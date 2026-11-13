"""Orchestration: paraphrase → metric mapping → gather → solve → answer, one StepEvent per step (spec §5)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence

from src.agent.answer import answer_text, leaf_detail, refusal_text
from src.agent.gather import Gatherer
from src.agent.llm import CountingLlm, LlmUnavailable
from src.agent.metrics import map_metrics
from src.agent.present import answer_data, gather_data, metrics_data, plan_data, solve_data
from src.agent.solver import Attempt, build_context, solve
from src.agent.types import AgentConfig, AgentRun, DataNeed, StepEvent
from src.calculation import Answer, FactSource
from src.calculation.metric_tree import MetricTree
from src.extraction.extractor import CellLocator, LeafExtractor
from src.extraction.llm_fallback import LlmCellLocator, PointerAnswer
from src.extraction.templates import TickerTemplates
from src.preprocess.cleaner import fold_text
from src.query_paraphrase.schemas import QATurn, QueryPlan

# Folded words of a full-year period; anything else in Subquery.period is a sub-annual period.
_ANNUAL_WORDS = frozenset({"nam", "ca", "tron", "tai", "chinh", "fy", "full", "fiscal", "year", "annual"})
_TWELVE_MONTHS = re.compile(r"\b12 thang\b")
# "cuối năm" (end-of-year balance) is annual; "cuối" alone or before another period is not.
_END_OF_YEAR = re.compile(r"\bcuoi nam\b")
_YEAR = re.compile(r"(?:19|20|21)\d{2}")

_TITLES = {
    "plan": "Phân tích câu hỏi",
    "metrics": "Xác định chỉ tiêu",
    "gather": "Thu thập số liệu",
    "answer": "Kết quả",
}


class _Finished(Exception):
    """Ends a run early with its final status (internal control flow only)."""

    def __init__(self, status: str, reason: str | None, text: str):
        super().__init__(reason)
        self.status, self.reason, self.text = status, reason, text


def _refuse(code: str, details: Sequence[str], status: str = "refused") -> _Finished:
    return _Finished(status, code, refusal_text(code, details))


def _source_line(source: FactSource) -> str:
    return f"{source.row_label} — {source.doc_id}, trang {source.page}"


class ReasoningAgent:
    def __init__(
        self,
        paraphraser,
        tree: MetricTree,
        templates: TickerTemplates,
        extractor_factory: Callable[[CellLocator | None], LeafExtractor],
        llm: CountingLlm,
        config: AgentConfig = AgentConfig(),
    ):
        self._paraphraser = paraphraser
        self._tree = tree
        self._templates = templates
        self._extractor_factory = extractor_factory
        self._llm = llm
        self._config = config

    def run(
        self,
        question: str,
        qa_history: Sequence[QATurn | Mapping] = (),
        on_event: Callable[[StepEvent], None] | None = None,
    ) -> AgentRun:
        started, calls_before = time.perf_counter(), self._llm.calls
        events: list[StepEvent] = []

        def emit(kind: str, data: Mapping[str, object], title: str | None = None) -> None:
            event = StepEvent(kind, title or _TITLES[kind], data)
            events.append(event)
            if on_event is not None:
                on_event(event)

        answer: Answer | None = None
        sources: tuple[FactSource, ...] = ()
        try:
            answer, sources = self._answer(question, qa_history, emit)
            status, reason, text = "answered", None, answer_text(answer, sources, None)
        except _Finished as finished:
            status, reason, text = finished.status, finished.reason, finished.text
        except LlmUnavailable as error:
            answer, sources = None, ()
            status, reason = "error", "llm_unavailable"
            text = refusal_text("llm_unavailable", [f"{type(error).__name__}: {error}"])
        emit("answer", answer_data(status, reason, text, [_source_line(s) for s in sources]))
        return AgentRun(status, text, reason, answer, sources, tuple(events),
                        self._llm.calls - calls_before, time.perf_counter() - started)

    # --- steps ----------------------------------------------------------------------------
    def _answer(self, question: str, qa_history, emit) -> tuple[Answer, tuple[FactSource, ...]]:
        plan = self._plan(question, qa_history, emit)
        mappings, tree = map_metrics(plan, self._tree, self._templates, self._llm)
        emit("metrics", metrics_data(mappings))
        unmapped = [sq for sq, m in zip(plan.subqueries, mappings) if m.node_id is None]
        if unmapped:
            raise _refuse("metric_unmapped", [f"{sq.id}: {sq.metric_text or sq.metric or sq.question}"
                                              for sq in unmapped])

        config, llm = self._config, self._llm
        locator = (LlmCellLocator(lambda p: llm(p, PointerAnswer), llm_errors=(LlmUnavailable,))
                   if config.locator_enabled else None)
        gatherer = Gatherer(tree, self._templates, self._extractor_factory, locator,
                            config.max_locator_calls, config.default_scope)
        node_ids = {m.subquery_id: m.node_id for m in mappings}
        logs = gatherer.gather([DataNeed(node_ids[sq.id], sq.ticker, sq.year, sq.scope)
                                for sq in plan.subqueries])
        emit("gather", gather_data(logs))

        def on_attempt(attempt: Attempt) -> None:
            if attempt.gathered:
                emit("gather", gather_data(attempt.gathered))
            emit("solve", solve_data(attempt), f"Lập trình tính toán — lần {attempt.number}")

        context = build_context(question, mappings, self._templates, config.default_scope)
        solved = solve(question, plan, mappings, gatherer, tree, context, llm, config, on_attempt)
        if solved.status == "facts_missing":
            raise _refuse("facts_missing", [leaf_detail(log) for log in solved.failed])
        if solved.status != "passed" or solved.result is None:
            problems = solved.attempts[-1].problems if solved.attempts else ()
            raise _refuse("gates_failed", problems)
        keys = solved.result.trace.facts_used()
        sources = tuple(dict.fromkeys(gatherer.facts.get(key).source for key in keys))
        return solved.result.answer, sources

    def _plan(self, question: str, qa_history, emit) -> QueryPlan:
        result = self._paraphraser.paraphrase(question, qa_history)
        if result.status == "invalid":  # one repair round with the validator's problems
            feedback = [f"{issue.subquery_id}: {issue.message}" if issue.subquery_id else issue.message
                        for issue in result.issues]
            result = self._paraphraser.paraphrase(question, qa_history, feedback=feedback)
        messages = [issue.message for issue in result.issues]
        if result.status == "unavailable":
            raise _refuse("llm_unavailable", messages, status="error")
        if result.status == "needs_clarification":
            raise _refuse("clarify", messages, status="clarify")
        if result.status != "valid" or result.plan is None:
            raise _refuse("plan_invalid", messages)
        plan = result.plan
        emit("plan", plan_data(plan))
        missing = [line for sq in plan.subqueries for line in _missing_fields(sq)]
        if missing:
            raise _refuse("plan_invalid", missing)
        return plan


def _missing_fields(sq) -> list[str]:
    lines = []
    if sq.ticker is None:
        lines.append(f"{sq.id}: thiếu mã cổ phiếu ({sq.question})")
    if sq.year is None:
        lines.append(f"{sq.id}: thiếu năm ({sq.question})")
    if not _annual(sq.period):
        lines.append(f"{sq.id}: kỳ {sq.period} không được hỗ trợ (chỉ có số liệu năm)")
    return lines


def _annual(period: str | None) -> bool:
    """True for no period or a full-year wording ("năm", "cả năm 2023", "FY2023", "12 tháng",
    "cuối năm" for an end-of-year balance).

    Facts are annual only, so any other period (quý 2, 6 tháng, H1, nửa đầu năm, cuối quý, ...)
    is refused rather than answered with the full-year value.
    """
    if period is None:
        return True
    folded = _END_OF_YEAR.sub(" ", _TWELVE_MONTHS.sub(" ", fold_text(period)))
    words = re.findall(r"[a-z]+|\d+", folded)
    return all(w in _ANNUAL_WORDS or _YEAR.fullmatch(w) for w in words)

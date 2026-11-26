"""Advanced RAG vs Agentic RAG: paired configurations and one evaluation record per question.

Both systems run the same ``ReasoningAgent`` pipeline (paraphrase and decompose → metric mapping
→ metadata-filtered hybrid BM25 + dense retrieval, RRF, cross-encoder rerank → leaf extraction
with the LLM cell locator → solution program → sandbox → verification gates → cited answer) with
the same models, indexes, prompts and retrieval parameters. They differ only in ``ADAPTIVE_FIELDS``:

- ``advanced_rag``: one plan, data gathered once from the plan's metrics (metric-tree leaves,
  including prior-year balances of the formulas), one solution program; any failure is final.
- ``agentic_rag``: plan repair with validator feedback, data gathering driven by the solution's
  declared needs, and solution repair with sandbox/gate feedback.

Counters here only observe calls; they never change what a system does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from src.agent.answer import format_value
from src.agent.llm import DEFAULT_MODEL, CountingLlm, StructuredLlm
from src.agent.types import AgentConfig, AgentRun
from src.extraction.retrieval import RankedTable, TableRetriever

SYSTEMS = ("advanced_rag", "agentic_rag")
ADAPTIVE_FIELDS = ("max_plan_repairs", "max_repairs", "max_gather_rounds")
_LOG = logging.getLogger(__name__)

# Every field not in ADAPTIVE_FIELDS is identical for both systems (checked in tests).
_SHARED = AgentConfig(retriever_mode="hybrid_rerank", locator_enabled=True, max_locator_calls=6,
                      default_scope="consolidated")
_ADAPTIVE: dict[str, dict[str, int]] = {
    "advanced_rag": {"max_plan_repairs": 0, "max_repairs": 0, "max_gather_rounds": 0},
    "agentic_rag": {"max_plan_repairs": 1, "max_repairs": 3, "max_gather_rounds": 2},
}


def experiment_config(system: str) -> AgentConfig:
    if system not in _ADAPTIVE:
        raise ValueError(f"system must be one of {SYSTEMS}, got {system!r}")
    return dataclasses.replace(_SHARED, **_ADAPTIVE[system])


def shared_fields(config: AgentConfig) -> dict[str, object]:
    """The config fields both systems must agree on."""
    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)
            if f.name not in ADAPTIVE_FIELDS}


# --- observers ------------------------------------------------------------------------------
class StageCounter:
    """Structured LLM wrapper counting calls per reply schema (QueryPlan, MetricChoices,
    PointerAnswer, SolutionDraft), failed calls included."""

    def __init__(self, inner: StructuredLlm):
        self._inner = inner
        self.counts: Counter[str] = Counter()

    def __call__(self, prompt, schema):
        self.counts[schema.__name__] += 1
        return self._inner(prompt, schema)


class RetrievalCounter:
    """``build_agent(retriever_wrapper=...)``: counts ``rank`` calls of every wrapped retriever."""

    def __init__(self):
        self.calls = 0

    def __call__(self, inner: TableRetriever) -> TableRetriever:
        return _CountedRetriever(inner, self)


class _CountedRetriever:
    def __init__(self, inner: TableRetriever, counter: RetrievalCounter):
        self._inner = inner
        self._counter = counter
        self.mode = inner.mode

    def rank(self, ticker, scope, report_year, leaf) -> tuple[RankedTable, ...]:
        self._counter.calls += 1  # failed calls count too
        return self._inner.rank(ticker, scope, report_year, leaf)


class TokenUsage(BaseCallbackHandler):
    """LangChain callback summing provider-reported tokens; never estimates missing usage."""

    def __init__(self):
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.reported = 0  # LLM responses that carried usage
        self.unreported = 0  # LLM responses without usage
        self.errors = 0  # LLM calls that raised (their usage, if any, is unknown)

    def on_llm_end(self, response, **kwargs) -> None:
        usage = _usage(response)
        if usage is None:
            self.unreported += 1
            return
        self.reported += 1
        self.input_tokens += usage[0]
        self.output_tokens += usage[1]

    def on_llm_error(self, error, **kwargs) -> None:
        self.errors += 1

    def snapshot(self) -> tuple[int, int, int, int, int]:
        return self.input_tokens, self.output_tokens, self.reported, self.unreported, self.errors


def _usage(response) -> tuple[int, int] | None:
    found = False
    tokens_in = tokens_out = 0
    for generations in getattr(response, "generations", None) or ():
        for generation in generations:
            meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if meta:
                found = True
                tokens_in += int(meta.get("input_tokens", 0))
                tokens_out += int(meta.get("output_tokens", 0))
    if found:
        return tokens_in, tokens_out
    usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    return None


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens; supplied by the caller (no price list is built in)."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal

    def __post_init__(self) -> None:
        for value in (self.input_per_mtok, self.output_per_mtok):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError("prices must be finite Decimals >= 0")


def token_record(before: Sequence[int], after: Sequence[int],
                 pricing: Pricing | None) -> dict[str, object] | None:
    """Per-question token use from two TokenUsage snapshots; None when no response reported usage.

    ``complete`` is False when some LLM response had no usage or a call raised, so totals are a
    lower bound; the cost is then not estimated.
    """
    tokens_in, tokens_out, reported, unreported, errors = (a - b for a, b in zip(after, before))
    if reported == 0:
        return None
    complete = unreported == 0 and errors == 0
    cost = None
    if pricing is not None and complete:
        cost = str((tokens_in * pricing.input_per_mtok + tokens_out * pricing.output_per_mtok)
                   / Decimal(1_000_000))
    return {"input": tokens_in, "output": tokens_out, "total": tokens_in + tokens_out,
            "complete": complete, "responses_without_usage": unreported, "failed_calls": errors,
            "estimated_cost_usd": cost}


# --- record ---------------------------------------------------------------------------------
def _answer_record(run: AgentRun) -> dict[str, object] | None:
    answer = run.answer
    if answer is None:
        return None
    value = answer.value
    if answer.kind == "quantity":
        value = {"value": str(value.value), "dimension": str(value.dimension)}
    return {"kind": answer.kind, "value": value, "unit": answer.unit, "display": format_value(answer)}


def run_record(run: AgentRun, *, question_id, question: str, system: str, config: AgentConfig,
               model: str, retrieval_calls: int | None, llm_calls_by_stage: Mapping[str, int] | None,
               tokens: dict[str, object] | None) -> dict[str, object]:
    """JSON-serializable record of one run. Nothing here is computed from ground truth."""
    solves = [e.data for e in run.events if e.kind == "solve"]
    gathers = [e for e in run.events if e.kind == "gather"]
    return {
        "id": question_id,
        "system": system,
        "model": model,
        "config": dataclasses.asdict(config),
        "question": question,
        "status": run.status,
        "reason": run.reason,
        "text": run.text,
        "answer": _answer_record(run),
        "sources": [{"chunk_id": s.chunk_id, "doc_id": s.doc_id, "page": s.page,
                     "row_label": s.row_label, "column_header": s.column_header} for s in run.sources],
        "evidence": [{"leaf_id": l.leaf_id, "ticker": l.ticker, "year": l.year, "scope": l.scope,
                      "outcome": l.outcome, "reason": l.reason,
                      "value": None if l.value is None else str(l.value), "chunk_id": l.chunk_id,
                      "doc_id": l.doc_id, "page": l.page, "row_label": l.row_label, "tool": l.tool}
                     for l in run.evidence],
        "calculation": {"code": solves[-1]["code"] if solves else None, "attempts": len(solves)},
        # An answer exists only when the sandbox ran and every gate passed (solver contract).
        "verification": {"passed": run.status == "answered",
                         "attempts": [{"number": s["number"], "outcome": s["outcome"],
                                       "problems": list(s["problems"])} for s in solves]},
        "telemetry": {"seconds": run.seconds, "llm_calls": run.llm_calls,
                      "llm_calls_by_stage": None if llm_calls_by_stage is None else dict(llm_calls_by_stage),
                      "retrieval_calls": retrieval_calls,
                      "solver_gather_rounds": max(len(gathers) - 1, 0),
                      "tokens": tokens},
    }


# --- manifest -------------------------------------------------------------------------------
MANIFEST_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, object]:
    """HEAD commit and whether tracked files differ from it; None values when git is unavailable."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def run_manifest(system: str, *, model: str, temperature: float | None, seed: int | None,
                 pricing: Pricing | None, questions: Path, db: Path, bm25_dir: Path,
                 chroma_dir: Path, repo_root: Path, bm25_name: str = "bm25-table") -> dict[str, object]:
    """Everything a run's records depend on. Large files (corpus, BM25 pickle) are identified by
    size and mtime, not hashed; the question file and the BM25 manifest are hashed."""
    import inspect

    from config.settings import DENSE_MODEL_NAME, RERANK_MODEL_NAME, RRF_CONSTANT
    from src.agent.llm import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
    from src.retrieval.table_retriever import open_table_retriever  # import-light module

    defaults = inspect.signature(open_table_retriever).parameters
    bm25_manifest = bm25_dir / f"{bm25_name}.json"
    return {
        "manifest_version": MANIFEST_VERSION,
        "system": system,
        "config": dataclasses.asdict(experiment_config(system)),
        "llm": {"provider": "openai", "model": model, "timeout": DEFAULT_TIMEOUT,
                "max_retries": DEFAULT_MAX_RETRIES, "temperature": temperature, "seed": seed},
        "pricing": None if pricing is None else {"input_per_mtok": str(pricing.input_per_mtok),
                                                 "output_per_mtok": str(pricing.output_per_mtok)},
        "code": _git_state(repo_root),
        "dataset": {"path": str(questions.resolve()), "sha256": _sha256(questions)},
        "corpus": _file_identity(db),
        "bm25": {"name": bm25_name,
                 "manifest_sha256": _sha256(bm25_manifest) if bm25_manifest.is_file() else None,
                 "pickle": _file_identity(bm25_dir / f"{bm25_name}.pkl")},
        "chroma": {"path": str(chroma_dir.resolve()), "name": chroma_dir.name},
        "retrieval": {"k": defaults["k"].default, "depth": defaults["depth"].default,
                      "fp16": defaults["fp16"].default, "rrf_constant": RRF_CONSTANT},
        "models": {"embedding": DENSE_MODEL_NAME, "reranker": RERANK_MODEL_NAME},
    }


def manifest_id(manifest: Mapping[str, object]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- runner ---------------------------------------------------------------------------------
class ExperimentRunner:
    """One system, one opened agent; ``run`` returns one record per question (sequential use).

    An exception the agent does not turn into a run becomes an error record that keeps the
    calls made before it (the exception message is never kept: it may carry secrets).
    """

    def __init__(self, system: str, agent, *, config: AgentConfig, model: str, stages: StageCounter,
                 retrieval: RetrievalCounter, usage: TokenUsage | None, pricing: Pricing | None = None,
                 manifest_id: str | None = None):
        self.system, self.config, self.model, self.manifest_id = system, config, model, manifest_id
        self._agent, self._stages, self._retrieval = agent, stages, retrieval
        self._usage, self._pricing = usage, pricing

    def run(self, question_id, question: str, qa_history=()) -> dict[str, object]:
        stages_before, retrieval_before = Counter(self._stages.counts), self._retrieval.calls
        usage_before = self._usage.snapshot() if self._usage is not None else None
        started = time.perf_counter()
        run = failure = None
        try:
            run = self._agent.run(question, qa_history)
        except Exception as error:
            _LOG.exception("agent run failed for id %s", question_id)  # traceback on the log only
            failure = error
        seconds = time.perf_counter() - started
        stages = dict(sorted((self._stages.counts - stages_before).items()))
        retrieval_calls = self._retrieval.calls - retrieval_before
        tokens = (None if self._usage is None
                  else token_record(usage_before, self._usage.snapshot(), self._pricing))
        if run is None:
            record = error_record(question_id=question_id, question=question, system=self.system,
                                  config=self.config, model=self.model, error=failure, seconds=seconds)
            record["telemetry"].update(llm_calls=sum(stages.values()), llm_calls_by_stage=stages,
                                       retrieval_calls=retrieval_calls, tokens=tokens)
        else:
            record = run_record(run, question_id=question_id, question=question, system=self.system,
                                config=self.config, model=self.model, retrieval_calls=retrieval_calls,
                                llm_calls_by_stage=stages, tokens=tokens)
        record["manifest_id"] = self.manifest_id
        return record


def open_runner(system: str, *, db: Path, bm25_dir: Path, chroma_dir: Path | None,
                model: str = DEFAULT_MODEL, temperature: float | None = None, seed: int | None = None,
                pricing: Pricing | None = None, manifest_id: str | None = None,
                llm: StructuredLlm | None = None, build: Callable | None = None) -> ExperimentRunner:
    """Open the real components for ``system``. ``llm`` replaces the OpenAI client (no token
    usage is then reported); ``build`` replaces ``build_agent`` (tests)."""
    config = experiment_config(system)
    if config.retriever_mode in ("hybrid", "hybrid_rerank") and chroma_dir is None:
        raise ValueError(f"{config.retriever_mode} retrieval needs a Chroma index directory")
    usage = None
    if llm is None:
        from src.agent.llm import openai_llm

        usage = TokenUsage()
        llm = openai_llm(model, callbacks=[usage], temperature=temperature, seed=seed)
    if build is None:
        from src.agent.factory import build_agent as build
    stages, retrieval = StageCounter(llm), RetrievalCounter()
    agent = build(config, llm=CountingLlm(stages), db=db, bm25_dir=bm25_dir, chroma_dir=chroma_dir,
                  retriever_wrapper=retrieval)
    return ExperimentRunner(system, agent, config=config, model=model, stages=stages,
                            retrieval=retrieval, usage=usage, pricing=pricing, manifest_id=manifest_id)


def error_record(*, question_id, question: str, system: str, config: AgentConfig, model: str,
                 error: BaseException, seconds: float) -> dict[str, object]:
    """Record for an exception the agent did not turn into a run (the message is never kept:
    it may carry secrets)."""
    return {"id": question_id, "system": system, "model": model, "config": dataclasses.asdict(config),
            "question": question, "status": "error", "reason": f"exception:{type(error).__name__}",
            "text": None, "answer": None, "sources": [], "evidence": [], "calculation": None,
            "verification": None, "telemetry": {"seconds": seconds}}

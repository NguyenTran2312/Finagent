"""Display rows for agent step events (pure; no UI imports).

``StepEvent.data`` is JSON-like. Shapes per kind (keys are the constants below):

- plan:    {OPERATION: str, SUBQUERIES: [{ID, TICKER, YEAR, SCOPE, METRIC, METRIC_TEXT}]}
- metrics: {MAPPINGS: [{SUBQUERY_ID, NODE_ID, HOW}]}
- gather:  {LEAVES: [{LEAF_ID, TICKER, YEAR, SCOPE, OUTCOME, REASON, VALUE, DOC_ID, PAGE, TOOL}]}
           VALUE is ``str(Decimal)`` or None.
- solve:   {NUMBER: int, CODE: str, OUTCOME: str, PROBLEMS: [str]}
- answer:  {STATUS: str, REASON: str | None, TEXT: str, SOURCES: [str]}

Builders turn the real objects into that data; ``rows`` consumes only the data.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal

from src.agent.answer import _vietnamese_number
from src.agent.metrics import MetricMapping
from src.agent.solver import Attempt
from src.agent.types import AgentRun, LeafLog, StepEvent
from src.query_paraphrase.schemas import QueryPlan

# plan
OPERATION = "operation"
SUBQUERIES = "subqueries"
# metrics
MAPPINGS = "mappings"
# gather
LEAVES = "leaves"
# solve
NUMBER = "number"
CODE = "code"
PROBLEMS = "problems"
# answer
STATUS = "status"
TEXT = "text"
SOURCES = "sources"
# shared item keys
ID = "id"
TICKER = "ticker"
YEAR = "year"
SCOPE = "scope"
METRIC = "metric"
METRIC_TEXT = "metric_text"
SUBQUERY_ID = "subquery_id"
NODE_ID = "node_id"
HOW = "how"
LEAF_ID = "leaf_id"
OUTCOME = "outcome"
REASON = "reason"
VALUE = "value"
DOC_ID = "doc_id"
PAGE = "page"
TOOL = "tool"


def plan_data(plan: QueryPlan) -> dict[str, object]:
    return {
        OPERATION: plan.operation,
        SUBQUERIES: [
            {ID: s.id, TICKER: s.ticker, YEAR: s.year, SCOPE: s.scope,
             METRIC: s.metric, METRIC_TEXT: s.metric_text}
            for s in plan.subqueries
        ],
    }


def metrics_data(mappings: Iterable[MetricMapping]) -> dict[str, object]:
    return {MAPPINGS: [{SUBQUERY_ID: m.subquery_id, NODE_ID: m.node_id, HOW: m.how}
                       for m in mappings]}


def gather_data(logs: Iterable[LeafLog]) -> dict[str, object]:
    return {LEAVES: [
        {LEAF_ID: log.leaf_id, TICKER: log.ticker, YEAR: log.year, SCOPE: log.scope,
         OUTCOME: log.outcome, REASON: log.reason,
         VALUE: None if log.value is None else str(log.value),
         DOC_ID: log.doc_id, PAGE: log.page, TOOL: log.tool}
        for log in logs
    ]}


def solve_data(attempt: Attempt) -> dict[str, object]:
    return {NUMBER: attempt.number, CODE: attempt.code, OUTCOME: attempt.outcome,
            PROBLEMS: list(attempt.problems)}


def answer_data(status: str, reason: str | None, text: str,
                sources: Sequence[str] = ()) -> dict[str, object]:
    return {STATUS: status, REASON: reason, TEXT: text, SOURCES: list(sources)}


def _s(value: object) -> str:
    return "" if value is None else str(value)


def _plan_rows(data: Mapping[str, object]) -> list[dict[str, str]]:
    return [
        {"Câu hỏi con": _s(i[ID]), "Mã": _s(i[TICKER]), "Năm": _s(i[YEAR]),
         "Phạm vi": _s(i[SCOPE]), "Chỉ tiêu": _s(i[METRIC]), "Cụm từ": _s(i[METRIC_TEXT])}
        for i in data[SUBQUERIES]  # type: ignore[attr-defined]
    ]


def _metrics_rows(data: Mapping[str, object]) -> list[dict[str, str]]:
    return [{"Câu hỏi con": _s(i[SUBQUERY_ID]), "Node": _s(i[NODE_ID]), "Cách": _s(i[HOW])}
            for i in data[MAPPINGS]]  # type: ignore[attr-defined]


def _gather_rows(data: Mapping[str, object]) -> list[dict[str, str]]:
    out = []
    for i in data[LEAVES]:  # type: ignore[attr-defined]
        found = i[OUTCOME] == "Found"
        result = _s(i[OUTCOME])
        if not found and i[REASON]:
            result = f"{result}: {i[REASON]}"
        value = "" if i[VALUE] is None else _vietnamese_number(Decimal(i[VALUE]))
        source = f"{i[DOC_ID]}, trang {i[PAGE]}" if found and i[DOC_ID] is not None else ""
        out.append({"Leaf": _s(i[LEAF_ID]), "Mã": _s(i[TICKER]), "Năm": _s(i[YEAR]),
                    "Phạm vi": _s(i[SCOPE]), "Kết quả": result, "Giá trị": value,
                    "Nguồn": source, "Công cụ": _s(i[TOOL])})
    return out


def _solve_rows(data: Mapping[str, object]) -> list[dict[str, str]]:
    return [{"Lần": _s(data[NUMBER]), "Kết quả": _s(data[OUTCOME]),
             "Vấn đề": "; ".join(data[PROBLEMS])}]  # type: ignore[arg-type]


def _answer_rows(data: Mapping[str, object]) -> list[dict[str, str]]:
    return [{"Trạng thái": _s(data[STATUS]), "Lý do": _s(data[REASON])}]


_ROWS = {"plan": _plan_rows, "metrics": _metrics_rows, "gather": _gather_rows,
         "solve": _solve_rows, "answer": _answer_rows}


def rows(event: StepEvent) -> list[dict[str, str]]:
    return _ROWS[event.kind](event.data)


def footer(run: AgentRun) -> str:
    return f"{run.llm_calls} lần gọi LLM · {run.seconds:.1f} giây"

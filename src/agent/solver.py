"""Solve: LLM-written solution() programs run in the sandbox, checked by the gates, repaired (spec §4.3)."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel

from src.agent.gather import Gatherer
from src.agent.llm import LlmMalformedOutput, StructuredLlm
from src.agent.metrics import MetricMapping
from src.agent.types import AgentConfig, DataNeed, LeafLog, Scope
from src.calculation import sandbox
from src.calculation.facts import FactSet
from src.calculation.gates import (
    ALLOWED_CONSTANTS,
    question_literals,
    requested_answer_kind,
    requested_unit,
    run_gates,
)
from src.calculation.helpers import Answer, ExecutionContext, Trace, build_helpers
from src.calculation.metric_tree import MetricTree
from src.calculation.sandbox import ExecutionFailure, ExecutionResult
from src.calculation.units import MONEY_SCALES, Dimension, Quantity
from src.extraction.templates import TickerTemplates
from src.query_paraphrase.schemas import QueryPlan

UNDECLARED_DATA = "code used data not declared in needs"
MALFORMED_DRAFT = "output did not match the SolutionDraft schema"
_DATA_FAILURES = frozenset({"MissingFact", "MetricUnavailable"})

# Calculation core spec §16, "Generated-code constraints".
CODE_CONSTRAINTS = """\
- No attribute access at all (`d.items()`, `xs.append(...)`); iterate with `for k in d` and
  build lists with comprehensions.
- Numeric literals are exact `Decimal`s, but `int / int` from builtins/helpers (e.g.
  `count_if(...) / len(xs)`) yields a float and fails closed; multiply by a literal first
  (e.g. `count_if(...) * 100 / len(xs)`).
- Thresholds go through `percent(n)` / `money(n, scale)`; for "1 nghìn tỷ" write
  `money(1, "nghin_ty")`, not `money(1000, "ty")` (1000 is not a question literal).
- Strings are allowed as plain values (ticker lists); f-strings are forbidden. The bare name
  `_` is allowed; every other `_`-prefixed name is rejected."""

_INSTRUCTIONS = f"""\
Write a Python program that answers the question from the reported facts.
Reply with:
- code: exactly one top-level `def solution():` without parameters, returning answer(value, unit);
  it may call only the helpers below and {", ".join(sandbox.ALLOWED_BUILTINS)};
  numeric literals must be {", ".join(str(c) for c in sorted(ALLOWED_CONSTANTS))}, a year or a
  number stated in the question;
- claimed_value: the value you expect solution() to return, as a plain decimal number in
  claimed_unit (a year, a ticker, or true/false for those units);
- claimed_unit: the unit passed to answer();
- needs: one entry per fact(), metric() or avg_balance() call the code makes (node_id, ticker,
  year, scope or null for the default, helper). Data not listed in needs is not fetched."""

_YEAR_UNITS = {"year": "year", "count": "count"}
_PERCENT_UNITS = ("percent", "percentage_points")
_PLAIN_UNITS = {"times": Dimension.RATIO, "days": Dimension.DAYS,
                "money_per_share": Dimension.MONEY_PER_SHARE}
CLAIM_UNITS = (*MONEY_SCALES, *_PERCENT_UNITS, *_PLAIN_UNITS, *_YEAR_UNITS, "ticker", "boolean")


class NeedModel(BaseModel):
    node_id: str
    ticker: str
    year: int
    scope: Literal["consolidated", "separate"] | None = None
    helper: Literal["fact", "metric", "avg_balance"] = "metric"


class SolutionDraft(BaseModel):
    code: str
    claimed_value: str
    claimed_unit: str
    needs: list[NeedModel]


@dataclass(frozen=True)
class Attempt:
    number: int
    code: str
    needs: tuple[DataNeed, ...]
    gathered: tuple[LeafLog, ...]
    problems: tuple[str, ...]
    outcome: Literal["passed", "repair", "facts_missing"]
    # Declared needs never gathered (no gather round left) when the code failed for missing data.
    ungathered: tuple[DataNeed, ...] = ()


@dataclass(frozen=True)
class SolveResult:
    status: Literal["passed", "facts_missing", "gates_failed"]
    result: ExecutionResult | None
    attempts: tuple[Attempt, ...]
    failed: tuple[LeafLog, ...]


def claimed_answer(draft: SolutionDraft) -> Answer | None:
    """The draft's claim as an Answer (the inverse of `to_display`); None when it does not parse."""
    unit, text = draft.claimed_unit.strip(), draft.claimed_value.strip()
    if unit == "ticker":
        return Answer("ticker", text, unit) if text else None
    if unit == "boolean":
        flag = {"true": True, "false": False}.get(text.casefold())
        return None if flag is None else Answer("boolean", flag, unit)
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if unit in _YEAR_UNITS:
        if number != number.to_integral_value():
            return None
        return Answer(_YEAR_UNITS[unit], int(number), unit)
    if unit in MONEY_SCALES:
        value = Quantity(number * MONEY_SCALES[unit], Dimension.MONEY)
    elif unit in _PERCENT_UNITS:
        value = Quantity(number / 100, Dimension.RATIO)
    elif unit in _PLAIN_UNITS:
        value = Quantity(number, _PLAIN_UNITS[unit])
    else:
        return None
    return Answer("quantity", value, unit)


def build_context(question: str, mappings: Sequence[MetricMapping], templates: TickerTemplates,
                  default_scope: Scope) -> ExecutionContext:
    return ExecutionContext(
        default_scope=default_scope,
        ticker_templates=templates.calculation_templates(),
        question_literals=question_literals(question),
        requested_unit=requested_unit(question),
        requested_kind=requested_answer_kind(question),
        named_metrics=frozenset(m.node_id for m in mappings if m.node_id is not None),
    )


def solve(question: str, plan: QueryPlan, mappings: Sequence[MetricMapping], gatherer: Gatherer,
          tree: MetricTree, context: ExecutionContext, llm: StructuredLlm, config: AgentConfig,
          on_attempt: Callable[[Attempt], None]) -> SolveResult:
    """At most 1 + max_repairs attempts; drafts' new needs are gathered first while rounds remain."""
    attempts: list[Attempt] = []
    gather_rounds = 0
    for number in range(1, config.max_repairs + 2):
        prompt = _prompt(question, plan, mappings, tree, gatherer.facts, context,
                         attempts[-1] if attempts else None)
        try:
            draft = llm(prompt, SolutionDraft)
        except LlmMalformedOutput:  # a reply that does not parse is repaired like a bad draft
            attempt = Attempt(number, "", (), (), (MALFORMED_DRAFT,), "repair")
            attempts.append(attempt)
            on_attempt(attempt)
            continue
        needs = tuple(DataNeed(n.node_id, n.ticker, n.year, n.scope, n.helper) for n in draft.needs)
        gathered: tuple[LeafLog, ...] = ()
        failed: tuple[LeafLog, ...] = ()
        ungathered: tuple[DataNeed, ...] = ()
        result: ExecutionResult | None = None
        claimed = claimed_answer(draft)
        claim_problems = () if claimed is not None else (
            f"claim does not parse: {draft.claimed_value} {draft.claimed_unit} "
            f"(claimed_unit must be one of: {', '.join(CLAIM_UNITS)})",)
        unknown = [n.node_id for n in needs if not _known(tree, n.node_id)]
        if unknown:
            problems = (*(f"unknown node id: {node_id}" for node_id in dict.fromkeys(unknown)),
                        *claim_problems)
        else:
            new = [n for n in needs if not gatherer.requested(n)]
            if new and gather_rounds < config.max_gather_rounds:
                gather_rounds += 1
                gathered = gatherer.gather(new)
            outcome = sandbox.run(draft.code, tree, gatherer.facts, context)
            if isinstance(outcome, ExecutionFailure):
                if outcome.detail in _DATA_FAILURES:
                    failed = tuple(log for n in needs for log in gatherer.unmet(n))
                    problems = () if failed else (UNDECLARED_DATA, *claim_problems)
                    if not failed:
                        ungathered = tuple(dict.fromkeys(n for n in needs if not gatherer.requested(n)))
                else:
                    problems = (*_failure_problems(outcome), *claim_problems)
            elif claimed is None:  # R8: never gate without a claim; the attempt cannot pass
                problems = claim_problems
            else:
                report = run_gates(draft.code, outcome, tree, gatherer.facts, context, claimed=claimed)
                problems = tuple(f"{i.code}: {i.message}" for i in report.issues if i.severity == "error")
                result = outcome if report.passed else None
        status = "facts_missing" if failed else "passed" if result is not None else "repair"
        attempt = Attempt(number, draft.code, needs, gathered, problems, status, ungathered)
        attempts.append(attempt)
        on_attempt(attempt)
        if status == "facts_missing":
            return SolveResult("facts_missing", None, tuple(attempts), tuple(dict.fromkeys(failed)))
        if status == "passed":
            return SolveResult("passed", result, tuple(attempts), ())
    return SolveResult("gates_failed", None, tuple(attempts), ())


# --- internals ----------------------------------------------------------------------------
def _known(tree: MetricTree, node_id: str) -> bool:
    try:
        tree.kind(node_id)
    except KeyError:
        return False
    return True


def _failure_problems(failure: ExecutionFailure) -> tuple[str, ...]:
    if failure.rejections:
        return tuple(f"rejected: {r.node} at line {r.line}: {r.reason}" for r in failure.rejections)
    return (f"{failure.kind}: {failure.detail}",)


def _helper_docs(tree: MetricTree, context: ExecutionContext) -> list[str]:
    helpers = build_helpers(tree, FactSet(()), context, Trace())
    lines = []
    for name, function in helpers.items():
        lines.append(f"{name}{inspect.signature(function)}")
        lines += [f"    {line}" for line in (inspect.getdoc(function) or "").splitlines()]
    return lines


def _prompt(question: str, plan: QueryPlan, mappings: Sequence[MetricMapping], tree: MetricTree,
            facts: FactSet, context: ExecutionContext, previous: Attempt | None) -> str:
    node_ids = {m.subquery_id: m.node_id for m in mappings}
    out = [_INSTRUCTIONS, f"\nQuestion: {question}",
           f"Requested answer kind: {context.requested_kind}; requested unit: "
           f"{context.requested_unit or 'not stated'}; default scope: {context.default_scope}.",
           "\nSubqueries (subquery id → node id):"]
    out += [f"- {sq.id} → {node_ids.get(sq.id) or 'unmapped'}: {sq.question}" for sq in plan.subqueries]
    out.append("\nHelpers:")
    out += _helper_docs(tree, context)
    out += ["\nCode constraints:", CODE_CONSTRAINTS, "\nAvailable facts (metric_id ticker year scope = value):"]
    inventory = sorted(f"{f.metric_id} {f.ticker} {f.fiscal_year} {f.scope} = {f.value.value} "
                       f"{f.value.dimension}" for f in facts)
    out += inventory or ["(none yet)"]
    if previous is not None:
        out += ["\nYour previous code:", previous.code, "Problems found:"]
        out += [f"- {p}" for p in previous.problems]
        out.append("Fix these problems and reply again.")
    return "\n".join(out)

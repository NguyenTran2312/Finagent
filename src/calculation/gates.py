"""Verification gates: decide whether an executed answer may be shown to a user (spec §11).

`run_gates` never trusts the generated code: it re-reads the ORIGINAL source for numeric
literals and re-checks every fact in the trace against the FactSet. Errors block the answer;
warnings (sanity checks, restated cross-check mismatches) are shown alongside it.

Question parsing (`question_literals`, `requested_answer_kind`, `requested_unit`) is what the
query layer uses to build the `ExecutionContext` the gates compare against.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from src.calculation.facts import FactKey, FactSet, FactSource
from src.calculation.helpers import Answer, AnswerKind, ExecutionContext, TraceStep
from src.calculation.metric_tree import DerivedDef, MetricTree
from src.calculation.sandbox import ExecutionResult, _SourceLines
from src.calculation.units import (
    MONEY_SCALES,
    Dimension,
    Quantity,
    UnitError,
    display_unit,
    round_display,
    to_display,
)
from src.preprocess.cleaner import parse_requested_unit, parse_vn_decimal

ALLOWED_CONSTANTS = frozenset(Decimal(x) for x in (0, 1, 2, 4, 12, 100, 365))
MIN_YEAR, MAX_YEAR = 1900, 2100
MONEY_LIMIT = Decimal(10) ** 16  # đồng
RELATIVE_TOLERANCE = Decimal("0.002")  # 0.2 %

GateCode = Literal["provenance", "formula", "unit_type", "sanity", "cross_check", "agreement"]
Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class GateIssue:
    code: GateCode
    severity: Severity
    message: str
    line: int | None = None
    trace_step: int | None = None


@dataclass(frozen=True)
class GateReport:
    issues: tuple[GateIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


# ---- question parsing ---------------------------------------------------------------------
_PERCENT_WORDS = ("%", "phần trăm")
_SCALE_WORDS: dict[str, Decimal] = {
    "nghìn tỷ": MONEY_SCALES["nghin_ty"], "ngàn tỷ": MONEY_SCALES["nghin_ty"],
    "nghìn tỉ": MONEY_SCALES["nghin_ty"], "ngàn tỉ": MONEY_SCALES["nghin_ty"],
    "trăm tỷ": MONEY_SCALES["tram_ty"], "trăm tỉ": MONEY_SCALES["tram_ty"],
    "tỷ": MONEY_SCALES["ty"], "tỉ": MONEY_SCALES["ty"],
    "triệu": MONEY_SCALES["trieu"],
    "nghìn": MONEY_SCALES["nghin"], "ngàn": MONEY_SCALES["nghin"],
}


# "tỷ" starts the words tỷ trọng / tỷ lệ / tỷ số / tỷ suất (share, ratio), not a money scale.
_NOT_A_SCALE = r"(?!\s+(?:trọng|lệ|số|suất)(?!\w))"


def _word(phrase: str) -> str:
    """Regex for a phrase of whole words with any whitespace between them."""
    pattern = r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)"
    return pattern + _NOT_A_SCALE if phrase.endswith(("tỷ", "tỉ")) else pattern


_UNIT_WORDS = sorted([*_SCALE_WORDS, "phần trăm"], key=len, reverse=True)
# Digits inside an alphanumeric token (tickers PC1, C32, L14) are not stated numbers.
_LITERAL_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)*)(?!\d)(?:\s*(%|"
                         + "|".join(map(_word, _UNIT_WORDS)) + r"))?")

_YEAR_KIND_RE = re.compile(r"(?<!\w)năm nào(?!\w)")
_TICKER_KIND_RE = re.compile(r"(?<!\w)(?:công ty|doanh nghiệp|ngân hàng) nào(?!\w)")
# "bao nhiêu" + an entity counts with or without "có"; "năm 2023" / "đơn vị tính" are not
# entities ("Doanh thu là bao nhiêu năm 2023?" asks for a quantity).
_COUNT_ENTITY = (r"(?:doanh nghiệp|dn|công ty|ctcp|ngân hàng|mã|đơn vị(?! (?:tính|tiền))"
                 r"|năm(?! ?\d))")
_COUNT_KIND_RE = re.compile(
    r"(?<!\w)(?:có bao nhiêu trong số|bao nhiêu " + _COUNT_ENTITY + r")(?!\w)")

# The requested unit is the one attached to the question word ("bao nhiêu tỷ đồng",
# "là bao nhiêu (%)", "(đơn vị: tỷ đồng)"); units elsewhere describe literals ("giảm 10%").
_TARGET_RE = re.compile(r"(?<!\w)(?:bao nhiêu|mấy|đơn vị(?: tính)?)(?!\w)")
_TARGET_LEAD = re.compile(r"[\s(:\-–]*")
_HEAD_UNITS: tuple[tuple[str, re.Pattern], ...] = (
    ("percentage_points", re.compile(r"điểm (?:phần trăm(?!\w)|%)")),
    ("percent", re.compile(r"%|phần trăm(?!\w)")),
    ("times", re.compile(r"(?:lần|vòng)(?!\w)")),
    ("days", re.compile(r"ngày(?!\w)")),
)
_MONEY_HEAD_RE = re.compile(
    r"(?:(?:(?:nghìn|ngàn|trăm) )?(?:tỷ|tỉ)|triệu|nghìn|ngàn)? ?(?:đồng|vnđ|vnd)(?!\w)")
_SCALE_NAMES = {scale: name for name, scale in MONEY_SCALES.items()}


def _normalize(question: object) -> str:
    if not isinstance(question, str):
        raise TypeError(f"question must be a str, got {type(question).__name__}")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", question).casefold()).strip()


def question_literals(question: str) -> frozenset[Decimal]:
    """Numbers stated in the question (Vietnamese format), with their derived forms.

    "10%" / "10 phần trăm" give 10 and 0.10; "20 tỷ" gives 20 and 20 000 000 000.
    Tokens that are not valid Vietnamese-format numbers are skipped.
    """
    found: set[Decimal] = set()
    for match in _LITERAL_RE.finditer(_normalize(question)):
        value = parse_vn_decimal(match.group(1), locale="vi")
        if value is None:
            continue
        found.add(value)
        unit = match.group(2)
        if unit is None:
            continue
        unit = re.sub(r"\s+", " ", unit)
        if unit in _PERCENT_WORDS:
            found.add(value / 100)
        else:
            found.add(value * _SCALE_WORDS[unit])
    return frozenset(found)


def requested_answer_kind(question: str) -> AnswerKind:
    """year for "năm nào", ticker for "công ty/doanh nghiệp/ngân hàng nào", count for
    "có bao nhiêu trong số" or "bao nhiêu" + an entity or "năm"; otherwise quantity."""
    text = _normalize(question)
    if _YEAR_KIND_RE.search(text):
        return "year"
    if _TICKER_KIND_RE.search(text):
        return "ticker"
    if _COUNT_KIND_RE.search(text):
        return "count"
    return "quantity"


def requested_unit(question: str) -> str | None:
    """The D5 unit name stated right after the last question word, or None.

    Only quantity questions carry a unit (year / ticker / count answers imply theirs).
    "điểm phần trăm" wins over "phần trăm"; "ngày" counts only as the asked-for unit, so a
    date ("ngày 31/12") never matches; money goes through `parse_requested_unit`.
    """
    text = _normalize(question)
    if requested_answer_kind(text) != "quantity":
        return None
    targets = list(_TARGET_RE.finditer(text))
    if not targets:
        return None
    head = text[targets[-1].end():]
    head = head[_TARGET_LEAD.match(head).end():]
    for name, pattern in _HEAD_UNITS:
        if pattern.match(head):
            return name
    money = _MONEY_HEAD_RE.match(head)
    if money is None:
        return None
    parsed = parse_requested_unit(money.group().replace("tỉ", "tỷ"))
    if (parsed.unit_status != "valid" or parsed.unit_kind != "monetary"
            or parsed.currency != "VND"):
        return None
    return _SCALE_NAMES.get(parsed.scale_multiplier)


# ---- answer comparison ----------------------------------------------------------------------
MAX_CLAIM_PLACES = 28  # executed values carry 28 significant digits; bounds round_display


def _places(value: Decimal) -> int:
    exponent = value.normalize().as_tuple().exponent
    places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return min(places, MAX_CLAIM_PLACES)


def compare_answers(claimed: Answer, executed: Answer) -> bool:
    """True when the claimed answer agrees with the executed one.

    Units must be identical. Quantities are shown in that unit and rounded to 2 places (or the
    claim's own precision when it has more), then may differ by at most 0.2 % relative; both
    zero is equal. Year, ticker, count and boolean answers must be equal.
    """
    if not isinstance(claimed, Answer) or not isinstance(executed, Answer):
        raise TypeError("compare_answers needs two Answer values")
    if claimed.kind != executed.kind or claimed.unit != executed.unit:
        return False
    c, e = claimed.value, executed.value
    if claimed.kind != "quantity":
        return type(c) is type(e) and c == e
    if not isinstance(c, Quantity) or not isinstance(e, Quantity) or c.dimension is not e.dimension:
        return False
    try:
        unit = display_unit(claimed.unit)
        shown_claim, shown_exec = to_display(c, unit), to_display(e, unit)
    except UnitError:
        return False
    places = max(2, _places(shown_claim))
    a, b = round_display(shown_claim, places), round_display(shown_exec, places)
    return a == b or abs(a - b) <= RELATIVE_TOLERANCE * max(abs(a), abs(b))


# ---- gates ----------------------------------------------------------------------------------
def _issue(code: GateCode, severity: Severity, message: str, *, line: int | None = None,
           step: int | None = None) -> GateIssue:
    return GateIssue(code, severity, message, line, step)


def _numeric_literals(module: ast.AST) -> list[ast.Constant]:
    nodes = (n for n in ast.walk(module)
             if isinstance(n, ast.Constant) and not isinstance(n.value, bool)
             and isinstance(n.value, (int, float, complex)))
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def _literal_allowed(value: Decimal, allowed: frozenset[Decimal]) -> bool:
    if value in allowed:
        return True
    return value == value.to_integral_value() and MIN_YEAR <= value <= MAX_YEAR


def _provenance_literals(code: str, context: ExecutionContext) -> list[GateIssue]:
    try:
        module = ast.parse(code, "<solution>", "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return [_issue("provenance", "error", "generated code cannot be parsed")]
    source = _SourceLines(code)
    allowed = ALLOWED_CONSTANTS | context.question_literals
    issues = []
    # A negative literal is UnaryOp(USub, Constant): the Constant itself is checked.
    for node in _numeric_literals(module):
        text = source.segment(node) or repr(node.value)
        try:
            value = Decimal(text)
            ok = value.is_finite() and _literal_allowed(value, allowed)
        except (InvalidOperation, ValueError):
            ok = False
        if not ok:
            issues.append(_issue(
                "provenance", "error",
                f"numeric literal {text} is not an allowed constant, a number stated in the "
                "question or a year", line=node.lineno))
    return issues


def _provenance_facts(steps: list[TraceStep], facts: FactSet) -> list[GateIssue]:
    issues = []
    seen: set[object] = set()
    for index, step in enumerate(steps):  # covers facts_used() and every cross_check step
        for key in step.fact_keys:
            if key in seen:
                continue
            seen.add(key)
            found = facts.find(key.metric_id, key.ticker, key.fiscal_year, key.scope,
                               key.period_kind) if isinstance(key, FactKey) else None
            if found is None:
                issues.append(_issue("provenance", "error",
                                     f"trace uses a fact that is not in the fact set: {key}",
                                     step=index))
            elif not isinstance(found.source, FactSource):
                issues.append(_issue("provenance", "error", f"fact without a source: {key}",
                                     step=index))
    return issues


def _formula(steps: list[TraceStep], tree: MetricTree, context: ExecutionContext,
             metrics_used: frozenset[str]) -> list[GateIssue]:
    # R2 + R17: a named leaf may be read by fact() or averaged by avg_balance()
    facts_read = {s.args[0] for s in steps if s.helper in ("fact", "avg_balance") and s.args}
    issues = []
    for node_id in sorted(context.named_metrics):
        try:
            kind = tree.kind(node_id)
        except KeyError:
            issues.append(_issue("formula", "error",
                                 f"named metric {node_id!r} is not in the metric tree"))
            continue
        if node_id in metrics_used or (kind == "leaf" and node_id in facts_read):
            continue
        how = "metric()" if kind == "derived" else "metric(), fact() or avg_balance()"
        issues.append(_issue("formula", "error",
                             f"the question names {node_id!r} but the program never reads it "
                             f"with {how}"))
    return issues


_MONEY_UNITS = frozenset(MONEY_SCALES) | {"money_per_share"}
_QUANTITY_DIMENSIONS: dict[str, tuple[Dimension, ...]] = {
    **{name: (Dimension.MONEY, Dimension.MONEY_PER_SHARE) for name in MONEY_SCALES},
    "money_per_share": (Dimension.MONEY_PER_SHARE,),
    "percent": (Dimension.RATIO,),
    "percentage_points": (Dimension.RATIO,),
    "times": (Dimension.RATIO,),
    "days": (Dimension.DAYS,),
}
_VALUE_TYPES: dict[str, type] = {"year": int, "count": int, "ticker": str, "boolean": bool}


def _units_compatible(requested: str, actual: str) -> bool:
    return requested == actual or (requested in _MONEY_UNITS and actual in _MONEY_UNITS)


def _unit_type(answer: Answer, context: ExecutionContext) -> list[GateIssue]:
    issues = []
    if answer.kind != context.requested_kind:
        issues.append(_issue("unit_type", "error",
                             f"the question asks for a {context.requested_kind} answer, "
                             f"the program returned a {answer.kind}"))
    unit_kind = ("quantity" if answer.unit in _QUANTITY_DIMENSIONS
                 else answer.unit if answer.unit in _VALUE_TYPES else None)
    if unit_kind is None or unit_kind != answer.kind:
        issues.append(_issue("unit_type", "error",
                             f"answer unit {answer.unit!r} does not fit kind {answer.kind}"))
    elif unit_kind == "quantity":
        if (not isinstance(answer.value, Quantity)
                or answer.value.dimension not in _QUANTITY_DIMENSIONS[answer.unit]):
            issues.append(_issue("unit_type", "error",
                                 f"answer value cannot be shown in {answer.unit!r}"))
    elif type(answer.value) is not _VALUE_TYPES[unit_kind]:
        issues.append(_issue("unit_type", "error",
                             f"{unit_kind} answer has a {type(answer.value).__name__} value"))
    if context.requested_unit is not None and not _units_compatible(context.requested_unit,
                                                                     answer.unit):
        issues.append(_issue("unit_type", "error",
                             f"the question asks for {context.requested_unit!r}, "
                             f"the program answered in {answer.unit!r}"))
    return issues


def _parse_result(text: str) -> Quantity | None:
    """Inverse of the helpers' rendering of a Quantity ("<value> <dimension>")."""
    value, _, dimension = text.rpartition(" ")
    try:
        return Quantity(Decimal(value), Dimension(dimension))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _sanity(answer: Answer, steps: list[TraceStep], tree: MetricTree) -> list[GateIssue]:
    issues = []
    last = next(((i, s) for i, s in reversed(list(enumerate(steps)))
                 if s.helper == "metric" and s.args), None)
    if last is not None:
        index, step = last
        node_id = step.args[0]
        node = _derived_node(tree, node_id)
        if node is not None and node.expected_range is not None:
            low, high = node.expected_range
            value = _parse_result(step.result)
            if value is None or value.dimension is not node.dimension:
                issues.append(_issue("sanity", "warning",
                                     f"cannot read the value of {node_id!r} for a range check",
                                     step=index))
            elif not low <= value.value <= high:
                issues.append(_issue("sanity", "warning",
                                     f"{node_id} = {value.value} is outside its expected range "
                                     f"[{low}, {high}]", step=index))
    value = answer.value
    if (isinstance(value, Quantity)
            and value.dimension in (Dimension.MONEY, Dimension.MONEY_PER_SHARE)
            and not abs(value.value) < MONEY_LIMIT):
        issues.append(_issue("sanity", "warning",
                             f"implausible money magnitude: {value.value} đồng"))
    return issues


def _derived_node(tree: MetricTree, node_id: object) -> DerivedDef | None:
    try:
        return tree.derived(node_id)  # type: ignore[arg-type]
    except (KeyError, TypeError):
        return None


def _restated(steps: list[TraceStep], index: int, facts: FactSet) -> bool:
    """Whether a fact used by the metric step this cross-check belongs to is restated."""
    check = steps[index]
    for step in reversed(steps[:index]):
        if step.helper == "metric" and step.args == check.args:
            return any(_is_restated(key, facts) for key in step.fact_keys)
    return False


def _is_restated(key: object, facts: FactSet) -> bool:
    if not isinstance(key, FactKey):
        return False
    found = facts.find(key.metric_id, key.ticker, key.fiscal_year, key.scope, key.period_kind)
    return found is not None and found.restated


def _decimals(detail, names: Iterable[str]) -> list[Decimal] | None:
    try:
        values = [Decimal(detail[name]) for name in names]
    except (KeyError, TypeError, InvalidOperation):
        return None
    return values if all(v.is_finite() for v in values) else None


def _cross_check(steps: list[TraceStep], tree: MetricTree, facts: FactSet) -> list[GateIssue]:
    issues = []
    for index, step in enumerate(steps):
        if step.helper != "cross_check":
            continue
        node = _derived_node(tree, step.detail.get("node"))
        values = _decimals(step.detail, ("primary", "alternative", "tolerance"))
        if values is None or node is None:
            issues.append(_issue("cross_check", "error", "malformed cross-check step",
                                 step=index))
            continue
        primary, alternative, tolerance = values
        difference = abs(primary - alternative)
        if node.dimension is Dimension.MONEY:
            mismatch = difference > tolerance  # D8: sum of the printed scales
        else:
            mismatch = difference > RELATIVE_TOLERANCE * abs(primary)
        if not mismatch:
            continue
        restated = _restated(steps, index, facts)
        issues.append(_issue(
            "cross_check", "warning" if restated else "error",
            f"{node.id}: primary {primary} vs alternative {alternative} "
            f"(tolerance {tolerance if node.dimension is Dimension.MONEY else '0.2%'})"
            + (" (a restated fact is involved)" if restated else ""), step=index))
    return issues


def run_gates(code: str, result: ExecutionResult, tree: MetricTree, facts: FactSet,
              context: ExecutionContext, claimed: Answer | None = None) -> GateReport:
    """Every gate issue for an executed program; `GateReport.passed` means no errors."""
    if not isinstance(result, ExecutionResult):
        raise TypeError(f"run_gates needs an ExecutionResult, got {type(result).__name__}")
    if not isinstance(code, str):
        raise TypeError("code must be a str")
    steps = list(result.trace.steps)
    issues = [
        *_provenance_literals(code, context),
        *_provenance_facts(steps, facts),
        *_formula(steps, tree, context, result.trace.metrics_used()),
        *_unit_type(result.answer, context),
        *_sanity(result.answer, steps, tree),
        *_cross_check(steps, tree, facts),
    ]
    if claimed is not None and not compare_answers(claimed, result.answer):
        issues.append(_issue("agreement", "error",
                             f"claimed {claimed.value!r} [{claimed.unit}] does not match executed "
                             f"{result.answer.value!r} [{result.answer.unit}]"))
    return GateReport(tuple(issues))

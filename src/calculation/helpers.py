"""Traced helper library: the only functions generated `solution()` code may call.

Every successful helper call appends one `TraceStep`; the trace feeds the verification gates.
Failed calls raise and record nothing. Missing data is never replaced by 0.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Literal, get_args

from src.calculation.expressions import Ref, evaluate
from src.calculation.facts import FactKey, FactSet, PeriodKind, Scope
from src.calculation.metric_tree import MetricTree, Template
from src.calculation.units import (
    MONEY_SCALES,
    CalculationError,
    Dimension,
    Quantity,
    UnitError,
    display_unit,
    to_display,
)

AnswerKind = Literal["quantity", "year", "ticker", "count", "boolean"]

_QUANTITY_UNITS = frozenset({*MONEY_SCALES, "percent", "percentage_points", "times", "days",
                             "money_per_share"})
_UNIT_KINDS: dict[str, AnswerKind] = {
    **{u: "quantity" for u in _QUANTITY_UNITS},
    "year": "year", "ticker": "ticker", "count": "count", "boolean": "boolean",
}
_OPS: dict[str, Callable[[object, object], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    # via ordering so Quantity dimension rules apply (dataclass == would silently be False)
    "==": lambda a, b: a <= b and a >= b,
}


class MetricUnavailable(LookupError):
    """The node does not apply to the ticker's template, or no alternative has all its facts."""


class _NoFacts(Exception):
    """Internal: an alternative cannot be evaluated because a fact is absent."""


@dataclass(frozen=True)
class Answer:
    kind: AnswerKind
    value: Quantity | int | str | bool
    unit: str


@dataclass(frozen=True)
class ExecutionContext:
    default_scope: Scope
    ticker_templates: Mapping[str, Template]
    question_literals: frozenset[Decimal]
    requested_unit: str | None
    requested_kind: AnswerKind
    named_metrics: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TraceStep:
    helper: str
    args: tuple
    result: str
    fact_keys: tuple[FactKey, ...]
    detail: Mapping[str, str] = field(default_factory=dict)


@dataclass
class Trace:
    steps: list[TraceStep] = field(default_factory=list)

    def facts_used(self) -> tuple[FactKey, ...]:
        """Unique fact keys in first-use order; cross-check alternatives are excluded."""
        seen: dict[FactKey, None] = {}
        for step in self.steps:
            if step.helper != "cross_check":
                seen.update(dict.fromkeys(step.fact_keys))
        return tuple(seen)

    def metrics_used(self) -> frozenset[str]:
        return frozenset(s.args[0] for s in self.steps if s.helper == "metric")

    def cross_checks(self) -> list[TraceStep]:
        return [s for s in self.steps if s.helper == "cross_check"]


# ---- argument validation --------------------------------------------------------------
def _number(value: object, what: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise TypeError(f"{what} must be an int or Decimal, got {type(value).__name__}")
    return Decimal(value)


def _integer(value: object, what: str) -> int:
    """Accept int or an integral Decimal (generated code passes literals as Decimal, D3)."""
    number = _number(value, what)
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{what} must be an integer")
    return int(number)


def _scope(scope: object, context: ExecutionContext) -> Scope:
    scope = context.default_scope if scope is None else scope
    if scope not in get_args(Scope):
        raise ValueError(f"invalid scope {scope!r}")
    return scope  # type: ignore[return-value]


def _quantity(value: object, what: str) -> Quantity:
    if not isinstance(value, Quantity):
        raise TypeError(f"{what} must be a Quantity, got {type(value).__name__}")
    return value


def _quantities(values: Iterable, what: str) -> list[Quantity]:
    items = [_quantity(v, what) for v in values]
    if not items:
        raise CalculationError(f"{what} of an empty collection")
    if any(q.dimension is not items[0].dimension for q in items):
        raise UnitError(f"{what} needs values of one dimension")
    return items


def _comparable(value: object) -> object:
    if isinstance(value, (bool, float)) or not isinstance(value, (Quantity, int, Decimal)):
        raise TypeError(f"cannot compare a {type(value).__name__}")
    return value


def _op(op: object) -> Callable[[object, object], bool]:
    if not isinstance(op, str) or op not in _OPS:
        raise ValueError(f"unsupported operator {op!r}")
    return _OPS[op]


def _render(value: object) -> str:
    if isinstance(value, Quantity):
        return f"{value.value} {value.dimension.value}"
    if isinstance(value, Answer):
        return f"{_render(value.value)} [{value.unit}]"
    return str(value)


def _freeze(value: object) -> object:
    """Immutable snapshot of a helper argument for the trace."""
    if isinstance(value, Mapping):
        return tuple((k, _freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


def _unique(keys: Iterable[FactKey]) -> tuple[FactKey, ...]:
    return tuple(dict.fromkeys(keys))


# ---- builder ------------------------------------------------------------------------
def build_helpers(tree: MetricTree, facts: FactSet, context: ExecutionContext,
                  trace: Trace) -> dict[str, Callable]:
    """Build the helper namespace for one execution; every call is appended to `trace`."""

    def record(helper: str, args: tuple, result: object, fact_keys=(), detail=None) -> None:
        trace.steps.append(TraceStep(helper, tuple(_freeze(a) for a in args), _render(result),
                                     _unique(fact_keys), dict(detail or {})))

    # -- tree evaluation --
    def eval_node(node_id: str, ticker: str, year: int, scope: Scope, template: Template,
                  sink: list[TraceStep]) -> tuple[Quantity, tuple[FactKey, ...]]:
        """Value and fact keys of a node; derived nodes append their steps to `sink`."""
        args = (node_id, ticker, year, scope)
        if not tree.applies(node_id, template):
            raise MetricUnavailable(f"{node_id!r} does not apply to template {template}")
        if tree.kind(node_id) == "leaf":
            leaf = tree.leaf(node_id, template)
            found = facts.find(node_id, ticker, year, scope, leaf.period_kind)
            if found is None:
                raise _NoFacts(node_id)
            return found.value, (found.key,)
        computed: list[tuple[Quantity, tuple[FactKey, ...], list[TraceStep]]] = []
        for formula in tree.derived(node_id).formulas:
            local: list[TraceStep] = []
            keys: list[FactKey] = []

            def resolve(ref: Ref, local=local, keys=keys) -> Quantity:
                value, used = eval_node(ref.node_id, ticker, year - ref.offset, scope,
                                        template, local)
                keys.extend(used)
                return value

            try:
                value = evaluate(formula, resolve)
            except _NoFacts:
                continue  # this alternative lacks facts; errors other than absence propagate
            computed.append((value, _unique(keys), local))
        if not computed:
            raise _NoFacts(node_id)
        primary, primary_keys, primary_steps = computed[0]
        sink.extend(primary_steps)
        sink.append(TraceStep("metric", args, _render(primary), primary_keys))
        for alt, alt_keys, _ in computed[1:]:
            tolerance = sum((facts.get(k).source.scale for k in {*primary_keys, *alt_keys}),
                            Decimal(0))
            sink.append(TraceStep("cross_check", args, _render(alt), alt_keys, {
                "node": node_id, "primary": str(primary.value), "alternative": str(alt.value),
                "tolerance": str(tolerance)}))
        return primary, primary_keys

    # -- data access --
    def fact(metric_id: str, ticker: str, year: int, period: PeriodKind = "flow",
             scope: Scope | None = None) -> Quantity:
        """Read one reported line item (a leaf) as a Quantity; money is in dong.

        period: "flow" (income/cash-flow lines) or "closing" (balance-sheet lines).
        scope: "consolidated" or "separate"; None uses the question's default scope.
        Raises MissingFact when the fact was not extracted. Never returns 0 for missing data.
        """
        if period not in get_args(PeriodKind):
            raise ValueError(f"invalid period {period!r}")
        key = FactKey(metric_id, ticker, _integer(year, "year"), _scope(scope, context), period)
        value = facts.get(key).value
        record("fact", (metric_id, ticker, key.fiscal_year, period, key.scope), value, (key,))
        return value

    def metric(node_id: str, ticker: str, year: int, scope: Scope | None = None) -> Quantity:
        """Evaluate a metric-tree node (derived metric or leaf) for one ticker and year.

        Uses the first formula alternative whose facts all exist; the other computable
        alternatives are recorded as cross-checks. Raises MetricUnavailable when the node does
        not apply to the ticker's template (e.g. gross_margin for a bank) or no alternative
        has all its facts.
        """
        year = _integer(year, "year")
        scope = _scope(scope, context)
        template = context.ticker_templates.get(ticker)
        if template is None:
            raise MetricUnavailable(f"unknown ticker {ticker!r}")
        steps: list[TraceStep] = []
        try:
            value, keys = eval_node(node_id, ticker, year, scope, template, steps)
        except _NoFacts:
            raise MetricUnavailable(f"no alternative of {node_id!r} has all its facts") from None
        if not steps:  # a leaf read through metric() (R2)
            steps.append(TraceStep("metric", (node_id, ticker, year, scope), _render(value), keys))
        trace.steps.extend(steps)
        return value

    def avg_balance(metric_id: str, ticker: str, year: int,
                    scope: Scope | None = None) -> Quantity:
        """Average of the closing balances of `year` and `year - 1`: (closing(t) + closing(t-1)) / 2.

        Raises MissingFact when either balance is absent.
        """
        year = _integer(year, "year")
        scope = _scope(scope, context)
        keys = (FactKey(metric_id, ticker, year, scope, "closing"),
                FactKey(metric_id, ticker, year - 1, scope, "closing"))
        closing, opening = (facts.get(k).value for k in keys)
        value = (closing + opening) / 2
        record("avg_balance", (metric_id, ticker, year, scope), value, keys)
        return value

    # -- calculations --
    def growth_pct(new: Quantity, old: Quantity) -> Quantity:
        """Growth rate (new - old) / abs(old) as a RATIO; old == 0 raises CalculationError."""
        new, old = _quantity(new, "new"), _quantity(old, "old")
        value = (new - old) / abs(old)
        record("growth_pct", (new, old), value)
        return value

    def cagr(end: Quantity, start: Quantity, years: int) -> Quantity:
        """Compound annual growth (end / start) ** (1 / years) - 1 as a RATIO.

        Needs end > 0, start > 0 (same dimension) and an integer years >= 1.
        """
        end, start = _quantity(end, "end"), _quantity(start, "start")
        n = _integer(years, "years")
        if n < 1:
            raise CalculationError("cagr needs years >= 1")
        if end.dimension is not start.dimension:
            raise UnitError(f"cagr needs one dimension, got {end.dimension} and {start.dimension}")
        if end.value <= 0 or start.value <= 0:
            raise CalculationError("cagr needs positive end and start")
        with localcontext() as ctx:
            ctx.prec = 28
            ratio = end.value / start.value
            value = Quantity(ratio ** (Decimal(1) / n) - 1, Dimension.RATIO)
        record("cagr", (end, start, n), value)
        return value

    def pct_point_change(new_ratio: Quantity, old_ratio: Quantity) -> Quantity:
        """Difference of two ratios (new - old), shown in percentage points."""
        new, old = _quantity(new_ratio, "new_ratio"), _quantity(old_ratio, "old_ratio")
        if new.dimension is not Dimension.RATIO or old.dimension is not Dimension.RATIO:
            raise UnitError("pct_point_change needs two ratios")
        value = new - old
        record("pct_point_change", (new, old), value)
        return value

    def mean(values: Iterable[Quantity]) -> Quantity:
        """Arithmetic mean of non-empty Quantities of one dimension."""
        items = _quantities(values, "mean")
        value = sum(items[1:], items[0]) / len(items)
        record("mean", (items,), value)
        return value

    def median(values: Iterable[Quantity]) -> Quantity:
        """Median of non-empty Quantities of one dimension; even count -> mean of middle two."""
        items = _quantities(values, "median")
        ordered = sorted(items, key=lambda q: q.value)
        mid = len(ordered) // 2
        value = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        record("median", (items,), value)
        return value

    def _extreme(name: str, mapping: Mapping, better: Callable[[object, object], bool]):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} needs a mapping")
        if not mapping:
            raise CalculationError(f"{name} of an empty mapping")
        items = [(k, _comparable(v)) for k, v in mapping.items()]
        best_key, best = items[0]
        tie = False
        for key, value in items[1:]:
            if better(value, best):
                best_key, best, tie = key, value, False
            elif not better(best, value):
                tie = True
        if tie:
            raise CalculationError(f"{name} has a tie")
        record(name, (mapping,), best_key)
        return best_key

    def argmax(mapping: Mapping):
        """Key of the largest value; a tie or an empty mapping raises CalculationError."""
        return _extreme("argmax", mapping, lambda a, b: a > b)

    def argmin(mapping: Mapping):
        """Key of the smallest value; a tie or an empty mapping raises CalculationError."""
        return _extreme("argmin", mapping, lambda a, b: a < b)

    def filter_by(mapping: Mapping, op: str, threshold) -> dict:
        """Sub-mapping whose values satisfy `value op threshold`; op in <, <=, >, >=, ==.

        Build thresholds with money(...) or percent(...); a bare number only compares with
        ratios and counts (or is exactly 0).
        """
        test = _op(op)
        if not isinstance(mapping, Mapping):
            raise TypeError("filter_by needs a mapping")
        threshold = _comparable(threshold)
        value = {k: v for k, v in mapping.items() if test(_comparable(v), threshold)}
        record("filter_by", (mapping, op, threshold), value)
        return value

    def count_if(values: Iterable, op: str, threshold) -> int:
        """Number of values satisfying `value op threshold`; same rules as filter_by."""
        test = _op(op)
        threshold = _comparable(threshold)
        items = list(values)
        value = sum(1 for v in items if test(_comparable(v), threshold))
        record("count_if", (items, op, threshold), value)
        return value

    # -- thresholds and result --
    def money(value, scale: str) -> Quantity:
        """Money threshold: value in the named scale (dong, nghin, trieu, ty, tram_ty, nghin_ty)."""
        number = _number(value, "money value")
        if scale not in MONEY_SCALES:
            raise UnitError(f"unknown money scale {scale!r}")
        result = Quantity(number * MONEY_SCALES[scale], Dimension.MONEY)
        record("money", (value, scale), result)
        return result

    def percent(value) -> Quantity:
        """Ratio threshold from a percentage: percent(50) is the ratio 0.5."""
        result = Quantity(_number(value, "percent value") / 100, Dimension.RATIO)
        record("percent", (value,), result)
        return result

    def answer(value, unit: str) -> Answer:
        """Final result. unit: a money scale, percent, percentage_points, times, days,
        money_per_share (value is a Quantity), year (int), ticker (str), count (int >= 0) or
        boolean (bool)."""
        kind = _UNIT_KINDS.get(unit) if isinstance(unit, str) else None
        if kind is None:
            raise ValueError(f"unknown answer unit {unit!r}")
        if kind == "quantity":
            if not isinstance(value, Quantity):
                raise ValueError(f"unit {unit!r} needs a Quantity")
            to_display(value, display_unit(unit))  # UnitError (a ValueError) on mismatch
        elif kind in ("year", "count"):
            try:
                value = _integer(value, kind)
            except TypeError:
                raise ValueError(f"unit {unit!r} needs an integer") from None
            if kind == "count" and value < 0:
                raise ValueError("count must be >= 0")
        elif kind == "ticker":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("ticker must be a non-empty string")
        elif not isinstance(value, bool):
            raise ValueError("boolean answer needs a bool")
        result = Answer(kind, value, unit)
        record("answer", (value, unit), result)
        return result

    return {
        "fact": fact, "metric": metric, "avg_balance": avg_balance, "growth_pct": growth_pct,
        "cagr": cagr, "pct_point_change": pct_point_change, "mean": mean, "median": median,
        "argmax": argmax, "argmin": argmin, "filter_by": filter_by, "count_if": count_if,
        "money": money, "percent": percent, "answer": answer,
    }

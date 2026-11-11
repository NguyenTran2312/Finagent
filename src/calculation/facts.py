"""Sourced facts and the conflict rules applied when a FactSet is built."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, get_args

from src.calculation.units import Quantity

Scope = Literal["consolidated", "separate"]
PeriodKind = Literal["flow", "closing"]


class FactConflict(ValueError):
    """Two facts for one key disagree beyond the printed-scale tolerance."""


class MissingFact(LookupError):
    """No fact exists for the requested key."""


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_year(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("fiscal_year must be an int")


@dataclass(frozen=True)
class FactSource:
    chunk_id: str
    doc_id: str
    page: int
    row_label: str
    raw_text: str
    unit_text: str
    scale: Decimal
    column_header: str | None = None

    def __post_init__(self) -> None:
        for name in ("chunk_id", "doc_id", "row_label", "raw_text", "unit_text"):
            _require_text(name, getattr(self, name))
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 0:
            raise ValueError("page must be an int >= 0")
        if (
            not isinstance(self.scale, Decimal)
            or not self.scale.is_finite()
            or self.scale <= 0
        ):
            raise ValueError("scale must be a finite Decimal > 0")


@dataclass(frozen=True)
class FactKey:
    metric_id: str
    ticker: str
    fiscal_year: int
    scope: Scope
    period_kind: PeriodKind


@dataclass(frozen=True)
class Fact:
    metric_id: str
    ticker: str
    fiscal_year: int
    scope: Scope
    period_kind: PeriodKind
    value: Quantity
    source: FactSource
    restated: bool = False

    def __post_init__(self) -> None:
        _require_text("metric_id", self.metric_id)
        _require_text("ticker", self.ticker)
        _require_year(self.fiscal_year)
        if self.scope not in get_args(Scope):
            raise ValueError(f"invalid scope: {self.scope!r}")
        if self.period_kind not in get_args(PeriodKind):
            raise ValueError(f"invalid period_kind: {self.period_kind!r}")
        if not isinstance(self.value, Quantity):
            raise TypeError("value must be a Quantity")
        if not isinstance(self.source, FactSource):
            raise TypeError("source must be a FactSource")
        if not isinstance(self.restated, bool):
            raise TypeError("restated must be a bool")

    @property
    def key(self) -> FactKey:
        return FactKey(
            self.metric_id, self.ticker, self.fiscal_year, self.scope, self.period_kind
        )


def _describe(key: FactKey) -> str:
    return (
        f"{key.metric_id}/{key.ticker}/{key.fiscal_year}/{key.scope}/{key.period_kind}"
    )


def _finest(group: list[Fact]) -> Fact:
    """Smallest printed scale; exact ties keep the first in input order."""
    kept = group[0]
    for fact in group[1:]:
        if fact.source.scale < kept.source.scale:
            kept = fact
    return kept


def _require_agreement(key: FactKey, group: list[Fact]) -> None:
    """Every pair must agree within the larger of the two printed scales."""
    for i, left in enumerate(group):
        for right in group[i + 1:]:
            tolerance = max(left.source.scale, right.source.scale)
            if abs(left.value.value - right.value.value) > tolerance:
                raise FactConflict(
                    f"conflicting facts for {_describe(key)}: "
                    f"{left.value.value} ({left.source.chunk_id}) vs "
                    f"{right.value.value} ({right.source.chunk_id})"
                )


def _resolve_group(key: FactKey, group: list[Fact], warnings: list[str]) -> Fact:
    """Resolve all facts of one key; independent of input order except exact ties."""
    first = group[0]
    for fact in group[1:]:
        if fact.value.dimension is not first.value.dimension:
            raise FactConflict(
                f"dimension mismatch for {_describe(key)}: "
                f"{first.value.dimension} vs {fact.value.dimension}"
            )
    restated = [f for f in group if f.restated]
    if len(restated) == 1:
        if len(group) > 1:
            warnings.append(f"restated value used for {_describe(key)}")
        return restated[0]
    candidates = restated or group
    _require_agreement(key, candidates)
    return _finest(candidates)


class FactSet:
    def __init__(self, facts: Iterable[Fact]) -> None:
        groups: dict[FactKey, list[Fact]] = {}
        for fact in facts:
            if not isinstance(fact, Fact):
                raise TypeError("FactSet items must be Fact instances")
            groups.setdefault(fact.key, []).append(fact)
        warnings: list[str] = []
        self._facts = {key: _resolve_group(key, group, warnings) for key, group in groups.items()}
        self.warnings: tuple[str, ...] = tuple(warnings)

    def get(self, key: FactKey) -> Fact:
        try:
            return self._facts[key]
        except KeyError:
            raise MissingFact(_describe(key)) from None

    def find(
        self,
        metric_id: str,
        ticker: str,
        year: int,
        scope: Scope,
        period_kind: PeriodKind,
    ) -> Fact | None:
        return self._facts.get(FactKey(metric_id, ticker, year, scope, period_kind))

    def opening(self, metric_id: str, ticker: str, year: int, scope: Scope) -> Fact:
        return self.get(FactKey(metric_id, ticker, year - 1, scope, "closing"))

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts.values())

    def __len__(self) -> int:
        return len(self._facts)

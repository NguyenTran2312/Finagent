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


def _resolve(kept: Fact, new: Fact, warnings: list[str]) -> Fact:
    """Apply the D1 conflict rules to the currently kept fact and the next one."""
    key = kept.key
    if kept.value.dimension is not new.value.dimension:
        raise FactConflict(
            f"dimension mismatch for {_describe(key)}: "
            f"{kept.value.dimension} vs {new.value.dimension}"
        )
    tolerance = max(kept.source.scale, new.source.scale)
    beyond = abs(kept.value.value - new.value.value) > tolerance

    if kept.restated != new.restated:
        winner = kept if kept.restated else new
        warnings.append(f"restated value used for {_describe(key)}")
        return winner
    if beyond:
        raise FactConflict(
            f"conflicting facts for {_describe(key)}: "
            f"{kept.value.value} ({kept.source.chunk_id}) vs "
            f"{new.value.value} ({new.source.chunk_id})"
        )
    if new.source.scale < kept.source.scale:
        return new
    return kept


class FactSet:
    def __init__(self, facts: Iterable[Fact]) -> None:
        by_key: dict[FactKey, Fact] = {}
        warnings: list[str] = []
        for fact in facts:
            if not isinstance(fact, Fact):
                raise TypeError("FactSet items must be Fact instances")
            current = by_key.get(fact.key)
            by_key[fact.key] = fact if current is None else _resolve(current, fact, warnings)
        self._facts = by_key
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

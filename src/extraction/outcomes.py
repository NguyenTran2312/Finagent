"""Typed per-request outcomes of leaf extraction (spec §7.1)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Union

from src.calculation.facts import Fact, Scope
from src.calculation.metric_tree import LeafRequest

Method = Literal["code", "label", "section_position", "llm_located"]


@dataclass(frozen=True)
class CellLocation:
    chunk_id: str
    doc_id: str
    page: int
    row_label: str
    column_header: str
    raw_text: str


@dataclass(frozen=True)
class Candidate:
    value: Decimal | None
    location: CellLocation


@dataclass(frozen=True)
class ExtractedFact:
    fact: Fact
    method: Method
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Found:
    ticker: str
    scope: Scope
    request: LeafRequest
    extracted: ExtractedFact


@dataclass(frozen=True)
class NotFound:
    ticker: str
    scope: Scope
    request: LeafRequest
    reason: str
    searched_doc_ids: tuple[str, ...]
    skipped_tables: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BlankValue:
    ticker: str
    scope: Scope
    request: LeafRequest
    location: CellLocation


@dataclass(frozen=True)
class Ambiguous:
    ticker: str
    scope: Scope
    request: LeafRequest
    reason: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class Conflict:
    ticker: str
    scope: Scope
    request: LeafRequest
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class UnitUnknown:
    ticker: str
    scope: Scope
    request: LeafRequest
    reason: str
    location: CellLocation


@dataclass(frozen=True)
class ReportMissing:
    ticker: str
    scope: Scope
    request: LeafRequest
    available_scopes: tuple[str, ...]


@dataclass(frozen=True)
class NotApplicable:
    ticker: str
    scope: Scope
    request: LeafRequest
    template: str


@dataclass(frozen=True)
class UnsupportedTemplate:
    ticker: str
    scope: Scope
    request: LeafRequest
    template: str


@dataclass(frozen=True)
class UnknownTicker:
    ticker: str
    scope: Scope
    request: LeafRequest


Outcome = Union[
    Found,
    NotFound,
    BlankValue,
    Ambiguous,
    Conflict,
    UnitUnknown,
    ReportMissing,
    NotApplicable,
    UnsupportedTemplate,
    UnknownTicker,
]

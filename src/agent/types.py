from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.calculation import Answer, FactSource

Scope = Literal["consolidated", "separate"]


@dataclass(frozen=True)
class DataNeed:
    node_id: str
    ticker: str
    year: int
    scope: Scope | None = None
    helper: Literal["fact", "metric", "avg_balance"] = "metric"


@dataclass(frozen=True)
class LeafLog:
    leaf_id: str
    ticker: str
    year: int
    scope: Scope
    outcome: str  # outcome class name, e.g. "Found", "NotFound"
    reason: str | None
    value: Decimal | None
    chunk_id: str | None
    doc_id: str | None
    page: int | None
    row_label: str | None
    tool: Literal["extractor", "locator", "locator_skipped"]


@dataclass(frozen=True)
class AgentConfig:
    retriever_mode: Literal["none", "bm25", "hybrid", "hybrid_rerank"] = "none"
    locator_enabled: bool = True
    max_repairs: int = 3  # code attempts = 1 + max_repairs
    max_gather_rounds: int = 2  # extra gather rounds after the first
    max_locator_calls: int = 6  # per question
    default_scope: Scope = "consolidated"


@dataclass(frozen=True)
class StepEvent:
    kind: Literal["plan", "metrics", "gather", "solve", "answer"]
    title: str  # Vietnamese, shown as the block header
    data: Mapping[str, object]  # JSON-like; present.py knows each kind's shape


@dataclass(frozen=True)
class AgentRun:
    status: Literal["answered", "clarify", "refused", "error"]
    text: str  # answer, clarification request, refusal or error message
    reason: str | None  # refusal / error code, None when answered
    answer: Answer | None
    sources: tuple[FactSource, ...]  # facts used by the accepted program
    events: tuple[StepEvent, ...]
    llm_calls: int
    seconds: float

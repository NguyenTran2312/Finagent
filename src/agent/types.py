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


RETRIEVER_MODES = ("none", "bm25", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class AgentConfig:
    retriever_mode: Literal["none", "bm25", "hybrid", "hybrid_rerank"] = "none"
    locator_enabled: bool = True
    max_repairs: int = 3  # code attempts = 1 + max_repairs
    max_gather_rounds: int = 2  # extra gather rounds after the first
    max_locator_calls: int = 6  # per question
    default_scope: Scope = "consolidated"
    max_plan_repairs: int = 1  # paraphrase calls = at most 1 + max_plan_repairs

    def __post_init__(self) -> None:
        if self.retriever_mode not in RETRIEVER_MODES:
            raise ValueError(f"retriever_mode must be one of {RETRIEVER_MODES}")
        if self.default_scope not in ("consolidated", "separate"):
            raise ValueError("default_scope must be 'consolidated' or 'separate'")
        if not isinstance(self.locator_enabled, bool):
            raise ValueError("locator_enabled must be a bool")
        for name in ("max_repairs", "max_gather_rounds", "max_locator_calls", "max_plan_repairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an int >= 0")


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
    evidence: tuple[LeafLog, ...] = ()  # every leaf outcome gathered during the run, in order

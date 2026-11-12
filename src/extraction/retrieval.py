"""Table retriever protocol and retrieval trace for leaf extraction.

Standard library only at runtime: this module defines the seam through which
the extractor asks for a table ranking. Concrete retrievers live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from src.calculation.facts import Scope
    from src.calculation.metric_tree import LeafDef


@dataclass(frozen=True)
class RankedTable:
    chunk_id: str
    rank: int


class TableRetriever(Protocol):
    mode: str

    def rank(
        self, ticker: str, scope: Scope, report_year: int, leaf: LeafDef
    ) -> tuple[RankedTable, ...]: ...


class TableRetrievalError(RuntimeError):
    """A retrieval backend failed while ranking tables."""


@dataclass(frozen=True)
class RetrievalTrace:
    mode: str
    step: Literal["retrieved", "full_scan"]
    rank: int | None


def ranking_of(answer: object) -> dict[str, int]:
    """Validate a retriever answer and return chunk_id -> rank."""
    if not isinstance(answer, tuple):
        raise ValueError("retriever answer must be a tuple")
    if not all(isinstance(item, RankedTable) for item in answer):
        raise ValueError("retriever answer items must be RankedTable")
    ranks = sorted(item.rank for item in answer)
    if ranks != list(range(1, len(answer) + 1)):
        raise ValueError("retriever ranks must be exactly 1..N")
    ranking = {item.chunk_id: item.rank for item in answer}
    if len(ranking) != len(answer):
        raise ValueError("retriever chunk ids must be distinct")
    return ranking

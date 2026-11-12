"""Table retriever for leaf extraction: BM25 (+ BGE-M3 dense via Chroma) with RRF.

Implements the ``TableRetriever`` protocol of ``src.extraction.retrieval``. This
module must stay import-light: torch, chromadb and embedding models are loaded
lazily by the code that opens the real indexes, never at import time.
"""

from __future__ import annotations

from config.settings import RRF_CONSTANT
from src.calculation.metric_tree import LeafDef
from src.extraction.retrieval import RankedTable, TableRetrievalError
from src.indexing.table_semantics import StatementKind
from src.main.refine_questions import retrieve_refined
from src.retrieval.filters import normalize_metadata_filters
from src.retrieval.hybrid import rrf_fuse

MODES = ("hybrid", "bm25")

STATEMENT_NAMES: dict[StatementKind, str] = {
    StatementKind.BALANCE_SHEET: "bảng cân đối kế toán",
    StatementKind.INCOME_STATEMENT: "báo cáo kết quả hoạt động kinh doanh",
    StatementKind.CASH_FLOW: "báo cáo lưu chuyển tiền tệ",
}


def table_query(leaf: LeafDef) -> str:
    name = STATEMENT_NAMES.get(leaf.statement_kind) if leaf.source == "statement" else None
    return f"{leaf.label_vi} {name}" if name else leaf.label_vi


def _is_int(value: object) -> bool:
    return type(value) is int


class HybridTableRetriever:
    def __init__(self, bm25, dense_store, *, mode: str = "hybrid", k: int = 10, depth: int = 20):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not _is_int(k) or k < 1:
            raise ValueError("k must be a positive integer")
        if not _is_int(depth) or depth < k:
            raise ValueError("depth must be an integer at least k")
        if mode == "hybrid" and dense_store is None:
            raise ValueError("hybrid mode requires a dense store")
        self.bm25, self.dense_store = bm25, dense_store
        self.mode, self.k, self.depth = mode, k, depth

    def rank(self, ticker: str, scope: str, report_year: int, leaf: LeafDef) -> tuple[RankedTable, ...]:
        query = table_query(leaf)
        filters = normalize_metadata_filters(
            {"tickers": [ticker.strip().upper()], "years": [report_year], "scopes": [scope]}
        )
        try:
            if self.mode == "bm25":
                documents = retrieve_refined(self.bm25, query, k=self.k, metadata_filters=filters)["documents"]
            else:
                lexical = retrieve_refined(
                    self.bm25, query, k=self.depth, metadata_filters=filters)["documents"]
                dense = self.dense_store.similarity_search(
                    query, k=self.depth, filter=_where(filters))
                documents = rrf_fuse(lexical, dense, k=self.k, c=RRF_CONSTANT)
            return tuple(
                RankedTable(document.metadata["chunk_id"], index)
                for index, document in enumerate(documents, 1)
            )
        except Exception as exc:
            raise TableRetrievalError(f"{self.mode} retrieval failed: {type(exc).__name__}") from exc


def _where(filters: dict) -> dict:
    scopes = filters["scopes"]
    if "separate" in scopes:
        scopes = [*scopes, "parent"]
    return {"$and": [
        {"ticker": {"$in": filters["tickers"]}},
        {"report_year": {"$in": filters["years"]}},
        {"scope": {"$in": scopes}},
    ]}

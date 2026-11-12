"""Table retriever for leaf extraction: BM25 (+ BGE-M3 dense via Chroma) with RRF.

Implements the ``TableRetriever`` protocol of ``src.extraction.retrieval``. This
module must stay import-light: torch, chromadb and embedding models are loaded
lazily by the code that opens the real indexes, never at import time. The source Chroma
index is never opened in place; a snapshot copy is opened instead (see ``open_table_retriever``).

Both branches search a per-report candidate set (spec §5.5): the BM25 documents of one
``(ticker, report_year, scope)`` are found once and kept in an LRU of ``CANDIDATE_CACHE_SIZE``
reports. BM25 scores only those documents (``rank_indices``). The dense branch fetches their
stored vectors once per report (``store.get``) and scores them exactly by dot product with the
query embedding (normalised vectors, so cosine), which is cached per query text; Chroma's
filtered ``similarity_search`` is not used.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import PROJECT_ROOT, RRF_CONSTANT
from src.calculation.metric_tree import LeafDef
from src.extraction.retrieval import RankedTable, TableRetrievalError
from src.indexing.bm25_store import load_bm25, sqlite_source_identity
from src.indexing.chroma import chroma_snapshot
from src.indexing.table_semantics import StatementKind
from src.main.refine_questions import filter_document_indices, rank_indices
from src.retrieval.filters import normalize_metadata_filters
from src.retrieval.hybrid import rrf_fuse

MODES = ("hybrid", "bm25")
COLLECTION = "vifinqa_tables"
CANDIDATE_CACHE_SIZE = 32

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


@dataclass
class _Candidates:
    """One report's BM25 document positions and chunk ids; ``dense`` is filled on first use."""

    indices: list[int]
    chunk_ids: list[str]
    dense: tuple | None = field(default=None, repr=False)  # (matrix, metadatas), aligned rows


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
        self._cache: OrderedDict[tuple, _Candidates] = OrderedDict()
        self._query_vectors: dict[str, list[float]] = {}

    def rank(self, ticker: str, scope: str, report_year: int, leaf: LeafDef) -> tuple[RankedTable, ...]:
        query = table_query(leaf)
        filters = normalize_metadata_filters(
            {"tickers": [ticker.strip().upper()], "years": [report_year], "scopes": [scope]}
        )
        candidates = self._candidates(filters)
        if not candidates.indices:
            return ()
        try:
            if self.mode == "bm25":
                documents = rank_indices(self.bm25, query, candidates.indices, self.k)[0]
            else:
                lexical = rank_indices(self.bm25, query, candidates.indices, self.depth)[0]
                dense = self._dense(query, candidates)
        except Exception as exc:
            raise TableRetrievalError(f"{self.mode} retrieval failed: {type(exc).__name__}") from exc
        if self.mode == "hybrid":
            documents = rrf_fuse(lexical, dense, k=self.k, c=RRF_CONSTANT)
        return tuple(
            RankedTable(document.metadata["chunk_id"], index)
            for index, document in enumerate(documents, 1)
        )

    def _candidates(self, filters: dict) -> _Candidates:
        key = (filters["tickers"][0], filters["years"][0], tuple(filters["scopes"]))
        candidates = self._cache.get(key)
        if candidates is not None:
            self._cache.move_to_end(key)
            return candidates
        indices = filter_document_indices(self.bm25.docs, filters)
        candidates = _Candidates(indices, [self.bm25.docs[i].metadata["chunk_id"] for i in indices])
        self._cache[key] = candidates
        while len(self._cache) > CANDIDATE_CACHE_SIZE:
            self._cache.popitem(last=False)
        return candidates

    def _dense(self, query: str, candidates: _Candidates) -> list:
        """Top ``depth`` candidates by exact cosine; ties keep candidate order."""
        import numpy as np
        from langchain_core.documents import Document

        if candidates.dense is None:
            fetched = self.dense_store.get(ids=candidates.chunk_ids, include=["embeddings", "metadatas"])
            rows = {chunk_id: position for position, chunk_id in enumerate(fetched["ids"])}
            order = [rows[chunk_id] for chunk_id in candidates.chunk_ids if chunk_id in rows]
            embeddings = fetched["embeddings"]
            matrix = np.asarray([embeddings[position] for position in order], dtype=np.float32)
            candidates.dense = (matrix, [fetched["metadatas"][position] for position in order])
        matrix, metadatas = candidates.dense
        if not metadatas:
            return []
        vector = self._query_vectors.get(query)
        if vector is None:
            vector = self.dense_store.embeddings.embed_query(query)
            self._query_vectors[query] = vector
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        top = np.argsort(-scores, kind="stable")[:self.depth]
        return [Document(page_content="", metadata=dict(metadatas[int(i)])) for i in top]


def _chroma_client(path):
    import chromadb

    return chromadb.PersistentClient(path=str(path))


def _embeddings(device, fp16):
    from src.indexing.embedding import get_embeddings

    return get_embeddings(device=device, fp16=fp16)


def _chroma_store(client, embeddings):
    from langchain_chroma import Chroma

    return Chroma(client=client, collection_name=COLLECTION, embedding_function=embeddings,
                  create_collection_if_not_exists=False)


def open_table_retriever(db_path, bm25_dir, *, bm25_name: str = "bm25-table", chroma_dir=None,
                         mode: str = "hybrid", device: str = "cuda", fp16: bool = True,
                         k: int = 10, depth: int = 20) -> HybridTableRetriever:
    """Open the table indexes, failing fast on any mismatch (arguments, corpus identity, counts).

    The source Chroma index is never opened directly (opening can migrate or rewrite it): hybrid
    mode opens a snapshot copy at ``<chroma_dir parent>/loaded/<sha>``. The first hybrid start
    copies the index once (~3 GB); later starts reuse the copy.
    """
    HybridTableRetriever(None, object(), mode=mode, k=k, depth=depth)  # validates mode / k / depth
    if mode == "hybrid":
        if chroma_dir is None:
            raise ValueError("hybrid mode requires chroma_dir")
        if fp16 is not True:
            raise ValueError("the table Chroma index holds fp16 BGE-M3 vectors; fp16 must be True")
        if not str(device).startswith("cuda"):
            raise ValueError("fp16 table embeddings need a CUDA device")
    bm25 = load_bm25(bm25_dir, bm25_name,
                     expected_identity=sqlite_source_identity(db_path, chunk_types=("table",)))
    dense_store = None
    if mode == "hybrid":
        snapshot = chroma_snapshot(chroma_dir, cache_root=Path(chroma_dir).resolve().parent)
        client = _chroma_client(snapshot)
        count = client.get_collection(COLLECTION).count()
        if count != len(bm25.docs):
            raise ValueError(
                f"Chroma collection {COLLECTION!r} has {count} documents but the BM25 index has "
                f"{len(bm25.docs)}; rebuild the stale index")
        dense_store = _chroma_store(client, _embeddings(device, fp16))
    return HybridTableRetriever(bm25, dense_store, mode=mode, k=k, depth=depth)


def add_retriever_arguments(parser) -> None:
    parser.add_argument("--retriever", choices=("none", "bm25", "hybrid"), default="none")
    parser.add_argument("--bm25-dir", default=str(Path(PROJECT_ROOT) / "data" / "indexes" / "bm25"))
    parser.add_argument("--bm25-name", default="bm25-table")
    parser.add_argument("--chroma-dir", default=None)
    parser.add_argument("--device", default="cuda")


def retriever_from_args(args, db_path) -> HybridTableRetriever | None:
    if args.retriever == "none":
        return None
    return open_table_retriever(db_path, args.bm25_dir, bm25_name=args.bm25_name,
                                chroma_dir=args.chroma_dir, mode=args.retriever, device=args.device)

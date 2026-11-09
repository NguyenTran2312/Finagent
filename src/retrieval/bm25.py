"""PostgreSQL chunks -> LangChain Documents -> BM25Retriever.

Dependencies: pip install psycopg langchain-core langchain-community rank-bm25
"""

import json
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.rows import dict_row

from config.settings import postgres_connection_params

from preprocess.cleaner import tokens

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever


def load_corpus_documents(
    connection_params: dict[str, Any] | None = None,
) -> list[Document]:
    """Load a completed corpus from one consistent, read-only snapshot."""
    from langchain_core.documents import Document

    params = connection_params or postgres_connection_params()
    with psycopg.connect(**params, row_factory=dict_row) as conn, conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        version = conn.execute(
            "SELECT value FROM corpus_meta WHERE key = 'corpus_version'"
        ).fetchone()
        if version is None or not version["value"]:
            raise RuntimeError("Corpus is not ready. Complete a successful corpus build before loading it.")
        documents = []
        for row in conn.execute("""
            SELECT c.chunk_id, c.doc_id, c.chunk_type, c.page_id,
                   c.start_char, c.end_char, c.text, c.metadata_json,
                   d.ticker, d.report_year, d.scope,
                   t.table_key, t.table_id, t.page, t.context, t.statement_kind,
                   t.unit_scale, t.rows_json, t.header_rows_json
            FROM chunks c
            LEFT JOIN documents d ON d.doc_id = c.doc_id
            LEFT JOIN tables t ON t.table_key = c.table_key AND t.doc_id = c.doc_id
            ORDER BY c.doc_id, c.chunk_id
        """):
            if row["ticker"] is None or row["page_id"] is None:
                raise ValueError(f"Chunk {row['chunk_id']} is missing its document or page")
            if row["chunk_type"] == "table" and row["table_id"] is None:
                raise ValueError(f"Table chunk {row['chunk_id']} is missing its table")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"Chunk {row['chunk_id']} has no text")
            try:
                chunk_metadata = json.loads(row["metadata_json"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid metadata_json for chunk {row['chunk_id']}") from exc
            if not isinstance(chunk_metadata, dict):
                raise ValueError(f"Chunk {row['chunk_id']} metadata must be an object")
            metadata = dict(row)
            text = metadata.pop("text")
            metadata.pop("metadata_json")
            metadata = {**chunk_metadata, **metadata}
            metadata["page_ids"] = [row["page_id"]]
            documents.append(Document(id=row["chunk_id"], page_content=text, metadata=metadata))
        if not documents:
            raise ValueError("No chunks found. Build the corpus with pages, tables and chunks first.")
        return documents


def build_bm25(documents: list[Document], k: int = 5) -> BM25Retriever:
    """Index Documents; use the same Vietnamese tokenization for text and queries."""
    if not documents:
        raise ValueError("Cannot build BM25 from an empty document list")
    if k < 1:
        raise ValueError("k must be at least 1")
    if not any(tokens(document.page_content) for document in documents):
        raise ValueError("Documents contain no searchable tokens")
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(documents, preprocess_func=tokens, k=k)


def build_bm25_from_corpus(
    connection_params: dict[str, Any] | None = None,
    k: int = 5,
) -> BM25Retriever:
    """Load corpus Documents from PostgreSQL and build an in-memory BM25 index."""
    if k < 1:
        raise ValueError("k must be at least 1")
    params = connection_params or postgres_connection_params()
    return build_bm25(load_corpus_documents(params), k=k)

"""Read-only SQLite corpus -> LangChain Documents for BM25 and dense indexing.

Mirrors ``retrieval.bm25.load_corpus_documents`` (PostgreSQL) for the local
``corpus.sqlite3`` snapshot: same SQL columns, same metadata precedence (row
columns override ``metadata_json``) and the same ``page_ids`` field.

Differences, all deliberate:
- The database is opened with ``mode=ro``; nothing is ever written.
- SQLite snapshots carry no ``corpus_meta`` table, so there is no version check.
- Per-cell unit maps and folded search copies (``_HEAVY_METADATA_KEYS``) are
  dropped from chunk metadata. They make up about 4 GB of ``metadata_json``,
  are not used by retrieval, and stay available in the ``tables`` table.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from langchain_core.documents import Document

CHUNK_TYPES = ("page", "table")
_HEAVY_METADATA_KEYS = ("cell_units_json", "folded_text", "table_unit_json")

_SQL = """
SELECT c.chunk_id, c.doc_id, c.chunk_type, c.page_id,
       c.start_char, c.end_char, c.text,
       json_remove(c.metadata_json, {removed}) AS metadata_json,
       d.ticker, d.report_year, d.scope,
       t.table_key, t.table_id, t.page, t.context, t.statement_kind,
       t.unit_scale, t.rows_json, t.header_rows_json
FROM chunks c
LEFT JOIN documents d ON d.doc_id = c.doc_id
LEFT JOIN tables t ON t.table_key = c.table_key AND t.doc_id = c.doc_id
{where}
ORDER BY c.doc_id, c.chunk_id
{limit}
"""


def connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite file read-only; never create a missing one."""
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite corpus not found: {path}")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _normalize_chunk_types(chunk_types: Iterable[str] | None) -> tuple[str, ...]:
    if chunk_types is None:
        return CHUNK_TYPES
    if isinstance(chunk_types, str):
        raise ValueError("chunk_types must be a collection of strings, not a string")
    selected = tuple(dict.fromkeys(chunk_types))
    unknown = [value for value in selected if value not in CHUNK_TYPES]
    if unknown or not selected:
        raise ValueError(f"chunk_types must be a non-empty subset of {CHUNK_TYPES}; got {list(selected)}")
    return selected


def _query(chunk_types: tuple[str, ...], limit: int | None) -> tuple[str, list]:
    removed = ", ".join(f"'$.{key}'" for key in _HEAVY_METADATA_KEYS)
    params: list = []
    where = ""
    if chunk_types != CHUNK_TYPES:
        where = f"WHERE c.chunk_type IN ({', '.join('?' for _ in chunk_types)})"
        params.extend(chunk_types)
    limit_sql = ""
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        limit_sql = "LIMIT ?"
        params.append(limit)
    return _SQL.format(removed=removed, where=where, limit=limit_sql), params


def _row_document(row: dict) -> Document:
    from langchain_core.documents import Document

    chunk_id = row["chunk_id"]
    if row["ticker"] is None or row["page_id"] is None:
        raise ValueError(f"Chunk {chunk_id} is missing its document or page")
    if row["chunk_type"] == "table" and row["table_id"] is None:
        raise ValueError(f"Table chunk {chunk_id} is missing its table")
    if not isinstance(row["text"], str) or not row["text"].strip():
        raise ValueError(f"Chunk {chunk_id} has no text")
    try:
        chunk_metadata = json.loads(row["metadata_json"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid metadata_json for chunk {chunk_id}") from exc
    if not isinstance(chunk_metadata, dict):
        raise ValueError(f"Chunk {chunk_id} metadata must be an object")
    metadata = dict(row)
    text = metadata.pop("text")
    metadata.pop("metadata_json")
    metadata = {**chunk_metadata, **metadata}
    metadata["page_ids"] = [row["page_id"]]
    return Document(id=chunk_id, page_content=text, metadata=metadata)


def load_sqlite_corpus_documents(
    db_path: str | Path,
    *,
    chunk_types: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[Document]:
    """Load chunks (all, or only ``chunk_types``) in a stable doc/chunk order.

    ``limit`` keeps the first N chunks of that order, for smoke runs only.
    """
    selected = _normalize_chunk_types(chunk_types)
    sql, params = _query(selected, limit)
    connection = connect_read_only(db_path)
    try:
        documents = [_row_document(dict(row)) for row in connection.execute(sql, params)]
    finally:
        connection.close()
    if not documents:
        raise ValueError(f"No {'/'.join(selected)} chunks found in {db_path}")
    return documents

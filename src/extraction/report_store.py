"""Read-only access to stored report documents and tables for leaf extraction."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_DOCUMENTS_SQL = """
SELECT doc_id, ticker, report_year, scope, scope_variant FROM documents
WHERE ticker = {p} AND report_year = {p}
ORDER BY scope, scope_variant IS NOT NULL, scope_variant, doc_id
"""

_TABLES_SQL = """
SELECT t.table_key, c.chunk_id, t.doc_id, t.table_id, t.page, COALESCE(t.context, ''),
       t.statement_kind, t.rows_json
FROM tables t
LEFT JOIN chunks c ON c.table_key = t.table_key AND c.doc_id = t.doc_id AND c.chunk_type = 'table'
WHERE t.doc_id = {p} ORDER BY t.page, t.table_id
"""


@dataclass(frozen=True)
class ReportDoc:
    doc_id: str
    ticker: str
    report_year: int
    scope: str
    scope_variant: int | None


@dataclass(frozen=True)
class TableRecord:
    table_key: str
    chunk_id: str | None
    doc_id: str
    table_id: int
    page: int
    context: str
    statement_kind: str | None
    rows_json: str


class ReportSession(Protocol):
    def documents(self, ticker: str, year: int) -> list[ReportDoc]: ...

    def tables(self, doc_id: str) -> list[TableRecord]: ...


class ReportStore(Protocol):
    def session(self) -> AbstractContextManager[ReportSession]: ...


class ReportStoreError(RuntimeError):
    """Store failure; carries the operation and exception class name only."""

    def __init__(self, operation: str, error_class: str):
        super().__init__(f"report store {operation} failed: {error_class}")
        self.operation = operation
        self.error_class = error_class


class _DbSession:
    def __init__(self, conn: Any, placeholder: str, db_errors: tuple[type[BaseException], ...]):
        self._conn = conn
        self._placeholder = placeholder
        self._db_errors = db_errors

    def _fetch(self, operation: str, sql: str, params: tuple) -> list[tuple]:
        try:
            return list(self._conn.execute(sql.format(p=self._placeholder), params).fetchall())
        except self._db_errors as exc:
            raise ReportStoreError(operation, type(exc).__name__) from None

    def documents(self, ticker: str, year: int) -> list[ReportDoc]:
        rows = self._fetch("documents", _DOCUMENTS_SQL, (ticker.strip().upper(), year))
        return [ReportDoc(*row) for row in rows]

    def tables(self, doc_id: str) -> list[TableRecord]:
        rows = self._fetch("tables", _TABLES_SQL, (doc_id,))
        return [TableRecord(*row) for row in rows]


class SqliteReportStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)

    @contextmanager
    def session(self) -> Iterator[ReportSession]:
        try:
            conn = sqlite3.connect(f"file:{self._path.as_posix()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise ReportStoreError("connect", type(exc).__name__) from None
        try:
            yield _DbSession(conn, "?", (sqlite3.Error,))
        finally:
            conn.close()


class PostgresReportStore:
    def __init__(self, connection_params: dict | None = None, *, connect: Callable | None = None):
        self._params = connection_params
        self._connect = connect

    @contextmanager
    def session(self) -> Iterator[ReportSession]:
        import psycopg

        params = self._params
        if params is None:
            from config.settings import postgres_connection_params

            params = postgres_connection_params()
        connect = self._connect or psycopg.connect
        try:
            conn = connect(**params)
        except psycopg.Error as exc:
            raise ReportStoreError("connect", type(exc).__name__) from None
        try:
            try:
                with conn.transaction():
                    conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                    yield _DbSession(conn, "%s", (psycopg.Error,))
            except psycopg.Error as exc:
                raise ReportStoreError("transaction", type(exc).__name__) from None
        finally:
            conn.close()

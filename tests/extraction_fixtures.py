"""Shared helpers for leaf-extraction tests."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from src.extraction.report_store import ReportDoc, TableRecord


def record(
    doc_id: str,
    table_id: int,
    page: int,
    rows: list[list[str]],
    *,
    kind: str | None,
    context: str = "",
    chunk: bool = True,
) -> TableRecord:
    return TableRecord(
        table_key=f"{doc_id}:{table_id}",
        chunk_id=f"{doc_id}:table:{table_id}" if chunk else None,
        doc_id=doc_id,
        table_id=table_id,
        page=page,
        context=context,
        statement_kind=kind,
        rows_json=json.dumps(rows, ensure_ascii=False),
    )


class _InMemorySession:
    def __init__(self, docs, tables):
        self._docs = docs
        self._tables = tables

    def documents(self, ticker: str, year: int) -> list[ReportDoc]:
        want = ticker.strip().upper()
        found = [d for d in self._docs if d.ticker == want and d.report_year == year]
        return sorted(
            found,
            key=lambda d: (d.scope, d.scope_variant is not None, d.scope_variant or 0, d.doc_id),
        )

    def tables(self, doc_id: str) -> list[TableRecord]:
        return sorted(self._tables.get(doc_id, ()), key=lambda t: (t.page, t.table_id))


class InMemoryReportStore:
    def __init__(self, docs: Sequence[ReportDoc], tables: Mapping[str, Sequence[TableRecord]]):
        self._docs = list(docs)
        self._tables = {k: list(v) for k, v in tables.items()}
        self.sessions_opened = 0

    @contextmanager
    def session(self) -> Iterator[_InMemorySession]:
        self.sessions_opened += 1
        yield _InMemorySession(self._docs, self._tables)

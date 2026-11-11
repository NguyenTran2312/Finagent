import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.extraction.report_store import (
    PostgresReportStore,
    ReportStoreError,
    SqliteReportStore,
)
from src.indexing.schema import SCHEMA


class FakeConnect:
    def __init__(self):
        self.sql = []
        self.closed = False

    def __call__(self, **params):
        return self

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def transaction(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        self.closed = True


class ReportStoreTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.tmp = self._dir.name
        self.path = Path(self.tmp) / "corpus.sqlite3"
        conn = sqlite3.connect(self.path)
        conn.executescript(SCHEMA)
        for doc_id, scope, variant in [
            ("D3", "separate", None),
            ("D2", "consolidated", 2),
            ("D1", "consolidated", 1),
        ]:
            conn.execute(
                "INSERT INTO documents (doc_id, ticker, report_year, scope, scope_variant,"
                " source_path, source_bytes) VALUES (?, 'ACB', 2022, ?, ?, ?, 0)",
                (doc_id, scope, variant, f"{doc_id}.md"),
            )
        conn.execute(
            "INSERT INTO documents (doc_id, ticker, report_year, scope, source_path, source_bytes)"
            " VALUES ('X1', 'ACB', 2021, 'consolidated', 'X1.md', 0)"
        )
        for page in (7, 8):
            conn.execute(
                "INSERT INTO pages VALUES (?, 'D1', ?, ?, 0, 0, '')",
                (f"D1:p{page}", page, page),
            )
        for table_id, page in [(1, 8), (2, 7), (3, 8)]:
            conn.execute(
                "INSERT INTO tables (table_key, doc_id, table_id, page, page_id, context,"
                " folded_text, n_rows, n_cols, statement_kind, rows_json, table_unit_json,"
                " cell_units_json, header_rows_json)"
                " VALUES (?, 'D1', ?, ?, ?, NULL, '', 1, 1, 'balance_sheet', ?, '{}', '[]', '[]')",
                (f"D1:{table_id}", table_id, page, f"D1:p{page}", json.dumps([["a"]])),
            )
        for table_id in (1, 2):
            page = 8 if table_id == 1 else 7
            conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, chunk_type, page_id, table_key, text)"
                " VALUES (?, 'D1', 'table', ?, ?, 't')",
                (f"D1:table:{table_id}", f"D1:p{page}", f"D1:{table_id}"),
            )
        conn.commit()
        conn.close()

    def test_documents_all_scopes_ordered(self):
        with SqliteReportStore(self.path).session() as s:
            docs = s.documents("acb", 2022)
        self.assertEqual(
            [(d.scope, d.scope_variant) for d in docs],
            [("consolidated", 1), ("consolidated", 2), ("separate", None)],
        )
        self.assertEqual(docs[0].ticker, "ACB")

    def test_tables_ordered_and_chunk_join(self):
        with SqliteReportStore(self.path).session() as s:
            tables = s.tables("D1")
        self.assertEqual([t.page for t in tables], [7, 8, 8])
        self.assertEqual(tables[0].chunk_id, "D1:table:2")
        self.assertIsNone(tables[2].chunk_id)
        self.assertEqual(tables[0].context, "")
        self.assertEqual(tables[0].rows_json, json.dumps([["a"]]))

    def test_sqlite_is_read_only(self):
        with SqliteReportStore(self.path).session() as s:
            with self.assertRaises(sqlite3.OperationalError):
                s._conn.execute("DELETE FROM documents")

    def test_store_error_carries_class_name_only(self):
        with self.assertRaises(ReportStoreError) as ctx:
            with SqliteReportStore(Path(self.tmp) / "absent.sqlite3").session() as s:
                s.documents("ACB", 2022)
        self.assertEqual(ctx.exception.error_class, "OperationalError")
        self.assertNotIn("absent", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_query_error_is_wrapped(self):
        bare = Path(self.tmp) / "bare.sqlite3"
        sqlite3.connect(bare).close()
        with self.assertRaises(ReportStoreError) as ctx:
            with SqliteReportStore(bare).session() as s:
                s.documents("ACB", 2022)
        self.assertEqual(ctx.exception.operation, "documents")
        self.assertEqual(ctx.exception.error_class, "OperationalError")

    def test_postgres_session_read_only_and_closed(self):
        fake = FakeConnect()
        with self.assertRaises(RuntimeError):
            with PostgresReportStore({}, connect=fake).session():
                raise RuntimeError("boom")
        self.assertIn("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY", fake.sql[0])
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()

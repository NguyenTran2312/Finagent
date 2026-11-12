import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from retrieval.sqlite_corpus import connect_read_only, load_sqlite_corpus_documents

_SCHEMA = """
CREATE TABLE documents (doc_id TEXT PRIMARY KEY, ticker TEXT, report_year INTEGER, scope TEXT);
CREATE TABLE tables (table_key TEXT, doc_id TEXT, table_id INTEGER, page INTEGER, page_id TEXT,
    context TEXT, statement_kind TEXT, unit_scale TEXT, rows_json TEXT, header_rows_json TEXT);
CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, chunk_type TEXT, page_id TEXT,
    table_key TEXT, start_char INTEGER, end_char INTEGER, text TEXT, search_text TEXT,
    metadata_json TEXT);
"""


def _write_corpus(path: Path, *, chunks=None) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(_SCHEMA)
    connection.execute("INSERT INTO documents VALUES ('doc', 'AAA', 2024, 'consolidated')")
    connection.execute(
        "INSERT INTO tables VALUES ('doc:table:0', 'doc', 0, 3, 'doc:page:0', 'Đơn vị: VND', "
        "'balance_sheet', '1', '[[\"Tổng tài sản\", \"100\"]]', '[]')"
    )
    default_chunks = [
        ("doc:page:0", "doc", "page", "doc:page:0", None, 0, 10, "Báo cáo 2024", "bao cao",
         json.dumps({"chunk_type": "page", "ticker": "WRONG", "report_year": 1999, "page_number": 1})),
        ("doc:table:0", "doc", "table", "doc:page:0", "doc:table:0", 11, 40, "Tổng tài sản 100", "tong",
         json.dumps({"chunk_type": "table", "table_index": 7, "cell_units_json": "x" * 50,
                     "folded_text": "tong tai san", "table_unit_json": "{}", "n_rows": 1})),
    ]
    connection.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)", chunks or default_chunks)
    connection.commit()
    connection.close()


class SqliteCorpusLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "corpus.sqlite3"
        _write_corpus(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_loads_page_and_table_chunks_with_row_metadata_winning(self) -> None:
        documents = load_sqlite_corpus_documents(self.db)

        self.assertEqual([document.id for document in documents], ["doc:page:0", "doc:table:0"])
        page, table = documents
        self.assertEqual(page.page_content, "Báo cáo 2024")
        self.assertEqual((page.metadata["ticker"], page.metadata["report_year"]), ("AAA", 2024))
        self.assertEqual(page.metadata["page_number"], 1)
        self.assertEqual(page.metadata["page_ids"], ["doc:page:0"])
        self.assertIsNone(page.metadata["table_id"])
        self.assertEqual(table.metadata["table_key"], "doc:table:0")
        self.assertEqual(table.metadata["statement_kind"], "balance_sheet")
        self.assertEqual(table.metadata["rows_json"], '[["Tổng tài sản", "100"]]')
        self.assertEqual(table.metadata["n_rows"], 1)
        self.assertNotIn("text", table.metadata)
        self.assertNotIn("metadata_json", table.metadata)

    def test_drops_heavy_per_cell_metadata(self) -> None:
        table = load_sqlite_corpus_documents(self.db, chunk_types=["table"])[0]

        for key in ("cell_units_json", "folded_text", "table_unit_json"):
            self.assertNotIn(key, table.metadata)

    def test_filters_chunk_types_and_limits(self) -> None:
        tables = load_sqlite_corpus_documents(self.db, chunk_types=["table"])
        first = load_sqlite_corpus_documents(self.db, limit=1)

        self.assertEqual([document.id for document in tables], ["doc:table:0"])
        self.assertEqual([document.id for document in first], ["doc:page:0"])

    def test_rejects_bad_selection(self) -> None:
        for chunk_types in ([], ["image"], "table"):
            with self.subTest(chunk_types=chunk_types), self.assertRaises(ValueError):
                load_sqlite_corpus_documents(self.db, chunk_types=chunk_types)
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                load_sqlite_corpus_documents(self.db, limit=limit)

    def test_missing_file_is_not_created(self) -> None:
        missing = Path(self._tmp.name) / "absent.sqlite3"

        with self.assertRaises(FileNotFoundError):
            load_sqlite_corpus_documents(missing)
        self.assertFalse(missing.exists())

    def test_connection_is_read_only(self) -> None:
        connection = connect_read_only(self.db)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM chunks")
        finally:
            connection.close()

    def test_invalid_rows_fail_loudly(self) -> None:
        cases = {
            "blank text": ("doc:page:9", "doc", "page", "doc:page:9", None, 0, 1, "  ", "", "{}"),
            "orphan document": ("x:page:0", "x", "page", "x:page:0", None, 0, 1, "t", "", "{}"),
            "table without table row": ("doc:table:9", "doc", "table", "doc:page:0", "missing", 0, 1,
                                        "t", "", "{}"),
            "metadata not an object": ("doc:page:9", "doc", "page", "doc:page:9", None, 0, 1, "t", "",
                                       "[1]"),
        }
        for name, row in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                db = Path(folder) / "bad.sqlite3"
                _write_corpus(db, chunks=[row])
                with self.assertRaises(ValueError):
                    load_sqlite_corpus_documents(db)

    def test_empty_selection_raises(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "pages.sqlite3"
            _write_corpus(db, chunks=[
                ("doc:page:0", "doc", "page", "doc:page:0", None, 0, 1, "t", "", "{}"),
            ])
            with self.assertRaises(ValueError):
                load_sqlite_corpus_documents(db, chunk_types=["table"])


if __name__ == "__main__":
    unittest.main()

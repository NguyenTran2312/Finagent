import os
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import psycopg
from psycopg import sql

from config.settings import postgres_connection_params
from main.build_corpus import _load_document, discover_documents, run_pipeline
from retrieval.bm25 import load_corpus_documents


@contextmanager
def isolated_corpus(connection_params):
    """Keep integration writes, including corpus_version, out of the real corpus."""
    schema = f"test_corpus_{uuid.uuid4().hex}"
    with psycopg.connect(**connection_params, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        params = dict(connection_params)
        params["options"] = f"{params.get('options', '')} -csearch_path={schema}"
        yield params
    finally:
        with psycopg.connect(**connection_params, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


class IndexingPipelineTests(unittest.TestCase):
    def test_discover_documents_defaults_to_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(101):
                (root / f"report_{index:03}_extracted.txt").touch()

            paths = discover_documents(root)

            self.assertEqual(len(paths), 101)

    def test_loads_document_pages_tables_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_path = (
                root / "AAA_financial_statements_2024_consolidated_extracted.txt"
            )
            report_path.write_text(
                "===== PAGE 3 =====\n"
                "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH\n"
                "<table><tr><th>Chỉ tiêu</th><th>2024</th></tr>"
                "<tr><td>Doanh thu</td><td>100</td></tr></table>\n",
                encoding="utf-8",
            )

            document = _load_document(report_path)

            self.assertEqual(document["metadata"]["ticker"], "AAA")
            self.assertEqual(document["metadata"]["report_year"], 2024)
            self.assertEqual(document["page_labels"], ["3"])
            self.assertEqual(len(document["pages"]), 1)
            self.assertEqual(len(document["tables"]), 1)
            self.assertEqual(document["tables"][0]["n_rows"], 2)


@unittest.skipUnless(
    os.environ.get("POSTGRES_TEST_DATABASE"),
    "Set POSTGRES_TEST_DATABASE to run PostgreSQL integration tests.",
)
class PostgresCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        database_name = os.environ["POSTGRES_TEST_DATABASE"]
        connection_params = postgres_connection_params()
        connection_params["dbname"] = database_name
        self.params = self.enterContext(isolated_corpus(connection_params))
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ticker = f"T{uuid.uuid4().hex[:8].upper()}"
        for year in (2023, 2024):
            report_path = root / (
                f"{self.ticker}_financial_statements_{year}_consolidated_extracted.txt"
            )
            report_path.write_text(
                f"===== PAGE 1 =====\nReport for {self.ticker} {year}\n",
                encoding="utf-8",
            )
        with patch("main.build_corpus.postgres_connection_params", return_value=self.params):
            self.result = run_pipeline(root, database_name=database_name, max_workers=2)

    def test_writes_documents_concurrently_to_postgres(self) -> None:
        self.assertEqual(self.result["valid_documents"], 2)
        self.assertEqual(self.result["pages"], 2)
        self.assertEqual(self.result["chunks"], 2)
        with psycopg.connect(**self.params) as connection:
            stored = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE ticker = %s", (self.ticker,),
            ).fetchone()[0]
        self.assertEqual(stored, 2)
        self.assertEqual(len(load_corpus_documents(self.params)), 2)

    def test_rejects_a_corpus_after_an_interrupted_build(self) -> None:
        with psycopg.connect(**self.params) as connection:
            connection.execute("DELETE FROM corpus_meta WHERE key = 'corpus_version'")
            connection.execute("UPDATE chunks SET text = 'Partially rebuilt data'")

        with self.assertRaisesRegex(RuntimeError, "Corpus is not ready"):
            load_corpus_documents(self.params)

    def test_reader_keeps_one_snapshot_when_a_rebuild_starts_after_version_check(self) -> None:
        real_connect = psycopg.connect
        writer_params = self.params

        class RebuildingCursor(psycopg.Cursor):
            def execute(self, query, params=None, **kwargs):
                if isinstance(query, str) and "FROM chunks c" in query:
                    with real_connect(**writer_params) as writer:
                        writer.execute("DELETE FROM corpus_meta WHERE key = 'corpus_version'")
                        writer.execute("UPDATE chunks SET text = 'New generation'")
                return super().execute(query, params, **kwargs)

        def connect_reader(**kwargs):
            return real_connect(**kwargs, cursor_factory=RebuildingCursor)

        with patch("retrieval.bm25.psycopg.connect", side_effect=connect_reader):
            documents = load_corpus_documents({**self.params, "autocommit": True})

        self.assertEqual(
            {document.page_content for document in documents},
            {f"Report for {self.ticker} {year}" for year in (2023, 2024)},
        )
        with real_connect(**self.params) as connection:
            self.assertEqual(connection.execute("SELECT DISTINCT text FROM chunks").fetchall(), [("New generation",)])
            self.assertIsNone(connection.execute("SELECT value FROM corpus_meta").fetchone())


if __name__ == "__main__":
    unittest.main()

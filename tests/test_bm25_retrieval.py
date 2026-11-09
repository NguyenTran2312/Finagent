import unittest
from unittest.mock import MagicMock, patch

from retrieval.bm25 import build_bm25_from_corpus, load_corpus_documents


class BM25RetrievalTests(unittest.TestCase):
    def test_builds_in_memory_index_from_postgres_corpus(self) -> None:
        documents = [object()]
        retriever = object()

        with (
            patch("retrieval.bm25.load_corpus_documents", return_value=documents) as load,
            patch("retrieval.bm25.build_bm25", return_value=retriever) as build,
        ):
            result = build_bm25_from_corpus({"dbname": "finance_corpus"}, k=3)

        self.assertIs(result, retriever)
        load.assert_called_once_with({"dbname": "finance_corpus"})
        build.assert_called_once_with(documents, k=3)

    def test_loads_page_and_table_chunks_from_postgres_rows(self) -> None:
        rows = [
            {
                "chunk_id": "doc:page:0",
                "doc_id": "doc",
                "chunk_type": "page",
                "page_id": "doc:page:0",
                "start_char": 0,
                "end_char": 10,
                "text": "Báo cáo 2024",
                "metadata_json": (
                    '{"chunk_type": "page", "page_number": 1, "ticker": "WRONG", '
                    '"report_year": 1999, "doc_id": "wrong"}'
                ),
                "ticker": "AAA",
                "report_year": 2024,
                "scope": "consolidated",
                "table_key": None,
                "table_id": None,
                "page": None,
                "context": None,
                "statement_kind": None,
                "unit_scale": None,
                "rows_json": None,
                "header_rows_json": None,
            },
            {
                "chunk_id": "doc:table:0",
                "doc_id": "doc",
                "chunk_type": "table",
                "page_id": "doc:page:0",
                "start_char": 11,
                "end_char": 40,
                "text": "Doanh thu 100",
                "metadata_json": '{"chunk_type": "table"}',
                "ticker": "AAA",
                "report_year": 2024,
                "scope": "consolidated",
                "table_key": "doc:table:0",
                "table_id": 0,
                "page": 1,
                "context": "Kết quả kinh doanh",
                "statement_kind": "income_statement",
                "unit_scale": 1.0,
                "rows_json": "[]",
                "header_rows_json": "[]",
            },
        ]
        connection = MagicMock()
        connection.__enter__.return_value = connection
        version_cursor = MagicMock()
        version_cursor.fetchone.return_value = {"value": "complete-build"}
        connection.execute.side_effect = [MagicMock(), version_cursor, rows]

        with patch("retrieval.bm25.psycopg.connect", return_value=connection) as connect:
            documents = load_corpus_documents({"dbname": "finance_corpus"})

        connect.assert_called_once()
        connection.transaction.return_value.__enter__.assert_called_once()
        statements = [call.args[0] for call in connection.execute.call_args_list]
        self.assertIn("REPEATABLE READ", statements[0])
        self.assertIn("READ ONLY", statements[0])
        self.assertIn("corpus_version", statements[1])
        self.assertIn("FROM chunks", statements[2])
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["page_ids"], ["doc:page:0"])
        self.assertEqual(documents[0].metadata["page_number"], 1)
        self.assertEqual(documents[0].metadata["ticker"], "AAA")
        self.assertEqual(documents[0].metadata["report_year"], 2024)
        self.assertEqual(documents[0].metadata["doc_id"], "doc")
        self.assertEqual(documents[1].metadata["table_id"], 0)
        self.assertEqual(documents[1].metadata["context"], "Kết quả kinh doanh")

    def test_rejects_missing_or_empty_corpus_version_before_reading_chunks(self) -> None:
        for marker in (None, {"value": ""}):
            with self.subTest(marker=marker):
                connection = MagicMock()
                connection.__enter__.return_value = connection
                connection.execute.return_value.fetchone.return_value = marker
                connection.execute.return_value.__iter__.return_value = [{
                    "chunk_id": "doc:page:0", "doc_id": "doc", "chunk_type": "page",
                    "page_id": "doc:page:0", "ticker": "AAA", "text": "Existing page",
                    "metadata_json": "{}",
                }]
                with patch("retrieval.bm25.psycopg.connect", return_value=connection):
                    with self.assertRaisesRegex(RuntimeError, "Corpus is not ready"):
                        load_corpus_documents({"dbname": "finance_corpus"})

                statements = [call.args[0] for call in connection.execute.call_args_list]
                self.assertFalse(any("FROM chunks" in statement for statement in statements))
                connection.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()

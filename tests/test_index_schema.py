import sqlite3
import unittest

from indexing.schema import SCHEMA


class IndexSchemaTests(unittest.TestCase):
    def test_schema_covers_pipeline_document_page_table_and_chunk_fields(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(SCHEMA)
            expected_columns = {
                "corpus_meta": {"key", "value"},
                "documents": {
                    "scope",
                    "scope_original",
                    "scope_variant",
                    "source_path",
                    "source_bytes",
                },
                "pages": {
                    "page_index",
                    "page_number",
                    "start_char",
                    "end_char",
                },
                "tables": {
                    "folded_text",
                    "n_rows",
                    "n_cols",
                    "table_unit_json",
                    "cell_units_json",
                    "header_rows_json",
                    "rows_json",
                },
                "chunks": {
                    "chunk_type",
                    "page_id",
                    "table_key",
                    "start_char",
                    "end_char",
                    "text",
                    "search_text",
                    "metadata_json",
                },
            }
            for table, expected in expected_columns.items():
                with self.subTest(table=table):
                    actual = {
                        row[1]
                        for row in connection.execute(f"PRAGMA table_info({table})")
                    }
                    self.assertTrue(expected.issubset(actual))
        finally:
            connection.close()

    def test_schema_accepts_canonical_scope_and_preserved_variant(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            connection.executescript(SCHEMA)
            connection.execute(
                """
                INSERT INTO documents (
                    doc_id, ticker, report_year, scope, scope_original, scope_variant,
                    source_path, source_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "AAA_2023",
                    "AAA",
                    2023,
                    "separate",
                    "separate_2",
                    2,
                    "financial_statements/AAA/report.txt",
                    100,
                ),
            )
            stored_scope = connection.execute(
                "SELECT scope, scope_original, scope_variant FROM documents"
            ).fetchone()
            self.assertEqual(stored_scope, ("separate", "separate_2", 2))
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()

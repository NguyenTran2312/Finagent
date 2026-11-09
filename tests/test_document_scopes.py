import unittest

from preprocess.build_document_metadata import build_document_metadata
from preprocess.chunker import table_record_to_payload


class DocumentScopeTests(unittest.TestCase):
    def test_build_document_metadata_normalizes_scope_categories(self) -> None:
        cases = [
            ("AAA_financial_statements_2023_consolidated_1_extracted.txt", "consolidated"),
            ("AAA_financial_statements_2023_consolidated_2_extracted.txt", "consolidated"),
            ("AAA_financial_statements_2023_separate_extracted.txt", "separate"),
            ("AAA_financial_statements_2023_separate_1_extracted.txt", "separate"),
            ("AAA_financial_statements_2023_separate_2_extracted.txt", "separate"),
            ("AAA_2023_financial_statement_explanations_1_extracted.txt", "explanations"),
            ("AAA_2023_explanatory_letters_extracted.txt", "letters"),
        ]

        for filename, expected_scope in cases:
            with self.subTest(filename=filename):
                metadata = build_document_metadata(filename)
                self.assertEqual(metadata["scope"], expected_scope)

    def test_preserves_scope_variant_separately_from_canonical_scope(self) -> None:
        metadata = build_document_metadata(
            "AAA_financial_statements_2023_separate_2_extracted.txt"
        )

        self.assertEqual(metadata["scope"], "separate")
        self.assertEqual(metadata["scope_original"], "separate_2")
        self.assertEqual(metadata["scope_variant"], 2)

    def test_table_payload_keeps_each_scope_distinct(self) -> None:
        cases = [
            ("consolidated", "hợp nhất"),
            ("consolidated_1", "hợp nhất"),
            ("separate", "công ty mẹ"),
            ("explanations", "thuyết minh"),
            ("letters", "thư giải trình"),
        ]

        for scope, expected_label in cases:
            with self.subTest(scope=scope):
                record = {
                    "doc_id": "AAA_2023",
                    "table_id": 1,
                    "report_year": 2023,
                    "scope": scope,
                    "ticker": "AAA",
                    "statement_kind": "balance_sheet",
                    "context": "",
                    "rows_json": "[[\"Chỉ tiêu\"]]",
                }
                text, metadata = table_record_to_payload(record, table_index=0)

                expected_scope = {
                    "consolidated_1": "consolidated",
                    "consolidated_2": "consolidated",
                    "separate_1": "separate",
                    "separate_2": "separate",
                }.get(scope, scope)
                self.assertEqual(metadata["scope"], expected_scope)
                self.assertIn(f"({expected_label})", text)


if __name__ == "__main__":
    unittest.main()

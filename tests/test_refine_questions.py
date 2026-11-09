import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from main.refine_questions import refine_question, retrieve_refined
from retrieval.bm25 import build_bm25


class RefineQuestionsTests(unittest.TestCase):
    def test_extracts_ticker_year_range_and_report_scope(self) -> None:
        filters = refine_question(
            "So sánh doanh thu AAA từ 2022 đến 2024, báo cáo riêng lẻ",
            {"AAA", "BBB"},
        )

        self.assertEqual(filters, {
            "tickers": ["AAA"],
            "years": [2022, 2023, 2024],
            "scopes": ["separate"],
        })

    def test_defaults_unspecified_scope_to_consolidated(self) -> None:
        filters = refine_question("Doanh thu năm 2024", {"AAA"})

        self.assertEqual(filters["scopes"], ["consolidated"])

    def test_extracts_common_vietnamese_year_range_phrasings(self) -> None:
        questions = (
            "Doanh thu từ năm 2020 đến năm 2022",
            "Doanh thu giai đoạn từ năm 2020 tới năm 2022",
            "Doanh thu trong giai đoạn 2020 cho đến 2022",
            "Doanh thu năm 2020 - năm 2022",
        )

        for question in questions:
            with self.subTest(question=question):
                filters = refine_question(question, {"AAA"})
                self.assertEqual(filters["years"], [2020, 2021, 2022])

    def test_retrieves_with_metadata_filters_and_original_filter_question(self) -> None:
        documents = [
            Document(
                page_content="Doanh thu AAA năm 2024 hợp nhất",
                metadata={
                    "ticker": "AAA",
                    "report_year": 2024,
                    "scope": "consolidated",
                },
            ),
            Document(
                page_content="Doanh thu AAA năm 2023 hợp nhất",
                metadata={
                    "ticker": "AAA",
                    "report_year": 2023,
                    "scope": "consolidated",
                },
            ),
            Document(
                page_content="Doanh thu BBB năm 2024 hợp nhất",
                metadata={
                    "ticker": "BBB",
                    "report_year": 2024,
                    "scope": "consolidated",
                },
            ),
        ]
        retriever = build_bm25(documents, k=3)

        result = retrieve_refined(
            retriever,
            "Doanh thu",
            filter_question="Doanh thu AAA năm 2024",
        )

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(result["documents"][0].metadata["ticker"], "AAA")
        self.assertEqual(result["documents"][0].metadata["report_year"], 2024)
        self.assertEqual(len(result["scores"]), len(result["documents"]))

    def test_excludes_documents_with_zero_bm25_scores(self) -> None:
        documents = [
            Document(
                page_content="document one",
                metadata={"ticker": "AAA", "report_year": 2024, "scope": "consolidated"},
            ),
            Document(
                page_content="document two",
                metadata={"ticker": "AAA", "report_year": 2024, "scope": "consolidated"},
            ),
        ]
        retriever = Mock(
            docs=documents,
            k=2,
            preprocess_func=lambda query: query.split(),
        )
        retriever.vectorizer.get_batch_scores.return_value = [0.0, 0.0]

        result = retrieve_refined(retriever, "AAA 2024 unknownterm", k=2)

        self.assertEqual(result["documents"], [])
        self.assertEqual(result["scores"], [])


if __name__ == "__main__":
    unittest.main()

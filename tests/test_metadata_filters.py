import copy
import unittest

from langchain_core.documents import Document
from pydantic import BaseModel

from src.retrieval.bm25 import build_bm25


def make_doc(ticker, year, scope, text, table_id=0):
    return Document(
        page_content=text,
        metadata={"ticker": ticker, "report_year": year, "scope": scope,
                  "doc_id": f"{ticker}_{year}_{scope}", "table_id": table_id},
    )


def corpus():
    return [
        make_doc("VJC", 2022, "consolidated", "Doanh thu VJC nam 2022 hop nhat"),
        make_doc("VJC", 2023, "consolidated", "Doanh thu VJC nam 2023 hop nhat"),
        make_doc("VJC", 2022, "separate", "Doanh thu VJC nam 2022 rieng"),
        make_doc("VJC", 2022, "parent", "Doanh thu VJC nam 2022 parent cu"),
        make_doc("HPG", 2022, "consolidated", "Doanh thu HPG nam 2022 hop nhat"),
    ]


class FakeDense:
    def __init__(self):
        self.calls = []

    def similarity_search(self, query, k, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return []


class MetadataFiltersModelTests(unittest.TestCase):
    def test_normalizes_and_serializes_plural_keys(self) -> None:
        from src.retrieval.filters import MetadataFilters

        filters = MetadataFilters(tickers=[" vjc ", "VJC", "hpg"], years=[2022, 2022, 2023],
                                  scopes=["parent", "separate", "consolidated"])
        self.assertEqual(filters.model_dump(), {
            "tickers": ["VJC", "HPG"], "years": [2022, 2023],
            "scopes": ["separate", "consolidated"]})

    def test_empty_lists_are_unrestricted(self) -> None:
        from src.retrieval.filters import MetadataFilters

        self.assertEqual(MetadataFilters(tickers=[], years=[], scopes=[]).model_dump(),
                         {"tickers": [], "years": [], "scopes": []})

    def test_all_scopes_accepted(self) -> None:
        from src.retrieval.filters import MetadataFilters

        scopes = ["consolidated", "separate", "explanations", "letters"]
        self.assertEqual(MetadataFilters(tickers=[], years=[], scopes=scopes).scopes, scopes)

    def test_rejects_invalid_values(self) -> None:
        from pydantic import ValidationError
        from src.retrieval.filters import MetadataFilters

        base = {"tickers": ["VJC"], "years": [2022], "scopes": ["consolidated"]}
        bad = [
            {**base, "tickers": [""]},
            {**base, "tickers": ["  "]},
            {**base, "tickers": [5]},
            {**base, "years": ["2022"]},
            {**base, "years": [True]},
            {**base, "years": [2022.0]},
            {**base, "years": [1899]},
            {**base, "years": [2200]},
            {**base, "scopes": ["quarterly"]},
            {**base, "scopes": [None]},
            {**base, "extra": 1},
            {"tickers": ["VJC"], "years": [2022]},
            {},
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                MetadataFilters(**value)

    def test_year_boundaries_accepted(self) -> None:
        from src.retrieval.filters import MetadataFilters

        self.assertEqual(MetadataFilters(tickers=[], years=[1900, 2199], scopes=[]).years,
                         [1900, 2199])


class NormalizeBoundaryTests(unittest.TestCase):
    def test_accepts_dict_model_and_foreign_basemodel(self) -> None:
        from src.retrieval.filters import MetadataFilters, normalize_metadata_filters

        raw = {"tickers": [" vjc"], "years": [2022], "scopes": ["parent"]}
        expected = {"tickers": ["VJC"], "years": [2022], "scopes": ["separate"]}

        class Foreign(BaseModel):
            tickers: list[str]
            years: list[int]
            scopes: list[str]

        for value in (raw, MetadataFilters(**raw), Foreign(**raw)):
            with self.subTest(type=type(value).__name__):
                self.assertEqual(normalize_metadata_filters(value), expected)

    def test_rejects_other_types_and_incomplete_mapping(self) -> None:
        from src.retrieval.filters import normalize_metadata_filters

        for value in (None, "VJC", [], 5, {}, {"tickers": ["VJC"]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_metadata_filters(value)

    def test_does_not_mutate_input(self) -> None:
        from src.retrieval.filters import normalize_metadata_filters

        raw = {"tickers": [" vjc"], "years": [2022], "scopes": ["parent"]}
        snapshot = copy.deepcopy(raw)
        result = normalize_metadata_filters(raw)
        self.assertEqual(raw, snapshot)
        result["tickers"].append("X")
        self.assertEqual(raw, snapshot)


class RetrieveRefinedStructuredTests(unittest.TestCase):
    def test_explicit_filters_override_text_extraction(self) -> None:
        from src.main.refine_questions import retrieve_refined

        bm25 = build_bm25(corpus(), k=10)
        result = retrieve_refined(
            bm25, "Doanh thu VJC nam 2022 2023 hop nhat", k=10,
            metadata_filters={"tickers": ["vjc"], "years": [2022], "scopes": ["consolidated"]})
        self.assertEqual(result["filters"], {"tickers": ["VJC"], "years": [2022],
                                             "scopes": ["consolidated"]})
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual([(d.metadata["ticker"], d.metadata["report_year"], d.metadata["scope"])
                          for d in result["documents"]], [("VJC", 2022, "consolidated")])

    def test_unrestricted_fields_and_parent_alias(self) -> None:
        from src.main.refine_questions import retrieve_refined

        bm25 = build_bm25(corpus(), k=10)
        result = retrieve_refined(bm25, "Doanh thu", k=10,
                                  metadata_filters={"tickers": [], "years": [], "scopes": ["parent"]})
        self.assertEqual(result["filters"]["scopes"], ["separate"])
        self.assertEqual(result["candidate_count"], 2)

    def test_explicit_empty_scopes_do_not_default_to_consolidated(self) -> None:
        from src.main.refine_questions import retrieve_refined

        bm25 = build_bm25(corpus(), k=10)
        result = retrieve_refined(bm25, "Doanh thu", k=10,
                                  metadata_filters={"tickers": ["HPG"], "years": [], "scopes": []})
        self.assertEqual(result["candidate_count"], 1)

    def test_conflicting_filter_question_rejected(self) -> None:
        from src.main.refine_questions import retrieve_refined

        bm25 = build_bm25(corpus(), k=10)
        with self.assertRaises(ValueError):
            retrieve_refined(bm25, "Doanh thu", filter_question="VJC",
                             metadata_filters={"tickers": [], "years": [], "scopes": []})

    def test_invalid_filters_rejected(self) -> None:
        from src.main.refine_questions import retrieve_refined

        bm25 = build_bm25(corpus(), k=10)
        with self.assertRaises(ValueError):
            retrieve_refined(bm25, "Doanh thu", metadata_filters={})

    def test_legacy_path_unchanged(self) -> None:
        from src.main.refine_questions import retrieve_refined

        bm25 = build_bm25(corpus(), k=10)
        result = retrieve_refined(bm25, "Doanh thu VJC nam 2022")
        self.assertEqual(result["filters"], {"tickers": ["VJC"], "years": [2022],
                                             "scopes": ["consolidated"]})
        result = retrieve_refined(bm25, "Doanh thu", filter_question="HPG 2022")
        self.assertEqual(result["filters"]["tickers"], ["HPG"])


class HybridStructuredTests(unittest.TestCase):
    def build(self):
        from src.retrieval.hybrid import build_hybrid_retriever

        dense = FakeDense()
        return build_hybrid_retriever(build_bm25(corpus(), k=10), dense, k=5, candidate_k=10), dense

    def test_bm25_and_chroma_use_identical_restrictions(self) -> None:
        retriever, dense = self.build()
        filters = {"tickers": [" vjc "], "years": [2022], "scopes": ["consolidated"]}
        result = retriever.invoke({"question": "Doanh thu VJC nam 2022 2023 hop nhat",
                                   "metadata_filters": filters})
        self.assertEqual([(d.metadata["ticker"], d.metadata["report_year"], d.metadata["scope"])
                          for d in result], [("VJC", 2022, "consolidated")])
        self.assertEqual(len(dense.calls), 1)
        self.assertEqual(dense.calls[0]["filter"], {"$and": [
            {"ticker": {"$in": ["VJC"]}},
            {"report_year": {"$in": [2022]}},
            {"scope": {"$in": ["consolidated"]}}]})
        self.assertEqual(filters, {"tickers": [" vjc "], "years": [2022], "scopes": ["consolidated"]})

    def test_parent_alias_adds_legacy_parent_to_where(self) -> None:
        retriever, dense = self.build()
        result = retriever.invoke({"question": "Doanh thu VJC",
                                   "metadata_filters": {"tickers": ["VJC"], "years": [2022],
                                                        "scopes": ["parent"]}})
        self.assertEqual({d.metadata["scope"] for d in result}, {"separate", "parent"})
        self.assertEqual(dense.calls[0]["filter"]["$and"][2], {"scope": {"$in": ["separate", "parent"]}})

    def test_empty_lists_produce_no_where(self) -> None:
        retriever, dense = self.build()
        retriever.invoke({"question": "Doanh thu",
                          "metadata_filters": {"tickers": [], "years": [], "scopes": []}})
        self.assertIsNone(dense.calls[0]["filter"])

    def test_no_candidates_skips_dense(self) -> None:
        retriever, dense = self.build()
        result = retriever.invoke({"question": "Doanh thu",
                                   "metadata_filters": {"tickers": ["HPG"], "years": [2023], "scopes": []}})
        self.assertEqual(result, [])
        self.assertEqual(dense.calls, [])

    def test_invalid_input_never_reaches_dense(self) -> None:
        retriever, dense = self.build()
        base = {"tickers": ["VJC"], "years": [2022], "scopes": ["consolidated"]}
        cases = [
            {**base, "years": ["2022"]},
            {**base, "years": [True]},
            {**base, "scopes": ["bogus"]},
            {**base, "tickers": [" "]},
            {**base, "extra": 1},
            {},
            "VJC",
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                retriever.invoke({"question": "Doanh thu", "metadata_filters": value})
        with self.assertRaises(ValueError):
            retriever.invoke({"question": "Doanh thu", "filter_question": "VJC",
                              "metadata_filters": base})
        self.assertEqual(dense.calls, [])

    def test_none_metadata_filters_is_absent(self) -> None:
        retriever, dense = self.build()
        retriever.invoke({"question": "Doanh thu VJC nam 2022", "metadata_filters": None})
        self.assertEqual(dense.calls[0]["filter"], {"$and": [
            {"ticker": {"$in": ["VJC"]}}, {"report_year": {"$in": [2022]}},
            {"scope": {"$in": ["consolidated"]}}]})

    def test_legacy_string_and_filter_question_unchanged(self) -> None:
        retriever, dense = self.build()
        result = retriever.invoke("Doanh thu VJC nam 2022")
        self.assertEqual([d.metadata["scope"] for d in result], ["consolidated"])
        retriever.invoke({"question": "Doanh thu", "filter_question": "HPG nam 2022",
                          "unknown_key": 1})
        self.assertEqual(dense.calls[1]["filter"]["$and"][0], {"ticker": {"$in": ["HPG"]}})


if __name__ == "__main__":
    unittest.main()

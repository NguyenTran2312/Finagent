"""Tests for retrieval-guided leaf extraction."""

from __future__ import annotations

import dataclasses
import unittest
from decimal import Decimal as D

from src.calculation.metric_tree import LeafRequest
from src.extraction.extractor import LeafExtractor, LocatorResult
from src.extraction.outcomes import Ambiguous, BlankValue, ExtractedFact, Found, NotFound, ReportMissing
from src.extraction.retrieval import RankedTable, RetrievalTrace, TableRetrievalError, ranking_of
from tests.extraction_fixtures import record
from tests.test_extraction_extractor import (
    INTEREST_2023,
    NET_REVENUE_2023,
    StubLocator,
    default_store,
    doc,
    is_rows,
    is_table,
    one,
    read_label,
    store_of,
    templates,
    tree,
)

_NOTE_ROWS = [
    ["", "", "2023 VND", "2022 VND"],
    ["", "Lãi tiền gửi ngân hàng", "120.000.000", "90.000.000"],
]


class FakeRetriever:
    mode = "fake"

    def __init__(self, answers=None, error=None):
        self.answers, self.error, self.calls = answers or {}, error, []

    def rank(self, ticker, scope, report_year, leaf):
        self.calls.append((ticker, scope, report_year, leaf.id))
        if self.error is not None:
            raise self.error
        return self.answers.get((report_year, leaf.id), ())


def ranked(*chunk_ids):
    return tuple(RankedTable(c, i) for i, c in enumerate(chunk_ids, 1))


def base_tables(doc_id="VJC23C"):
    return [is_table(doc_id, is_rows(2023)), record(doc_id, 2, 30, _NOTE_ROWS, kind="note")]


def base_store():
    return store_of((doc("VJC23C", 2023), base_tables()))


def extract_with(retriever, requests, store=None, fallback=None):
    store = store if store is not None else base_store()
    extractor = LeafExtractor(store, tree, templates, fallback=fallback, retriever=retriever)
    return extractor.extract("VJC", "consolidated", requests).outcomes[0]


def one_with(retriever, store=None, request=INTEREST_2023, fallback=None):
    return extract_with(retriever, [request], store, fallback)


def _interest(*chunk_ids):
    return FakeRetriever({(2023, "interest_expense"): ranked(*chunk_ids)})


class RankingValidationTests(unittest.TestCase):
    def test_valid_answer_maps_chunks_to_ranks(self):
        answer = (RankedTable("D:table:2", 2), RankedTable("D:table:1", 1))
        self.assertEqual(ranking_of(answer), {"D:table:2": 2, "D:table:1": 1})
        self.assertEqual(ranking_of(()), {})

    def test_invalid_answers_raise(self):
        bad = {
            "list": [RankedTable("a", 1)],
            "rank_zero": (RankedTable("a", 0),),
            "rank_gap": (RankedTable("a", 1), RankedTable("b", 3)),
            "duplicate_rank": (RankedTable("a", 1), RankedTable("b", 1)),
            "duplicate_chunk": (RankedTable("a", 1), RankedTable("a", 2)),
            "not_ranked_table": (("a", 1),),
        }
        for name, answer in bad.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    ranking_of(answer)

    def test_extracted_fact_has_no_trace_by_default(self):
        self.assertIsNone(dataclasses.fields(ExtractedFact)[-1].default)
        self.assertEqual(dataclasses.fields(ExtractedFact)[-1].name, "retrieval")


class RetrievalStepTests(unittest.TestCase):
    def test_value_in_ranked_table_is_retrieved(self):
        out = one_with(_interest("VJC23C:table:2", "VJC23C:table:1"))
        self.assertEqual(out.extracted.fact.value.value, D("-1950254195864"))
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 2))

    def test_row_outside_ranked_tables_falls_back_to_full_scan(self):
        out = one_with(_interest("VJC23C:table:2"))
        self.assertEqual(out.extracted.fact.value.value, D("-1950254195864"))
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "full_scan", None))

    def test_blank_cell_in_ranked_table_is_final(self):
        store = store_of((doc("VJC23C", 2023), [
            is_table("VJC23C", is_rows(2023, current="")),
            is_table("VJC23C", is_rows(2023), table_id=3, page=40),
        ]))
        out = one_with(_interest("VJC23C:table:1"), store)
        self.assertIsInstance(out, BlankValue)
        self.assertEqual(out.location.chunk_id, "VJC23C:table:1")

    def test_empty_answer_reads_full_report(self):
        r = FakeRetriever()
        out = one_with(r)
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "full_scan", None))
        self.assertEqual(r.calls, [("VJC", "consolidated", 2023, "interest_expense")])

    def test_r1_hidden_conflict_is_accepted_risk(self):
        store_r1 = store_of((doc("VJC23C", 2023), base_tables() + [
            is_table("VJC23C", is_rows(2023, current="(5.000.000.000)"), table_id=3, page=40),
        ]))
        self.assertIsInstance(one(store_r1), Ambiguous)  # no retriever
        out = one_with(_interest("VJC23C:table:3"), store_r1)
        self.assertEqual(out.extracted.fact.value.value, D("-5000000000"))
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 1))

    def test_next_year_source_retrieves_with_its_own_year(self):
        store = store_of(
            (doc("VJC22C", 2022), [is_table("VJC22C", is_rows(2022, interest=False))]),
            (doc("VJC23C", 2023), base_tables()),
        )
        r = FakeRetriever()
        out = extract_with(r, [LeafRequest("interest_expense", 2022, "flow")], store)
        self.assertIsInstance(out, Found)
        self.assertEqual(out.extracted.fact.value.value, D("-1383641245573"))
        self.assertEqual([c[2] for c in r.calls], [2022, 2023])

    def test_one_rank_call_per_year_and_leaf(self):
        store = store_of(
            (doc("VJC22C", 2022), [is_table("VJC22C", is_rows(2022, interest=False))]),
            (doc("VJC23C", 2023), base_tables()),
            (doc("VJC23C1", 2023, variant=1), base_tables("VJC23C1")),
        )
        r = FakeRetriever()
        LeafExtractor(store, tree, templates, retriever=r).extract(
            "VJC", "consolidated", [LeafRequest("interest_expense", 2022, "flow"), INTEREST_2023])
        self.assertEqual(sorted(r.calls), [("VJC", "consolidated", 2022, "interest_expense"),
                                           ("VJC", "consolidated", 2023, "interest_expense")])

    def test_variant_without_ranked_table_is_read_in_full(self):
        store = store_of(
            (doc("VJC23C", 2023), base_tables()),
            (doc("VJC23C1", 2023, variant=1), base_tables("VJC23C1")),
        )
        out = one_with(_interest("VJC23C:table:1"), store)
        self.assertIsInstance(out, Found)
        self.assertFalse(any(w.startswith("not_in_variant") for w in out.extracted.warnings))
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 1))

    def test_report_missing_year_is_not_retrieved(self):
        r = FakeRetriever()
        out = one_with(r, request=LeafRequest("interest_expense", 2021, "flow"))
        self.assertIsInstance(out, ReportMissing)
        self.assertEqual(r.calls, [])

    def test_no_retriever_keeps_outcomes_and_has_no_trace(self):
        runs = [
            ("VJC", [INTEREST_2023, NET_REVENUE_2023]),
            ("ACB", [LeafRequest("total_assets", 2022, "closing"),
                     LeafRequest("customer_loans_gross", 2022, "closing")]),
        ]
        for ticker, requests in runs:
            without = LeafExtractor(default_store(), tree, templates).extract(ticker, "consolidated", requests)
            with_empty = LeafExtractor(default_store(), tree, templates, retriever=FakeRetriever()).extract(
                ticker, "consolidated", requests)
            self.assertEqual(len(without.outcomes), len(with_empty.outcomes))
            self.assertTrue(any(isinstance(a, Found) for a in without.outcomes))
            for a, b in zip(without.outcomes, with_empty.outcomes):
                with self.subTest(ticker=ticker, request=a.request):
                    if isinstance(a, Found):
                        self.assertIsNone(a.extracted.retrieval)
                        self.assertEqual(b.extracted.retrieval, RetrievalTrace("fake", "full_scan", None))
                        self.assertEqual(a, dataclasses.replace(
                            b, extracted=dataclasses.replace(b.extracted, retrieval=None)))
                    else:
                        self.assertEqual(a, b)


def _no_interest_store():
    """Table 1 has no interest row; table 2 is the note table: the deterministic search gives NotFound."""
    return store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, interest=False)),
                                           record("VJC23C", 2, 30, _NOTE_ROWS, kind="note")]))


class ViewsLocator(StubLocator):
    """``StubLocator`` that also records the chunk ids of the views it received."""

    def __init__(self, respond):
        super().__init__(respond)
        self.views = []

    def locate(self, leaf, request, role, views):
        self.views.append([v.record.chunk_id for v in views])
        return super().locate(leaf, request, role, views)


def _not_present(views):
    return LocatorResult(None, None, ())


class RetrievalFailureTests(unittest.TestCase):
    def test_retrieval_error_degrades_to_full_scan(self):
        for error, code in ((TableRetrievalError("x"), "TableRetrievalError"),
                            (FileNotFoundError("x"), "FileNotFoundError")):
            with self.subTest(code):
                out = one_with(FakeRetriever(error=error))
                self.assertEqual(out.extracted.fact.value.value, D("-1950254195864"))
                self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "full_scan", None))
                self.assertIn(f"retrieval_error:{code}", out.extracted.warnings)

    def test_programming_error_from_retriever_propagates(self):
        with self.assertRaises(KeyError):
            one_with(FakeRetriever(error=KeyError("x")))

    def test_invalid_answer_raises(self):
        with self.assertRaises(ValueError):
            one_with(FakeRetriever({(2023, "interest_expense"): (RankedTable("VJC23C:table:1", 2),)}))

    def test_unknown_chunk_is_ignored_with_warning(self):
        out = one_with(_interest("XYZ:table:9", "VJC23C:table:1"))
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 2))
        self.assertIn("retrieval_unknown_chunks:1", out.extracted.warnings)

    def test_only_unknown_chunks_reads_full_report(self):
        out = one_with(_interest("XYZ:table:9", "XYZ:table:8"))
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "full_scan", None))
        self.assertIn("retrieval_unknown_chunks:2", out.extracted.warnings)

    def test_ranked_malformed_table_is_not_unknown(self):
        bad_json = dataclasses.replace(is_table("VJC23C", is_rows(2023), table_id=2), rows_json="{not json")
        store = store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023)), bad_json]))
        out = one_with(_interest("VJC23C:table:2", "VJC23C:table:1"), store)
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 2))
        self.assertIn("skipped_table:VJC23C:2", out.extracted.warnings)
        self.assertFalse(any(w.startswith("retrieval_unknown_chunks") for w in out.extracted.warnings))

    def test_not_found_carries_retrieval_warnings(self):
        out = one_with(FakeRetriever(error=TableRetrievalError("x")), _no_interest_store())
        self.assertIsInstance(out, NotFound)
        self.assertIn("retrieval_error:TableRetrievalError", out.warnings)

    def test_failed_retrieval_is_asked_once_per_source(self):
        store = store_of(
            (doc("VJC22C", 2022), [is_table("VJC22C", is_rows(2022, interest=False))]),
            (doc("VJC23C", 2023), base_tables()),
        )
        r = FakeRetriever(error=TableRetrievalError("x"))
        LeafExtractor(store, tree, templates, retriever=r).extract(
            "VJC", "consolidated", [LeafRequest("interest_expense", 2022, "flow"), INTEREST_2023])
        self.assertEqual(sorted(r.calls), [("VJC", "consolidated", 2022, "interest_expense"),
                                           ("VJC", "consolidated", 2023, "interest_expense")])


class RetrievalLocatorTests(unittest.TestCase):
    def test_locator_receives_only_ranked_views(self):
        for name, retriever, seen in (
            ("ranked", _interest("VJC23C:table:2"), [["VJC23C:table:2"]]),
            ("empty answer", FakeRetriever(), [["VJC23C:table:1", "VJC23C:table:2"]]),
            ("retrieval error", FakeRetriever(error=TableRetrievalError("x")),
             [["VJC23C:table:1", "VJC23C:table:2"]]),
        ):
            with self.subTest(name):
                locator = ViewsLocator(_not_present)
                out = one_with(retriever, _no_interest_store(), fallback=locator)
                self.assertIsInstance(out, NotFound)
                self.assertEqual(locator.views, seen)
                self.assertEqual(len(retriever.calls), 1)  # the locator's ranking is a cache hit

    def test_locator_found_in_ranked_table_is_retrieved(self):
        locator = StubLocator(read_label("Lãi tiền gửi ngân hàng", col=2))
        out = one_with(_interest("VJC23C:table:2"), _no_interest_store(), fallback=locator)
        self.assertEqual(out.extracted.method, "llm_located")
        self.assertEqual(out.extracted.fact.source.chunk_id, "VJC23C:table:2")
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 1))

    def test_locator_found_outside_ranked_tables_is_full_scan(self):
        for name, retriever in (("empty answer", FakeRetriever()),
                                ("retrieval error", FakeRetriever(error=TableRetrievalError("x")))):
            with self.subTest(name):
                locator = StubLocator(read_label("- Trong đó: Chi phí lãi vay"))
                out = one_with(retriever, _no_interest_store(), fallback=locator)
                self.assertEqual(out.extracted.method, "llm_located")
                self.assertEqual(out.extracted.fact.source.chunk_id, "VJC23C:table:1")
                self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "full_scan", None))

    def test_located_found_keeps_retrieval_warnings(self):
        locator = StubLocator(read_label("- Trong đó: Chi phí lãi vay"))
        out = one_with(FakeRetriever(error=TableRetrievalError("x")), _no_interest_store(), fallback=locator)
        self.assertIn("retrieval_error:TableRetrievalError", out.extracted.warnings)

    def test_next_year_locator_uses_next_year_ranking(self):
        only_next = store_of((doc("VJC24C", 2024), [is_table("VJC24C", is_rows(2024, interest=False)),
                                                    record("VJC24C", 2, 30, _NOTE_ROWS, kind="note")]))
        r = FakeRetriever({(2024, "interest_expense"): ranked("VJC24C:table:1")})
        locator = ViewsLocator(read_label("- Trong đó: Chi phí lãi vay", col=4))
        out = one_with(r, only_next, fallback=locator)
        self.assertEqual(locator.views, [["VJC24C:table:1"]])
        self.assertEqual(r.calls, [("VJC", "consolidated", 2024, "interest_expense")])
        self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 1))
        self.assertIn("primary_report_missing", out.extracted.warnings)

    def test_no_retriever_locator_has_no_trace(self):
        locator = ViewsLocator(read_label("- Trong đó: Chi phí lãi vay"))
        out = one_with(None, _no_interest_store(), fallback=locator)
        self.assertEqual(locator.views, [["VJC23C:table:1", "VJC23C:table:2"]])
        self.assertIsNone(out.extracted.retrieval)


if __name__ == "__main__":
    unittest.main()

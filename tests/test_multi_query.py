import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from langchain_core.documents import Document

from config.settings import ParaphraseSettings
from src.query_paraphrase.entities import EntityRegistry
from src.query_paraphrase.metric_catalog import METRIC_CATALOG
from src.query_paraphrase.paraphraser import QueryParaphraser
from src.query_paraphrase.schemas import ParaphraseResult, QueryPlan, ValidationIssue
from src.retrieval.bm25 import build_bm25
from src.retrieval.hybrid import build_hybrid_retriever
from src.retrieval.multi_query import (EvidenceGroup, RetrievalPolicy, SupplementalEvidence,
                                       overall_status, retrieve_subqueries)
from src.retrieval.reranker import build_reranked_retriever

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = EntityRegistry.from_csv(ROOT / "data" / "code_stock.csv")
SETTINGS = ParaphraseSettings()
LABELS = {metric.id: metric.label_vi for metric in METRIC_CATALOG.values()}
SCOPE_TEXT = {"consolidated": "báo cáo hợp nhất", "separate": "báo cáo riêng", None: None}


def sq(sid="q1", *, ticker="VJC", year=2022, metric="revenue", scope=None, period=None,
       unit=None, types=None):
    parts = (LABELS.get(metric), ticker, year, period, SCOPE_TEXT[scope], unit)
    text = " ".join(str(part) for part in parts if part) or "Thông tin tài chính"
    return {"id": sid, "question": text + "?", "search_query": text, "ticker": ticker,
            "company_name": None, "year": year, "period": period, "scope": scope,
            "metric": metric, "metric_text": None, "unit": unit,
            "document_types": types or [], "context_refs": {}}


def plan(question, *subqueries, operation="lookup", relationships=()):
    return {"original_question": question, "operation": operation,
            "operation_description": None, "subqueries": list(subqueries),
            "relationships": list(relationships)}


def valid(*subqueries, operation="lookup", relationships=()):
    payload = plan("câu hỏi", *subqueries, operation=operation, relationships=relationships)
    return ParaphraseResult(status="valid", plan=QueryPlan.model_validate(payload))


def doc(text, **metadata):
    return Document(page_content=text, metadata={"doc_id": text, "table_id": 0, **metadata})


class FakeRetriever:
    """Records deep copies of every mapping; responses come from a callable or a queue."""

    def __init__(self, respond=None):
        self.calls = []
        self._respond = respond or (lambda call: [])

    def invoke(self, mapping):
        self.calls.append(copy.deepcopy(mapping))
        outcome = self._respond(mapping)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def mapping(question, scope, tickers=("VJC",), years=(2022,)):
    return {"question": question,
            "metadata_filters": {"tickers": list(tickers), "years": list(years),
                                 "scopes": [scope]}}


class NonValidTests(unittest.TestCase):
    def test_non_valid_results_make_zero_calls(self):
        issue = ValidationIssue(code="x", message="y", subquery_id=None)
        for status in ("invalid", "needs_clarification", "unavailable"):
            with self.subTest(status=status):
                retriever = FakeRetriever()
                result = ParaphraseResult(status=status, plan=None, issues=[issue])
                self.assertEqual(retrieve_subqueries(result, retriever), [])
                self.assertEqual(retriever.calls, [])

    def test_valid_status_without_plan_makes_zero_calls(self):
        retriever = FakeRetriever()
        self.assertEqual(retrieve_subqueries(ParaphraseResult(status="valid", plan=None),
                                             retriever), [])
        self.assertEqual(retriever.calls, [])

    def test_rejects_non_result_and_bad_arguments_before_any_call(self):
        retriever = FakeRetriever()
        result = valid(sq())
        with self.assertRaises(TypeError):
            retrieve_subqueries({"status": "valid"}, retriever)
        with self.assertRaises(TypeError):
            retrieve_subqueries(result, object())
        with self.assertRaises(TypeError):
            retrieve_subqueries(result, retriever, policy="consolidated")
        for bad in ((), [TimeoutError], (ValueError, 5), TimeoutError, (BaseException,)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                retrieve_subqueries(result, retriever, retrieval_errors=bad)
        self.assertEqual(retriever.calls, [])

    def test_result_is_revalidated_from_model_dump_not_isinstance(self):
        class Foreign:
            def __init__(self, inner):
                self._inner = inner

            def model_dump(self):
                return self._inner.model_dump()

        retriever = FakeRetriever(lambda call: [doc("x")])
        groups = retrieve_subqueries(Foreign(valid(sq())), retriever)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(retriever.calls), 1)

        class Broken:
            def model_dump(self):
                return {"status": "valid", "plan": None, "unexpected": 1}

        with self.assertRaises(ValueError):
            retrieve_subqueries(Broken(), retriever)
        self.assertEqual(len(retriever.calls), 1)


class PolicyTests(unittest.TestCase):
    def test_defaults_validation_and_from_settings(self):
        self.assertEqual(RetrievalPolicy(), RetrievalPolicy("consolidated", True))
        for bad in ({"default_report_scope": "explanations"}, {"default_report_scope": None},
                    {"notes_fallback": 1}, {"notes_fallback": "true"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RetrievalPolicy(**bad)
        settings = ParaphraseSettings(default_report_scope="separate", notes_fallback=False)
        self.assertEqual(RetrievalPolicy.from_settings(settings),
                         RetrievalPolicy("separate", False))

    def test_policy_none_uses_loaded_settings(self):
        retriever = FakeRetriever()
        with mock.patch.dict("os.environ", {"FINAGENT_PARAPHRASE_DEFAULT_REPORT_SCOPE": "separate",
                                            "FINAGENT_PARAPHRASE_NOTES_FALLBACK": "false"}):
            groups = retrieve_subqueries(valid(sq()), retriever)
        self.assertEqual(retriever.calls[0]["metadata_filters"]["scopes"], ["separate"])
        self.assertEqual(len(retriever.calls), 1)  # fallback disabled by environment
        self.assertEqual(groups[0].scope_source, "policy_default")


class PrimaryPassTests(unittest.TestCase):
    def test_two_years_make_two_exact_mappings_in_order(self):
        first, second = doc("a2022"), doc("a2023")
        retriever = FakeRetriever(lambda call: [first] if call["metadata_filters"]["years"] == [2022]
                                  else [second])
        result = valid(sq("q1", year=2022), sq("q2", year=2023), operation="compare",
                       relationships=[{"kind": "compare", "subquery_ids": ["q1", "q2"],
                                       "description": None}])
        groups = retrieve_subqueries(result, retriever, policy=RetrievalPolicy())
        self.assertEqual(retriever.calls, [
            mapping("Doanh thu VJC 2022", "consolidated", years=(2022,)),
            mapping("Doanh thu VJC 2023", "consolidated", years=(2023,))])
        for call in retriever.calls:
            self.assertEqual(set(call), {"question", "metadata_filters"})
        self.assertEqual([g.subquery.id for g in groups], ["q1", "q2"])
        self.assertEqual([g.primary_evidence for g in groups], [[first], [second]])
        self.assertEqual([g.status for g in groups], ["ok", "ok"])
        self.assertEqual(groups[0].primary_filters, retriever.calls[0]["metadata_filters"])

    def test_explicit_scopes_and_policy_default_labels(self):
        cases = [("consolidated", RetrievalPolicy("separate"), "consolidated", "requested"),
                 ("separate", RetrievalPolicy("consolidated"), "separate", "requested"),
                 (None, RetrievalPolicy("separate"), "separate", "policy_default"),
                 (None, RetrievalPolicy("consolidated"), "consolidated", "policy_default")]
        for scope, policy, effective, source in cases:
            with self.subTest(scope=scope, policy=policy.default_report_scope):
                retriever = FakeRetriever(lambda call: [doc("x")])
                group, = retrieve_subqueries(valid(sq(scope=scope)), retriever, policy=policy)
                self.assertEqual(retriever.calls[0]["metadata_filters"]["scopes"], [effective])
                self.assertEqual(group.requested_scope, scope)
                self.assertEqual(group.effective_scope, effective)
                self.assertEqual(group.scope_source, source)
                self.assertEqual(group.status, "ok")

    def test_missing_ticker_and_year_are_unrestricted(self):
        retriever = FakeRetriever(lambda call: [doc("x")])
        retrieve_subqueries(valid(sq(ticker=None, year=None, metric="revenue")), retriever,
                            policy=RetrievalPolicy(notes_fallback=False))
        self.assertEqual(retriever.calls[0]["metadata_filters"],
                         {"tickers": [], "years": [], "scopes": ["consolidated"]})

    def test_financial_statements_only_never_falls_back(self):
        retriever = FakeRetriever()
        group, = retrieve_subqueries(valid(sq(types=["financial_statements"])), retriever,
                                     policy=RetrievalPolicy(notes_fallback=True))
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(group.status, "empty")

    def test_period_and_unit_are_kept_and_warned_without_downgrade(self):
        retriever = FakeRetriever(lambda call: [doc("x")])
        group, = retrieve_subqueries(valid(sq(period="quý 2", unit="triệu đồng")), retriever)
        self.assertIs(group.period_verified, False)
        self.assertIs(group.unit_verified, False)
        self.assertEqual(group.subquery.period, "quý 2")
        self.assertEqual(group.subquery.unit, "triệu đồng")
        self.assertEqual(group.status, "ok")
        self.assertTrue(any("period" in w for w in group.warnings))
        self.assertTrue(any("unit" in w for w in group.warnings))
        self.assertEqual(retriever.calls[0]["metadata_filters"]["years"], [2022])
        plain, = retrieve_subqueries(valid(sq()), retriever)
        self.assertIsNone(plain.period_verified)
        self.assertIsNone(plain.unit_verified)


class SupplementalPassTests(unittest.TestCase):
    def test_notes_only_goes_to_supplemental_with_unverified_scope(self):
        note = doc("note")
        retriever = FakeRetriever(lambda call: [note])
        group, = retrieve_subqueries(valid(sq(types=["explanations"])), retriever)
        self.assertEqual(retriever.calls, [mapping("Doanh thu VJC 2022", "explanations")])
        self.assertEqual(group.primary_evidence, [])
        self.assertIsNone(group.primary_filters)
        item, = group.supplemental
        self.assertEqual((item.document_type, item.status, item.source), ("explanations", "ok",
                                                                         "requested"))
        self.assertEqual(item.documents, [note])
        self.assertIs(item.scope_verified, False)
        self.assertIsNone(item.report_scope)
        self.assertEqual(item.filters["scopes"], ["explanations"])
        self.assertEqual(group.status, "ok")
        self.assertIsNone(group.effective_scope)

    def test_letters_and_combined_types_stay_distinct_in_order(self):
        by_scope = {"consolidated": [doc("fin")], "explanations": [doc("note")],
                    "letters": [doc("letter")]}
        retriever = FakeRetriever(lambda call: by_scope[call["metadata_filters"]["scopes"][0]])
        types = ["letters", "explanations", "financial_statements"]
        group, = retrieve_subqueries(valid(sq(types=types)), retriever)
        self.assertEqual([c["metadata_filters"]["scopes"] for c in retriever.calls],
                         [["consolidated"], ["explanations"], ["letters"]])
        self.assertEqual([d.page_content for d in group.primary_evidence], ["fin"])
        self.assertEqual([(s.document_type, [d.page_content for d in s.documents])
                          for s in group.supplemental],
                         [("explanations", ["note"]), ("letters", ["letter"])])
        self.assertTrue(all(not s.scope_verified and s.report_scope is None
                            for s in group.supplemental))
        self.assertEqual(group.status, "ok")
        only_letters, = retrieve_subqueries(valid(sq(types=["letters"])),
                                            FakeRetriever(lambda call: [doc("l")]))
        self.assertEqual(only_letters.supplemental[0].document_type, "letters")
        self.assertEqual(only_letters.primary_evidence, [])

    def test_notes_with_explicit_report_scope_is_partial(self):
        for types in (["explanations"], ["letters"], ["financial_statements", "explanations"]):
            with self.subTest(types=types):
                retriever = FakeRetriever(lambda call: [doc("x")])
                group, = retrieve_subqueries(valid(sq(scope="consolidated", types=types)),
                                             retriever)
                self.assertEqual(group.status, "partial")
                self.assertTrue(any("cannot be verified" in w for w in group.warnings))
                self.assertEqual(group.requested_scope, "consolidated")

    def test_fallback_runs_once_when_primary_is_empty_and_stays_supplemental(self):
        note = doc("note")
        retriever = FakeRetriever(lambda call: [note] if call["metadata_filters"]["scopes"] ==
                                  ["explanations"] else [])
        group, = retrieve_subqueries(valid(sq()), retriever, policy=RetrievalPolicy())
        self.assertEqual(retriever.calls, [mapping("Doanh thu VJC 2022", "consolidated"),
                                           mapping("Doanh thu VJC 2022", "explanations")])
        self.assertEqual(group.primary_evidence, [])
        item, = group.supplemental
        self.assertEqual((item.source, item.status, item.documents), ("fallback", "ok", [note]))
        self.assertFalse(item.scope_verified)
        self.assertEqual(group.status, "partial")

    def test_fallback_empty_too_gives_empty_and_filters_never_widen(self):
        retriever = FakeRetriever()
        group, = retrieve_subqueries(valid(sq(scope="separate")), retriever)
        self.assertEqual(group.status, "empty")
        self.assertEqual(len(retriever.calls), 2)
        for call in retriever.calls:
            self.assertEqual(call["metadata_filters"]["tickers"], ["VJC"])
            self.assertEqual(call["metadata_filters"]["years"], [2022])
            self.assertEqual(len(call["metadata_filters"]["scopes"]), 1)

    def test_no_fallback_when_disabled_or_primary_has_documents_or_notes_requested(self):
        cases = [(RetrievalPolicy(notes_fallback=False), lambda call: [], []),
                 (RetrievalPolicy(), lambda call: [doc("x")], []),
                 (RetrievalPolicy(), lambda call: [], ["financial_statements", "explanations"])]
        for policy, respond, types in cases:
            with self.subTest(policy=policy, types=types):
                retriever = FakeRetriever(respond)
                retrieve_subqueries(valid(sq(types=types)), retriever, policy=policy)
                explanation_calls = [c for c in retriever.calls
                                     if c["metadata_filters"]["scopes"] == ["explanations"]]
                self.assertEqual(len(explanation_calls), 1 if types else 0)

    def test_no_fallback_after_primary_error_and_error_is_visible(self):
        retriever = FakeRetriever(lambda call: TimeoutError("secret-token-123"))
        group, = retrieve_subqueries(valid(sq()), retriever, policy=RetrievalPolicy())
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(group.status, "error")
        self.assertEqual(group.errors, ["financial_statements: TimeoutError"])
        self.assertNotIn("secret", json.dumps([group.errors, group.warnings]))
        self.assertEqual(group.supplemental, [])

    def test_supplemental_error_is_visible_and_other_groups_continue(self):
        def respond(call):
            filters = call["metadata_filters"]
            if filters["years"] == [2022] and filters["scopes"] == ["explanations"]:
                return ConnectionError("password=hunter2")
            return [doc("ok")]

        retriever = FakeRetriever(respond)
        groups = retrieve_subqueries(
            valid(sq("q1", year=2022, types=["financial_statements", "explanations"]),
                  sq("q2", year=2023), operation="compare",
                  relationships=[{"kind": "compare", "subquery_ids": ["q1", "q2"],
                                  "description": None}]), retriever)
        first, second = groups
        self.assertEqual(first.status, "partial")
        self.assertEqual(first.supplemental[0].status, "error")
        self.assertEqual(first.supplemental[0].error, "ConnectionError")
        self.assertEqual(first.errors, ["explanations: ConnectionError"])
        self.assertEqual(len(first.primary_evidence), 1)
        self.assertEqual(second.status, "ok")
        self.assertNotIn("hunter2", repr(groups[0].errors) + repr(first.supplemental[0].error))
        self.assertEqual(overall_status(groups), "partial")

    def test_error_in_one_group_does_not_stop_the_next(self):
        retriever = FakeRetriever(lambda call: OSError("boom") if call["metadata_filters"]["years"]
                                  == [2022] else [doc("x")])
        groups = retrieve_subqueries(valid(sq("q1", year=2022), sq("q2", year=2023),
                                           operation="compare",
                                           relationships=[{"kind": "compare",
                                                           "subquery_ids": ["q1", "q2"],
                                                           "description": None}]), retriever)
        self.assertEqual([g.status for g in groups], ["error", "ok"])
        self.assertEqual(len(retriever.calls), 2)

    def test_success_empty_plus_error_pass_is_partial_not_error(self):
        def respond(call):
            if call["metadata_filters"]["scopes"] == ["letters"]:
                return TimeoutError()
            return []

        group, = retrieve_subqueries(valid(sq(types=["financial_statements", "letters"])),
                                     FakeRetriever(respond))
        self.assertEqual(group.status, "partial")


class OverallStatusTests(unittest.TestCase):
    def _two(self, respond):
        result = valid(sq("q1", year=2022), sq("q2", year=2023), operation="compare",
                       relationships=[{"kind": "compare", "subquery_ids": ["q1", "q2"],
                                       "description": None}])
        return retrieve_subqueries(result, FakeRetriever(respond),
                                   policy=RetrievalPolicy(notes_fallback=False))

    def test_all_empty_all_error_all_ok_and_mixed(self):
        self.assertEqual(overall_status(self._two(lambda c: [])), "empty")
        self.assertEqual(overall_status(self._two(lambda c: TimeoutError())), "error")
        self.assertEqual(overall_status(self._two(lambda c: [doc("x")])), "ok")
        mixed = self._two(lambda c: [] if c["metadata_filters"]["years"] == [2022] else [doc("x")])
        self.assertEqual(overall_status(mixed), "partial")
        error_and_ok = self._two(lambda c: TimeoutError() if c["metadata_filters"]["years"]
                                 == [2022] else [doc("x")])
        self.assertEqual(overall_status(error_and_ok), "partial")
        self.assertEqual(overall_status([]), "not_run")


class FailureSemanticsTests(unittest.TestCase):
    def test_programmer_errors_propagate_and_are_not_masked(self):
        for error in (ValueError("v"), KeyError("k"), TypeError("t")):
            with self.subTest(error=type(error).__name__):
                retriever = FakeRetriever(lambda call, error=error: error)
                with self.assertRaises(type(error)):
                    retrieve_subqueries(valid(sq()), retriever)
                self.assertEqual(len(retriever.calls), 1)

    def test_only_injected_retrieval_errors_are_recorded(self):
        retriever = FakeRetriever(lambda call: TimeoutError())
        with self.assertRaises(TimeoutError):
            retrieve_subqueries(valid(sq()), retriever, retrieval_errors=(ConnectionError,))
        group, = retrieve_subqueries(valid(sq()), retriever, retrieval_errors=(TimeoutError,))
        self.assertEqual(group.status, "error")

    def test_non_list_retriever_output_is_a_type_error(self):
        with self.assertRaises(TypeError):
            retrieve_subqueries(valid(sq()), FakeRetriever(lambda call: None))

    def test_empty_scopes_make_zero_calls(self):
        class NoScopes:
            def __init__(self, **kwargs):
                self.scopes = []

        retriever = FakeRetriever()
        with mock.patch("src.retrieval.multi_query.MetadataFilters", NoScopes):
            with self.assertRaises(ValueError):
                retrieve_subqueries(valid(sq()), retriever)
        self.assertEqual(retriever.calls, [])

    def test_all_filters_are_validated_before_the_first_call(self):
        from src.retrieval.filters import MetadataFilters

        def factory(**kwargs):
            if kwargs["tickers"] == ["HPG"]:
                raise ValueError("bad filter")
            return MetadataFilters(**kwargs)

        result = valid(sq("q1", ticker="VJC"), sq("q2", ticker="HPG"), operation="compare",
                       relationships=[{"kind": "compare", "subquery_ids": ["q1", "q2"],
                                       "description": None}])
        retriever = FakeRetriever(lambda call: [doc("x")])
        with mock.patch("src.retrieval.multi_query.MetadataFilters", factory):
            with self.assertRaises(ValueError):
                retrieve_subqueries(result, retriever)
        self.assertEqual(retriever.calls, [])

    def test_call_count_is_bounded_by_subqueries_times_passes(self):
        subqueries = [sq(f"q{n}", year=2015 + n, types=["financial_statements", "explanations",
                                                        "letters"]) for n in range(1, 4)]
        retriever = FakeRetriever()
        retrieve_subqueries(valid(*subqueries), retriever)
        self.assertEqual(len(retriever.calls), 9)
        fallback = FakeRetriever()
        retrieve_subqueries(valid(sq("q1", year=2020), sq("q2", year=2021), operation="compare",
                                  relationships=[{"kind": "compare", "subquery_ids": ["q1", "q2"],
                                                  "description": None}]), fallback)
        self.assertLessEqual(len(fallback.calls), 2 * 2)

    def test_documents_are_returned_untouched_and_never_merged(self):
        shared = doc("shared", ticker="VJC", report_year=2022)
        snapshot = copy.deepcopy(shared)
        retriever = FakeRetriever(lambda call: [shared])
        groups = retrieve_subqueries(valid(sq("q1", year=2022), sq("q2", year=2023),
                                           operation="compare",
                                           relationships=[{"kind": "compare",
                                                           "subquery_ids": ["q1", "q2"],
                                                           "description": None}]), retriever)
        for group in groups:
            self.assertEqual(len(group.primary_evidence), 1)
            self.assertIs(group.primary_evidence[0], shared)
        self.assertEqual(shared, snapshot)
        self.assertIsNot(groups[0].primary_evidence, groups[1].primary_evidence)

    def test_returned_filters_are_independent_of_retriever_mutation(self):
        def respond(call):
            call["metadata_filters"]["tickers"].append("HPG")
            return [doc("x")]

        group, = retrieve_subqueries(valid(sq()), FakeRetriever(respond))
        self.assertEqual(group.primary_filters["tickers"], ["VJC"])


class RefineQuestionNeverCalledTests(unittest.TestCase):
    def test_real_hybrid_does_not_call_refine_question(self):
        documents = corpus()
        dense = RecordingDense(documents)
        hybrid = build_hybrid_retriever(build_bm25(documents, k=10), dense, k=5, candidate_k=10)

        def forbidden(*args, **kwargs):
            raise AssertionError("refine_question must not be called")

        with mock.patch("src.main.refine_questions.refine_question", side_effect=forbidden) as spy:
            groups = retrieve_subqueries(valid(sq()), hybrid, policy=RetrievalPolicy())
        spy.assert_not_called()
        self.assertEqual(groups[0].status, "ok")
        self.assertEqual(groups[0].primary_evidence[0].metadata["ticker"], "VJC")


def _where_matches(where, metadata):
    if where is None:
        return True
    if "$and" in where:
        return all(_where_matches(part, metadata) for part in where["$and"])
    (field, condition), = where.items()
    return metadata.get(field) in condition["$in"]


class RecordingDense:
    """Fake Chroma store that records the where filter and applies it to its documents."""

    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def similarity_search(self, query, k, filter=None):
        self.calls.append({"query": query, "k": k, "filter": copy.deepcopy(filter)})
        return [d for d in self.documents if _where_matches(filter, d.metadata)][:k]


class FakeEncoder:
    def score(self, pairs):
        return [float(len(text)) for query, text in pairs]


def corpus():
    def entry(ticker, year, scope, text, table_id=0):
        return Document(page_content=text, metadata={
            "doc_id": f"{ticker}_{year}_{scope}", "table_id": table_id, "ticker": ticker,
            "report_year": year, "scope": scope, "page_ids": [1]})

    return [
        entry("VJC", 2022, "consolidated", "Doanh thu VJC nam 2022 hop nhat"),
        entry("VJC", 2022, "consolidated", "Loi nhuan sau thue VJC nam 2022 hop nhat", 1),
        entry("VJC", 2023, "consolidated", "Doanh thu VJC nam 2023 hop nhat"),
        entry("VJC", 2023, "consolidated", "Loi nhuan sau thue VJC nam 2023 hop nhat", 1),
        entry("VJC", 2022, "separate", "Doanh thu VJC nam 2022 rieng"),
        entry("VJC", 2022, "explanations", "Thuyet minh doanh thu VJC nam 2022"),
        entry("HPG", 2022, "consolidated", "Doanh thu HPG nam 2022 hop nhat"),
        entry("HPG", 2023, "consolidated", "Doanh thu HPG nam 2023 hop nhat"),
        entry("FPT", 2021, "consolidated", "Gia von FPT nam 2021 hop nhat"),
        entry("FPT", 2021, "separate", "Tai san FPT nam 2021 rieng"),
        entry("MWG", 2020, "consolidated", "Von chu so huu MWG nam 2020"),
        entry("MWG", 2020, "separate", "No phai tra MWG nam 2020"),
    ]


COMPARE_QUESTION = "So sánh doanh thu VJC năm 2022 và 2023"


def comparison_plan():
    return plan(COMPARE_QUESTION,
                sq("q1", year=2022), sq("q2", year=2023), operation="compare",
                relationships=[{"kind": "compare", "subquery_ids": ["q1", "q2"],
                                "description": "so sánh theo năm"}])


class EndToEndTests(unittest.TestCase):
    def build(self, payload):
        documents = corpus()
        dense = RecordingDense(documents)
        hybrid = build_hybrid_retriever(build_bm25(documents, k=10), dense, k=10, candidate_k=10)
        reranked = build_reranked_retriever(hybrid, FakeEncoder(), k=3)
        prompts = []

        def llm(prompt):
            prompts.append(prompt)
            return json.dumps(payload, ensure_ascii=False)

        paraphraser = QueryParaphraser(llm, registry=REGISTRY, settings=SETTINGS)
        return paraphraser, reranked, dense, prompts, documents

    def test_string_entrypoint_retrieves_per_subquery_with_recorded_filters(self):
        from src.main.query_pipeline import PipelineResult, build_query_pipeline

        paraphraser, retriever, dense, prompts, documents = self.build(comparison_plan())
        snapshot = copy.deepcopy(documents)
        with mock.patch("src.main.refine_questions.refine_question",
                        side_effect=AssertionError("refine_question called")):
            result = build_query_pipeline(paraphraser, retriever,
                                          policy=RetrievalPolicy(notes_fallback=False)
                                          ).invoke(COMPARE_QUESTION)
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(result.paraphrase.status, "valid")
        self.assertEqual(result.retrieval_status, "ok")
        self.assertEqual([c["filter"] for c in dense.calls], [
            {"$and": [{"ticker": {"$in": ["VJC"]}}, {"report_year": {"$in": [2022]}},
                      {"scope": {"$in": ["consolidated"]}}]},
            {"$and": [{"ticker": {"$in": ["VJC"]}}, {"report_year": {"$in": [2023]}},
                      {"scope": {"$in": ["consolidated"]}}]}])
        first, second = result.evidence_groups
        for group, year in ((first, 2022), (second, 2023)):
            self.assertTrue(group.primary_evidence)
            for found in group.primary_evidence:
                self.assertEqual((found.metadata["ticker"], found.metadata["report_year"],
                                  found.metadata["scope"]), ("VJC", year, "consolidated"))
                self.assertIn("rrf_score", found.metadata)
                self.assertIn("rerank_score", found.metadata)
                self.assertIn("bm25_rank", found.metadata)
            scores = [d.metadata["rerank_score"] for d in group.primary_evidence]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertEqual(group.primary_filters["years"], [year])
        self.assertEqual(documents, snapshot)
        plan_ = result.paraphrase.plan
        self.assertEqual([r.kind for r in plan_.relationships], ["compare"])
        self.assertEqual(plan_.relationships[0].subquery_ids, ["q1", "q2"])

    def test_history_entrypoint_and_extra_keys(self):
        from src.main.query_pipeline import build_query_pipeline

        paraphraser, retriever, dense, prompts, _ = self.build(comparison_plan())
        pipeline = build_query_pipeline(paraphraser, retriever)
        history = [{"question": "Xin chào", "answer": "Chào bạn"}]
        result = pipeline.invoke({"question": COMPARE_QUESTION, "qa_history": history})
        self.assertEqual(result.retrieval_status, "ok")
        self.assertIn("Xin chào", prompts[0])
        self.assertEqual(len(dense.calls), 2)
        for bad in ({"question": COMPARE_QUESTION, "other": 1}, {"qa_history": []}, 5, None, []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pipeline.invoke(bad)
        self.assertEqual(len(prompts), 1)

    def test_multiple_metrics_and_entities_make_one_group_each(self):
        from src.main.query_pipeline import build_query_pipeline

        question = "Doanh thu và lợi nhuận sau thuế của VJC và HPG năm 2022"
        payload = plan(question,
                       sq("q1", ticker="VJC", year=2022, metric="revenue"),
                       sq("q2", ticker="VJC", year=2022, metric="profit_after_tax"),
                       sq("q3", ticker="HPG", year=2022, metric="revenue"),
                       sq("q4", ticker="HPG", year=2022, metric="profit_after_tax"),
                       operation="lookup")
        paraphraser, retriever, dense, _, _ = self.build(payload)
        result = build_query_pipeline(paraphraser, retriever,
                                      policy=RetrievalPolicy(notes_fallback=False)).invoke(question)
        self.assertEqual(result.paraphrase.status, "valid", result.paraphrase.issues)
        self.assertEqual(len(result.evidence_groups), 4)
        self.assertEqual([len(c["filter"]["$and"]) for c in dense.calls
                          if c["filter"] is not None], [3] * len(dense.calls))

    def test_non_valid_paraphrase_is_not_run_with_zero_calls(self):
        from src.main.query_pipeline import build_query_pipeline

        paraphraser, retriever, dense, _, _ = self.build({"not": "a plan"})
        result = build_query_pipeline(paraphraser, retriever).invoke(COMPARE_QUESTION)
        self.assertNotEqual(result.paraphrase.status, "valid")
        self.assertEqual(result.evidence_groups, [])
        self.assertEqual(result.retrieval_status, "not_run")
        self.assertEqual(dense.calls, [])

    def test_paraphraser_input_error_propagates(self):
        from src.main.query_pipeline import build_query_pipeline

        paraphraser, retriever, dense, prompts, _ = self.build(comparison_plan())
        with self.assertRaises(ValueError):
            build_query_pipeline(paraphraser, retriever).invoke("   ")
        self.assertEqual(prompts, [])
        self.assertEqual(dense.calls, [])

    def test_builder_validates_arguments_and_calls_paraphrase_once(self):
        from src.main.query_pipeline import build_query_pipeline

        calls = []

        class Paraphraser:
            def paraphrase(self, question, qa_history=None):
                calls.append((question, qa_history))
                return valid(sq())

        retriever = FakeRetriever(lambda call: [doc("x")])
        with self.assertRaises(TypeError):
            build_query_pipeline(object(), retriever)
        with self.assertRaises(TypeError):
            build_query_pipeline(Paraphraser(), object())
        with self.assertRaises(TypeError):
            build_query_pipeline(Paraphraser(), retriever, retrieval_errors=())
        result = build_query_pipeline(Paraphraser(), retriever).invoke(
            {"question": "q", "qa_history": [1]})
        self.assertEqual(calls, [("q", [1])])
        self.assertEqual(result.retrieval_status, "ok")

    def test_pipeline_reports_error_and_partial_statuses(self):
        from src.main.query_pipeline import build_query_pipeline

        class Paraphraser:
            def paraphrase(self, question, qa_history=None):
                return valid(sq("q1", year=2022), sq("q2", year=2023), operation="compare",
                             relationships=[{"kind": "compare", "subquery_ids": ["q1", "q2"],
                                             "description": None}])

        errors = FakeRetriever(lambda call: TimeoutError())
        self.assertEqual(build_query_pipeline(Paraphraser(), errors).invoke("q").retrieval_status,
                         "error")
        half = FakeRetriever(lambda call: [] if call["metadata_filters"]["years"] == [2022]
                             else [doc("x")])
        policy = RetrievalPolicy(notes_fallback=False)
        self.assertEqual(build_query_pipeline(Paraphraser(), half,
                                              policy=policy).invoke("q").retrieval_status,
                         "partial")


class DataclassContractTests(unittest.TestCase):
    def test_types_are_exposed(self):
        self.assertTrue(hasattr(EvidenceGroup, "__dataclass_fields__"))
        self.assertTrue(hasattr(SupplementalEvidence, "__dataclass_fields__"))
        item = SupplementalEvidence("letters", {}, [], "empty")
        self.assertFalse(item.scope_verified)
        self.assertIsNone(item.report_scope)
        self.assertIsNone(item.error)


if __name__ == "__main__":
    unittest.main()

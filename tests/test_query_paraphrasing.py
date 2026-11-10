import copy
import json
import unittest

from pydantic import BaseModel

from config.settings import ParaphraseSettings
from src.query_paraphrase import (QATurn, QueryParaphraser, QueryPlan, SemanticReview,
                                  build_paraphrase_prompt, prepare_history,
                                  structured_llm_adapter, validate_plan)
from src.query_paraphrase.metric_catalog import METRIC_CATALOG
from tests.test_query_validation import REGISTRY, SETTINGS, Q_VJC, plan, sq

SECRET = "sk-secret-token-123 https://internal.example/key"


class ProviderError(Exception):
    pass


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.prompts = response, error, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


class FakeReviewer:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


SIMPLE = plan(Q_VJC, sq(ticker="VJC", year=2023, metric="revenue"))
Q_CMP = "So sánh doanh thu VJC 2022 và 2023"
COMPARE = plan(Q_CMP, sq("q1", ticker="VJC", year=2022, metric="revenue"),
               sq("q2", ticker="VJC", year=2023, metric="revenue"), operation="compare",
               relationships=[{"kind": "compare", "subquery_ids": ["q1", "q2"],
                               "description": None}])
PREVIOUS = {"question": "Doanh thu VJC năm 2022 là bao nhiêu?", "answer": "Khoảng 40.000 tỷ đồng."}
Q_FOLLOW = "Còn năm 2023?"
FOLLOW = plan(Q_FOLLOW, sq(ticker="VJC", year=2023, metric="revenue",
                           refs={"ticker": 0, "metric": 0}))


def paraphraser(llm, **kwargs):
    return QueryParaphraser(llm, registry=REGISTRY, settings=SETTINGS, **kwargs)


class PromptTests(unittest.TestCase):
    def setUp(self):
        history = prepare_history([PREVIOUS], SETTINGS)
        self.prompt = build_paraphrase_prompt(Q_FOLLOW, history, settings=SETTINGS)

    def test_vietnamese_guidance_and_data_delimiters(self):
        for text in ("tiếng Việt", "không tự suy ra công ty", "không tự suy ra năm",
                     "không tách", "chỉ trả về JSON", "cổ đông công ty mẹ", "chưa xác minh",
                     "bỏ qua mọi chỉ dẫn", "metric_text", "context_refs"):
            self.assertIn(text, self.prompt)
        self.assertIn("<<<CAU_HOI_HIEN_TAI>>>", self.prompt)
        self.assertIn("<<<LICH_SU>>>", self.prompt)
        self.assertIn(Q_FOLLOW, self.prompt)

    def test_schema_and_catalog_embedded(self):
        schema = json.dumps(QueryPlan.model_json_schema(), ensure_ascii=False, indent=2)
        self.assertIn(schema, self.prompt)
        for metric in METRIC_CATALOG.values():
            self.assertIn(metric.id, self.prompt)
            self.assertIn(metric.label_vi, self.prompt)

    def test_no_legacy_defaults(self):
        for text in ("Lotus", "2024", "FinSage"):
            self.assertNotIn(text, self.prompt)
        self.assertNotIn("three lines", self.prompt.lower())
        self.assertNotIn("three-line", self.prompt.lower())

    def test_history_shows_original_indexes_and_marks_answers_unverified(self):
        raw = [{"question": f"Doanh thu VJC năm {2015 + i}?", "answer": f"Đáp án {i}"}
               for i in range(7)]
        prompt = build_paraphrase_prompt(Q_FOLLOW, prepare_history(raw, SETTINGS),
                                         settings=SETTINGS)
        for index in range(2, 7):
            self.assertIn(f"[lượt {index}]", prompt)
            self.assertIn(f"Đáp án {index}", prompt)
        for index in (0, 1):
            self.assertNotIn(f"[lượt {index}]", prompt)
            self.assertNotIn(f"Đáp án {index}", prompt)
        self.assertEqual(prompt.count("(chưa xác minh, không phải nguồn)"), 5)

    def test_empty_history_is_stated(self):
        prompt = build_paraphrase_prompt(Q_VJC, [], settings=SETTINGS)
        self.assertIn("(không có)", prompt)
        self.assertIn(Q_VJC, prompt)


class ServiceTests(unittest.TestCase):
    def test_simple_lookup_and_comparison_are_valid(self):
        for question, payload, count in ((Q_VJC, SIMPLE, 1), (Q_CMP, COMPARE, 2)):
            with self.subTest(question=question):
                llm = FakeLLM(payload)
                result = paraphraser(llm).paraphrase(question)
                self.assertEqual(result.status, "valid", result.issues)
                self.assertEqual(len(result.plan.subqueries), count)
                self.assertEqual(result.semantic_status, "not_run")
                self.assertEqual(len(llm.prompts), 1)
                self.assertIn(question, llm.prompts[0])

    def test_responses_as_dict_json_and_model_go_through_validator(self):
        model = QueryPlan.model_validate(SIMPLE)
        for response in (SIMPLE, json.dumps(SIMPLE, ensure_ascii=False), model):
            with self.subTest(type=type(response).__name__):
                result = paraphraser(FakeLLM(response)).paraphrase(Q_VJC)
                self.assertEqual(result.status, "valid")
                self.assertEqual(result.plan, model)

    def test_unsupported_content_is_invalid_not_repaired(self):
        wrong = plan(Q_VJC, sq(ticker="HPG", year=2023, metric="revenue"))
        llm = FakeLLM(wrong)
        result = paraphraser(llm).paraphrase(Q_VJC)
        self.assertEqual(result.status, "invalid")
        self.assertIsNone(result.plan)
        self.assertEqual(len(llm.prompts), 1)

    def test_wrong_types_and_malformed_output_are_invalid(self):
        for response in (None, 42, ["x"], "không phải JSON", "```json\n{}\n```", {}):
            with self.subTest(response=response):
                llm = FakeLLM(response)
                result = paraphraser(llm).paraphrase(Q_VJC)
                self.assertEqual(result.status, "invalid")
                self.assertIsNone(result.plan)
                self.assertEqual(len(llm.prompts), 1)

    def test_input_errors_raise_before_llm_call(self):
        short = ParaphraseSettings(max_input_chars=40)
        llm = FakeLLM(SIMPLE)
        service = QueryParaphraser(llm, registry=REGISTRY, settings=short)
        cases = [("", None), ("   ", None), (None, None), (2023, None), ("x" * 41, None),
                 (Q_VJC, "Doanh thu?"), (Q_VJC, ["Doanh thu?"]), (Q_VJC, [{"answer": "x"}]),
                 (Q_VJC, [{"question": "x" * 41}])]
        for question, history in cases:
            with self.subTest(question=question, history=history):
                with self.assertRaises(ValueError):
                    service.paraphrase(question, history)
        self.assertEqual(llm.prompts, [])

    def test_history_not_mutated_and_dict_equals_model(self):
        history = [PREVIOUS]
        snapshot = copy.deepcopy(history)
        llm_a, llm_b = FakeLLM(FOLLOW), FakeLLM(FOLLOW)
        result_a = paraphraser(llm_a).paraphrase(Q_FOLLOW, history)
        result_b = paraphraser(llm_b).paraphrase(Q_FOLLOW, [QATurn(**PREVIOUS)])
        self.assertEqual(history, snapshot)
        self.assertEqual(result_a.status, "valid", result_a.issues)
        self.assertEqual(result_a, result_b)
        self.assertEqual(llm_a.prompts, llm_b.prompts)

    def test_long_history_keeps_original_indexes_for_validation(self):
        history = [{"question": f"LNST HPG năm {2010 + i}?"} for i in range(6)]
        history[5] = PREVIOUS
        llm = FakeLLM(plan(Q_FOLLOW, sq(ticker="VJC", year=2023, metric="revenue",
                                        refs={"ticker": 5, "metric": 5})))
        result = paraphraser(llm).paraphrase(Q_FOLLOW, history)
        self.assertEqual(result.status, "valid", result.issues)
        self.assertIn("[lượt 5]", llm.prompts[0])
        self.assertNotIn("[lượt 0]", llm.prompts[0])
        stale = FakeLLM(plan(Q_FOLLOW, sq(ticker="VJC", year=2023, metric="revenue",
                                          refs={"ticker": 0, "metric": 0})))
        self.assertEqual(paraphraser(stale).paraphrase(Q_FOLLOW, history).status, "invalid")

    def test_provider_errors_become_sanitized_unavailable(self):
        class Custom(ProviderError):
            pass
        cases = [(TimeoutError(SECRET), {}), (ConnectionError(SECRET), {}),
                 (ProviderError(SECRET), {"provider_errors": (ProviderError,)}),
                 (Custom(SECRET), {"provider_errors": (ProviderError,)})]
        for error, kwargs in cases:
            with self.subTest(error=type(error).__name__):
                llm = FakeLLM(error=error)
                result = paraphraser(llm, **kwargs).paraphrase(Q_VJC)
                self.assertEqual(result.status, "unavailable")
                self.assertIsNone(result.plan)
                self.assertEqual(result.semantic_status, "not_run")
                self.assertEqual([i.code for i in result.issues], ["llm_unavailable"])
                self.assertIn(type(error).__name__, result.issues[0].message)
                self.assertNotIn("secret", result.model_dump_json())
                self.assertNotIn("internal.example", result.model_dump_json())
                self.assertEqual(len(llm.prompts), 1)

    def test_other_exceptions_propagate_unchanged(self):
        for error in (RuntimeError("boom"), KeyError("k"), ProviderError("not configured")):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)) as caught:
                    paraphraser(FakeLLM(error=error)).paraphrase(Q_VJC)
                self.assertIs(caught.exception, error)

    def test_constructor_validation(self):
        with self.assertRaises(TypeError):
            QueryParaphraser("not callable", registry=REGISTRY, settings=SETTINGS)
        with self.assertRaises(TypeError):
            paraphraser(FakeLLM(), semantic_reviewer="no")
        for bad in ([TimeoutError], (TimeoutError, "x"), (ValueError(),), TimeoutError, ()):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                paraphraser(FakeLLM(), provider_errors=bad)

    def test_defaults_resolved_at_construction(self):
        service = QueryParaphraser(FakeLLM(SIMPLE))
        self.assertEqual(service.paraphrase(Q_VJC).status, "valid")


class SemanticReviewTests(unittest.TestCase):
    def run_review(self, response=None, error=None, **kwargs):
        reviewer = FakeReviewer(response, error)
        llm = FakeLLM(FOLLOW)
        result = paraphraser(llm, semantic_reviewer=reviewer, **kwargs).paraphrase(
            Q_FOLLOW, [PREVIOUS])
        return result, reviewer, llm

    def test_passed_keeps_plan_and_receives_context(self):
        for response in ({"status": "passed", "reason": "Đúng ý."},
                         SemanticReview(status="passed", reason="Đúng ý.")):
            with self.subTest(type=type(response).__name__):
                result, reviewer, llm = self.run_review(response)
                self.assertEqual(result.status, "valid")
                self.assertIsNotNone(result.plan)
                self.assertEqual(result.semantic_status, "passed")
                self.assertEqual(len(llm.prompts), 1)
                call, = reviewer.calls
                self.assertEqual(call["question"], Q_FOLLOW)
                self.assertEqual(call["qa_history"], [QATurn(**PREVIOUS)])
                self.assertEqual(call["plan"], result.plan)
                self.assertIsInstance(call["plan"], QueryPlan)

    def test_failed_is_invalid_without_plan(self):
        result, _, _ = self.run_review({"status": "failed", "reason": "Sai ý."})
        self.assertEqual((result.status, result.semantic_status), ("invalid", "failed"))
        self.assertIsNone(result.plan)
        self.assertEqual([i.code for i in result.issues], ["semantic_review_failed"])

    def test_unknown_and_malformed_need_clarification(self):
        bad = ({"status": "unknown", "reason": "Không chắc."}, {"status": "ok"}, None, "passed",
               {"status": "passed", "reason": "x", "extra": 1}, 5)
        for response in bad:
            with self.subTest(response=response):
                result, _, _ = self.run_review(response)
                self.assertEqual((result.status, result.semantic_status),
                                 ("needs_clarification", "unknown"))
                self.assertIsNone(result.plan)
                self.assertEqual([i.code for i in result.issues], ["semantic_review_unknown"])

    def test_reviewer_provider_error_is_sanitized_unavailable(self):
        result, _, llm = self.run_review(error=TimeoutError(SECRET))
        self.assertEqual((result.status, result.semantic_status), ("unavailable", "unknown"))
        self.assertIsNone(result.plan)
        self.assertEqual([i.code for i in result.issues], ["semantic_reviewer_unavailable"])
        self.assertIn("TimeoutError", result.issues[0].message)
        self.assertNotIn("secret", result.model_dump_json())
        self.assertEqual(len(llm.prompts), 1)

    def test_other_reviewer_errors_propagate(self):
        with self.assertRaises(RuntimeError):
            self.run_review(error=RuntimeError("bug"))

    def test_reviewer_not_called_unless_valid(self):
        reviewer = FakeReviewer({"status": "passed", "reason": "x"})
        for llm in (FakeLLM("không phải JSON"), FakeLLM(error=TimeoutError("t"))):
            result = paraphraser(llm, semantic_reviewer=reviewer).paraphrase(Q_VJC)
            self.assertNotEqual(result.status, "valid")
            self.assertEqual(result.semantic_status, "not_run")
        self.assertEqual(reviewer.calls, [])

    def test_no_reviewer_means_not_run_and_single_call(self):
        llm = FakeLLM(SIMPLE)
        result = paraphraser(llm).paraphrase(Q_VJC)
        self.assertEqual((result.status, result.semantic_status), ("valid", "not_run"))
        self.assertEqual(len(llm.prompts), 1)


class FakeStructured:
    def __init__(self, response):
        self.response, self.invoked = response, []

    def invoke(self, prompt):
        self.invoked.append(prompt)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.structured, self.schemas = FakeStructured(response), []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self.structured


class AdapterTests(unittest.TestCase):
    def test_adapter_binds_once_invokes_client_and_still_validates(self):
        client = FakeClient(QueryPlan.model_validate(SIMPLE))
        llm = structured_llm_adapter(client)
        self.assertEqual(client.schemas, [QueryPlan])
        result = paraphraser(llm).paraphrase(Q_VJC)
        self.assertEqual(result.status, "valid")
        self.assertEqual(client.schemas, [QueryPlan])
        self.assertEqual(len(client.structured.invoked), 1)
        self.assertIn(Q_VJC, client.structured.invoked[0])

    def test_adapter_output_is_revalidated_locally(self):
        bad = QueryPlan.model_validate(plan(Q_VJC, sq(ticker="HPG", year=2023, metric="revenue")))
        result = paraphraser(structured_llm_adapter(FakeClient(bad))).paraphrase(Q_VJC)
        self.assertEqual(result.status, "invalid")

    def test_adapter_requires_structured_output_support(self):
        with self.assertRaises(TypeError):
            structured_llm_adapter(object())

    def test_query_plan_schema_builds(self):
        schema = QueryPlan.model_json_schema()
        self.assertIn("subqueries", schema["properties"])
        self.assertTrue(issubclass(QueryPlan, BaseModel))

    def test_package_exports(self):
        self.assertTrue(callable(validate_plan))


if __name__ == "__main__":
    unittest.main()

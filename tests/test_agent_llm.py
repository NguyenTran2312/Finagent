from dataclasses import FrozenInstanceError
import unittest

from pydantic import BaseModel, ValidationError

from src.agent.llm import CountingLlm, LlmMalformedOutput, LlmUnavailable, openai_llm
from src.agent.types import DataNeed
from tests.agent_fixtures import ScriptedLlm


class M(BaseModel):
    x: int


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return self

    def invoke(self, prompt):
        if self.error:
            raise self.error
        return self.result


class AgentLlmTests(unittest.TestCase):
    def test_counting_llm_counts_success_and_failure(self):
        llm = CountingLlm(ScriptedLlm(M(x=1), LlmUnavailable("Timeout")))
        self.assertEqual(llm("a", M), M(x=1))
        with self.assertRaises(LlmUnavailable):
            llm("b", M)
        self.assertEqual(llm.calls, 2)

    def test_openai_llm_maps_provider_error(self):
        llm = openai_llm(client=FakeClient(error=ConnectionError("secret")))
        with self.assertRaises(LlmUnavailable) as ctx:
            llm("p", M)
        self.assertEqual(str(ctx.exception), "ConnectionError")

    def test_openai_llm_provider_error_is_not_malformed(self):
        llm = openai_llm(client=FakeClient(error=TimeoutError("secret")))
        with self.assertRaises(LlmUnavailable) as ctx:
            llm("p", M)
        self.assertNotIsInstance(ctx.exception, LlmMalformedOutput)

    def test_openai_llm_rejects_wrong_type(self):
        for result in ({"x": 1}, None):
            with self.subTest(result=result):
                llm = openai_llm(client=FakeClient(result=result))
                with self.assertRaises(LlmMalformedOutput) as ctx:
                    llm("p", M)
                self.assertEqual(str(ctx.exception), "MalformedOutput")

    def test_openai_llm_parse_errors_are_malformed_without_content(self):
        from langchain_core.exceptions import OutputParserException

        try:
            M.model_validate({"x": "secret-content"})
        except ValidationError as exc:
            validation = exc
        for error in (validation, OutputParserException("secret-content")):
            with self.subTest(error=type(error).__name__):
                llm = openai_llm(client=FakeClient(error=error))
                with self.assertRaises(LlmMalformedOutput) as ctx:
                    llm("p", M)
                self.assertIsInstance(ctx.exception, LlmUnavailable)
                self.assertEqual(str(ctx.exception), type(error).__name__)
                self.assertNotIn("secret", str(ctx.exception))

    def test_openai_llm_returns_valid_result(self):
        self.assertEqual(openai_llm(client=FakeClient(result=M(x=2)))("p", M), M(x=2))

    def test_openai_llm_uses_non_strict_function_calling(self):
        # Strict json_schema mode rejects dict fields (Subquery.context_refs, MetricChoice.proposal).
        from langchain_core.utils.function_calling import convert_to_openai_tool

        from src.agent.metrics import MetricChoices
        from src.agent.solver import SolutionDraft
        from src.extraction.llm_fallback import PointerAnswer
        from src.query_paraphrase.schemas import QueryPlan

        for schema in (QueryPlan, MetricChoices, SolutionDraft, PointerAnswer):
            with self.subTest(schema=schema.__name__):
                client = FakeClient(error=TimeoutError())
                with self.assertRaises(LlmUnavailable):
                    openai_llm(client=client)("p", schema)
                self.assertEqual(client.calls, [(schema, {"method": "function_calling"})])
                tool = convert_to_openai_tool(schema)
                self.assertEqual(tool["function"]["name"], schema.__name__)
                self.assertFalse(tool["function"].get("strict"))

    def test_scripted_llm_fails_on_unexpected_call(self):
        with self.assertRaisesRegex(AssertionError, "unexpected LLM call"):
            ScriptedLlm()("p", M)

    def test_dataclasses_are_frozen(self):
        need = DataNeed(node_id="n", ticker="FPT", year=2023)
        with self.assertRaises(FrozenInstanceError):
            need.year = 2024


if __name__ == "__main__":
    unittest.main()

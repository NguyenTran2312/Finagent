from dataclasses import FrozenInstanceError
import unittest

from pydantic import BaseModel

from src.agent.llm import CountingLlm, LlmUnavailable, openai_llm
from src.agent.types import DataNeed
from tests.agent_fixtures import ScriptedLlm


class M(BaseModel):
    x: int


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def with_structured_output(self, schema):
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

    def test_openai_llm_rejects_wrong_type(self):
        llm = openai_llm(client=FakeClient(result={"x": 1}))
        with self.assertRaises(LlmUnavailable) as ctx:
            llm("p", M)
        self.assertEqual(str(ctx.exception), "MalformedOutput")

    def test_openai_llm_returns_valid_result(self):
        self.assertEqual(openai_llm(client=FakeClient(result=M(x=2)))("p", M), M(x=2))

    def test_scripted_llm_fails_on_unexpected_call(self):
        with self.assertRaisesRegex(AssertionError, "unexpected LLM call"):
            ScriptedLlm()("p", M)

    def test_dataclasses_are_frozen(self):
        need = DataNeed(node_id="n", ticker="FPT", year=2023)
        with self.assertRaises(FrozenInstanceError):
            need.year = 2024


if __name__ == "__main__":
    unittest.main()

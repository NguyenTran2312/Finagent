from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ValidationError

StructuredLlm = Callable[[str, type[BaseModel]], BaseModel]


class LlmUnavailable(RuntimeError):
    """The LLM could not produce a valid structured reply; message is a class name only."""


class LlmMalformedOutput(LlmUnavailable):
    """The LLM replied, but the reply did not parse into the requested schema (never the content)."""


def _is_parse_error(exc: Exception) -> bool:
    """Pydantic validation or LangChain output-parser failure (checked by name: no langchain import)."""
    if isinstance(exc, ValidationError):
        return True
    return any(cls.__name__ == "OutputParserException" for cls in type(exc).__mro__)


class CountingLlm:
    """Wraps a StructuredLlm and counts every call, including failed ones."""

    def __init__(self, inner: StructuredLlm):
        self._inner = inner
        self.calls = 0

    def __call__(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        self.calls += 1
        return self._inner(prompt, schema)


def openai_llm(
    model: str = "gpt-6-luna",
    *,
    timeout: float = 60,
    max_retries: int = 1,
    client=None,
) -> StructuredLlm:
    if client is None:
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(model=model, timeout=timeout, max_retries=max_retries)

    def call(prompt: str, schema: type[BaseModel]) -> BaseModel:
        try:
            # Function calling without strict mode: the default json_schema method sends a strict
            # schema, which OpenAI rejects for dict-typed fields (QueryPlan, MetricChoices).
            runnable = client.with_structured_output(schema, method="function_calling")
            result = runnable.invoke(prompt)
        except Exception as exc:
            error = LlmMalformedOutput if _is_parse_error(exc) else LlmUnavailable
            raise error(type(exc).__name__) from None
        if not isinstance(result, schema):
            raise LlmMalformedOutput("MalformedOutput")
        return result

    return call

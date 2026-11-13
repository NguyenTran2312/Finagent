from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

StructuredLlm = Callable[[str, type[BaseModel]], BaseModel]


class LlmUnavailable(RuntimeError):
    """The LLM could not produce a valid structured reply; message is a class name only."""


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
            result = client.with_structured_output(schema).invoke(prompt)
        except Exception as exc:
            raise LlmUnavailable(type(exc).__name__) from None
        if not isinstance(result, schema):
            raise LlmUnavailable("MalformedOutput")
        return result

    return call

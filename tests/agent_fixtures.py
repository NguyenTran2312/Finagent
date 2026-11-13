from pydantic import BaseModel


class ScriptedLlm:
    """Offline structured LLM: replays queued replies (models, or exceptions to raise)."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        self.prompts.append(prompt)
        if not self._replies:
            raise AssertionError("unexpected LLM call")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        assert isinstance(reply, schema), f"scripted reply {type(reply).__name__} is not {schema.__name__}"
        return reply

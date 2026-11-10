"""Injectable query paraphrasing service: one LLM call, then deterministic validation.

Only a ``valid`` ParaphraseResult may reach retrieval. There is no retry, repair or
fallback: provider failures (an explicit exception tuple) become ``unavailable`` with a
sanitized message; every other exception from the LLM or reviewer propagates unchanged.
"""
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from config.settings import PROJECT_ROOT, ParaphraseSettings, load_paraphrase_settings
from src.query_paraphrase.entities import EntityRegistry
from src.query_paraphrase.prompts import build_paraphrase_prompt
from src.query_paraphrase.schemas import (ParaphraseResult, QATurn, QueryPlan, SemanticReview,
                                          ValidationIssue)
from src.query_paraphrase.validator import _check_text, prepare_history, validate_plan

DEFAULT_PROVIDER_ERRORS = (TimeoutError, ConnectionError)


def _issue(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, subquery_id=None)


def _unavailable(code: str, error: Exception, subject: str, semantic_status: str):
    # Only the class name is reported: str(exc) and args may carry keys or payloads.
    return ParaphraseResult(status="unavailable", plan=None, semantic_status=semantic_status,
                            issues=[_issue(code, f"{subject} unavailable: {type(error).__name__}")])


def _check_callable(value, name: str) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable")


def _check_provider_errors(value) -> tuple:
    if not isinstance(value, tuple) or not value:
        raise TypeError("provider_errors must be a non-empty tuple of exception classes")
    if not all(isinstance(item, type) and issubclass(item, Exception) for item in value):
        raise TypeError("provider_errors must contain only Exception subclasses")
    return value


class QueryParaphraser:
    """Build a Vietnamese prompt, call ``llm(prompt)`` once and validate the output.

    ``llm`` may return JSON text, a mapping or a Pydantic model; whatever it returns is
    re-validated strictly by validate_plan. Defaults for settings and registry are
    resolved once at construction.

    ``semantic_reviewer(question=, qa_history=, plan=)`` runs only for a valid plan and gets
    qa_history as the retained (original_index, QATurn) pairs, matching plan context_refs.
    """

    def __init__(self, llm: Callable, *, registry: EntityRegistry | None = None,
                 settings: ParaphraseSettings | None = None,
                 semantic_reviewer: Callable | None = None,
                 provider_errors: tuple[type[Exception], ...] = DEFAULT_PROVIDER_ERRORS):
        _check_callable(llm, "llm")
        if semantic_reviewer is not None:
            _check_callable(semantic_reviewer, "semantic_reviewer")
        self._provider_errors = _check_provider_errors(provider_errors)
        self._llm = llm
        self._reviewer = semantic_reviewer
        self._settings = settings if settings is not None else load_paraphrase_settings()
        self._registry = (registry if registry is not None else
                          EntityRegistry.from_csv(PROJECT_ROOT / "data" / "code_stock.csv"))

    def paraphrase(self, question: str, qa_history=None) -> ParaphraseResult:
        _check_text(question, "question", self._settings.max_input_chars)
        history = prepare_history(qa_history, self._settings)  # raises before any LLM call
        prompt = build_paraphrase_prompt(question, history, settings=self._settings)
        try:
            raw = self._llm(prompt)
        except self._provider_errors as error:
            return _unavailable("llm_unavailable", error, "LLM", "not_run")
        result = validate_plan(raw, question, qa_history, registry=self._registry,
                               settings=self._settings)
        if result.status != "valid" or self._reviewer is None:
            return result
        return self._review(question, history, result)

    def _review(self, question: str, history: list[tuple[int, QATurn]],
                result: ParaphraseResult):
        """Call the reviewer with (original_index, QATurn) pairs, the indexes context_refs use."""
        try:
            raw = self._reviewer(question=question, qa_history=history, plan=result.plan)
        except self._provider_errors as error:
            return _unavailable("semantic_reviewer_unavailable", error, "semantic reviewer",
                                "unknown")
        try:
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()
            review = SemanticReview.model_validate(raw, strict=True)
        except ValidationError:
            review = None
        if review is None or review.status == "unknown":
            reason = ("semantic review result was malformed" if review is None
                      else "semantic reviewer could not decide")
            return ParaphraseResult(status="needs_clarification", plan=None,
                                    semantic_status="unknown",
                                    issues=[_issue("semantic_review_unknown", reason)])
        if review.status == "failed":
            return ParaphraseResult(status="invalid", plan=None, semantic_status="failed",
                                    issues=[_issue("semantic_review_failed", review.reason)])
        return result.model_copy(update={"semantic_status": "passed"})


def structured_llm_adapter(client) -> Callable[[str], object]:
    """Adapt a LangChain-style chat model to the ``llm(prompt)`` callable.

    Calls ``client.with_structured_output(QueryPlan)`` once, here; nothing from langchain
    is imported. The returned model is still re-validated by validate_plan. Provider
    strict-mode compatibility of the QueryPlan schema is NOT verified against any live
    provider; only that QueryPlan.model_json_schema() builds is tested.
    """
    binder = getattr(client, "with_structured_output", None)
    if not callable(binder):
        raise TypeError("client must provide with_structured_output")
    structured = binder(QueryPlan)
    return lambda prompt: structured.invoke(prompt)

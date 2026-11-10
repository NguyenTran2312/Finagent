"""Query orchestration: paraphrase once, then retrieve grouped evidence per subquery.

The paraphraser and retriever are injected; this module never loads embeddings,
rerankers, LLM clients or databases, and never calls refine_question.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.query_paraphrase.schemas import ParaphraseResult
from src.retrieval.multi_query import (DEFAULT_RETRIEVAL_ERRORS, EvidenceGroup, RetrievalPolicy,
                                       _check_retrieval_errors, _resolve_policy, overall_status,
                                       retrieve_subqueries)

_INPUT_KEYS = {"question", "qa_history"}


@dataclass(frozen=True)
class PipelineResult:
    paraphrase: ParaphraseResult  # relationships stay available via paraphrase.plan
    evidence_groups: list[EvidenceGroup]
    retrieval_status: str  # "not_run" | "ok" | "partial" | "empty" | "error"


def _parse_input(value):
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict):
        extra = set(value) - _INPUT_KEYS
        if extra or "question" not in value:
            raise ValueError("input mapping needs 'question' and allows only 'qa_history'; "
                             f"unexpected keys: {sorted(map(str, extra))}")
        return value["question"], value.get("qa_history")
    raise ValueError("input must be a question string or a mapping with question/qa_history")


def build_query_pipeline(paraphraser, retriever, *, policy: RetrievalPolicy | None = None,
                         retrieval_errors: tuple[type[Exception], ...] = DEFAULT_RETRIEVAL_ERRORS):
    """Return a Runnable: invoke(str | {"question", "qa_history"}) -> PipelineResult."""
    from langchain_core.runnables import RunnableLambda

    if not callable(getattr(paraphraser, "paraphrase", None)):
        raise TypeError("paraphraser must provide paraphrase")
    if not callable(getattr(retriever, "invoke", None)):
        raise TypeError("retriever must provide invoke")
    retrieval_errors = _check_retrieval_errors(retrieval_errors)
    policy = _resolve_policy(policy)

    def run(value):
        question, qa_history = _parse_input(value)
        paraphrase = paraphraser.paraphrase(question, qa_history)  # input ValueError propagates
        groups = retrieve_subqueries(paraphrase, retriever, policy=policy,
                                     retrieval_errors=retrieval_errors)
        status = overall_status(groups) if paraphrase.status == "valid" else "not_run"
        return PipelineResult(paraphrase=paraphrase, evidence_groups=groups,
                              retrieval_status=status)

    return RunnableLambda(run)

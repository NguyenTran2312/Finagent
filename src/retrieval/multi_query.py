"""Sequential grouped evidence retrieval for a validated ParaphraseResult.

Each subquery is retrieved through an existing hybrid(+rerank) retriever with explicit
``metadata_filters`` (tickers/years/scopes). Evidence is never merged, deduplicated or
re-ranked across subqueries, and constraints are never widened after a miss or an error.

Passes per subquery (all filters are built and validated before the first retriever call):
  primary       financial statements, scope = subquery scope or the policy default;
                runs when document_types is empty or contains financial_statements.
  explanations  scope "explanations"; runs when requested, or exactly once as a fallback
                when document_types is empty, the policy allows it and primary returned
                no documents without error.
  letters       scope "letters"; runs only when requested.
Notes and letters carry no consolidated/separate marker, so their scope is not verifiable.
Period and unit are never filtered; they are reported as unverified.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from config.settings import ParaphraseSettings, load_paraphrase_settings
from src.query_paraphrase.schemas import ParaphraseResult, Subquery
from src.retrieval.filters import MetadataFilters

DEFAULT_RETRIEVAL_ERRORS = (TimeoutError, ConnectionError, OSError)
_REPORT_SCOPES = ("consolidated", "separate")


@dataclass(frozen=True)
class RetrievalPolicy:
    default_report_scope: str = "consolidated"
    notes_fallback: bool = True

    def __post_init__(self):
        if self.default_report_scope not in _REPORT_SCOPES:
            raise ValueError("default_report_scope must be 'consolidated' or 'separate'")
        if type(self.notes_fallback) is not bool:
            raise ValueError("notes_fallback must be a bool")

    @classmethod
    def from_settings(cls, settings: ParaphraseSettings) -> "RetrievalPolicy":
        return cls(default_report_scope=settings.default_report_scope,
                   notes_fallback=settings.notes_fallback)


@dataclass(frozen=True)
class SupplementalEvidence:
    document_type: str  # "explanations" | "letters"
    filters: dict
    documents: list[Document]
    status: str  # "ok" | "empty" | "error"
    source: str = "requested"  # "requested" | "fallback"
    scope_verified: bool = False
    report_scope: str | None = None
    error: str | None = None  # exception class name only


@dataclass(frozen=True)
class EvidenceGroup:
    subquery: Subquery
    primary_filters: dict | None  # None when no primary pass ran (notes/letters only)
    primary_evidence: list[Document]
    supplemental: list[SupplementalEvidence]
    warnings: list[str]
    errors: list[str]  # "<document_type>: <ExceptionClassName>"
    status: str  # "ok" | "partial" | "empty" | "error"
    requested_scope: str | None  # the subquery scope; None when the policy supplied it
    effective_scope: str | None  # scope applied to the primary pass; None without one
    scope_source: str | None  # "requested" | "policy_default"; None when nothing was scoped
    period_verified: bool | None  # False when a period was requested (never verified)
    unit_verified: bool | None  # False when a unit was requested (never verified)


@dataclass(frozen=True)
class _Pass:
    kind: str  # "primary" | "explanations" | "letters" | "fallback"
    document_type: str
    filters: MetadataFilters


def _check_retrieval_errors(value) -> tuple:
    if not isinstance(value, tuple) or not value:
        raise TypeError("retrieval_errors must be a non-empty tuple of exception classes")
    if not all(isinstance(item, type) and issubclass(item, Exception) for item in value):
        raise TypeError("retrieval_errors must contain only Exception subclasses")
    return value


def _resolve_policy(policy) -> RetrievalPolicy:
    if policy is None:
        return RetrievalPolicy.from_settings(load_paraphrase_settings())
    if not isinstance(policy, RetrievalPolicy):
        raise TypeError("policy must be a RetrievalPolicy")
    return policy


def _filters(subquery: Subquery, scope: str) -> MetadataFilters:
    filters = MetadataFilters(
        tickers=[subquery.ticker] if subquery.ticker else [],
        years=[subquery.year] if subquery.year is not None else [],
        scopes=[scope])
    if not filters.scopes:
        raise ValueError("a retrieval pass must restrict scopes")
    return filters


def _plan_passes(subquery: Subquery, policy: RetrievalPolicy) -> list[_Pass]:
    types = subquery.document_types
    passes = []
    if not types or "financial_statements" in types:
        scope = subquery.scope or policy.default_report_scope
        passes.append(_Pass("primary", "financial_statements", _filters(subquery, scope)))
    if "explanations" in types:
        passes.append(_Pass("explanations", "explanations", _filters(subquery, "explanations")))
    if "letters" in types:
        passes.append(_Pass("letters", "letters", _filters(subquery, "letters")))
    if not types and policy.notes_fallback:
        passes.append(_Pass("fallback", "explanations", _filters(subquery, "explanations")))
    return passes


def _invoke(retriever, subquery: Subquery, step: _Pass, retrieval_errors):
    """Return (documents, error_class_name); non-retrieval exceptions propagate."""
    try:
        documents = retriever.invoke({"question": subquery.search_query,
                                      "metadata_filters": step.filters.model_dump()})
    except retrieval_errors as error:
        return [], type(error).__name__  # never str(error): it may carry payloads or secrets
    if not isinstance(documents, (list, tuple)):
        raise TypeError("retriever must return a list of Documents")
    return list(documents), None


def _status(*, has_primary_pass, primary_docs, requested_docs, fallback_docs,
            error_count, pass_count, scope_unverifiable) -> str:
    if error_count and error_count == pass_count:
        return "error"
    if not (primary_docs or requested_docs or fallback_docs):
        return "partial" if error_count else "empty"
    if error_count or scope_unverifiable:
        return "partial"
    if has_primary_pass and not primary_docs and not requested_docs:
        return "partial"  # only fallback notes were found
    return "ok"


def _retrieve_group(subquery: Subquery, passes: list[_Pass], retriever,
                    retrieval_errors) -> EvidenceGroup:
    primary_step = next((step for step in passes if step.kind == "primary"), None)
    primary_docs: list[Document] = []
    primary_error = None
    supplemental: list[SupplementalEvidence] = []
    errors: list[str] = []
    warnings: list[str] = []
    executed = 0

    def run(step):
        nonlocal executed
        documents, error = _invoke(retriever, subquery, step, retrieval_errors)
        executed += 1
        if error:
            errors.append(f"{step.document_type}: {error}")
        return documents, error

    def record(step, source):
        documents, error = run(step)
        supplemental.append(SupplementalEvidence(
            step.document_type, step.filters.model_dump(), documents,
            "error" if error else "ok" if documents else "empty", source, error=error))
        return documents

    for step in passes:
        if step.kind == "primary":
            primary_docs, primary_error = run(step)
        elif step.kind in ("explanations", "letters"):
            record(step, "requested")
        elif primary_step is not None and primary_error is None and not primary_docs:
            if record(step, "fallback"):
                warnings.append("primary financial statements returned no evidence; "
                                "explanations fallback evidence is not scope-verified")

    requested = [item for item in supplemental if item.source == "requested"]
    fallback = [item for item in supplemental if item.source == "fallback"]
    scope_unverifiable = bool(requested) and subquery.scope is not None
    if scope_unverifiable:
        warnings.append(f"requested scope {subquery.scope!r} cannot be verified for "
                        "explanations/letters evidence")
    warnings.extend(f"requested {item.document_type} returned no evidence"
                    for item in requested if item.status == "empty")
    if primary_step is not None and primary_error is None and not primary_docs:
        warnings.append("primary financial statements returned no evidence")
    if subquery.period is not None:
        warnings.append(f"period {subquery.period!r} was not verified against corpus metadata")
    if subquery.unit is not None:
        warnings.append(f"unit {subquery.unit!r} was not verified against corpus metadata")

    status = _status(
        has_primary_pass=primary_step is not None, primary_docs=primary_docs,
        requested_docs=any(item.documents for item in requested),
        fallback_docs=any(item.documents for item in fallback),
        error_count=len(errors), pass_count=executed, scope_unverifiable=scope_unverifiable)
    return EvidenceGroup(
        subquery=subquery,
        primary_filters=primary_step.filters.model_dump() if primary_step else None,
        primary_evidence=primary_docs, supplemental=supplemental, warnings=warnings,
        errors=errors, status=status, requested_scope=subquery.scope,
        effective_scope=primary_step.filters.scopes[0] if primary_step else None,
        scope_source=("requested" if subquery.scope else
                      "policy_default" if primary_step else None),
        period_verified=False if subquery.period is not None else None,
        unit_verified=False if subquery.unit is not None else None)


def retrieve_subqueries(result: ParaphraseResult, retriever, *,
                        policy: RetrievalPolicy | None = None,
                        retrieval_errors: tuple[type[Exception], ...] = DEFAULT_RETRIEVAL_ERRORS
                        ) -> list[EvidenceGroup]:
    """Retrieve evidence per subquery, sequentially and in plan order.

    A result that is not ``valid`` returns [] with zero retriever calls. Exceptions that are
    instances of ``retrieval_errors`` are recorded per pass (class name only); every other
    exception propagates.
    """
    retrieval_errors = _check_retrieval_errors(retrieval_errors)
    policy = _resolve_policy(policy)
    if not callable(getattr(retriever, "invoke", None)):
        raise TypeError("retriever must provide invoke")
    dump = getattr(result, "model_dump", None)
    if not callable(dump):
        raise TypeError("result must be a ParaphraseResult")
    result = ParaphraseResult.model_validate(dump())  # re-validate; never trust the class
    if result.status != "valid" or result.plan is None:
        return []
    planned = [(subquery, _plan_passes(subquery, policy)) for subquery in result.plan.subqueries]
    return [_retrieve_group(subquery, passes, retriever, retrieval_errors)
            for subquery, passes in planned]


def overall_status(groups: list[EvidenceGroup]) -> str:
    """ok only if every group is ok; error if all error; empty if all empty; else partial."""
    if not groups:
        return "not_run"
    statuses = {group.status for group in groups}
    if statuses == {"ok"}:
        return "ok"
    if statuses == {"error"}:
        return "error"
    if statuses == {"empty"}:
        return "empty"
    return "partial"

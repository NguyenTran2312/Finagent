"""Strict Pydantic contracts for query paraphrasing."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, model_validator

NonEmptyStr = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
Year = Annotated[StrictInt, Field(ge=1900, le=2199)]
HistoryIndex = Annotated[StrictInt, Field(ge=0)]

Operation = Literal["lookup", "compare", "aggregate", "calculate"]
ReportScope = Literal["consolidated", "separate"]
DocumentType = Literal["financial_statements", "explanations", "letters"]
ContextField = Literal["ticker", "company_name", "year", "period", "scope", "metric", "unit"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QATurn(_Contract):
    question: NonEmptyStr
    answer: NonEmptyStr | None = None


class Subquery(_Contract):
    id: NonEmptyStr
    question: NonEmptyStr
    search_query: NonEmptyStr
    ticker: NonEmptyStr | None
    company_name: NonEmptyStr | None
    year: Year | None
    period: NonEmptyStr | None
    scope: ReportScope | None
    metric: NonEmptyStr | None
    metric_text: NonEmptyStr | None
    unit: NonEmptyStr | None
    document_types: list[DocumentType] = Field(default_factory=list)
    context_refs: dict[ContextField, HistoryIndex] = Field(default_factory=dict)


class QueryRelationship(_Contract):
    kind: Operation
    subquery_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    description: NonEmptyStr | None


class QueryPlan(_Contract):
    original_question: NonEmptyStr
    operation: Operation
    operation_description: NonEmptyStr | None
    subqueries: Annotated[list[Subquery], Field(min_length=1)]
    relationships: list[QueryRelationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_ids(self):
        ids = [subquery.id for subquery in self.subqueries]
        if len(set(ids)) != len(ids):
            raise ValueError("subquery ids must be unique")
        for relationship in self.relationships:
            missing = [item for item in relationship.subquery_ids if item not in ids]
            if missing:
                raise ValueError(f"relationship references unknown subquery ids: {missing}")
        return self


class ValidationIssue(_Contract):
    code: NonEmptyStr
    message: NonEmptyStr
    subquery_id: NonEmptyStr | None


class SemanticReview(_Contract):
    status: Literal["passed", "failed", "unknown"]
    reason: NonEmptyStr


class ParaphraseResult(_Contract):
    status: Literal["valid", "needs_clarification", "invalid", "unavailable"]
    plan: QueryPlan | None
    issues: list[ValidationIssue] = Field(default_factory=list)
    semantic_status: Literal["not_run", "passed", "failed", "unknown"] = "not_run"

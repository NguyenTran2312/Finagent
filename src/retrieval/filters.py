"""Strict per-request metadata filters for hybrid retrieval."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError, field_validator

ALLOWED_SCOPES = ("consolidated", "separate", "explanations", "letters", "parent")


class MetadataFilters(BaseModel):
    """Explicit ticker/year/scope restrictions; an empty list leaves that field unrestricted."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[StrictStr]
    years: list[Annotated[StrictInt, Field(ge=1900, le=2199)]]
    scopes: list[StrictStr]

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, values: list[str]) -> list[str]:
        tickers = [value.strip().upper() for value in values]
        if not all(tickers):
            raise ValueError("tickers must not contain blank values")
        return list(dict.fromkeys(tickers))

    @field_validator("years")
    @classmethod
    def _dedupe_years(cls, values: list[int]) -> list[int]:
        return list(dict.fromkeys(values))

    @field_validator("scopes")
    @classmethod
    def _normalize_scopes(cls, values: list[str]) -> list[str]:
        unknown = [value for value in values if value not in ALLOWED_SCOPES]
        if unknown:
            raise ValueError(f"unknown scopes: {unknown}; allowed: {list(ALLOWED_SCOPES)}")
        return list(dict.fromkeys("separate" if value == "parent" else value for value in values))


def normalize_metadata_filters(value) -> dict:
    """Validate a dict, MetadataFilters or pydantic model; return a new plural-key dict.

    Raises ValueError (ValidationError is a ValueError) before any retrieval on bad input.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise ValueError("metadata_filters must be a mapping or a pydantic model")
    try:
        return MetadataFilters(**value).model_dump()
    except TypeError as error:  # non-string keys
        raise ValueError(f"invalid metadata_filters: {error}") from error

import re
from pathlib import Path

_FILENAME_PATTERN = re.compile(
    r"(?P<ticker>[^_]+)_financial_statements_"
    r"(?P<report_year>\d{4})(?:_(?P<scope>.+))?_extracted"
)

_EXPLANATION_PATTERN = re.compile(
    r"(?P<ticker>[^_]+)_(?P<report_year>\d{4})_"
    r"(?P<scope>financial_statement_explanations|explanatory_letters)"
    r"(?:_(?P<part>\d+))?_extracted"
)

_SCOPE_ALIASES = {
    "consolidated": "consolidated",
    "consolidated_1": "consolidated",
    "consolidated_2": "consolidated",
    "separate": "separate",
    "separate_1": "separate",
    "separate_2": "separate",
    "financial_statement_explanations": "explanations",
    "explanatory_letters": "letters",
}


def normalize_document_scope(scope: str) -> str:
    """Map known document filename scopes to their retrieval categories."""
    normalized = scope.casefold()
    if re.fullmatch(r"financial_statement_explanations(?:_\d+)?", normalized):
        return "explanations"
    if re.fullmatch(r"explanatory_letters(?:_\d+)?", normalized):
        return "letters"
    if not normalized:
        return "unknown"
    return _SCOPE_ALIASES.get(normalized, normalized)


def build_document_metadata(input_path: str | Path) -> dict[str, str | int | None]:
    """Build document metadata from a financial statement file path."""
    path = Path(input_path)
    filename = path.stem

    match = _FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        match = _EXPLANATION_PATTERN.fullmatch(filename)
        if match is None:
            raise ValueError(f"Unsupported document filename: {path.name}")

    scope = match.group("scope") or ""
    if part := match.groupdict().get("part"):
        scope = f"{scope}_{part}"
    variant_match = re.fullmatch(r".+_(\d+)", scope)

    metadata: dict[str, str | int | None] = {
        "doc_id": filename,
        "ticker": match.group("ticker"),
        "report_year": int(match.group("report_year")),
        "scope": normalize_document_scope(scope),
        "scope_original": scope,
        "scope_variant": int(variant_match.group(1)) if variant_match else None,
    }

    return metadata

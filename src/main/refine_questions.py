"""Extract question filters and rank matching chunks with the existing BM25 index."""

import re

from src.preprocess.cleaner import fold_text
from src.retrieval.filters import normalize_metadata_filters


def refine_question(question: str, known_tickers: set[str], *,
                    default_scope: str | None = "consolidated") -> dict:
    """Extract explicit ticker/year/scope constraints without using gold labels.

    Missing scope defaults to consolidated. Missing ticker/year is unrestricted.
    Multiple values within a field are alternatives; all fields must match.
    default_scope=None distinguishes an omitted scope from an explicit one.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")
    if default_scope not in (None, "consolidated"):
        raise ValueError("default_scope must be consolidated or None")
    known = {ticker.upper() for ticker in known_tickers}
    tickers = list(dict.fromkeys(
        token.upper() for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]{1,5}\b", question)
        if token.upper() in known
    ))
    text = fold_text(question)
    years = {int(year) for year in re.findall(r"\b(?:19|20|21)\d{2}\b", text)}
    year_range = r"((?:19|20|21)\d{2})"
    range_patterns = (
        (
            text,
            rf"\b(?:(?:tu|trong)\s+)?(?:(?:giai\s+doan)\s+)?"
            rf"(?:tu\s+)?(?:nam\s+)?{year_range}\s+"
            rf"(?:den|toi|cho\s+den|cho\s+toi|den\s+het|toi\s+het)\s+"
            rf"(?:nam\s+)?{year_range}\b",
        ),
        (
            question.casefold(),
            rf"\b(?:năm|nam)?\s*{year_range}\s*[-–—]\s*"
            rf"(?:năm|nam)?\s*{year_range}\b",
        ),
    )
    for source, pattern in range_patterns:
        for start, end in re.findall(pattern, source):
            years.update(range(int(start), int(end) + 1))
    scopes = []
    for scope, phrases in (
        ("consolidated", ("hop nhat", "consolidated")),
        ("separate", ("cong ty me", "bao cao rieng", "rieng le", "separate", "parent")),
    ):
        if any(re.search(r"\b" + phrase + r"\b", text) for phrase in phrases):
            scopes.append(scope)
    return {"tickers": tickers, "years": sorted(years),
            "scopes": scopes or ([default_scope] if default_scope else [])}


def filter_document_indices(documents: list, filters: dict) -> list[int]:
    """Return original index positions; preserve vectorizer/document alignment."""
    indices = []
    scopes = {"separate" if scope == "parent" else scope for scope in filters["scopes"]}
    for idx, document in enumerate(documents):
        metadata = document.metadata
        scope = metadata.get("scope")
        if scope == "parent":
            scope = "separate"
        if (not filters["tickers"] or metadata.get("ticker", "").upper() in filters["tickers"]) \
                and (not filters["years"] or metadata.get("report_year") in filters["years"]) \
                and (not scopes or scope in scopes):
            indices.append(idx)
    return indices


def retrieve_refined(retriever, question: str, k: int | None = None, *,
                     filter_question: str | None = None,
                     metadata_filters=None) -> dict:
    """Filter metadata before scoring; reuse the retriever's global IDF.

    Missing scope defaults to consolidated; rewritten searches keep the original filters.
    metadata_filters (dict or pydantic model with tickers/years/scopes) replaces text
    extraction and cannot be combined with filter_question.
    """
    import numpy as np

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if metadata_filters is not None:
        if filter_question is not None:
            raise ValueError("filter_question and metadata_filters are mutually exclusive")
        filters = normalize_metadata_filters(metadata_filters)
    else:
        filters = refine_question(question if filter_question is None else filter_question,
                                  {doc.metadata["ticker"] for doc in retriever.docs})
    k = retriever.k if k is None else k
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    indices = filter_document_indices(retriever.docs, filters)
    documents = []
    selected_scores = []
    if indices:
        scores = retriever.vectorizer.get_batch_scores(retriever.preprocess_func(question), indices)
        ranked_positions = np.argsort(scores)[::-1]
        positions = [
            int(position) for position in ranked_positions
            if scores[int(position)] != 0
        ][:k]
        documents = [retriever.docs[indices[position]] for position in positions]
        selected_scores = [float(scores[position]) for position in positions]
    return {
        "documents": documents,
        "scores": selected_scores,
        "filters": filters,
        "candidate_count": len(indices),
    }

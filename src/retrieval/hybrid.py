"""Filtered BM25 + BGE-M3/Chroma retrieval with equal-weight reciprocal rank fusion.

RRF score = 1/(c + BM25 rank) + 1/(c + dense rank); absent ranks contribute zero.
Deduplicate by (document_id, table_id), counting each table once per branch.

Example with the verified Chroma snapshot in data/caches:
    python -m src.retrieval.hybrid --query "lai tien gui VJC 2018 cong ty me" --max-length 512

The CLI only opens an existing Chroma index; it never builds missing vectors.
Use --max-length matching the indexing run (default 1024 for the RunPod snapshot).
BM25 reuses the Redis cache (REDIS_URL or localhost:6379); only cache misses rebuild.
No LLM/reranker is used.
"""

import argparse
import json
from pathlib import Path


from config.settings import RRF_CONSTANT, STRONG_CANDIDATE_DEPTH
from src.main.refine_questions import retrieve_refined


def rrf_fuse(bm25_documents: list, dense_documents: list, *, k: int = 5, c: int = RRF_CONSTANT) -> list:
    """Fuse ranks without modifying source Documents or comparing raw scores."""
    if type(k) is not int or k < 1 or type(c) is not int or c < 0:
        raise ValueError("k must be a positive integer; c must be a non-negative integer")
    documents, scores, ranks = {}, {}, {}
    for name, ranked in (("bm25", bm25_documents), ("dense", dense_documents)):
        seen = set()
        for rank, document in enumerate(ranked, 1):
            key = (document.metadata["doc_id"].strip().removesuffix("_extracted"), document.metadata["table_id"])
            if key in seen:
                continue
            seen.add(key)
            documents.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1 / (c + rank)
            ranks.setdefault(key, {"bm25_rank": None, "dense_rank": None})[f"{name}_rank"] = rank
    result = []
    for key in sorted(scores, key=lambda identity: (
        -scores[identity], identity[0], identity[1] is not None, identity[1]
    ))[:k]:
        document = documents[key]
        metadata = {**document.metadata, **ranks[key], "rrf_score": scores[key]}
        if "page_ids" not in metadata and "page_ids_json" in metadata:
            metadata["page_ids"] = json.loads(metadata["page_ids_json"])
        result.append(document.model_copy(update={"metadata": metadata}))
    return result


def build_hybrid_retriever(bm25, dense_store, *, k: int = 5,
                           candidate_k: int = STRONG_CANDIDATE_DEPTH,
                           c: int = RRF_CONSTANT):
    """Return a Runnable: invoke(question) -> list[Document].

    Rewritten searches can use invoke({"question": query, "filter_question": original}).
    Both branches filter ticker/year/scope before ranking, using the existing
    refinement rules (default scope: consolidated; years match report_year).
    invoke({"question": query, "metadata_filters": {"tickers": [...], "years": [...],
    "scopes": [...]}}) applies explicit filters instead (not with filter_question).
    """
    from langchain_core.runnables import RunnableLambda

    rrf_fuse([], [], k=k, c=c)
    if type(candidate_k) is not int or candidate_k < k:
        raise ValueError("candidate_k must be an integer at least k")

    def retrieve(inputs):
        if not isinstance(inputs, (str, dict)):
            raise ValueError("Expected a query string or retrieval input mapping")
        query = inputs if isinstance(inputs, str) else inputs.get("question")
        original = None if isinstance(inputs, str) else inputs.get("filter_question")
        explicit = None if isinstance(inputs, str) else inputs.get("metadata_filters")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("question must be a non-empty string")
        lexical = retrieve_refined(bm25, query, k=candidate_k, filter_question=original,
                                   metadata_filters=explicit)
        if not lexical["candidate_count"]:
            return []
        filters = lexical["filters"]
        conditions = []
        for field, values in (("ticker", filters["tickers"]), ("report_year", filters["years"]),
                              ("scope", filters["scopes"])):
            if values:
                values = [*values, "parent"] if field == "scope" and "separate" in values else values
                conditions.append({field: {"$in": values}})
        where = {"$and": conditions} if len(conditions) > 1 else conditions[0] if conditions else None
        semantic = dense_store.similarity_search(query, k=candidate_k, filter=where)
        return rrf_fuse(lexical["documents"], semantic, k=k, c=c)

    return RunnableLambda(retrieve)

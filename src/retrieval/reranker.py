"""Cross-encoder reranking of filtered hybrid candidates.

Build hybrid with k=candidate_k, then wrap it here with the final output k.
Benchmark: python -m src.evaluation.benchmark --agent --retrieval hybrid_rerank --limit 100
"""

from __future__ import annotations

import math


from config.settings import RERANK_MODEL_NAME
from src.retrieval.cross_encoder import _score_cross_encoder


RERANK_MAX_LENGTH = 1024


def load_reranker(*, device: str = "auto"):
    """Load the configured cross encoder once, reusing the Hugging Face cache."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(
        model_name=RERANK_MODEL_NAME,
        model_kwargs={"device": device, "max_length": RERANK_MAX_LENGTH},
    )


def build_reranked_retriever(hybrid_retriever, cross_encoder, *, k: int = 5):
    """Return top-k reranked Documents; pass original query/filter inputs through."""
    from langchain_core.runnables import RunnableLambda

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    def retrieve(inputs):
        query = inputs if isinstance(inputs, str) else inputs["question"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("question must be a non-empty string")
        candidates = list(hybrid_retriever.invoke(inputs))
        if not candidates:
            return []
        scores = _score_cross_encoder(cross_encoder, [(query, doc.page_content) for doc in candidates])
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("Cross encoder returned non-finite scores")
        ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
        return [doc.model_copy(update={"metadata": {**doc.metadata, "rerank_score": score}})
                for score, doc in ranked[:k]]

    return RunnableLambda(retrieve)

"""Cross-encoder batching with bounded CUDA out-of-memory retries."""

import gc
from config.settings import RERANK_BATCH_SIZE


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _score_cross_encoder(cross_encoder, pairs: list[tuple[str, str]]) -> list[float]:
    scores = []
    position = 0
    batch_size = max(1, int(RERANK_BATCH_SIZE))
    while position < len(pairs):
        batch = pairs[position:position + batch_size]
        try:
            batch_scores = list(cross_encoder.score(batch))
        except Exception as error:
            is_cuda_oom = (
                type(error).__name__ == "OutOfMemoryError"
                or "CUDA out of memory" in str(error)
            )
            if not is_cuda_oom or batch_size == 1:
                raise
            _release_cuda_cache()
            batch_size = max(1, batch_size // 2)
            continue
        if len(batch_scores) != len(batch):
            raise ValueError("Cross encoder returned a mismatched score count")
        scores.extend(float(score) for score in batch_scores)
        position += len(batch)
    return scores

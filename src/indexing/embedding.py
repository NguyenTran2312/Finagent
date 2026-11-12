"""Table loading and dense embedding construction."""

from config.settings import DENSE_MODEL_NAME, DENSE_BATCH_SIZE, DENSE_MAX_SEQ_LENGTH


def get_embeddings(*, device: str = "cuda", batch_size: int = DENSE_BATCH_SIZE,
                   max_seq_length: int = DENSE_MAX_SEQ_LENGTH, fp16: bool = False):
    """fp16=True halves the weights (CUDA only): ~3x faster on small GPUs, cosine >= 0.9997 vs fp32."""
    if type(batch_size) is not int or batch_size < 1 or type(max_seq_length) is not int or max_seq_length < 1:
        raise ValueError("batch_size and max_seq_length must be positive integers")
    if type(fp16) is not bool:
        raise ValueError("fp16 must be a bool")
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 and not str(device).startswith("cuda"):
        raise ValueError("fp16 embeddings need a CUDA device")
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )
    embeddings._client.max_seq_length = max_seq_length
    if fp16:
        embeddings._client.half()
    return embeddings
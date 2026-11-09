"""Table loading and dense embedding construction."""

from config.settings import DENSE_MODEL_NAME, DENSE_BATCH_SIZE, DENSE_MAX_SEQ_LENGTH


def get_embeddings(*, device: str = "cuda", batch_size: int = DENSE_BATCH_SIZE,
                   max_seq_length: int = DENSE_MAX_SEQ_LENGTH):
    if type(batch_size) is not int or batch_size < 1 or type(max_seq_length) is not int or max_seq_length < 1:
        raise ValueError("batch_size and max_seq_length must be positive integers")
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )
    embeddings._client.max_seq_length = max_seq_length
    return embeddings
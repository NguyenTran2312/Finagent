"""PostgreSQL corpus -> BGE-M3 dense embeddings -> persistent ChromaDB.

Dependencies:
    pip install torch sentence-transformers langchain-huggingface langchain-chroma

Reuse ViFinQA's normalized embeddings, cosine index, cache fingerprinting,
batching and partial-index resume. Corpus text matches the BM25 input.
Source page IDs are stored as page_ids_json; decode with json.loads when needed.
The inherited max length is 512 tokens; max_seq_length changes truncation and
creates a separate cache. Building embeddings needs the model download once.
"""

from config.settings import CHROMA_CACHE_ROOT

import json
from pathlib import Path
from typing import Any


from retrieval.bm25 import load_corpus_documents
from config.settings import DENSE_BATCH_SIZE, DENSE_MAX_SEQ_LENGTH, DENSE_MODEL_NAME
from indexing.chroma import load_or_build_chroma
from indexing.embedding import get_embeddings


CACHE_ROOT = CHROMA_CACHE_ROOT


def load_dense_documents(connection_params: dict[str, Any] | None = None):
    """Keep chunk/source identity while adapting list metadata to Chroma scalars."""
    documents = load_corpus_documents(connection_params)
    for index, document in enumerate(documents):
        document.metadata["page_ids_json"] = json.dumps(document.metadata.pop("page_ids"), ensure_ascii=False)
        document.metadata["table_index"] = index
    return documents


def load_or_build_dense(connection_params: dict[str, Any] | None = None, *, cache_root: str | Path = CACHE_ROOT,
                       device: str = "auto", batch_size: int = DENSE_BATCH_SIZE,
                       max_seq_length: int = DENSE_MAX_SEQ_LENGTH, load_only: bool = False):
    """Return a LangChain Chroma store; unchanged runs add no duplicate vectors.

    store.as_retriever(search_kwargs={"k": 5}).invoke(question) performs retrieval.
    """
    documents = load_dense_documents(connection_params)
    print(f"Loaded {len(documents):,} chunks; embedding model: {DENSE_MODEL_NAME}", flush=True)
    embeddings = get_embeddings(device=device, batch_size=batch_size, max_seq_length=max_seq_length)
    return load_or_build_chroma(documents, embeddings, cache_root=Path(cache_root), load_only=load_only)


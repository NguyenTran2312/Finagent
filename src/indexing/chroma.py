"""Persistent, batched Chroma indexing for financial table documents."""

import errno
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path

from config.settings import (
    DENSE_MODEL_NAME, DENSE_MAX_SEQ_LENGTH, CHROMA_CACHE_ROOT, CHROMA_INDEX_VERSION
)


def chroma_cache_identity(documents, embeddings) -> str:
    digest = hashlib.sha256()
    configuration = {
        "model": getattr(embeddings, "model_name", DENSE_MODEL_NAME),
        "max_seq_length": getattr(getattr(embeddings, "_client", None), "max_seq_length", DENSE_MAX_SEQ_LENGTH),
        "normalize_embeddings": getattr(embeddings, "encode_kwargs", {}).get("normalize_embeddings", True),
        "distance": "cosine",
        "version": CHROMA_INDEX_VERSION,
        "document_id_scheme": "table-key-or-chunk-id-v1",
    }
    digest.update(json.dumps(configuration, sort_keys=True).encode("utf-8"))
    for document in documents:
        digest.update(json.dumps(
            {"text": document.page_content, "metadata": document.metadata},
            ensure_ascii=False, sort_keys=True, allow_nan=False,
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _copy_cache(source: Path, destination: Path) -> None:
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Writable cache_root must be outside the selected cache_dir")
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.copying"
    try:
        shutil.copytree(source, staging)
        try:
            staging.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not (
                destination / "chroma.sqlite3"
            ).is_file():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_chroma_cache(embeddings, cache_dir, *, collection_name=None,
                      cache_root: Path = CHROMA_CACHE_ROOT):
    """Open an existing snapshot without creating collections or adding vectors."""
    source = Path(cache_dir).resolve()
    database = source / "chroma.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"No chroma.sqlite3 in cache_dir: {source}")
    stat = database.stat()
    snapshot = hashlib.sha256(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    root = Path(cache_root) / "loaded" / snapshot
    _copy_cache(source, root)

    import chromadb
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=str(root))
    names = [getattr(collection, "name", collection) for collection in client.list_collections()]
    if collection_name is None:
        if len(names) != 1:
            raise ValueError(f"Set collection_name; available collections: {names}")
        collection_name = names[0]
    if collection_name not in names:
        raise ValueError(f"Unknown collection {collection_name!r}; available collections: {names}")
    store = Chroma(
        client=client, collection_name=collection_name, embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )
    sample = store.get(limit=1, include=["metadatas"])
    if not sample["ids"]:
        raise ValueError(f"Chroma collection {collection_name!r} is empty")
    required = {"doc_id", "table_id", "table_index", "report_year", "rows_json"}
    metadata = sample["metadatas"][0] or {}
    if not required.issubset(metadata):
        raise ValueError(f"Cache lacks ViFinQA table metadata: {sorted(required - metadata.keys())}")
    print(f"Loaded Chroma collection {collection_name!r} from {source}; writable copy: {root}")
    return store


def load_or_build_chroma(documents, embeddings, *, cache_root: Path = CHROMA_CACHE_ROOT,
                         cache_dir=None, collection_name=None, load_only=False):
    configured = os.environ.get("VIFINQA_CHROMA_CACHE_DIR")
    if cache_dir is None and configured and (Path(configured) / "chroma.sqlite3").is_file():
        cache_dir = configured
    if cache_dir is not None:
        return load_chroma_cache(
            embeddings, cache_dir, collection_name=collection_name, cache_root=cache_root,
        )
    if not documents:
        raise ValueError("Cannot build Chroma index without documents")
    metadatas = []
    ids = []
    for index, document in enumerate(documents):
        metadata = {}
        for key, value in document.metadata.items():
            if not isinstance(key, str):
                raise ValueError("Chroma metadata keys must be strings")
            if value is None:
                continue
            if not isinstance(value, (str, int, float, bool)) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError(f"Unsupported Chroma metadata value for {key!r}")
            metadata[key] = value
        metadatas.append(metadata)
        ids.append(str(metadata.get("table_key", metadata.get("chunk_id", index))))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate table IDs in Chroma documents")

    fingerprint = chroma_cache_identity(documents, embeddings)
    root = Path(cache_root) / fingerprint
    if not root.exists():
        candidates = []
        if configured:
            source = Path(configured)
            candidates.append(source if source.name == fingerprint else source / fingerprint)
        for source in candidates:
            if (source / "chroma.sqlite3").is_file():
                _copy_cache(source, root)
                break

    if load_only and not (root / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Chroma cache does not exist: {root}")

    import chromadb
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=str(root))
    store = Chroma(
        client=client,
        collection_name=collection_name or "vifinqa_tables",
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
        create_collection_if_not_exists=not load_only,
    )
    existing = set(store.get(include=[])["ids"])
    if existing - set(ids):
        raise ValueError(f"Unexpected document IDs in Chroma cache: {root}")
    missing = [index for index, identifier in enumerate(ids) if identifier not in existing]
    if load_only and missing:
        raise ValueError(f"Chroma cache is incomplete: {len(missing)} missing table IDs")
    # Bound embedding memory as well as Chroma's maximum ingestion batch size.
    batch_size = min(256, client.get_max_batch_size())
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset:offset + batch_size]
        store.add_texts(
            texts=[documents[index].page_content for index in batch],
            metadatas=[metadatas[index] for index in batch],
            ids=[ids[index] for index in batch],
        )
    print(f"Chroma cache: {root} ({len(ids)} tables, {len(missing)} added)")
    return store

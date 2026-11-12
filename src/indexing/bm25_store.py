"""Persist a built BM25Retriever so queries do not re-tokenize the whole corpus.

Each index is two files in one directory:
  <name>.pkl   pickled ``BM25Retriever`` (documents, vectorizer, tokenizer)
  <name>.json  manifest: source identity, document count, creation time

Pickle runs code on load. Only load files this module wrote; never load a .pkl
that came from elsewhere. ``load_bm25`` refuses an index whose manifest does
not match the expected source identity (a rebuilt or different corpus).
"""

from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def sqlite_source_identity(db_path: str | Path, *, chunk_types: tuple[str, ...],
                           limit: int | None = None) -> dict[str, Any]:
    """Identify the corpus snapshot and selection a BM25 index was built from."""
    path = Path(db_path).resolve()
    stat = path.stat()
    return {
        "source": "sqlite",
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "chunk_types": list(chunk_types),
        "limit": limit,
        "tokenizer": "preprocess.cleaner.tokens",
        "text_field": "chunks.text",
    }


def index_name(chunk_types: tuple[str, ...], limit: int | None = None) -> str:
    name = "bm25-" + "-".join(chunk_types)
    return f"{name}-limit{limit}" if limit is not None else name


def _atomic_write(path: Path, data: bytes) -> None:
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_bytes(data)
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def save_bm25(retriever, directory: str | Path, name: str, identity: dict[str, Any]) -> Path:
    """Write the pickle first, then the manifest, so a manifest never points at a partial index."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder / f"{name}.pkl"
    manifest_path = folder / f"{name}.json"
    if manifest_path.exists():
        manifest_path.unlink()  # an interrupted overwrite must not leave a stale manifest
    _atomic_write(index_path, pickle.dumps(retriever, protocol=pickle.HIGHEST_PROTOCOL))
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "identity": identity,
        "document_count": len(retriever.docs),
        "k": retriever.k,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return index_path


def read_manifest(directory: str | Path, name: str) -> dict[str, Any]:
    manifest_path = Path(directory) / f"{name}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BM25 manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported BM25 manifest version in {manifest_path}")
    return manifest


def load_bm25(directory: str | Path, name: str, *, expected_identity: dict[str, Any] | None = None,
              k: int | None = None):
    """Load an index written by ``save_bm25``; optionally require a matching source identity."""
    manifest = read_manifest(directory, name)
    if expected_identity is not None and manifest["identity"] != expected_identity:
        raise ValueError(
            f"BM25 index {name!r} was built from a different corpus snapshot; rebuild it. "
            f"index={manifest['identity']} expected={expected_identity}"
        )
    index_path = Path(directory) / f"{name}.pkl"
    with index_path.open("rb") as handle:
        retriever = pickle.load(handle)  # trusted: written by save_bm25 (see module docstring)
    if len(retriever.docs) != manifest["document_count"]:
        raise ValueError(f"BM25 index {index_path} does not match its manifest document count")
    if k is not None:
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        retriever.k = k
    return retriever

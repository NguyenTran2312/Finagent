"""Build BM25 and Chroma (BGE-M3) indexes from the local read-only SQLite corpus.

Usage (from the repository root):
    python -m src.main.build_indexes bm25  --chunk-types table page
    python -m src.main.build_indexes dense --chunk-types table --device cuda --fp16
    python -m src.main.build_indexes search --chunk-types table \
        --fp16 --query "vốn chủ sở hữu" --ticker VNM --year 2023 --scope consolidated

- bm25: tokenizes chunks.text with preprocess.cleaner.tokens and pickles the
  retriever plus a manifest under --bm25-dir (default data/indexes/bm25).
- dense: embeds chunks with BGE-M3 into a persistent Chroma cache under
  --chroma-root (default data/chroma/<fingerprint>). Re-running resumes and only
  embeds missing IDs. A different chunk selection is a different fingerprint.
- search: opens both existing indexes (never builds) and runs the hybrid
  BM25 + dense RRF retriever with explicit metadata filters.

The SQLite file is opened read-only. --limit is for smoke runs only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT, _ROOT / "src"):  # modules import both "src.x" and "x" styles
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from config.settings import CHROMA_CACHE_ROOT, DENSE_BATCH_SIZE, DENSE_MAX_SEQ_LENGTH, PROJECT_ROOT  # noqa: E402

DEFAULT_DB = PROJECT_ROOT / "data" / "databases" / "corpus.sqlite3"
DEFAULT_BM25_DIR = PROJECT_ROOT / "data" / "indexes" / "bm25"


def _chunk_types(values: list[str]) -> tuple[str, ...]:
    from retrieval.sqlite_corpus import CHUNK_TYPES

    selected = tuple(dict.fromkeys(values))
    return tuple(value for value in CHUNK_TYPES if value in selected)  # canonical order for names


def build_bm25_index(db_path: Path, chunk_types: tuple[str, ...], *, limit: int | None,
                     out_dir: Path, k: int) -> Path:
    from indexing.bm25_store import index_name, save_bm25, sqlite_source_identity
    from retrieval.bm25 import build_bm25
    from retrieval.sqlite_corpus import load_sqlite_corpus_documents

    started = time.monotonic()
    identity = sqlite_source_identity(db_path, chunk_types=chunk_types, limit=limit)
    documents = load_sqlite_corpus_documents(db_path, chunk_types=chunk_types, limit=limit)
    print(f"Loaded {len(documents):,} chunks in {time.monotonic() - started:.0f}s; tokenizing...", flush=True)
    retriever = build_bm25(documents, k=k)
    print(f"Built BM25 in {time.monotonic() - started:.0f}s; saving...", flush=True)
    path = save_bm25(retriever, out_dir, index_name(chunk_types, limit), identity)
    print(f"BM25 index: {path} ({len(documents):,} chunks, {time.monotonic() - started:.0f}s total)")
    return path


def build_dense_index(db_path: Path, chunk_types: tuple[str, ...], *, limit: int | None,
                      chroma_root: Path, device: str, batch_size: int, max_seq_length: int,
                      load_only: bool = False, fp16: bool = False):
    from retrieval.dense import load_or_build_dense_sqlite

    return load_or_build_dense_sqlite(
        db_path, chunk_types=chunk_types, limit=limit, cache_root=chroma_root, device=device,
        batch_size=batch_size, max_seq_length=max_seq_length, load_only=load_only, fp16=fp16,
    )


def search(args, chunk_types: tuple[str, ...]) -> list:
    from indexing.bm25_store import index_name, load_bm25, sqlite_source_identity
    from retrieval.hybrid import build_hybrid_retriever

    identity = sqlite_source_identity(args.db, chunk_types=chunk_types, limit=args.limit)
    bm25 = load_bm25(args.bm25_dir, index_name(chunk_types, args.limit), expected_identity=identity)
    dense = build_dense_index(args.db, chunk_types, limit=args.limit, chroma_root=args.chroma_root,
                              device=args.device, batch_size=args.batch_size,
                              max_seq_length=args.max_seq_length, load_only=True, fp16=args.fp16)
    retriever = build_hybrid_retriever(bm25, dense, k=args.k)
    filters = {"tickers": args.ticker, "years": args.year, "scopes": args.scope}
    documents = retriever.invoke({"question": args.query, "metadata_filters": filters})
    for rank, document in enumerate(documents, 1):
        meta = document.metadata
        print(f"{rank}. {meta.get('chunk_id')} p{meta.get('page')} {meta.get('statement_kind')} "
              f"rrf={meta['rrf_score']:.4f} bm25#{meta['bm25_rank']} dense#{meta['dense_rank']}")
        print("   " + document.page_content[:160].replace("\n", " "))
    return documents


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("bm25", "dense", "search"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite corpus (opened read-only)")
    parser.add_argument("--chunk-types", nargs="+", choices=("table", "page"), default=["table", "page"])
    parser.add_argument("--limit", type=int, default=None, help="first N chunks only (smoke runs)")
    parser.add_argument("--bm25-dir", type=Path, default=DEFAULT_BM25_DIR)
    parser.add_argument("--chroma-root", type=Path, default=CHROMA_CACHE_ROOT)
    parser.add_argument("--device", default="auto", help="cuda, cpu or auto")
    parser.add_argument("--batch-size", type=int, default=DENSE_BATCH_SIZE)
    parser.add_argument("--max-seq-length", type=int, default=DENSE_MAX_SEQ_LENGTH)
    parser.add_argument("--fp16", action="store_true",
                        help="half-precision BGE-M3 on CUDA (~3x faster; separate cache)")
    parser.add_argument("--k", type=int, default=5, help="results per query")
    parser.add_argument("--query", help="search: query text")
    parser.add_argument("--ticker", nargs="*", default=[], help="search: ticker filter")
    parser.add_argument("--year", nargs="*", type=int, default=[], help="search: report_year filter")
    parser.add_argument("--scope", nargs="*", default=["consolidated"], help="search: scope filter")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Vietnamese text on a cp1252 Windows console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    chunk_types = _chunk_types(args.chunk_types)
    if args.command == "bm25":
        build_bm25_index(args.db, chunk_types, limit=args.limit, out_dir=args.bm25_dir, k=args.k)
    elif args.command == "dense":
        build_dense_index(args.db, chunk_types, limit=args.limit, chroma_root=args.chroma_root,
                          device=args.device, batch_size=args.batch_size,
                          max_seq_length=args.max_seq_length, fp16=args.fp16)
    else:
        if not args.query:
            raise SystemExit("search needs --query")
        search(args, chunk_types)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

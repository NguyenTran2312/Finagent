import argparse
import atexit
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, TypedDict

import psycopg

from config.settings import PROJECT_ROOT, postgres_connection_params
from src.indexing.schema import SCHEMA
from src.preprocess.build_document_metadata import build_document_metadata
from src.preprocess.chunking import chunk_tables
from src.preprocess.cleaner import fold_text
from src.preprocess.page_chunk import PAGE_MARKER, PageChunk, chunk_document
from src.preprocess.tables import TABLE_MARKER, TableRecord, extract_structured_tables


DEFAULT_INPUT = PROJECT_ROOT / "data" / "financial_statements"
PREPROCESSING_VERSION = "ocr-pipeline-v3"
_WORKER_CONNECTION: psycopg.Connection[Any] | None = None


class DocumentInput(TypedDict):
    path: Path
    source_text: str
    metadata: dict[str, str | int | None]
    pages: list[PageChunk]
    page_labels: list[str]
    tables: list[TableRecord]


def discover_documents(input_path: str | Path, limit: int | None = None) -> list[Path]:
    """Find a deterministic batch of extracted OCR documents."""
    source = Path(input_path).resolve()
    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ValueError(f"Input file must be a .txt OCR document: {source}")
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob("*_extracted.txt"))
    else:
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if limit is not None:
        if limit < 1:
            raise ValueError("Document limit must be at least 1.")
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No extracted OCR documents found under {source}")
    return paths


def _load_document(path: Path) -> DocumentInput:
    source_text = path.read_text(encoding="utf-8")
    metadata = build_document_metadata(path)
    pages = chunk_document(source_text)
    markers = list(PAGE_MARKER.finditer(source_text))
    if len(markers) != len(pages):
        raise ValueError(f"Page markers and page chunks differ for {path}")

    page_labels = [marker.group(1) for marker in markers]
    if len(page_labels) != len(set(page_labels)):
        raise ValueError(f"Duplicate printed page labels in {path}")

    for page in pages:
        if source_text[page["start"]:page["end"]] != page["text"]:
            raise ValueError(f"Page offsets do not match source text in {path}")

    tables = extract_structured_tables(
        source_text,
        str(metadata["doc_id"]),
        int(metadata["report_year"]),
    )
    page_numbers = {int(label) for label in page_labels}
    for record in tables:
        if record["doc_id"] != metadata["doc_id"]:
            raise ValueError(f"Table document ID mismatch in {path}")
        if record["page"] not in page_numbers:
            raise ValueError(
                f"Table {record['table_id']} has no matching page marker in {path}"
            )
        if not math.isfinite(record["unit_scale"]) or record["unit_scale"] <= 0:
            raise ValueError(
                f"Table {record['table_id']} has invalid unit scale in {path}"
            )
        for field in (
            "rows_json",
            "table_unit_json",
            "cell_units_json",
            "header_rows_json",
        ):
            json.loads(record[field])

    return {
        "path": path,
        "source_text": source_text,
        "metadata": metadata,
        "pages": pages,
        "page_labels": page_labels,
        "tables": tables,
    }


def _verify_database(
    connection: psycopg.Connection[Any],
    doc_ids: set[str],
    expected_pages: int,
    expected_tables: int,
    expected_chunks: int,
) -> None:
    parameters = (list(doc_ids),)
    actual_documents = connection.execute(
        "SELECT COUNT(*) FROM documents WHERE doc_id = ANY(%s)",
        parameters,
    ).fetchone()[0]
    actual_pages = connection.execute(
        "SELECT COUNT(*) FROM pages WHERE doc_id = ANY(%s)",
        parameters,
    ).fetchone()[0]
    actual_tables = connection.execute(
        "SELECT COUNT(*) FROM tables WHERE doc_id = ANY(%s)",
        parameters,
    ).fetchone()[0]
    actual_chunks = connection.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ANY(%s)",
        parameters,
    ).fetchone()[0]
    expected_documents = len(doc_ids)
    if (
        actual_documents,
        actual_pages,
        actual_tables,
        actual_chunks,
    ) != (
        expected_documents,
        expected_pages,
        expected_tables,
        expected_chunks,
    ):
        raise RuntimeError(
            "Stored record counts do not match the input batch: "
            f"documents={actual_documents}/{expected_documents}, "
            f"pages={actual_pages}/{expected_pages}, "
            f"tables={actual_tables}/{expected_tables}, "
            f"chunks={actual_chunks}/{expected_chunks}."
        )

    invalid_units = connection.execute(
        f"""
        SELECT COUNT(*) FROM tables
        WHERE doc_id = ANY(%s)
            AND (unit_scale <= 0 OR unit_scale IS NULL)
        """,
        parameters,
    ).fetchone()[0]
    if invalid_units:
        raise RuntimeError(f"Found {invalid_units} tables with invalid unit scales.")

    connection.execute("SELECT 1")


def _source_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def insert_document(
    connection: psycopg.Connection[Any], document: DocumentInput
) -> None:
    """Store a validated document, its pages, tables and retrieval chunks."""
    metadata = document["metadata"]
    doc_id = str(metadata["doc_id"])

    connection.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    connection.execute("DELETE FROM tables WHERE doc_id = %s", (doc_id,))
    connection.execute("DELETE FROM pages WHERE doc_id = %s", (doc_id,))
    connection.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
    connection.execute(
        """
        INSERT INTO documents (
            doc_id, ticker, report_year, scope, scope_original, scope_variant,
            source_path, source_bytes, source_text, preprocessing_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            doc_id,
            metadata["ticker"],
            metadata["report_year"],
            metadata["scope"],
            metadata["scope_original"],
            metadata["scope_variant"],
            _source_path(document["path"]),
            document["path"].stat().st_size,
            document["source_text"],
            PREPROCESSING_VERSION,
        ),
    )

    page_id_by_label: dict[str, str] = {}
    for page_index, page in enumerate(document["pages"]):
        page_id = f"{doc_id}:page:{page_index}"
        page_label = document["page_labels"][page_index]
        page_id_by_label[page_label] = page_id
        connection.execute(
            """
            INSERT INTO pages (
                page_id, doc_id, page_index, page_number,
                start_char, end_char, text
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                page_id,
                doc_id,
                page_index,
                int(page_label),
                page["start"],
                page["end"],
                page["text"],
            ),
        )

    for record in document["tables"]:
        page_id = page_id_by_label[str(record["page"])]
        table_key = f"{doc_id}:table:{record['table_id']}"
        connection.execute(
            """
            INSERT INTO tables (
                table_key, doc_id, table_id, page, page_id, context,
                folded_text, n_rows, n_cols, statement_kind, unit_scale,
                rows_json, table_unit_json, cell_units_json, header_rows_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                table_key,
                doc_id,
                record["table_id"],
                record["page"],
                page_id,
                record["context"],
                record["folded_text"],
                record["n_rows"],
                record["n_cols"],
                record["statement_kind"],
                record["unit_scale"],
                record["rows_json"],
                record["table_unit_json"],
                record["cell_units_json"],
                record["header_rows_json"],
            ),
        )

    for page_index, page in enumerate(document["pages"]):
        text = page["text"].strip()
        if not text:
            continue
        page_label = document["page_labels"][page_index]
        chunk_metadata = {
            **metadata,
            "source_path": _source_path(document["path"]),
            "chunk_type": "page",
            "page_index": page_index,
            "page_number": int(page_label),
            "start": page["start"],
            "end": page["end"],
        }
        connection.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, chunk_type, page_id, start_char, end_char,
                text, search_text, metadata_json
            ) VALUES (%s, %s, 'page', %s, %s, %s, %s, %s, %s)
            """,
            (
                f"{doc_id}:page:{page_index}",
                doc_id,
                page_id_by_label[page_label],
                page["start"],
                page["end"],
                text,
                fold_text(text),
                json.dumps(chunk_metadata, ensure_ascii=False),
            ),
        )
    table_chunks = chunk_tables(document["tables"], metadata)
    table_matches = list(TABLE_MARKER.finditer(document["source_text"]))
    if len(table_chunks) != len(document["tables"]) or len(table_matches) != len(
        document["tables"]
    ):
        raise ValueError(f"Table records and source table markers differ for {doc_id}")

    for chunk, record, match in zip(
        table_chunks, document["tables"], table_matches, strict=True
    ):
        table_key = f"{doc_id}:table:{record['table_id']}"
        chunk_metadata = {**chunk["metadata"], "chunk_type": "table"}
        text = chunk["page_content"]
        connection.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, chunk_type, page_id, table_key,
                start_char, end_char, text, search_text, metadata_json
            ) VALUES (%s, %s, 'table', %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                f"{doc_id}:table:{record['table_id']}",
                doc_id,
                page_id_by_label[str(record["page"])],
                table_key,
                match.start(),
                match.end(),
                text,
                fold_text(text),
                json.dumps(chunk_metadata, ensure_ascii=False),
            ),
        )


def _initialize_worker(connection_params: dict[str, Any]) -> None:
    global _WORKER_CONNECTION
    _WORKER_CONNECTION = psycopg.connect(**connection_params)
    atexit.register(_WORKER_CONNECTION.close)


def _process_and_store(path: Path) -> dict[str, int]:
    if _WORKER_CONNECTION is None:
        raise RuntimeError("PostgreSQL worker connection was not initialized.")

    document = _load_document(path)
    with _WORKER_CONNECTION.transaction():
        insert_document(_WORKER_CONNECTION, document)

    return {
        "pages": len(document["pages"]),
        "tables": len(document["tables"]),
        "chunks": sum(
            bool(page["text"].strip()) for page in document["pages"]
        ) + len(document["tables"]),
    }


def run_pipeline(
    input_path: str | Path = DEFAULT_INPUT,
    database_name: str | None = None,
    limit: int | None = None,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Process and store documents concurrently in PostgreSQL."""
    paths = discover_documents(input_path, limit)
    if max_workers is not None and max_workers < 1:
        raise ValueError("Worker count must be at least 1.")
    worker_count = min(max_workers or os.cpu_count() or 1, len(paths))

    doc_ids = [path.stem for path in paths]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Input contains duplicate document IDs (filename stems).")

    connection_params = postgres_connection_params()
    if database_name:
        connection_params["dbname"] = database_name
    page_total = 0
    table_total = 0
    chunk_total = 0

    with psycopg.connect(**connection_params) as connection:
        for statement in SCHEMA.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute(
            "DELETE FROM corpus_meta WHERE key = 'corpus_version'"
        )

    path_iterator = iter(paths)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_worker,
        initargs=(connection_params,),
    ) as executor:
        pending = set()
        for _ in range(worker_count):
            pending.add(executor.submit(_process_and_store, next(path_iterator)))

        while pending:
            completed, pending = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                counts = future.result()
                page_total += counts["pages"]
                table_total += counts["tables"]
                chunk_total += counts["chunks"]

                try:
                    path = next(path_iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(_process_and_store, path))

    with psycopg.connect(**connection_params) as connection:
        _verify_database(
            connection,
            set(doc_ids),
            page_total,
            table_total,
            chunk_total,
        )
        connection.execute(
            """
            INSERT INTO corpus_meta (key, value)
            VALUES ('corpus_version', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (uuid.uuid4().hex,),
        )

    return {
        "ocr_files": len(paths),
        "valid_documents": len(doc_ids),
        "pages": page_total,
        "tables": table_total,
        "units": table_total,
        "chunks": chunk_total,
        "broken_foreign_keys": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate OCR documents and store document, page, table and chunk data."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Local OCR document or directory (default: data/financial_statements)",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="PostgreSQL database name (default: config/config.yml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all documents)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel processing and database-writing processes (default: all available CPU cores)",
    )
    args = parser.parse_args()

    result = run_pipeline(
        args.input_path,
        args.database,
        args.limit,
        max_workers=args.workers,
    )
    print(f"{result['ocr_files']} OCR files")
    print(f"{result['valid_documents']} valid documents")
    print(f"{result['pages']} pages stored")
    print(f"{result['tables']} tables stored")
    print(f"{result['chunks']} chunks stored")
    print(f"{result['broken_foreign_keys']} broken foreign keys")


if __name__ == "__main__":
    main()

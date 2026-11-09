"""Orchestrate preprocessing and chunk export for the report corpus."""

import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .build_document_metadata import build_document_metadata
from .chunking import chunk_tables
from .page_chunk import PAGE_MARKER, chunk_document
from .tables import extract_structured_tables


def process_dataset(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Preprocess every extracted report and write chunks plus a corpus report."""
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    reports_dir = data_dir / "financial_statements"
    if not reports_dir.is_dir():
        raise FileNotFoundError(f"Financial statements directory not found: {reports_dir}")

    report_paths = sorted(reports_dir.rglob("*_extracted.txt"))
    if not report_paths:
        raise ValueError(f"No extracted reports found under {reports_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    report_file_path = output_dir / "preprocess_report.json"
    chunks_temp_path: Path | None = None
    report_temp_path: Path | None = None
    chunks_backup_path: Path | None = None
    documents: list[dict[str, Any]] = []
    scope_counts: Counter[str] = Counter()
    page_chunk_count = 0
    table_chunk_count = 0
    empty_page_count = 0
    documents_without_tables = 0
    source_bytes = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=".chunks-",
            suffix=".tmp",
            delete=False,
        ) as chunks_file:
            chunks_temp_path = Path(chunks_file.name)

            for source_path in report_paths:
                relative_path = source_path.relative_to(data_dir).as_posix()
                metadata = build_document_metadata(source_path)
                source_text = source_path.read_text(encoding="utf-8")
                pages = chunk_document(source_text)
                page_numbers = [
                    int(marker.group(1))
                    for marker in PAGE_MARKER.finditer(source_text)
                ]
                tables = extract_structured_tables(
                    source_text,
                    str(metadata["doc_id"]),
                    int(metadata["report_year"]),
                )
                table_chunks = chunk_tables(tables, metadata)

                document_page_chunks = 0
                for page in pages:
                    page_text = page["text"].strip()
                    if not page_text:
                        empty_page_count += 1
                        continue
                    chunk = {
                        "page_content": page_text,
                        "metadata": {
                            **metadata,
                            "source_path": relative_path,
                            "chunk_type": "page",
                            "page_index": page["page_index"],
                            "page_number": page_numbers[page["page_index"]],
                            "start": page["start"],
                            "end": page["end"],
                        },
                    }
                    chunks_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    document_page_chunks += 1

                for chunk in table_chunks:
                    chunk["metadata"]["source_path"] = relative_path
                    chunk["metadata"]["chunk_type"] = "table"
                    chunks_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                scope = str(metadata["scope"])
                scope_counts[scope] += 1
                page_chunk_count += document_page_chunks
                table_chunk_count += len(table_chunks)
                source_size = source_path.stat().st_size
                source_bytes += source_size
                if not tables:
                    documents_without_tables += 1
                documents.append(
                    {
                        **metadata,
                        "source_path": relative_path,
                        "source_bytes": source_size,
                        "pages": len(pages),
                        "page_chunks": document_page_chunks,
                        "tables": len(tables),
                        "table_chunks": len(table_chunks),
                    }
                )

        summary = {
            "documents": len(documents),
            "page_chunks": page_chunk_count,
            "table_chunks": table_chunk_count,
            "total_chunks": page_chunk_count + table_chunk_count,
            "documents_without_tables": documents_without_tables,
            "empty_pages": empty_page_count,
            "source_bytes": source_bytes,
            "scopes": dict(sorted(scope_counts.items())),
        }
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_dir": str(data_dir),
            "chunks_file": chunks_path.name,
            "summary": summary,
            "documents": documents,
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=".report-",
            suffix=".tmp",
            delete=False,
        ) as report_file:
            report_temp_path = Path(report_file.name)
            json.dump(report, report_file, ensure_ascii=False, indent=2)
            report_file.write("\n")

        if chunks_path.exists():
            with tempfile.NamedTemporaryFile(
                dir=output_dir, prefix=".chunks-backup-", suffix=".tmp", delete=False,
            ) as backup_file:
                chunks_backup_path = Path(backup_file.name)
                with chunks_path.open("rb") as previous_chunks:
                    shutil.copyfileobj(previous_chunks, backup_file)

        os.replace(chunks_temp_path, chunks_path)
        chunks_temp_path = None
        try:
            os.replace(report_temp_path, report_file_path)
        except OSError:
            if chunks_backup_path is None:
                chunks_path.unlink()
            else:
                # Retain the backup for recovery if restoring it also fails.
                restore_path = chunks_backup_path
                chunks_backup_path = None
                try:
                    os.replace(restore_path, chunks_path)
                except OSError as restore_error:
                    raise RuntimeError(
                        f"Could not restore chunks; previous chunks retained at {restore_path}"
                    ) from restore_error
            raise
        report_temp_path = None
        return summary
    finally:
        for temporary_path in (chunks_temp_path, report_temp_path, chunks_backup_path):
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

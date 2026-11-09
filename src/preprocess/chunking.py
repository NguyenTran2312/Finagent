"""Build table-level retrieval chunks."""

from typing import Any

from src.preprocess.chunker import table_record_to_payload
from src.preprocess.tables import TableRecord


def chunk_tables(
    tables: list[TableRecord], document_metadata: dict[str, str | int | None],
) -> list[dict[str, Any]]:
    """Keep each complete table as page content plus its source metadata."""
    chunks = []
    for index, table in enumerate(tables):
        if table["doc_id"] != document_metadata["doc_id"]:
            raise ValueError(f"Table {table['table_id']} document ID does not match metadata")
        text, metadata = table_record_to_payload({**table, **document_metadata}, index)
        chunks.append({"page_content": text, "metadata": metadata})
    return chunks

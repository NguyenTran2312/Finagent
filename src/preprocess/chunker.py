"""Conversion of indexed table records into retrieval payloads."""

import json
from collections.abc import Mapping
from typing import Any

from src.preprocess.build_document_metadata import normalize_document_scope


_SCOPE_LABELS = {
    "consolidated": "hợp nhất",
    "separate": "công ty mẹ",
    "explanations": "thuyết minh",
    "letters": "thư giải trình",
}


def table_record_to_payload(record: Mapping[str, Any], table_index: int) -> tuple[str, dict[str, Any]]:
    metadata = dict(record)
    metadata["doc_id"] = str(record["doc_id"])
    metadata["table_id"] = int(record["table_id"])
    metadata["report_year"] = int(record["report_year"])
    scope = normalize_document_scope(str(record["scope"]))
    metadata["scope"] = scope
    metadata["table_key"] = f"{record['doc_id']}|table_{record['table_id']}"
    metadata["table_index"] = table_index
    try:
        table_text = " | ".join(
            " ; ".join(str(cell) for cell in cells)
            for cells in json.loads(record["rows_json"])
        )
    except (TypeError, ValueError):
        table_text = str(record["rows_json"])
    scope_label = _SCOPE_LABELS.get(scope, scope or "không xác định")
    text = " ".join((
        f"mã cổ phiếu {record['ticker']}",
        f"năm báo cáo {metadata['report_year']}",
        f"phạm vi {scope} ({scope_label})",
        f"loại bảng {record['statement_kind']}",
        str(record.get("context") or ""),
        table_text,
    )).strip()
    return text, metadata
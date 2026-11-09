import argparse
import json
import re
from pathlib import Path
from download_data import download_dataset
from typing import TypedDict


from src.preprocess.cleaner import (
    ParsedFinancialUnit,
    clean_text,
    fold_text,
    parse_financial_unit,
)
from src.preprocess.loader import parse_html_table
from src.indexing.table_semantics import TableAnalyzer, statement_kind

TABLE_MARKER = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
PAGE_MARKER = re.compile(r"===== PAGE\s+(\d+)\s+=====")
HTML_TAG = re.compile(r"<[^>]+>")


class TableRecord(TypedDict):
    doc_id: str
    table_id: int
    page: int
    context: str
    folded_text: str
    rows_json: str
    n_rows: int
    n_cols: int
    statement_kind: str
    unit_scale: float
    table_unit_json: str
    cell_units_json: str
    header_rows_json: str


def _unit_data(unit: ParsedFinancialUnit) -> dict[str, str | None]:
    return {
        "currency": unit.currency,
        "scale_multiplier": (
            str(unit.scale_multiplier) if unit.scale_multiplier is not None else None
        ),
        "unit_kind": unit.unit_kind,
        "unit_status": unit.unit_status,
        "source_text": unit.source_text,
        "reason": unit.reason,
    }


def _page_at(markers: list[re.Match[str]], offset: int) -> int:
    page = 0
    for marker in markers:
        if marker.start() > offset:
            break
        page = int(marker.group(1))
    return page


def _plain_context(text: str) -> str:
    return clean_text(HTML_TAG.sub(" ", text))


def extract_structured_tables(source_text: str, doc_id: str, report_year: int) -> list[TableRecord]:
    """Keep every table and original row, using the runtime index format."""
    table_matches = list(TABLE_MARKER.finditer(source_text))
    page_markers = list(PAGE_MARKER.finditer(source_text))
    records: list[TableRecord] = []
    previous_end = 0

    for table_id, match in enumerate(table_matches, start=1):
        grid = parse_html_table(match.group(0))
        page_start = max(
            (marker.start() for marker in page_markers if marker.start() <= match.start()),
            default=0,
        )
        context_before = _plain_context(
            source_text[max(previous_end, page_start):match.start()]
        )[-1200:]
        next_break = source_text.find("\n\n", match.end())
        if next_break < 0:
            next_break = min(len(source_text), match.end() + 400)
        context_after = _plain_context(source_text[match.end():next_break])[:300]

        context = clean_text(f"{context_before} {context_after}")

        analyzer = TableAnalyzer(grid, context=context, report_year=report_year)
        table_unit = parse_financial_unit(
            context,
            table_header=analyzer.table_unit,
        )
        cell_units = [
            [_unit_data(analyzer.cell(row_idx, col_idx).unit)
             for col_idx, _ in enumerate(row)]
            for row_idx, row in enumerate(grid)
        ]
        unit_scale = (
            float(table_unit.scale_multiplier)
            if table_unit.unit_status == "valid" and table_unit.scale_multiplier is not None
            else 1.0
        )
        records.append(
            {
                "doc_id": doc_id,
                "table_id": table_id,
                "page": _page_at(page_markers, match.start()),
                "context": context,
                "folded_text": fold_text(context + " " + " ".join(" ".join(row) for row in grid)),
                "rows_json": json.dumps(grid, ensure_ascii=False, separators=(",", ":")),
                "n_rows": len(grid),
                "n_cols": max((len(row) for row in grid), default=0),
                "statement_kind": statement_kind(context, grid).value,
                "unit_scale": unit_scale,
                "table_unit_json": json.dumps(
                    _unit_data(table_unit), ensure_ascii=False, separators=(",", ":")
                ),
                "cell_units_json": json.dumps(
                    cell_units, ensure_ascii=False, separators=(",", ":")
                ),
                "header_rows_json": json.dumps(list(analyzer.header_indices), separators=(",", ":")),
            }
        )
        previous_end = match.end()

    return records

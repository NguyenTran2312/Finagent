"""Semantic interpretation of flattened financial-report tables.

The OCR corpus stores every HTML table as a rectangular grid.  Rowspan and
colspan expansion is useful for execution, but it loses the distinction
between a value inherited from a header and an ordinary data cell.  This
module rebuilds the small amount of structure needed by all deterministic
solvers: hierarchical column headers, section ancestry, period roles, units
and statement type.

The functions deliberately accept plain ``list[list[str]]`` values so they
can be used while building the PostgreSQL index as well as at query time.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal
    
from src.preprocess.cleaner import (
    ParsedFinancialUnit, clean_text, fold_text, parse_vn_number, parse_financial_unit,
)


YEAR_RE = re.compile(r"\b((?:19|20|21)\d{2})\b")
CODE_RE = re.compile(r"^\d{1,3}$")
UNIT_MARKERS = (
    "nghin ty",
    "tram ty",
    "ty dong",
    "trieu dong",
    "trieu vnd",
    "nghin dong",
    "ngan dong",
    "nghin vnd",
    "ngan vnd",
    "don vi vnd",
    "don vi dong",
    "vnd",
)
TOTAL_MARKERS = (
    "tong cong",
    "cong",
    "tong so",
    "tong tai san",
    "tong nguon von",
    "tong doanh thu",
    "tong chi phi",
)
COMPONENT_MARKERS = (
    "trong do",
    "bao gom",
    "chi tiet",
    "phan bo",
    "nam hien hanh",
    "nam truoc",
)


class StatementKind(StrEnum):
    UNKNOWN = "unknown"
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    EQUITY_MOVEMENT = "equity_movement"
    NOTE = "note"


class PeriodRole(StrEnum):
    UNKNOWN = "unknown"
    CURRENT = "current"
    PRIOR = "prior"
    OPENING = "opening"
    CLOSING = "closing"


@dataclass(frozen=True, slots=True)
class PeriodInfo:
    role: PeriodRole
    year: int | None
    explicit: bool
    period_type: Literal["start", "end"] | None = None


@dataclass(frozen=True, slots=True)
class CellSemantics:
    row_idx: int
    col_idx: int
    row_label: str
    section: str
    column_header: str
    period: PeriodInfo
    unit_scale: float
    is_total: bool
    is_component: bool
    unit: ParsedFinancialUnit = ParsedFinancialUnit(None, None, "unknown", "unknown", "", "No explicit unit")


class TableAnalyzer:
    """Precomputed semantic view for repeated cell access within one table."""

    def __init__(
        self,
        rows: list[list[str]],
        *,
        context: str = "",
        report_year: int | None = None,
    ) -> None:
        self.rows = rows
        self.context = context
        self.report_year = report_year
        self.header_indices = tuple(idx for idx, row in enumerate(rows) if is_header_row(row))
        self.labels = tuple(row_label(row) for row in rows)
        sections: list[str] = []
        active: list[str] = []
        self._header_set = set(self.header_indices)
        self._column_header_cache: dict[int, tuple[int, set[str], str]] = {}
        for idx, row in enumerate(rows):
            sections.append(_dedupe(active[-3:]))
            if idx in self._header_set:
                continue
            populated = [clean_text(cell) for cell in row if clean_text(cell)]
            if populated and not any(parse_vn_number(cell) is not None for cell in populated):
                label = self.labels[idx]
                if label:
                    active.append(label)
        self.sections = tuple(sections)
        header_text = " | ".join(" ".join(rows[idx]) for idx in self.header_indices if idx < len(rows))
        self.table_unit = parse_financial_unit(table_header=header_text)
        self.context_unit = parse_financial_unit(context)
        self.table_scale = float(self.table_unit.scale_multiplier) if self.table_unit.unit_status == "valid" else None
        self.context_scale = float(self.context_unit.scale_multiplier) if self.context_unit.unit_status == "valid" else None

    def column_header(self, row_idx: int, col_idx: int) -> str:
        start, seen, header = self._column_header_cache.get(col_idx, (0, set(), ""))
        if row_idx < start:
            return _column_header_for_block(self.rows, row_idx, col_idx, self.header_indices)

        # During row-wise extraction, normalize each header cell only once.
        for idx in range(start, row_idx):
            row = self.rows[idx]
            if idx in self._header_set:
                if col_idx < len(row):
                    value = clean_text(row[col_idx])
                    folded = fold_text(value)
                    if value and folded and folded not in seen:
                        seen.add(folded)
                        header = f"{header} | {value}" if header else value
            elif any(clean_text(cell) for cell in row):
                seen = set()
                header = ""
        self._column_header_cache[col_idx] = (row_idx, seen, header)
        return header

    def cell(self, row_idx: int, col_idx: int, *, code_idx: int | None = None) -> CellSemantics:
        label = row_label(self.rows[row_idx], code_idx=code_idx) if code_idx is not None else self.labels[row_idx]
        section = self.sections[row_idx]
        header = self.column_header(row_idx, col_idx)
        unit = parse_financial_unit(self.context, column_header=header, table_header=self.table_unit)
        scale = float(unit.scale_multiplier) if unit.unit_status == "valid" else 1.0
        folded_label = fold_text(label)
        folded_scope = fold_text(f"{section} {label}")
        is_total = any(
            folded_label == marker or folded_label.startswith(f"{marker} ")
            for marker in TOTAL_MARKERS
        )
        is_component = any(marker in folded_scope for marker in COMPONENT_MARKERS)
        return CellSemantics(
            row_idx=row_idx,
            col_idx=col_idx,
            row_label=label,
            section=section,
            column_header=header,
            period=period_info(header, report_year=self.report_year),
            unit_scale=scale,
            is_total=is_total,
            is_component=is_component,
            unit=unit,
        )


def _dedupe(parts: list[str]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = clean_text(part)
        folded = fold_text(value)
        if not value or not folded or folded in seen:
            continue
        seen.add(folded)
        result.append(value)
    return " | ".join(result)


def _column_header_for_block(
    rows: list[list[str]],
    row_idx: int,
    col_idx: int,
    header_indices: tuple[int, ...],
) -> str:
    header_set = set(header_indices)
    preceding_headers: list[int] = []
    for idx in range(row_idx - 1, -1, -1):
        if idx in header_set:
            preceding_headers.append(idx)
        elif any(clean_text(cell) for cell in rows[idx]):
            break
    parts = [
        rows[idx][col_idx]
        for idx in reversed(preceding_headers)
        if col_idx < len(rows[idx])
    ]
    return _dedupe(parts)


def row_label(row: list[str], *, code_idx: int | None = None) -> str:
    """Return the most informative non-numeric label in a data row."""

    labels: list[str] = []
    for idx, raw in enumerate(row):
        text = clean_text(raw)
        if not text or idx == code_idx or parse_vn_number(text) is not None:
            continue
        # A bare line-item code is not a semantic label.
        if CODE_RE.fullmatch(text):
            continue
        labels.append(text)
    return max(labels, key=lambda item: (len(fold_text(item).split()), len(item)), default="")


def is_header_row(row: list[str]) -> bool:
    """Conservative classifier for global or repeated local header rows."""

    populated = [clean_text(cell) for cell in row if clean_text(cell)]
    if not populated:
        return False
    # Missing amounts must not let a financial row masquerade as a period header.
    if any(parse_vn_number(cell) is not None and not YEAR_RE.fullmatch(cell) for cell in populated):
        return False
    folded = fold_text(" ".join(populated))
    header_cues = (
        "nam nay",
        "nam truoc",
        "ky nay",
        "ky truoc",
        "so cuoi",
        "so dau",
        "tai ngay",
        "don vi",
        "thuyet minh",
        "chi tieu",
    )
    if any(cue in folded for cue in header_cues) or YEAR_RE.search(folded):
        return True
    # Colspan expansion frequently creates repeated text across a header.
    folded_cells = [fold_text(cell) for cell in populated]
    repeated = len(folded_cells) >= 2 and len(set(folded_cells)) < len(folded_cells)
    return repeated


def column_header(rows: list[list[str]], row_idx: int, col_idx: int) -> str:
    """Reconstruct the nearest header block above one data cell."""

    header_indices = tuple(idx for idx, row in enumerate(rows) if is_header_row(row))
    return _column_header_for_block(rows, row_idx, col_idx, header_indices)


def section_text(rows: list[list[str]], row_idx: int, *, lookback: int = 10) -> str:
    """Return nearby non-numeric section labels preceding ``row_idx``."""

    parts: list[str] = []
    for idx in range(max(0, row_idx - lookback), row_idx):
        row = rows[idx]
        if is_header_row(row):
            continue
        populated = [clean_text(cell) for cell in row if clean_text(cell)]
        numeric = sum(parse_vn_number(cell) is not None for cell in populated)
        if not populated or numeric > 0:
            continue
        label = row_label(row)
        if label:
            parts.append(label)
    return _dedupe(parts[-3:])


def period_info(header: str, *, report_year: int | None = None) -> PeriodInfo:
    folded = fold_text(header)
    years = [int(value) for value in YEAR_RE.findall(folded)]
    year = years[-1] if years else None
    explicit = bool(years)
    opening_date = bool(re.search(r"\b0?1 0?1(?: 20\d{2})?\b", folded))
    closing_date = bool(re.search(r"\b31 12(?: 20\d{2})?\b", folded))
    if opening_date or any(marker in folded for marker in ("so dau", "dau ky", "dau nam")):
        role = PeriodRole.OPENING
        period_type: Literal["start", "end"] | None = "start"
    elif closing_date or any(marker in folded for marker in ("so cuoi", "cuoi ky", "cuoi nam")):
        role = PeriodRole.CLOSING
        period_type = "end"
    elif any(marker in folded for marker in ("nam truoc", "ky truoc", "so sanh")):
        role = PeriodRole.PRIOR
        period_type = None
    elif any(marker in folded for marker in ("nam nay", "ky nay", "hien tai")):
        role = PeriodRole.CURRENT
        period_type = None
    elif report_year is not None and year == report_year:
        role = PeriodRole.CURRENT
        period_type = None
    elif report_year is not None and year == report_year - 1:
        role = PeriodRole.PRIOR
        period_type = None
    else:
        role = PeriodRole.UNKNOWN
        period_type = None
    return PeriodInfo(role=role, year=year, explicit=explicit, period_type=period_type)


def _explicit_unit_scale(text: str) -> float | None:
    unit = parse_financial_unit(text)
    return float(unit.scale_multiplier) if unit.unit_status == "valid" else None


def unit_scale(
    rows: list[list[str]],
    row_idx: int,
    col_idx: int,
    *,
    context: str = "",
) -> float:
    """Infer a cell's source unit using nearest/most-specific evidence first."""

    return TableAnalyzer(rows, context=context).cell(row_idx, col_idx).unit_scale


def statement_kind(context: str, rows: list[list[str]]) -> StatementKind:
    lead = " ".join(" ".join(row) for row in rows[: min(8, len(rows))])
    folded = fold_text(f"{context} {lead}")
    if any(marker in folded for marker in ("bao cao luu chuyen tien te", "luu chuyen tien te")):
        return StatementKind.CASH_FLOW
    if any(marker in folded for marker in ("bao cao ket qua hoat dong kinh doanh", "ket qua kinh doanh")):
        return StatementKind.INCOME_STATEMENT
    if any(marker in folded for marker in ("bang can doi ke toan", "bao cao tinh hinh tai chinh")):
        return StatementKind.BALANCE_SHEET
    if any(marker in folded for marker in ("bao cao bien dong von chu so huu", "thay doi von chu so huu")):
        return StatementKind.EQUITY_MOVEMENT
    if any(marker in folded for marker in ("thuyet minh", "chi tiet")):
        return StatementKind.NOTE
    return StatementKind.UNKNOWN


def cell_semantics(
    rows: list[list[str]],
    row_idx: int,
    col_idx: int,
    *,
    context: str = "",
    report_year: int | None = None,
    code_idx: int | None = None,
) -> CellSemantics:
    label = row_label(rows[row_idx], code_idx=code_idx)
    section = section_text(rows, row_idx)
    header = column_header(rows, row_idx, col_idx)
    folded_label = fold_text(label)
    folded_scope = fold_text(f"{section} {label}")
    is_total = any(
        folded_label == marker or folded_label.startswith(f"{marker} ")
        for marker in TOTAL_MARKERS
    )
    is_component = any(marker in folded_scope for marker in COMPONENT_MARKERS)
    unit = TableAnalyzer(rows, context=context, report_year=report_year).cell(row_idx, col_idx).unit
    return CellSemantics(
        row_idx=row_idx,
        col_idx=col_idx,
        row_label=label,
        section=section,
        column_header=header,
        period=period_info(header, report_year=report_year),
        unit_scale=float(unit.scale_multiplier) if unit.unit_status == "valid" else 1.0,
        is_total=is_total,
        is_component=is_component,
        unit=unit,
    )


def question_statement_preference(question: str) -> tuple[StatementKind, ...]:
    folded = fold_text(question)
    if any(marker in folded for marker in ("luu chuyen", "dong tien", "tien thuan")):
        return (StatementKind.CASH_FLOW, StatementKind.NOTE)
    if any(marker in folded for marker in ("doanh thu", "loi nhuan", "chi phi", "lai lo")):
        return (StatementKind.INCOME_STATEMENT, StatementKind.NOTE)
    if any(
        marker in folded
        for marker in (
            "tai san",
            "nguon von",
            "so du",
            "cuoi nam",
            "dau nam",
            "tai ngay",
            "phai thu",
            "phai tra",
        )
    ):
        return (StatementKind.BALANCE_SHEET, StatementKind.NOTE)
    return ()


def asks_total(question: str) -> bool:
    folded = fold_text(question)
    return any(
        marker in folded
        for marker in ("tong cong", "tong so", "tong tai san", "tong nguon von", "tong doanh thu")
    )


def period_match_score(period: PeriodInfo, *, requested_year: int, report_year: int) -> float:
    """Score whether a header represents the requested reporting period."""

    if requested_year <= 0:
        return 0.0
    if period.year is not None:
        if period.year == requested_year:
            return 8.0 if period.period_type == "start" else 10.0
        return -12.0
    if requested_year == report_year:
        if period.role in (PeriodRole.CURRENT, PeriodRole.CLOSING):
            return 8.0
        if period.role in (PeriodRole.PRIOR, PeriodRole.OPENING):
            return -7.0
    if requested_year == report_year - 1:
        if period.role in (PeriodRole.PRIOR, PeriodRole.OPENING):
            return 8.0
        if period.role in (PeriodRole.CURRENT, PeriodRole.CLOSING):
            return -7.0
    return 0.0

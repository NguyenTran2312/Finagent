"""Deterministic reading of one stored table: codes, labels, periods, numbers, units (spec §6.1, §6.5-§6.7)."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from src.calculation.facts import FactSource
from src.calculation.units import Dimension, Quantity
from src.extraction.outcomes import CellLocation
from src.extraction.report_store import TableRecord
from src.indexing.table_semantics import PeriodRole, TableAnalyzer, period_info, row_label
from src.preprocess.cleaner import (
    ParsedFinancialUnit,
    clean_text,
    fold_text,
    parse_financial_number,
    parse_financial_unit,
)

SourceRole = Literal["primary", "fallback"]
Locale = Literal["vi", "en", "mixed"]
Level = Literal["top", "section", "item", "sub_item"]


def _roman(n: int) -> str:
    return "X" * (n // 10) + ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX")[n % 10]


_ROMANS = frozenset(_roman(n) for n in range(1, 31))
# A leading enumerator token: optional "(", the token, then punctuation ("VI.", "1.", "a)"), space or end.
_ENUM_RE = re.compile(r"\(?(?P<tok>[IVX]+|\d{1,2}|[A-Za-z])(?:[.):]+\s*|\s+|$)")
_CODE_DIGITS = re.compile(r"\d{1,3}")
_ITEM = re.compile(r"\d{1,2}")
_LETTER = re.compile(r"[A-Za-z]")
_RESTATED = ("trinh bay lai", "dieu chinh lai", "restated")
_NO_UNIT = ParsedFinancialUnit(None, None, "unknown", "unknown", "", "No explicit supported unit")

# §6.7: dates, four-digit years and period phrases are removed before a header cell is parsed as a unit.
_DATE_RES = (
    re.compile(r"(?<!\d)\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{4}(?!\d)"),
    re.compile(r"(?:ng[àa]y\s*)?(?<!\d)\d{1,2}\s*th[áa]ng\s*\d{1,2}(?:\s*n[ăa]m(?:\s*\d{4}(?!\d))?)?", re.I),
)
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20|21)\d{2}(?!\d)")
_PERIOD_PHRASES = tuple(
    sorted(
        (
            tuple(fold_text(p).split())
            for p in (
                "số cuối năm", "số đầu năm", "số cuối kỳ", "số đầu kỳ", "cuối năm", "đầu năm",
                "năm nay", "năm trước", "kỳ này", "kỳ trước", "tại ngày", "vào ngày", "tại", "ngày",
                "cho năm tài chính kết thúc", "năm tài chính kết thúc", "cho năm", "năm",
            )
        ),
        key=len,
        reverse=True,
    )
)
# §6.7 unit spellings seen in the corpus that the shared parser does not read (the cleaner is not changed here).
_DECLARATION_OCR = re.compile(r"\b(?:đơn|don)\s*(?:vị|vi)\s*(?:tính|tinh)\b", re.I)
_DECLARATION = re.compile(
    r"(?:(?:đơn|don)\s*(?:vị|vi)(?:\s*(?:tính|tinh))?|units?(?:\s+of\s+measurement)?|currency)\s*:\s*[^\n;]+", re.I
)
_VND_WORDS = re.compile(r"\b(?:đồng\s+việt\s+nam|việt\s+nam\s+đồng)\b", re.I)
_GLUED_VND = re.compile(r"(?<=[^\W\d_])(?=VND\b)")
_CONTINUED = re.compile(r"\(?\s*(?:tiếp\s+theo|continued)\s*\)?", re.I)
_TRAILING_UNIT_WORDS = 3


class MalformedTable(ValueError):
    def __init__(self, table_key: str, reason: Literal["no_chunk", "json", "shape"]):
        super().__init__(f"malformed table {table_key}: {reason}")
        self.table_key = table_key
        self.reason = reason


def normalize_label(text: str) -> str:
    """Folded label with one leading enumerator token removed.

    The enumerator is decided on the raw token (uppercase roman I-XXX, 1-2 digits, one ASCII
    letter), so words that only fold to a roman numeral ("Xí nghiệp", "Vị thế") are kept.
    """
    raw = clean_text(unicodedata.normalize("NFC", clean_text(text)))
    match = _ENUM_RE.match(raw)
    if match and (match.group("tok") in _ROMANS or len(match.group("tok")) == 1 or match.group("tok").isdigit()):
        raw = raw[match.end():]
    return fold_text(raw)


def enumerator_level(code: str) -> Level | None:
    """Bank enumerator level of a code cell; case-sensitive (§6.4)."""
    code = clean_text(code)
    if code in _ROMANS:
        return "section"
    if _ITEM.fullmatch(code):
        return "item"
    if _LETTER.fullmatch(code):
        return "top" if code.isupper() else "sub_item"
    return None


def _code_like(cell: str) -> bool:
    return bool(_CODE_DIGITS.fullmatch(cell) or cell in _ROMANS or _LETTER.fullmatch(cell))


def _strip_periods(cell: str) -> str:
    text = unicodedata.normalize("NFC", cell)
    for pattern in _DATE_RES:
        text = pattern.sub(" ", text)
    words = _YEAR_RE.sub(" ", text).split()
    folded = [fold_text(w) for w in words]
    kept: list[str] = []
    i = 0
    while i < len(words):
        phrase = next((p for p in _PERIOD_PHRASES if tuple(folded[i:i + len(p)]) == p), None)
        if phrase:
            i += len(phrase)
        else:
            kept.append(words[i])
            i += 1
    return " ".join(kept)


def _unit_spelling(text: str) -> str:
    """Rewrite corpus unit spellings into forms the shared parser reads (§6.7)."""
    text = unicodedata.normalize("NFC", text)
    text = _DECLARATION_OCR.sub("Đơn vị tính", text)
    text = _VND_WORDS.sub("VND", text)
    return _GLUED_VND.sub(" ", text)


def header_unit(cells: Sequence[str]) -> ParsedFinancialUnit:
    """Column unit from its header cells, nearest header row first (§6.7).

    ``source_text`` of a valid result is the header cell the unit came from.
    """
    found: ParsedFinancialUnit | None = None
    for cell in cells:
        cell = clean_text(cell)
        remainder = _strip_periods(_unit_spelling(cell))
        if not re.search(r"\w|%", remainder):
            continue
        parsed = parse_financial_unit("", column_header=remainder)
        if parsed.unit_status == "ambiguous":
            return replace(parsed, source_text=cell)
        if parsed.unit_status != "valid":
            continue
        if found is None:
            found = replace(parsed, source_text=cell)
        elif (parsed.unit_kind, parsed.currency, parsed.scale_multiplier) != (
            found.unit_kind, found.currency, found.scale_multiplier
        ):
            return ParsedFinancialUnit(None, None, "unknown", "ambiguous", cell, "Conflicting header units")
    return found or _NO_UNIT


def context_unit(context: str) -> ParsedFinancialUnit:
    """Table unit from the text above the table (§6.7), after header and table-header units.

    A declaration ("Đơn vị tính: …") is parsed as before. Otherwise a unit of at most three words
    that ends the context right after its last date or year ("… năm 2016 Ngàn VND") is the unit.
    ``source_text`` of a result is the declaration or the trailing unit, not the whole context.
    """
    original = clean_text(context or "")
    text = _unit_spelling(original)
    parsed = parse_financial_unit(text)
    if parsed.unit_status != "unknown":
        declaration = _DECLARATION.search(unicodedata.normalize("NFC", original))
        return replace(parsed, source_text=declaration.group(0).strip() if declaration else original)
    ends = [m.end() for pattern in (*_DATE_RES, _YEAR_RE) for m in pattern.finditer(text)]
    if not ends:
        return _NO_UNIT
    tail = _CONTINUED.sub(" ", text[max(ends):]).strip(" .:;,-–()")
    if not tail or len(tail.split()) > _TRAILING_UNIT_WORDS:
        return _NO_UNIT
    unit = parse_financial_unit("", column_header=tail)
    return replace(unit, source_text=tail) if unit.unit_status == "valid" else _NO_UNIT


@dataclass(frozen=True)
class CellRead:
    status: Literal["ok", "blank", "unparsed", "unit"]
    location: CellLocation
    reason: str | None = None
    value: Quantity | None = None
    source: FactSource | None = None


def _decode(record: TableRecord) -> list[list[str]]:
    if record.chunk_id is None:
        raise MalformedTable(record.table_key, "no_chunk")
    try:
        rows = json.loads(record.rows_json)
    except (TypeError, ValueError):
        raise MalformedTable(record.table_key, "json") from None
    if not isinstance(rows, list) or not all(
        isinstance(row, list) and all(isinstance(cell, str) for cell in row) for row in rows
    ):
        raise MalformedTable(record.table_key, "shape")
    return rows


def _number_status(raw: str, locale: str) -> str:
    return parse_financial_number(raw, locale=locale).status


class TableView:
    """One table's deterministic structure; raises ``MalformedTable`` for undecodable records."""

    def __init__(self, record: TableRecord, report_year: int):
        self.record = record
        self.report_year = report_year
        self._rows = _decode(record)
        self._analyzer = TableAnalyzer(self._rows, context=record.context, report_year=report_year)
        headers = self._analyzer.header_indices
        header_set = set(headers)
        n_cols = max((len(row) for row in self._rows), default=0)
        self.data_rows = tuple(i for i in range(len(self._rows)) if i not in header_set)
        # Header cells per column, top row first.
        self._header_cells = tuple(
            tuple(c for c in (clean_text(self._cell(i, col)) for i in headers) if c) for col in range(n_cols)
        )
        self._headers = tuple(_join_header(cells) for cells in self._header_cells)
        # Space-insensitive so OCR-joined "Thuyếtminh" / "Mãsố" headers are recognised too.
        squashed = [fold_text(h).replace(" ", "") for h in self._headers]
        note_cols = {col for col in range(n_cols) if "thuyetminh" in squashed[col]}
        self.code_col = self._find_code_col(n_cols, squashed, note_cols)
        self._labels = tuple(row_label(row, code_idx=self.code_col) for row in self._rows)
        self._norm_labels = tuple(normalize_label(label) for label in self._labels)
        self.value_cols = tuple(
            col for col in range(n_cols)
            if col != self.code_col and col not in note_cols
            and any(self._parses(self._cell(r, col)) for r in self.data_rows)
        )
        self.locale = self._detect_locale()
        self._multi_block = self._has_conflicting_header_block(headers)
        self._units: dict[int, ParsedFinancialUnit] = {}

    # --- structure -------------------------------------------------------
    def _cell(self, row_idx: int, col_idx: int) -> str:
        row = self._rows[row_idx]
        return row[col_idx] if col_idx < len(row) else ""

    def _find_code_col(self, n_cols: int, squashed: list[str], note_cols: set[int]) -> int | None:
        by_header = [c for c in range(n_cols) if "maso" in squashed[c] and c not in note_cols]
        if by_header:
            return by_header[0] if len(by_header) == 1 else None
        qualifying = []
        for col in range(n_cols):
            if col in note_cols:
                continue
            cells = [c for c in (clean_text(self._cell(r, col)) for r in self.data_rows) if c]
            coded = sum(_code_like(c) for c in cells)
            if coded >= 3 and coded >= 0.8 * len(cells):
                qualifying.append(col)
        return qualifying[0] if len(qualifying) == 1 else None

    def _has_conflicting_header_block(self, headers: Sequence[int]) -> bool:
        """True when a header block after the first data row re-labels a value column's period.

        Only cells with period evidence (a year or a period role) are compared, so empty cells and
        colspan-repeated section labels ("LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG ĐẦU TƯ") never trigger it;
        a block that repeats the leading block's period text (page-break repeat) does not either.
        """
        first_data = next((r for r in self.data_rows if any(clean_text(c) for c in self._rows[r])), None)
        if first_data is None:
            return False
        leading = [i for i in headers if i < first_data]
        blocks: list[list[int]] = []
        for i in (i for i in headers if i > first_data):
            if blocks and blocks[-1][-1] == i - 1:
                blocks[-1].append(i)
            else:
                blocks.append([i])
        for col in self.value_cols:
            lead_text = self._period_text(leading, col)
            if any((text := self._period_text(block, col)) and text != lead_text for block in blocks):
                return True
        return False

    def _period_text(self, header_rows: Sequence[int], col: int) -> str:
        parts = []
        for i in header_rows:
            folded = fold_text(self._cell(i, col))
            info = period_info(folded, report_year=self.report_year)
            if folded and (info.year is not None or info.role != PeriodRole.UNKNOWN):
                parts.append(folded)
        return " | ".join(parts)

    @staticmethod
    def _parses(raw: str) -> bool:
        return any(_number_status(raw, loc) == "valid" for loc in ("vi", "en"))

    def _detect_locale(self) -> Locale:
        vi_only = en_only = False
        for r in self.data_rows:
            for col in self.value_cols:
                raw = self._cell(r, col)
                vi, en = (_number_status(raw, loc) == "valid" for loc in ("vi", "en"))
                vi_only |= vi and not en
                en_only |= en and not vi
        if vi_only and en_only:
            return "mixed"
        return "en" if en_only else "vi"

    def label(self, row_idx: int) -> str:
        return self._labels[row_idx]

    def norm_label(self, row_idx: int) -> str:
        return self._norm_labels[row_idx]

    def code(self, row_idx: int) -> str:
        return "" if self.code_col is None else clean_text(self._cell(row_idx, self.code_col))

    def column_header(self, col_idx: int) -> str:
        return self._headers[col_idx] if col_idx < len(self._headers) else ""

    def tokens(self) -> frozenset[str]:
        """Distinct folded tokens of the data rows' normalised labels."""
        return frozenset(t for r in self.data_rows for t in self._norm_labels[r].split())

    def restatement_marker(self, col_idx: int) -> bool:
        text = f"{fold_text(self.column_header(col_idx))} {fold_text(self.record.context)}"
        return any(marker in text for marker in _RESTATED)

    # --- periods (§6.5, ruling R1) ----------------------------------------
    def period_columns(self, year: int, period_kind: str, role: SourceRole) -> tuple[int, ...]:
        if period_kind not in ("flow", "closing"):
            raise ValueError("period_kind must be flow or closing")
        expected = {"primary": self.report_year, "fallback": self.report_year - 1}.get(role)
        if expected is None or year != expected:
            raise ValueError("role must be primary (year == report_year) or fallback (year == report_year - 1)")
        if role == "primary":
            roles = {PeriodRole.CURRENT} if period_kind == "flow" else {PeriodRole.CLOSING, PeriodRole.CURRENT}
        else:
            roles = {PeriodRole.PRIOR} if period_kind == "flow" else {PeriodRole.OPENING, PeriodRole.PRIOR}
        if self._multi_block:  # one header per column cannot describe several blocks: fail safe
            return ()
        matched = []
        for col in self.value_cols:
            info = period_info(self.column_header(col), report_year=self.report_year)
            role_year = self.report_year - 1 if info.role == PeriodRole.PRIOR else self.report_year
            by_role = info.role in roles and (info.year is None or info.year == role_year)
            by_year = info.year == year and (
                info.period_type != "start" if period_kind == "flow" else info.period_type == "end"
            )
            if by_role or by_year:
                matched.append(col)
        return tuple(matched)

    # --- numbers and units (§6.6, §6.7) -----------------------------------
    def _unit(self, col_idx: int) -> ParsedFinancialUnit:
        if col_idx not in self._units:
            cells = self._header_cells[col_idx] if col_idx < len(self._header_cells) else ()
            unit = header_unit(tuple(reversed(cells)))
            if unit.unit_status == "unknown":
                table_unit = self._analyzer.table_unit
                unit = table_unit if table_unit.unit_status != "unknown" else context_unit(self.record.context)
            self._units[col_idx] = unit
        return self._units[col_idx]

    def read(self, row_idx: int, col_idx: int, dimension: Dimension) -> CellRead:
        raw = clean_text(self._cell(row_idx, col_idx))
        location = CellLocation(
            chunk_id=self.record.chunk_id,
            doc_id=self.record.doc_id,
            page=self.record.page,
            row_label=self.label(row_idx),
            column_header=self.column_header(col_idx),
            raw_text=raw,
        )
        if self.locale == "mixed":
            return CellRead("unparsed", location, "number_locale")
        number = parse_financial_number(raw, locale=self.locale)
        if number.status == "missing":
            return CellRead("blank", location)
        if number.status != "valid":
            return CellRead("unparsed", location, "number_unparsed")
        unit = self._unit(col_idx)
        problem = _unit_problem(unit, dimension, cell_is_percent=number.is_percent)
        if problem:
            return CellRead("unit", location, problem)
        source = FactSource(
            chunk_id=self.record.chunk_id,
            doc_id=self.record.doc_id,
            page=self.record.page,
            row_label=location.row_label or self.code(row_idx) or f"row:{row_idx}",
            raw_text=raw,
            unit_text=unit.source_text,
            scale=unit.scale_multiplier,
            column_header=location.column_header or None,
        )
        return CellRead("ok", location, value=Quantity(number.value * unit.scale_multiplier, dimension), source=source)


def _join_header(cells: Sequence[str]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        folded = fold_text(cell)
        if folded and folded not in seen:
            seen.add(folded)
            parts.append(cell)
    return " | ".join(parts)


def _unit_problem(unit: ParsedFinancialUnit, dimension: Dimension, *, cell_is_percent: bool) -> str | None:
    if dimension not in (Dimension.MONEY, Dimension.MONEY_PER_SHARE):
        raise ValueError(f"unsupported leaf dimension: {dimension}")
    if unit.unit_status == "unknown":
        return "missing"
    if unit.unit_status == "ambiguous":
        return "ambiguous"
    if unit.unit_kind != "monetary" or cell_is_percent:
        return "kind_mismatch"
    if unit.currency != "VND":
        return "currency"
    if dimension == Dimension.MONEY_PER_SHARE and unit.scale_multiplier != 1:
        return "kind_mismatch"
    return None

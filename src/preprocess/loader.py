"""Expansion of OCR HTML tables, including merged cells."""

from html.parser import HTMLParser

from src.preprocess.cleaner import clean_text


_MAX_TABLE_ROWS = 10_000
_MAX_TABLE_COLUMNS = 1_000
_MAX_TABLE_CELLS = 100_000


def _positive_int(value: str | None, default: int = 1) -> int:
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, dict[str, str | None]]]] = []
        self._row: list[tuple[str, dict[str, str | None]]] | None = None
        self._attrs: dict[str, str | None] | None = None
        self._text: list[str] = []
        self._cell_count = 0

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            if len(self.rows) >= _MAX_TABLE_ROWS:
                raise ValueError("HTML table exceeds the expanded row limit")
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            if len(self._row) >= _MAX_TABLE_COLUMNS or self._cell_count >= _MAX_TABLE_CELLS:
                raise ValueError("HTML table exceeds the expanded size limits")
            self._attrs = dict(attrs)
            self._text = []

    def handle_data(self, data):
        if self._attrs is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self._row is not None and self._attrs is not None:
            self._row.append((clean_text(" ".join(self._text)), self._attrs))
            self._cell_count += 1
            self._attrs = None
            self._text = []
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def parse_html_table(fragment: str) -> list[list[str]]:
    parser = _TableParser()
    parser.feed(fragment)
    height = len(parser.rows)
    if height > _MAX_TABLE_ROWS:
        raise ValueError("HTML table exceeds the expanded row limit")
    pending: dict[int, tuple[str, int]] = {}
    sparse_rows: list[dict[int, str]] = []
    width = 0
    for row_index, cells in enumerate(parser.rows):
        current = {col: text for col, (text, _) in pending.items()}
        next_pending = {
            col: (text, remaining - 1)
            for col, (text, remaining) in pending.items()
            if remaining > 1
        }
        col = 0
        for text, attrs in cells:
            while col in current:
                col += 1
            colspan = _positive_int(attrs.get("colspan"))
            rowspan = _positive_int(attrs.get("rowspan"))
            height = max(height, row_index + rowspan)
            if height > _MAX_TABLE_ROWS or colspan > _MAX_TABLE_COLUMNS:
                raise ValueError("HTML table span exceeds the expanded dimension limits")
            while any(col + offset in current for offset in range(colspan)):
                col += 1
            width = max(width, col + colspan)
            if width > _MAX_TABLE_COLUMNS or height * width > _MAX_TABLE_CELLS:
                raise ValueError("HTML table exceeds the expanded size limits")
            for offset in range(colspan):
                current[col + offset] = text
                if rowspan > 1:
                    next_pending[col + offset] = (text, rowspan - 1)
            col += colspan
        pending = next_pending
        if current:
            width = max(width, max(current) + 1)
            sparse_rows.append(current)
    while pending:
        current = {col: text for col, (text, _) in pending.items()}
        pending = {
            col: (text, remaining - 1)
            for col, (text, remaining) in pending.items()
            if remaining > 1
        }
        width = max(width, max(current) + 1)
        sparse_rows.append(current)
    return [[row.get(col, "") for col in range(width)] for row in sparse_rows]

"""Shared helpers for leaf-extraction tests."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from src.extraction.report_store import ReportDoc, TableRecord
from src.extraction.table_view import TableView


def record(
    doc_id: str,
    table_id: int,
    page: int,
    rows: list[list[str]],
    *,
    kind: str | None,
    context: str = "",
    chunk: bool = True,
) -> TableRecord:
    return TableRecord(
        table_key=f"{doc_id}:{table_id}",
        chunk_id=f"{doc_id}:table:{table_id}" if chunk else None,
        doc_id=doc_id,
        table_id=table_id,
        page=page,
        context=context,
        statement_kind=kind,
        rows_json=json.dumps(rows, ensure_ascii=False),
    )


class _InMemorySession:
    def __init__(self, store: InMemoryReportStore):
        self._docs = store._docs
        self._tables = store._tables
        self._store = store

    def documents(self, ticker: str, year: int) -> list[ReportDoc]:
        self._store.documents_queried.append((ticker, year))
        want = ticker.strip().upper()
        found = [d for d in self._docs if d.ticker == want and d.report_year == year]
        return sorted(
            found,
            key=lambda d: (d.scope, d.scope_variant is not None, d.scope_variant or 0, d.doc_id),
        )

    def tables(self, doc_id: str) -> list[TableRecord]:
        self._store.tables_read[doc_id] += 1
        return sorted(self._tables.get(doc_id, ()), key=lambda t: (t.page, t.table_id))


class InMemoryReportStore:
    def __init__(self, docs: Sequence[ReportDoc], tables: Mapping[str, Sequence[TableRecord]]):
        self._docs = list(docs)
        self._tables = {k: list(v) for k, v in tables.items()}
        self.sessions_opened = 0
        self.documents_queried: list[tuple[str, int]] = []  # (ticker, year) as passed
        self.tables_read: Counter[str] = Counter()  # doc_id -> tables() calls

    @contextmanager
    def session(self) -> Iterator[_InMemorySession]:
        self.sessions_opened += 1
        yield _InMemorySession(self)


# --- Table fixtures (Task 4). Rows marked "real" are copied from the corpus; others are synthetic.

_VJC_PERIOD = "Năm tài chính kết thúc ngày 31 tháng 12 năm"

VJC_IS_2023 = [
    ["Mã số", "", "Thuyết minh", _VJC_PERIOD, _VJC_PERIOD],  # real
    ["Mã số", "", "Thuyết minh", "2023 VND", "2022 VND"],  # real
    ["01", "Doanh thu bán hàng và cung cấp dịch vụ", "", "58.340.618.715.308", "40.141.863.488.029"],
    ["10", "Doanh thu thuần về bán hàng và cung cấp dịch vụ", "29", "58.340.618.715.308", "40.141.863.488.029"],
    ["11", "Giá vốn hàng bán và dịch vụ cung cấp", "30", "(55.819.871.829.489)", "(42.135.151.389.707)"],
    ["20", "Lợi nhuận/(lỗ) gộp về bán hàng và cung cấp dịch vụ", "", "2.520.746.885.819", "(1.993.287.901.678)"],
    ["23", "- Trong đó: Chi phí lãi vay", "32", "(1.950.254.195.864)", "(1.383.641.245.573)"],
    # synthetic
    ["50", "Tổng lợi nhuận kế toán trước thuế", "", "606.103.607.021", "(2.648.659.574.470)"],
    ["60", "Lợi nhuận sau thuế thu nhập doanh nghiệp", "", "231.413.415.351", "(2.261.974.843.785)"],
]

ACB_BS_2022 = [
    ["", "", "Thuyết minh", "Tại ngày", "Tại ngày"],  # real
    ["", "", "Thuyết minh", "31.12.2022 Triệu VND", "31.12.2021 Triệu VND"],  # real
    ["A", "TÀI SẢN", "", "", ""],
    ["I", "Tiền mặt, vàng bạc, đá quý", "4", "8.460.892", "7.509.877"],
    ["III", "Tiền gửi và cho vay các tổ chức tín dụng khác", "6", "85.971.313", "49.818.776"],
    ["1", "Tiền gửi tại các tổ chức tín dụng khác", "6.1", "82.338.762", "43.713.230"],
    ["2", "Cho vay các tổ chức tín dụng khác", "6.1", "3.682.551", "6.105.546"],
    ["3", "Dự phòng rủi ro", "6.3", "(50.000)", "-"],
    ["4", "Cho vay khách hàng", "", "1.000", "900"],  # synthetic decoy inside section III
    ["VI", "Cho vay khách hàng", "9", "408.856.508", "356.050.950"],
    ["1", "Cho vay khách hàng", "", "413.706.244", "361.912.538"],
    ["2", "Dự phòng rủi ro cho vay khách hàng", "9.7", "(4.849.736)", "(5.861.588)"],
    ["VIII", "Chứng khoán đầu tư", "10", "75.534.794", "59.474.604"],
    ["1", "Chứng khoán đầu tư sẵn sàng để bán", "10.1", "31.854.915", "9.739.539"],
    ["", "TỔNG TÀI SẢN", "", "607.875.185", "527.769.944"],
]

ACB_IS_2022_AS_NOTE = [
    ["", "", "Thuyết minh", "Năm 2022 Triệu VND", "Năm 2021 Triệu VND"],  # real
    ["1", "Thu nhập lãi và các khoản thu nhập tương tự", "22", "40.698.834", "33.713.810"],  # real
    ["2", "Chi phí lãi và các chi phí tương tự", "23", "(17.165.305)", "(14.769.018)"],  # real
    ["I", "Thu nhập lãi thuần", "", "23.533.529", "18.944.792"],
    ["II", "Lãi thuần từ hoạt động dịch vụ", "", "3.526.244", "2.893.963"],
    ["VIII", "Chi phí hoạt động", "31", "(11.605.045)", "(8.229.776)"],
    ["IX", "Lợi nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng", "",
     "17.185.022", "15.334.221"],
    ["X", "Chi phí dự phòng rủi ro tín dụng", "32", "(70.814)", "(3.336.109)"],
    ["XI", "Tổng lợi nhuận trước thuế", "", "17.114.208", "11.998.112"],
    ["XIII", "Lợi nhuận sau thuế", "", "13.688.193", "9.602.746"],
]

_SEG_AT = "Tại ngày 31 tháng 12 năm 2022"
_SEG_FOR = "Cho năm tài chính kết thúc ngày 31 tháng 12 năm 2022"
ACB_SEGMENT_NOTE = [  # real (page 98, 2022 block)
    ["", "Miền BắcTriệu VND", "Miền TrungTriệu VND", "Miền NamTriệu VND", "Loại trừTriệu VND",
     "Tổng cộngTriệu VND"],
    [_SEG_AT] * 6,
    ["Tài sản", "86.875.381", "64.568.858", "589.651.441", "(133.220.495)", "607.875.185"],
    ["Nợ phải trả", "84.784.267", "62.881.838", "531.315.549", "(129.545.132)", "549.436.522"],
    [_SEG_FOR] * 6,
    ["Kết quả kinh doanh bộ phận", "2.113.741", "1.687.022", "13.454.496", "(141.051)", "17.114.208"],
]


def vjc_view():
    return TableView(record("VJC23C", 1, 12, VJC_IS_2023, kind="income_statement"), 2023)


def acb_bs_view():
    return TableView(record("ACB22C", 2, 7, ACB_BS_2022, kind="balance_sheet"), 2022)


def acb_is_view(kind: str | None = "note"):
    return TableView(record("ACB22C", 5, 10, ACB_IS_2022_AS_NOTE, kind=kind), 2022)


def segment_view(kind: str | None = "income_statement"):
    return TableView(record("ACB22C", 40, 98, ACB_SEGMENT_NOTE, kind=kind), 2022)


def view_with_cell(
    cell: str,
    *,
    header: str = "VND",
    period: str = "Năm nay",
    context: str = "",
    report_year: int = 2023,
):
    """A one-data-row view whose value cell is (2, 3)."""
    rows = [
        ["", "", "Thuyết minh", period],
        ["", "", "Thuyết minh", header],
        ["01", "Doanh thu thuần", "", cell],
    ]
    return TableView(record("D", 1, 3, rows, kind="income_statement", context=context), report_year)


def row_of(view, text: str) -> int:
    """First data row whose code or label equals ``text``."""
    for idx in view.data_rows:
        if view.code(idx) == text or view.label(idx) == text:
            return idx
    raise LookupError(text)

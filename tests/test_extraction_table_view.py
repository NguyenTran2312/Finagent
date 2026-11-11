import dataclasses
import unittest
from decimal import Decimal as D

from src.calculation.units import Dimension
from src.extraction.table_view import (
    MalformedTable,
    TableView,
    enumerator_level,
    header_unit,
    normalize_label,
)
from tests.extraction_fixtures import (
    ACB_BS_2022,
    VJC_IS_2023,
    acb_bs_view,
    acb_is_view,
    record,
    row_of,
    segment_view,
    view_with_cell,
    vjc_view,
)


def view(rows, *, year=2023, kind="note", context=""):
    return TableView(record("D", 1, 3, rows, kind=kind, context=context), year)


class TableViewTests(unittest.TestCase):
    def test_vjc_real_header_unit_and_periods(self):
        v = TableView(record("D", 1, 12, VJC_IS_2023, kind="income_statement"), 2023)
        self.assertEqual(v.code_col, 0)
        self.assertEqual(v.period_columns(2023, "flow", "primary"), (3,))
        self.assertEqual(v.period_columns(2022, "flow", "fallback"), (4,))
        r = v.read(row_of(v, "11"), 3, Dimension.MONEY)
        self.assertEqual((r.status, r.value.value, r.source.scale), ("ok", D("-55819871829489"), D(1)))
        self.assertEqual(r.source.unit_text, "2023 VND")
        self.assertEqual(r.source.row_label, "Giá vốn hàng bán và dịch vụ cung cấp")
        self.assertEqual(r.source.raw_text, "(55.819.871.829.489)")
        self.assertEqual(r.source.chunk_id, "D:table:1")
        self.assertEqual(r.location.column_header, "Năm tài chính kết thúc ngày 31 tháng 12 năm | 2023 VND")

    def test_vjc_view_builder_rows(self):
        v = vjc_view()
        self.assertEqual(v.data_rows, tuple(range(2, len(VJC_IS_2023))))
        self.assertEqual(v.value_cols, (3, 4))
        self.assertEqual(v.locale, "vi")
        self.assertEqual(v.code(row_of(v, "23")), "23")
        self.assertEqual(v.label(row_of(v, "23")), "- Trong đó: Chi phí lãi vay")

    def test_joined_ocr_header_unit_and_period(self):  # Review Focus 1
        rows = [["", "", "Thuyết minh", "Tại ngày"], ["", "", "Thuyết minh", "31.12.2022Triệu VND"],
                ["I", "Tiền mặt, vàng bạc, đá quý", "4", "8.460.892"]]
        v = TableView(record("D", 1, 7, rows, kind="balance_sheet"), 2022)
        self.assertEqual(v.period_columns(2022, "closing", "primary"), (3,))
        self.assertEqual(v.read(2, 3, Dimension.MONEY).value.value, D("8460892000000"))

    def test_explicit_year_period_rule(self):  # ruling R1
        v = acb_bs_view()
        self.assertEqual(v.code_col, 0)
        self.assertEqual(v.period_columns(2022, "closing", "primary"), (3,))
        self.assertEqual(v.period_columns(2021, "closing", "fallback"), (4,))
        r = v.read(row_of(v, "TỔNG TÀI SẢN"), 4, Dimension.MONEY)
        self.assertEqual((r.status, r.value.value, r.source.unit_text), ("ok", D("527769944000000"),
                                                                         "31.12.2021 Triệu VND"))
        for opening in ("01.01.2022", "Số đầu năm"):
            rows = [["", "Thuyết minh", "Số cuối năm", opening], ["Tiền", "4", "1.000", "900"]]
            v = view(rows, year=2022, kind="balance_sheet")
            self.assertEqual(v.period_columns(2022, "closing", "primary"), (2,), opening)
            self.assertEqual(v.period_columns(2021, "closing", "fallback"), (3,), opening)

    def test_flow_periods(self):
        v = acb_is_view()
        self.assertEqual(v.period_columns(2022, "flow", "primary"), (3,))
        self.assertEqual(v.period_columns(2021, "flow", "fallback"), (4,))
        v = view([["Chỉ tiêu", "Năm nay", "Năm trước", "Giá trị"], ["Doanh thu", "1.000", "900", "5"]])
        self.assertEqual(v.period_columns(2023, "flow", "primary"), (1,))
        self.assertEqual(v.period_columns(2022, "flow", "fallback"), (2,))
        with self.assertRaises(ValueError):
            v.period_columns(2022, "flow", "primary")
        with self.assertRaises(ValueError):
            v.period_columns(2023, "stock", "primary")

    def test_header_unit_rules(self):
        self.assertEqual(header_unit(["Số cuối năm Triệu đồng"]).scale_multiplier, D("1E+6"))
        self.assertEqual(header_unit(["2023 VND", "Năm tài chính kết thúc ngày 31 tháng 12 năm"]).unit_status, "valid")
        self.assertEqual(header_unit(["Triệu VND", "VND"]).unit_status, "ambiguous")
        joined = header_unit(["31.12.2022Triệu VND", "Tại ngày"])
        self.assertEqual((joined.currency, joined.scale_multiplier, joined.source_text),
                         ("VND", D("1E+6"), "31.12.2022Triệu VND"))
        self.assertEqual(header_unit(["Thuyết minh", "Tại ngày"]).unit_status, "unknown")
        self.assertEqual(header_unit([]).unit_status, "unknown")

    def test_two_header_units_make_read_ambiguous(self):
        r = view_with_cell("1.000", header="Triệu VND", period="Năm nay VND").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.reason), ("unit", "ambiguous"))

    def test_context_declaration_unit(self):
        rows = [["Mã số", "Chỉ tiêu", "Năm nay", "Năm trước"], ["01", "Doanh thu thuần", "1.000", "900"]]
        v = view(rows, context="Đơn vị tính: triệu đồng")
        r = v.read(1, 2, Dimension.MONEY)
        self.assertEqual((r.status, r.value.value, r.source.scale), ("ok", D("1000000000"), D("1E+6")))
        self.assertEqual(r.source.unit_text, "Đơn vị tính: triệu đồng")
        r = view(rows).read(1, 2, Dimension.MONEY)
        self.assertEqual((r.status, r.reason), ("unit", "missing"))

    def test_header_unit_corpus_forms(self):  # §6.7 amendment, real headers
        for cell, scale in (("Tại 31/12/2025 VND", "1"), ("Số cuối nămVND", "1"),
                            ("Vào ngày 31/12/2023 Triệu VND", "1E+6"), ("31 tháng 12 năm 2025 VND", "1")):
            with self.subTest(cell=cell):
                unit = header_unit([cell])
                self.assertEqual((unit.unit_status, unit.currency, unit.scale_multiplier, unit.source_text),
                                 ("valid", "VND", D(scale), cell))
        self.assertEqual(header_unit(["Số cuối năm"]).unit_status, "unknown")

    def test_context_unit_corpus_forms(self):  # §6.7 amendment, real table contexts
        rows = [["Mã số", "TÀI SẢN", "Thuyết minh", "Số cuối năm", "Số đầu năm"],
                ["100", "TÀI SẢN NGẮN HẠN", "", "1.234", "1.000"]]
        cases = (
            ("Công ty Cổ phần Đầu tư Thế Giới Di Động B01-DN/HN BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT "
             "vào ngày 31 tháng 12 năm 2016 VND", "1", "VND"),
            ("Công ty Cổ phần Hoàng Anh Gia Lai B01-DN BẢNG CÂN ĐỐI KẾ TOÁN TỔNG HỢP "
             "ngày 31 tháng 12 năm 2017 Ngàn VND", "1E+3", "Ngàn VND"),
            ("BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT (tiếp theo) vào ngày 31 tháng 12 năm 2016 VND", "1", "VND"),
            ("BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH cho năm tài chính kết thúc ngày 31/12/2016 Triệu VND",
             "1E+6", "Triệu VND"),
            ("BÀNG CÂN ĐÔI KÊ TOÁN RIÊNG ngày 31 tháng 12 năm 2017 Đơn vị tinh: VND", "1",
             "Đơn vị tinh: VND"),
            ("BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG Tại ngày 31 tháng 12 năm 2016 (Tiếp theo) "
             "Đơn vị tính: đồng Việt Nam", "1", "Đơn vị tính: đồng Việt Nam"),
        )
        for context, scale, unit_text in cases:
            with self.subTest(context=context[-40:]):
                r = view(rows, kind="balance_sheet", context=context).read(1, 3, Dimension.MONEY)
                self.assertEqual((r.status, r.source.scale, r.source.unit_text), ("ok", D(scale), unit_text))

    def test_context_without_unit_stays_missing(self):
        rows = [["Mã số", "TÀI SẢN", "Số cuối năm"], ["100", "TÀI SẢN NGẮN HẠN", "1.234"]]
        for context in ("BẢNG CÂN ĐỐI KẾ TOÁN cho năm tài chính kết thúc ngày 31/12/2016 Công ty TNHH ABC",
                        "BẢNG CÂN ĐỐI KẾ TOÁN ngày 31 tháng 12 năm 2016",
                        "Phải thu của đơn vị khác 2016 Đầu tư vào đơn vị trực thuộc",
                        "BẢNG CÂN ĐỐI KẾ TOÁN VND năm 2016 Công ty"):
            with self.subTest(context=context):
                r = view(rows, kind="balance_sheet", context=context).read(1, 2, Dimension.MONEY)
                self.assertEqual((r.status, r.reason), ("unit", "missing"))

    def test_unit_declared_in_another_header_cell(self):  # real FPT 2018: "Đơn vị: VND" sits over the next column
        rows = [["", "TÀI SẢN", "Mã số", "Số cuối năm", "Đơn vị: VND Số đầu năm"],
                ["A.", "TÀI SẢN NGẮN HẠN", "100", "18.406.087.226.041", "16.059.937.820.645"],
                ["", "TỔNG CỘNG TÀI SẢN (270=100+200)", "270", "29.757.067.149.568", "23.884.267.040.722"]]
        r = view(rows, year=2018, kind="balance_sheet").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.source.scale, r.source.unit_text), ("ok", D(1), "Đơn vị: VND Số đầu năm"))

    def test_header_rows_below_data_carry_no_unit(self):  # real GAS 2018: signature rows ("Người lập") at the bottom
        rows = [["CHỈ TIÊU", "Mã số", "Năm nay", "Năm trước"],
                ["1. Doanh thu bán hàng", "01", "75.976.345.569.671", "64.830.157.981.142"],
                ["3. Doanh thu thuần (10=01-02)", "10", "75.611.546.239.412", "64.500.000.000.000"],
                ["Đặng Thị Hồng Yên Người lập biểu", "Trần Thị Hoàng Anh Kế toán",
                 "Dương Mạnh Sơn Tổng Giám đốc", "Dương Mạnh Sơn Tổng Giám đốc"]]
        v = view(rows, year=2018, kind="income_statement", context="BÁO CÁO KẾT QUẢ năm 2018 Đơn vị: VND")
        self.assertEqual(v._analyzer.header_indices, (0, 3))  # the signature row is taken for a header row
        r = v.read(2, 2, Dimension.MONEY)
        self.assertEqual((r.status, r.source.scale, r.source.unit_text), ("ok", D(1), "Đơn vị: VND"))

    def test_header_unit_beats_context_unit(self):
        rows = [["Mã số", "TÀI SẢN", "Số cuối năm Triệu VND"], ["100", "TÀI SẢN NGẮN HẠN", "1.234"]]
        r = view(rows, context="BẢNG CÂN ĐỐI KẾ TOÁN ngày 31 tháng 12 năm 2016 VND").read(1, 2, Dimension.MONEY)
        self.assertEqual((r.status, r.source.scale), ("ok", D("1E+6")))

    def test_code_column_rules(self):
        self.assertEqual(vjc_view().code_col, 0)  # "Mã số" header
        codes = [["Chỉ tiêu", "", "Năm nay"], ["Doanh thu", "01", "1.000"], ["Giá vốn", "02", "900"],
                 ["Lãi gộp", "10", "100"], ["Khác", "", "5"]]
        self.assertEqual(view(codes).code_col, 1)  # content: >= 3 codes, all code-like
        self.assertIsNone(view(codes[:3]).code_col)  # only 2 code-like values
        mostly_text = codes + [["X1", "abc", "1.100"], ["X2", "def", "1.200"]]
        self.assertIsNone(view(mostly_text).code_col)  # 3 of 5 code-like < 80%
        notes = [["", "", "Thuyết minh", "Năm nay"], ["01", "Doanh thu", "29", "1.000"],
                 ["02", "Giá vốn", "30", "900"], ["10", "Lãi gộp", "31", "100"]]
        v = view(notes)
        self.assertEqual(v.code_col, 0)  # "Thuyết minh" never qualifies
        self.assertEqual(v.value_cols, (3,))
        joined = [row[:2] + ["Thuyếtminh" if i == 0 else row[2]] + row[3:] for i, row in enumerate(notes)]
        self.assertEqual(view(joined).code_col, 0)  # OCR-joined header still excluded
        two = [["", "", "Năm nay"], ["01", "A1", "1.000"], ["02", "A2", "900"], ["10", "A3", "100"]]
        two = [[r[0], r[0] if i else "", r[1], r[2]] for i, r in enumerate(two)]
        self.assertIsNone(view(two).code_col)  # two qualifying columns

    def test_merged_code_and_note_header(self):  # real DCM 2020 balance sheet: one cell spans both columns
        rows = [["", "TÀI SẢN", "Mã số Thuyết minh", "Mã số Thuyết minh", "Số cuối năm", "Số đầu năm"],
                ["A.", "TÀI SẢN NGẮN HẠN", "100", "", "3.978.192.126.939", "4.250.036.958.838"],
                ["I.", "Tiền và các khoản tương đương tiền", "110", "4", "509.995.350.873", "471.173.323.541"],
                ["1.", "Tiền", "111", "", "209.995.350.873", "171.173.323.541"],
                ["", "TỔNG CỘNG TÀI SẢN (270=100+200)", "270", "", "8.717.480.691.611", "10.172.594.752.746"]]
        v = view(rows, kind="balance_sheet")
        self.assertEqual((v.code_col, v.value_cols), (2, (4, 5)))
        self.assertEqual(v.code(4), "270")
        # Two merged columns with equally many code-like cells stay undecided.
        tie = [r[:3] + [r[2]] + r[4:] for r in rows]
        self.assertIsNone(view(tie, kind="balance_sheet").code_col)

    def test_note_reference_column_by_content(self):  # real HUT/SNZ: unlabelled note column "5.1", "5.13"
        rows = [["", "TÀI SẢN", "", "31/12/2019 VND", "01/01/2019 VND"],
                ["100", "TÀI SẢN NGẮN HẠN", "", "2.311.130.883.461", "2.117.560.373.052"],
                ["110", "Tiền và các khoản tương đương tiền", "5.1", "146.076.986.910", "142.526.664.337"],
                ["130", "Các khoản phải thu ngắn hạn", "5.13", "115.602.160.741", "45.772.170.869"],
                ["140", "Hàng tồn kho", "5.2", "1.234.567", "2.345.678"]]
        v = view(rows, year=2019, kind="balance_sheet")
        self.assertEqual((v.value_cols, v.locale), ((3, 4), "vi"))
        r = v.read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.value.value), ("ok", D("146076986910")))
        # A column with a period header is a value column even when its cells look like note numbers.
        decimals = [r[:2] + [r[3]] for r in rows]
        decimals = [decimals[0]] + [[a, b, c] for a, b, c in
                                    (("100", "Tài sản", "5.1"), ("110", "Tiền", "5.13"), ("130", "Phải thu", "5.2"),
                                     ("140", "Hàng tồn kho", "6.1"))]
        self.assertEqual(view(decimals, year=2019, kind="balance_sheet").value_cols, (2,))
        # Real DPM 2021: a row-number column "1.", "2.", "18.1", "18.2" left of the labels.
        numbered = [["", "CHỈ TIÊU", "Mã số", "Thuyết minh", "Năm nay", "Năm trước"],
                    ["1.", "Doanh thu bán hàng", "01", "25", "12.786.141.193.352", "7.761.818.604.563"],
                    ["2.", "Giá vốn hàng bán", "11", "26", "9.100.000.000.000", "6.000.000.000.000"],
                    ["18.", "Lợi nhuận sau thuế", "60", "", "3.171.516.300.559", "1.000.000.000.000"],
                    ["18.1", "Lợi nhuận sau thuế của Công ty mẹ", "61", "", "3.000.000.000.000", "900.000.000.000"],
                    ["18.2", "Lợi nhuận sau thuế của cổ đông không kiểm soát", "62", "", "171.516.300.559", "1"]]
        v = view(numbered, year=2021, kind="income_statement")
        self.assertEqual((v.code_col, v.value_cols, v.locale), (2, (4, 5), "vi"))

    def test_normalize_label_and_levels(self):
        for raw in ("TỔNG CỘNG TÀI SẢN (270=100+200)", "Tổng cộng tài sản (270 = 100 + 200)",
                    "TỔNG CỘNG TÀI SẢN (270 ="):  # OCR-truncated formula
            self.assertEqual(normalize_label(raw), "tong cong tai san", raw)  # trailing formula dropped
        self.assertEqual(normalize_label("Lợi nhuận thuần từ hoạt động kinh doanh (30 = 20 + (21 - 22) - (25 + 26))"),
                         "loi nhuan thuan tu hoat dong kinh doanh")
        self.assertEqual(normalize_label("Vay (ngắn hạn)"), "vay ngan han")  # ordinary parentheses kept
        self.assertEqual(normalize_label("VI Cho vay khách hàng"), "cho vay khach hang")
        self.assertEqual([enumerator_level(c) for c in ("A", "VI", "X", "2", "b", "6.1")],
                         ["top", "section", "section", "item", "sub_item", None])
        self.assertEqual(normalize_label("12. Tiền"), "tien")
        self.assertEqual(normalize_label("Tiền gửi"), "tien gui")
        self.assertEqual(normalize_label("I II Tiền"), "ii tien")  # one token only
        self.assertEqual(normalize_label("Xí nghiệp A"), "xi nghiep a")  # raw token decides
        self.assertEqual(normalize_label("Vị thế ngoại tệ"), "vi the ngoai te")
        self.assertEqual(normalize_label("Á Châu"), "a chau")
        self.assertEqual(normalize_label("vi phạm"), "vi pham")  # lower-case roman is not a section
        for raw in ("VI. Cho vay", "1. Cho vay", "a) Cho vay", "(b) Cho vay", "1.Cho vay", "x Cho vay"):
            self.assertEqual(normalize_label(raw), "cho vay", raw)
        self.assertEqual(normalize_label("XXXI Cho vay"), "xxxi cho vay")
        self.assertEqual(normalize_label("123 Cho vay"), "123 cho vay")
        self.assertEqual(normalize_label("A"), "")
        self.assertEqual([enumerator_level(c) for c in ("I", "V", "XXX", "XXXI", "123", "")],
                         ["section", "section", "section", None, None, None])
        v = acb_bs_view()
        self.assertEqual(v.norm_label(row_of(v, "VI")), "cho vay khach hang")
        self.assertEqual(v.label(row_of(v, "VI")), "Cho vay khách hàng")

    def test_locale(self):
        self.assertEqual(acb_bs_view().locale, "vi")
        en = [["Chỉ tiêu", "Năm nay"], ["Doanh thu", "1,234,567"], ["Giá vốn", "1,000"]]
        self.assertEqual(view(en).locale, "en")
        mixed = [["Chỉ tiêu", "Năm nay VND"], ["Doanh thu", "1,234,567"], ["Giá vốn", "8.460.892"]]
        v = view(mixed)
        self.assertEqual(v.locale, "mixed")
        r = v.read(1, 1, Dimension.MONEY)
        self.assertEqual((r.status, r.reason, r.value), ("unparsed", "number_locale", None))
        neither = [["Chỉ tiêu", "Năm nay"], ["Doanh thu", "1.000"], ["Giá vốn", "1,000"]]
        self.assertEqual(view(neither).locale, "vi")

    def test_blank_and_unit_problems(self):
        r = view_with_cell("-").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.reason, r.value, r.location.raw_text), ("blank", None, None, "-"))
        r = view_with_cell("12,5", header="%").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.reason), ("unit", "kind_mismatch"))
        r = view_with_cell("1.000", header="USD").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.reason), ("unit", "currency"))
        r = view_with_cell("1.500", header="Triệu VND").read(2, 3, Dimension.MONEY_PER_SHARE)
        self.assertEqual((r.status, r.reason), ("unit", "kind_mismatch"))
        r = view_with_cell("1.500", header="VND").read(2, 3, Dimension.MONEY_PER_SHARE)
        self.assertEqual((r.status, r.value.value), ("ok", D(1500)))
        r = view_with_cell("1.000", header="").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.reason), ("unit", "missing"))
        r = view_with_cell("12%").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.reason), ("unit", "kind_mismatch"))

    def test_footnote_cell_is_unparsed(self):  # Review Focus 2
        r = view_with_cell("1.234.567 (*)").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.reason), ("unparsed", "number_unparsed"))
        r = view_with_cell("1.234.567 2.345.678").read(2, 3, Dimension.MONEY)
        self.assertEqual((r.status, r.reason, r.value), ("unparsed", "number_unparsed", None))

    def test_sole_unlabelled_column_not_used(self):  # Review Focus 4
        rows = [["Chỉ tiêu", "Giá trị"], ["Doanh thu thuần", "1.000"]]
        v = TableView(record("D", 1, 3, rows, kind="note"), 2023)
        self.assertEqual(v.period_columns(2023, "flow", "primary"), ())
        self.assertEqual(v.period_columns(2023, "closing", "primary"), ())
        self.assertEqual(v.period_columns(2022, "flow", "fallback"), ())

    def test_multi_block_header_fails_safe(self):
        rows = [["Chỉ tiêu", "Năm 2022", "Năm 2022"], ["Doanh thu", "1.000", "2.000"],
                ["Chỉ tiêu", "Năm 2021", "Năm 2021"], ["Doanh thu", "900", "1.900"]]
        v = view(rows, year=2022)
        self.assertEqual(v.period_columns(2022, "flow", "primary"), ())
        self.assertEqual(v.period_columns(2021, "flow", "fallback"), ())
        seg = segment_view()  # real ACB segment note: header blocks between data rows
        self.assertEqual(seg.period_columns(2022, "closing", "primary"), ())

    def test_repeated_header_block_still_works(self):
        head = ["Chỉ tiêu", "Năm nay", "Năm trước"]
        rows = [head, ["Doanh thu", "1.000", "900"], list(head), ["Giá vốn", "800", "700"]]
        v = view(rows)
        self.assertEqual(v.period_columns(2023, "flow", "primary"), (1,))
        self.assertEqual(v.period_columns(2022, "flow", "fallback"), (2,))

    def test_mid_table_rows_without_period_cells_do_not_trigger(self):
        section_with_year = ["Số dư tại ngày 31/12/2022", "", ""]
        rows = [["Chỉ tiêu", "Năm nay", "Năm trước"], ["Doanh thu", "1.000", "900"], section_with_year,
                ["Giá vốn", "800", "700"]]
        self.assertEqual(view(rows).period_columns(2023, "flow", "primary"), (1,))
        label = "LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG ĐẦU TƯ"  # colspan-repeated section label (VJC 2023 cash flow)
        rows = [["Mã số", "Chỉ tiêu", "Năm nay", "Năm trước"], ["01", "Lợi nhuận", "1.000", "900"],
                [label] * 4, ["21", "Tiền chi mua sắm", "(100)", "(90)"]]
        v = view(rows)
        self.assertEqual(v.period_columns(2023, "flow", "primary"), (2,))
        self.assertEqual(v.period_columns(2022, "flow", "fallback"), (3,))

    def test_continuation_table_without_header(self):  # spec §6.9
        v = view([["01", "Doanh thu thuần", "1.000", "900"]])
        self.assertEqual(v.period_columns(2023, "flow", "primary"), ())
        self.assertEqual(v.read(0, 2, Dimension.MONEY).reason, "missing")

    def test_restatement_marker(self):
        rows = [["Chỉ tiêu", "Năm 2023", "Năm 2022 (trình bày lại)"], ["Doanh thu", "1.000", "900"]]
        v = view(rows)
        self.assertEqual((v.restatement_marker(1), v.restatement_marker(2)), (False, True))
        self.assertTrue(view(rows, context="Restated figures").restatement_marker(1))
        self.assertTrue(view(rows, context="số liệu đã được điều chỉnh lại").restatement_marker(1))

    def test_tokens(self):
        tokens = acb_bs_view().tokens()
        self.assertIn("vay", tokens)
        self.assertIn("tong", tokens)
        self.assertNotIn("vi", tokens)

    def test_malformed(self):
        cases = [
            (record("D", 1, 3, ACB_BS_2022, kind=None, chunk=False), "no_chunk"),
            (dataclasses.replace(record("D", 1, 3, [], kind=None), rows_json="not json"), "json"),
            (dataclasses.replace(record("D", 1, 3, [], kind=None), rows_json='[["a"], 1]'), "shape"),
            (dataclasses.replace(record("D", 1, 3, [], kind=None), rows_json='[["a", 2]]'), "shape"),
            (dataclasses.replace(record("D", 1, 3, [], kind=None), rows_json='{"a": 1}'), "shape"),
        ]
        for rec, reason in cases:
            with self.assertRaises(MalformedTable) as ctx:
                TableView(rec, 2022)
            self.assertEqual((ctx.exception.table_key, ctx.exception.reason), ("D:1", reason))
            self.assertIsInstance(ctx.exception, ValueError)


if __name__ == "__main__":
    unittest.main()

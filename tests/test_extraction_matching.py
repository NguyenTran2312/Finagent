import dataclasses
import unittest

from src.calculation.metric_tree import MetricTree
from src.calculation.units import Dimension
from src.extraction.matching import RowHit, anchor_kinds, find_rows, label_keys
from src.extraction.table_view import TableView
from src.indexing.table_semantics import StatementKind
from tests.extraction_fixtures import (
    ACB_IS_2022_AS_NOTE,
    VJC_IS_2023,
    acb_bs_view,
    acb_is_view,
    record,
    row_of,
    segment_view,
    vjc_view,
)

tree = MetricTree.load()
bank = tree.leaves("bank")
corp = tree.leaves("corporate")

_BANK_BS_HEADER = [
    ["", "", "Thuyết minh", "Tại ngày", "Tại ngày"],
    ["", "", "Thuyết minh", "31.12.2022 Triệu VND", "31.12.2021 Triệu VND"],
]


def view(rows, *, kind, table_id=1, page=3, year=2023):
    return TableView(record("D", table_id, page, rows, kind=kind), year)


def labels_of(hits):
    return [h.view.label(h.row_idx) for h in hits]


def coded_is(kind):
    """A corporate income statement (codes 01-20) carrying a 'Mã số' column."""
    return view(
        [
            ["Mã số", "Chỉ tiêu", "Năm nay"],
            ["01", "Doanh thu bán hàng và cung cấp dịch vụ", "100"],
            ["02", "Các khoản giảm trừ doanh thu", "(5)"],
            ["10", "Doanh thu thuần về bán hàng và cung cấp dịch vụ", "95"],
            ["11", "Giá vốn hàng bán", "(60)"],
            ["20", "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "35"],
        ],
        kind=kind,
    )


def coded_cf(kind):
    """A corporate cash flow statement carrying only its two leaf labels."""
    return view(
        [
            ["Mã số", "Chỉ tiêu", "Năm nay"],
            ["01", "Lợi nhuận trước thuế", "50"],
            ["02", "Khấu hao TSCĐ", "10"],
            ["03", "Các khoản dự phòng", "1"],
            ["20", "Lưu chuyển tiền thuần từ hoạt động kinh doanh", "61"],
        ],
        kind=kind,
        table_id=2,
    )


class LabelKeysTests(unittest.TestCase):
    def test_folded_label_and_aliases(self):
        keys = label_keys(tree.leaf("total_assets", "bank"))
        self.assertIn("tong tai san", keys)
        self.assertIn("tong no phai tra va von chu so huu", keys)
        self.assertNotIn("Tổng tài sản", keys)


class CorporateMatchingTests(unittest.TestCase):
    def test_corporate_codes(self):
        hits = find_rows(tree.leaf("interest_expense", "corporate"), [vjc_view()], corp)
        self.assertEqual([(h.method, h.view.code(h.row_idx)) for h in hits], [("code", "23")])
        self.assertEqual(hits[0].warnings, ())

    def test_code_compared_as_integer(self):
        v = view(
            [
                ["Mã số", "Chỉ tiêu", "Năm nay"],
                ["1", "Doanh thu bán hàng", "100"],
                ["2", "Các khoản giảm trừ", "(5)"],
                ["10", "Doanh thu thuần", "95"],
            ],
            kind="income_statement",
        )
        hits = find_rows(tree.leaf("gross_revenue", "corporate"), [v], corp)
        self.assertEqual([(h.method, h.view.code(h.row_idx)) for h in hits], [("code", "1")])

    def test_coded_table_ignores_labels(self):
        # The code column decides; an exact label on a row with another code is not a hit.
        v = view(
            [
                ["Mã số", "Chỉ tiêu", "Năm nay"],
                ["01", "Doanh thu bán hàng", "100"],
                ["10", "Doanh thu thuần", "95"],
                ["11", "Chi phí lãi vay", "(3)"],
            ],
            kind="income_statement",
        )
        self.assertEqual(find_rows(tree.leaf("interest_expense", "corporate"), [v], corp), [])

    def test_no_code_column_falls_back_to_label(self):
        v = view(
            [
                ["Chỉ tiêu", "Năm nay", "Năm trước"],
                ["Doanh thu bán hàng và cung cấp dịch vụ", "100", "90"],
                ["Doanh thu thuần về bán hàng và cung cấp dịch vụ", "95", "85"],
                ["Giá vốn hàng bán", "(60)", "(50)"],
            ],
            kind="income_statement",
        )
        self.assertIsNone(v.code_col)
        hits = find_rows(tree.leaf("net_revenue", "corporate"), [v], corp)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].method, "label")
        self.assertEqual(hits[0].warnings, ("matched_by_label",))
        self.assertEqual(hits[0].view.label(hits[0].row_idx), "Doanh thu thuần về bán hàng và cung cấp dịch vụ")

    def test_no_substring_matching(self):
        rows = [
            ["Chỉ tiêu", "Số cuối năm"],
            ["Dự phòng giảm giá hàng tồn kho", "(7)"],
            ["Hàng tồn kho", "70"],
            ["Hàng tồn kho đi đường", "3"],
        ]
        v = view(rows, kind="balance_sheet")
        hits = find_rows(tree.leaf("inventories", "corporate"), [v], corp)
        self.assertEqual(labels_of(hits), ["Hàng tồn kho"])
        no_exact = view([r for r in rows if r[0] != "Hàng tồn kho"], kind="balance_sheet")
        self.assertEqual(find_rows(tree.leaf("inventories", "corporate"), [no_exact], corp), [])

    def test_wrong_statement_kind_not_searched(self):
        hits = find_rows(tree.leaf("interest_expense", "corporate"), [vjc_view()], corp)
        self.assertEqual(len(hits), 1)
        bs = TableView(record("VJC23C", 1, 12, VJC_IS_2023, kind="balance_sheet"), 2023)
        self.assertEqual(find_rows(tree.leaf("interest_expense", "corporate"), [bs], corp), [])

    def test_note_leaf_exact_label(self):
        rows = [
            ["", "Số cuối năm", "Số đầu năm"],
            ["Chi phí lãi vay", "5", "4"],
            ["Chi phí lãi vay phải trả", "12", "9"],
            ["Chi phí phải trả khác", "3", "2"],
        ]
        note = view(rows, kind="note")
        leaf = tree.leaf("accrued_interest_payable", "corporate")
        hits = find_rows(leaf, [note], corp)
        self.assertEqual([(h.method, h.warnings) for h in hits], [("label", ())])
        self.assertEqual(labels_of(hits), ["Chi phí lãi vay phải trả"])
        unknown = TableView(record("D", 1, 3, rows, kind="unknown"), 2023)
        self.assertEqual(len(find_rows(leaf, [unknown], corp)), 1)
        # Note leaves never search statement tables.
        statement = TableView(record("D", 1, 3, rows, kind="balance_sheet"), 2023)
        self.assertEqual(find_rows(leaf, [statement], corp), [])


class BankMatchingTests(unittest.TestCase):
    def test_bank_section_and_children(self):
        v = acb_bs_view()
        for leaf_id, value in [("customer_loans_section", "408.856.508"),
                               ("customer_loans_gross", "413.706.244"),
                               ("customer_loan_provision", "(4.849.736)")]:
            hits = find_rows(tree.leaf(leaf_id, "bank"), [v], bank)
            self.assertEqual(len(hits), 1, leaf_id)
            self.assertEqual(hits[0].method, "section_position")
            self.assertEqual(v.read(hits[0].row_idx, 3, Dimension.MONEY).location.raw_text, value)
        # the decoy "4 Cho vay khách hàng" under III is never hit
        decoy = 8
        self.assertEqual((v.code(decoy), v.norm_label(decoy)), ("4", "cho vay khach hang"))

    def test_section_block_ends_at_total_marker(self):
        rows = _BANK_BS_HEADER + [
            ["A", "TÀI SẢN", "", "", ""],
            ["I", "Tiền mặt, vàng bạc, đá quý", "4", "8.000", "7.000"],
            ["II", "Cho vay khách hàng", "9", "408.000", "356.000"],
            ["1", "Cho vay khách hàng", "", "413.000", "361.000"],
            ["", "Cộng", "", "413.000", "361.000"],
            ["2", "Dự phòng rủi ro cho vay khách hàng", "9.7", "(5.000)", "(5.000)"],
        ]
        v = view(rows, kind="balance_sheet", year=2022)
        self.assertEqual(v.code_col, 0)
        self.assertEqual(len(find_rows(tree.leaf("customer_loans_gross", "bank"), [v], bank)), 1)
        self.assertEqual(find_rows(tree.leaf("customer_loan_provision", "bank"), [v], bank), [])

    def test_block_end_marker_is_a_label_prefix(self):
        # A total marker inside a label ("... cộng dồn") does not end the block; a leading one does.
        body = [
            ["A", "TÀI SẢN", "", "", ""],
            ["I", "Tiền mặt, vàng bạc, đá quý", "4", "8.000", "7.000"],
            ["II", "Cho vay khách hàng", "9", "408.000", "356.000"],
            ["1", "Cho vay khách hàng", "", "413.000", "361.000"],
            ["", "Lãi cộng dồn", "", "1.000", "1.000"],
            ["2", "Dự phòng rủi ro cho vay khách hàng", "9.7", "(5.000)", "(5.000)"],
        ]
        v = view(_BANK_BS_HEADER + body, kind="balance_sheet", year=2022)
        self.assertEqual(v.code_col, 0)
        hits = find_rows(tree.leaf("customer_loan_provision", "bank"), [v], bank)
        self.assertEqual(labels_of(hits), ["Dự phòng rủi ro cho vay khách hàng"])
        body[4][1] = "Tổng cộng cho vay"
        v = view(_BANK_BS_HEADER + body, kind="balance_sheet", year=2022)
        self.assertEqual(find_rows(tree.leaf("customer_loan_provision", "bank"), [v], bank), [])

    def test_section_needs_child_after_section_row(self):
        # A section row with the child's label but no child row after it is not the section (R2).
        rows = _BANK_BS_HEADER + [
            ["A", "TÀI SẢN", "", "", ""],
            ["I", "Tiền mặt, vàng bạc, đá quý", "4", "8.000", "7.000"],
            ["II", "Cho vay khách hàng", "9", "408.000", "356.000"],
            ["1", "Cho vay khác", "", "1.000", "1.000"],
            ["III", "Chứng khoán đầu tư", "10", "75.000", "59.000"],
        ]
        v = view(rows, kind="balance_sheet", year=2022)
        self.assertEqual(v.code_col, 0)
        for leaf_id in ("customer_loans_section", "customer_loans_gross", "customer_loan_provision"):
            self.assertEqual(find_rows(tree.leaf(leaf_id, "bank"), [v], bank), [], leaf_id)

    def test_two_qualifying_sections_both_returned(self):
        rows = _BANK_BS_HEADER + [
            ["I", "Cho vay khách hàng", "9", "408.000", "356.000"],
            ["1", "Cho vay khách hàng", "", "413.000", "361.000"],
            ["II", "Cho vay khách hàng thuần", "9", "500.000", "400.000"],
            ["1", "Dự phòng rủi ro cho vay khách hàng", "", "(5.000)", "(5.000)"],
            ["III", "Chứng khoán đầu tư", "10", "75.000", "59.000"],
        ]
        v = view(rows, kind="balance_sheet", year=2022)
        self.assertEqual(v.code_col, 0)
        hits = find_rows(tree.leaf("customer_loans_section", "bank"), [v], bank)
        self.assertEqual([h.view.code(h.row_idx) for h in hits], ["I", "II"])

    def test_section_without_enumerators(self):
        rows = [
            ["", "Thuyết minh", "31.12.2022 Triệu VND", "31.12.2021 Triệu VND"],
            ["Tiền mặt, vàng bạc, đá quý", "4", "8.460.892", "7.509.877"],
            ["Cho vay khách hàng", "9", "408.856.508", "356.050.950"],
            ["Cho vay khách hàng", "", "413.706.244", "361.912.538"],
            ["Dự phòng rủi ro cho vay khách hàng", "9.7", "(4.849.736)", "(5.861.588)"],
            ["Chứng khoán đầu tư", "10", "75.534.794", "59.474.604"],
        ]
        v = view(rows, kind="balance_sheet", year=2022)
        self.assertIsNone(v.code_col)
        for leaf_id, value in [("customer_loans_section", "408.856.508"),
                               ("customer_loans_gross", "413.706.244"),
                               ("customer_loan_provision", "(4.849.736)")]:
            hits = find_rows(tree.leaf(leaf_id, "bank"), [v], bank)
            self.assertEqual(len(hits), 1, leaf_id)
            self.assertEqual(hits[0].method, "section_position")
            self.assertEqual(v.read(hits[0].row_idx, 2, Dimension.MONEY).location.raw_text, value)

        other_shapes = {
            "single X": [rows[0], rows[1], rows[2], rows[4], rows[5]],
            "X X without other child": [rows[0], rows[1], rows[2], rows[3], rows[5]],
            "other child beyond 3 rows": [rows[0], rows[2], rows[3], rows[1], rows[5], rows[1], rows[4]],
        }
        for name, shape in other_shapes.items():
            sv = view(shape, kind="balance_sheet", year=2022)
            self.assertIsNone(sv.code_col, name)
            for leaf_id in ("customer_loans_section", "customer_loans_gross", "customer_loan_provision"):
                self.assertEqual(find_rows(tree.leaf(leaf_id, "bank"), [sv], bank), [], f"{name}: {leaf_id}")

    def test_total_assets_two_labels(self):
        rows = _BANK_BS_HEADER + [
            ["A", "TÀI SẢN", "", "", ""],
            ["I", "Tiền mặt, vàng bạc, đá quý", "4", "8.000", "7.000"],
            ["II", "Chứng khoán đầu tư", "10", "75.000", "59.000"],
            ["", "TỔNG TÀI SẢN", "", "83.000", "66.000"],
            ["B", "NGUỒN VỐN", "", "", ""],
            ["I", "Tiền gửi của khách hàng", "15", "80.000", "60.000"],
            ["", "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU", "", "83.000", "66.000"],
        ]
        v = view(rows, kind="balance_sheet", year=2022)
        self.assertEqual(v.code_col, 0)
        hits = find_rows(tree.leaf("total_assets", "bank"), [v], bank)
        self.assertEqual(labels_of(hits), ["TỔNG TÀI SẢN", "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU"])
        self.assertEqual({h.method for h in hits}, {"label"})

    def test_heading_row_is_never_a_label_hit(self):
        # "B NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU" equals a total_assets alias but is a blank top-level heading.
        rows = _BANK_BS_HEADER + [
            ["A", "TÀI SẢN", "", "", ""],
            ["I", "Tiền mặt, vàng bạc, đá quý", "4", "8.460.892", "7.509.877"],
            ["", "TỔNG TÀI SẢN", "", "607.875.185", "527.769.944"],
            ["B", "NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU", "", "", ""],
            ["I", "Tiền gửi của khách hàng", "15", "80.000", "60.000"],
            ["", "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU", "", "607.875.185", "527.769.944"],
        ]
        v = view(rows, kind="balance_sheet", year=2022)
        self.assertEqual(v.code_col, 0)
        hits = find_rows(tree.leaf("total_assets", "bank"), [v], bank)
        self.assertEqual(labels_of(hits), ["TỔNG TÀI SẢN", "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU"])
        # A top-level row that carries a value is not a heading.
        rows[5] = ["B", "NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU", "", "607.875.185", ""]
        v = view(rows, kind="balance_sheet", year=2022)
        self.assertEqual(len(find_rows(tree.leaf("total_assets", "bank"), [v], bank)), 3)

    def test_unknown_parent_raises(self):
        orphan = dataclasses.replace(tree.leaf("customer_loans_gross", "bank"), parent="no_such_section")
        self.assertRaises(ValueError, find_rows, orphan, [acb_bs_view()], bank)

    def test_bank_label_leaf(self):
        hits = find_rows(tree.leaf("net_interest_income", "bank"), [acb_is_view("income_statement")], bank)
        self.assertEqual([(h.method, h.view.code(h.row_idx)) for h in hits], [("label", "I")])


class CandidateTableTests(unittest.TestCase):
    def test_anchor_rule_bank_income(self):
        views = [segment_view(kind="income_statement"), acb_is_view(kind="note")]
        hits = find_rows(tree.leaf("net_interest_income", "bank"), views, bank)
        self.assertEqual([h.view.record.table_id for h in hits], [acb_is_view().record.table_id])

    def test_dcm_total_assets_from_balance_sheet_not_subsidiary_note(self):  # real DCM 2020 wrong value
        bs = view([["", "TÀI SẢN", "Mã số Thuyết minh", "Mã số Thuyết minh", "Số cuối năm", "Số đầu năm"],
                   ["A.", "TÀI SẢN NGẮN HẠN", "100", "", "3.978.192.126.939", "4.250.036.958.838"],
                   ["I.", "Tiền và các khoản tương đương tiền", "110", "4", "509.995.350.873", "471.173.323.541"],
                   ["B.", "TÀI SẢN DÀI HẠN", "200", "", "4.739.288.564.672", "5.922.557.793.908"],
                   ["", "TỔNG CỘNG TÀI SẢN (270=100+200)", "270", "", "8.717.480.691.611", "10.172.594.752.746"]],
                  kind="balance_sheet", table_id=5, page=10, year=2020)
        nci_note = view([["", "Số cuối nămVND", "Số đầu nămVND"],
                         ["Tổng tài sản", "50.688.973.610", "67.941.730.594"],
                         ["Tổng nợ phải trả", "(24.341.547.506)", "(40.663.837.246)"],
                         ["Tài sản thuần", "26.347.426.104", "27.277.893.348"],
                         ["Vốn điều lệ", "23.503.490.000", "23.503.490.000"]],
                        kind="note", table_id=37, page=37, year=2020)
        hits = find_rows(tree.leaf("total_assets", "corporate"), [bs, nci_note], corp)
        self.assertEqual([(h.view.record.table_id, h.method) for h in hits], [(5, "code")])

    def test_formula_suffix_label_matches(self):
        bs = view([["Chỉ tiêu", "Số cuối năm"], ["TỔNG CỘNG TÀI SẢN (270=100+200)", "8.717"]], kind="balance_sheet")
        self.assertEqual(labels_of(find_rows(tree.leaf("total_assets", "corporate"), [bs], corp)),
                         ["TỔNG CỘNG TÀI SẢN (270=100+200)"])

    def test_tagged_hit_suppresses_anchor_tables(self):
        tagged = TableView(record("ACB22C", 9, 20, ACB_IS_2022_AS_NOTE, kind="income_statement"), 2022)
        views = [acb_is_view(kind="note"), tagged]
        hits = find_rows(tree.leaf("net_interest_income", "bank"), views, bank)
        self.assertEqual([h.view.record.table_id for h in hits], [9])

    def test_anchor_threshold_and_cross_kind(self):
        # Two bank income-statement labels (of nine leaves) do not anchor.
        two = view(
            [
                ["", "Năm 2022 Triệu VND"],
                ["Thu nhập lãi thuần", "23.533.529"],
                ["Chi phí hoạt động", "(11.605.045)"],
                ["Thu nhập khác", "5"],
            ],
            kind="note",
            year=2022,
        )
        self.assertEqual(anchor_kinds(two, bank), frozenset())
        self.assertEqual(find_rows(tree.leaf("net_interest_income", "bank"), [two], bank), [])

        # Corporate cash flow has two leaves, so its two labels anchor.
        cf = coded_cf("unknown")
        self.assertEqual(anchor_kinds(cf, corp), frozenset({StatementKind.CASH_FLOW}))
        hits = find_rows(tree.leaf("cfo", "corporate"), [cf], corp)
        self.assertEqual([(h.method, h.view.code(h.row_idx)) for h in hits], [("code", "20")])

        # An income statement tagged unknown has codes 02 and 20 but anchors by labels only.
        inc = coded_is("unknown")
        self.assertEqual(anchor_kinds(inc, corp), frozenset({StatementKind.INCOME_STATEMENT}))
        self.assertEqual(find_rows(tree.leaf("cfo", "corporate"), [inc], corp), [])
        self.assertEqual(find_rows(tree.leaf("depreciation", "corporate"), [inc], corp), [])
        self.assertEqual(len(find_rows(tree.leaf("net_revenue", "corporate"), [inc], corp)), 1)

        # A table anchoring two kinds is a candidate for neither.
        both = view(
            [
                ["Mã số", "Chỉ tiêu", "Năm nay"],
                ["01", "Doanh thu bán hàng và cung cấp dịch vụ", "100"],
                ["10", "Doanh thu thuần về bán hàng và cung cấp dịch vụ", "95"],
                ["11", "Giá vốn hàng bán", "(60)"],
                ["02", "Khấu hao TSCĐ", "10"],
                ["20", "Lưu chuyển tiền thuần từ hoạt động kinh doanh", "61"],
            ],
            kind="note",
        )
        self.assertEqual(
            anchor_kinds(both, corp), frozenset({StatementKind.INCOME_STATEMENT, StatementKind.CASH_FLOW})
        )
        self.assertEqual(find_rows(tree.leaf("cfo", "corporate"), [both], corp), [])
        self.assertEqual(find_rows(tree.leaf("net_revenue", "corporate"), [both], corp), [])

    def test_anchor_needs_distinct_labels(self):
        loans = view(
            [
                ["", "31.12.2022 Triệu VND"],
                ["Cho vay khách hàng", "413.706.244"],
                ["Dự phòng rủi ro cho vay khách hàng", "(4.849.736)"],
            ],
            kind="note",
            year=2022,
        )
        self.assertEqual(anchor_kinds(loans, bank), frozenset())
        self.assertEqual(find_rows(tree.leaf("customer_loan_provision", "bank"), [loans], bank), [])
        # Two leaves sharing one label: three matched leaves but two labels do not anchor.
        shared = tuple(
            dataclasses.replace(leaf, aliases=(*leaf.aliases, "cho vay khách hàng"))
            if leaf.id == "customer_loans_section" else leaf
            for leaf in bank
        )
        self.assertEqual(anchor_kinds(loans, shared), frozenset())
        three = view(
            [
                ["", "31.12.2022 Triệu VND"],
                ["Cho vay khách hàng", "413.706.244"],
                ["Dự phòng rủi ro cho vay khách hàng", "(4.849.736)"],
                ["Tiền gửi của khách hàng", "449.000.000"],
            ],
            kind="note",
            year=2022,
        )
        self.assertEqual(anchor_kinds(three, shared), frozenset({StatementKind.BALANCE_SHEET}))

    def test_untagged_table_counts_as_unknown(self):
        rows = [
            ["", "Số cuối năm", "Số đầu năm"],
            ["Chi phí lãi vay phải trả", "12", "9"],
        ]
        untagged = view(rows, kind=None)
        hits = find_rows(tree.leaf("accrued_interest_payable", "corporate"), [untagged], corp)
        self.assertEqual(labels_of(hits), ["Chi phí lãi vay phải trả"])
        anchored = acb_is_view(kind=None)
        hits = find_rows(tree.leaf("net_interest_income", "bank"), [anchored], bank)
        self.assertEqual([h.view.code(h.row_idx) for h in hits], ["I"])

    def test_anchor_only_note_or_unknown_tables(self):
        # A balance sheet carrying income statement labels is not an anchor candidate.
        mis_tagged = coded_is("balance_sheet")
        self.assertEqual(find_rows(tree.leaf("net_revenue", "corporate"), [mis_tagged], corp), [])

    def test_hits_ordered_by_page_table_row(self):
        late = TableView(record("D", 1, 9, VJC_IS_2023, kind="income_statement"), 2023)
        early_hi = TableView(record("D", 7, 4, VJC_IS_2023, kind="income_statement"), 2023)
        early_lo = TableView(record("D", 3, 4, VJC_IS_2023, kind="income_statement"), 2023)
        hits = find_rows(tree.leaf("interest_expense", "corporate"), [late, early_hi, early_lo], corp)
        self.assertEqual([(h.view.record.page, h.view.record.table_id) for h in hits], [(4, 3), (4, 7), (9, 1)])
        self.assertTrue(all(isinstance(h, RowHit) for h in hits))
        self.assertEqual(hits[0].row_idx, row_of(early_lo, "23"))


if __name__ == "__main__":
    unittest.main()

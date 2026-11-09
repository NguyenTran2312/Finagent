import unittest
from unittest.mock import patch

from indexing import table_semantics

from indexing.table_semantics import (
    TableAnalyzer,
    column_header,
    is_header_row,
    period_info,
    period_match_score,
)


class PeriodInfoTests(unittest.TestCase):
    def test_period_info_identifies_start_and_end(self) -> None:
        opening = period_info("Số đầu năm 2023", report_year=2023)
        closing = period_info("Số cuối năm 2023", report_year=2023)

        self.assertEqual(opening.year, 2023)
        self.assertEqual(closing.year, 2023)
        self.assertEqual(opening.period_type, "start")
        self.assertEqual(closing.period_type, "end")

    def test_period_match_score_distinguishes_start_and_end(self) -> None:
        opening = period_info("Số đầu năm 2023", report_year=2023)
        closing = period_info("Số cuối năm 2023", report_year=2023)

        self.assertEqual(
            period_match_score(opening, requested_year=2023, report_year=2023),
            8.0,
        )
        self.assertEqual(
            period_match_score(closing, requested_year=2023, report_year=2023),
            10.0,
        )
        self.assertEqual(
            period_match_score(opening, requested_year=2022, report_year=2023),
            -12.0,
        )
        self.assertEqual(
            period_match_score(
                period_info("Số đầu năm 2022", report_year=2023),
                requested_year=2022,
                report_year=2023,
            ),
            8.0,
        )


class ColumnHeaderTests(unittest.TestCase):
    def test_repeated_merged_headers_require_linear_normalization_work(self) -> None:
        rows = [["Disclosure paragraph", "Disclosure paragraph"] for _ in range(100)]
        analyzer = TableAnalyzer(rows)
        with patch.object(table_semantics, "fold_text", wraps=table_semantics.fold_text) as normalize:
            headers = [
                analyzer.column_header(row_idx, col_idx)
                for row_idx, row in enumerate(rows)
                for col_idx in range(len(row))
            ]
        self.assertEqual(headers[:2], ["", ""])
        self.assertEqual(set(headers[2:]), {"Disclosure paragraph"})
        self.assertLessEqual(normalize.call_count, 400, "Header cells are repeatedly normalized")

    def test_header_results_survive_out_of_order_and_repeated_lookups(self) -> None:
        rows = [
            ["Chỉ tiêu", "2023", "USD"],
            ["", "Năm nay"],
            ["", "NAM NAY", "usd"],
            ["", "", ""],
            ["Tiền", "100", "200"],
            ["Phải thu", "300", "400"],
            ["Chỉ tiêu", "2024"],
            ["Tiền", "500"],
        ]
        analyzer = TableAnalyzer(rows)
        for row_idx, col_idx, expected in (
            (4, 1, "2023 | Năm nay"),
            (1, 1, "2023"),
            (4, 2, "USD"),
            (5, 1, ""),
            (4, 1, "2023 | Năm nay"),
            (7, 1, "2024"),
            (7, 1, "2024"),
            (0, 1, ""),
        ):
            with self.subTest(row=row_idx, column=col_idx):
                self.assertEqual(analyzer.column_header(row_idx, col_idx), expected)

    def test_column_header_uses_nearest_block_and_ignores_blank_rows(self) -> None:
        rows = [
            ["Chỉ tiêu", "2022"],
            ["", "Số cuối năm"],
            ["Tiền", "100"],
            ["Phải thu", "200"],
            ["", ""],
            ["Chỉ tiêu", "2023"],
            ["", "Số cuối năm"],
            ["Tiền", "300"],
        ]

        self.assertEqual(column_header(rows, 7, 1), "2023 | Số cuối năm")
        self.assertEqual(
            TableAnalyzer(rows).column_header(7, 1),
            "2023 | Số cuối năm",
        )

    def test_column_header_keeps_initial_block_before_first_data_row(self) -> None:
        rows = [
            ["Chỉ tiêu", "2023"],
            ["", "Số cuối năm"],
            ["Tiền", "300"],
        ]

        self.assertEqual(column_header(rows, 2, 1), "2023 | Số cuối năm")


class HeaderRowTests(unittest.TestCase):
    def test_recognizes_common_financial_report_headers(self) -> None:
        cases = [
            ["Chỉ tiêu", "2022", "2023"],
            ["Năm trước", "Năm nay"],
            ["Kỳ trước", "Kỳ này"],
            ["Số đầu năm", "Số cuối năm"],
            ["Tại ngày 31/12/2023"],
            ["Đơn vị tính: triệu đồng"],
            ["Thuyết minh", "2023"],
            ["Chỉ tiêu", "Chỉ tiêu"],
        ]

        for row in cases:
            with self.subTest(row=row):
                self.assertTrue(is_header_row(row))

    def test_rejects_empty_rows_and_data_rows_with_amounts(self) -> None:
        cases = [
            [],
            ["", "  "],
            ["Tiền và tương đương tiền", "100"],
            ["Doanh thu thuần", "1.250,5", "900"],
            ["01", "Tiền mặt", "500"],
        ]

        for row in cases:
            with self.subTest(row=row):
                self.assertFalse(is_header_row(row))

    def test_only_year_values_are_valid_header_numbers(self) -> None:
        self.assertTrue(is_header_row(["Chỉ tiêu", "2022", "2023"]))
        self.assertFalse(is_header_row(["Chỉ tiêu", "2023", "100"]))

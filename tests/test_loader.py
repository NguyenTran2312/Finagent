import unittest
from unittest.mock import patch

from src.preprocess import loader
from src.preprocess.loader import parse_html_table


class HtmlTableResourceTests(unittest.TestCase):
    def test_rejects_raw_limits_before_parsing_the_remaining_input(self) -> None:
        cases = [
            (
                "rows", (3, 10, 100),
                "<tr><td>x</td></tr>" * 3,
                "<tr><td>overflow</td></tr>", "row limit",
            ),
            (
                "columns", (10, 3, 100),
                "<tr>" + "<td>x</td>" * 3,
                "<td>overflow</td></tr>", "size limits",
            ),
            (
                "cells", (10, 10, 3),
                "<tr><td>a</td><td>b</td></tr><tr><td>c</td>",
                "<td>overflow</td></tr>", "size limits",
            ),
        ]
        for name, (rows, columns, cells), prefix, overflow, message in cases:
            with self.subTest(limit=name):
                with patch.multiple(
                    loader, _MAX_TABLE_ROWS=rows,
                    _MAX_TABLE_COLUMNS=columns, _MAX_TABLE_CELLS=cells,
                ):
                    parser = loader._TableParser()
                    parser.feed("<table>" + prefix)
                    with self.assertRaisesRegex(ValueError, message):
                        parser.feed(overflow + "</table>")

    def test_allows_raw_tables_exactly_at_each_limit(self) -> None:
        with patch.multiple(loader, _MAX_TABLE_ROWS=2, _MAX_TABLE_COLUMNS=2, _MAX_TABLE_CELLS=4):
            self.assertEqual(
                parse_html_table(
                    "<table><tr><td>a</td><th>b</th></tr>"
                    "<tr><td>c</td><td>d</td></tr></table>"
                ),
                [["a", "b"], ["c", "d"]],
            )

    def test_rejects_excessive_spans(self) -> None:
        for attribute in ("rowspan", "colspan"):
            with self.subTest(attribute=attribute):
                fragment = f'<table><tr><td {attribute}="1000000000">x</td></tr></table>'
                with self.assertRaisesRegex(ValueError, "dimension limits"):
                    parse_html_table(fragment)

    def test_rejects_combined_span_cell_count_before_expansion(self) -> None:
        fragment = '<table><tr><td rowspan="10000" colspan="1000">x</td></tr></table>'
        with self.assertRaisesRegex(ValueError, "size limits"):
            parse_html_table(fragment)

    def test_checks_dimensions_and_dense_cell_budget(self) -> None:
        cases = [
            ("<tr><td>x</td></tr>" * 4, "row limit"),
            ('<tr><td colspan="2">x</td><td>y</td></tr>', "size limits"),
            ('<tr><td>x</td></tr><tr><td rowspan="3">y</td></tr>', "dimension limits"),
            ('<tr><td colspan="2">x</td></tr><tr><td>y</td></tr>', "size limits"),
            ('<tr><td rowspan="2">x</td></tr><tr><td colspan="2">y</td></tr>', "size limits"),
        ]
        with patch.multiple(loader, _MAX_TABLE_ROWS=3, _MAX_TABLE_COLUMNS=2, _MAX_TABLE_CELLS=3):
            for content, message in cases:
                with self.subTest(content=content):
                    with self.assertRaisesRegex(ValueError, message):
                        parse_html_table(f"<table>{content}</table>")

    def test_allows_boundary_sized_tables_and_trailing_rowspans(self) -> None:
        with patch.multiple(loader, _MAX_TABLE_ROWS=3, _MAX_TABLE_COLUMNS=2, _MAX_TABLE_CELLS=6):
            self.assertEqual(
                parse_html_table('<table><tr><td rowspan="3" colspan="2">x</td></tr></table>'),
                [["x", "x"], ["x", "x"], ["x", "x"]],
            )

    def test_keeps_existing_invalid_span_fallbacks(self) -> None:
        for span in ("invalid", "0", "-1", ""):
            with self.subTest(span=span):
                self.assertEqual(
                    parse_html_table(f'<table><tr><td rowspan="{span}" colspan="{span}">x</td></tr></table>'),
                    [["x"]],
                )


if __name__ == "__main__":
    unittest.main()

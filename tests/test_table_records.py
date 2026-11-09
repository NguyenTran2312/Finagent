import json
import unittest

from preprocess.tables import extract_structured_tables


class TableRecordUnitTests(unittest.TestCase):
    def test_saves_table_unit_from_header_metadata(self) -> None:
        source = """
        <table>
          <tr><th>Chỉ tiêu</th><th>2024</th></tr>
          <tr><td>Đơn vị tính: triệu đồng</td><td></td></tr>
          <tr><td>Doanh thu</td><td>100</td></tr>
        </table>
        """

        record = extract_structured_tables(source, "AAA_2024", 2024)[0]
        table_unit = json.loads(record["table_unit_json"])
        cell_units = json.loads(record["cell_units_json"])

        self.assertEqual(record["unit_scale"], 1_000_000.0)
        self.assertEqual(table_unit["unit_status"], "valid")
        self.assertEqual(table_unit["scale_multiplier"], "1E+6")
        self.assertEqual(cell_units[2][1]["scale_multiplier"], "1E+6")

    def test_saves_distinct_units_for_cells_in_different_columns(self) -> None:
        source = """
        <table>
          <tr>
            <th>Chỉ tiêu</th>
            <th>Số cuối năm (triệu đồng)</th>
            <th>Số đầu năm (nghìn đồng)</th>
          </tr>
          <tr><td>Doanh thu</td><td>100</td><td>200</td></tr>
        </table>
        """

        record = extract_structured_tables(source, "AAA_2024", 2024)[0]
        table_unit = json.loads(record["table_unit_json"])
        cell_units = json.loads(record["cell_units_json"])

        self.assertEqual(record["unit_scale"], 1.0)
        self.assertEqual(table_unit["unit_status"], "ambiguous")
        self.assertEqual(cell_units[1][1]["scale_multiplier"], "1E+6")
        self.assertEqual(cell_units[1][2]["scale_multiplier"], "1E+3")


class ExtractStructuredTablesTests(unittest.TestCase):
    def test_extracts_table_rows_dimensions_and_document_fields(self) -> None:
        source = """===== PAGE 2 =====
        BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
        <table>
          <tr><th>Chỉ tiêu</th><th>2024</th></tr>
          <tr><td>Doanh thu thuần</td><td>1.250</td></tr>
        </table>
        """

        records = extract_structured_tables(source, "AAA_2024", 2024)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["doc_id"], "AAA_2024")
        self.assertEqual(record["table_id"], 1)
        self.assertEqual(record["page"], 2)
        self.assertEqual(record["n_rows"], 2)
        self.assertEqual(record["n_cols"], 2)
        self.assertEqual(
            json.loads(record["rows_json"]),
            [["Chỉ tiêu", "2024"], ["Doanh thu thuần", "1.250"]],
        )
        self.assertEqual(json.loads(record["header_rows_json"]), [0])
        self.assertEqual(record["statement_kind"], "income_statement")
        self.assertIn("bao cao ket qua hoat dong kinh doanh", record["folded_text"])

    def test_extracts_multiple_tables_with_sequential_ids_and_page_numbers(self) -> None:
        source = """===== PAGE 1 =====
        <table><tr><td>Bảng một</td></tr></table>
        ===== PAGE 3 =====
        <table><tr><td>Bảng hai</td></tr></table>
        """

        records = extract_structured_tables(source, "AAA_2024", 2024)

        self.assertEqual(
            [(record["table_id"], record["page"]) for record in records],
            [(1, 1), (2, 3)],
        )
        self.assertEqual(
            [json.loads(record["rows_json"]) for record in records],
            [[["Bảng một"]], [["Bảng hai"]]],
        )

    def test_expands_rowspan_and_colspan_cells(self) -> None:
        source = """
        <table>
          <tr><th rowspan="2">Chỉ tiêu</th><th colspan="2">2024</th></tr>
          <tr><th>Số đầu năm</th><th>Số cuối năm</th></tr>
          <tr><td>Tiền mặt</td><td>10</td><td>20</td></tr>
        </table>
        """

        record = extract_structured_tables(source, "AAA_2024", 2024)[0]

        self.assertEqual(
            json.loads(record["rows_json"]),
            [
                ["Chỉ tiêu", "2024", "2024"],
                ["Chỉ tiêu", "Số đầu năm", "Số cuối năm"],
                ["Tiền mặt", "10", "20"],
            ],
        )
        self.assertEqual(record["n_rows"], 3)
        self.assertEqual(record["n_cols"], 3)

    def test_uses_unit_declared_before_table_as_table_and_cell_unit(self) -> None:
        source = """===== PAGE 1 =====
        <p>Đơn vị tính: triệu đồng</p>
        <table>
          <tr><th>Chỉ tiêu</th><th>2024</th></tr>
          <tr><td>Doanh thu</td><td>100</td></tr>
        </table>
        """

        record = extract_structured_tables(source, "AAA_2024", 2024)[0]

        self.assertIn("triệu đồng", record["context"])
        self.assertEqual(record["unit_scale"], 1_000_000.0)
        self.assertEqual(
            json.loads(record["table_unit_json"])["unit_status"],
            "valid",
        )
        self.assertEqual(
            json.loads(record["cell_units_json"])[1][1]["scale_multiplier"],
            "1E+6",
        )

    def test_no_tables_returns_empty_list(self) -> None:
        for source in ("", "Văn bản không có bảng", "<table><tr><td>Chưa đóng bảng"):
            with self.subTest(source=source):
                self.assertEqual(extract_structured_tables(source, "AAA_2024", 2024), [])

    def test_empty_table_is_returned_as_an_empty_record(self) -> None:
        records = extract_structured_tables("<table></table>", "AAA_2024", 2024)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["n_rows"], 0)
        self.assertEqual(records[0]["n_cols"], 0)
        self.assertEqual(json.loads(records[0]["rows_json"]), [])
        self.assertEqual(json.loads(records[0]["cell_units_json"]), [])


if __name__ == "__main__":
    unittest.main()

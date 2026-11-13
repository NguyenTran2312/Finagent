import unittest
from decimal import Decimal

from src.agent.answer import answer_text, format_value, leaf_detail, refusal_text
from src.agent.types import LeafLog
from src.calculation import Answer, FactSource
from src.calculation.units import Dimension, Quantity


def money(value, unit="ty"):
    return Answer("quantity", Quantity(Decimal(value), Dimension.MONEY), unit)


def source(row="Lợi nhuận sau thuế", doc="VJC_2022", page=12):
    return FactSource(chunk_id=f"{doc}:{page}", doc_id=doc, page=page, row_label=row,
                      raw_text="1", unit_text="VND", scale=Decimal(1))


class FormatValueTests(unittest.TestCase):
    def test_money_in_ty(self):
        self.assertEqual(format_value(money("-1950254195864")), "-1.950,25 tỷ đồng")

    def test_percent(self):
        answer = Answer("quantity", Quantity(Decimal("0.123456"), Dimension.RATIO), "percent")
        self.assertEqual(format_value(answer), "12,35 %")

    def test_negative_zero_normalised(self):
        self.assertEqual(format_value(money("-1000")), "0,00 tỷ đồng")

    def test_boolean_and_year(self):
        self.assertEqual(format_value(Answer("boolean", True, "boolean")), "Có")
        self.assertEqual(format_value(Answer("boolean", False, "boolean")), "Không")
        self.assertEqual(format_value(Answer("year", 2022, "year")), "2022")
        self.assertEqual(format_value(Answer("ticker", "VJC", "ticker")), "VJC")
        self.assertEqual(format_value(Answer("count", 3, "count")), "3")

    def test_small_values_keep_two_decimals(self):
        answer = Answer("quantity", Quantity(Decimal("2.5"), Dimension.RATIO), "times")
        self.assertEqual(format_value(answer), "2,50 lần")


class AnswerTextTests(unittest.TestCase):
    def test_description_and_sources(self):
        text = answer_text(money("5000000000"), [source()], "LNST VJC 2022")
        self.assertEqual(text.splitlines(), [
            "LNST VJC 2022: 5,00 tỷ đồng.", "Nguồn:",
            "- Lợi nhuận sau thuế — VJC_2022, trang 12"])

    def test_default_prefix(self):
        self.assertTrue(answer_text(money("0"), [], None).startswith("Trả lời: 0,00 tỷ đồng."))

    def test_sources_deduplicated(self):
        text = answer_text(money("1"), [source(), source(), source(page=13)], None)
        self.assertEqual(text.count("- Lợi nhuận sau thuế"), 2)


class RefusalTests(unittest.TestCase):
    def test_each_refusal_code_has_text(self):
        expected = {
            "plan_invalid": "Không phân tích được câu hỏi.",
            "metric_unmapped": "Không xác định được chỉ tiêu cần tính.",
            "facts_missing": "Không đủ số liệu để trả lời.",
            "gates_failed": "Không tạo được phép tính hợp lệ sau nhiều lần thử.",
            "clarify": "Câu hỏi cần làm rõ thêm.",
            "llm_unavailable": "Không gọi được mô hình ngôn ngữ, vui lòng thử lại.",
        }
        for code, sentence in expected.items():
            self.assertEqual(refusal_text(code, []), sentence)

    def test_details_become_bullets(self):
        self.assertEqual(refusal_text("clarify", ["a", "b"]),
                         "Câu hỏi cần làm rõ thêm.\n- a\n- b")

    def test_unknown_code_raises(self):
        with self.assertRaises(KeyError):
            refusal_text("nope", [])


class LeafDetailTests(unittest.TestCase):
    def log(self, **kw):
        base = dict(leaf_id="net_profit", ticker="VJC", year=2022, scope="consolidated",
                    outcome="Ambiguous", reason="2 bảng cho 2 giá trị", value=None,
                    chunk_id=None, doc_id=None, page=None, row_label=None, tool="extractor")
        return LeafLog(**{**base, **kw})

    def test_leaf_detail_names_scope(self):
        self.assertEqual(leaf_detail(self.log()),
                         "net_profit VJC 2022 (hợp nhất): Ambiguous — 2 bảng cho 2 giá trị")
        self.assertEqual(leaf_detail(self.log(scope="separate", reason=None, outcome="Found")),
                         "net_profit VJC 2022 (riêng): Found")


if __name__ == "__main__":
    unittest.main()

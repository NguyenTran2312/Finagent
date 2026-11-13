import json
import unittest
from decimal import Decimal

from src.agent import present
from src.agent.metrics import MetricMapping
from src.agent.solver import Attempt
from src.agent.types import AgentRun, LeafLog, StepEvent
from src.query_paraphrase.schemas import QueryPlan, Subquery


def _subquery(**overrides):
    base = dict(
        id="q1", question="Doanh thu VNM 2023?", search_query="doanh thu VNM 2023",
        ticker="VNM", company_name=None, year=2023, period=None, scope="consolidated",
        metric="revenue", metric_text="doanh thu thuần", unit=None,
    )
    base.update(overrides)
    return Subquery(**base)


def _log(**overrides):
    base = dict(
        leaf_id="revenue", ticker="VNM", year=2023, scope="consolidated", outcome="Found",
        reason=None, value=Decimal("1234567.5"), chunk_id="c1", doc_id="VNM_2023_BCTC",
        page=12, row_label="Doanh thu", tool="extractor",
    )
    base.update(overrides)
    return LeafLog(**base)


class PresentTests(unittest.TestCase):
    def test_plan_rows_and_operation(self):
        plan = QueryPlan(
            original_question="q", operation="lookup", operation_description=None,
            subqueries=[_subquery(), _subquery(id="q2", ticker=None, year=None, scope=None,
                                               metric=None, metric_text=None)],
        )
        data = present.plan_data(plan)
        json.dumps(data)
        self.assertEqual(data[present.OPERATION], "lookup")
        event = StepEvent("plan", "Kế hoạch", data)
        table = present.rows(event)
        self.assertEqual(
            list(table[0]),
            ["Câu hỏi con", "Mã", "Năm", "Phạm vi", "Chỉ tiêu", "Cụm từ"],
        )
        self.assertEqual(table[0]["Mã"], "VNM")
        self.assertEqual(table[0]["Năm"], "2023")
        self.assertEqual(table[1]["Mã"], "")
        self.assertEqual(table[1]["Năm"], "")

    def test_metrics_rows(self):
        data = present.metrics_data([MetricMapping("q1", "revenue", "catalog"),
                                     MetricMapping("q2", None, "unmapped")])
        json.dumps(data)
        table = present.rows(StepEvent("metrics", "Chỉ tiêu", data))
        self.assertEqual(list(table[0]), ["Câu hỏi con", "Node", "Cách"])
        self.assertEqual(table[0]["Node"], "revenue")
        self.assertEqual(table[1]["Node"], "")
        self.assertEqual(table[1]["Cách"], "unmapped")

    def test_gather_found_row(self):
        data = present.gather_data([_log()])
        json.dumps(data)
        table = present.rows(StepEvent("gather", "Thu thập", data))
        self.assertEqual(
            list(table[0]),
            ["Leaf", "Mã", "Năm", "Phạm vi", "Kết quả", "Giá trị", "Nguồn", "Công cụ"],
        )
        self.assertEqual(table[0]["Giá trị"], "1.234.567,50")
        self.assertEqual(table[0]["Nguồn"], "VNM_2023_BCTC, trang 12")
        self.assertEqual(table[0]["Kết quả"], "Found")
        self.assertEqual(table[0]["Công cụ"], "extractor")

    def test_gather_failure_row_shows_reason(self):
        log = _log(outcome="NotFound", reason="không thấy dòng", value=None,
                   doc_id=None, page=None, row_label=None, tool="locator")
        table = present.rows(StepEvent("gather", "Thu thập", present.gather_data([log])))
        self.assertEqual(table[0]["Kết quả"], "NotFound: không thấy dòng")
        self.assertEqual(table[0]["Giá trị"], "")
        self.assertEqual(table[0]["Nguồn"], "")

    def test_solve_rows_and_code(self):
        attempt = Attempt(2, "answer = 1", (), (), ("thiếu đơn vị", "sai năm"), "repair")
        data = present.solve_data(attempt)
        json.dumps(data)
        self.assertEqual(data[present.CODE], "answer = 1")
        table = present.rows(StepEvent("solve", "Tính", data))
        self.assertEqual(list(table[0]), ["Lần", "Kết quả", "Vấn đề"])
        self.assertEqual(table[0]["Lần"], "2")
        self.assertEqual(table[0]["Kết quả"], "repair")
        self.assertEqual(table[0]["Vấn đề"], "thiếu đơn vị; sai năm")

    def test_answer_rows(self):
        data = present.answer_data("refused", "facts_missing", "Không đủ số liệu.")
        json.dumps(data)
        table = present.rows(StepEvent("answer", "Trả lời", data))
        self.assertEqual(list(table[0]), ["Trạng thái", "Lý do"])
        self.assertEqual(table[0]["Trạng thái"], "refused")
        self.assertEqual(table[0]["Lý do"], "facts_missing")
        self.assertEqual(data[present.TEXT], "Không đủ số liệu.")
        answered = present.answer_data("answered", None, "ok")
        self.assertEqual(present.rows(StepEvent("answer", "t", answered))[0]["Lý do"], "")

    def test_footer(self):
        run = AgentRun("answered", "x", None, None, (), (), 3, 4.26)
        self.assertEqual(present.footer(run), "3 lần gọi LLM · 4.3 giây")


if __name__ == "__main__":
    unittest.main()

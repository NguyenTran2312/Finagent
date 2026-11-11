import unittest
from decimal import Decimal as D

from src.calculation.metric_tree import LeafRequest, MetricTree
from src.extraction.extractor import LeafExtractor
from src.extraction.llm_fallback import (
    CellPointer,
    LlmCellLocator,
    PointerAnswer,
    build_prompt,
    shortlist,
    structured_pointer_adapter,
)
from src.extraction.outcomes import Found, NotFound
from src.extraction.report_store import ReportDoc
from src.extraction.table_view import TableView
from src.extraction.templates import TickerTemplates
from tests.extraction_fixtures import InMemoryReportStore, record, row_of, vjc_view

tree = MetricTree.load()
templates = TickerTemplates({"VJC": "corporate"})

INTEREST_2023 = LeafRequest("interest_expense", 2023, "flow")
NET_REVENUE_2023 = LeafRequest("net_revenue", 2023, "flow")
_P = "Năm tài chính kết thúc ngày 31 tháng 12 năm"
_VJC_HEADER = [["Mã số", "", "Thuyết minh", _P, _P], ["Mã số", "", "Thuyết minh", "2023 VND", "2022 VND"]]


def leaf(metric_id):
    return tree.leaf(metric_id, "corporate")


def returning(pointer):
    """A fake ``llm`` answering ``found`` at ``pointer`` = (table_ref, row_idx, col_idx)."""
    ref, row, col = pointer
    return lambda prompt: PointerAnswer(status="found", pointer=CellPointer(table_ref=ref, row_idx=row, col_idx=col))


def locator_returning(pointer, **kwargs):
    return LlmCellLocator(returning(pointer), **kwargs)


def locate_interest(locator, views=None):
    return locator.locate(leaf("interest_expense"), INTEREST_2023, "primary", views or [vjc_view()])


def long_table_view(rows=80, leaf_row=70):
    """A VJC-shaped income statement with ``rows`` data rows; row ``leaf_row`` holds code 23 (interest)."""
    grid = [list(r) for r in _VJC_HEADER]
    for idx in range(len(grid), len(grid) + rows):
        if idx == leaf_row:
            grid.append(["23", "Chi phí lãi vay", "", "(1.950.254.195.864)", "(1.383.641.245.573)"])
        else:
            grid.append([str(idx), f"Khoản mục {idx}", "", "1.000", "900"])
    return TableView(record("VJC23C", 1, 12, grid, kind="income_statement"), 2023)


def cell_view(cell, *, header="VND"):
    """Net revenue at (2, 3) holds ``cell``; a second data row keeps column 3 a value column."""
    rows = [["Mã số", "", "Thuyết minh", "Năm nay"], ["Mã số", "", "Thuyết minh", header],
            ["10", "Doanh thu thuần", "", cell], ["11", "Giá vốn hàng bán", "", "(500)"]]
    return TableView(record("D", 1, 3, rows, kind="income_statement"), 2023)


def simple_view(table_id, page, labels, *, kind):
    rows = [["", "", "Năm nay VND"], *(["", label, "1.000"] for label in labels)]
    return TableView(record("D", table_id, page, rows, kind=kind), 2023)


class Recorder:
    """A fake ``llm`` that records prompts and returns ``answer`` (or raises it)."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


class FakeStructured:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, prompt):
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self, *responses):
        self.bind_calls = []
        self._structured = FakeStructured(responses)

    def with_structured_output(self, schema, **kwargs):
        self.bind_calls.append((schema, kwargs))
        return self._structured


class PointerModelTest(unittest.TestCase):
    def test_pointer_required_iff_found(self):
        with self.assertRaises(ValueError):
            PointerAnswer(status="found", pointer=None)
        with self.assertRaises(ValueError):
            PointerAnswer(status="not_present", pointer=CellPointer(table_ref="T1", row_idx=0, col_idx=0))
        with self.assertRaises(ValueError):
            CellPointer(table_ref="T1", row_idx=-1, col_idx=0)
        with self.assertRaises(ValueError):
            CellPointer(table_ref="T1", row_idx=0, col_idx=0, value="1")
        self.assertIsNone(PointerAnswer(status="not_present", pointer=None).pointer)


class LlmCellLocatorTest(unittest.TestCase):
    def test_valid_pointer_found(self):
        row23 = row_of(vjc_view(), "23")
        res = locate_interest(locator_returning(("T1", row23, 3)))
        self.assertEqual(res.read.status, "ok")
        self.assertEqual(res.read.value.value, D("-1950254195864"))
        self.assertEqual(res.read.source.raw_text, "(1.950.254.195.864)")
        self.assertIsNone(res.reason)
        self.assertEqual(res.warnings, ())

    def test_rejections(self):
        view = vjc_view()
        row23, row_revenue = row_of(view, "23"), row_of(view, "01")
        for pointer, check in [(("T9", 2, 3), "unknown_table"), (("T1", 999, 3), "out_of_bounds"),
                               (("T1", row23, 99), "out_of_bounds"),
                               (("T1", 0, 3), "not_data_row"), (("T1", row23, 0), "not_value_column"),
                               (("T1", row23, 2), "not_value_column"),
                               (("T1", row23, 4), "period"), (("T1", row_revenue, 3), "label")]:
            with self.subTest(pointer=pointer, check=check):
                res = locate_interest(locator_returning(pointer))
                self.assertIsNone(res.read)
                self.assertEqual(res.reason, "llm_pointer_rejected")
                self.assertEqual(res.warnings, (f"llm_pointer_rejected:{check}",))

    def test_pointer_beyond_rendered_rows_rejected(self):  # Review Focus 5, E7
        view = long_table_view(rows=80)  # valid leaf row at index 70
        self.assertEqual(view.read(70, 3, leaf("interest_expense").dimension).status, "ok")
        llm = Recorder(PointerAnswer(status="found", pointer=CellPointer(table_ref="T1", row_idx=70, col_idx=3)))
        res = LlmCellLocator(llm, max_rows=60).locate(leaf("interest_expense"), INTEREST_2023, "primary", [view])
        self.assertIsNone(res.read)
        self.assertEqual(res.reason, "llm_pointer_rejected")
        self.assertIn("llm_pointer_rejected:not_rendered", res.warnings)
        self.assertNotIn("Chi phí lãi vay", llm.prompts[0].split("\n\n", 1)[1])
        # The same pointer is accepted once the row is rendered.
        res = LlmCellLocator(llm, max_rows=80).locate(leaf("interest_expense"), INTEREST_2023, "primary", [view])
        self.assertEqual(res.read.value.value, D("-1950254195864"))

    def test_blank_unparsed_and_unit_cells_rejected(self):
        for cell, header, check in [("-", "VND", "number"), ("1.234.567 (*)", "VND", "number"),
                                    ("1.000", "USD", "unit")]:
            with self.subTest(cell=cell, header=header):
                view = cell_view(cell, header=header)
                res = locator_returning(("T1", 2, 3)).locate(leaf("net_revenue"), NET_REVENUE_2023, "primary", [view])
                self.assertIsNone(res.read)
                self.assertEqual(res.reason, "llm_pointer_rejected")
                self.assertEqual(res.warnings, (f"llm_pointer_rejected:{check}",))

    def test_differing_period_columns_rejected(self):
        rows = [["Mã số", "", "Thuyết minh", "Năm nay", "Năm 2023"],
                ["Mã số", "", "Thuyết minh", "VND", "VND"],
                ["10", "Doanh thu thuần", "", "1.000", "2.000"]]
        view = TableView(record("D", 1, 3, rows, kind="income_statement"), 2023)
        self.assertEqual(view.period_columns(2023, "flow", "primary"), (3, 4))
        res = locator_returning(("T1", 2, 3)).locate(leaf("net_revenue"), NET_REVENUE_2023, "primary", [view])
        self.assertEqual(res.warnings, ("llm_pointer_rejected:period",))
        rows[2][4] = "1.000"  # equal values: the pointed cell is read
        view = TableView(record("D", 1, 3, rows, kind="income_statement"), 2023)
        res = locator_returning(("T1", 2, 3)).locate(leaf("net_revenue"), NET_REVENUE_2023, "primary", [view])
        self.assertEqual(res.read.value.value, D("1000"))

    def test_failures(self):
        not_present = PointerAnswer(status="not_present", pointer=None)
        for answer, warnings in [(TimeoutError("secret raw text"), ("llm_fallback_error:TimeoutError",)),
                                 (ConnectionRefusedError("x"), ("llm_fallback_error:ConnectionRefusedError",)),
                                 (None, ("llm_fallback_unparsed",)),
                                 ({"status": "found"}, ("llm_fallback_unparsed",)),
                                 (not_present, ())]:
            with self.subTest(answer=answer):
                llm = Recorder(answer)
                res = locate_interest(LlmCellLocator(llm))
                self.assertEqual(len(llm.prompts), 1)  # one call, no retry
                self.assertIsNone(res.read)
                self.assertIsNone(res.reason)  # the extractor keeps the deterministic reason
                self.assertEqual(res.warnings, warnings)
                self.assertNotIn("secret", " ".join(res.warnings))
        with self.assertRaises(ValueError):
            locate_interest(LlmCellLocator(Recorder(ValueError("boom"))))
        with self.assertRaises(TimeoutError):  # only the configured errors are caught
            locate_interest(LlmCellLocator(Recorder(TimeoutError()), llm_errors=(ConnectionError,)))

    def test_no_table_rendered_skips_llm(self):
        llm = Recorder(None)
        res = locate_interest(LlmCellLocator(llm, max_chars=50))
        self.assertEqual((res.read, res.reason, res.warnings), (None, None, ()))
        self.assertEqual(llm.prompts, [])
        res = LlmCellLocator(llm).locate(leaf("interest_expense"), INTEREST_2023, "primary", [])
        self.assertEqual((res.read, res.reason, res.warnings), (None, None, ()))
        self.assertEqual(llm.prompts, [])

    def test_constructor_validation(self):
        with self.assertRaises(TypeError):
            LlmCellLocator(None)
        with self.assertRaises(TypeError):
            LlmCellLocator(Recorder(None), llm_errors=(TimeoutError, "x"))
        for kwargs in ({"max_tables": 0}, {"max_rows": 0}, {"max_chars": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                LlmCellLocator(Recorder(None), **kwargs)


class AdapterTest(unittest.TestCase):
    def test_adapter(self):
        answer = PointerAnswer(status="found", pointer=CellPointer(table_ref="T1", row_idx=6, col_idx=3))
        client = FakeClient(
            {"raw": "RAW TEXT", "parsed": answer, "parsing_error": None},
            {"raw": "RAW TEXT", "parsed": None, "parsing_error": ValueError("bad json")},
            {"raw": "RAW TEXT", "parsed": {"status": "found", "pointer": {"table_ref": "T2", "row_idx": 1,
                                                                          "col_idx": 4}}, "parsing_error": None},
            {"raw": "RAW TEXT", "parsed": {"status": "found", "pointer": None}, "parsing_error": None},
            {"raw": "RAW TEXT", "parsed": {"status": "not_present", "pointer": None, "value": "12"},
             "parsing_error": None},
            {"raw": "RAW TEXT", "parsed": {"status": "found", "pointer": {"table_ref": "T1", "row_idx": "6",
                                                                          "col_idx": 3}}, "parsing_error": None},
            {"raw": "RAW TEXT", "parsed": "RAW TEXT", "parsing_error": None},
            "RAW TEXT",
        )
        llm = structured_pointer_adapter(client)
        self.assertEqual(client.bind_calls, [(PointerAnswer, {"include_raw": True})])
        self.assertEqual(llm("p"), answer)
        self.assertIsNone(llm("p"))  # parsing_error
        self.assertEqual(llm("p"), PointerAnswer(status="found", pointer=CellPointer(table_ref="T2", row_idx=1,
                                                                                     col_idx=4)))
        for _ in range(5):  # invalid mappings, a raw string as parsed, a non-mapping response
            self.assertIsNone(llm("p"))
        self.assertEqual(len(client.bind_calls), 1)  # bound once

        class NoStructured:
            pass

        with self.assertRaises(TypeError):
            structured_pointer_adapter(NoStructured())


class ShortlistAndPromptTest(unittest.TestCase):
    def test_shortlist_ranking(self):
        a = simple_view(1, 5, ["Doanh thu"], kind="income_statement")  # 0 shared tokens + 2
        b = simple_view(2, 20, ["Chi phí lãi vay"], kind="note")  # 4 shared tokens
        c = simple_view(3, 3, ["Chi phí khác"], kind="note")  # 2 shared tokens, earlier page than a
        d = simple_view(4, 1, ["Chi phí lãi vay"], kind="balance_sheet")  # other statement: not a candidate
        e = simple_view(1, 20, ["Chi phí lãi vay trả trước"], kind=None)  # 4 shared, same page as b, lower id
        f = simple_view(5, 30, ["Lãi khác"], kind="unknown")  # 1 shared token
        ranked = shortlist(leaf("interest_expense"), [a, b, c, d, e, f], max_tables=10)
        self.assertEqual([v.record.table_id for v in ranked], [1, 2, 3, 1, 5])
        self.assertEqual([v.record.page for v in ranked], [20, 20, 3, 5, 30])
        top = shortlist(leaf("interest_expense"), [a, b, c, d, e, f], max_tables=3)
        self.assertEqual(top, ranked[:3])

    def test_prompt_frame(self):
        prompt, refs = build_prompt(leaf("interest_expense"), INTEREST_2023, [vjc_view()], max_rows=60,
                                    max_chars=12000)
        self.assertTrue(prompt.startswith(
            "Bạn là trợ lý đọc báo cáo tài chính. Hãy tìm ô chứa giá trị của chỉ tiêu dưới đây.\n"
            "Chỉ tiêu: Chi phí lãi vay (tên khác: chi phí lãi vay, trong đó chi phí lãi vay)\n"
            "Loại báo cáo: income_statement. Năm tài chính: 2023. Kỳ: trong năm.\n"
            'Chỉ trả về vị trí ô (table_ref, row_idx, col_idx) hoặc status "not_present". '
            "Không được trả về con số.\n"))
        self.assertIn("\nT1\n", prompt)
        self.assertIn("6 | 0: 23 | 1: - Trong đó: Chi phí lãi vay | 2: 32 | 3: (1.950.254.195.864)", prompt)
        self.assertEqual(list(refs), ["T1"])
        self.assertEqual(refs["T1"][1], 9)  # all rows rendered
        closing = LeafRequest("cash_and_equivalents", 2023, "closing")
        prompt, _ = build_prompt(leaf("cash_and_equivalents"), closing, [vjc_view()], max_rows=60, max_chars=12000)
        self.assertIn("Kỳ: số dư cuối năm.", prompt)

    def test_prompt_budget(self):
        views = [vjc_view(), long_table_view(rows=10), simple_view(9, 40, ["Chi phí lãi vay"], kind="note")]
        full, refs = build_prompt(leaf("interest_expense"), INTEREST_2023, views, max_rows=60, max_chars=100000)
        self.assertEqual(list(refs), ["T1", "T2", "T3"])
        one, _ = build_prompt(leaf("interest_expense"), INTEREST_2023, views[:1], max_rows=60, max_chars=100000)
        prompt, refs = build_prompt(leaf("interest_expense"), INTEREST_2023, views, max_rows=60,
                                    max_chars=len(one) + 10)
        self.assertEqual(list(refs), ["T1"])  # lower-ranked tables dropped whole
        self.assertEqual(prompt, one)
        self.assertLessEqual(len(prompt), len(one) + 10)
        self.assertIn("Không được trả về con số", prompt)
        self.assertNotIn("\nT2\n", prompt)
        prompt, refs = build_prompt(leaf("interest_expense"), INTEREST_2023, views, max_rows=60, max_chars=len(one))
        self.assertEqual(list(refs), ["T1"])
        prompt, refs = build_prompt(leaf("interest_expense"), INTEREST_2023, views, max_rows=60, max_chars=100)
        self.assertEqual(refs, {})

    def test_render_limit(self):
        view = long_table_view(rows=80)
        prompt, refs = build_prompt(leaf("interest_expense"), INTEREST_2023, [view], max_rows=60, max_chars=100000)
        self.assertEqual(refs["T1"], (view, 62))  # 2 header rows + 60 data rows
        self.assertIn("\n61 | ", prompt)
        self.assertNotIn("\n62 | ", prompt)


class ExtractorIntegrationTest(unittest.TestCase):
    def _store(self, interest_code):
        rows = [*(list(r) for r in _VJC_HEADER),
                ["01", "Doanh thu bán hàng và cung cấp dịch vụ", "", "58.340.618.715.308", "40.141.863.488.029"],
                [interest_code, "- Trong đó: Chi phí lãi vay", "32", "(1.950.254.195.864)", "(1.383.641.245.573)"]]
        doc = ReportDoc("VJC23C", "VJC", 2023, "consolidated", None)
        return InMemoryReportStore([doc], {"VJC23C": [record("VJC23C", 1, 12, rows, kind="income_statement")]})

    def test_extractor_integration(self):
        llm = Recorder(PointerAnswer(status="found", pointer=CellPointer(table_ref="T1", row_idx=3, col_idx=3)))
        extractor = LeafExtractor(self._store(""), tree, templates, fallback=LlmCellLocator(llm))
        out = extractor.extract("VJC", "consolidated", [INTEREST_2023]).outcomes[0]
        self.assertEqual(len(llm.prompts), 1)
        self.assertIsInstance(out, Found)
        self.assertEqual(out.extracted.method, "llm_located")
        f = out.extracted.fact
        self.assertEqual(f.value.value, D("-1950254195864"))
        self.assertEqual((f.ticker, f.fiscal_year, f.scope, f.period_kind, f.restated),
                         ("VJC", 2023, "consolidated", "flow", False))
        s = f.source
        self.assertEqual((s.chunk_id, s.doc_id, s.page, s.row_label, s.raw_text, s.unit_text, s.scale),
                         ("VJC23C:table:1", "VJC23C", 12, "- Trong đó: Chi phí lãi vay", "(1.950.254.195.864)",
                          "2023 VND", 1))
        self.assertEqual(s.column_header, f"{_P} | 2023 VND")

    def test_extractor_rejection_and_failure(self):
        bad = Recorder(PointerAnswer(status="found", pointer=CellPointer(table_ref="T1", row_idx=3, col_idx=4)))
        out = LeafExtractor(self._store(""), tree, templates, fallback=LlmCellLocator(bad)).extract(
            "VJC", "consolidated", [INTEREST_2023]).outcomes[0]
        self.assertIsInstance(out, NotFound)
        self.assertEqual(out.reason, "llm_pointer_rejected")
        self.assertIn("llm_pointer_rejected:period", out.warnings)
        failing = Recorder(TimeoutError("raw"))
        out = LeafExtractor(self._store(""), tree, templates, fallback=LlmCellLocator(failing)).extract(
            "VJC", "consolidated", [INTEREST_2023]).outcomes[0]
        self.assertIsInstance(out, NotFound)
        self.assertEqual(out.reason, "no_row")
        self.assertIn("llm_fallback_error:TimeoutError", out.warnings)

    def test_deterministic_hit_never_calls_llm(self):
        llm = Recorder(None)
        out = LeafExtractor(self._store("23"), tree, templates, fallback=LlmCellLocator(llm)).extract(
            "VJC", "consolidated", [INTEREST_2023]).outcomes[0]
        self.assertEqual(out.extracted.method, "code")
        self.assertEqual(llm.prompts, [])


if __name__ == "__main__":
    unittest.main()

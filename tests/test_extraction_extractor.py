import dataclasses
import unittest
from decimal import Decimal as D

from src.calculation.facts import FactSet
from src.calculation.metric_tree import LeafRequest, MetricTree
from src.calculation.units import Dimension
from src.extraction.extractor import ExtractionResult, LeafExtractor, LocatorResult
from src.extraction.outcomes import (
    Ambiguous,
    BlankValue,
    Conflict,
    Found,
    NotApplicable,
    NotFound,
    ReportMissing,
    UnitUnknown,
    UnknownTicker,
    UnsupportedTemplate,
)
from src.extraction.report_store import ReportDoc
from src.extraction.templates import TickerTemplates
from tests.extraction_fixtures import ACB_BS_2022, InMemoryReportStore, record, row_of

tree = MetricTree.load()
templates = TickerTemplates({"VJC": "corporate", "ACB": "bank", "FTS": "securities"})

_P = "Năm tài chính kết thúc ngày 31 tháng 12 năm"
_BANK_BS_HEADER = [
    ["", "", "Thuyết minh", "Tại ngày", "Tại ngày"],
    ["", "", "Thuyết minh", "31.12.2022 Triệu VND", "31.12.2021 Triệu VND"],
]

INTEREST_2023 = LeafRequest("interest_expense", 2023, "flow")
NET_REVENUE_2023 = LeafRequest("net_revenue", 2023, "flow")


def is_rows(year, current="(1.950.254.195.864)", prior="(1.383.641.245.573)", *, unit="VND", interest=True,
            interest_label="- Trong đó: Chi phí lãi vay"):
    """A VJC-shaped income statement for report ``year``; row 23 (interest expense) is optional."""
    rows = [
        ["Mã số", "", "Thuyết minh", _P, _P],
        ["Mã số", "", "Thuyết minh", f"{year} {unit}".strip(), f"{year - 1} {unit}".strip()],
        ["01", "Doanh thu bán hàng và cung cấp dịch vụ", "", "58.340.618.715.308", "40.141.863.488.029"],
        ["10", "Doanh thu thuần về bán hàng và cung cấp dịch vụ", "29", "58.340.618.715.308",
         "40.141.863.488.029"],
    ]
    if interest:
        rows.append(["23", interest_label, "32", current, prior])
    else:
        rows.append(["", interest_label, "32", current, prior])  # no code: never matched by code
    return rows


def doc(doc_id, year, scope="consolidated", variant=None, ticker="VJC"):
    return ReportDoc(doc_id, ticker, year, scope, variant)


def is_table(doc_id, rows, *, table_id=1, page=12, context=""):
    return record(doc_id, table_id, page, rows, kind="income_statement", context=context)


def store_of(*docs_and_tables):
    """``store_of((doc, [records]), ...)``."""
    return InMemoryReportStore([d for d, _ in docs_and_tables], {d.doc_id: t for d, t in docs_and_tables})


def default_store():
    return store_of(
        (doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023))]),
        (doc("ACB22C", 2022, ticker="ACB"), [record("ACB22C", 2, 7, ACB_BS_2022, kind="balance_sheet")]),
    )


def extract(ticker, scope, requests, *, store=None, fallback=None) -> ExtractionResult:
    store = store if store is not None else default_store()
    return LeafExtractor(store, tree, templates, fallback=fallback).extract(ticker, scope, requests)


def one(store, request=INTEREST_2023, ticker="VJC", fallback=None):
    return extract(ticker, "consolidated", [request], store=store, fallback=fallback).outcomes[0]


class StubLocator:
    def __init__(self, respond):
        self._respond = respond
        self.calls = []

    def locate(self, leaf, request, role, views):
        self.calls.append((leaf.id, request, role, tuple(sorted({v.record.doc_id for v in views}))))
        return self._respond(views)


def read_label(label, col=3):
    """Stub response: the cell of the first view's row ``label``."""
    def respond(views):
        view = views[0]
        return LocatorResult(view.read(row_of(view, label), col, Dimension.MONEY), None, ("llm_ok",))
    return respond


class LeafExtractorTest(unittest.TestCase):
    def test_vjc_found_with_full_source(self):
        out = extract("VJC", "consolidated", [LeafRequest("interest_expense", 2023, "flow")]).outcomes[0]
        self.assertIsInstance(out, Found)
        f = out.extracted.fact
        self.assertEqual(f.value.value, D("-1950254195864"))
        self.assertEqual((f.source.chunk_id, f.source.page, f.source.raw_text),
                         ("VJC23C:table:1", 12, "(1.950.254.195.864)"))
        self.assertEqual(out.extracted.method, "code")
        self.assertEqual((f.ticker, f.fiscal_year, f.scope, f.period_kind, f.restated),
                         ("VJC", 2023, "consolidated", "flow", False))
        self.assertEqual(out.extracted.warnings, ())

    def test_primary_missing_uses_next_year(self):
        store = store_of((doc("VJC24C", 2024), [is_table("VJC24C", is_rows(2024, "(9)", "(1.383.641.245.573)"))]))
        out = one(store)
        self.assertIsInstance(out, Found)
        f = out.extracted.fact
        self.assertEqual(f.value.value, D("-1383641245573"))
        self.assertEqual(f.fiscal_year, 2023)
        self.assertFalse(f.restated)
        self.assertIn("from_next_year_report:VJC24C", out.extracted.warnings)
        self.assertIn("primary_report_missing", out.extracted.warnings)

    def test_restatement_marker_sets_restated(self):
        table = is_table("VJC24C", is_rows(2024), context="Số liệu so sánh đã được trình bày lại")
        out = one(store_of((doc("VJC24C", 2024), [table])))
        self.assertTrue(out.extracted.fact.restated)
        # The marker is ignored for a primary (current-column) fact.
        table23 = is_table("VJC23C", is_rows(2023), context="Số liệu so sánh đã được trình bày lại")
        self.assertFalse(one(store_of((doc("VJC23C", 2023), [table23]))).extracted.fact.restated)

    def test_closing_from_opening_column(self):
        bs = [
            ["Mã số", "TÀI SẢN", "Thuyết minh", "Số cuối năm VND", "Số đầu năm VND"],
            ["100", "Tài sản ngắn hạn", "", "1.000", "900"],
            ["110", "Tiền", "", "500", "400"],
            ["270", "TỔNG CỘNG TÀI SẢN", "", "5.000", "4.000"],
        ]
        store = store_of((doc("VJC22C", 2022), [record("VJC22C", 1, 5, bs, kind="balance_sheet")]))
        out = one(store, LeafRequest("total_assets", 2021, "closing"))
        self.assertIsInstance(out, Found)
        self.assertEqual(out.extracted.fact.value.value, D("4000"))
        self.assertEqual(out.extracted.fact.source.column_header, "Số đầu năm VND")
        self.assertEqual(out.extracted.fact.fiscal_year, 2021)

    def test_report_missing_lists_scopes(self):
        store = store_of(
            (doc("VJC23S", 2023, "separate"), [is_table("VJC23S", is_rows(2023))]),
            (doc("VJC23U", 2023, "unknown"), [is_table("VJC23U", is_rows(2023))]),
        )
        out = one(store)
        self.assertIsInstance(out, ReportMissing)
        self.assertEqual(out.available_scopes, ("separate", "unknown"))
        self.assertEqual(store.tables_read, {})  # other scopes are never read

    def test_combination_table(self):
        no_row = is_rows(2023, interest=False)
        no_row24 = is_rows(2024, interest=False)
        with self.subTest("ReportMissing + ReportMissing"):
            out = one(store_of((doc("VJC23S", 2023, "separate"), [])))
            self.assertEqual((type(out), out.available_scopes), (ReportMissing, ("separate",)))
        with self.subTest("ReportMissing + NotFound"):
            out = one(store_of((doc("VJC24C", 2024), [is_table("VJC24C", no_row24)])))
            self.assertIsInstance(out, NotFound)
            self.assertEqual(out.reason, "no_row")
            self.assertEqual(out.searched_doc_ids, ("VJC24C",))
            self.assertIn("primary_report_missing", out.warnings)
        with self.subTest("NotFound + ReportMissing"):
            out = one(store_of((doc("VJC23C", 2023), [is_table("VJC23C", no_row)])))
            self.assertIsInstance(out, NotFound)
            self.assertEqual(out.searched_doc_ids, ("VJC23C",))
            self.assertNotIn("primary_report_missing", out.warnings)
        with self.subTest("NotFound + NotFound"):
            out = one(store_of((doc("VJC23C", 2023), [is_table("VJC23C", no_row)]),
                               (doc("VJC24C", 2024), [is_table("VJC24C", no_row24)])))
            self.assertIsInstance(out, NotFound)
            self.assertEqual(out.searched_doc_ids, ("VJC23C", "VJC24C"))
        with self.subTest("NotFound + Found"):
            out = one(store_of((doc("VJC23C", 2023), [is_table("VJC23C", no_row)]),
                               (doc("VJC24C", 2024), [is_table("VJC24C", is_rows(2024))])))
            self.assertIsInstance(out, Found)
            self.assertEqual(out.extracted.warnings, ("from_next_year_report:VJC24C",))
        with self.subTest("ReportMissing + BlankValue"):
            out = one(store_of((doc("VJC24C", 2024), [is_table("VJC24C", is_rows(2024, prior="-"))])))
            self.assertIsInstance(out, BlankValue)
            self.assertEqual(out.location.doc_id, "VJC24C")

    def test_not_found_reasons(self):
        stale = store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2021))]))  # columns 2021, 2020
        out = one(stale)
        self.assertEqual((type(out), out.reason), (NotFound, "no_period_column"))
        no_section = _BANK_BS_HEADER + [
            ["", "Cho vay khách hàng", "", "1.000", "900"],
            ["", "TỔNG TÀI SẢN", "", "607.875.185", "527.769.944"],
        ]
        store = store_of((doc("ACB22C", 2022, ticker="ACB"),
                          [record("ACB22C", 2, 7, no_section, kind="balance_sheet")]))
        out = one(store, LeafRequest("customer_loans_gross", 2022, "closing"), "ACB")
        self.assertEqual((type(out), out.reason), (NotFound, "section_unresolved"))

    def test_primary_ambiguous_does_not_fall_back(self):
        store = store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, "1.234.567 (*)"))]),
                         (doc("VJC24C", 2024), [is_table("VJC24C", is_rows(2024))]))
        out = one(store)
        self.assertIsInstance(out, Ambiguous)
        self.assertEqual(store.documents_queried, [("VJC", 2023)])

    def test_variants_agree_disagree_onesided(self):
        d1, d2 = doc("VJC23C1", 2023, variant=1), doc("VJC23C2", 2023, variant=2)
        millions = is_table("VJC23C1", is_rows(2023, "(1.950.254)", "(1.383.641)", unit="Triệu VND"))

        def with_second(rows):
            return store_of((d1, [millions]), (d2, [is_table("VJC23C2", rows)]))

        with self.subTest("agree: finer scale kept"):
            out = one(with_second(is_rows(2023)))
            self.assertIsInstance(out, Found)
            self.assertEqual(out.extracted.fact.source.doc_id, "VJC23C2")
            self.assertEqual(out.extracted.fact.source.scale, D(1))
            self.assertEqual(out.extracted.warnings, ())
        with self.subTest("disagree: Conflict"):
            out = one(with_second(is_rows(2023, "(1.000.000.000.000)")))
            self.assertIsInstance(out, Conflict)
            self.assertEqual([c.location.doc_id for c in out.candidates], ["VJC23C1", "VJC23C2"])
            self.assertEqual([c.value for c in out.candidates], [D("-1950254000000"), D("-1000000000000")])
        with self.subTest("one-sided: not_in_variant"):
            out = one(with_second(is_rows(2023, interest=False)))
            self.assertIsInstance(out, Found)
            self.assertEqual(out.extracted.fact.source.doc_id, "VJC23C1")
            self.assertEqual(out.extracted.warnings, ("not_in_variant:VJC23C2",))
        with self.subTest("other outcome beside Found: variant_outcome"):
            out = one(with_second(is_rows(2023, "-")))
            self.assertIsInstance(out, Found)
            self.assertEqual(out.extracted.warnings, ("variant_outcome:VJC23C2:blank_value",))
        with self.subTest("no Found: first non-NotFound in doc-id order"):
            store = store_of((d1, [is_table("VJC23C1", is_rows(2023, interest=False))]),
                             (d2, [is_table("VJC23C2", is_rows(2023, "-"))]))
            self.assertIsInstance(one(store), BlankValue)

    def test_multiple_rows_and_columns(self):
        def acb(total_on_p8):
            p8 = _BANK_BS_HEADER + [
                ["", "Tiền gửi của khách hàng", "", "1.000", "900"],
                ["", "TỔNG NGUỒN VỐN", "", total_on_p8, "527.769.944"],
            ]
            return store_of((doc("ACB22C", 2022, ticker="ACB"), [
                record("ACB22C", 2, 7, ACB_BS_2022, kind="balance_sheet"),
                record("ACB22C", 3, 8, p8, kind="balance_sheet"),
            ]))

        request = LeafRequest("total_assets", 2022, "closing")
        out = one(acb("607.875.185"), request, "ACB")
        self.assertIsInstance(out, Found)
        self.assertEqual(out.extracted.fact.source.page, 7)
        self.assertEqual(out.extracted.fact.value.value, D("607875185000000"))
        out = one(acb("600.000.000"), request, "ACB")
        self.assertIsInstance(out, Ambiguous)
        self.assertEqual(out.reason, "multiple_rows")
        self.assertEqual(sorted(c.location.page for c in out.candidates), [7, 8])

        def two_current(left, right):
            rows = [
                ["Mã số", "", "Thuyết minh", "Năm nay VND", "2023 VND"],
                ["01", "Doanh thu", "", "5", "5"],
                ["10", "Doanh thu thuần", "", "5", "5"],
                ["23", "Chi phí lãi vay", "", left, right],
            ]
            return store_of((doc("VJC23C", 2023), [is_table("VJC23C", rows)]))

        out = one(two_current("7", "7"))
        self.assertIsInstance(out, Found)
        self.assertEqual(out.extracted.fact.source.column_header, "Năm nay VND")
        out = one(two_current("7", "8"))
        self.assertIsInstance(out, Ambiguous)
        self.assertEqual(out.reason, "multiple_period_columns")
        self.assertEqual([c.value for c in out.candidates], [D(7), D(8)])

    def test_blank_unit_unparsed_outcomes(self):
        blank = one(store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, "-"))])))
        self.assertIsInstance(blank, BlankValue)
        self.assertEqual(blank.location.raw_text, "-")
        unit = one(store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, unit=""))])))
        self.assertIsInstance(unit, UnitUnknown)
        self.assertEqual(unit.reason, "missing")
        unparsed = one(store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, "12,3.4"))])))
        self.assertIsInstance(unparsed, Ambiguous)
        self.assertEqual(unparsed.reason, "number_unparsed")

    def test_unparsed_cell_outcome(self):
        out = one(store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, "1.234.567 (*)"))])))
        self.assertIsInstance(out, Ambiguous)
        self.assertEqual(out.reason, "number_unparsed")
        self.assertEqual(len(out.candidates), 1)
        self.assertIsNone(out.candidates[0].value)
        self.assertEqual(out.candidates[0].location.raw_text, "1.234.567 (*)")

    def test_template_outcomes(self):
        out = extract("FTS", "consolidated", [INTEREST_2023]).outcomes[0]
        self.assertEqual((type(out), out.template), (UnsupportedTemplate, "securities"))
        self.assertIsInstance(extract("ZZZ", "consolidated", [INTEREST_2023]).outcomes[0], UnknownTicker)
        out = extract("ACB", "consolidated", [LeafRequest("gross_revenue", 2022, "flow")]).outcomes[0]
        self.assertEqual((type(out), out.template), (NotApplicable, "bank"))

    def test_lowercase_ticker(self):
        out = extract("acb", "consolidated", [LeafRequest("total_assets", 2022, "closing")]).outcomes[0]
        self.assertEqual(out.extracted.fact.ticker, "ACB")
        store = default_store()
        result = extract(" acb ", "consolidated", [LeafRequest("total_assets", 2022, "closing")], store=store)
        self.assertEqual((result.ticker, result.outcomes[0].ticker), ("ACB", "ACB"))
        self.assertEqual(store.documents_queried, [("ACB", 2022)])

    def test_validation_errors(self):
        store = default_store()
        with self.assertRaises(ValueError):
            extract("VJC", "parent", [INTEREST_2023], store=store)
        with self.assertRaises(TypeError):
            extract("VJC", "consolidated", [("interest_expense", 2023, "flow")], store=store)
        with self.assertRaises(ValueError):
            extract("VJC", "consolidated", [LeafRequest("interest_expense", 2023, "closing")], store=store)
        self.assertEqual(store.sessions_opened, 0)
        # R4: no leaf for the ticker's template (or no template) -> no period check, typed outcome.
        out = extract("ACB", "consolidated", [LeafRequest("gross_revenue", 2022, "closing")]).outcomes[0]
        self.assertIsInstance(out, NotApplicable)
        out = extract("ZZZ", "consolidated", [LeafRequest("interest_expense", 2023, "closing")]).outcomes[0]
        self.assertIsInstance(out, UnknownTicker)

    def test_request_order_and_dedup(self):
        missing = LeafRequest("interest_expense", 2019, "flow")
        result = extract("VJC", "consolidated", [NET_REVENUE_2023, missing, INTEREST_2023,
                                                 LeafRequest("net_revenue", 2023, "flow")])
        self.assertEqual([o.request for o in result.outcomes], [NET_REVENUE_2023, missing, INTEREST_2023])
        self.assertEqual([type(o) for o in result.outcomes], [Found, ReportMissing, Found])
        self.assertEqual([f.fact.metric_id for f in result.facts], ["net_revenue", "interest_expense"])
        facts = result.fact_set()
        self.assertIsInstance(facts, FactSet)
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts.find("interest_expense", "VJC", 2023, "consolidated", "flow").value.value,
                         D("-1950254195864"))

    def test_malformed_table_reported(self):
        bad_json = dataclasses.replace(is_table("VJC23C", is_rows(2023), table_id=2), rows_json="{not json")
        no_chunk = is_table("VJC23C", is_rows(2023), table_id=3)
        no_chunk = dataclasses.replace(no_chunk, chunk_id=None)
        good = is_table("VJC23C", is_rows(2023, interest=False))
        out = one(store_of((doc("VJC23C", 2023), [good, bad_json, no_chunk])))
        self.assertIsInstance(out, NotFound)
        self.assertEqual(out.skipped_tables, ("VJC23C:2", "VJC23C:3"))
        self.assertEqual(out.warnings, ("skipped_table:VJC23C:2", "table_without_chunk:VJC23C:3"))
        # A fact from the same document keeps the skip warnings visible.
        found = one(store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023)), bad_json])))
        self.assertIsInstance(found, Found)
        self.assertEqual(found.extracted.warnings, ("skipped_table:VJC23C:2",))

    def test_one_session_per_call(self):
        store = store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, interest=False))]),
                         (doc("VJC24C", 2024), [is_table("VJC24C", is_rows(2024))]))
        result = extract("VJC", "consolidated", [INTEREST_2023, NET_REVENUE_2023,
                                                 LeafRequest("gross_revenue", 2023, "flow")], store=store)
        self.assertEqual([type(o) for o in result.outcomes], [Found, Found, Found])
        self.assertEqual(store.sessions_opened, 1)
        self.assertEqual(store.tables_read, {"VJC23C": 1, "VJC24C": 1})
        self.assertEqual(store.documents_queried, [("VJC", 2023), ("VJC", 2024)])

    def test_fallback_called_only_for_not_found(self):
        no_code = store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, interest=False))]))
        stub = StubLocator(read_label("- Trong đó: Chi phí lãi vay"))
        out = one(no_code, fallback=stub)
        self.assertEqual(stub.calls, [("interest_expense", INTEREST_2023, "primary", ("VJC23C",))])
        self.assertIsInstance(out, Found)
        self.assertEqual(out.extracted.method, "llm_located")
        self.assertEqual(out.extracted.warnings, ("llm_ok",))
        self.assertEqual(out.extracted.fact.value.value, D("-1950254195864"))

        with self.subTest("not called for Ambiguous / ReportMissing"):
            stub = StubLocator(read_label("x"))
            unparsed = store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, "1.234.567 (*)"))]))
            self.assertIsInstance(one(unparsed, fallback=stub), Ambiguous)
            self.assertIsInstance(one(no_code, LeafRequest("interest_expense", 2019, "flow"), fallback=stub),
                                  ReportMissing)
            self.assertEqual(stub.calls, [])

        with self.subTest("rejected pointer"):
            stub = StubLocator(lambda views: LocatorResult(None, "llm_pointer_rejected",
                                                           ("llm_pointer_rejected:period",)))
            out = one(no_code, fallback=stub)
            self.assertEqual((type(out), out.reason), (NotFound, "llm_pointer_rejected"))
            self.assertIn("llm_pointer_rejected:period", out.warnings)

        with self.subTest("not_present keeps the deterministic reason"):
            out = one(no_code, fallback=StubLocator(lambda views: LocatorResult(None, None, ())))
            self.assertEqual((type(out), out.reason), (NotFound, "no_row"))

        with self.subTest("fallback documents when the primary report is missing"):
            only_next = store_of((doc("VJC24C", 2024), [is_table("VJC24C", is_rows(2024, interest=False))]))
            stub = StubLocator(read_label("- Trong đó: Chi phí lãi vay", col=4))
            out = one(only_next, fallback=stub)
            self.assertEqual(stub.calls, [("interest_expense", INTEREST_2023, "fallback", ("VJC24C",))])
            self.assertIsInstance(out, Found)
            self.assertEqual(set(out.extracted.warnings),
                             {"llm_ok", "from_next_year_report:VJC24C", "primary_report_missing"})


if __name__ == "__main__":
    unittest.main()

import unittest
from decimal import Decimal as D

from src.calculation.facts import Fact, FactKey, FactSet, FactSource, MissingFact
from src.calculation.helpers import (
    Answer,
    ExecutionContext,
    MetricUnavailable,
    Trace,
    TraceStep,
    build_helpers,
)
from src.calculation.metric_tree import MetricTree
from src.calculation.units import (
    CalculationError,
    Dimension as Dm,
    Quantity,
    UnitError,
    round_display,
)


def M(v):
    return Quantity(D(v), Dm.MONEY)


def R(v):
    return Quantity(D(v), Dm.RATIO)


def leaf(id, *, period="flow", codes=("1",), aliases=None, kind="income_statement"):
    return {"id": id, "label_vi": id, "aliases": aliases or [f"alias {id}", f"nhan {id}"],
            "statement_kind": kind, "codes": list(codes), "period_kind": period,
            "source": "statement", "dimension": "money"}


def derived(id, formulas, *, dim="money", templates=("corporate",)):
    return {"id": id, "label_vi": id, "aliases": [f"alias {id}"], "templates": list(templates),
            "dimension": dim, "formulas": list(formulas)}


def make_tree() -> MetricTree:
    corporate = [
        leaf("net_revenue"), leaf("cost_of_goods_sold"), leaf("gross_profit_reported"),
        leaf("profit_before_tax"), leaf("interest_expense"),
        leaf("total_assets", period="closing", kind="balance_sheet"),
    ]
    bank = [leaf("profit_before_tax", codes=()),
            leaf("total_assets", period="closing", codes=(), kind="balance_sheet")]
    return MetricTree.from_entries({"corporate": corporate, "bank": bank}, [
        derived("gross_profit", ["gross_profit_reported", "net_revenue - abs(cost_of_goods_sold)"]),
        derived("gross_margin", ["gross_profit / net_revenue"], dim="ratio"),
        derived("ebit", ["profit_before_tax + abs(interest_expense)"]),
        derived("interest_coverage", ["ebit / abs(interest_expense)"], dim="ratio"),
    ])


def fact(metric_id, value, *, ticker="VJC", year=2023, period="flow",
         scope="consolidated", scale="1"):
    src = FactSource(chunk_id=f"c-{metric_id}-{year}", doc_id="d1", page=1, row_label=metric_id,
                     raw_text=str(value), unit_text="VND", scale=D(scale))
    return Fact(metric_id, ticker, year, scope, period, M(str(value)), src)


VJC_2023 = [
    fact("profit_before_tax", 606103607021),
    fact("interest_expense", -1950254195864),
]


def context(**overrides) -> ExecutionContext:
    kwargs = dict(default_scope="consolidated",
                  ticker_templates={"VJC": "corporate", "HPG": "corporate", "ACB": "bank"},
                  question_literals=frozenset(), requested_unit=None, requested_kind="quantity")
    kwargs.update(overrides)
    return ExecutionContext(**kwargs)


def helpers(facts=(), **ctx):
    trace = Trace()
    return build_helpers(make_tree(), FactSet(facts), context(**ctx), trace), trace


class MetricTests(unittest.TestCase):
    def test_icr_with_negative_printed_interest(self):
        h, _ = helpers(VJC_2023)
        q = h["metric"]("interest_coverage", "VJC", 2023)
        self.assertIs(q.dimension, Dm.RATIO)
        self.assertGreater(q.value, 0)
        self.assertEqual(round_display(q.value, 4), D("1.3108"))

    def test_trace_records_facts_and_metrics(self):
        h, trace = helpers(VJC_2023)
        h["metric"]("interest_coverage", "VJC", 2023)
        self.assertEqual(trace.metrics_used(), frozenset({"interest_coverage", "ebit"}))
        self.assertEqual(trace.facts_used(), (
            FactKey("profit_before_tax", "VJC", 2023, "consolidated", "flow"),
            FactKey("interest_expense", "VJC", 2023, "consolidated", "flow"),
        ))
        metric_steps = [s for s in trace.steps if s.helper == "metric"]
        self.assertEqual([s.args[0] for s in metric_steps], ["ebit", "interest_coverage"])
        self.assertEqual(metric_steps[-1].args, ("interest_coverage", "VJC", 2023, "consolidated"))
        self.assertEqual(len(metric_steps[-1].fact_keys), 2)
        self.assertEqual(trace.cross_checks(), [])

    def test_alternative_cross_check_recorded(self):
        h, trace = helpers([
            fact("gross_profit_reported", 300, scale="1000000"),
            fact("net_revenue", 1000),
            fact("cost_of_goods_sold", -705),
        ])
        q = h["metric"]("gross_profit", "VJC", 2023)
        self.assertEqual(q, M("300"))  # first alternative wins
        checks = trace.cross_checks()
        self.assertEqual(len(checks), 1)
        self.assertIsInstance(checks[0], TraceStep)
        self.assertEqual(dict(checks[0].detail), {
            "node": "gross_profit", "primary": "300", "alternative": "295",
            "tolerance": "1000002"})
        # facts of the cross-check alternative are not facts the answer used
        self.assertEqual([k.metric_id for k in trace.facts_used()], ["gross_profit_reported"])

    def test_second_alternative_used_when_first_missing(self):
        h, trace = helpers([fact("net_revenue", 1000), fact("cost_of_goods_sold", -700)])
        self.assertEqual(h["metric"]("gross_profit", "VJC", 2023), M("300"))
        self.assertEqual(trace.cross_checks(), [])

    def test_nested_cross_check_and_margin(self):
        h, trace = helpers([
            fact("gross_profit_reported", 300), fact("net_revenue", 1000),
            fact("cost_of_goods_sold", -700),
        ])
        self.assertEqual(h["metric"]("gross_margin", "VJC", 2023), R("0.3"))
        self.assertEqual(len(trace.cross_checks()), 1)
        self.assertEqual(trace.metrics_used(), frozenset({"gross_margin", "gross_profit"}))
        self.assertEqual([s.args[0] for s in trace.steps if s.helper == "metric"][-1],
                         "gross_margin")

    def test_metric_wrong_template_raises(self):
        h, trace = helpers([fact("net_revenue", 1000, ticker="ACB", year=2022)])
        with self.assertRaises(MetricUnavailable):
            h["metric"]("gross_margin", "ACB", 2022)
        self.assertEqual(trace.steps, [])

    def test_metric_unknown_ticker_or_node_raises(self):
        h, _ = helpers(VJC_2023)
        with self.assertRaises(MetricUnavailable):
            h["metric"]("interest_coverage", "ZZZ", 2023)
        with self.assertRaises(MetricUnavailable):
            h["metric"]("no_such_metric", "VJC", 2023)

    def test_metric_no_alternative_available_raises(self):
        h, trace = helpers([fact("profit_before_tax", 1)])
        with self.assertRaises(MetricUnavailable):
            h["metric"]("interest_coverage", "VJC", 2023)
        self.assertEqual(trace.steps, [])

    def test_metric_accepts_leaf_ids(self):
        h, trace = helpers([fact("total_assets", 500, ticker="ACB", year=2022, period="closing")])
        self.assertEqual(h["metric"]("total_assets", "ACB", 2022), M("500"))
        self.assertEqual(trace.metrics_used(), frozenset({"total_assets"}))
        self.assertEqual(trace.facts_used(),
                         (FactKey("total_assets", "ACB", 2022, "consolidated", "closing"),))
        with self.assertRaises(MetricUnavailable):
            h["metric"]("total_assets", "ACB", 2021)

    def test_metric_scope_and_decimal_year(self):
        h, trace = helpers([fact("profit_before_tax", 7, scope="separate")])
        self.assertEqual(h["metric"]("profit_before_tax", "VJC", D("2023"), scope="separate"),
                         M("7"))
        self.assertEqual(trace.steps[-1].args, ("profit_before_tax", "VJC", 2023, "separate"))
        with self.assertRaises(MetricUnavailable):
            h["metric"]("profit_before_tax", "VJC", 2023)
        with self.assertRaises(ValueError):
            h["metric"]("profit_before_tax", "VJC", 2023, scope="group")
        with self.assertRaises(ValueError):
            h["metric"]("profit_before_tax", "VJC", D("2023.5"))
        with self.assertRaises(TypeError):
            h["metric"]("profit_before_tax", "VJC", True)


class FactTests(unittest.TestCase):
    def test_missing_fact_raises(self):
        h, trace = helpers(VJC_2023)
        with self.assertRaises(MissingFact):
            h["fact"]("net_revenue", "VJC", 2023)
        self.assertEqual(trace.steps, [])

    def test_fact_reads_and_traces(self):
        h, trace = helpers(VJC_2023)
        self.assertEqual(h["fact"]("profit_before_tax", "VJC", 2023), M("606103607021"))
        self.assertEqual(trace.steps[0].helper, "fact")
        self.assertEqual(trace.facts_used(),
                         (FactKey("profit_before_tax", "VJC", 2023, "consolidated", "flow"),))
        self.assertEqual(trace.metrics_used(), frozenset())
        with self.assertRaises(ValueError):
            h["fact"]("profit_before_tax", "VJC", 2023, period="opening")

    def test_avg_balance(self):
        h, trace = helpers([
            fact("total_assets", 1200, year=2024, period="closing"),
            fact("total_assets", 1000, year=2023, period="closing"),
        ])
        self.assertEqual(h["avg_balance"]("total_assets", "VJC", 2024), M("1100"))
        self.assertEqual(len(trace.facts_used()), 2)
        with self.assertRaises(MissingFact):
            h["avg_balance"]("total_assets", "VJC", 2023)


class CalculationHelperTests(unittest.TestCase):
    def setUp(self):
        self.h, self.trace = helpers()

    def test_growth_and_cagr(self):
        self.assertEqual(self.h["growth_pct"](M(120), M(100)), R("0.2"))
        self.assertEqual(self.h["growth_pct"](M(-50), M(-100)), R("0.5"))
        self.assertEqual(round(self.h["cagr"](M(121), M(100), 2).value, 6), D("0.100000"))
        self.assertEqual(round(self.h["cagr"](M(121), M(100), D("2")).value, 6), D("0.100000"))
        self.assertIs(self.h["cagr"](M(121), M(100), 2).dimension, Dm.RATIO)
        with self.assertRaises(CalculationError):
            self.h["growth_pct"](M(1), M(0))
        with self.assertRaises(UnitError):
            self.h["growth_pct"](M(1), R(1))

    def test_cagr_rejects_nonpositive(self):
        for end, start, years in ((M(0), M(100), 2), (M(121), M(-100), 2),
                                  (M(-121), M(-100), 2), (M(121), M(100), 0)):
            with self.subTest(end=end, start=start, years=years):
                with self.assertRaises(CalculationError):
                    self.h["cagr"](end, start, years)
        with self.assertRaises(TypeError):
            self.h["cagr"](M(121), M(100), True)

    def test_pct_point_change(self):
        self.assertEqual(self.h["pct_point_change"](R("0.25"), R("0.2")), R("0.05"))
        with self.assertRaises(UnitError):
            self.h["pct_point_change"](M(1), M(2))

    def test_median_even_and_odd(self):
        self.assertEqual(self.h["median"]([M(3), M(1), M(2)]), M(2))
        self.assertEqual(self.h["median"]([M(4), M(1), M(3), M(2)]), M("2.5"))
        self.assertEqual(self.h["mean"]([R("0.1"), R("0.2"), R("0.6")]), R("0.3"))
        for name in ("median", "mean"):
            with self.subTest(name=name):
                with self.assertRaises(CalculationError):
                    self.h[name]([])
                with self.assertRaises(UnitError):
                    self.h[name]([M(1), R(1)])

    def test_argmax_tie_raises(self):
        with self.assertRaises(CalculationError):
            self.h["argmax"]({"VJC": R("1.5"), "HPG": R("1.50"), "ACB": R("1")})
        with self.assertRaises(CalculationError):
            self.h["argmin"]({"VJC": R("1"), "HPG": R("1")})
        with self.assertRaises(CalculationError):
            self.h["argmax"]({})

    def test_argmax_argmin(self):
        data = {"VJC": R("1.2"), "HPG": R("1.5"), "ACB": R("0.9")}
        self.assertEqual(self.h["argmax"](data), "HPG")
        self.assertEqual(self.h["argmin"](data), "ACB")
        # a tie that is later beaten is not a tie
        self.assertEqual(self.h["argmax"]({"a": M(1), "b": M(1), "c": M(2)}), "c")

    def test_filter_and_count(self):
        percent = self.h["percent"]
        self.assertEqual(self.h["filter_by"]({"A": R("0.4"), "B": R("0.8")}, ">", percent(50)),
                         {"B": R("0.8")})
        self.assertEqual(self.h["count_if"]([R("1.2"), R("1.6")], "<", D("1.5")), 1)
        self.assertEqual(self.h["count_if"]([R("1.5"), R("1.6")], "==", D("1.50")), 1)
        self.assertEqual(self.h["filter_by"]({"A": M(5), "B": M(-1)}, ">=", 0), {"A": M(5)})
        with self.assertRaises(ValueError):
            self.h["filter_by"]({"A": R(1)}, "!=", D(1))
        with self.assertRaises(ValueError):
            self.h["count_if"]([], "<>", D(1))
        with self.assertRaises(UnitError):
            self.h["count_if"]([M(1)], "<", D(5))

    def test_money_threshold(self):
        self.assertEqual(self.h["money"](20, "ty"), M("20000000000"))
        self.assertEqual(self.h["money"](D("1.5"), "trieu"), M("1500000"))
        self.assertEqual(self.h["percent"](D("12.5")), R("0.125"))
        for bad in (1.5, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.h["money"](bad, "ty")
                with self.assertRaises(TypeError):
                    self.h["percent"](bad)
        with self.assertRaises(ValueError):
            self.h["money"](1, "bogus")

    def test_answer_kinds(self):
        answer = self.h["answer"]
        self.assertEqual(answer(2023, "year"), Answer("year", 2023, "year"))
        self.assertEqual(answer(D("2023"), "year").value, 2023)
        self.assertIsInstance(answer(D("2023"), "year").value, int)
        q = M("5000000000")
        self.assertEqual(answer(q, "ty"), Answer("quantity", q, "ty"))
        self.assertEqual(answer(R("0.3"), "percent").kind, "quantity")
        self.assertEqual(answer("VJC", "ticker").kind, "ticker")
        self.assertEqual(answer(3, "count"), Answer("count", 3, "count"))
        self.assertEqual(answer(True, "boolean"), Answer("boolean", True, "boolean"))
        bad = [(1, "bogus"), (q, "year"), (2023, "ty"), (R("0.3"), "ty"), (-1, "count"),
               (True, "count"), (1, "boolean"), ("", "ticker"), (D("2023.5"), "year")]
        for value, unit in bad:
            with self.subTest(value=value, unit=unit):
                with self.assertRaises(ValueError):
                    answer(value, unit)

    def test_every_call_is_traced(self):
        self.h["money"](20, "ty")
        self.h["growth_pct"](M(120), M(100))
        self.h["answer"](2023, "year")
        self.assertEqual([s.helper for s in self.trace.steps], ["money", "growth_pct", "answer"])
        self.assertTrue(all(isinstance(s.result, str) for s in self.trace.steps))

    def test_build_helpers_names(self):
        self.assertEqual(set(self.h), {
            "fact", "metric", "avg_balance", "growth_pct", "cagr", "pct_point_change", "mean",
            "median", "argmax", "argmin", "filter_by", "count_if", "money", "percent", "answer"})

    def test_builtin_sum_over_money(self):
        self.assertEqual(sum([M(1), M(2)]), M(3))


if __name__ == "__main__":
    unittest.main()

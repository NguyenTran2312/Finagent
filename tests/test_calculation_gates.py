import subprocess
import sys
import unittest
from decimal import Decimal as D
from pathlib import Path

from src.calculation.facts import Fact, FactKey, FactSet, FactSource
from src.calculation.gates import (
    ALLOWED_CONSTANTS,
    GateIssue,
    GateReport,
    compare_answers,
    question_literals,
    requested_answer_kind,
    requested_unit,
    run_gates,
)
from src.calculation.helpers import Answer, ExecutionContext, Trace, TraceStep, build_helpers
from src.calculation.metric_tree import MetricTree
from src.calculation.sandbox import ExecutionFailure, ExecutionResult
from src.calculation.units import Dimension as Dm, Quantity

TREE = MetricTree.load()
ROOT = Path(__file__).resolve().parents[1]


def M(v):
    return Quantity(D(v), Dm.MONEY)


def R(v):
    return Quantity(D(v), Dm.RATIO)


def fact(metric_id, value, *, ticker="VJC", year=2023, scale="1", restated=False,
         template="corporate", scope="consolidated"):
    period = TREE.leaf(metric_id, template).period_kind
    source = FactSource(chunk_id=f"syn:gates:{metric_id}:{ticker}:{year}", doc_id="syn-doc",
                        page=1, row_label=metric_id, raw_text=str(value), unit_text="VND",
                        scale=D(scale))
    return Fact(metric_id, ticker, year, scope, period, M(value), source, restated)


VJC = [
    fact("profit_before_tax", "1217635208744"),
    fact("interest_expense", "-1950254195864"),
    fact("net_revenue", "71925960000000", scale="1000000"),
    fact("cost_of_goods_sold", "-67000000000000", scale="1000000"),
    fact("gross_profit_reported", "4925960000000", scale="1000000"),
]


def context(question="", *, unit=None, kind="quantity", named=(), literals=None):
    return ExecutionContext(
        default_scope="consolidated", ticker_templates={"VJC": "corporate", "ACB": "bank"},
        question_literals=question_literals(question) if literals is None else literals,
        requested_unit=unit, requested_kind=kind, named_metrics=frozenset(named))


def execute(program, facts=VJC, ctx=None):
    """Run `program(h)` in-process against real helpers; no sandbox spawn."""
    ctx = ctx or context()
    facts = facts if isinstance(facts, FactSet) else FactSet(facts)
    trace = Trace()
    result = program(build_helpers(TREE, facts, ctx, trace))
    return ExecutionResult(result, trace), facts, ctx


def gates(code, program, *, facts=VJC, ctx=None, claimed=None):
    result, factset, ctx = execute(program, facts, ctx)
    return run_gates(code, result, TREE, factset, ctx, claimed)


def errors(report, code):
    return [i for i in report.issues if i.code == code and i.severity == "error"]


ICR_CODE = """\
def solution():
    return answer(metric("interest_coverage", "VJC", 2023), "times")
"""


def icr_program(h):
    return h["answer"](h["metric"]("interest_coverage", "VJC", 2023), "times")


class QuestionParsingTests(unittest.TestCase):
    def test_question_literals(self):
        ql = question_literals(
            "có bao nhiêu DN có chi phí lớn hơn 20 tỷ đồng và ICR dưới 1,5 lần, giảm 10%")
        self.assertLessEqual({D(20), D("20000000000"), D("1.5"), D(10), D("0.10")}, ql)

    def test_question_literals_formats(self):
        ql = question_literals("Năm 2019–2023, doanh thu 1.234,5 tỷ đồng, 3 nghìn tỷ, "
                               "tăng 12,5 phần trăm, 2 trăm tỷ, 7 triệu đồng")
        self.assertLessEqual({D(2019), D(2023), D("1234.5"), D("1234500000000"),
                              D(3), D("3000000000000"), D("12.5"), D("0.125"),
                              D("200000000000"), D("7000000")}, ql)
        self.assertEqual(question_literals("Doanh thu là bao nhiêu?"), frozenset())

    def test_requested_kind_without_unit(self):
        question = "Năm nào có số dư lớn nhất?"
        self.assertEqual(requested_answer_kind(question), "year")
        self.assertIsNone(requested_unit(question))
        self.assertEqual(requested_answer_kind("có bao nhiêu doanh nghiệp có ...?"), "count")
        self.assertIsNone(requested_unit("có bao nhiêu doanh nghiệp có ICR dưới 1,5 lần?"))
        self.assertEqual(requested_answer_kind("Công ty nào có ROE cao nhất?"), "ticker")
        self.assertEqual(requested_answer_kind("Ngân hàng nào có LDR thấp nhất?"), "ticker")
        plain = "Lợi nhuận gộp năm 2023 là bao nhiêu?"
        self.assertEqual(requested_answer_kind(plain), "quantity")
        self.assertIsNone(requested_unit(plain))

    def test_requested_unit_variants(self):
        cases = {
            "Biên lãi gộp thay đổi bao nhiêu điểm phần trăm?": "percentage_points",
            "Tăng trưởng doanh thu là bao nhiêu %?": "percent",
            "Biên lợi nhuận gộp là bao nhiêu (%)?": "percent",
            "Tăng bao nhiêu phần trăm so với năm 2022?": "percent",
            "ICR đạt bao nhiêu lần?": "times",
            "Vòng quay tổng tài sản là bao nhiêu vòng?": "times",
            "Doanh thu là bao nhiêu tỷ đồng?": "ty",
            "Vốn chủ sở hữu là bao nhiêu nghìn tỷ đồng?": "nghin_ty",
            "Chi phí là bao nhiêu triệu đồng?": "trieu",
            "EPS là bao nhiêu đồng?": "dong",
            "Số ngày tồn kho là bao nhiêu ngày?": "days",
            "Lợi nhuận là bao nhiêu tỷ đồng nếu doanh thu giảm 10%?": "ty",
            "Tổng tài sản năm 2023 là bao nhiêu? (đơn vị: tỷ đồng)": "ty",
        }
        for question, unit in cases.items():
            with self.subTest(question):
                self.assertEqual(requested_unit(question), unit)

    def test_requested_unit_ignores_dates_and_unknown(self):
        self.assertIsNone(requested_unit("Tổng tài sản tại ngày 31/12/2023 là bao nhiêu?"))
        self.assertIsNone(requested_unit("Doanh thu giảm 10% thì lợi nhuận gộp là bao nhiêu?"))
        self.assertIsNone(requested_unit("Doanh thu là bao nhiêu USD?"))


class ProvenanceTests(unittest.TestCase):
    def test_invented_literal_fails_provenance(self):
        code = ("def solution():\n"
                "    rev = fact(\"net_revenue\", \"VJC\", 2023)\n"
                "    return answer(rev * 0.37, \"ty\")\n")
        report = gates(code, lambda h: h["answer"](
            h["fact"]("net_revenue", "VJC", 2023) * D("0.37"), "ty"), ctx=context(unit="ty"))
        found = errors(report, "provenance")
        self.assertFalse(report.passed)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].line, 3)
        self.assertIn("0.37", found[0].message)

    def test_year_literal_allowed(self):
        code = ("def solution():\n"
                "    years = [2022, 2023]\n"
                "    x = metric(\"interest_coverage\", \"VJC\", years[-1]) * 100 / 100\n"
                "    return answer(x * 365 / 365 + 0 * 12 * 4 * 2, \"times\")\n")
        self.assertTrue(gates(code, icr_program).passed)
        for bad in ("1899", "2101", "2023.5", "-7"):
            with self.subTest(bad):
                report = gates(f"def solution():\n    x = {bad}\n" + ICR_CODE.splitlines()[1],
                               icr_program)
                self.assertEqual(len(errors(report, "provenance")), 1)
                self.assertEqual(errors(report, "provenance")[0].line, 2)

    def test_question_literals_allowed(self):
        code = ("def solution():\n"
                "    icr = metric(\"interest_coverage\", \"VJC\", 2023) * (1 - 0.10)\n"
                "    return answer(count_if([icr], \"<\", 1.5), \"count\")\n")
        question = "Nếu doanh thu giảm 10%, có bao nhiêu doanh nghiệp có ICR dưới 1,5 lần?"
        ctx = context(question, kind="count")

        def program(h):
            icr = h["metric"]("interest_coverage", "VJC", 2023) * (1 - D("0.10"))
            return h["answer"](h["count_if"]([icr], "<", D("1.5")), "count")

        self.assertTrue(gates(code, program, ctx=ctx).passed)
        self.assertFalse(gates(code, program, ctx=context(kind="count")).passed)

    def test_string_constants_ignored(self):
        code = 'def solution():\n    x = "0.37"\n' + ICR_CODE.splitlines()[1]
        self.assertTrue(gates(code, icr_program).passed)

    def test_unknown_fact_in_trace_fails_provenance(self):
        result, factset, ctx = execute(icr_program)
        ghost = FactKey("net_revenue", "HVN", 2023, "consolidated", "flow")
        result.trace.steps.append(TraceStep("fact", ("net_revenue", "HVN", 2023), "1 money",
                                            (ghost,)))
        report = run_gates(ICR_CODE, result, TREE, factset, ctx)
        found = errors(report, "provenance")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].trace_step, len(result.trace.steps) - 1)

    def test_unknown_cross_check_fact_fails_provenance(self):
        result, factset, ctx = execute(icr_program)
        ghost = FactKey("gross_profit_reported", "HVN", 2023, "consolidated", "flow")
        result.trace.steps.append(TraceStep(
            "cross_check", ("gross_profit", "HVN", 2023, "consolidated"), "1 money", (ghost,),
            {"node": "gross_profit", "primary": "1", "alternative": "1", "tolerance": "1"}))
        self.assertEqual(len(errors(run_gates(ICR_CODE, result, TREE, factset, ctx),
                                    "provenance")), 1)

    def test_unparseable_code_fails_provenance(self):
        result, factset, ctx = execute(icr_program)
        report = run_gates("def solution(:\n", result, TREE, factset, ctx)
        self.assertEqual(len(errors(report, "provenance")), 1)


class FormulaTests(unittest.TestCase):
    def test_hand_rolled_named_metric_fails_formula(self):
        code = ("def solution():\n"
                "    interest = abs(fact(\"interest_expense\", \"VJC\", 2023))\n"
                "    ebit = fact(\"profit_before_tax\", \"VJC\", 2023) + interest\n"
                "    return answer(ebit / interest, \"times\")\n")

        def program(h):
            interest = abs(h["fact"]("interest_expense", "VJC", 2023))
            ebit = h["fact"]("profit_before_tax", "VJC", 2023) + interest
            return h["answer"](ebit / interest, "times")

        ctx = context(unit="times", named={"interest_coverage"})
        report = gates(code, program, ctx=ctx)
        self.assertEqual(len(errors(report, "formula")), 1)
        self.assertIn("interest_coverage", errors(report, "formula")[0].message)
        self.assertTrue(gates(ICR_CODE, icr_program, ctx=ctx).passed)

    def test_named_leaf_satisfied_by_fact(self):
        code = 'def solution():\n    return answer(fact("net_revenue", "VJC", 2023), "ty")\n'
        ctx = context(unit="ty", named={"net_revenue"})
        report = gates(code, lambda h: h["answer"](h["fact"]("net_revenue", "VJC", 2023), "ty"),
                       ctx=ctx)
        self.assertTrue(report.passed, report.issues)

    def test_unknown_named_metric_is_error(self):
        report = gates(ICR_CODE, icr_program, ctx=context(named={"no_such_metric"}))
        self.assertEqual(len(errors(report, "formula")), 1)


class UnitTypeTests(unittest.TestCase):
    def test_ratio_answer_to_money_question_fails_unit(self):
        code = ICR_CODE.replace('"times"', '"percent"')
        report = gates(code, lambda h: h["answer"](
            h["metric"]("interest_coverage", "VJC", 2023), "percent"), ctx=context(unit="ty"))
        self.assertEqual(len(errors(report, "unit_type")), 1)

    def test_unit_compatibility(self):
        def money_program(unit):
            return lambda h: h["answer"](h["fact"]("net_revenue", "VJC", 2023), unit)

        self.assertTrue(gates("def solution(): pass", money_program("trieu"),
                              ctx=context(unit="ty")).passed)
        self.assertFalse(gates("def solution(): pass", lambda h: h["answer"](
            h["pct_point_change"](R("0.2"), R("0.1")), "percent"),
            ctx=context(unit="percentage_points")).passed)
        self.assertFalse(gates("def solution(): pass", money_program("ty"),
                               ctx=context(unit="times")).passed)

    def test_money_per_share_compatible_with_money_scale(self):
        eps = Answer("quantity", Quantity(D("2500"), Dm.MONEY_PER_SHARE), "money_per_share")
        result = ExecutionResult(eps, Trace())
        report = run_gates("def solution(): pass", result, TREE, FactSet([]), context(unit="dong"))
        self.assertTrue(report.passed, report.issues)

    def test_kind_mismatch(self):
        report = gates("def solution(): pass", lambda h: h["answer"](2021, "year"),
                       ctx=context(kind="count"))
        self.assertEqual(len(errors(report, "unit_type")), 1)

    def test_dimension_must_match_own_unit(self):
        bad = Answer("quantity", M("5"), "times")
        report = run_gates("def solution(): pass", ExecutionResult(bad, Trace()), TREE,
                           FactSet([]), context())
        self.assertEqual(len(errors(report, "unit_type")), 1)


class CrossCheckTests(unittest.TestCase):
    CODE = 'def solution():\n    return answer(metric("gross_profit", "VJC", 2023), "ty")\n'

    @staticmethod
    def program(h):
        return h["answer"](h["metric"]("gross_profit", "VJC", 2023), "ty")

    def with_reported(self, value, restated=False):
        others = [f for f in VJC if f.metric_id != "gross_profit_reported"]
        return others + [fact("gross_profit_reported", value, scale="1000000", restated=restated)]

    def test_cross_check_within_tolerance_passes(self):
        # tolerance = 3 facts x 10^6 (D8); 2,000,000 off is within it
        report = gates(self.CODE, self.program, facts=self.with_reported("4925962000000"))
        self.assertTrue(report.passed, report.issues)
        self.assertEqual(report.issues, ())

    def test_cross_check_mismatch_error(self):
        report = gates(self.CODE, self.program, facts=self.with_reported("4925970000000"))
        found = errors(report, "cross_check")
        self.assertEqual(len(found), 1)
        self.assertIsNotNone(found[0].trace_step)

    def test_restated_mismatch_is_warning(self):
        report = gates(self.CODE, self.program,
                       facts=self.with_reported("4925970000000", restated=True))
        self.assertTrue(report.passed)
        self.assertEqual([(i.code, i.severity) for i in report.issues],
                         [("cross_check", "warning")])

    def test_non_money_node_uses_relative_tolerance(self):
        def report_for(primary, alternative):
            trace = Trace([TraceStep("metric", ("gross_margin", "VJC", 2023, "consolidated"),
                                     f"{primary} ratio", ()),
                           TraceStep("cross_check", ("gross_margin", "VJC", 2023, "consolidated"),
                                     f"{alternative} ratio", (),
                                     {"node": "gross_margin", "primary": primary,
                                      "alternative": alternative, "tolerance": "0"})])
            result = ExecutionResult(Answer("quantity", R(primary), "percent"), trace)
            return run_gates("def solution(): pass", result, TREE, FactSet([]), context())

        self.assertTrue(report_for("0.2", "0.2004").passed)
        self.assertFalse(report_for("0.2", "0.2005").passed)
        self.assertTrue(report_for("0", "0").passed)
        self.assertFalse(report_for("0", "0.0001").passed)


class SanityTests(unittest.TestCase):
    def test_sanity_out_of_range_warning_only(self):
        facts = [fact("customer_loans_gross", "12000000000000", ticker="ACB", template="bank"),
                 fact("customer_deposits", "1000000000000", ticker="ACB", template="bank")]
        code = 'def solution():\n    return answer(metric("loan_to_deposit", "ACB", 2023), "percent")\n'
        report = gates(code, lambda h: h["answer"](h["metric"]("loan_to_deposit", "ACB", 2023),
                                                   "percent"), facts=facts)
        self.assertTrue(report.passed)
        self.assertEqual([(i.code, i.severity) for i in report.issues], [("sanity", "warning")])

    def test_huge_money_warning(self):
        big = Answer("quantity", M("10000000000000000"), "ty")
        report = run_gates("def solution(): pass", ExecutionResult(big, Trace()), TREE,
                           FactSet([]), context())
        self.assertTrue(report.passed)
        self.assertEqual([(i.code, i.severity) for i in report.issues], [("sanity", "warning")])
        ok = Answer("quantity", M("9999999999999999"), "ty")
        self.assertEqual(run_gates("def solution(): pass", ExecutionResult(ok, Trace()), TREE,
                                   FactSet([]), context()).issues, ())


class AgreementTests(unittest.TestCase):
    def test_agreement(self):
        executed = Answer("quantity", R("1.3108"), "times")
        self.assertTrue(compare_answers(Answer("quantity", R("1.31"), "times"), executed))
        self.assertFalse(compare_answers(Answer("quantity", R("1.35"), "times"), executed))
        self.assertTrue(compare_answers(Answer("year", 2021, "year"), Answer("year", 2021, "year")))
        self.assertFalse(compare_answers(Answer("year", 2022, "year"),
                                         Answer("year", 2021, "year")))

    def test_agreement_gate(self):
        claimed_bad = Answer("quantity", R("1.31"), "times")
        result, factset, ctx = execute(icr_program)
        executed = result.answer.value.value  # 3167889404608 / 1950254195864 = 1.6243...
        self.assertEqual(str(executed)[:6], "1.6243")
        report = run_gates(ICR_CODE, result, TREE, factset, ctx,
                           Answer("quantity", R("1.62"), "times"))
        self.assertTrue(report.passed, report.issues)
        report = run_gates(ICR_CODE, result, TREE, factset, ctx, claimed_bad)
        self.assertEqual(len(errors(report, "agreement")), 1)

    def test_agreement_rules(self):
        q = lambda v, unit: Answer("quantity", R(v), unit)
        # claim precision above 2 places is honoured
        self.assertTrue(compare_answers(q("0.123456", "percent"), q("0.1234561", "percent")))
        self.assertTrue(compare_answers(q("0", "percent"), q("0.00001", "percent")))
        self.assertFalse(compare_answers(q("0", "percent"), q("0.0001", "percent")))
        self.assertFalse(compare_answers(q("1.31", "percent"), q("1.31", "times")))
        self.assertTrue(compare_answers(q("1000", "times"), q("1001.9", "times")))
        self.assertFalse(compare_answers(q("1000", "times"), q("1002.1", "times")))
        money = Answer("quantity", M("60368920000000"), "ty")
        self.assertTrue(compare_answers(money, Answer("quantity", M("60368915000000"), "ty")))
        self.assertFalse(compare_answers(Answer("ticker", "VJC", "ticker"),
                                         Answer("ticker", "HVN", "ticker")))
        self.assertFalse(compare_answers(Answer("count", 2, "count"), Answer("count", 3, "count")))
        self.assertTrue(compare_answers(Answer("boolean", True, "boolean"),
                                        Answer("boolean", True, "boolean")))
        tiny = "0." + "0" * 120 + "1"  # precision is capped, never an InvalidOperation
        self.assertTrue(compare_answers(q(tiny, "times"), q("0", "times")))


class ReportTests(unittest.TestCase):
    def test_constants_and_report(self):
        self.assertEqual(ALLOWED_CONSTANTS, frozenset(D(x) for x in (0, 1, 2, 4, 12, 100, 365)))
        warning = GateIssue("sanity", "warning", "w")
        self.assertTrue(GateReport((warning,)).passed)
        self.assertFalse(GateReport((warning, GateIssue("formula", "error", "e", 1))).passed)

    def test_failure_is_rejected(self):
        with self.assertRaises(TypeError):
            run_gates("", ExecutionFailure("timeout", "Timeout"), TREE, FactSet([]), context())


class PackageExportTests(unittest.TestCase):
    def test_public_names_exported(self):
        import src.calculation as calculation
        for name in ("run_gates", "GateReport", "question_literals", "run", "MetricTree",
                     "FactSet", "Quantity", "validate_proposal", "build_helpers",
                     "parse_formula"):
            self.assertIn(name, calculation.__all__)
        for name in calculation.__all__:
            self.assertTrue(hasattr(calculation, name), name)

    def test_import_has_no_heavy_dependencies(self):
        heavy = ("torch", "transformers", "sentence_transformers", "langchain", "langchain_core",
                 "chromadb", "psycopg", "psycopg2", "rank_bm25", "numpy")
        probe = ("import sys, src.calculation; "
                 f"print(sorted(m for m in {heavy!r} if m in sys.modules))")
        out = subprocess.run([sys.executable, "-B", "-c", probe], cwd=ROOT, capture_output=True,
                             text=True, timeout=60, check=True)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()

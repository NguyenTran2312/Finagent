"""Golden programs: LLM-style `solution()` code over synthetic facts, end to end.

Each test builds the facts an extractor would produce (FactSource chunk ids "syn:<qid>:<n>",
printed values in the report's own scale), the ExecutionContext the query layer would build
from the question, runs the code in the spawn sandbox and requires every gate to pass with no
issues. Expected values are hand-computed in the comments. One spawn per test (~0.5 s each).
"""

import unittest
from decimal import Decimal as D

from src.calculation.facts import Fact, FactKey, FactSet, FactSource
from src.calculation.gates import (
    question_literals,
    requested_answer_kind,
    requested_unit,
    run_gates,
)
from src.calculation.helpers import Answer, ExecutionContext
from src.calculation.metric_tree import MetricTree
from src.calculation.sandbox import ExecutionResult, run
from src.calculation.units import Dimension, Quantity, display_unit, round_display, to_display
from src.preprocess.cleaner import parse_vn_decimal

TREE = MetricTree.load()
SCALES = {"VND": D(1), "triệu VND": D(10) ** 6}


class Report:
    """Synthetic extracted facts for one question."""

    def __init__(self, qid, templates):
        self.qid = qid
        self.templates = templates
        self.facts = []

    def add(self, metric_id, ticker, year, printed, unit="triệu VND", *, restated=False):
        leaf = TREE.leaf(metric_id, self.templates[ticker])
        scale = SCALES[unit]
        n = len(self.facts) + 1
        source = FactSource(chunk_id=f"syn:{self.qid}:{n}", doc_id=f"syn:{self.qid}:{ticker}",
                            page=n, row_label=leaf.label_vi, raw_text=printed, unit_text=unit,
                            scale=scale, column_header=str(year))
        value = parse_vn_decimal(printed, locale="vi") * scale
        self.facts.append(Fact(metric_id, ticker, year, "consolidated", leaf.period_kind,
                               Quantity(value, leaf.dimension), source, restated))
        return self


def claim(display_value, unit, dimension=Dimension.MONEY):
    """An answer as an LLM would state it, in the display unit."""
    value = D(display_value)
    if unit in ("percent", "percentage_points"):
        value /= 100
    elif dimension is Dimension.MONEY:
        value *= display_unit(unit).scale
    return Answer("quantity", Quantity(value, dimension), unit)


class GoldenPrograms(unittest.TestCase):
    def check(self, question, report, code, *, kind="quantity", unit, expected, named=(),
              claimed=None):
        context = ExecutionContext(
            default_scope="consolidated", ticker_templates=dict(report.templates),
            question_literals=question_literals(question),
            requested_unit=requested_unit(question),
            requested_kind=requested_answer_kind(question), named_metrics=frozenset(named))
        self.assertEqual(context.requested_kind, kind)
        self.assertEqual(context.requested_unit, unit if kind == "quantity" else None)
        facts = FactSet(report.facts)
        result = run(code, TREE, facts, context)
        self.assertIsInstance(result, ExecutionResult, result)
        answer = result.answer
        self.assertEqual((answer.kind, answer.unit), (kind, unit))
        if kind == "quantity":
            shown = round_display(to_display(answer.value, display_unit(unit)))
            self.assertEqual(shown, D(expected))
        else:
            self.assertEqual(answer.value, expected)
        gates = run_gates(code, result, TREE, facts, context, claimed)
        self.assertTrue(gates.passed, gates.issues)
        self.assertEqual(gates.issues, ())
        return result

    # 1 -- lookup
    def test_01_lookup(self):
        q = "Doanh thu thuần năm 2023 của VNM là bao nhiêu tỷ đồng?"
        report = Report(1, {"VNM": "corporate"}).add("net_revenue", "VNM", 2023, "60.368.915")
        code = '''
def solution():
    revenue = fact("net_revenue", "VNM", 2023)
    return answer(revenue, "ty")
'''
        # 60.368.915 triệu = 60 368 915 000 000 đồng = 60 368,915 tỷ -> 60368.92
        self.check(q, report, code, unit="ty", expected="60368.92", named={"net_revenue"},
                   claimed=claim("60368.92", "ty"))

    # 760 -- difference between two companies (reports in different scales)
    def test_02_difference_between_companies(self):
        q = "Lợi nhuận sau thuế năm 2022 của HPG cao hơn HSG bao nhiêu tỷ đồng?"
        report = (Report(760, {"HPG": "corporate", "HSG": "corporate"})
                  .add("profit_after_tax", "HPG", 2022, "8.444.411")
                  .add("profit_after_tax", "HSG", 2022, "251.077.445.010", "VND"))
        code = '''
def solution():
    hpg = fact("profit_after_tax", "HPG", 2022)
    hsg = fact("profit_after_tax", "HSG", 2022)
    return answer(hpg - hsg, "ty")
'''
        # 8 444 411 000 000 - 251 077 445 010 = 8 193 333 554 990 đồng = 8193.33355499 tỷ
        self.check(q, report, code, unit="ty", expected="8193.33")

    # 606 -- growth 2019 -> 2023
    def test_03_growth(self):
        q = "Doanh thu thuần của FPT năm 2023 tăng bao nhiêu % so với năm 2019?"
        report = (Report(606, {"FPT": "corporate"})
                  .add("net_revenue", "FPT", 2019, "27.716.960")
                  .add("net_revenue", "FPT", 2023, "52.617.901"))
        code = '''
def solution():
    new = fact("net_revenue", "FPT", 2023)
    old = fact("net_revenue", "FPT", 2019)
    return answer(growth_pct(new, old), "percent")
'''
        # (52 617 901 - 27 716 960) / 27 716 960 = 24 900 941 / 27 716 960 = 0.898400870... -> 89.84 %
        self.check(q, report, code, unit="percent", expected="89.84")

    # 976 -- mean of four companies' shares
    def test_04_mean_of_shares(self):
        q = ("Tỷ trọng tài sản dài hạn trên tổng tài sản trung bình của HPG, HSG, NKG và TLH "
             "năm 2023 là bao nhiêu %?")
        tickers = {"HPG": ("112.200.000", "187.000.000"), "HSG": ("8.000.000", "20.000.000"),
                   "NKG": ("3.500.000", "14.000.000"), "TLH": ("1.400.000", "4.000.000")}
        report = Report(976, dict.fromkeys(tickers, "corporate"))
        for ticker, (long_term, total) in tickers.items():
            report.add("long_term_assets", ticker, 2023, long_term)
            report.add("total_assets", ticker, 2023, total)
        code = '''
def solution():
    tickers = ["HPG", "HSG", "NKG", "TLH"]
    shares = [metric("long_term_assets_share", t, 2023) for t in tickers]
    return answer(mean(shares), "percent")
'''
        # shares 0.6, 0.4, 0.25, 0.35 -> mean 1.6 / 4 = 0.4 -> 40.00 %
        self.check(q, report, code, unit="percent", expected="40.00",
                   named={"long_term_assets_share"})

    # 363 -- argmax year by liabilities_to_equity, then interest_coverage of that year
    def test_05_argmax_year_then_icr(self):
        q = ("Hệ số khả năng thanh toán lãi vay của VJC là bao nhiêu lần trong năm có hệ số nợ "
             "phải trả trên vốn chủ sở hữu cao nhất giai đoạn 2021–2023?")
        report = Report(363, {"VJC": "corporate"})
        for year, liabilities, equity in ((2021, "35.000.000", "15.000.000"),
                                          (2022, "48.000.000", "14.000.000"),
                                          (2023, "52.000.000", "16.000.000")):
            report.add("liabilities", "VJC", year, liabilities)
            report.add("owners_equity", "VJC", year, equity)
        report.add("profit_before_tax", "VJC", 2022, "1.217.635.208.744", "VND")
        report.add("interest_expense", "VJC", 2022, "(1.950.254.195.864)", "VND")
        code = '''
def solution():
    years = [2021, 2022, 2023]
    leverage = {y: metric("liabilities_to_equity", "VJC", y) for y in years}
    worst = argmax(leverage)
    return answer(metric("interest_coverage", "VJC", worst), "times")
'''
        # D/E: 2.333, 3.429, 3.25 -> 2022.
        # ICR 2022 = (1 217 635 208 744 + |-1 950 254 195 864|) / 1 950 254 195 864
        #          = 3 167 889 404 608 / 1 950 254 195 864 = 1.62434... -> 1.62 lần
        self.check(q, report, code, unit="times", expected="1.62",
                   named={"interest_coverage", "liabilities_to_equity"})

    # 443 -- median filter on D/E, argmax of revenue growth, then gross_margin
    def test_06_median_filter_growth_margin(self):
        q = ("Trong các doanh nghiệp HPG, HSG, NKG, TLH, SMC, doanh nghiệp có hệ số nợ phải trả "
             "trên vốn chủ sở hữu năm 2023 thấp hơn trung vị và có tăng trưởng doanh thu thuần "
             "năm 2023 cao nhất thì có biên lợi nhuận gộp năm 2023 là bao nhiêu %?")
        leverage = {"HPG": ("80.000.000", "100.000.000"), "HSG": ("12.000.000", "10.000.000"),
                    "NKG": ("7.500.000", "5.000.000"), "TLH": ("1.200.000", "2.000.000"),
                    "SMC": ("5.000.000", "2.000.000")}
        report = Report(443, dict.fromkeys(leverage, "corporate"))
        for ticker, (liabilities, equity) in leverage.items():
            report.add("liabilities", ticker, 2023, liabilities)
            report.add("owners_equity", ticker, 2023, equity)
        (report.add("net_revenue", "HPG", 2022, "141.409.274")
               .add("net_revenue", "HPG", 2023, "120.355.080")
               .add("net_revenue", "TLH", 2022, "4.000.000")
               .add("net_revenue", "TLH", 2023, "4.400.000")
               .add("cost_of_goods_sold", "TLH", 2023, "(4.114.000)")
               .add("gross_profit_reported", "TLH", 2023, "286.000"))
        code = '''
def solution():
    tickers = ["HPG", "HSG", "NKG", "TLH", "SMC"]
    leverage = {t: metric("liabilities_to_equity", t, 2023) for t in tickers}
    middle = median([leverage[t] for t in tickers])
    low = filter_by(leverage, "<", middle)
    growth = {t: growth_pct(fact("net_revenue", t, 2023), fact("net_revenue", t, 2022))
              for t in low}
    best = argmax(growth)
    return answer(metric("gross_margin", best, 2023), "percent")
'''
        # D/E 0.8, 1.2, 1.5, 0.6, 2.5 -> median 1.2 -> {HPG, TLH};
        # growth HPG -14.89 %, TLH +10 % -> TLH; gross_profit 286 000 (reported) cross-checks
        # with 4 400 000 - 4 114 000; margin 286 000 / 4 400 000 = 0.065 -> 6.50 %
        result = self.check(q, report, code, unit="percent", expected="6.50",
                            named={"liabilities_to_equity", "gross_margin"})
        self.assertEqual(len(result.trace.cross_checks()), 1)

    # 435 -- scenario (revenue -10 %) and count_if of ICR < 1.5
    def test_07_scenario_count(self):
        q = ("Nếu doanh thu thuần năm 2023 giảm 10% và biên EBIT không đổi, có bao nhiêu doanh "
             "nghiệp trong nhóm VJC, HVN, ACV có hệ số khả năng thanh toán lãi vay dưới 1,5 lần?")
        report = Report(435, dict.fromkeys(("VJC", "HVN", "ACV"), "corporate"))
        for ticker, pbt, interest in (("VJC", "1.170.000", "(1.950.000)"),
                                      ("HVN", "(2.000.000)", "(4.000.000)"),
                                      ("ACV", "3.800.000", "(200.000)")):
            report.add("profit_before_tax", ticker, 2023, pbt)
            report.add("interest_expense", ticker, 2023, interest)
        code = '''
def solution():
    tickers = ["VJC", "HVN", "ACV"]
    # constant EBIT margin: EBIT, hence ICR, falls with revenue by 10%
    scenario = [metric("interest_coverage", t, 2023) * (1 - percent(10)) for t in tickers]
    return answer(count_if(scenario, "<", 1.5), "count")
'''
        # ICR: VJC 3 120 000 / 1 950 000 = 1.6 -> 1.44; HVN 2 000 000 / 4 000 000 = 0.5 -> 0.45;
        # ACV 4 000 000 / 200 000 = 20 -> 18. Below 1.5: VJC, HVN -> 2
        self.check(q, report, code, kind="count", unit="count", expected=2,
                   named={"interest_coverage"}, claimed=Answer("count", 2, "count"))

    # 393 -- percentage-point change of gross_margin
    def test_08_pct_point_change(self):
        q = "Biên lợi nhuận gộp của MWG năm 2023 thay đổi bao nhiêu điểm phần trăm so với năm 2022?"
        report = (Report(393, {"MWG": "corporate"})
                  .add("net_revenue", "MWG", 2022, "133.400.000")
                  .add("cost_of_goods_sold", "MWG", 2022, "(103.385.000)")
                  .add("net_revenue", "MWG", 2023, "118.280.000")
                  .add("cost_of_goods_sold", "MWG", 2023, "(93.441.200)"))
        code = '''
def solution():
    new = metric("gross_margin", "MWG", 2023)
    old = metric("gross_margin", "MWG", 2022)
    return answer(pct_point_change(new, old), "percentage_points")
'''
        # 2022: 30 015 000 / 133 400 000 = 0.225; 2023: 24 838 800 / 118 280 000 = 0.21
        # change = -0.015 -> -1.50 điểm phần trăm
        self.check(q, report, code, unit="percentage_points", expected="-1.50",
                   named={"gross_margin"},
                   claimed=claim("-1.5", "percentage_points", Dimension.RATIO))

    # 370 -- CAGR selection then net_margin
    def test_09_cagr_then_net_margin(self):
        q = ("Trong ba doanh nghiệp VNM, MSN, SAB, doanh nghiệp có tốc độ tăng trưởng kép (CAGR) "
             "doanh thu thuần giai đoạn 2019–2023 cao nhất có biên lợi nhuận ròng năm 2023 là "
             "bao nhiêu %?")
        report = Report(370, dict.fromkeys(("VNM", "MSN", "SAB"), "corporate"))
        for ticker, start, end in (("VNM", "56.318.123", "60.368.915"),
                                   ("MSN", "37.354.000", "78.252.000"),
                                   ("SAB", "37.899.000", "30.462.000")):
            report.add("net_revenue", ticker, 2019, start)
            report.add("net_revenue", ticker, 2023, end)
        report.add("profit_after_tax", "MSN", 2023, "3.912.600")
        code = '''
def solution():
    tickers = ["VNM", "MSN", "SAB"]
    growth = {t: cagr(fact("net_revenue", t, 2023), fact("net_revenue", t, 2019), 2023 - 2019)
              for t in tickers}
    best = argmax(growth)
    return answer(metric("net_margin", best, 2023), "percent")
'''
        # CAGR: VNM 1.75 %, MSN 20.31 %, SAB -5.31 % -> MSN; 3 912 600 / 78 252 000 = 0.05 -> 5.00 %
        self.check(q, report, code, unit="percent", expected="5.00", named={"net_margin"})

    # 1000 -- "năm nào" argmax
    def test_10_which_year(self):
        q = "Trong giai đoạn 2020–2023, năm nào HPG có lợi nhuận sau thuế cao nhất?"
        report = Report(1000, {"HPG": "corporate"})
        for year, profit in ((2020, "13.506.164"), (2021, "34.520.977"), (2022, "8.444.411"),
                             (2023, "6.835.000")):
            report.add("profit_after_tax", "HPG", year, profit)
        code = '''
def solution():
    profits = {y: fact("profit_after_tax", "HPG", y) for y in range(2020, 2024)}
    return answer(argmax(profits), "year")
'''
        # largest: 34 520 977 triệu in 2021
        self.check(q, report, code, kind="year", unit="year", expected=2021,
                   claimed=Answer("year", 2021, "year"))

    # 965 -- "có bao nhiêu" with a money(20, "ty") threshold
    def test_11_count_with_money_threshold(self):
        q = ("Có bao nhiêu công ty trong nhóm VJC, HVN, ACV, SCS có chi phí lãi vay năm 2023 lớn "
             "hơn 20 tỷ đồng?")
        report = (Report(965, dict.fromkeys(("VJC", "HVN", "ACV", "SCS"), "corporate"))
                  .add("interest_expense", "VJC", 2023, "(1.950.254.195.864)", "VND")
                  .add("interest_expense", "HVN", 2023, "(4.012.345)")
                  .add("interest_expense", "ACV", 2023, "(19.800)")
                  .add("interest_expense", "SCS", 2023, "(1.234.567.890)", "VND"))
        code = '''
def solution():
    tickers = ["VJC", "HVN", "ACV", "SCS"]
    costs = [abs(fact("interest_expense", t, 2023)) for t in tickers]
    return answer(count_if(costs, ">", money(20, "ty")), "count")
'''
        # 1950.25 tỷ, 4012.35 tỷ, 19.8 tỷ, 1.23 tỷ -> above 20 tỷ: VJC, HVN -> 2
        self.check(q, report, code, kind="count", unit="count", expected=2,
                   named={"interest_expense"})

    # bank template -- loan_to_deposit
    def test_12_bank_loan_to_deposit(self):
        q = ("Tỷ lệ cho vay khách hàng trên tiền gửi của khách hàng (LDR) của ACB tại ngày "
             "31/12/2023 là bao nhiêu %?")
        report = (Report("ldr", {"ACB": "bank"})
                  .add("customer_loans_gross", "ACB", 2023, "480.000.000")
                  .add("customer_deposits", "ACB", 2023, "500.000.000"))
        code = '''
def solution():
    return answer(metric("loan_to_deposit", "ACB", 2023), "percent")
'''
        # 480 000 000 / 500 000 000 = 0.96 -> 96.00 %
        self.check(q, report, code, unit="percent", expected="96.00", named={"loan_to_deposit"})

    # days_inventory with average balances; opening balance from two reports in two scales
    def test_13_days_inventory_average_balances(self):
        q = "Số ngày tồn kho bình quân của MWG năm 2023 là bao nhiêu ngày?"
        report = (Report("dio", {"MWG": "corporate"})
                  .add("inventories", "MWG", 2022, "36.500.000")                    # 2022 report
                  .add("inventories", "MWG", 2022, "36.500.000.400.000", "VND")     # 2023 opening
                  .add("inventories", "MWG", 2023, "22.900.000")
                  .add("cost_of_goods_sold", "MWG", 2023, "(99.000.000)"))
        code = '''
def solution():
    return answer(metric("days_inventory", "MWG", 2023), "days")
'''
        # 2022 closing: |36 500 000 000 000 - 36 500 000 400 000| <= 10^6 -> no conflict, the VND
        # value is kept (D1). average = (22 900 000 000 000 + 36 500 000 400 000) / 2
        # = 29 700 000 200 000; / 99 000 000 000 000 * 365 = 109.50000074 -> 109.50 ngày
        result = self.check(q, report, code, unit="days", expected="109.50",
                            named={"days_inventory"},
                            claimed=claim("109.5", "days", Dimension.DAYS))
        step = next(s for s in result.trace.steps if s.helper == "metric")
        self.assertIn(FactKey("inventories", "MWG", 2022, "consolidated", "closing"),
                      step.fact_keys)

    # asset_turnover: [t-1] closing balance through the tree formula
    def test_14_asset_turnover(self):
        q = "Vòng quay tổng tài sản của VNM năm 2023 là bao nhiêu vòng?"
        report = (Report("turnover", {"VNM": "corporate"})
                  .add("net_revenue", "VNM", 2023, "60.368.915")
                  .add("total_assets", "VNM", 2023, "52.673.371")
                  .add("total_assets", "VNM", 2022, "48.482.664"))
        code = '''
def solution():
    return answer(metric("asset_turnover", "VNM", 2023), "times")
'''
        # 60 368 915 / ((52 673 371 + 48 482 664) / 2) = 60 368 915 / 50 578 017.5 = 1.19358 -> 1.19
        result = self.check(q, report, code, unit="times", expected="1.19",
                            named={"asset_turnover"})
        step = next(s for s in result.trace.steps if s.helper == "metric")
        self.assertEqual(set(step.fact_keys), {
            FactKey("net_revenue", "VNM", 2023, "consolidated", "flow"),
            FactKey("total_assets", "VNM", 2023, "consolidated", "closing"),
            FactKey("total_assets", "VNM", 2022, "consolidated", "closing")})


if __name__ == "__main__":
    unittest.main()

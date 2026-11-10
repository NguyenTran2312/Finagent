import pickle
import unittest
from decimal import Decimal as D

from src.calculation.expressions import (
    Formula,
    FormulaError,
    Ref,
    evaluate,
    infer_dimension,
    parse_formula,
)
from src.calculation.units import CalculationError, Dimension as Dm, Quantity, UnitError


def M(v):
    return Quantity(D(v), Dm.MONEY)


def R(v):
    return Quantity(D(v), Dm.RATIO)


class ExpressionsTests(unittest.TestCase):
    def test_parse_refs_and_offsets(self):
        f = parse_formula("net_revenue / ((total_assets + total_assets[t-1]) / 2)")
        self.assertIsInstance(f, Formula)
        self.assertEqual(
            f.refs, {Ref("net_revenue", 0), Ref("total_assets", 0), Ref("total_assets", 1)}
        )
        self.assertEqual(parse_formula("x[t - 2]").refs, {Ref("x", 2)})

    def test_rejected_constructs(self):
        bad_cases = [
            "x ** 2", "max(a, b)", "a.b", "a < b", "a and b", "x[t+1]", "x[0]",
            "x[t-1.5]", "'s'", "lambda: 1", "__import__('os')", "a if b else c",
            "1j", "x[t-0]", "x[t-True]", "x[u-1]", "x[t-1][t-1]", "abs(a, b)",
            "abs(a=1)", "abs(*a)", "abs()", "True", "None", "a % b", "a // b",
            "+a", "~a", "a.__class__", "__x", "a__b", "(a, b)", "[a]", "",
            "a +", "1e999", "x[t-y]", "abs(a)(b)", "a\nb", "\x00",
            "f'{a}'", "a := 1", "x[1:2]",
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad):
                with self.assertRaises(FormulaError):
                    parse_formula(bad)

    def test_non_string_rejected(self):
        with self.assertRaises(FormulaError):
            parse_formula(None)  # type: ignore[arg-type]

    def test_abs_is_only_call(self):
        self.assertEqual(parse_formula("abs(interest_expense)").refs, {Ref("interest_expense", 0)})

    def test_infer_dimension_and_mismatch(self):
        dims = {"a": Dm.MONEY, "b": Dm.MONEY, "r": Dm.RATIO, "s": Dm.SHARES}.__getitem__
        self.assertIs(infer_dimension(parse_formula("a / b"), dims), Dm.RATIO)
        self.assertIs(infer_dimension(parse_formula("a * 2"), dims), Dm.MONEY)
        self.assertIs(infer_dimension(parse_formula("a / s"), dims), Dm.MONEY_PER_SHARE)
        self.assertIs(infer_dimension(parse_formula("abs(-a)"), dims), Dm.MONEY)
        with self.assertRaises(UnitError):
            infer_dimension(parse_formula("a + r"), dims)
        with self.assertRaises(UnitError):
            infer_dimension(parse_formula("a * b"), dims)
        with self.assertRaises(UnitError):
            infer_dimension(parse_formula("a + 1"), dims)

    def test_infer_never_divides_by_zero(self):
        dims = {"a": Dm.MONEY, "b": Dm.MONEY}.__getitem__
        self.assertIs(infer_dimension(parse_formula("a / (a - b)"), dims), Dm.RATIO)
        self.assertIs(infer_dimension(parse_formula("a / (1 - 1)"), dims), Dm.MONEY)

    def test_days_form(self):
        f = parse_formula("((inv + inv[t-1]) / 2) / abs(cogs) * 365")
        self.assertTrue(f.is_days_form)
        dims = {"inv": Dm.MONEY, "cogs": Dm.MONEY}.__getitem__
        self.assertIs(infer_dimension(f, dims), Dm.DAYS)
        self.assertFalse(parse_formula("a * 365 + 1").is_days_form)
        self.assertFalse(parse_formula("365 * a").is_days_form)

    def test_days_form_requires_ratio(self):
        f = parse_formula("a * 365")
        with self.assertRaises(UnitError):
            infer_dimension(f, {"a": Dm.MONEY}.__getitem__)
        with self.assertRaises(UnitError):
            evaluate(f, {Ref("a", 0): M("1")}.__getitem__)

    def test_evaluate_days(self):
        f = parse_formula("(a / b) * 365")
        vals = {Ref("a", 0): M("10"), Ref("b", 0): M("20")}
        self.assertEqual(evaluate(f, vals.__getitem__), Quantity(D("182.5"), Dm.DAYS))

    def test_evaluate_exact(self):
        vals = {Ref("a", 0): M("10"), Ref("b", 0): M("4")}
        self.assertEqual(evaluate(parse_formula("(a - b) / b"), vals.__getitem__), R("1.5"))
        self.assertEqual(evaluate(parse_formula("0.1 + 0.2"), vals.__getitem__), R("0.3"))
        self.assertEqual(evaluate(parse_formula("abs(-a) - b"), vals.__getitem__), M("6"))
        self.assertEqual(evaluate(parse_formula("-a"), vals.__getitem__), M("-10"))

    def test_evaluate_offsets_use_resolver(self):
        vals = {Ref("a", 0): M("10"), Ref("a", 1): M("6")}
        self.assertEqual(evaluate(parse_formula("(a + a[t-1]) / 2"), vals.__getitem__), M("8"))

    def test_evaluate_division_by_zero(self):
        vals = {Ref("a", 0): M("10"), Ref("b", 0): M("0")}
        with self.assertRaises(CalculationError):
            evaluate(parse_formula("a / b"), vals.__getitem__)

    def test_evaluate_dimension_mismatch(self):
        vals = {Ref("a", 0): M("10"), Ref("r", 0): R("2")}
        with self.assertRaises(UnitError):
            evaluate(parse_formula("a + r"), vals.__getitem__)

    def test_pickle_roundtrip(self):
        f = parse_formula("a / b * 365")
        g = pickle.loads(pickle.dumps(f))
        self.assertEqual(g.text, f.text)
        self.assertEqual(g.refs, f.refs)
        self.assertEqual(g.is_days_form, f.is_days_form)

    def test_ref_validation(self):
        with self.assertRaises(ValueError):
            Ref("a", -1)


if __name__ == "__main__":
    unittest.main()

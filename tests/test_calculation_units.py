import unittest
from decimal import Decimal as D

from src.calculation.units import (
    CalculationError,
    Dimension as Dm,
    Quantity,
    UnitError,
    display_unit,
    round_display,
    to_display,
)


def M(v):
    return Quantity(D(v), Dm.MONEY)


def R(v):
    return Quantity(D(v), Dm.RATIO)


class UnitsTests(unittest.TestCase):
    def test_money_algebra(self):
        self.assertEqual(M("5") + M("3"), M("8"))
        self.assertEqual(M("6") / M("3"), R("2"))
        self.assertIs((M("10") / Quantity(D("2"), Dm.SHARES)).dimension, Dm.MONEY_PER_SHARE)
        self.assertEqual(M("10") * R("0.5"), M("5.0"))
        self.assertEqual(M("10") * 2, M("20"))
        self.assertEqual(2 * M("10"), M("20"))
        self.assertEqual(-M("3"), M("-3"))
        self.assertEqual(abs(M("-3")), M("3"))
        self.assertEqual(M("10") / 4, M("2.5"))

    def test_invalid_algebra_raises(self):
        with self.assertRaises(UnitError):
            M("1") + R("1")
        with self.assertRaises(UnitError):
            M("1") * M("1")
        with self.assertRaises(UnitError):
            R("1") - Quantity(D(1), Dm.DAYS)
        with self.assertRaises(UnitError):
            R("1") / M("1")

    def test_division_by_zero(self):
        with self.assertRaises(CalculationError):
            M("1") / M("0")
        with self.assertRaises(CalculationError):
            M("1") / 0

    def test_rejects_float_and_nonfinite(self):
        with self.assertRaises(TypeError):
            Quantity(0.1, Dm.MONEY)
        with self.assertRaises(TypeError):
            M("1") + 0.1
        with self.assertRaises(UnitError):
            Quantity(D("NaN"), Dm.MONEY)
        with self.assertRaises(UnitError):
            Quantity(D("Infinity"), Dm.RATIO)

    def test_compare_with_number_only_ratio_count(self):
        self.assertTrue(R("1.6") > D("1.5"))
        self.assertTrue(Quantity(D(3), Dm.COUNT) >= 3)
        with self.assertRaises(UnitError):
            M("5") > 3
        with self.assertRaises(UnitError):
            M("5") < R("3")
        self.assertTrue(M("5") > M("3"))

    def test_exact_zero_is_additive_identity_for_any_dimension(self):
        days = Quantity(D(30), Dm.DAYS)
        self.assertEqual(0 + M("5"), M("5"))
        self.assertEqual(M("5") + 0, M("5"))
        self.assertEqual(D(0) + days, days)
        self.assertEqual(M("5") - D("0.00"), M("5"))
        self.assertEqual(0 - M("5"), M("-5"))
        self.assertEqual(sum([M("1"), M("2"), M("3")]), M("6"))
        self.assertEqual(sum([R("0.1"), R("0.2")]), R("0.3"))
        with self.assertRaises(UnitError):
            M("5") + 1
        with self.assertRaises(UnitError):
            1 + M("5")
        with self.assertRaises(TypeError):
            M("5") + False
        with self.assertRaises(TypeError):
            M("5") + 0.0

    def test_exact_zero_is_comparable_for_any_dimension(self):
        self.assertTrue(M("5") > 0)
        self.assertTrue(M("-5") < D(0))
        self.assertTrue(M("0") >= 0)
        self.assertTrue(0 < M("5"))
        self.assertFalse(Quantity(D(-1), Dm.DAYS) >= 0)
        with self.assertRaises(UnitError):
            M("5") > 1
        with self.assertRaises(TypeError):
            M("5") > False
        with self.assertRaises(TypeError):
            M("5") > 0.0

    def test_display_money_scales(self):
        self.assertEqual(to_display(M("1234567000000"), display_unit("ty")), D("1234.567"))
        self.assertEqual(to_display(M("2500000000000"), display_unit("nghin_ty")), D("2.5"))

    def test_display_percent_and_points(self):
        self.assertEqual(to_display(R("0.1234"), display_unit("percent")), D("12.34"))
        self.assertEqual(to_display(R("-0.015"), display_unit("percentage_points")), D("-1.5"))
        with self.assertRaises(UnitError):
            to_display(M("1"), display_unit("percent"))

    def test_display_money_accepts_money_per_share(self):
        q = Quantity(D("1500"), Dm.MONEY_PER_SHARE)
        self.assertEqual(to_display(q, display_unit("nghin")), D("1.5"))

    def test_display_passthrough_and_rejections(self):
        self.assertEqual(to_display(R("2.5"), display_unit("times")), D("2.5"))
        self.assertEqual(to_display(Quantity(D(30), Dm.DAYS), display_unit("days")), D(30))
        self.assertEqual(to_display(Quantity(D(7), Dm.COUNT), display_unit("count")), D(7))
        self.assertEqual(to_display(Quantity(D(7), Dm.SHARES), display_unit("shares")), D(7))
        with self.assertRaises(UnitError):
            to_display(M("1"), display_unit("days"))
        with self.assertRaises(UnitError):
            display_unit("year")
        with self.assertRaises(UnitError):
            display_unit("bogus")

    def test_round_half_up(self):
        self.assertEqual(round_display(D("2.345")), D("2.35"))
        self.assertEqual(round_display(D("-2.345")), D("-2.35"))
        self.assertEqual(round_display(D("1234567890123456789012345678.125"), 2),
                         D("1234567890123456789012345678.13"))
        self.assertEqual(round_display(D("2.5"), 0), D("3"))

    def test_decimal_exact(self):
        self.assertEqual(M("0.1") + M("0.2"), M("0.3"))


if __name__ == "__main__":
    unittest.main()

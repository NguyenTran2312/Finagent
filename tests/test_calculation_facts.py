import unittest
from decimal import Decimal as D

from src.calculation.facts import (
    Fact,
    FactConflict,
    FactKey,
    FactSet,
    FactSource,
    MissingFact,
)
from src.calculation.units import Dimension, Quantity


def M(text: str) -> Quantity:
    return Quantity(D(text), Dimension.MONEY)


def source(scale: str = "1", **overrides) -> FactSource:
    kwargs = dict(
        chunk_id="c1",
        doc_id="d1",
        page=3,
        row_label="Tong tai san",
        raw_text="1.000",
        unit_text="VND",
        scale=D(scale) if isinstance(scale, str) else scale,
    )
    kwargs.update(overrides)
    return FactSource(**kwargs)


def fact(
    value="1000",
    scale="1",
    metric_id="total_assets",
    ticker="VJC",
    year=2023,
    scope="consolidated",
    period_kind="closing",
    restated=False,
    **fact_overrides,
) -> Fact:
    kwargs = dict(
        metric_id=metric_id,
        ticker=ticker,
        fiscal_year=year,
        scope=scope,
        period_kind=period_kind,
        value=M(value) if isinstance(value, str) else value,
        source=source(scale),
        restated=restated,
    )
    kwargs.update(fact_overrides)
    return Fact(**kwargs)


class FactSourceTests(unittest.TestCase):
    def test_source_required(self):
        with self.assertRaises(TypeError):
            fact(source=None)
        with self.assertRaises(ValueError):
            source(chunk_id="")

    def test_blank_strings_rejected(self):
        for name in ("chunk_id", "doc_id", "row_label", "raw_text", "unit_text"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    source(**{name: "   "})

    def test_column_header_may_be_empty_or_none(self):
        self.assertIsNone(source().column_header)
        self.assertEqual(source(column_header="").column_header, "")

    def test_page_validation(self):
        self.assertEqual(source(page=0).page, 0)
        for bad in (-1, True, 1.0, "1"):
            with self.subTest(page=bad):
                with self.assertRaises(ValueError):
                    source(page=bad)

    def test_scale_validation(self):
        for bad in (D("0"), D("-1"), D("NaN"), D("Infinity"), 1, 1.0):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    source(scale=bad)

    def test_non_string_field_rejected(self):
        with self.assertRaises(ValueError):
            source(chunk_id=None)

    def test_fact_field_validation(self):
        with self.assertRaises(TypeError):
            fact(value=D("1"))
        with self.assertRaises(ValueError):
            fact(scope="group")
        with self.assertRaises(ValueError):
            fact(period_kind="opening")
        for bad in (True, "2023", 2023.0):
            with self.subTest(year=bad):
                with self.assertRaises(ValueError):
                    fact(year=bad)


class FactSetTests(unittest.TestCase):
    def test_get_and_missing(self):
        a = fact()
        s = FactSet([a])
        self.assertEqual(len(s), 1)
        self.assertEqual(list(s), [a])
        self.assertIs(s.get(a.key), a)
        with self.assertRaises(MissingFact):
            s.get(FactKey("total_assets", "VJC", 1999, "consolidated", "closing"))

    def test_find(self):
        a = fact()
        s = FactSet([a])
        self.assertIs(s.find("total_assets", "VJC", 2023, "consolidated", "closing"), a)
        self.assertIsNone(s.find("total_assets", "VJC", 2022, "consolidated", "closing"))

    def test_conflict_rejected(self):
        with self.assertRaises(FactConflict):
            FactSet([fact("1000"), fact("5000")])

    def test_conflict_is_value_error(self):
        self.assertTrue(issubclass(FactConflict, ValueError))
        self.assertTrue(issubclass(MissingFact, LookupError))

    def test_dimension_mismatch_is_conflict(self):
        ratio = Quantity(D("1000"), Dimension.RATIO)
        with self.assertRaises(FactConflict):
            FactSet([fact("1000"), fact(ratio)])

    def test_same_key_within_scale_tolerance_is_not_conflict(self):
        a = fact(value=M("1234567000000"), scale="1000000")  # trieu report
        b = fact(value=M("1234567123456"), scale="1")  # VND report
        s = FactSet([a, b])
        self.assertEqual(len(s), 1)
        self.assertEqual(s.get(a.key).value, b.value)  # finer scale kept (D1)
        self.assertEqual(s.warnings, ())

    def test_finer_scale_kept_regardless_of_order(self):
        a = fact(value=M("1234567000000"), scale="1000000")
        b = fact(value=M("1234567123456"), scale="1")
        self.assertEqual(FactSet([b, a]).get(a.key).value, b.value)

    def test_tolerance_boundary(self):
        a = fact(value=M("1000000"), scale="1000")
        within = fact(value=M("1001000"), scale="1")
        beyond = fact(value=M("1001001"), scale="1")
        self.assertEqual(len(FactSet([a, within])), 1)
        with self.assertRaises(FactConflict):
            FactSet([a, beyond])

    def test_equal_scales_within_tolerance_keeps_first_without_warning(self):
        a = fact(value=M("1000"), scale="1000")
        b = fact(value=M("1500"), scale="1000")
        s = FactSet([a, b])
        self.assertIs(s.get(a.key), a)
        self.assertEqual(s.warnings, ())

    def test_restated_wins_with_warning(self):
        old = fact("1000")
        new = fact("9000", restated=True)
        for order in ([old, new], [new, old]):
            with self.subTest(order=[f.restated for f in order]):
                s = FactSet(order)
                self.assertEqual(s.get(old.key).value, new.value)
                self.assertTrue(s.warnings)
                self.assertIn("total_assets", s.warnings[0])

    def test_restated_within_tolerance_still_wins(self):
        old = fact("1000", scale="1")
        new = fact("1000", scale="1000", restated=True)
        s = FactSet([old, new])
        self.assertIs(s.get(old.key), new)
        self.assertEqual(len(s.warnings), 1)

    def test_two_restated_conflict(self):
        with self.assertRaises(FactConflict):
            FactSet([fact("1000", restated=True), fact("9000", restated=True)])

    def test_two_restated_within_tolerance_ok(self):
        s = FactSet([fact("1000", restated=True), fact("1000", restated=True)])
        self.assertEqual(len(s), 1)

    def test_three_facts_pairwise_in_input_order(self):
        a = fact("1000", scale="1000")
        b = fact("1200", scale="1")  # within tolerance of a, finer -> kept
        c = fact("5000", scale="1")  # beyond tolerance of b -> conflict
        with self.assertRaises(FactConflict):
            FactSet([a, b, c])
        d = fact("1100", scale="1")  # vs b: diff 100 <= 1? no -> conflict
        with self.assertRaises(FactConflict):
            FactSet([a, b, d])

    def test_opening_is_prior_closing(self):
        prior = fact(year=2023)
        s = FactSet([prior, fact("2000", year=2024)])
        self.assertIs(s.opening("total_assets", "VJC", 2024, "consolidated"), prior)

    def test_opening_missing(self):
        s = FactSet([fact(year=2024)])
        with self.assertRaises(MissingFact):
            s.opening("total_assets", "VJC", 2024, "consolidated")

    def test_opening_ignores_flow_and_other_scope(self):
        s = FactSet([
            fact(year=2023, period_kind="flow"),
            fact(year=2023, scope="separate"),
        ])
        with self.assertRaises(MissingFact):
            s.opening("total_assets", "VJC", 2024, "consolidated")

    def test_sign_preserved(self):
        f = fact(metric_id="interest_expense", value=M("-1950254195864"), period_kind="flow")
        self.assertEqual(FactSet([f]).get(f.key).value, M("-1950254195864"))

    def test_warnings_is_tuple(self):
        self.assertIsInstance(FactSet([]).warnings, tuple)


if __name__ == "__main__":
    unittest.main()

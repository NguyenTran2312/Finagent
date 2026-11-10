import tempfile
import unittest
from pathlib import Path

import yaml

from src.calculation.metric_tree import (
    DEFAULT_METRICS_DIR,
    DerivedDef,
    LeafRequest,
    MetricTree,
    TreeError,
)
from src.calculation.expressions import parse_formula
from src.calculation.units import Dimension


def leaf(id, *, period="flow", dim="money", aliases=None, codes=("1",), **extra):
    entry = {"id": id, "label_vi": id, "aliases": aliases or [f"alias {id}"],
             "statement_kind": "income_statement", "codes": list(codes),
             "period_kind": period, "source": "statement", "dimension": dim}
    entry.update(extra)
    return entry


def derived(id, formulas, *, dim="ratio", templates=("corporate",), aliases=None, **extra):
    entry = {"id": id, "label_vi": id, "aliases": aliases or [], "templates": list(templates),
             "dimension": dim, "formulas": list(formulas)}
    entry.update(extra)
    return entry


def base_leaves():
    return {"corporate": [leaf("a"), leaf("b"), leaf("c", period="closing")], "bank": []}


def tree(derived_entries=(), leaves=None):
    return MetricTree.from_entries(leaves or base_leaves(), list(derived_entries))


class MetricTreeTests(unittest.TestCase):
    def test_valid_tree_and_expansion(self):
        t = tree([derived("r", ["a / ((c + c[t-1]) / 2)"])])
        self.assertEqual(t.required_leaves("r", "corporate", 2024), [frozenset({
            LeafRequest("a", 2024, "flow"), LeafRequest("c", 2024, "closing"),
            LeafRequest("c", 2023, "closing")})])
        self.assertEqual(t.kind("r"), "derived")
        self.assertEqual(t.kind("a"), "leaf")
        self.assertEqual(t.dimension_of("r", "corporate"), Dimension.RATIO)
        self.assertTrue(t.applies("r", "corporate"))
        self.assertFalse(t.applies("r", "bank"))
        self.assertEqual(t.required_leaves("a", "corporate", 2024),
                         [frozenset({LeafRequest("a", 2024, "flow")})])

    def test_derived_of_derived_expands_to_leaves(self):
        t = tree([derived("m", ["a - b"], dim="money"),
                  derived("r", ["m[t-1] / c"])])
        self.assertEqual(t.required_leaves("r", "corporate", 2024), [frozenset({
            LeafRequest("a", 2023, "flow"), LeafRequest("b", 2023, "flow"),
            LeafRequest("c", 2024, "closing")})])

    def test_alternatives_in_order(self):
        t = tree([derived("g_rep", ["a"], dim="money"),
                  derived("g", ["g_rep", "a - b"], dim="money")])
        self.assertEqual(t.required_leaves("g", "corporate", 2024), [
            frozenset({LeafRequest("a", 2024, "flow")}),
            frozenset({LeafRequest("a", 2024, "flow"), LeafRequest("b", 2024, "flow")})])

    def test_nested_alternatives_are_all_requested(self):
        t = tree([derived("m", ["a", "a - b"], dim="money"),
                  derived("r", ["m / c"])])
        self.assertEqual(t.required_leaves("r", "corporate", 2024), [frozenset({
            LeafRequest("a", 2024, "flow"), LeafRequest("b", 2024, "flow"),
            LeafRequest("c", 2024, "closing")})])

    def test_tree_errors(self):
        dup_alias = base_leaves()
        dup_alias["corporate"] = [leaf("a", aliases=["x"]), leaf("b", aliases=["X"])]
        bank_leaf = base_leaves()
        bank_leaf["bank"] = [leaf("k", codes=(), aliases=["only one"])]
        bad_parent = base_leaves()
        bad_parent["corporate"].append(leaf("p", parent="zzz"))
        bank_deriv = base_leaves()
        cases = [
            ("unknown reference", dict(derived_entries=[derived("r", ["zz / a"])])),
            ("cycle", dict(derived_entries=[derived("r", ["r + a"], dim="money")])),
            ("cycle", dict(derived_entries=[derived("p", ["q"], dim="money"),
                                            derived("q", ["p"], dim="money")])),
            ("formula syntax", dict(derived_entries=[derived("r", ["a +"])])),
            ("dimension mismatch", dict(derived_entries=[derived("r", ["a"])])),
            ("dimension mismatch", dict(derived_entries=[derived("r", ["a / b", "a"])])),
            ("dimension mismatch", dict(derived_entries=[derived("r", ["a / b * 365"])])),
            ("duplicate id", dict(derived_entries=[derived("a", ["b"], dim="money")])),
            ("duplicate id", dict(derived_entries=[derived("r", ["a"], dim="money"),
                                                   derived("r", ["b"], dim="money")])),
            ("duplicate alias", dict(derived_entries=[], leaves=dup_alias)),
            ("duplicate alias", dict(derived_entries=[derived("r", ["a / b"], aliases=["ALIAS a"])])),
            ("needs codes or two aliases", dict(derived_entries=[], leaves=bank_leaf)),
            ("days form", dict(derived_entries=[derived("d", ["a / b"], dim="days")])),
            ("unknown reference", dict(derived_entries=[
                derived("r", ["a / b"], templates=("corporate", "bank"))])),
            ("unknown parent", dict(derived_entries=[], leaves=bad_parent)),
            ("reserved id", dict(derived_entries=[derived("t", ["a"], dim="money")])),
            ("reserved id", dict(derived_entries=[derived("abs", ["a"], dim="money")])),
            ("offset too large", dict(derived_entries=[derived("r", ["a[t-11] / b"])])),
            ("template mismatch", dict(derived_entries=[],
                                       leaves={"corporate": [leaf("a", template="bank")]})),
            ("unknown key", dict(derived_entries=[derived("r", ["a / b"], bogus=1)])),
            ("missing key", dict(derived_entries=[{"id": "r"}])),
        ]
        for reason, kwargs in cases:
            with self.subTest(reason=reason, kwargs=str(kwargs)[:80]):
                with self.assertRaises(TreeError) as ctx:
                    tree(**kwargs)
                self.assertIn(reason, ctx.exception.reason)
                self.assertTrue(str(ctx.exception).startswith("<memory>: "))

    def test_days_form_accepted_and_non_days_rejected(self):
        t = tree([derived("d", ["a / b * 365"], dim="days")])
        self.assertEqual(t.dimension_of("d", "corporate"), Dimension.DAYS)

    def test_error_names_file_and_node(self):
        with self.assertRaises(TreeError) as ctx:
            tree([derived("r", ["zz / a"])])
        err = ctx.exception
        self.assertTrue(str(err).startswith("<memory>: r: "))
        self.assertEqual((err.file, err.node_id), ("<memory>", "r"))

    def test_bank_leaf_with_two_aliases_and_parent(self):
        leaves = {"corporate": [], "bank": [
            leaf("sec", codes=(), aliases=["cho vay", "cho vay khach hang"]),
            leaf("child", codes=(), aliases=["x", "y"], parent="sec")]}
        t = MetricTree.from_entries(leaves, [])
        self.assertEqual(t.leaf("child", "bank").parent, "sec")
        with self.assertRaises(KeyError):
            t.leaf("child", "corporate")

    def test_find_by_alias_longest_first(self):
        leaves = base_leaves()
        leaves["corporate"] = [
            leaf("a", aliases=["lợi nhuận sau thuế"]),
            leaf("b", aliases=["lợi nhuận sau thuế thuộc cổ đông công ty mẹ"])]
        t = tree(leaves=leaves)
        text = ("LNST thuộc cổ đông công ty mẹ ... lợi nhuận sau thuế thuộc cổ đông công ty mẹ")
        self.assertEqual(t.find_by_alias(text, "corporate"), {"b"})
        self.assertEqual(t.find_by_alias("Lợi nhuận sau thuế", "corporate"), {"a"})
        self.assertEqual(t.find_by_alias("lợi nhuận sau thuế", "bank"), set())

    def test_with_derived_revalidates(self):
        t = tree([derived("m", ["a - b"], dim="money")])
        cyclic = DerivedDef("x", "x", (), ("corporate",), Dimension.MONEY, None,
                            (parse_formula("x + a"),), None)
        with self.assertRaises(TreeError) as ctx:
            t.with_derived(cyclic)
        self.assertIn("cycle", ctx.exception.reason)
        self.assertFalse(t.applies("x", "corporate"))
        good = DerivedDef("y", "y", (), ("corporate",), Dimension.MONEY, None,
                          (parse_formula("m + a"),), None)
        t2 = t.with_derived(good)
        self.assertTrue(t2.applies("y", "corporate"))
        self.assertFalse(t.applies("y", "corporate"))

    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "corporate_leaves.yaml").write_text(
                yaml.safe_dump(base_leaves()["corporate"], allow_unicode=True), encoding="utf-8")
            (d / "bank_leaves.yaml").write_text("[]", encoding="utf-8")
            (d / "derived.yaml").write_text(yaml.safe_dump([
                derived("r", ["a / b"], expected_range=[0, 1.5])]), encoding="utf-8")
            t = MetricTree.load(d)
            low, high = t.derived("r").expected_range
            self.assertEqual((str(low), str(high)), ("0", "1.5"))

            (d / "derived.yaml").write_text(yaml.safe_dump([derived("r", ["zz"])]), encoding="utf-8")
            with self.assertRaises(TreeError) as ctx:
                MetricTree.load(d)
            self.assertEqual(ctx.exception.file, "derived.yaml")

            (d / "bank_leaves.yaml").write_text(
                yaml.safe_dump([leaf("k", template="corporate", codes=("1",))]), encoding="utf-8")
            (d / "derived.yaml").write_text("[]", encoding="utf-8")
            with self.assertRaises(TreeError) as ctx:
                MetricTree.load(d)
            self.assertEqual(ctx.exception.file, "bank_leaves.yaml")
            self.assertIn("template mismatch", ctx.exception.reason)

            (d / "bank_leaves.yaml").unlink()
            with self.assertRaises(TreeError) as ctx:
                MetricTree.load(d)
            self.assertIn("missing file", ctx.exception.reason)

    def test_default_metrics_dir(self):
        self.assertEqual(DEFAULT_METRICS_DIR.parts[-2:], ("config", "metrics"))
        self.assertTrue((DEFAULT_METRICS_DIR.parent / "settings.py").is_file())


if __name__ == "__main__":
    unittest.main()

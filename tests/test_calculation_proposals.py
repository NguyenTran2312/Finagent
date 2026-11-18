import copy
import unittest

from src.calculation.metric_tree import MetricTree
from src.calculation.proposals import ProposalResult, validate_proposal


def good(**over):
    p = {"id": "cash_to_assets", "label_vi": "Tiền trên tổng tài sản", "aliases": [],
         "templates": ["corporate"], "dimension": "ratio",
         "formulas": ["cash_and_equivalents / total_assets"]}
    p.update(over)
    return p


class ProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = MetricTree.load()

    def test_valid_proposal_accepted(self):
        r = validate_proposal(self.tree, good(), "corporate")
        self.assertIsInstance(r, ProposalResult)
        self.assertTrue(r.accepted, r.issues)
        self.assertEqual(r.issues, ())
        self.assertTrue(r.tree.applies("cash_to_assets", "corporate"))

    def test_redefinition_rejected(self):
        r = validate_proposal(self.tree, good(id="gross_margin"), "corporate")
        self.assertFalse(r.accepted)
        self.assertIsNone(r.tree)
        self.assertTrue(any("redefines existing node" in i for i in r.issues))

    def test_redefinition_of_leaf_in_other_template_rejected(self):
        r = validate_proposal(self.tree, good(id="total_assets"), "corporate")
        self.assertFalse(r.accepted)
        self.assertTrue(any("redefines existing node" in i for i in r.issues))

    def test_proposal_wrong_template_leaf_rejected(self):
        r = validate_proposal(self.tree, good(templates=["bank"]), "bank")
        self.assertFalse(r.accepted)
        self.assertTrue(any("unknown reference" in i for i in r.issues))

    def test_template_not_listed(self):
        r = validate_proposal(self.tree, good(), "bank")
        self.assertFalse(r.accepted)
        self.assertTrue(any("template not listed" in i for i in r.issues))

    def test_bad_formula_rejected_not_raised(self):
        r = validate_proposal(self.tree, good(formulas=["total_assets ** 2"]), "corporate")
        self.assertFalse(r.accepted)
        self.assertTrue(any("formula syntax" in i for i in r.issues))

    def test_dimension_mismatch_rejected(self):
        r = validate_proposal(self.tree, good(dimension="money"), "corporate")
        self.assertFalse(r.accepted)
        self.assertTrue(any("dimension mismatch" in i for i in r.issues))

    def test_cycle_rejected(self):
        r = validate_proposal(self.tree, good(formulas=["cash_to_assets"]), "corporate")
        self.assertFalse(r.accepted)
        self.assertTrue(any("cycle" in i for i in r.issues))

    def test_alias_collision_rejected(self):
        existing = self.tree.derived("gross_margin").aliases[0]
        r = validate_proposal(self.tree, good(aliases=[existing.upper()]), "corporate")
        self.assertFalse(r.accepted)
        self.assertTrue(any("alias collision" in i for i in r.issues))

    def test_alias_containing_existing_alias_accepted(self):
        # A longer alias that merely CONTAINS an existing alias is not a collision.
        cases = (
            good(id="inventory_turnover", label_vi="Vòng quay hàng tồn kho",
                 aliases=["vòng quay hàng tồn kho"], display="times",
                 formulas=["abs(cost_of_goods_sold) / ((inventories + inventories[t-1]) / 2)"]),
            good(id="pretax_margin", label_vi="Biên lợi nhuận trước thuế",
                 aliases=["biên lợi nhuận trước thuế"], display="percent",
                 formulas=["profit_before_tax / net_revenue"]),
        )
        for p in cases:
            r = validate_proposal(self.tree, p, "corporate")
            self.assertTrue(r.accepted, r.issues)
            self.assertEqual(r.tree.find_by_alias(p["aliases"][0], "corporate"), {p["id"]})

    def test_alias_accent_variant_collision_rejected(self):
        r = validate_proposal(self.tree, good(aliases=["Hang Ton Kho"]), "corporate")
        self.assertFalse(r.accepted)
        self.assertTrue(any("alias collision" in i and "inventories" in i for i in r.issues), r.issues)

    def test_alias_in_unlisted_template_not_a_collision(self):
        existing = self.tree.derived("gross_margin").aliases[0]
        self.assertEqual(self.tree.find_by_alias(existing, "bank"), set())
        r = validate_proposal(self.tree, good(templates=["bank"], aliases=[existing],
                                              formulas=["total_assets / total_assets"]), "bank")
        self.assertFalse(any("alias collision" in i for i in r.issues))

    def test_malformed_shapes_do_not_raise(self):
        for bad in (None, "x", [1], {}, good(formulas="a / b"), good(templates="corporate"),
                    {k: v for k, v in good().items() if k != "formulas"}, good(extra=1)):
            r = validate_proposal(self.tree, bad, "corporate")
            self.assertFalse(r.accepted, bad)
            self.assertIsNone(r.tree)
            self.assertTrue(r.issues)

    def test_non_tree_raises_type_error(self):
        with self.assertRaises(TypeError):
            validate_proposal(object(), good(), "corporate")

    def test_original_tree_unchanged(self):
        validate_proposal(self.tree, good(), "corporate")
        self.assertFalse(self.tree.applies("cash_to_assets", "corporate"))

    def test_proposal_record_is_immutable_copy(self):
        p = good()
        r = validate_proposal(self.tree, p, "corporate")
        snapshot = copy.deepcopy(r.proposal["formulas"])
        p["formulas"].append("x / y")
        p["id"] = "changed"
        self.assertEqual(r.proposal["id"], "cash_to_assets")
        self.assertEqual(list(r.proposal["formulas"]), list(snapshot))
        with self.assertRaises(TypeError):
            r.proposal["id"] = "z"


if __name__ == "__main__":
    unittest.main()

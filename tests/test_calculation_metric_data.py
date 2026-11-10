"""Data lint for the shipped metric tree (config/metrics/*.yaml)."""

import unittest

from src.calculation.metric_tree import MetricTree
from src.calculation.units import Dimension
from src.query_paraphrase.metric_catalog import METRIC_CATALOG

CORPORATE_LEAVES = (
    "gross_revenue", "net_revenue", "cost_of_goods_sold", "gross_profit_reported",
    "financial_income", "financial_expenses", "interest_expense", "selling_expenses",
    "admin_expenses", "operating_profit", "other_income", "other_expenses",
    "profit_before_tax", "profit_after_tax", "profit_after_tax_parent_share", "basic_eps",
    "current_assets", "cash_and_equivalents", "inventories", "long_term_assets",
    "construction_in_progress", "total_assets", "liabilities", "current_liabilities",
    "short_term_borrowings", "long_term_borrowings", "owners_equity", "cfo", "depreciation",
)
BANK_LEAVES = (
    "interest_and_similar_income", "interest_and_similar_expenses", "net_interest_income",
    "net_fee_income", "operating_expenses", "credit_risk_provision_expense",
    "profit_before_tax", "profit_after_tax", "total_assets", "customer_loans_section",
    "customer_loans_gross", "customer_loan_provision", "deposits_at_other_credit_institutions",
    "customer_deposits", "liabilities", "owners_equity",
)
DERIVED = {
    "gross_profit": ("corporate",), "gross_margin": ("corporate",), "net_margin": ("corporate",),
    "ebit": ("corporate",), "interest_coverage": ("corporate",),
    "liabilities_to_equity": ("corporate", "bank"), "borrowings_to_equity": ("corporate",),
    "current_ratio": ("corporate",), "quick_ratio": ("corporate",),
    "cfo_to_current_liabilities": ("corporate",), "cfo_to_profit_after_tax": ("corporate",),
    "cfo_margin": ("corporate",), "asset_turnover": ("corporate",),
    "days_inventory": ("corporate",), "selling_and_admin_expenses": ("corporate",),
    "long_term_assets_share": ("corporate",), "roe": ("corporate", "bank"),
    "roa": ("corporate", "bank"), "loan_to_deposit": ("bank",),
    "loan_provision_ratio": ("bank",),
}
CATALOG_STATEMENT_IDS = ("net_revenue", "profit_before_tax", "profit_after_tax",
                         "profit_after_tax_parent_share")


class MetricDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = MetricTree.load()

    def corporate_leaves(self):
        return [self.tree.leaf(i, "corporate") for i in CORPORATE_LEAVES]

    def bank_leaves(self):
        return [self.tree.leaf(i, "bank") for i in BANK_LEAVES]

    def test_real_tree_loads(self):
        self.assertIsInstance(MetricTree.load(), MetricTree)

    def test_required_nodes_present(self):
        for node_id in ("interest_coverage", "quick_ratio", "days_inventory", "asset_turnover",
                        "roe", "loan_to_deposit"):
            self.assertEqual(self.tree.derived(node_id).id, node_id)
        self.assertTrue(self.tree.applies("gross_margin", "corporate"))
        self.assertFalse(self.tree.applies("gross_margin", "bank"))

    def test_all_tabled_nodes_present_with_templates(self):
        self.assertEqual(len(self.corporate_leaves()), len(CORPORATE_LEAVES))
        self.assertEqual(len(self.bank_leaves()), len(BANK_LEAVES))
        for node_id, templates in DERIVED.items():
            self.assertEqual(self.tree.derived(node_id).templates, templates, node_id)

    def test_every_corporate_statement_leaf_has_code(self):
        for leaf in self.corporate_leaves():
            if leaf.source == "statement":
                self.assertTrue(leaf.codes, leaf.id)

    def test_bank_leaves_have_two_aliases(self):
        for leaf in self.bank_leaves():
            if not leaf.codes:
                self.assertGreaterEqual(len(leaf.aliases), 2, leaf.id)

    def test_bank_loan_children_name_section_parent(self):
        for child in ("customer_loans_gross", "customer_loan_provision"):
            self.assertEqual(self.tree.leaf(child, "bank").parent, "customer_loans_section")

    def test_catalog_ids_reused(self):
        for node_id in CATALOG_STATEMENT_IDS:
            self.assertIn(node_id, METRIC_CATALOG)
            self.assertEqual(self.tree.kind(node_id), "leaf", node_id)
            self.assertTrue(self.tree.applies(node_id, "corporate"), node_id)

    def test_tabled_codes_and_periods(self):
        expected = {
            "gross_revenue": ("01", "flow"), "net_revenue": ("10", "flow"),
            "interest_expense": ("23", "flow"), "profit_before_tax": ("50", "flow"),
            "profit_after_tax": ("60", "flow"), "basic_eps": ("70", "flow"),
            "total_assets": ("270", "closing"), "owners_equity": ("400", "closing"),
            "cfo": ("20", "flow"), "depreciation": ("02", "flow"),
        }
        for node_id, (code, period) in expected.items():
            leaf = self.tree.leaf(node_id, "corporate")
            self.assertIn(code, leaf.codes, node_id)
            self.assertEqual(leaf.period_kind, period, node_id)
        self.assertIs(self.tree.leaf("basic_eps", "corporate").dimension, Dimension.MONEY_PER_SHARE)

    def test_question_phrasings_resolve(self):
        cases = {
            "lợi nhuận sau thuế TNDN": "profit_after_tax",
            "lãi cơ bản trên cổ phiếu": "basic_eps",
            "lưu chuyển tiền thuần từ hoạt động kinh doanh": "cfo",
            "chi phí lãi vay": "interest_expense",
            "vốn chủ sở hữu": "owners_equity",
            "nợ phải trả": "liabilities",
            "hệ số khả năng thanh toán lãi vay": "interest_coverage",
            "hệ số thanh toán nhanh": "quick_ratio",
            "vòng quay tổng tài sản": "asset_turnover",
            "số ngày tồn kho": "days_inventory",
            "biên lợi nhuận gộp": "gross_margin",
            "hệ số nợ phải trả trên vốn chủ sở hữu": "liabilities_to_equity",
            "lnst thuộc cổ đông công ty mẹ": "profit_after_tax_parent_share",
            "hệ số thanh toán hiện hành": "current_ratio",
            "Hệ số dòng tiền hoạt động trên nợ ngắn hạn": "cfo_to_current_liabilities",
            "hệ số chuyển đổi lợi nhuận (CFO/LNST)": "cfo_to_profit_after_tax",
            "tỷ trọng tài sản dài hạn trên tổng tài sản": "long_term_assets_share",
            "Khấu hao và phân bổ": "depreciation",
        }
        for text, node_id in cases.items():
            self.assertEqual(self.tree.find_by_alias(text, "corporate"), {node_id}, text)
        self.assertEqual(self.tree.find_by_alias("Dự phòng rủi ro cho vay khách hàng", "bank"),
                         {"customer_loan_provision"})
        self.assertEqual(self.tree.find_by_alias("Tiền gửi của khách hàng", "bank"),
                         {"customer_deposits"})

    def test_bare_cong_ty_me_is_scope_not_parent_share(self):
        # "công ty mẹ" alone names the separate report (query_paraphrase validator/prompts),
        # so this phrasing must not resolve to the code-61 parent-share leaf.
        found = self.tree.find_by_alias("Lợi nhuận sau thuế của công ty mẹ VJC năm 2023", "corporate")
        self.assertEqual(found, {"profit_after_tax"})

    def test_ratio_display_units(self):
        for node_id in DERIVED:
            node = self.tree.derived(node_id)
            if node.dimension is Dimension.RATIO:
                self.assertIn(node.display, ("percent", "times"), node_id)


if __name__ == "__main__":
    unittest.main()

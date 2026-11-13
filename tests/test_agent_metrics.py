import unittest

from src.agent.metrics import (
    CATALOG_TO_TREE, MetricChoice, MetricChoices, MetricMapping, map_metrics,
)
from src.calculation.metric_tree import MetricTree
from src.extraction.templates import TickerTemplates
from src.query_paraphrase.schemas import QueryPlan
from tests.agent_fixtures import ScriptedLlm
from tests.test_query_validation import plan, sq

TEMPLATES = TickerTemplates({"VJC": "corporate", "ACB": "bank"})
GOOD_PROPOSAL = {"id": "cash_to_assets", "label_vi": "Tiền trên tổng tài sản", "aliases": [],
                 "templates": ["corporate"], "dimension": "ratio",
                 "formulas": ["cash_and_equivalents / total_assets"]}


def make_plan(*subqueries):
    return QueryPlan.model_validate(plan("câu hỏi", *subqueries))


class MetricMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = MetricTree.load()

    def run_map(self, p, llm):
        return map_metrics(p, self.tree, TEMPLATES, llm)

    def test_catalog_map_targets_exist_in_tree(self):
        for target in CATALOG_TO_TREE.values():
            self.assertTrue(self.tree.applies(target, "corporate"), target)

    def test_catalog_metric_maps_without_llm(self):
        llm = ScriptedLlm()
        mappings, tree = self.run_map(make_plan(sq(ticker="VJC", year=2023, metric="revenue")), llm)
        self.assertEqual(mappings, (MetricMapping("q1", "net_revenue", "catalog"),))
        self.assertIs(tree, self.tree)
        self.assertEqual(llm.prompts, [])

    def test_metric_text_alias_maps_without_llm(self):
        llm = ScriptedLlm()
        mappings, _ = self.run_map(
            make_plan(sq(ticker="VJC", year=2023, metric_text="chi phí lãi vay")), llm)
        self.assertEqual(mappings, (MetricMapping("q1", "interest_expense", "alias"),))

    def test_unresolved_texts_use_one_llm_call(self):
        llm = ScriptedLlm(MetricChoices(choices=[
            MetricChoice(subquery_id="q1", node_id="gross_margin"),
            MetricChoice(subquery_id="q2", node_id="total_assets")]))
        p = make_plan(sq("q1", ticker="VJC", year=2023, metric_text="zzz một"),
                      sq("q2", ticker="VJC", year=2023, metric_text="zzz hai"))
        mappings, _ = self.run_map(p, llm)
        self.assertEqual(mappings, (MetricMapping("q1", "gross_margin", "llm"),
                                    MetricMapping("q2", "total_assets", "llm")))
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("gross_margin", llm.prompts[0])

    def test_prompt_requires_exact_line_item_match(self):
        llm = ScriptedLlm(MetricChoices(choices=[MetricChoice(subquery_id="q1")]))
        p = make_plan(sq("q1", ticker="VJC", year=2023, metric_text="Lãi tiền gửi"))
        mappings, _ = self.run_map(p, llm)
        self.assertEqual(mappings, (MetricMapping("q1", None, "unmapped"),))
        self.assertIn("đúng cùng một khoản mục", llm.prompts[0])
        self.assertIn("Lãi tiền gửi", llm.prompts[0])
        self.assertIn("trả null", llm.prompts[0])

    def test_llm_node_must_apply_to_template(self):
        llm = ScriptedLlm(MetricChoices(choices=[MetricChoice(subquery_id="q1", node_id="loan_to_deposit")]))
        mappings, _ = self.run_map(make_plan(sq(ticker="VJC", year=2023, metric_text="zzz")), llm)
        self.assertEqual(mappings, (MetricMapping("q1", None, "unmapped"),))

    def test_missing_choice_is_unmapped(self):
        llm = ScriptedLlm(MetricChoices(choices=[]))
        mappings, _ = self.run_map(make_plan(sq(ticker="VJC", year=2023, metric_text="zzz")), llm)
        self.assertEqual(mappings, (MetricMapping("q1", None, "unmapped"),))

    def test_valid_proposal_extends_tree(self):
        llm = ScriptedLlm(MetricChoices(choices=[MetricChoice(subquery_id="q1", proposal=GOOD_PROPOSAL)]))
        mappings, tree = self.run_map(make_plan(sq(ticker="VJC", year=2023, metric_text="zzz")), llm)
        self.assertEqual(mappings, (MetricMapping("q1", "cash_to_assets", "proposal"),))
        self.assertTrue(tree.applies("cash_to_assets", "corporate"))
        self.assertFalse(self.tree.applies("cash_to_assets", "corporate"))

    def test_rejected_proposal_is_unmapped(self):
        bad = dict(GOOD_PROPOSAL, formulas=["total_assets ** 2"])
        llm = ScriptedLlm(MetricChoices(choices=[MetricChoice(subquery_id="q1", proposal=bad)]))
        mappings, tree = self.run_map(make_plan(sq(ticker="VJC", year=2023, metric_text="zzz")), llm)
        self.assertEqual(mappings, (MetricMapping("q1", None, "unmapped"),))
        self.assertIs(tree, self.tree)

    def test_bank_template_used_for_bank_ticker(self):
        llm = ScriptedLlm(MetricChoices(choices=[MetricChoice(subquery_id="q1", node_id="loan_to_deposit")]))
        mappings, _ = self.run_map(make_plan(sq(ticker="ACB", year=2023, metric_text="zzz")), llm)
        self.assertEqual(mappings, (MetricMapping("q1", "loan_to_deposit", "llm"),))


if __name__ == "__main__":
    unittest.main()

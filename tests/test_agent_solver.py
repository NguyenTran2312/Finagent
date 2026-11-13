import unittest
from decimal import Decimal as D

from src.agent.gather import Gatherer
from src.agent.metrics import MetricMapping
from src.agent.solver import (
    NeedModel,
    SolutionDraft,
    build_context,
    claimed_answer,
    solve,
)
from src.agent.types import AgentConfig, DataNeed
from src.calculation import MONEY_SCALES, Answer, Dimension, Quantity, display_unit, to_display
from src.extraction.extractor import LeafExtractor
from src.query_paraphrase.schemas import QueryPlan
from tests.agent_fixtures import ScriptedLlm
from tests.test_extraction_extractor import default_store, templates, tree
from tests.test_query_validation import plan, sq

QUESTION = "Chi phí lãi vay của VJC năm 2023 là bao nhiêu tỷ đồng?"
PLAN = QueryPlan.model_validate(
    plan(QUESTION, sq(ticker="VJC", year=2023, metric_text="chi phí lãi vay")))
MAPPINGS = (MetricMapping("q1", "interest_expense", "alias"),)
INTEREST = NeedModel(node_id="interest_expense", ticker="VJC", year=2023)
GOOD_CODE = 'def solution():\n    return answer(fact("interest_expense", "VJC", 2023), "ty")\n'
UNDECLARED = "code used data not declared in needs"


def draft(code=GOOD_CODE, value="-1950.25", unit="ty", needs=(INTEREST,)):
    return SolutionDraft(code=code, claimed_value=value, claimed_unit=unit, needs=list(needs))


def make_gatherer():
    factory = lambda loc: LeafExtractor(default_store(), tree, templates, fallback=loc)
    return Gatherer(tree, templates, factory, None, 6, "consolidated")


class ClaimedAnswerTest(unittest.TestCase):
    def test_claimed_answer_round_trips_every_quantity_unit(self):
        units = [*MONEY_SCALES, "percent", "percentage_points", "times", "days", "money_per_share"]
        for unit in units:
            for value in ("12.5", "-1950.25", "0"):
                with self.subTest(unit=unit, value=value):
                    claimed = claimed_answer(draft(value=value, unit=unit))
                    self.assertEqual((claimed.kind, claimed.unit), ("quantity", unit))
                    self.assertEqual(to_display(claimed.value, display_unit(unit)), D(value))

    def test_claimed_answer_dimensions(self):
        self.assertEqual(claimed_answer(draft(value="2", unit="ty")).value,
                         Quantity(D("2000000000"), Dimension.MONEY))
        self.assertEqual(claimed_answer(draft(value="12.5", unit="percent")).value,
                         Quantity(D("0.125"), Dimension.RATIO))
        self.assertEqual(claimed_answer(draft(value="3", unit="days")).value.dimension, Dimension.DAYS)

    def test_claimed_answer_other_kinds(self):
        self.assertEqual(claimed_answer(draft(value="2023", unit="year")), Answer("year", 2023, "year"))
        self.assertEqual(claimed_answer(draft(value="3", unit="count")), Answer("count", 3, "count"))
        self.assertEqual(claimed_answer(draft(value="VJC", unit="ticker")), Answer("ticker", "VJC", "ticker"))
        self.assertEqual(claimed_answer(draft(value="true", unit="boolean")), Answer("boolean", True, "boolean"))
        self.assertEqual(claimed_answer(draft(value="False", unit="boolean")), Answer("boolean", False, "boolean"))

    def test_claimed_answer_unparseable_is_none(self):
        for value, unit in (("khoảng 2 tỷ", "ty"), ("NaN", "ty"), ("2", "parsecs"), ("2023.5", "year"),
                            ("maybe", "boolean"), ("", "ticker")):
            with self.subTest(value=value, unit=unit):
                self.assertIsNone(claimed_answer(draft(value=value, unit=unit)))


class BuildContextTest(unittest.TestCase):
    def test_context_from_question_and_mappings(self):
        mappings = (*MAPPINGS, MetricMapping("q2", None, "unmapped"))
        context = build_context(QUESTION, mappings, templates, "separate")
        self.assertEqual(context.default_scope, "separate")
        self.assertEqual(dict(context.ticker_templates), {"VJC": "corporate", "ACB": "bank"})
        self.assertEqual(context.question_literals, frozenset({D(2023)}))
        self.assertEqual((context.requested_unit, context.requested_kind), ("ty", "quantity"))
        self.assertEqual(context.named_metrics, frozenset({"interest_expense"}))


class SolveTest(unittest.TestCase):
    def setUp(self):
        self.gatherer = make_gatherer()
        self.context = build_context(QUESTION, MAPPINGS, templates, "consolidated")
        self.seen = []

    def run_solve(self, *drafts, config=AgentConfig()):
        self.llm = ScriptedLlm(*drafts)
        return solve(QUESTION, PLAN, MAPPINGS, self.gatherer, tree, self.context, self.llm, config,
                     self.seen.append)

    def pregather(self):
        self.gatherer.gather([DataNeed("interest_expense", "VJC", 2023)])

    def test_good_draft_passes_first_attempt(self):
        self.pregather()
        result = self.run_solve(draft())
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.result.answer.value, Quantity(D("-1950254195864"), Dimension.MONEY))
        (attempt,) = result.attempts
        self.assertEqual((attempt.number, attempt.outcome, attempt.problems, attempt.gathered),
                         (1, "passed", (), ()))
        self.assertEqual(attempt.needs, (DataNeed("interest_expense", "VJC", 2023),))
        self.assertEqual(result.failed, ())
        self.assertEqual(self.seen, list(result.attempts))
        prompt = self.llm.prompts[0]
        self.assertIn(QUESTION, prompt)
        self.assertIn("q1 → interest_expense", prompt)
        self.assertIn("interest_expense VJC 2023 consolidated = -1950254195864", prompt)
        self.assertIn("Read one reported line item", prompt)  # fact() docstring
        self.assertIn("No attribute access at all", prompt)  # code constraints

    def test_bad_then_good_draft(self):
        self.pregather()
        bad = 'def solution():\n    import os\n    return answer(fact("interest_expense", "VJC", 2023), "ty")\n'
        result = self.run_solve(draft(code=bad), draft())
        self.assertEqual(result.status, "passed")
        self.assertEqual([a.outcome for a in result.attempts], ["repair", "passed"])
        self.assertTrue(any("import is forbidden" in p for p in result.attempts[0].problems))
        self.assertIn("import is forbidden", self.llm.prompts[1])
        self.assertIn("import os", self.llm.prompts[1])
        self.assertNotIn("import is forbidden", self.llm.prompts[0])
        self.assertEqual(self.seen, list(result.attempts))

    def test_gate_errors_every_attempt_fail(self):
        self.pregather()
        result = self.run_solve(*(draft(value="5") for _ in range(4)))
        self.assertEqual(result.status, "gates_failed")
        self.assertIsNone(result.result)
        self.assertEqual(len(result.attempts), 4)
        self.assertEqual({a.outcome for a in result.attempts}, {"repair"})
        self.assertTrue(any(p.startswith("agreement") for p in result.attempts[-1].problems))
        self.assertEqual(result.failed, ())
        self.assertEqual(len(self.llm.prompts), 4)

    def test_new_need_gathers_before_running(self):
        self.assertEqual(len(self.gatherer.facts), 0)
        result = self.run_solve(draft())
        self.assertEqual(result.status, "passed")
        (attempt,) = result.attempts
        self.assertEqual([(l.leaf_id, l.outcome) for l in attempt.gathered], [("interest_expense", "Found")])
        self.assertEqual(attempt.outcome, "passed")
        self.assertEqual(len(self.gatherer.facts), 1)

    def test_failed_declared_need_is_facts_missing(self):
        code = 'def solution():\n    return answer(fact("interest_expense", "VJC", 2030), "ty")\n'
        need = NeedModel(node_id="interest_expense", ticker="VJC", year=2030)
        result = self.run_solve(draft(code=code, needs=[need]))
        self.assertEqual(result.status, "facts_missing")
        self.assertIsNone(result.result)
        (attempt,) = result.attempts
        self.assertEqual(attempt.outcome, "facts_missing")
        self.assertEqual([(l.leaf_id, l.year, l.outcome) for l in result.failed],
                         [("interest_expense", 2030, "ReportMissing")])
        self.assertEqual(self.seen, [attempt])

    def test_undeclared_data_is_repair(self):
        self.pregather()
        code = 'def solution():\n    return answer(fact("interest_expense", "VJC", 2022), "ty")\n'
        result = self.run_solve(draft(code=code, needs=[]), draft())
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.attempts[0].problems, (UNDECLARED,))
        self.assertEqual(result.attempts[0].gathered, ())
        self.assertIn(UNDECLARED, self.llm.prompts[1])

    def test_unknown_node_in_needs_is_repair_without_running(self):
        self.pregather()
        unknown = NeedModel(node_id="no_such_node", ticker="VJC", year=2023)
        result = self.run_solve(draft(needs=[INTEREST, unknown]), draft())
        self.assertEqual(result.status, "passed")
        first = result.attempts[0]
        # The code would pass if it ran; the repair proves it was not run.
        self.assertEqual((first.outcome, first.problems, first.gathered),
                         ("repair", ("unknown node id: no_such_node",), ()))
        self.assertIn("unknown node id: no_such_node", self.llm.prompts[1])

    def test_exhausted_gather_rounds_runs_without_gathering(self):
        config = AgentConfig(max_repairs=0, max_gather_rounds=0)
        result = self.run_solve(draft(), config=config)
        self.assertEqual(result.status, "gates_failed")
        (attempt,) = result.attempts
        self.assertEqual((attempt.gathered, attempt.problems), ((), (UNDECLARED,)))
        self.assertEqual(self.gatherer.log, ())


if __name__ == "__main__":
    unittest.main()

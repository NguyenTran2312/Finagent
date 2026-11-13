import unittest

from src.agent.agent import ReasoningAgent, _annual
from src.agent.llm import CountingLlm, LlmMalformedOutput, LlmUnavailable
from src.agent.metrics import MetricChoice, MetricChoices
from src.agent.present import plan_data
from src.agent.solver import NeedModel, SolutionDraft
from src.agent.types import AgentConfig
from src.extraction.extractor import LeafExtractor
from src.query_paraphrase import QATurn, QueryParaphraser, QueryPlan
from tests.agent_fixtures import ScriptedLlm
from tests.test_extraction_extractor import default_store, templates, tree
from tests.test_query_validation import REGISTRY, SETTINGS, plan, sq

Q_LOOKUP = "Chi phí lãi vay năm 2023 của VJC là bao nhiêu tỷ đồng?"
LOOKUP_PLAN = QueryPlan.model_validate(
    plan(Q_LOOKUP, sq(ticker="VJC", year=2023, metric_text="chi phí lãi vay", unit="tỷ đồng")))
INTEREST_2023 = NeedModel(node_id="interest_expense", ticker="VJC", year=2023)
GOOD_CODE = 'def solution():\n    return answer(fact("interest_expense", "VJC", 2023), "ty")\n'
BAD_CODE = 'def solution():\n    import os\n    return answer(fact("interest_expense", "VJC", 2023), "ty")\n'


def draft(code=GOOD_CODE, value="-1950.25", needs=(INTEREST_2023,)):
    return SolutionDraft(code=code, claimed_value=value, claimed_unit="ty", needs=list(needs))


def make_agent(*replies, config=AgentConfig()):
    scripted = ScriptedLlm(*replies)
    llm = CountingLlm(scripted)
    paraphraser = QueryParaphraser(lambda p: llm(p, QueryPlan), registry=REGISTRY, settings=SETTINGS,
                                   provider_errors=(LlmUnavailable,))
    factory = lambda locator: LeafExtractor(default_store(), tree, templates, fallback=locator)
    return ReasoningAgent(paraphraser, tree, templates, factory, llm, config), scripted, llm


def kinds(run):
    return [event.kind for event in run.events]


class ReasoningAgentTest(unittest.TestCase):
    def assertSingleAnswerEvent(self, run):
        self.assertEqual(kinds(run).count("answer"), 1)
        last = run.events[-1]
        self.assertEqual(last.kind, "answer")
        self.assertEqual((last.data["status"], last.data["reason"], last.data["text"]),
                         (run.status, run.reason, run.text))

    def test_lookup_answered(self):
        agent, scripted, _ = make_agent(LOOKUP_PLAN, draft())
        run = agent.run(Q_LOOKUP)
        self.assertEqual(run.status, "answered", run.text)
        self.assertIsNone(run.reason)
        self.assertTrue(run.text.startswith("Trả lời: -1.950,25 tỷ đồng."), run.text)
        self.assertIn("Nguồn:", run.text)
        self.assertEqual(len(run.sources), 1)
        (source,) = run.sources
        self.assertEqual(source.doc_id, "VJC23C")
        self.assertIn(f"{source.row_label} — VJC23C, trang {source.page}", run.text)
        self.assertEqual(run.answer.unit, "ty")
        self.assertEqual(run.llm_calls, 2)
        self.assertEqual(len(scripted.prompts), 2)
        self.assertGreaterEqual(run.seconds, 0)
        self.assertEqual(kinds(run), ["plan", "metrics", "gather", "solve", "answer"])
        self.assertSingleAnswerEvent(run)
        self.assertEqual(run.events[-1].data["sources"], [f"{source.row_label} — VJC23C, trang {source.page}"])
        self.assertEqual(run.events[1].data["mappings"],
                         [{"subquery_id": "q1", "node_id": "interest_expense", "how": "alias"}])
        self.assertEqual([(l["leaf_id"], l["outcome"]) for l in run.events[2].data["leaves"]],
                         [("interest_expense", "Found")])

    def test_llm_calls_is_a_per_run_delta(self):
        agent, _, llm = make_agent(LOOKUP_PLAN, draft(), LOOKUP_PLAN, draft())
        agent.run(Q_LOOKUP)
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.llm_calls, llm.calls), ("answered", 2, 4))

    def test_history_forwarded_and_clarify(self):
        question = "Còn năm 2023 của công ty đó?"
        history = [QATurn(question="Doanh thu VJC và HPG năm 2022?")]
        scripted_plan = QueryPlan.model_validate(
            plan(question, sq(ticker="VJC", year=2023, metric="revenue", refs={"ticker": 0, "metric": 0})))
        agent, scripted, _ = make_agent(scripted_plan)
        run = agent.run(question, history)
        self.assertEqual((run.status, run.reason, run.answer), ("clarify", "clarify", None))
        self.assertIn("Doanh thu VJC và HPG năm 2022?", scripted.prompts[0])
        self.assertTrue(run.text.startswith("Câu hỏi cần làm rõ thêm."), run.text)
        self.assertIn("- ticker inherited from turn 0 has several plausible antecedents", run.text)
        self.assertEqual(run.llm_calls, 1)
        self.assertEqual(kinds(run), ["answer"])
        self.assertSingleAnswerEvent(run)

    def test_mapping_history_is_accepted(self):
        question = "Còn năm 2023?"
        history = [{"question": "Doanh thu VJC năm 2022 là bao nhiêu?", "answer": "Khoảng 40.000 tỷ đồng."}]
        scripted_plan = QueryPlan.model_validate(
            plan(question, sq(ticker="VJC", year=2023, metric="revenue", refs={"ticker": 0, "metric": 0})))
        agent, scripted, _ = make_agent(scripted_plan, LlmUnavailable("TimeoutError"))
        run = agent.run(question, history)
        self.assertIn("Doanh thu VJC năm 2022 là bao nhiêu?", scripted.prompts[0])
        self.assertEqual(kinds(run)[:2], ["plan", "metrics"])  # the plan was valid

    def test_invalid_plan_refused(self):
        # An invalid plan gets one repair paraphrase; a second invalid plan is refused.
        scripted_plan = QueryPlan.model_validate(
            plan(Q_LOOKUP, sq(ticker="VJC", year=2022, metric_text="chi phí lãi vay", unit="tỷ đồng")))
        agent, scripted, _ = make_agent(scripted_plan, scripted_plan)
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason, run.answer), ("refused", "plan_invalid", None))
        self.assertTrue(run.text.startswith("Không phân tích được câu hỏi."), run.text)
        self.assertIn("- year 2022 is not in the question", run.text)
        self.assertEqual((run.llm_calls, len(scripted.prompts)), (2, 2))
        self.assertEqual(kinds(run), ["answer"])
        self.assertSingleAnswerEvent(run)

    def test_invalid_plan_repaired_once_with_validator_feedback(self):
        bad_plan = QueryPlan.model_validate(
            plan(Q_LOOKUP, sq(ticker="VJC", year=2022, metric_text="chi phí lãi vay", unit="tỷ đồng")))
        agent, scripted, _ = make_agent(bad_plan, LOOKUP_PLAN, draft())
        run = agent.run(Q_LOOKUP)
        self.assertEqual(run.status, "answered", run.text)
        self.assertEqual(run.llm_calls, 3)
        self.assertNotIn("<<<LOI_LAN_TRUOC>>>", scripted.prompts[0])
        self.assertIn("<<<LOI_LAN_TRUOC>>>", scripted.prompts[1])
        # Subquery issues carry their id; plan-level issues (no id) are the bare message.
        self.assertIn("- q1: year 2022 is not in the question", scripted.prompts[1])
        self.assertIn("- question year not covered by any subquery: ['2023']", scripted.prompts[1])
        self.assertEqual(kinds(run), ["plan", "metrics", "gather", "solve", "answer"])
        self.assertEqual(run.events[0].data, plan_data(LOOKUP_PLAN))

    def test_clarify_and_unavailable_are_not_repaired(self):
        question = "Doanh thu ZZZ năm 2023?"
        unknown = QueryPlan.model_validate(plan(question, sq(ticker="VJC", year=2023, metric="revenue")))
        agent, scripted, _ = make_agent(unknown)
        run = agent.run(question)
        self.assertEqual((run.status, run.llm_calls, len(scripted.prompts)), ("clarify", 1, 1))
        agent, scripted, _ = make_agent(LlmUnavailable("TimeoutError"))
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.llm_calls, len(scripted.prompts)), ("error", 1, 1))

    def test_missing_ticker_refused(self):
        question = "Doanh thu năm 2023 là bao nhiêu?"
        agent, _, _ = make_agent(QueryPlan.model_validate(plan(question, sq(year=2023, metric="revenue"))))
        run = agent.run(question)
        self.assertEqual((run.status, run.reason), ("refused", "plan_invalid"))
        self.assertIn("- q1", run.text)
        self.assertIn("mã cổ phiếu", run.text)
        self.assertEqual((run.llm_calls, kinds(run)), (1, ["plan", "answer"]))
        self.assertSingleAnswerEvent(run)

    def test_sub_annual_period_refused(self):
        question = "Doanh thu quý 2 năm 2023 của VJC là bao nhiêu?"
        scripted_plan = QueryPlan.model_validate(
            plan(question, sq(ticker="VJC", year=2023, metric="revenue", period="quý 2")))
        agent, _, _ = make_agent(scripted_plan)
        run = agent.run(question)
        self.assertEqual((run.status, run.reason, run.answer), ("refused", "plan_invalid", None))
        self.assertIn("- q1: kỳ quý 2 không được hỗ trợ (chỉ có số liệu năm)", run.text)
        self.assertEqual((run.llm_calls, kinds(run)), (1, ["plan", "answer"]))
        self.assertSingleAnswerEvent(run)

    def test_annual_periods_supported(self):
        for period in (None, "năm", "cả năm 2023", "Năm tài chính 2023", "FY2023", "12 tháng", "2023",
                       "cuối năm", "Cuối năm 2022", "tại cuối năm"):
            with self.subTest(period=period):
                self.assertTrue(_annual(period))
        for period in ("quý 2", "Q2/2023", "6 tháng đầu năm", "H1", "nửa đầu năm", "9 tháng", "tháng 12",
                       "bán niên", "lũy kế", "cuối quý 2", "cuối kỳ", "cuối", "cuối tháng 6 năm 2023"):
            with self.subTest(period=period):
                self.assertFalse(_annual(period))

    def test_malformed_draft_is_repaired(self):
        agent, _, _ = make_agent(LOOKUP_PLAN, LlmMalformedOutput("ValidationError"), draft())
        run = agent.run(Q_LOOKUP)
        self.assertEqual(run.status, "answered", run.text)
        self.assertEqual(kinds(run), ["plan", "metrics", "gather", "solve", "solve", "answer"])
        self.assertEqual((run.events[3].data["outcome"], run.events[3].data["problems"]),
                         ("repair", ["output did not match the SolutionDraft schema"]))
        self.assertEqual(run.llm_calls, 3)

    def test_malformed_output_outside_solve_is_error(self):
        agent, _, _ = make_agent(LlmMalformedOutput("ValidationError"))
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason), ("error", "llm_unavailable"))
        self.assertSingleAnswerEvent(run)

    def test_unmapped_metric_refused(self):
        question = "Chỉ số hạnh phúc của VJC năm 2023 là bao nhiêu?"
        scripted_plan = QueryPlan.model_validate(
            plan(question, sq(ticker="VJC", year=2023, metric_text="chỉ số hạnh phúc")))
        agent, _, _ = make_agent(scripted_plan, MetricChoices(choices=[MetricChoice(subquery_id="q1")]))
        run = agent.run(question)
        self.assertEqual((run.status, run.reason, run.answer), ("refused", "metric_unmapped", None))
        self.assertTrue(run.text.startswith("Không xác định được chỉ tiêu cần tính."), run.text)
        self.assertIn("- q1: chỉ số hạnh phúc", run.text)
        self.assertEqual((run.llm_calls, kinds(run)), (2, ["plan", "metrics", "answer"]))
        self.assertSingleAnswerEvent(run)

    def test_facts_missing_refused(self):
        question = "Chi phí lãi vay năm 2021 của VJC là bao nhiêu tỷ đồng?"
        scripted_plan = QueryPlan.model_validate(
            plan(question, sq(ticker="VJC", year=2021, metric_text="chi phí lãi vay", unit="tỷ đồng")))
        code = 'def solution():\n    return answer(fact("interest_expense", "VJC", 2021), "ty")\n'
        need = NeedModel(node_id="interest_expense", ticker="VJC", year=2021)
        agent, _, _ = make_agent(scripted_plan, draft(code=code, value="-1", needs=(need,)))
        run = agent.run(question)
        self.assertEqual((run.status, run.reason, run.answer), ("refused", "facts_missing", None))
        self.assertTrue(run.text.startswith("Không đủ số liệu để trả lời."), run.text)
        self.assertIn("- interest_expense VJC 2021 (hợp nhất): ReportMissing", run.text)
        self.assertEqual(run.sources, ())
        self.assertEqual(kinds(run), ["plan", "metrics", "gather", "solve", "answer"])
        self.assertEqual(run.events[3].data["outcome"], "facts_missing")
        self.assertSingleAnswerEvent(run)

    def test_gates_failed_refused_with_last_problems(self):
        config = AgentConfig(max_repairs=1)
        agent, _, _ = make_agent(LOOKUP_PLAN, draft(code=BAD_CODE), draft(code=BAD_CODE), config=config)
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason), ("refused", "gates_failed"))
        self.assertTrue(run.text.startswith("Không tạo được phép tính hợp lệ sau nhiều lần thử."), run.text)
        self.assertIn("import is forbidden", run.text)
        self.assertEqual(kinds(run), ["plan", "metrics", "gather", "solve", "solve", "answer"])
        self.assertSingleAnswerEvent(run)

    def test_llm_failure_mid_solve_is_error(self):
        agent, _, _ = make_agent(LOOKUP_PLAN, draft(code=BAD_CODE), LlmUnavailable("TimeoutError"))
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason, run.answer, run.sources), ("error", "llm_unavailable", None, ()))
        self.assertTrue(run.text.startswith("Không gọi được mô hình ngôn ngữ, vui lòng thử lại."), run.text)
        self.assertIn("- LlmUnavailable: TimeoutError", run.text)
        self.assertEqual(kinds(run), ["plan", "metrics", "gather", "solve", "answer"])
        self.assertEqual((run.events[3].data["number"], run.events[3].data["outcome"]), (1, "repair"))
        self.assertEqual(run.llm_calls, 3)
        self.assertSingleAnswerEvent(run)

    def test_paraphraser_unavailable_is_error(self):
        agent, _, _ = make_agent(LlmUnavailable("TimeoutError"))
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason, run.answer), ("error", "llm_unavailable", None))
        self.assertTrue(run.text.startswith("Không gọi được mô hình ngôn ngữ, vui lòng thử lại."), run.text)
        self.assertIn("LlmUnavailable", run.text)
        self.assertEqual((run.llm_calls, kinds(run)), (1, ["answer"]))
        self.assertSingleAnswerEvent(run)

    def test_mapping_llm_failure_is_error(self):
        question = "Chỉ số hạnh phúc của VJC năm 2023 là bao nhiêu?"
        scripted_plan = QueryPlan.model_validate(
            plan(question, sq(ticker="VJC", year=2023, metric_text="chỉ số hạnh phúc")))
        agent, _, _ = make_agent(scripted_plan, LlmUnavailable("TimeoutError"))
        run = agent.run(question)
        self.assertEqual((run.status, run.reason), ("error", "llm_unavailable"))
        self.assertEqual(kinds(run), ["plan", "answer"])
        self.assertSingleAnswerEvent(run)

    def test_on_event_called_in_order(self):
        extra = NeedModel(node_id="net_revenue", ticker="VJC", year=2023)
        agent, _, llm = make_agent(LOOKUP_PLAN, draft(code=BAD_CODE), draft(needs=(INTEREST_2023, extra)))
        seen = []
        run = agent.run(Q_LOOKUP, on_event=lambda event: seen.append((event, llm.calls)))
        self.assertEqual(run.status, "answered", run.text)
        self.assertEqual([event for event, _ in seen], list(run.events))
        self.assertEqual([(event.kind, calls) for event, calls in seen],
                         [("plan", 1), ("metrics", 1), ("gather", 1), ("solve", 2),
                          ("gather", 3), ("solve", 3), ("answer", 3)])
        self.assertEqual([l["leaf_id"] for l in run.events[4].data["leaves"]], ["net_revenue"])
        self.assertEqual([e.data["number"] for e in run.events if e.kind == "solve"], [1, 2])
        self.assertTrue(all(e.title for e in run.events))

    def test_locator_disabled_passes_none_to_factory(self):
        seen = []
        scripted = ScriptedLlm(LOOKUP_PLAN, draft())
        llm = CountingLlm(scripted)
        paraphraser = QueryParaphraser(lambda p: llm(p, QueryPlan), registry=REGISTRY, settings=SETTINGS,
                                       provider_errors=(LlmUnavailable,))

        def factory(locator):
            seen.append(locator)
            return LeafExtractor(default_store(), tree, templates, fallback=locator)

        for enabled in (False, True):
            agent = ReasoningAgent(paraphraser, tree, templates, factory, llm,
                                   AgentConfig(locator_enabled=enabled))
            if enabled:
                scripted._replies.extend([LOOKUP_PLAN, draft()])
            self.assertEqual(agent.run(Q_LOOKUP).status, "answered")
        self.assertIsNone(seen[0])
        self.assertIsNotNone(seen[1])


if __name__ == "__main__":
    unittest.main()

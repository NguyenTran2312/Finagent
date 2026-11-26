"""Advanced RAG baseline: config presets, isolated agentic behaviour, telemetry and the runner."""

import dataclasses
import importlib.util
import io
import json
import os
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal as D
from pathlib import Path
from unittest import mock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.agent.agent import ReasoningAgent
from src.agent.experiment import (
    ADAPTIVE_FIELDS,
    SYSTEMS,
    Pricing,
    RetrievalCounter,
    StageCounter,
    TokenUsage,
    error_record,
    experiment_config,
    open_runner,
    run_record,
    shared_fields,
    token_record,
)
from src.agent.factory import build_agent, reset_retriever_cache
from src.agent.llm import CountingLlm, LlmMalformedOutput, LlmUnavailable, openai_llm
from src.agent.solver import MALFORMED_DRAFT, NeedModel, SolutionDraft
from src.agent.types import AgentConfig
from src.calculation import Dimension, Quantity
from src.extraction.extractor import LeafExtractor
from src.extraction.retrieval import RankedTable
from src.query_paraphrase import QueryParaphraser, QueryPlan
from tests.agent_fixtures import ScriptedLlm
from tests.extraction_fixtures import record
from tests.test_extraction_extractor import default_store, doc, is_rows, is_table, store_of, templates, tree
from tests.test_query_validation import REGISTRY, SETTINGS, plan, sq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADVANCED, AGENTIC = experiment_config("advanced_rag"), experiment_config("agentic_rag")

# --- single hop: one subquery, one fact ---------------------------------------------------------
Q_LOOKUP = "Chi phí lãi vay năm 2023 của VJC là bao nhiêu tỷ đồng?"
LOOKUP_PLAN = QueryPlan.model_validate(
    plan(Q_LOOKUP, sq(ticker="VJC", year=2023, metric_text="chi phí lãi vay", unit="tỷ đồng")))
BAD_YEAR_PLAN = QueryPlan.model_validate(  # the validator rejects a year absent from the question
    plan(Q_LOOKUP, sq(ticker="VJC", year=2022, metric_text="chi phí lãi vay", unit="tỷ đồng")))
INTEREST_2023 = NeedModel(node_id="interest_expense", ticker="VJC", year=2023)
INTEREST_2022 = NeedModel(node_id="interest_expense", ticker="VJC", year=2022)
GOOD_CODE = 'def solution():\n    return answer(fact("interest_expense", "VJC", 2023), "ty")\n'
CODE_2022 = 'def solution():\n    return answer(fact("interest_expense", "VJC", 2022), "ty")\n'

# --- multi hop: two subqueries (two years) and a calculation -----------------------------------
Q_CHANGE = "Chi phí lãi vay của VJC năm 2023 thay đổi bao nhiêu tỷ đồng so với năm 2022?"
CHANGE_PLAN = QueryPlan.model_validate(plan(
    Q_CHANGE,
    sq("q1", ticker="VJC", year=2023, metric_text="chi phí lãi vay", unit="tỷ đồng"),
    sq("q2", ticker="VJC", year=2022, metric_text="chi phí lãi vay", unit="tỷ đồng"),
    operation="calculate",
    relationships=[{"kind": "calculate", "subquery_ids": ["q1", "q2"], "description": None}]))
CHANGE_CODE = ('def solution():\n    return answer(fact("interest_expense", "VJC", 2023)'
               ' - fact("interest_expense", "VJC", 2022), "ty")\n')

# --- metric with a prior-year balance: ROE = PAT / avg(owners_equity[t], owners_equity[t-1]) ----
Q_ROE = "ROE của VJC năm 2023 là bao nhiêu phần trăm?"
ROE_PLAN = QueryPlan.model_validate(plan(Q_ROE, sq(ticker="VJC", year=2023, metric_text="roe", unit="%")))
ROE_CODE = 'def solution():\n    return answer(metric("roe", "VJC", 2023), "percent")\n'
ROE_NEED = NeedModel(node_id="roe", ticker="VJC", year=2023)


def roe_store():
    income = is_rows(2023) + [["60", "Lợi nhuận sau thuế thu nhập doanh nghiệp", "", "1.000.000.000",
                               "900.000.000"]]
    balance = [["Mã số", "", "Thuyết minh", "31.12.2023 VND", "31.12.2022 VND"],
               ["400", "Vốn chủ sở hữu", "", "12.000.000.000", "8.000.000.000"]]
    return store_of((doc("VJC23C", 2023), [is_table("VJC23C", income),
                                          record("VJC23C", 2, 14, balance, kind="balance_sheet")]))


def draft(code=GOOD_CODE, value="-1950.25", unit="ty", needs=(INTEREST_2023,)):
    return SolutionDraft(code=code, claimed_value=value, claimed_unit=unit, needs=list(needs))


def make_agent(*replies, config, store=None):
    scripted = ScriptedLlm(*replies)
    stages = StageCounter(scripted)
    llm = CountingLlm(stages)
    paraphraser = QueryParaphraser(lambda p: llm(p, QueryPlan), registry=REGISTRY, settings=SETTINGS,
                                   provider_errors=(LlmUnavailable,))
    store = store if store is not None else default_store()
    factory = lambda locator: LeafExtractor(store, tree, templates, fallback=locator)
    return ReasoningAgent(paraphraser, tree, templates, factory, llm, config), scripted, stages


def kinds(run):
    return [event.kind for event in run.events]


class ExperimentConfigTest(unittest.TestCase):
    def test_systems_share_every_non_adaptive_field(self):
        self.assertEqual(shared_fields(ADVANCED), shared_fields(AGENTIC))
        self.assertEqual(set(f.name for f in dataclasses.fields(AgentConfig)) - set(shared_fields(ADVANCED)),
                         set(ADAPTIVE_FIELDS))

    def test_both_systems_use_hybrid_rerank_and_the_locator(self):
        for config in (ADVANCED, AGENTIC):
            self.assertEqual((config.retriever_mode, config.locator_enabled), ("hybrid_rerank", True))

    def test_adaptive_fields(self):
        self.assertEqual([getattr(ADVANCED, f) for f in ADAPTIVE_FIELDS], [0, 0, 0])
        self.assertEqual([getattr(AGENTIC, f) for f in ADAPTIVE_FIELDS], [1, 3, 2])

    def test_unknown_system_is_rejected(self):
        self.assertEqual(SYSTEMS, ("advanced_rag", "agentic_rag"))
        with self.assertRaises(ValueError):
            experiment_config("naive_rag")

    def test_agent_config_defaults_keep_the_agentic_behaviour(self):
        config = AgentConfig()
        self.assertEqual((config.max_plan_repairs, config.max_repairs, config.max_gather_rounds), (1, 3, 2))

    def test_agent_config_rejects_invalid_values(self):
        for kwargs in ({"max_repairs": -1}, {"max_gather_rounds": -1}, {"max_plan_repairs": -1},
                       {"max_locator_calls": -1}, {"max_repairs": True}, {"max_repairs": 1.5},
                       {"retriever_mode": "dense"}, {"default_scope": "parent"},
                       {"locator_enabled": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AgentConfig(**kwargs)


class AdvancedRagPipelineTest(unittest.TestCase):
    def test_single_hop_lookup_answered_with_citation(self):
        agent, scripted, stages = make_agent(LOOKUP_PLAN, draft(), config=ADVANCED)
        run = agent.run(Q_LOOKUP)
        self.assertEqual(run.status, "answered", run.text)
        self.assertTrue(run.text.startswith("Trả lời: -1.950,25 tỷ đồng."), run.text)
        (source,) = run.sources
        self.assertEqual((source.doc_id, source.chunk_id), ("VJC23C", "VJC23C:table:1"))
        self.assertIn(f"{source.row_label} — VJC23C, trang {source.page}", run.text)
        self.assertEqual(kinds(run), ["plan", "metrics", "gather", "solve", "answer"])
        self.assertEqual([(l.leaf_id, l.year, l.outcome) for l in run.evidence],
                         [("interest_expense", 2023, "Found")])
        self.assertEqual(stages.counts, Counter({"QueryPlan": 1, "SolutionDraft": 1}))

    def test_multi_hop_calculation_runs_in_the_sandbox(self):
        agent, _, stages = make_agent(CHANGE_PLAN, draft(code=CHANGE_CODE, value="-566.612950291",
                                                         needs=(INTEREST_2023, INTEREST_2022)),
                                      config=ADVANCED)
        run = agent.run(Q_CHANGE)
        self.assertEqual(run.status, "answered", run.text)
        self.assertEqual(run.answer.value, Quantity(D("-566612950291"), Dimension.MONEY))
        # Both years come from the initial collection (decomposition), none from the solver.
        self.assertEqual(sorted((l.leaf_id, l.year) for l in run.evidence),
                         [("interest_expense", 2022), ("interest_expense", 2023)])
        self.assertEqual(kinds(run).count("gather"), 1)
        self.assertEqual(len(run.sources), 2)
        self.assertEqual(stages.counts, Counter({"QueryPlan": 1, "SolutionDraft": 1}))

    def test_metric_dependencies_with_prior_year_balance_are_collected_up_front(self):
        agent, _, _ = make_agent(ROE_PLAN, draft(code=ROE_CODE, value="10", unit="percent", needs=(ROE_NEED,)),
                                 config=ADVANCED, store=roe_store())
        run = agent.run(Q_ROE)
        self.assertEqual(run.status, "answered", run.text)
        self.assertEqual(run.answer.value, Quantity(D("0.1"), Dimension.RATIO))
        found = {(l.leaf_id, l.year) for l in run.evidence if l.outcome == "Found"}
        self.assertLessEqual({("profit_after_tax", 2023), ("owners_equity", 2023), ("owners_equity", 2022)},
                             found)
        self.assertEqual(kinds(run).count("gather"), 1)  # no solver-driven gathering

    def test_invalid_plan_is_refused_without_repair(self):
        agent, scripted, stages = make_agent(BAD_YEAR_PLAN, config=ADVANCED)
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason), ("refused", "plan_invalid"))
        self.assertIn("- year 2022 is not in the question", run.text)
        self.assertEqual((run.llm_calls, stages.counts), (1, Counter({"QueryPlan": 1})))
        self.assertEqual(kinds(run), ["answer"])

    def test_agentic_config_repairs_the_same_invalid_plan(self):
        agent, scripted, _ = make_agent(BAD_YEAR_PLAN, LOOKUP_PLAN, draft(), config=AGENTIC)
        run = agent.run(Q_LOOKUP)
        self.assertEqual(run.status, "answered", run.text)
        self.assertIn("year 2022 is not in the question", scripted.prompts[1])

    def test_failed_verification_is_final(self):
        agent, scripted, stages = make_agent(LOOKUP_PLAN, draft(value="5"), config=ADVANCED)
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason, run.answer), ("refused", "gates_failed", None))
        self.assertIn("agreement", run.text)  # the gate's failure reason is kept
        self.assertEqual(stages.counts["SolutionDraft"], 1)
        (solve,) = [e for e in run.events if e.kind == "solve"]
        self.assertEqual(solve.data["outcome"], "repair")

    def test_agentic_config_regenerates_after_the_same_failure(self):
        agent, scripted, stages = make_agent(LOOKUP_PLAN, draft(value="5"), draft(), config=AGENTIC)
        run = agent.run(Q_LOOKUP)
        self.assertEqual(run.status, "answered", run.text)
        self.assertEqual(stages.counts["SolutionDraft"], 2)
        self.assertIn("Problems found:", scripted.prompts[-1])

    def test_sandbox_failure_is_final(self):
        bad = 'def solution():\n    import os\n    return answer(fact("interest_expense", "VJC", 2023), "ty")\n'
        agent, _, stages = make_agent(LOOKUP_PLAN, draft(code=bad), config=ADVANCED)
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason), ("refused", "gates_failed"))
        self.assertIn("import is forbidden", run.text)
        self.assertEqual(stages.counts["SolutionDraft"], 1)

    def test_malformed_solution_is_final(self):
        agent, _, stages = make_agent(LOOKUP_PLAN, LlmMalformedOutput("ValidationError"), config=ADVANCED)
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason), ("refused", "gates_failed"))
        self.assertIn(MALFORMED_DRAFT, run.text)
        self.assertEqual(stages.counts["SolutionDraft"], 1)

    def test_solver_cannot_gather_after_the_initial_collection(self):
        agent, _, stages = make_agent(LOOKUP_PLAN, draft(code=CODE_2022, needs=(INTEREST_2022,)),
                                      config=ADVANCED)
        run = agent.run(Q_LOOKUP)
        self.assertEqual((run.status, run.reason), ("refused", "data_not_gathered"))
        self.assertTrue(run.text.startswith("Phép tính cần số liệu chưa được thu thập."), run.text)
        self.assertIn("- metric: interest_expense VJC 2022 (hợp nhất)", run.text)
        self.assertEqual([(l.leaf_id, l.year) for l in run.evidence], [("interest_expense", 2023)])
        self.assertEqual(kinds(run).count("gather"), 1)
        self.assertEqual(stages.counts["SolutionDraft"], 1)

    def test_agentic_config_gathers_the_solvers_new_needs(self):
        replies = [draft(code=CODE_2022, needs=(INTEREST_2022,))] * 4
        agent, _, _ = make_agent(LOOKUP_PLAN, *replies, config=AGENTIC)
        run = agent.run(Q_LOOKUP)
        self.assertIn(("interest_expense", 2022), [(l.leaf_id, l.year) for l in run.evidence])
        self.assertEqual(kinds(run).count("gather"), 2)
        self.assertNotEqual(run.reason, "data_not_gathered")

    def test_missing_evidence_is_refused(self):
        question = "Chi phí lãi vay năm 2030 của VJC là bao nhiêu tỷ đồng?"
        missing_plan = QueryPlan.model_validate(
            plan(question, sq(ticker="VJC", year=2030, metric_text="chi phí lãi vay", unit="tỷ đồng")))
        code = GOOD_CODE.replace("2023", "2030")
        need = NeedModel(node_id="interest_expense", ticker="VJC", year=2030)
        agent, _, stages = make_agent(missing_plan, draft(code=code, needs=(need,)), config=ADVANCED)
        run = agent.run(question)
        self.assertEqual((run.status, run.reason, run.answer, run.sources),
                         ("refused", "facts_missing", None, ()))
        self.assertTrue(run.text.startswith("Không đủ số liệu để trả lời."), run.text)
        self.assertIn("interest_expense VJC 2030 (hợp nhất): ReportMissing", run.text)
        self.assertEqual(stages.counts["SolutionDraft"], 1)

    def test_first_attempt_prompts_are_identical_across_systems(self):
        prompts = []
        for config in (ADVANCED, AGENTIC):
            agent, scripted, _ = make_agent(LOOKUP_PLAN, draft(), config=config)
            self.assertEqual(agent.run(Q_LOOKUP).status, "answered")
            prompts.append(scripted.prompts)
        self.assertEqual(prompts[0], prompts[1])


class TelemetryTest(unittest.TestCase):
    def test_stage_counter_counts_failures_by_schema(self):
        stages = StageCounter(ScriptedLlm(LOOKUP_PLAN, LlmUnavailable("TimeoutError")))
        stages("p", QueryPlan)
        with self.assertRaises(LlmUnavailable):
            stages("p", SolutionDraft)
        self.assertEqual(stages.counts, Counter({"QueryPlan": 1, "SolutionDraft": 1}))

    def test_retrieval_counter_counts_and_delegates(self):
        class Inner:
            mode = "hybrid_rerank"

            def __init__(self):
                self.fail = False

            def rank(self, ticker, scope, report_year, leaf):
                if self.fail:
                    raise RuntimeError("down")
                return (RankedTable("c1", 1),)

        inner, counter = Inner(), RetrievalCounter()
        wrapped = counter(inner)
        self.assertEqual(wrapped.mode, "hybrid_rerank")
        self.assertEqual(wrapped.rank("VJC", "consolidated", 2023, None), (RankedTable("c1", 1),))
        inner.fail = True
        with self.assertRaises(RuntimeError):
            counter(inner).rank("VJC", "consolidated", 2023, None)
        self.assertEqual(counter.calls, 2)

    def test_token_usage_reads_usage_metadata_and_llm_output(self):
        usage = TokenUsage()
        message = AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        usage.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        usage.on_llm_end(LLMResult(generations=[[]], llm_output={"token_usage": {"prompt_tokens": 3,
                                                                              "completion_tokens": 2}}))
        usage.on_llm_end(LLMResult(generations=[[]]))
        usage.on_llm_error(TimeoutError())
        self.assertEqual(usage.snapshot(), (13, 6, 2, 1, 1))

    def test_token_usage_is_wired_through_model_callbacks(self):
        usage = TokenUsage()
        message = AIMessage(content="x", usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9})
        GenericFakeChatModel(messages=iter([message]), callbacks=[usage]).invoke("p")
        self.assertEqual(usage.snapshot(), (7, 2, 1, 0, 0))

    def test_token_record_cost_only_when_complete(self):
        pricing = Pricing(D("2"), D("8"))
        complete = token_record((0, 0, 0, 0, 0), (1_000_000, 500_000, 3, 0, 0), pricing)
        self.assertEqual(complete["total"], 1_500_000)
        self.assertEqual(D(complete["estimated_cost_usd"]), D("6"))
        partial = token_record((0, 0, 0, 0, 0), (100, 50, 1, 1, 0), pricing)
        self.assertEqual((partial["complete"], partial["estimated_cost_usd"]), (False, None))
        self.assertIsNone(token_record((5, 5, 1, 0, 0), (5, 5, 1, 0, 1), pricing))  # nothing reported
        self.assertIsNone(token_record((0, 0, 0, 0, 0), (10, 5, 1, 0, 0), None)["estimated_cost_usd"])

    def test_pricing_is_validated(self):
        for values in ((D("-1"), D("1")), (D("NaN"), D("1")), (1, D("1"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Pricing(*values)

    def test_openai_llm_passes_callbacks_to_the_client_it_builds(self):
        usage = TokenUsage()
        with mock.patch("langchain_openai.ChatOpenAI") as chat:
            openai_llm("m", callbacks=[usage])
        self.assertEqual(chat.call_args.kwargs["callbacks"], [usage])
        with self.assertRaises(ValueError):
            openai_llm(client=object(), callbacks=[usage])

    def test_build_agent_wraps_the_shared_retriever(self):
        reset_retriever_cache()
        self.addCleanup(reset_retriever_cache)
        shared, counter = mock.Mock(mode="hybrid_rerank"), RetrievalCounter()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("src.retrieval.table_retriever.open_table_retriever", return_value=shared):
                agent = build_agent(ADVANCED, llm=CountingLlm(ScriptedLlm()), db=Path(tmp) / "c.sqlite3",
                                    bm25_dir=Path(tmp) / "bm25", chroma_dir=Path(tmp) / "chroma",
                                    retriever_wrapper=counter)
                retriever = agent._extractor_factory(None)._retriever
        retriever.rank("VJC", "consolidated", 2023, None)
        shared.rank.assert_called_once_with("VJC", "consolidated", 2023, None)
        self.assertEqual((retriever.mode, counter.calls), ("hybrid_rerank", 1))


class RunRecordTest(unittest.TestCase):
    def runner(self, *replies, config=ADVANCED):
        built = {}

        def build(config, *, llm, db, bm25_dir, chroma_dir, retriever_wrapper):
            built.update(config=config, wrapper=retriever_wrapper)
            agent, _, _ = make_agent(config=config)
            # Route the agent's LLM through the runner's counters.
            return ReasoningAgent(QueryParaphraser(lambda p: llm(p, QueryPlan), registry=REGISTRY,
                                                   settings=SETTINGS, provider_errors=(LlmUnavailable,)),
                                  tree, templates, agent._extractor_factory, llm, config)

        runner = open_runner("advanced_rag" if config is ADVANCED else "agentic_rag", db=Path("db"),
                             bm25_dir=Path("bm25"), chroma_dir=Path("chroma"), model="test-model",
                             llm=ScriptedLlm(*replies), build=build)
        return runner, built

    def test_record_of_an_answered_run(self):
        runner, built = self.runner(LOOKUP_PLAN, draft())
        self.assertEqual(built["config"], ADVANCED)
        self.assertIsInstance(built["wrapper"], RetrievalCounter)
        rec = runner.run(7, Q_LOOKUP)
        json.dumps(rec, ensure_ascii=False)  # serializable
        self.assertEqual((rec["id"], rec["system"], rec["model"], rec["status"], rec["reason"]),
                         (7, "advanced_rag", "test-model", "answered", None))
        self.assertEqual(rec["config"]["max_repairs"], 0)
        self.assertEqual(rec["answer"], {"kind": "quantity", "value": {"value": "-1950254195864",
                                                                       "dimension": "money"},
                                         "unit": "ty", "display": "-1.950,25 tỷ đồng"})
        self.assertEqual([s["chunk_id"] for s in rec["sources"]], ["VJC23C:table:1"])
        self.assertEqual([(e["leaf_id"], e["outcome"], e["value"]) for e in rec["evidence"]],
                         [("interest_expense", "Found", "-1950254195864")])
        self.assertEqual(rec["calculation"], {"code": GOOD_CODE, "attempts": 1})
        self.assertEqual(rec["verification"], {"passed": True, "attempts": [
            {"number": 1, "outcome": "passed", "problems": []}]})
        telemetry = rec["telemetry"]
        self.assertEqual((telemetry["llm_calls"], telemetry["retrieval_calls"], telemetry["tokens"],
                          telemetry["solver_gather_rounds"]), (2, 0, None, 0))
        self.assertEqual(telemetry["llm_calls_by_stage"], {"QueryPlan": 1, "SolutionDraft": 1})

    def test_counters_are_per_question(self):
        runner, _ = self.runner(LOOKUP_PLAN, draft(), BAD_YEAR_PLAN)
        runner.run(1, Q_LOOKUP)
        rec = runner.run(2, Q_LOOKUP)
        self.assertEqual((rec["status"], rec["reason"]), ("refused", "plan_invalid"))
        self.assertEqual(rec["telemetry"]["llm_calls_by_stage"], {"QueryPlan": 1})
        self.assertEqual((rec["calculation"], rec["verification"]["passed"]),
                         ({"code": None, "attempts": 0}, False))

    def test_hybrid_retrieval_requires_a_chroma_dir(self):
        with self.assertRaises(ValueError):
            open_runner("advanced_rag", db=Path("db"), bm25_dir=Path("bm25"), chroma_dir=None,
                        llm=ScriptedLlm(), build=lambda *a, **k: None)

    def test_error_record_keeps_only_the_class_name(self):
        rec = error_record(question_id=1, question="q", system="advanced_rag", config=ADVANCED, model="m",
                           error=RuntimeError("secret"), seconds=0.5)
        self.assertEqual((rec["status"], rec["reason"]), ("error", "exception:RuntimeError"))
        self.assertNotIn("secret", json.dumps(rec))


def _load_script():
    spec = importlib.util.spec_from_file_location("run_experiment", PROJECT_ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunExperimentScriptTest(unittest.TestCase):
    def setUp(self):
        self.script = _load_script()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        (self.tmp / "db.sqlite3").write_bytes(b"")
        for name in ("bm25", "chroma"):
            (self.tmp / name).mkdir()
        self.questions = self.tmp / "q.jsonl"
        self.questions.write_text(
            json.dumps({"id": 1, "question": "Câu một?", "answer": "GOLD", "program": "GOLD"}) + "\n"
            + json.dumps({"id": 2, "question": "Câu hai?"}) + "\n", encoding="utf-8")
        self.out = self.tmp / "out.jsonl"

    def argv(self, *extra):
        return ["--system", "advanced_rag", "--questions", str(self.questions), "--out", str(self.out),
                "--db", str(self.tmp / "db.sqlite3"), "--bm25-dir", str(self.tmp / "bm25"),
                "--chroma-dir", str(self.tmp / "chroma"), *extra]

    def main(self, *extra, runner=None):
        runner = runner or FakeRunner()
        with mock.patch.object(self.script, "open_runner", return_value=runner) as opener, \
                mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = self.script.main(self.argv(*extra))
        return code, runner, opener

    def records(self):
        return [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]

    def test_read_questions_drops_every_other_field(self):
        items = self.script.read_questions(self.questions)
        self.assertEqual(items, [{"id": 1, "question": "Câu một?", "qa_history": []},
                                 {"id": 2, "question": "Câu hai?", "qa_history": []}])

    def test_read_questions_rejects_bad_lines(self):
        for line in ('{"question": "x"}', '{"id": 1, "question": " "}', '{"id": 1, "question": "x", "qa_history": 1}'):
            self.questions.write_text(line + "\n", encoding="utf-8")
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.script.read_questions(self.questions)
        self.questions.write_text('{"id": 1, "question": "a"}\n{"id": "1", "question": "b"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.script.read_questions(self.questions)

    def test_runs_every_question_and_never_passes_gold_fields(self):
        code, runner, opener = self.main()
        self.assertEqual(code, 0)
        self.assertEqual(opener.call_args.args, ("advanced_rag",))
        self.assertEqual(runner.calls, [(1, "Câu một?", []), (2, "Câu hai?", [])])
        self.assertEqual([r["id"] for r in self.records()], [1, 2])
        self.assertNotIn("GOLD", self.out.read_text(encoding="utf-8"))

    def test_refuses_a_non_empty_output_and_resumes(self):
        self.main("--limit", "1")
        with self.assertRaises(SystemExit):
            self.main()
        _, runner, _ = self.main("--resume")
        self.assertEqual([c[0] for c in runner.calls], [2])
        self.assertEqual([r["id"] for r in self.records()], [1, 2])

    def test_an_exception_becomes_an_error_record(self):
        self.main(runner=FakeRunner(fail_on={1}))
        first, second = self.records()
        self.assertEqual((first["status"], first["reason"]), ("error", "exception:RuntimeError"))
        self.assertEqual(second["status"], "answered")

    def test_rejects_unknown_ids_and_half_pricing(self):
        for extra in (("--ids", "9"), ("--input-price-per-mtok", "1")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.main(*extra)


class FakeRunner:
    config, model = ADVANCED, "m"

    def __init__(self, fail_on=()):
        self.calls = []
        self._fail_on = set(fail_on)

    def run(self, question_id, question, qa_history=()):
        self.calls.append((question_id, question, qa_history))
        if question_id in self._fail_on:
            raise RuntimeError("secret")
        return {"id": question_id, "status": "answered", "reason": None}


if __name__ == "__main__":
    unittest.main()

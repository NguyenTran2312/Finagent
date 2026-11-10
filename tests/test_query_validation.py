import copy
import json
from pathlib import Path
import unicodedata
import unittest

from pydantic import BaseModel, ConfigDict

from config.settings import ParaphraseSettings
from src.query_paraphrase.entities import EntityRegistry
from src.query_paraphrase.metric_catalog import METRIC_CATALOG
from src.query_paraphrase.schemas import QATurn, QueryPlan, Subquery
from src.query_paraphrase.validator import prepare_history, validate_plan


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = EntityRegistry.from_csv(ROOT / "data" / "code_stock.csv")
SETTINGS = ParaphraseSettings()
LABELS = {metric.id: metric.label_vi for metric in METRIC_CATALOG.values()}
SCOPE_TEXT = {"consolidated": "báo cáo hợp nhất", "separate": "báo cáo riêng", None: None}
Q_VJC = "Doanh thu VJC năm 2023 là bao nhiêu?"


def sq(sid="q1", *, ticker=None, year=None, metric=None, metric_text=None, scope=None,
       period=None, unit=None, company_name=None, refs=None, search=None, question=None):
    """Subquery payload whose texts are derived from its own metadata unless overridden."""
    parts = (LABELS.get(metric) or metric_text, ticker, year, period, SCOPE_TEXT[scope], unit)
    text = " ".join(str(part) for part in parts if part) or "Thông tin tài chính"
    return {"id": sid, "question": question or text + "?", "search_query": search or text,
            "ticker": ticker, "company_name": company_name, "year": year, "period": period,
            "scope": scope, "metric": metric, "metric_text": metric_text, "unit": unit,
            "context_refs": refs or {}}


def plan(question, *subqueries, operation="lookup", description=None, relationships=()):
    return {"original_question": question, "operation": operation,
            "operation_description": description, "subqueries": list(subqueries),
            "relationships": list(relationships)}


def run(raw, question, history=None, *, registry=REGISTRY, settings=SETTINGS):
    return validate_plan(raw, question, history, registry=registry, settings=settings)


class AdapterPlan(BaseModel):
    """Stand-in for a model returned by a structured-output adapter (different class)."""
    model_config = ConfigDict(extra="allow")


class ValidationTestCase(unittest.TestCase):
    def assertValid(self, result):
        self.assertEqual(result.status, "valid", result.issues)
        self.assertIsInstance(result.plan, QueryPlan)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.semantic_status, "not_run")

    def assertRejected(self, result, status, *codes):
        self.assertEqual(result.status, status, result.issues)
        self.assertIsNone(result.plan)
        self.assertEqual(result.semantic_status, "not_run")
        found = {issue.code for issue in result.issues}
        self.assertTrue(found, "non-valid result must explain itself")
        self.assertTrue(set(codes) <= found, f"expected {codes}, got {result.issues}")

    def assertInvalid(self, result, *codes):
        self.assertRejected(result, "invalid", *codes)

    def assertClarify(self, result, *codes):
        self.assertRejected(result, "needs_clarification", *codes)


class StructuralValidationTests(ValidationTestCase):
    def test_simple_lookup_is_valid_as_mapping_json_and_model(self):
        payload = plan(Q_VJC, sq(ticker="VJC", year=2023, metric="revenue"))
        expected = QueryPlan.model_validate(payload)
        for raw in (payload, json.dumps(payload, ensure_ascii=False), expected):
            with self.subTest(raw_type=type(raw).__name__):
                result = run(raw, Q_VJC)
                self.assertValid(result)
                self.assertEqual(result.plan, expected)
        self.assertIsNot(run(expected, Q_VJC).plan, expected)

    def test_multiple_years_metrics_and_entities_are_valid(self):
        cases = {
            "Doanh thu VJC năm 2022 và 2023": [
                sq("q1", ticker="VJC", year=2022, metric="revenue"),
                sq("q2", ticker="VJC", year=2023, metric="revenue")],
            "Doanh thu thuần và LNST của VJC năm 2023": [
                sq("q1", ticker="VJC", year=2023, metric="net_revenue"),
                sq("q2", ticker="VJC", year=2023, metric="profit_after_tax")],
            "Doanh thu VJC và HPG năm 2023": [
                sq("q1", ticker="VJC", year=2023, metric="revenue"),
                sq("q2", ticker="HPG", year=2023, metric="revenue")],
        }
        for question, subqueries in cases.items():
            with self.subTest(question=question):
                self.assertValid(run(plan(question, *subqueries), question))

    def test_duplicate_ids_needs_and_unknown_relationship_ids_are_invalid(self):
        question = "Doanh thu VJC năm 2022 và 2023"
        first = sq("q1", ticker="VJC", year=2022, metric="revenue")
        second = sq("q2", ticker="VJC", year=2023, metric="revenue")
        self.assertInvalid(run(plan(question, first, dict(second, id="q1")), question),
                           "schema_invalid")
        dangling = plan(question, first, second, relationships=[
            {"kind": "compare", "subquery_ids": ["q1", "q9"], "description": None}])
        self.assertInvalid(run(dangling, question), "schema_invalid")
        repeated = dict(first, id="q3", search_query="  doanh thu  vjc 2022 ")
        result = run(plan(question, first, second, repeated), question)
        self.assertInvalid(result, "duplicate_subquery")
        self.assertIn("q3", {issue.subquery_id for issue in result.issues})

    def test_subquery_limit_defaults_to_eight_and_is_configurable(self):
        question = "Doanh thu VJC từ 2016 đến 2023"
        payload = plan(question, *(sq(f"q{year}", ticker="VJC", year=year, metric="revenue")
                                   for year in range(2016, 2024)))
        self.assertValid(run(payload, question))
        self.assertInvalid(run(payload, question, settings=ParaphraseSettings(max_subqueries=7)),
                           "too_many_subqueries")
        question = "Doanh thu VJC từ 2015 đến 2023"
        payload = plan(question, *(sq(f"q{year}", ticker="VJC", year=year, metric="revenue")
                                   for year in range(2015, 2024)))
        self.assertInvalid(run(payload, question), "too_many_subqueries")

    def test_original_question_must_match_input(self):
        subquery = sq(ticker="VJC", year=2023, metric="revenue")
        self.assertInvalid(run(plan("Doanh thu VJC năm 2022 là bao nhiêu?", subquery), Q_VJC),
                           "original_question_mismatch")
        decomposed = unicodedata.normalize("NFD", Q_VJC)
        self.assertValid(run(plan(decomposed, subquery), Q_VJC))

    def test_malformed_json_markdown_extra_fields_and_types_are_invalid(self):
        payload = plan(Q_VJC, sq(ticker="VJC", year=2023, metric="revenue"))
        text = json.dumps(payload, ensure_ascii=False)
        for raw in ("không phải JSON", f"```json\n{text}\n```", "Kế hoạch: " + text,
                    text + " xong", ""):
            with self.subTest(raw=raw[:20]):
                self.assertInvalid(run(raw, Q_VJC), "malformed_json")
        extra = copy.deepcopy(payload)
        extra["confidence"] = 0.9
        string_year = copy.deepcopy(payload)
        string_year["subqueries"][0]["year"] = "2023"
        for raw in (extra, string_year, json.dumps(extra), json.dumps(string_year)):
            with self.subTest(raw=str(raw)[:40]):
                self.assertInvalid(run(raw, Q_VJC), "schema_invalid")
        for raw in (None, 42, True, text.encode(), [payload]):
            with self.subTest(raw_type=type(raw).__name__):
                self.assertInvalid(run(raw, Q_VJC), "unsupported_output")

    def test_model_instances_are_revalidated_after_mutation(self):
        payload = plan(Q_VJC, sq(ticker="VJC", year=2023, metric="revenue"))
        mutated = QueryPlan.model_validate(payload)
        mutated.subqueries[0].year = "2023"
        self.assertInvalid(run(mutated, Q_VJC), "schema_invalid")
        appended = QueryPlan.model_validate(payload)
        appended.subqueries.append(appended.subqueries[0].model_copy())
        self.assertInvalid(run(appended, Q_VJC), "schema_invalid")
        changed = QueryPlan.model_validate(payload)
        changed.subqueries[0].year = 2022
        self.assertInvalid(run(changed, Q_VJC), "invented_year")
        constructed = QueryPlan.model_construct(original_question=Q_VJC, operation="lookup",
                                                operation_description=None, subqueries=[])
        self.assertInvalid(run(constructed, Q_VJC), "schema_invalid")

    def test_adapter_returned_models_are_revalidated_via_model_dump(self):
        payload = plan(Q_VJC, sq(ticker="VJC", year=2023, metric="revenue"))
        result = run(AdapterPlan(**payload), Q_VJC)
        self.assertValid(result)
        self.assertEqual(result.plan, QueryPlan.model_validate(payload))
        self.assertInvalid(run(AdapterPlan(**payload, confidence="high"), Q_VJC),
                           "schema_invalid")


class PreservationTests(ValidationTestCase):
    def check(self, asked, expected_code, **fields):
        result = run(plan(asked, sq(**fields)), asked)
        if expected_code is None:
            self.assertValid(result)
        else:
            self.assertInvalid(result, expected_code)
            if expected_code.startswith(("invented", "unknown", "metric", "search", "company")):
                self.assertIn("q1", {issue.subquery_id for issue in result.issues
                                     if issue.code == expected_code})
        return result

    def test_omitted_or_invented_ticker_and_year_are_invalid(self):
        cases = [
            (Q_VJC, "missing_ticker", dict(year=2023, metric="revenue")),
            ("Doanh thu năm 2023 là bao nhiêu?", "invented_ticker",
             dict(ticker="VJC", year=2023, metric="revenue")),
            (Q_VJC, "missing_year", dict(ticker="VJC", metric="revenue")),
            (Q_VJC, "invented_year", dict(ticker="VJC", year=2022, metric="revenue")),
            (Q_VJC, "unknown_ticker", dict(ticker="ZZZ", year=2023, metric="revenue")),
            (Q_VJC, "unknown_ticker", dict(ticker="vjc", year=2023, metric="revenue")),
            ("Doanh thu VJC năm 2022 và 2023", "missing_year",
             dict(ticker="VJC", year=2022, metric="revenue")),
        ]
        for question, code, fields in cases:
            with self.subTest(question=question, code=code):
                self.check(question, code, **fields)

    def test_recognized_metrics_and_distinct_variants_are_preserved(self):
        cases = [
            (Q_VJC, "missing_metric", dict(ticker="VJC", year=2023)),
            (Q_VJC, "invented_metric", dict(ticker="VJC", year=2023, metric="profit_after_tax")),
            ("Doanh thu thuần VJC năm 2023", "invented_metric",
             dict(ticker="VJC", year=2023, metric="revenue")),
            ("LNST hợp nhất của VJC năm 2023", "invented_metric",
             dict(ticker="VJC", year=2023, metric="profit_after_tax")),
            ("LNST thuộc cổ đông công ty mẹ VJC năm 2023", "metric_mismatch",
             dict(ticker="VJC", year=2023, metric="profit_after_tax_parent_share",
                  metric_text="LNST")),
            (Q_VJC, "metric_not_normalized",
             dict(ticker="VJC", year=2023, metric_text="doanh thu")),
            (Q_VJC, "unknown_metric", dict(ticker="VJC", year=2023, metric="sales")),
            ("LNST hợp nhất của VJC năm 2023", None,
             dict(ticker="VJC", year=2023, metric="consolidated_profit_after_tax")),
        ]
        for question, code, fields in cases:
            with self.subTest(question=question, code=code):
                self.check(question, code, **fields)

    def test_unknown_metric_keeps_metric_text_without_forcing_catalog(self):
        question = "Vốn lưu động ròng của VJC năm 2023?"
        self.check(question, None, ticker="VJC", year=2023, metric_text="vốn lưu động ròng")
        self.check(question, "invented_metric_text", ticker="VJC", year=2023,
                   metric_text="tiền và tương đương tiền")
        self.check(question, "unknown_metric", ticker="VJC", year=2023,
                   metric="working_capital", metric_text="vốn lưu động ròng")

    def test_explicit_scope_is_preserved_and_never_invented(self):
        deposit = "Lãi tiền gửi năm 2018 của công ty mẹ VJC?"
        fields = dict(ticker="VJC", year=2018, metric_text="lãi tiền gửi")
        self.check(deposit, None, scope="separate", **fields)
        self.check(deposit, "missing_scope", **fields)
        self.check(deposit, "invented_scope", scope="consolidated", **fields)
        consolidated = "Doanh thu hợp nhất VJC năm 2023"
        self.check(consolidated, None, ticker="VJC", year=2023, metric="revenue",
                   scope="consolidated")
        self.check(consolidated, "missing_scope", ticker="VJC", year=2023, metric="revenue")
        self.check(Q_VJC, "invented_scope", ticker="VJC", year=2023, metric="revenue",
                   scope="consolidated")

    def test_parent_share_profit_keeps_scope_null(self):
        question = "LNST thuộc cổ đông công ty mẹ VJC 2023"
        fields = dict(ticker="VJC", year=2023, metric="profit_after_tax_parent_share")
        result = self.check(question, None, **fields)
        self.assertIsNone(result.plan.subqueries[0].scope)
        self.check(question, "invented_scope", scope="separate", **fields)

    def test_explicit_period_is_preserved_and_never_invented(self):
        quarter = "Doanh thu VJC quý 1 năm 2023"
        fields = dict(ticker="VJC", year=2023, metric="revenue")
        self.check(quarter, None, period="quý 1", **fields)
        self.check(quarter, None, period="Q1", **fields)
        self.check(quarter, "missing_period", **fields)
        self.check(quarter, "invented_period", period="quý 2", **fields)
        self.check("Doanh thu VJC 6 tháng đầu năm 2023", None, period="6 tháng đầu năm", **fields)
        self.check("Doanh thu lũy kế 9 tháng của VJC năm 2023", None,
                   period="lũy kế 9 tháng", **fields)
        self.check("Doanh thu lũy kế 9 tháng của VJC năm 2023", "missing_period",
                   period="9 tháng", **fields)
        self.check(Q_VJC, "invented_period", period="cả năm", **fields)
        self.check(Q_VJC, "invented_period", period="Q4", **fields)

    def test_explicit_unit_is_preserved_and_never_invented(self):
        question = "Doanh thu VJC năm 2023 là bao nhiêu tỷ đồng?"
        fields = dict(ticker="VJC", year=2023, metric="revenue")
        self.check(question, None, unit="tỷ đồng", **fields)
        self.check(question, "missing_unit", **fields)
        self.check(question, "invented_unit", unit="triệu đồng", **fields)
        self.check(Q_VJC, "invented_unit", unit="USD", **fields)
        ratio = "Tỷ lệ LNST trên doanh thu của VJC năm 2023 (%)"
        subqueries = (sq("q1", ticker="VJC", year=2023, metric="profit_after_tax"),
                      sq("q2", ticker="VJC", year=2023, metric="revenue"))
        relation = [{"kind": "calculate", "subquery_ids": ["q1", "q2"], "description": None}]
        self.assertValid(run(plan(ratio, *subqueries, operation="calculate",
                                  description="q1 chia q2, đơn vị %", relationships=relation),
                             ratio))
        self.assertInvalid(run(plan(ratio, *subqueries, operation="calculate",
                                    relationships=relation), ratio), "missing_unit")

    def test_metadata_must_not_contradict_search_text(self):
        fields = dict(ticker="VJC", year=2023, metric="revenue")
        for override in (dict(search="Doanh thu HPG 2023"), dict(search="Doanh thu VJC 2022"),
                         dict(search="LNST VJC 2023"),
                         dict(search="Doanh thu VJC 2023 báo cáo riêng"),
                         dict(search="Doanh thu VJC quý 2 2023"),
                         dict(question="Doanh thu ZZZ năm 2023?")):
            with self.subTest(override=override):
                self.check(Q_VJC, "search_text_conflict", **fields, **override)
        self.check(Q_VJC, "company_name_mismatch", company_name="CTCP Tập đoàn Hòa Phát",
                   **fields)

    def test_operation_intent_must_match_question(self):
        two_years = (sq("q1", ticker="VJC", year=2022, metric="revenue"),
                     sq("q2", ticker="VJC", year=2023, metric="revenue"))
        cases = [
            ("So sánh doanh thu VJC năm 2022 và 2023", "compare", "lookup"),
            ("Tăng trưởng doanh thu VJC năm 2023 so với 2022", "calculate", "compare"),
            ("Chênh lệch doanh thu VJC năm 2022 và 2023", "calculate", "lookup"),
            ("Doanh thu VJC năm 2022 cộng năm 2023", "aggregate", "lookup"),
        ]
        for question, right, wrong in cases:
            with self.subTest(question=question):
                self.assertValid(run(plan(question, *two_years, operation=right), question))
                self.assertInvalid(run(plan(question, *two_years, operation=wrong), question),
                                   "operation_mismatch")
        assets = "Tổng tài sản của VJC năm 2023"
        self.check(assets, None, ticker="VJC", year=2023, metric_text="tổng tài sản")
        airport = "Doanh thu Tổng Công ty Cảng Hàng không Việt Nam - CTCP năm 2023"
        self.check(airport, None, ticker="ACV", year=2023, metric="revenue")
        question = "Doanh thu VJC năm 2022 và 2023"
        lookup_relation = plan(question, *two_years, relationships=[
            {"kind": "compare", "subquery_ids": ["q1", "q2"], "description": None}])
        self.assertInvalid(run(lookup_relation, question), "operation_mismatch")
        single = plan(question, *two_years, operation="compare", relationships=[
            {"kind": "compare", "subquery_ids": ["q1"], "description": None}])
        self.assertInvalid(run(single, question), "invalid_relationship")
        unsupported = plan(question, *two_years, operation="forecast")
        self.assertInvalid(run(unsupported, question), "schema_invalid")


class PairingTests(ValidationTestCase):
    QUESTION = "Doanh thu VJC năm 2022 và HPG năm 2023"

    def test_verifiable_pairs_reject_swaps_and_cartesian_products(self):
        def needs(*pairs):
            return [sq(f"q{i}", ticker=ticker, year=year, metric="revenue")
                    for i, (ticker, year) in enumerate(pairs)]
        self.assertValid(run(plan(self.QUESTION, *needs(("VJC", 2022), ("HPG", 2023))),
                             self.QUESTION))
        swapped = plan(self.QUESTION, *needs(("VJC", 2023), ("HPG", 2022)))
        self.assertInvalid(run(swapped, self.QUESTION), "pairing_mismatch")
        cartesian = plan(self.QUESTION, *needs(("VJC", 2022), ("VJC", 2023),
                                               ("HPG", 2022), ("HPG", 2023)))
        self.assertInvalid(run(cartesian, self.QUESTION), "pairing_mismatch")
        named = "Doanh thu VJC (CTCP Hàng không Vietjet) năm 2022 và HPG năm 2023"
        self.assertValid(run(plan(named, *needs(("VJC", 2022), ("HPG", 2023))), named))
        distributive = "Doanh thu VJC và HPG năm 2022 và 2023"
        self.assertValid(run(plan(distributive, *needs(("VJC", 2022), ("VJC", 2023),
                                                       ("HPG", 2022), ("HPG", 2023))),
                             distributive))

    def test_single_company_comparison_permits_exactly_the_two_needs(self):
        question = "So sánh doanh thu VJC 2022 và 2023"
        first = sq("q1", ticker="VJC", year=2022, metric="revenue")
        second = sq("q2", ticker="VJC", year=2023, metric="revenue")
        relation = [{"kind": "compare", "subquery_ids": ["q1", "q2"], "description": None}]
        result = run(plan(question, first, second, operation="compare",
                          relationships=relation), question)
        self.assertValid(result)
        self.assertEqual([(s.ticker, s.year) for s in result.plan.subqueries],
                         [("VJC", 2022), ("VJC", 2023)])
        self.assertInvalid(run(plan(question, first, operation="compare"), question),
                           "missing_year")
        extra = sq("q3", ticker="HPG", year=2023, metric="revenue")
        self.assertInvalid(run(plan(question, first, second, extra, operation="compare"),
                               question), "invented_ticker")
        self.assertInvalid(run(plan(question, first, second, dict(second, id="q3"),
                                    operation="compare"), question), "duplicate_subquery")

    def test_ambiguous_binding_needs_clarification(self):
        for question in ("Doanh thu năm 2022 của VJC và năm 2023 của HPG",
                         "Doanh thu VJC 2022, HPG và FPT 2023"):
            subqueries = [sq("q1", ticker="VJC", year=2022, metric="revenue"),
                          sq("q2", ticker="HPG", year=2023, metric="revenue")]
            if "FPT" in question:
                subqueries.append(sq("q3", ticker="FPT", year=2023, metric="revenue"))
            with self.subTest(question=question):
                self.assertClarify(run(plan(question, *subqueries), question),
                                   "ambiguous_binding")


class EntityTests(ValidationTestCase):
    def test_unknown_entity_blocks_even_when_plan_picks_a_real_ticker(self):
        question = "Doanh thu ZZZ năm 2023?"
        for ticker in ("VJC", None):
            with self.subTest(ticker=ticker):
                self.assertClarify(run(plan(question, sq(ticker=ticker, year=2023,
                                                         metric="revenue")), question),
                                   "unknown_entity")

    def test_ambiguous_alias_needs_clarification(self):
        registry = EntityRegistry.from_csv(ROOT / "data" / "code_stock.csv",
                                           aliases={"hãng bay": ["VJC", "ACV"]})
        question = "Doanh thu hãng bay năm 2023"
        result = run(plan(question, sq(ticker="VJC", year=2023, metric="revenue")), question,
                     registry=registry)
        self.assertClarify(result, "ambiguous_entity")

    def test_missing_company_without_mention_can_remain_null(self):
        question = "Doanh thu năm 2023 là bao nhiêu?"
        self.assertValid(run(plan(question, sq(year=2023, metric="revenue")), question))

    def test_full_name_resolution_and_metadata_consistency(self):
        question = "Doanh thu CTCP Hàng không Vietjet năm 2023"
        fields = dict(year=2023, metric="revenue")
        self.assertValid(run(plan(question, sq(ticker="VJC", company_name="CTCP Hàng không Vietjet",
                                               **fields)), question))
        self.assertInvalid(run(plan(question, sq(**fields)), question), "missing_ticker")
        fox = "LNST CTCP Viễn thông FPT năm 2023"
        fields = dict(year=2023, metric="profit_after_tax")
        self.assertValid(run(plan(fox, sq(ticker="FOX", **fields)), fox))
        self.assertInvalid(run(plan(fox, sq(ticker="FPT", **fields)), fox), "invented_ticker")

    def test_no_fuzzy_alias_without_injection(self):
        question = "Doanh thu Vietjet năm 2023"
        payload = plan(question, sq(ticker="VJC", company_name="Vietjet", year=2023,
                                    metric="revenue"))
        self.assertInvalid(run(payload, question), "invented_ticker", "company_name_mismatch")
        registry = EntityRegistry.from_csv(ROOT / "data" / "code_stock.csv",
                                           aliases={"Vietjet": "VJC"})
        self.assertValid(run(payload, question, registry=registry))

    def test_finance_acronyms_and_period_tokens_are_not_entities(self):
        question = "ROE của VJC FY2023?"
        self.assertValid(run(plan(question, sq(ticker="VJC", year=2023, metric_text="ROE")),
                             question))
        question = "EBIT của VJC FY23?"
        self.assertValid(run(plan(question, sq(ticker="VJC", metric_text="EBIT")), question))

    def test_lowercase_ticker_words_support_but_never_force(self):
        question = "doanh thu gas năm 2023"
        self.assertValid(run(plan(question, sq(ticker="GAS", year=2023, metric="revenue")),
                             question))
        # Final-review ruling: a ticker-word mention with no subquery ticker is ambiguous.
        self.assertClarify(run(plan(question, sq(year=2023, metric="revenue")), question),
                           "ambiguous_entity_mention")
        deficit = "Thâm hụt ngân sách năm 2023?"
        self.assertValid(run(plan(deficit, sq(year=2023, metric_text="thâm hụt ngân sách")),
                             deficit))


class HistoryTests(ValidationTestCase):
    PREVIOUS = QATurn(question="Doanh thu VJC năm 2022 là bao nhiêu?",
                      answer="Khoảng 40.000 tỷ đồng.")
    FOLLOW_UP = "Còn năm 2023?"

    def follow_up(self, history, refs, question=None, **fields):
        question = question or self.FOLLOW_UP
        values = dict(ticker="VJC", year=2023, metric="revenue")
        values.update(fields)
        return run(plan(question, sq(refs=refs, **values)), question, history)

    def test_prepare_history_keeps_last_turns_with_original_indexes(self):
        self.assertEqual(prepare_history(None, SETTINGS), [])
        raw = [{"question": f"Doanh thu VJC năm {2015 + i}?", "answer": None} for i in range(7)]
        raw[3] = QATurn(question="LNST HPG năm 2018?")
        snapshot = copy.deepcopy(raw)
        retained = prepare_history(raw, SETTINGS)
        self.assertEqual([index for index, _ in retained], [2, 3, 4, 5, 6])
        self.assertTrue(all(isinstance(turn, QATurn) for _, turn in retained))
        self.assertEqual(retained[1][1].question, "LNST HPG năm 2018?")
        self.assertEqual(raw, snapshot)
        self.assertEqual(prepare_history(raw, ParaphraseSettings(max_history_turns=0)), [])
        as_dicts = prepare_history([turn.model_dump() for _, turn in retained], SETTINGS)
        self.assertEqual([turn for _, turn in as_dicts], [turn for _, turn in retained])

    def test_malformed_history_and_question_arguments_raise(self):
        short = ParaphraseSettings(max_input_chars=40)
        for history in ("Doanh thu?", ["Doanh thu?"], [{"answer": "x"}], [{"question": " "}],
                        [{"question": "Doanh thu?", "answer": 1}],
                        [{"question": "Doanh thu?", "extra": 1}], {"question": "Doanh thu?"}):
            with self.subTest(history=history), self.assertRaises(ValueError):
                prepare_history(history, SETTINGS)
        for history in ([{"question": "x" * 41}], [{"question": "Doanh thu?", "answer": "x" * 41}]):
            with self.subTest(limit=True), self.assertRaises(ValueError):
                prepare_history(history, short)
        payload = plan(Q_VJC, sq(ticker="VJC", year=2023, metric="revenue"))
        for question in ("", "   ", None, 2023, "x" * 41):
            with self.subTest(question=question), self.assertRaises(ValueError):
                run(payload, question, settings=short)
        with self.assertRaises(ValueError):
            run(payload, Q_VJC, ["Doanh thu?"])

    def test_follow_up_inherits_missing_fields_only_with_valid_refs(self):
        for history in ([self.PREVIOUS], [self.PREVIOUS.model_dump()]):
            with self.subTest(history_type=type(history[0]).__name__):
                snapshot = copy.deepcopy(history)
                self.assertValid(self.follow_up(history, {"ticker": 0, "metric": 0}))
                self.assertEqual(history, snapshot)
        result = self.follow_up([self.PREVIOUS], {})
        self.assertInvalid(result, "invented_ticker", "invented_metric")

    def test_current_question_year_overrides_history(self):
        result = self.follow_up([self.PREVIOUS], {"ticker": 0, "metric": 0, "year": 0}, year=2022)
        self.assertInvalid(result, "context_overrides_question")
        stated = "Chỉ tiêu này của HPG năm 2023?"
        history = [QATurn(question="Doanh thu thuần VJC năm 2022?")]
        self.assertValid(self.follow_up(history, {"metric": 0}, question=stated, ticker="HPG",
                                        metric="net_revenue"))
        self.assertInvalid(self.follow_up(history, {"metric": 0, "ticker": 0}, question=stated,
                                          ticker="HPG", metric="net_revenue"),
                           "context_overrides_question")

    def test_answer_only_and_unrelated_context_fail(self):
        answer_only = [QATurn(question="Hãng bay nào có doanh thu lớn nhất năm 2022?",
                              answer="VJC có doanh thu lớn nhất.")]
        self.assertInvalid(self.follow_up(answer_only, {"ticker": 0, "metric": 0}),
                           "context_value_not_in_history")
        unrelated = [QATurn(question="LNST HPG năm 2022?")]
        self.assertInvalid(self.follow_up(unrelated, {"ticker": 0, "metric": 0}),
                           "context_value_not_in_history")
        self.assertInvalid(self.follow_up([self.PREVIOUS], {"ticker": 0, "metric": 0,
                                                            "period": 0}),
                           "context_ref_unused")

    def test_ambiguous_antecedents_need_clarification(self):
        question = "Còn năm 2023 của công ty đó?"
        both = [QATurn(question="Doanh thu VJC và HPG năm 2022?")]
        self.assertClarify(self.follow_up(both, {"ticker": 0, "metric": 0}, question=question),
                           "ambiguous_context")
        payload = plan(question, sq("q1", ticker="VJC", year=2023, metric="revenue",
                                    refs={"ticker": 0, "metric": 0}),
                       sq("q2", ticker="HPG", year=2023, metric="revenue",
                          refs={"ticker": 0, "metric": 0}))
        self.assertValid(run(payload, question, both))
        later = [self.PREVIOUS, QATurn(question="Doanh thu HPG năm 2022?")]
        self.assertClarify(self.follow_up(later, {"ticker": 0, "metric": 0}), "ambiguous_context")
        self.assertValid(self.follow_up(later, {"ticker": 1, "metric": 1}, ticker="HPG"))

    def test_refs_use_original_indexes_within_retained_window(self):
        history = ([QATurn(question=f"Doanh thu VJC năm {2015 + i}?") for i in range(2)]
                   + [QATurn(question=f"Tổng tài sản HPG năm {2017 + i}?") for i in range(4)]
                   + [QATurn(question="Doanh thu VJC năm 2022?")])
        self.assertValid(self.follow_up(history, {"ticker": 6, "metric": 6}))
        for refs in ({"ticker": 0, "metric": 6}, {"ticker": 9, "metric": 6}):
            with self.subTest(refs=refs):
                self.assertInvalid(self.follow_up(history, refs), "context_ref_out_of_window")
        self.assertInvalid(self.follow_up(history, {"ticker": 4, "metric": 6}),
                           "context_value_not_in_history")
        self.assertInvalid(self.follow_up([], {"ticker": 0, "metric": 0}),
                           "context_ref_out_of_window")

    def test_inheritance_requires_follow_up_marker(self):
        question = "Doanh thu năm 2023 là bao nhiêu?"
        self.assertClarify(self.follow_up([self.PREVIOUS], {"ticker": 0}, question=question),
                           "unsupported_context")

    def test_parent_share_history_does_not_imply_separate_scope(self):
        history = [QATurn(question="LNST thuộc cổ đông công ty mẹ VJC năm 2022?")]
        fields = dict(metric="profit_after_tax_parent_share")
        refs = {"ticker": 0, "metric": 0}
        self.assertValid(self.follow_up(history, refs, **fields))
        self.assertInvalid(self.follow_up(history, dict(refs, scope=0), scope="separate", **fields),
                           "context_value_not_in_history")
        self.assertInvalid(self.follow_up(history, refs, scope="separate", **fields),
                           "invented_scope")


class ReviewFixRoundOneTests(ValidationTestCase):
    """Regressions for review findings on the first Task 3 implementation."""
    HISTORY = [QATurn(question="Doanh thu VJC năm 2022 là bao nhiêu?")]

    def test_nested_models_in_mapping_are_revalidated_and_not_aliased(self):
        bad_types = Subquery.model_validate(sq(ticker="VJC", year=2023, metric="revenue"))
        bad_types.document_types = ["everything"]
        bad_refs = Subquery.model_validate(sq(ticker="VJC", year=2023, metric="revenue"))
        bad_refs.context_refs["bogus"] = 0
        for nested in (bad_types, bad_refs):
            with self.subTest(nested=nested.document_types):
                payload = plan(Q_VJC)
                payload["subqueries"] = [nested]
                self.assertInvalid(run(payload, Q_VJC), "schema_invalid")
        good = Subquery.model_validate(sq(ticker="VJC", year=2023, metric="revenue"))
        payload = plan(Q_VJC)
        payload["subqueries"] = [good]
        result = run(payload, Q_VJC)
        self.assertValid(result)
        self.assertIsNot(result.plan.subqueries[0], good)
        self.assertIsNot(result.plan.subqueries[0].document_types, good.document_types)
        self.assertIsNot(result.plan.subqueries, payload["subqueries"])

    def test_reworded_duplicate_need_is_invalid(self):
        question = "Tổng doanh thu VJC năm 2022 và 2023"
        needs = [sq("q1", ticker="VJC", year=2022, metric="revenue"),
                 sq("q2", ticker="VJC", year=2023, metric="revenue"),
                 sq("q3", ticker="VJC", year=2022, metric="revenue",
                    search="Doanh thu năm 2022 của VJC")]
        relation = [{"kind": "aggregate", "subquery_ids": ["q1", "q2", "q3"],
                     "description": None}]
        result = run(plan(question, *needs, operation="aggregate", relationships=relation),
                     question)
        self.assertInvalid(result, "duplicate_subquery")

    def test_metric_inheritance_needs_metric_reference_or_bare_follow_up(self):
        question = "Còn vốn chủ sở hữu năm 2023?"
        inherited = plan(question, sq(ticker="VJC", year=2023, metric="revenue",
                                      refs={"ticker": 0, "metric": 0}))
        self.assertClarify(run(inherited, question, self.HISTORY),
                           "unverified_metric_inheritance")
        own_metric = plan(question, sq(ticker="VJC", year=2023, metric_text="vốn chủ sở hữu",
                                       refs={"ticker": 0}))
        self.assertValid(run(own_metric, question, self.HISTORY))

    def test_follow_up_resolving_nothing_needs_clarification(self):
        question = "Còn năm 2023?"
        result = run(plan(question, sq(year=2023)), question, self.HISTORY)
        self.assertClarify(result, "unresolved_follow_up")

    def test_single_year_group_inside_entities_is_ambiguous(self):
        cases = {"Doanh thu VJC năm 2022 và HPG": [("VJC", 2022), ("HPG", 2022)],
                 "Doanh thu VJC 2022 và 2023 so với HPG": [("VJC", 2022), ("VJC", 2023),
                                                          ("HPG", 2022), ("HPG", 2023)]}
        for question, pairs in cases.items():
            needs = [sq(f"q{i}", ticker=t, year=y, metric="revenue")
                     for i, (t, y) in enumerate(pairs)]
            with self.subTest(question=question):
                self.assertClarify(run(plan(question, *needs), question), "ambiguous_binding")
        leading = "Doanh thu năm 2022 và 2023 của VJC và HPG"
        needs = [sq(f"q{i}", ticker=t, year=y, metric="revenue")
                 for i, (t, y) in enumerate(cases["Doanh thu VJC 2022 và 2023 so với HPG"])]
        self.assertValid(run(plan(leading, *needs), leading))

    def test_relationship_kind_must_match_operation(self):
        question = "So sánh doanh thu VJC 2022 và 2023"
        needs = (sq("q1", ticker="VJC", year=2022, metric="revenue"),
                 sq("q2", ticker="VJC", year=2023, metric="revenue"))
        relation = [{"kind": "aggregate", "subquery_ids": ["q1", "q2"], "description": None}]
        self.assertInvalid(run(plan(question, *needs, operation="compare",
                                    relationships=relation), question), "operation_mismatch")

    def test_line_items_are_not_operation_cues_and_aggregate_needs_operands(self):
        cases = [("Tổng doanh thu của VJC năm 2023", dict(ticker="VJC", metric="revenue")),
                 ("Tổng lợi nhuận kế toán trước thuế của VJC năm 2023",
                  dict(ticker="VJC", metric_text="tổng lợi nhuận kế toán trước thuế")),
                 ("Tỷ lệ nợ xấu của VCB năm 2023", dict(ticker="VCB", metric_text="tỷ lệ nợ xấu")),
                 ("Tỷ lệ an toàn vốn của VCB năm 2023",
                  dict(ticker="VCB", metric_text="tỷ lệ an toàn vốn"))]
        for question, fields in cases:
            with self.subTest(question=question):
                self.assertValid(run(plan(question, sq(year=2023, **fields)), question))
        question = cases[0][0]
        self.assertInvalid(run(plan(question, sq(ticker="VJC", year=2023, metric="revenue"),
                                    operation="aggregate"), question), "insufficient_operands")

    def test_amount_word_ty_is_billion_vnd_unit(self):
        question = "Doanh thu VJC năm 2023 là bao nhiêu tỷ?"
        fields = dict(ticker="VJC", year=2023, metric="revenue")
        self.assertValid(run(plan(question, sq(unit="tỷ đồng", **fields)), question))
        self.assertInvalid(run(plan(question, sq(unit="triệu đồng", **fields)), question),
                           "invented_unit")
        self.assertInvalid(run(plan(question, sq(**fields)), question), "missing_unit")


class FinalReviewFixTests(ValidationTestCase):
    """Regressions for the final whole-branch review."""
    HISTORY = [QATurn(question="Doanh thu VJC năm 2022 là bao nhiêu?")]

    def test_follow_up_dropping_history_ticker_or_year_needs_clarification(self):
        cases = [("Còn năm 2023 thì sao?", dict(year=2023, metric="revenue", refs={"metric": 0})),
                 ("Còn lợi nhuận sau thuế năm 2023?", dict(year=2023, metric="profit_after_tax")),
                 ("Còn HPG thì sao?", dict(ticker="HPG", metric="revenue", refs={"metric": 0}))]
        for question, fields in cases:
            with self.subTest(question=question):
                self.assertClarify(run(plan(question, sq(**fields)), question, self.HISTORY),
                                   "unresolved_follow_up")

    def test_follow_up_with_inherited_ticker_or_year_stays_valid(self):
        cases = [("Còn năm 2023 thì sao?", dict(ticker="VJC", year=2023, metric="revenue",
                                                refs={"ticker": 0, "metric": 0})),
                 ("Còn lợi nhuận sau thuế năm 2023?",
                  dict(ticker="VJC", year=2023, metric="profit_after_tax", refs={"ticker": 0})),
                 ("Còn HPG thì sao?", dict(ticker="HPG", year=2022, metric="revenue",
                                           refs={"year": 0, "metric": 0}))]
        for question, fields in cases:
            with self.subTest(question=question):
                self.assertValid(run(plan(question, sq(**fields)), question, self.HISTORY))

    def test_lowercase_ticker_mention_without_subquery_ticker_needs_clarification(self):
        question = "doanh thu vjc năm 2023"
        self.assertClarify(run(plan(question, sq(year=2023, metric="revenue")), question),
                           "ambiguous_entity_mention")
        self.assertValid(run(plan(question, sq(ticker="VJC", year=2023, metric="revenue")),
                             question))

    def test_document_types_need_question_cues(self):
        def with_types(question, types):
            subquery = dict(sq(ticker="VJC", year=2023, metric="revenue"), document_types=types)
            return run(plan(question, subquery), question)

        for types in (["letters"], ["explanations"], ["financial_statements", "letters"]):
            with self.subTest(types=types):
                result = with_types(Q_VJC, types)
                self.assertInvalid(result, "invented_document_type")
                self.assertIn("q1", {i.subquery_id for i in result.issues
                                     if i.code == "invented_document_type"})
        self.assertValid(with_types(Q_VJC, ["financial_statements"]))
        notes = "Thuyết minh doanh thu VJC năm 2023"
        self.assertValid(with_types(notes, ["explanations"]))
        self.assertInvalid(with_types(notes, []), "missing_document_type")
        letters = "Thư giải trình doanh thu VJC năm 2023"
        self.assertValid(with_types(letters, ["letters"]))
        self.assertInvalid(with_types(letters, ["explanations"]),
                           "invented_document_type", "missing_document_type")


if __name__ == "__main__":
    unittest.main()

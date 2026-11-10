import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from config.settings import ParaphraseSettings, load_paraphrase_settings
from src.preprocess.cleaner import fold_text
from src.query_paraphrase.entities import EntityRegistry
from src.query_paraphrase.metric_catalog import find_metrics, normalize_metric
from src.query_paraphrase.schemas import (
    ParaphraseResult, QATurn, QueryPlan, QueryRelationship, SemanticReview,
    Subquery, ValidationIssue,
)


ROOT = Path(__file__).resolve().parents[1]


def subquery_payload():
    return {
        "id": "q1", "question": "Doanh thu VJC?", "search_query": "Doanh thu VJC",
        "ticker": None, "company_name": None, "year": None, "period": None,
        "scope": None, "metric": None, "metric_text": None, "unit": None,
    }


def plan_payload():
    return {"original_question": "Doanh thu VJC?", "operation": "lookup",
            "operation_description": None, "subqueries": [subquery_payload()]}


class ContractTests(unittest.TestCase):
    def test_nullable_metadata_must_be_explicit(self):
        for field in ("ticker", "company_name", "year", "period", "scope", "metric",
                      "metric_text", "unit"):
            payload = subquery_payload()
            del payload[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Subquery.model_validate(payload)

    def test_other_nullable_contract_fields_are_required(self):
        cases = [(QueryPlan, plan_payload(), "operation_description"),
                 (QueryRelationship, {"kind": "compare", "subquery_ids": ["q1"],
                                      "description": None}, "description"),
                 (ValidationIssue, {"code": "invalid", "message": "Invalid year",
                                    "subquery_id": None}, "subquery_id"),
                 (ParaphraseResult, {"status": "invalid", "plan": None}, "plan")]
        for model, payload, field in cases:
            del payload[field]
            with self.subTest(model=model.__name__), self.assertRaises(ValidationError):
                model.model_validate(payload)

    def test_valid_null_metadata_and_empty_optional_collections(self):
        plan = QueryPlan.model_validate(plan_payload())
        self.assertIsNone(plan.subqueries[0].scope)
        self.assertEqual(plan.subqueries[0].document_types, [])
        self.assertEqual(plan.subqueries[0].context_refs, {})
        self.assertEqual(plan.relationships, [])
        self.assertIsNone(QATurn(question="Còn năm 2023?").answer)
        self.assertEqual(SemanticReview(status="unknown", reason="No reviewer evidence").status,
                         "unknown")

    def test_rejects_empty_extra_and_non_string_fields(self):
        for field, value in (("question", "  "), ("search_query", ""), ("id", 1),
                             ("ticker", " "), ("metric_text", ""), ("surprise", True)):
            payload = subquery_payload()
            payload[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                Subquery.model_validate(payload)
        with self.assertRaises(ValidationError):
            QATurn(question=" ")
        with self.assertRaises(ValidationError):
            SemanticReview(status="passed", reason=" ")

    def test_year_is_strict_and_bounded(self):
        for year in (True, "2023", 2023.0, 1899, 2200):
            payload = subquery_payload()
            payload["year"] = year
            with self.subTest(year=year), self.assertRaises(ValidationError):
                Subquery.model_validate(payload)
        for year in (1900, 2199):
            payload = subquery_payload()
            payload["year"] = year
            self.assertEqual(Subquery.model_validate(payload).year, year)

    def test_collections_and_enums_reject_invalid_metadata(self):
        for field, value in (("scope", "parent"), ("document_types", ["notes"]),
                             ("context_refs", {"answer": 0}),
                             ("context_refs", {"year": True}),
                             ("context_refs", {"year": -1})):
            payload = subquery_payload()
            payload[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                Subquery.model_validate(payload)

    def test_empty_subqueries_duplicate_ids_and_dangling_relationships_fail(self):
        payload = plan_payload()
        for subqueries in ([], [subquery_payload(), subquery_payload()]):
            payload["subqueries"] = subqueries
            with self.subTest(subqueries=subqueries), self.assertRaises(ValidationError):
                QueryPlan.model_validate(payload)
        payload = plan_payload()
        payload["relationships"] = [{"kind": "compare", "subquery_ids": ["missing"],
                                     "description": None}]
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(payload)

    def test_relationship_operand_order_is_preserved(self):
        payload = plan_payload()
        second = copy.deepcopy(payload["subqueries"][0])
        second["id"] = "q2"
        payload["subqueries"].append(second)
        payload["operation"] = "calculate"
        payload["relationships"] = [{"kind": "calculate", "subquery_ids": ["q2", "q1"],
                                     "description": "q2 trừ q1"}]
        self.assertEqual(QueryPlan.model_validate(payload).relationships[0].subquery_ids,
                         ["q2", "q1"])


class SettingsTests(unittest.TestCase):
    def test_default_policy_and_call_time_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = load_paraphrase_settings()
            self.assertEqual((settings.max_subqueries, settings.max_history_turns,
                              settings.max_input_chars, settings.default_report_scope,
                              settings.notes_fallback), (8, 5, 8000, "consolidated", True))
            os.environ["FINAGENT_PARAPHRASE_MAX_SUBQUERIES"] = "3"
            os.environ["FINAGENT_PARAPHRASE_MAX_HISTORY_TURNS"] = "0"
            os.environ["FINAGENT_PARAPHRASE_MAX_INPUT_CHARS"] = "100"
            os.environ["FINAGENT_PARAPHRASE_DEFAULT_REPORT_SCOPE"] = "separate"
            os.environ["FINAGENT_PARAPHRASE_NOTES_FALLBACK"] = "false"
            changed = load_paraphrase_settings(max_subqueries=4)
            self.assertEqual((changed.max_subqueries, changed.max_history_turns,
                              changed.max_input_chars, changed.default_report_scope,
                              changed.notes_fallback), (4, 0, 100, "separate", False))

    def test_malformed_environment_is_rejected(self):
        for name, value in (("MAX_SUBQUERIES", "bad"), ("MAX_SUBQUERIES", "0"),
                            ("MAX_HISTORY_TURNS", "-1"), ("MAX_INPUT_CHARS", "0"),
                            ("NOTES_FALLBACK", "maybe"), ("DEFAULT_REPORT_SCOPE", "parent")):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {}, clear=True):
                os.environ["FINAGENT_PARAPHRASE_" + name] = value
                with self.assertRaises(ValueError):
                    load_paraphrase_settings()

    def test_constructor_overrides_are_strict(self):
        for override in ({"max_subqueries": True}, {"max_history_turns": "5"},
                         {"notes_fallback": "false"}, {"unknown": 1}):
            with self.subTest(override=override), self.assertRaises(ValidationError):
                ParaphraseSettings(**override)


class MetricCatalogTests(unittest.TestCase):
    def test_metric_aliases_preserve_accounting_distinctions(self):
        cases = {"LNST": "profit_after_tax", "lntt": "profit_before_tax",
                 "doanh thu": "revenue", "DOANH THU THUẦN": "net_revenue",
                 "lợi nhuận sau thuế hợp nhất": "consolidated_profit_after_tax",
                 "LNST thuộc cổ đông công ty mẹ": "profit_after_tax_parent_share"}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize_metric(text), expected)

    def test_longest_metric_suppresses_contained_generic_alias(self):
        self.assertEqual(find_metrics("Doanh thu thuần và lợi nhuận sau thuế hợp nhất VJC"),
                         {"net_revenue", "consolidated_profit_after_tax"})
        self.assertEqual(find_metrics("LNST thuộc cổ đông công ty mẹ và LNTT"),
                         {"profit_after_tax_parent_share", "profit_before_tax"})
        self.assertEqual(find_metrics("doanh thu thuần và doanh thu"),
                         {"net_revenue", "revenue"})

    def test_unknown_and_multiple_metrics_do_not_normalize_to_one(self):
        self.assertIsNone(normalize_metric("vốn lưu động ròng"))
        self.assertIsNone(normalize_metric("doanh thu và LNST"))
        self.assertEqual(find_metrics("ALNST không phải tên chỉ tiêu"), set())


class EntityRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = EntityRegistry.from_csv(ROOT / "data" / "code_stock.csv")

    def test_actual_ticker_and_full_company_name(self):
        self.assertEqual(self.registry.resolve(" vjc "), {"VJC"})
        self.assertEqual(self.registry.resolve("CTCP Hàng không Vietjet"), {"VJC"})
        self.assertEqual(self.registry.company_name("VJC"), "CTCP Hàng không Vietjet")
        self.assertIsNone(self.registry.company_name("ZZZ"))
        self.assertEqual(self.registry.find_mentions("Doanh thu vjc năm 2023"), {"VJC"})

    def test_unknown_ticker_is_inspectable_and_finance_acronyms_are_excluded(self):
        self.assertEqual(self.registry.resolve("ZZZ"), set())
        self.assertEqual(self.registry.find_mentions("LNST LNTT EBITDA BCTC ZZZ"), set())
        self.assertEqual(self.registry.unknown_ticker_candidates("LNST LNTT EBITDA BCTC ZZZ"),
                         {"ZZZ"})

    def test_finance_acronyms_and_period_tokens_are_not_unknown_tickers(self):
        text = ("EBIT ROE ROA ROIC NIM NPL CASA GTGT KQKD BCTC LCTT TNDN VCSH EPS "
                "FY23 FY2023 Q1 H1 Q123 H224 ZZZ")
        self.assertEqual(self.registry.unknown_ticker_candidates(text), {"ZZZ"})
        self.assertEqual(self.registry.find_mentions(text), set())

    def test_mention_spans_report_folded_offsets_in_order(self):
        text = "Doanh thu VJC 2022 và CTCP Tập đoàn Hòa Phát 2023"
        spans = self.registry.mention_spans(text)
        self.assertEqual([tickers for _, _, tickers in spans],
                         [frozenset({"VJC"}), frozenset({"HPG"})])
        folded = fold_text(text)
        self.assertEqual([folded[start:end] for start, end, _ in spans],
                         ["vjc", "ctcp tap doan hoa phat"])

    def test_alias_collisions_remain_ambiguous(self):
        registry = EntityRegistry.from_csv(ROOT / "data" / "code_stock.csv",
                                           aliases={"hãng bay": ["VJC", "ACV"],
                                                    "vietjet": "VJC", "VJC": "ACV"})
        self.assertEqual(registry.resolve("HÃNG BAY"), {"VJC", "ACV"})
        self.assertEqual(registry.find_mentions("Doanh thu hãng bay"), {"VJC", "ACV"})
        self.assertEqual(registry.resolve("vjc"), {"VJC", "ACV"})
        self.assertEqual(registry.resolve("vietjet"), {"VJC"})

    def test_invalid_alias_target_is_rejected(self):
        for aliases in ({"airline": "ZZZ"}, {"airline": []}, {" ": "VJC"}):
            with self.subTest(aliases=aliases), self.assertRaises(ValueError):
                EntityRegistry.from_csv(ROOT / "data" / "code_stock.csv", aliases=aliases)

    def test_duplicate_tickers_and_invalid_headers_are_rejected(self):
        for content in ("Mã CK,Tên công ty\nVJC,First\nvjc,Second\n",
                        "ticker,name\nVJC,Vietjet\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "companies.csv"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    EntityRegistry.from_csv(path)

    def test_full_company_name_does_not_imply_embedded_fpt_ticker(self):
        self.assertEqual(self.registry.find_mentions("LNST CTCP Viễn thông FPT năm 2023"),
                         {"FOX"})
        self.assertEqual(self.registry.find_mentions("CTCP Viễn thông FPT và FPT"),
                         {"FOX", "FPT"})
        self.assertEqual(self.registry.find_mentions("CTCP Chứng khoán FPT"), {"FTS"})


if __name__ == "__main__":
    unittest.main()

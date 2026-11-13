import unittest
from decimal import Decimal as D

from src.agent.gather import CappedLocator, Gatherer
from src.agent.types import DataNeed
from src.calculation.facts import FactKey
from src.extraction.extractor import LeafExtractor, LocatorResult
from tests.test_extraction_extractor import (
    StubLocator,
    default_store,
    doc,
    is_rows,
    is_table,
    store_of,
    templates,
    tree,
)


class CountingStore:
    """Wraps a store and counts the sessions opened, one per ``extract`` call."""

    def __init__(self, inner):
        self._inner = inner
        self.sessions = 0

    def session(self):
        self.sessions += 1
        return self._inner.session()


def gatherer(store=None, locator=None, max_calls=6, default_scope="consolidated"):
    store = store if store is not None else default_store()
    factory = lambda loc: LeafExtractor(store, tree, templates, fallback=loc)
    return Gatherer(tree, templates, factory, locator, max_calls, default_scope)


class GatherTest(unittest.TestCase):
    def test_found_leaf_enters_factset(self):
        g = gatherer()
        logs = g.gather([DataNeed("interest_expense", "VJC", 2023)])
        fact = g.facts.get(FactKey("interest_expense", "VJC", 2023, "consolidated", "flow"))
        self.assertEqual(fact.value.value, D("-1950254195864"))
        (log,) = logs
        self.assertEqual((log.leaf_id, log.outcome, log.tool), ("interest_expense", "Found", "extractor"))
        self.assertEqual(log.value, D("-1950254195864"))
        self.assertEqual(log.doc_id, "VJC23C")
        self.assertEqual(g.log, logs)

    def test_derived_need_requests_all_alternative_leaves(self):
        g = gatherer()
        g.gather([DataNeed("gross_margin", "VJC", 2023)])
        ids = {log.leaf_id for log in g.log}
        self.assertTrue({"net_revenue", "cost_of_goods_sold", "gross_profit_reported"} <= ids)

    def test_missing_scope_report_is_logged(self):
        g = gatherer()
        need = DataNeed("interest_expense", "VJC", 2023, scope="separate")
        (log,) = g.gather([need])
        self.assertEqual((log.outcome, log.scope), ("ReportMissing", "separate"))
        self.assertIsNone(log.value)
        self.assertEqual(g.failures(need), (log,))

    def test_no_leaf_requested_twice(self):
        store = CountingStore(default_store())
        g = gatherer(store)
        need = DataNeed("interest_expense", "VJC", 2023)
        self.assertEqual(len(g.gather([need])), 1)
        self.assertTrue(g.requested(need))
        self.assertEqual(g.gather([need]), ())
        self.assertEqual(store.sessions, 1)
        self.assertEqual(len(g.log), 1)
        self.assertEqual(g.failures(need), ())

    def test_avg_balance_requests_prior_year(self):
        g = gatherer()
        g.gather([DataNeed("interest_expense", "VJC", 2023, helper="avg_balance")])
        self.assertEqual({(l.leaf_id, l.year) for l in g.log},
                         {("interest_expense", 2023), ("interest_expense", 2022)})

    def test_unknown_ticker_logged_with_node_id(self):
        g = gatherer()
        need = DataNeed("gross_margin", "XXX", 2023)
        (log,) = g.gather([need])
        self.assertEqual((log.leaf_id, log.outcome), ("gross_margin", "UnknownTicker"))
        self.assertEqual(g.failures(need), (log,))

    def test_unsupported_template_logged(self):
        from src.extraction.templates import TickerTemplates

        factory = lambda loc: LeafExtractor(default_store(), tree, TickerTemplates({"ZZZ": "insurance"}), fallback=loc)
        g = Gatherer(tree, TickerTemplates({"ZZZ": "insurance"}), factory, None, 6, "consolidated")
        (log,) = g.gather([DataNeed("net_revenue", "ZZZ", 2023)])
        self.assertEqual(log.outcome, "UnsupportedTemplate")

    def test_locator_cap_marks_skipped(self):
        store = store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, interest=False))]))
        stub = StubLocator(lambda views: LocatorResult(None, None, ()))
        absent = [l.id for l in tree.leaves("corporate") if l.id not in ("net_revenue", "gross_revenue", "interest_expense")][:7]
        g = gatherer(store, stub, max_calls=6)
        g.gather([DataNeed(i, "VJC", 2023) for i in absent])
        self.assertEqual(len(stub.calls), 6)
        self.assertEqual([l.outcome for l in g.log].count("NotFound"), 7)
        self.assertEqual([l.tool for l in g.log].count("locator_skipped"), 1)

    def test_locator_found_is_tagged_locator(self):
        from tests.test_extraction_extractor import read_label

        store = store_of((doc("VJC23C", 2023), [is_table("VJC23C", is_rows(2023, interest=False))]))
        stub = StubLocator(read_label("- Trong đó: Chi phí lãi vay", col=3))
        g = gatherer(store, stub)
        (log,) = g.gather([DataNeed("interest_expense", "VJC", 2023)])
        self.assertEqual((log.outcome, log.tool), ("Found", "locator"))


class CappedLocatorTest(unittest.TestCase):
    def test_past_cap_skips_without_calling_inner(self):
        stub = StubLocator(lambda views: LocatorResult(None, "x", ()))
        capped = CappedLocator(stub, 1)
        leaf = tree.leaf("net_revenue", "corporate")
        req = next(iter(tree.required_leaves("net_revenue", "corporate", 2023)[0]))
        capped.locate(leaf, req, "primary", ())
        result = capped.locate(leaf, req, "primary", ())
        self.assertEqual((result.read, result.reason, result.warnings), (None, "locator_skipped", ()))
        self.assertEqual((capped.calls, len(stub.calls)), (1, 1))
        self.assertEqual(capped.skipped, {("net_revenue", 2023)})


if __name__ == "__main__":
    unittest.main()

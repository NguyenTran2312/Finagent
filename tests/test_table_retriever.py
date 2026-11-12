import unittest

from langchain_core.documents import Document

from config.settings import RRF_CONSTANT
from src.calculation.metric_tree import MetricTree
from src.extraction.retrieval import RankedTable, TableRetrievalError
from src.retrieval.bm25 import build_bm25
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.table_retriever import HybridTableRetriever, table_query

tree = MetricTree.load()
leaf_is = tree.leaf("gross_revenue", "corporate")
leaf_bs = tree.leaf("current_assets", "corporate")
leaf_cf = tree.leaf("cfo", "corporate")
leaf_note = tree.leaf("accrued_interest_payable", "corporate")


def doc(chunk_id, ticker, year, scope, text):
    table_id, doc_id = chunk_id.split(":")[1], chunk_id.split(":")[0]
    return Document(
        id=chunk_id,
        page_content=text,
        metadata={"chunk_id": chunk_id, "doc_id": doc_id, "table_id": table_id,
                  "ticker": ticker, "report_year": year, "scope": scope},
    )


TEXT = "doanh thu bán hàng và cung cấp dịch vụ báo cáo kết quả hoạt động kinh doanh"
DOCS = [
    doc("VJC23C:1", "VJC", 2023, "consolidated", TEXT + " doanh thu"),
    doc("VJC23C:2", "VJC", 2023, "consolidated", TEXT),
    doc("VJC23S:1", "VJC", 2023, "separate", TEXT),
    doc("VJC23P:1", "VJC", 2023, "parent", TEXT + " công ty mẹ"),
    doc("VJC22C:1", "VJC", 2022, "consolidated", TEXT),
    doc("ACB23C:1", "ACB", 2023, "consolidated", TEXT),
]
bm25 = build_bm25(DOCS, k=5)


class FakeDenseStore:
    def __init__(self, docs=(), error=None):
        self.docs, self.error, self.calls = list(docs), error, []

    def similarity_search(self, query, k, filter=None):
        self.calls.append((query, k, filter))
        if self.error:
            raise self.error
        return self.docs


class HybridTableRetrieverTests(unittest.TestCase):
    def test_query_appends_statement_name(self):
        self.assertEqual(table_query(leaf_is), f"{leaf_is.label_vi} báo cáo kết quả hoạt động kinh doanh")
        self.assertEqual(table_query(leaf_bs), f"{leaf_bs.label_vi} bảng cân đối kế toán")
        self.assertEqual(table_query(leaf_cf), f"{leaf_cf.label_vi} báo cáo lưu chuyển tiền tệ")
        self.assertEqual(table_query(leaf_note), leaf_note.label_vi)

    def test_bm25_mode_filters_ranks_and_truncates(self):
        r = HybridTableRetriever(bm25, None, mode="bm25", k=2, depth=2)
        out = r.rank("vjc", "consolidated", 2023, leaf_is)
        self.assertEqual([t.rank for t in out], [1, 2])
        self.assertTrue(all(isinstance(t, RankedTable) for t in out))
        self.assertTrue(all(t.chunk_id.startswith("VJC23C:") for t in out))

    def test_separate_scope_also_accepts_parent(self):
        store = FakeDenseStore([])
        HybridTableRetriever(bm25, store, k=2, depth=3).rank("VJC", "separate", 2023, leaf_is)
        query, k, where = store.calls[0]
        self.assertEqual(query, table_query(leaf_is))
        self.assertEqual(k, 3)
        self.assertIn({"scope": {"$in": ["separate", "parent"]}}, where["$and"])
        self.assertIn({"ticker": {"$in": ["VJC"]}}, where["$and"])
        self.assertIn({"report_year": {"$in": [2023]}}, where["$and"])
        out = HybridTableRetriever(bm25, None, mode="bm25", k=5, depth=5).rank("VJC", "separate", 2023, leaf_is)
        self.assertEqual({t.chunk_id for t in out}, {"VJC23S:1", "VJC23P:1"})

    def test_hybrid_fuses_both_branches(self):
        a, b = DOCS[0], DOCS[1]
        store = FakeDenseStore([b, a])
        r = HybridTableRetriever(bm25, store, k=2, depth=5)
        out = r.rank("VJC", "consolidated", 2023, leaf_is)
        lexical = HybridTableRetriever(bm25, None, mode="bm25", k=5, depth=5).rank(
            "VJC", "consolidated", 2023, leaf_is)
        lexical_docs = [d for t in lexical for d in DOCS if d.metadata["chunk_id"] == t.chunk_id]
        expected = [d.metadata["chunk_id"] for d in rrf_fuse(lexical_docs, [b, a], k=2, c=RRF_CONSTANT)]
        self.assertEqual([t.chunk_id for t in out], expected)
        self.assertEqual([t.rank for t in out], [1, 2])

    def test_backend_error_is_wrapped(self):
        store = FakeDenseStore(error=RuntimeError("chroma down"))
        with self.assertRaises(TableRetrievalError) as caught:
            HybridTableRetriever(bm25, store).rank("VJC", "consolidated", 2023, leaf_is)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception), "hybrid retrieval failed: RuntimeError")

    def test_bad_scope_is_input_error(self):
        store = FakeDenseStore([])
        with self.assertRaises(ValueError) as caught:
            HybridTableRetriever(bm25, store).rank("VJC", "nonsense", 2023, leaf_is)
        self.assertNotIsInstance(caught.exception, TableRetrievalError)
        self.assertEqual(store.calls, [])

    def test_constructor_validates(self):
        for kwargs in ({"mode": "dense"}, {"k": 0}, {"k": 10, "depth": 5}, {"mode": "hybrid"},
                       {"mode": "bm25", "k": "3"}, {"mode": "bm25", "depth": 20.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                HybridTableRetriever(bm25, None, **kwargs)

    def test_defaults(self):
        r = HybridTableRetriever(bm25, FakeDenseStore())
        self.assertEqual((r.mode, r.k, r.depth), ("hybrid", 10, 20))


if __name__ == "__main__":
    unittest.main()

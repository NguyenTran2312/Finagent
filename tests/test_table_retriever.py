import argparse
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document

from config.settings import RRF_CONSTANT
from src.calculation.metric_tree import MetricTree
from src.extraction.retrieval import RankedTable, TableRetrievalError
from src.indexing.bm25_store import save_bm25, sqlite_source_identity
from src.main.refine_questions import filter_document_indices, rank_indices
from src.retrieval import table_retriever
from src.retrieval.bm25 import build_bm25
from src.retrieval.filters import normalize_metadata_filters
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.table_retriever import (
    HybridTableRetriever,
    add_retriever_arguments,
    open_table_retriever,
    retriever_from_args,
    table_query,
)

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
    doc("VJC23C:3", "VJC", 2023, "consolidated", TEXT + " thuyết minh"),
    doc("VJC23S:1", "VJC", 2023, "separate", TEXT),
    doc("VJC23P:1", "VJC", 2023, "parent", TEXT + " công ty mẹ"),
    doc("VJC22C:1", "VJC", 2022, "consolidated", TEXT),
    doc("ACB23C:1", "ACB", 2023, "consolidated", TEXT),
]
bm25 = build_bm25(DOCS, k=5)


def unit(*values):
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


# Cosine order against QUERY_VECTOR: VJC23C:2, VJC23C:1, VJC23C:3 (others below).
VECTORS = {
    "VJC23C:1": unit(0.8, 0.6), "VJC23C:2": unit(1.0, 0.0), "VJC23C:3": unit(0.0, 1.0),
    "VJC23S:1": unit(0.6, 0.8), "VJC23P:1": unit(0.9, 0.1),
    "VJC22C:1": unit(1.0, 0.1), "ACB23C:1": unit(1.0, 0.2),
}
QUERY_VECTOR = unit(1.0, 0.0)


class FakeEmbeddings:
    def __init__(self, store):
        self.store = store

    def embed_query(self, text):
        self.store.embedded.append(text)
        if self.store.embed_error:
            raise self.store.embed_error
        return list(self.store.query_vectors.get(text, QUERY_VECTOR))


class FakeDenseStore:
    """Chroma-like store: ``get`` returns rows in reverse order of the ids asked for."""

    def __init__(self, docs=DOCS, error=None, *, vectors=VECTORS, query_vectors=None,
                 metadatas=None, embed_error=None):
        self.metadatas = {d.metadata["chunk_id"]: dict(d.metadata) for d in docs}
        self.metadatas.update(metadatas or {})
        self.vectors, self.query_vectors = dict(vectors), dict(query_vectors or {})
        self.error, self.embed_error = error, embed_error
        self.calls, self.embedded, self.similarity_calls = [], [], 0
        self.embeddings = FakeEmbeddings(self)

    @property
    def get_calls(self):
        return len(self.calls)

    @property
    def embed_calls(self):
        return len(self.embedded)

    def get(self, ids=None, include=None):
        self.calls.append((list(ids), list(include)))
        if self.error:
            raise self.error
        found = [i for i in ids if i in self.metadatas][::-1]
        return {"ids": found,
                "embeddings": np.array([self.vectors[i] for i in found]),
                "metadatas": [self.metadatas[i] for i in found]}

    def similarity_search(self, query, k, filter=None):
        self.similarity_calls += 1
        raise AssertionError("similarity_search must not be called")


class HybridTableRetrieverTests(unittest.TestCase):
    def test_query_appends_statement_name(self):
        self.assertEqual(table_query(leaf_is), f"{leaf_is.label_vi} báo cáo kết quả hoạt động kinh doanh")
        self.assertEqual(table_query(leaf_bs), f"{leaf_bs.label_vi} bảng cân đối kế toán")
        self.assertEqual(table_query(leaf_cf), f"{leaf_cf.label_vi} báo cáo lưu chuyển tiền tệ")
        self.assertEqual(table_query(leaf_note), leaf_note.label_vi)

    def test_bm25_mode_filters_ranks_and_truncates(self):
        r = HybridTableRetriever(bm25, None, mode="bm25", k=2, depth=2)
        out = r.rank("vjc", "consolidated", 2023, leaf_is)
        self.assertEqual(len(out), 2)  # three VJC/2023/consolidated candidates exist
        self.assertEqual([t.rank for t in out], [1, 2])
        self.assertTrue(all(isinstance(t, RankedTable) for t in out))
        self.assertTrue(all(t.chunk_id.startswith("VJC23C:") for t in out))

    def test_separate_scope_also_accepts_parent(self):
        store = FakeDenseStore()
        out = HybridTableRetriever(bm25, store, k=2, depth=3).rank("VJC", "separate", 2023, leaf_is)
        ids, include = store.calls[0]
        self.assertEqual(store.embedded, [table_query(leaf_is)])
        self.assertEqual(sorted(ids), ["VJC23P:1", "VJC23S:1"])  # ticker, year and scope filtered
        self.assertEqual(include, ["embeddings", "metadatas"])
        self.assertEqual({t.chunk_id for t in out}, {"VJC23S:1", "VJC23P:1"})
        out = HybridTableRetriever(bm25, None, mode="bm25", k=5, depth=5).rank("VJC", "separate", 2023, leaf_is)
        self.assertEqual({t.chunk_id for t in out}, {"VJC23S:1", "VJC23P:1"})

    def test_hybrid_fuses_both_branches(self):
        a, b, c = DOCS[0], DOCS[1], DOCS[2]
        store = FakeDenseStore()  # dense order by cosine: b, a, c
        r = HybridTableRetriever(bm25, store, k=2, depth=5)
        out = r.rank("VJC", "consolidated", 2023, leaf_is)
        lexical = HybridTableRetriever(bm25, None, mode="bm25", k=5, depth=5).rank(
            "VJC", "consolidated", 2023, leaf_is)
        lexical_docs = [d for t in lexical for d in DOCS if d.metadata["chunk_id"] == t.chunk_id]
        expected = [d.metadata["chunk_id"]
                    for d in rrf_fuse(lexical_docs, [b, a, c], k=2, c=RRF_CONSTANT)]
        self.assertEqual([t.chunk_id for t in out], expected)
        self.assertEqual([t.rank for t in out], [1, 2])
        self.assertEqual(len(out), 2)  # three candidates exist (C:1, C:2, C:3)
        self.assertGreater(len(lexical), 2)

    def test_backend_error_is_wrapped(self):
        store = FakeDenseStore(error=RuntimeError("chroma down"))
        with self.assertRaises(TableRetrievalError) as caught:
            HybridTableRetriever(bm25, store).rank("VJC", "consolidated", 2023, leaf_is)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception), "hybrid retrieval failed: RuntimeError")

    def test_missing_metadata_is_not_wrapped(self):
        store = FakeDenseStore(metadatas={"VJC23C:3": {"ticker": "VJC"}})
        with self.assertRaises(KeyError):
            HybridTableRetriever(bm25, store, k=2, depth=5).rank("VJC", "consolidated", 2023, leaf_is)

    def test_bad_scope_is_input_error(self):
        store = FakeDenseStore()
        with self.assertRaises(ValueError) as caught:
            HybridTableRetriever(bm25, store).rank("VJC", "nonsense", 2023, leaf_is)
        self.assertNotIsInstance(caught.exception, TableRetrievalError)
        self.assertEqual((store.calls, store.embedded), ([], []))

    def test_constructor_validates(self):
        for kwargs in ({"mode": "dense"}, {"k": 0}, {"k": 10, "depth": 5}, {"mode": "hybrid"},
                       {"mode": "bm25", "k": "3"}, {"mode": "bm25", "depth": 20.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                HybridTableRetriever(bm25, None, **kwargs)

    def test_defaults(self):
        r = HybridTableRetriever(bm25, FakeDenseStore())
        self.assertEqual((r.mode, r.k, r.depth), ("hybrid", 10, 20))


def report_filters(ticker, year, scope):
    return normalize_metadata_filters({"tickers": [ticker], "years": [year], "scopes": [scope]})


class CandidateCacheTests(unittest.TestCase):
    def test_cache_size(self):
        self.assertEqual(table_retriever.CANDIDATE_CACHE_SIZE, 32)

    def test_candidates_computed_once_per_report(self):
        r = HybridTableRetriever(bm25, FakeDenseStore(), k=2, depth=3)
        with patch.object(table_retriever, "filter_document_indices",
                          wraps=table_retriever.filter_document_indices) as spy:
            r.rank("VJC", "consolidated", 2023, leaf_is)
            r.rank("vjc", "consolidated", 2023, leaf_bs)
            self.assertEqual(spy.call_count, 1)
            r.rank("VJC", "consolidated", 2022, leaf_is)
            self.assertEqual(spy.call_count, 2)
        spy.assert_called_with(bm25.docs, report_filters("VJC", 2022, "consolidated"))

    def test_least_recently_used_report_is_evicted(self):
        r = HybridTableRetriever(bm25, None, mode="bm25", k=2, depth=2)
        with patch.object(table_retriever, "CANDIDATE_CACHE_SIZE", 2), \
                patch.object(table_retriever, "filter_document_indices",
                             wraps=table_retriever.filter_document_indices) as spy:
            for year in (2023, 2022, 2023, 2024, 2023):  # the 2023 hit keeps it; 2022 is evicted
                r.rank("VJC", "consolidated", year, leaf_is)
            self.assertEqual(spy.call_count, 3)
            r.rank("VJC", "consolidated", 2022, leaf_is)
            self.assertEqual(spy.call_count, 4)

    def test_dense_vectors_fetched_once_and_query_embedded_once(self):
        store = FakeDenseStore()
        r = HybridTableRetriever(bm25, store, k=2, depth=3)
        first = r.rank("VJC", "consolidated", 2023, leaf_is)
        self.assertEqual(r.rank("VJC", "consolidated", 2023, leaf_is), first)
        self.assertEqual(store.get_calls, 1)
        self.assertEqual(store.embed_calls, 1)
        self.assertEqual(store.similarity_calls, 0)
        self.assertEqual(sorted(store.calls[0][0]), ["VJC23C:1", "VJC23C:2", "VJC23C:3"])
        r.rank("VJC", "consolidated", 2022, leaf_is)  # same query text, another report
        self.assertEqual((store.get_calls, store.embed_calls), (2, 1))
        r.rank("VJC", "consolidated", 2023, leaf_bs)  # another query text, cached report
        self.assertEqual((store.get_calls, store.embed_calls), (2, 2))
        self.assertEqual(store.embedded, [table_query(leaf_is), table_query(leaf_bs)])

    def test_dense_order_is_exact_cosine(self):
        docs = [doc("VJC23C:1", "VJC", 2023, "consolidated", "alpha"),
                doc("VJC23C:2", "VJC", 2023, "consolidated", "beta"),
                doc("VJC23C:3", "VJC", 2023, "consolidated", "gamma")]
        lexical = build_bm25(docs, k=3)
        store = FakeDenseStore(docs, vectors={
            "VJC23C:1": unit(1.0, 0.5), "VJC23C:2": unit(0.0, 1.0), "VJC23C:3": unit(1.0, 0.1)})
        r = HybridTableRetriever(lexical, store, k=3, depth=3)
        out = r.rank("VJC", "consolidated", 2023, leaf_is)
        self.assertEqual([t.chunk_id for t in out], ["VJC23C:3", "VJC23C:1", "VJC23C:2"][:r.k])
        bm25_only = HybridTableRetriever(lexical, None, mode="bm25", k=3, depth=3)
        self.assertEqual(bm25_only.rank("VJC", "consolidated", 2023, leaf_is), ())  # no shared tokens

    def test_dense_ties_keep_candidate_order(self):
        docs = [doc(f"VJC23C:{n}", "VJC", 2023, "consolidated", "alpha") for n in (1, 2, 3)]
        store = FakeDenseStore(docs, vectors={f"VJC23C:{n}": unit(1.0, 1.0) for n in (1, 2, 3)})
        r = HybridTableRetriever(build_bm25(docs, k=3), store, k=3, depth=3)
        out = r.rank("VJC", "consolidated", 2023, leaf_is)  # get returns rows reversed
        self.assertEqual([t.chunk_id for t in out], ["VJC23C:1", "VJC23C:2", "VJC23C:3"])

    def test_hybrid_equals_rrf_of_cached_branches(self):
        query = table_query(leaf_is)
        indices = filter_document_indices(bm25.docs, report_filters("VJC", 2023, "consolidated"))
        exact_dense = [DOCS[1], DOCS[0], DOCS[2]]  # cosine order of the three candidates
        # (1, 1): an untruncated dense list would put VJC23C:3 first instead of VJC23C:2.
        for k, depth, first in ((2, 2, "VJC23C:2"), (1, 1, "VJC23C:2"), (2, 3, "VJC23C:2")):
            with self.subTest(k=k, depth=depth):
                r = HybridTableRetriever(bm25, FakeDenseStore(), k=k, depth=depth)
                lexical = rank_indices(bm25, query, indices, depth)[0]
                expected = [d.metadata["chunk_id"] for d in
                            rrf_fuse(lexical, exact_dense[:depth], k=k, c=RRF_CONSTANT)]
                out = r.rank("VJC", "consolidated", 2023, leaf_is)
                self.assertEqual([t.chunk_id for t in out], expected)
                self.assertEqual(out[0].chunk_id, first)
                self.assertEqual([t.rank for t in out], list(range(1, k + 1)))

    def test_dense_get_error_is_wrapped(self):
        store = FakeDenseStore(error=RuntimeError("chroma down"))
        r = HybridTableRetriever(bm25, store)
        with self.assertRaises(TableRetrievalError) as caught:
            r.rank("VJC", "consolidated", 2023, leaf_is)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception), "hybrid retrieval failed: RuntimeError")
        store.error = None  # a failed fetch is not cached; the next call fetches again
        self.assertTrue(r.rank("VJC", "consolidated", 2023, leaf_is))
        self.assertEqual(store.get_calls, 2)

    def test_embed_and_scoring_errors_are_wrapped(self):
        store = FakeDenseStore(embed_error=OSError("cuda"))
        with self.assertRaises(TableRetrievalError) as caught:
            HybridTableRetriever(bm25, store).rank("VJC", "consolidated", 2023, leaf_is)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        for mode, store in (("bm25", None), ("hybrid", FakeDenseStore())):
            with self.subTest(mode=mode), \
                    patch.object(table_retriever, "rank_indices", side_effect=MemoryError) as ranker, \
                    self.assertRaises(TableRetrievalError) as caught:
                HybridTableRetriever(bm25, store, mode=mode).rank("VJC", "consolidated", 2023, leaf_is)
            self.assertEqual(str(caught.exception), f"{mode} retrieval failed: MemoryError")
            ranker.assert_called_once()

    def test_no_candidates_returns_empty_without_dense_call(self):
        store = FakeDenseStore()
        r = HybridTableRetriever(bm25, store)
        self.assertEqual(r.rank("ZZZ", "consolidated", 2023, leaf_is), ())
        self.assertEqual((store.get_calls, store.embed_calls), (0, 0))
        bm25_only = HybridTableRetriever(bm25, None, mode="bm25")
        self.assertEqual(bm25_only.rank("ZZZ", "consolidated", 2023, leaf_is), ())


class FakeClient:
    def __init__(self, count):
        self.count, self.names = count, []

    def get_collection(self, name):
        self.names.append(name)
        return MagicMock(count=lambda: self.count)


SNAP = Path("snapshot-sentinel")


class OpenTableRetrieverTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.db = root / "corpus.sqlite3"
        self.db.write_bytes(b"corpus")
        self.idx = root / "bm25"
        self.chroma = root / "chroma"
        self.chroma.mkdir()
        save_bm25(bm25, self.idx, "bm25-table", sqlite_source_identity(self.db, chunk_types=("table",)))

    def test_bm25_mode_opens_saved_index(self):
        r = open_table_retriever(self.db, self.idx, bm25_name="bm25-table", mode="bm25")
        self.assertEqual((r.mode, r.k, r.depth), ("bm25", 10, 20))
        self.assertEqual(len(r.bm25.docs), len(bm25.docs))

    def test_identity_mismatch_fails(self):
        self.db.write_bytes(b"changed corpus")
        with self.assertRaises(ValueError):
            open_table_retriever(self.db, self.idx, mode="bm25")

    def test_chroma_count_must_match_bm25(self):
        (self.chroma / "chroma.sqlite3").write_bytes(b"")
        client = FakeClient(count=len(bm25.docs) + 1)
        with patch.object(table_retriever, "chroma_snapshot", return_value=SNAP),                 patch.object(table_retriever, "_chroma_client", return_value=client),                 patch.object(table_retriever, "_embeddings") as emb:
            with self.assertRaises(ValueError) as caught:
                open_table_retriever(self.db, self.idx, chroma_dir=self.chroma)
        emb.assert_not_called()
        self.assertEqual(client.names, ["vifinqa_tables"])
        self.assertIn(str(len(bm25.docs) + 1), str(caught.exception))
        self.assertIn(str(len(bm25.docs)), str(caught.exception))

    def test_missing_chroma_file_fails(self):
        with patch.object(table_retriever, "_chroma_client") as make:
            with self.assertRaises(FileNotFoundError):
                open_table_retriever(self.db, self.idx, chroma_dir=self.chroma)
        make.assert_not_called()

    def test_hybrid_opens_with_fake_backends(self):
        (self.chroma / "chroma.sqlite3").write_bytes(b"")
        client = FakeClient(count=len(bm25.docs))
        store = FakeDenseStore()
        with patch.object(table_retriever, "chroma_snapshot", return_value=SNAP) as snap,                 patch.object(table_retriever, "_chroma_client", return_value=client) as make_client,                 patch.object(table_retriever, "_embeddings", return_value="emb") as emb,                 patch.object(table_retriever, "_chroma_store", return_value=store) as make_store:
            r = open_table_retriever(self.db, self.idx, chroma_dir=self.chroma)
        snap.assert_called_once_with(self.chroma, cache_root=self.chroma.resolve().parent)
        make_client.assert_called_once_with(SNAP)
        self.assertNotIn(self.chroma, [c.args[0] for c in make_client.call_args_list])
        self.assertEqual(r.mode, "hybrid")
        self.assertIs(r.dense_store, store)
        emb.assert_called_once_with("cuda", True)
        make_store.assert_called_once_with(client, "emb")

    def test_bad_arguments_fail_before_loading(self):
        with patch.object(table_retriever, "load_bm25") as load:
            for kwargs in ({"mode": "dense"}, {"k": 0}, {"depth": 5}, {"chroma_dir": None},
                           {"fp16": False}, {"device": "cpu"}):
                with self.subTest(**kwargs), self.assertRaises(ValueError):
                    open_table_retriever(self.db, self.idx, **{"chroma_dir": self.chroma, **kwargs})
            load.assert_not_called()

    def test_cli_none_builds_no_retriever(self):
        parser = argparse.ArgumentParser()
        add_retriever_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual((args.retriever, args.bm25_name, args.device), ("none", "bm25-table", "cuda"))
        self.assertIsNone(args.chroma_dir)
        self.assertTrue(str(args.bm25_dir).replace("\\", "/").endswith("data/indexes/bm25"))
        self.assertIsNone(retriever_from_args(args, self.db))

    def test_cli_bm25_builds_retriever(self):
        parser = argparse.ArgumentParser()
        add_retriever_arguments(parser)
        args = parser.parse_args(["--retriever", "bm25", "--bm25-dir", str(self.idx)])
        self.assertEqual(retriever_from_args(args, self.db).mode, "bm25")


class FakeCrossEncoder:
    """Scores a pair by the first matching marker in the text; records every batch."""

    def __init__(self, scores=None, error=None, result=None):
        self.scores, self.error, self.result = dict(scores or {}), error, result
        self.batches = []

    def score(self, pairs):
        self.batches.append(list(pairs))
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return [next((v for marker, v in self.scores.items() if marker in text), 0.0)
                for _, text in pairs]


class RerankedTableRetrieverTests(unittest.TestCase):
    def fused_ids(self, k, depth=5):
        out = HybridTableRetriever(bm25, FakeDenseStore(), k=k, depth=depth).rank(
            "VJC", "consolidated", 2023, leaf_is)
        return [t.chunk_id for t in out]

    def test_rerank_scores_fused_pool_with_bm25_text(self):
        encoder = FakeCrossEncoder()
        r = HybridTableRetriever(bm25, FakeDenseStore(), mode="hybrid_rerank", k=2, depth=5,
                                 cross_encoder=encoder)
        r.rank("VJC", "consolidated", 2023, leaf_is)
        texts = {d.metadata["chunk_id"]: d.page_content for d in DOCS}
        pool = self.fused_ids(k=5)
        self.assertEqual(len(encoder.batches), 1)
        self.assertEqual(encoder.batches[0], [(table_query(leaf_is), texts[i]) for i in pool])

    def test_rerank_orders_by_score_and_truncates_to_k(self):
        # C:3 ends with "thuyết minh", C:1 with "kinh doanh doanh thu"; C:2 scores 0.
        encoder = FakeCrossEncoder({"thuyết minh": 3.0, "kinh doanh doanh thu": 1.0})
        r = HybridTableRetriever(bm25, FakeDenseStore(), mode="hybrid_rerank", k=2, depth=5,
                                 cross_encoder=encoder)
        out = r.rank("VJC", "consolidated", 2023, leaf_is)
        self.assertNotEqual(self.fused_ids(k=2), ["VJC23C:3", "VJC23C:1"])  # rerank changed it
        self.assertEqual([t.chunk_id for t in out], ["VJC23C:3", "VJC23C:1"])
        self.assertEqual([t.rank for t in out], [1, 2])

    def test_rerank_ties_keep_fused_order(self):
        r = HybridTableRetriever(bm25, FakeDenseStore(), mode="hybrid_rerank", k=3, depth=5,
                                 cross_encoder=FakeCrossEncoder())
        out = r.rank("VJC", "consolidated", 2023, leaf_is)
        self.assertEqual([t.chunk_id for t in out], self.fused_ids(k=3))

    def test_rerank_with_no_candidates_skips_encoder(self):
        encoder = FakeCrossEncoder()
        r = HybridTableRetriever(bm25, FakeDenseStore(), mode="hybrid_rerank", cross_encoder=encoder)
        self.assertEqual(r.rank("VJC", "consolidated", 2019, leaf_is), ())
        self.assertEqual(encoder.batches, [])

    def test_encoder_failure_is_wrapped(self):
        r = HybridTableRetriever(bm25, FakeDenseStore(), mode="hybrid_rerank",
                                 cross_encoder=FakeCrossEncoder(error=RuntimeError("gpu")))
        with self.assertRaises(TableRetrievalError) as caught:
            r.rank("VJC", "consolidated", 2023, leaf_is)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception), "hybrid_rerank retrieval failed: RuntimeError")

    def test_bad_scores_fail_explicitly(self):
        for result in ([float("nan"), 1.0, 2.0], [1.0]):
            with self.subTest(result=result):
                r = HybridTableRetriever(bm25, FakeDenseStore(), mode="hybrid_rerank", k=2, depth=5,
                                         cross_encoder=FakeCrossEncoder(result=result))
                with self.assertRaises(TableRetrievalError):
                    r.rank("VJC", "consolidated", 2023, leaf_is)

    def test_constructor_requires_encoder_only_for_rerank(self):
        with self.assertRaises(ValueError):
            HybridTableRetriever(bm25, FakeDenseStore(), mode="hybrid_rerank")
        with self.assertRaises(ValueError):
            HybridTableRetriever(bm25, None, mode="hybrid_rerank", cross_encoder=FakeCrossEncoder())
        with self.assertRaises(ValueError):
            HybridTableRetriever(bm25, FakeDenseStore(), cross_encoder=FakeCrossEncoder())


class OpenRerankedTableRetrieverTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.db = root / "corpus.sqlite3"
        self.db.write_bytes(b"corpus")
        self.idx = root / "bm25"
        self.chroma = root / "chroma"
        self.chroma.mkdir()
        (self.chroma / "chroma.sqlite3").write_bytes(b"")
        save_bm25(bm25, self.idx, "bm25-table", sqlite_source_identity(self.db, chunk_types=("table",)))

    def open(self, **kwargs):
        encoder = FakeCrossEncoder()
        with patch.object(table_retriever, "chroma_snapshot", return_value=SNAP), \
                patch.object(table_retriever, "_chroma_client", return_value=FakeClient(len(bm25.docs))), \
                patch.object(table_retriever, "_embeddings", return_value="emb"), \
                patch.object(table_retriever, "_chroma_store", return_value=FakeDenseStore()), \
                patch.object(table_retriever, "_cross_encoder", return_value=encoder) as load:
            r = open_table_retriever(self.db, self.idx, chroma_dir=self.chroma, **kwargs)
        return r, encoder, load

    def test_rerank_mode_loads_cross_encoder_on_device(self):
        r, encoder, load = self.open(mode="hybrid_rerank")
        load.assert_called_once_with("cuda")
        self.assertEqual(r.mode, "hybrid_rerank")
        self.assertIs(r.cross_encoder, encoder)

    def test_hybrid_mode_loads_no_cross_encoder(self):
        r, _, load = self.open(mode="hybrid")
        load.assert_not_called()
        self.assertIsNone(r.cross_encoder)

    def test_rerank_needs_chroma_before_loading(self):
        with patch.object(table_retriever, "load_bm25") as bm, \
                patch.object(table_retriever, "_cross_encoder") as load:
            for kwargs in ({"chroma_dir": None}, {"device": "cpu"}, {"fp16": False}):
                with self.subTest(**kwargs), self.assertRaises(ValueError):
                    open_table_retriever(self.db, self.idx, mode="hybrid_rerank",
                                         **{"chroma_dir": self.chroma, **kwargs})
        bm.assert_not_called()
        load.assert_not_called()

    def test_cli_accepts_rerank_mode(self):
        parser = argparse.ArgumentParser()
        add_retriever_arguments(parser)
        args = parser.parse_args(["--retriever", "hybrid_rerank"])
        self.assertEqual(args.retriever, "hybrid_rerank")


if __name__ == "__main__":
    unittest.main()

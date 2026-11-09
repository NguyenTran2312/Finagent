import os
from pathlib import Path
import subprocess
import sys
import unittest

from langchain_core.documents import Document


class HybridRerankerTests(unittest.TestCase):
    def test_retrieval_modules_import_from_project_root(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        for module in ("src.retrieval.hybrid", "src.retrieval.reranker", "src.retrieval.cross_encoder"):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-B", "-c", f"import {module}"],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_tied_page_and_table_scores_are_ordered_without_mutating_sources(self) -> None:
        from src.retrieval.hybrid import rrf_fuse

        page = Document(page_content="page", metadata={"doc_id": "doc", "table_id": None})
        table = Document(page_content="table", metadata={"doc_id": "doc", "table_id": 0})
        for lexical, dense in (([page, table], [table, page]), ([table, page], [page, table])):
            with self.subTest(lexical_first=lexical[0].page_content):
                result = rrf_fuse(lexical, dense, c=0)
                self.assertEqual([doc.page_content for doc in result], ["page", "table"])
                self.assertEqual([doc.metadata["rrf_score"] for doc in result], [1.5, 1.5])
        self.assertEqual(page.metadata, {"doc_id": "doc", "table_id": None})
        self.assertEqual(table.metadata, {"doc_id": "doc", "table_id": 0})

    def test_tied_tables_keep_numeric_order(self) -> None:
        from src.retrieval.hybrid import rrf_fuse

        two = Document(page_content="two", metadata={"doc_id": "doc", "table_id": 2})
        ten = Document(page_content="ten", metadata={"doc_id": "doc", "table_id": 10})
        result = rrf_fuse([ten, two], [two, ten], c=0)
        self.assertEqual([doc.metadata["table_id"] for doc in result], [2, 10])

    def test_reranker_scores_candidates_and_preserves_query_mapping(self) -> None:
        from langchain_core.runnables import RunnableLambda
        from src.retrieval.reranker import build_reranked_retriever

        inputs = {"question": "rewritten", "filter_question": "original"}
        candidates = [Document(page_content="low"), Document(page_content="high")]
        received = []

        def retrieve(value):
            received.append(value)
            return candidates

        class Encoder:
            def score(self, pairs):
                return [0.1 if text == "low" else 0.9 for query, text in pairs]

        result = build_reranked_retriever(RunnableLambda(retrieve), Encoder(), k=1).invoke(inputs)
        self.assertEqual(received, [inputs])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "high")
        self.assertEqual(result[0].metadata["rerank_score"], 0.9)
        self.assertEqual(candidates[1].metadata, {})


if __name__ == "__main__":
    unittest.main()

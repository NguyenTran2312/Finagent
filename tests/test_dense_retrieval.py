import json
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document


class DenseRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        from retrieval import dense

        self.dense = dense

    def test_loads_postgres_chunks_and_preserves_source_identity(self) -> None:
        documents = [
            Document(id="doc:page:0", page_content="Report page", metadata={
                "chunk_id": "doc:page:0", "page_ids": ["doc:page:0"],
                "table_key": None,
            }),
            Document(id="doc:table:0", page_content="Report table", metadata={
                "chunk_id": "doc:table:0", "page_ids": ["doc:page:0"],
                "table_key": "doc:table:0",
            }),
        ]
        params = {"dbname": "finance_corpus"}
        with patch.object(self.dense, "load_corpus_documents", return_value=documents) as load:
            result = self.dense.load_dense_documents(connection_params=params)

        load.assert_called_once_with(params)
        self.assertEqual([document.id for document in result], ["doc:page:0", "doc:table:0"])
        for index, document in enumerate(result):
            self.assertNotIn("page_ids", document.metadata)
            self.assertEqual(json.loads(document.metadata["page_ids_json"]), ["doc:page:0"])
            self.assertEqual(document.metadata["table_index"], index)

    def test_default_connection_is_resolved_by_the_shared_corpus_loader(self) -> None:
        with patch.object(self.dense, "load_corpus_documents", return_value=[]) as load:
            self.assertEqual(self.dense.load_dense_documents(), [])

        load.assert_called_once_with(None)

    def test_build_forwards_connection_and_embedding_options(self) -> None:
        params = {"dbname": "finance_corpus"}
        documents = [object()]
        embeddings = object()
        store = object()
        with (
            patch.object(self.dense, "load_dense_documents", return_value=documents) as load,
            patch.object(self.dense, "get_embeddings", return_value=embeddings) as embed,
            patch.object(self.dense, "load_or_build_chroma", return_value=store) as build,
        ):
            result = self.dense.load_or_build_dense(
                connection_params=params, cache_root="test-cache", device="cpu",
                batch_size=2, max_seq_length=64, load_only=True,
            )

        self.assertIs(result, store)
        load.assert_called_once_with(params)
        embed.assert_called_once_with(device="cpu", batch_size=2, max_seq_length=64)
        build.assert_called_once_with(
            documents, embeddings, cache_root=Path("test-cache"), load_only=True,
        )

    def test_unfinished_corpus_is_rejected_before_loading_the_model(self) -> None:
        with (
            patch.object(self.dense, "load_corpus_documents", side_effect=RuntimeError("Corpus is not ready")),
            patch.object(self.dense, "get_embeddings") as embed,
        ):
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                self.dense.load_or_build_dense({"dbname": "finance_corpus"})

        embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()

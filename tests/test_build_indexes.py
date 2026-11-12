import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.main import build_indexes


class BuildIndexesCliTests(unittest.TestCase):
    def test_chunk_types_use_canonical_order(self) -> None:
        self.assertEqual(build_indexes._chunk_types(["page", "table", "page"]), ("page", "table"))
        self.assertEqual(build_indexes._chunk_types(["table"]), ("table",))

    def test_dense_command_forwards_options(self) -> None:
        with patch.object(build_indexes, "build_dense_index") as build:
            build_indexes.main(["dense", "--db", "c.sqlite3", "--chunk-types", "table",
                                "--device", "cuda", "--batch-size", "16", "--limit", "10",
                                "--chroma-root", "out"])

        build.assert_called_once_with(
            Path("c.sqlite3"), ("table",), limit=10, chroma_root=Path("out"), device="cuda",
            batch_size=16, max_seq_length=build_indexes.DENSE_MAX_SEQ_LENGTH, fp16=False,
        )

    def test_bm25_command_forwards_options(self) -> None:
        with patch.object(build_indexes, "build_bm25_index") as build:
            build_indexes.main(["bm25", "--db", "c.sqlite3", "--bm25-dir", "idx", "--k", "20"])

        build.assert_called_once_with(Path("c.sqlite3"), ("page", "table"), limit=None,
                                      out_dir=Path("idx"), k=20)

    def test_search_requires_query(self) -> None:
        with self.assertRaises(SystemExit):
            build_indexes.main(["search"])

    def test_dense_sqlite_build_prepares_scalar_metadata(self) -> None:
        from retrieval import dense

        documents = [Document(id="d:table:0", page_content="t", metadata={"page_ids": ["d:page:0"]})]
        with (
            patch("retrieval.sqlite_corpus.load_sqlite_corpus_documents", return_value=documents) as load,
            patch.object(dense, "get_embeddings", return_value="emb") as embed,
            patch.object(dense, "load_or_build_chroma", return_value="store") as chroma,
        ):
            result = dense.load_or_build_dense_sqlite("c.sqlite3", chunk_types=("table",), limit=3,
                                                      cache_root="root", device="cpu", batch_size=8)

        self.assertEqual(result, "store")
        load.assert_called_once_with("c.sqlite3", chunk_types=("table",), limit=3)
        embed.assert_called_once_with(device="cpu", batch_size=8, max_seq_length=dense.DENSE_MAX_SEQ_LENGTH,
                                      fp16=False)
        prepared = chroma.call_args.args[0]
        self.assertEqual(prepared[0].metadata, {"page_ids_json": '["d:page:0"]', "table_index": 0})
        self.assertEqual(chroma.call_args.kwargs, {"cache_root": Path("root"), "load_only": False})

    def test_fp16_flag_is_forwarded(self) -> None:
        with patch.object(build_indexes, "build_dense_index") as build:
            build_indexes.main(["dense", "--fp16"])

        self.assertIs(build.call_args.kwargs["fp16"], True)


class Fp16CacheIdentityTests(unittest.TestCase):
    def _embeddings(self, dtype):
        import torch

        class Client:
            max_seq_length = 512

            def parameters(self):
                return iter([torch.zeros(1, dtype=dtype)])

        class Embeddings:
            model_name = "m"
            encode_kwargs = {"normalize_embeddings": True}
            _client = Client()

        return Embeddings()

    def test_only_half_precision_changes_the_fingerprint(self) -> None:
        import torch

        from indexing.chroma import chroma_cache_identity

        documents = [Document(id="a", page_content="t", metadata={"table_key": "a"})]

        class Plain:  # no torch client: the pre-existing identity
            model_name = "m"
            encode_kwargs = {"normalize_embeddings": True}

            class _client:
                max_seq_length = 512

        fp32 = chroma_cache_identity(documents, self._embeddings(torch.float32))
        self.assertEqual(fp32, chroma_cache_identity(documents, Plain()))
        self.assertNotEqual(fp32, chroma_cache_identity(documents, self._embeddings(torch.float16)))

    def test_fp16_requires_cuda(self) -> None:
        from indexing.embedding import get_embeddings

        with self.assertRaises(ValueError):
            get_embeddings(device="cpu", fp16=True)
        with self.assertRaises(ValueError):
            get_embeddings(device="cuda", fp16="yes")


if __name__ == "__main__":
    unittest.main()

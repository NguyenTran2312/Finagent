import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from indexing.bm25_store import index_name, load_bm25, read_manifest, save_bm25, sqlite_source_identity
from retrieval.bm25 import build_bm25


def _retriever():
    documents = [
        Document(id="a", page_content="Tổng tài sản 100", metadata={"ticker": "AAA"}),
        Document(id="b", page_content="Doanh thu thuần 50", metadata={"ticker": "BBB"}),
    ]
    return build_bm25(documents, k=1)


class BM25StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.db = self.folder / "corpus.sqlite3"
        self.db.write_bytes(b"snapshot")
        self.identity = sqlite_source_identity(self.db, chunk_types=("table",))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_trip_keeps_documents_and_ranking(self) -> None:
        save_bm25(_retriever(), self.folder / "idx", "bm25-table", self.identity)

        loaded = load_bm25(self.folder / "idx", "bm25-table", expected_identity=self.identity, k=2)

        self.assertEqual([document.id for document in loaded.docs], ["a", "b"])
        self.assertEqual(loaded.k, 2)
        self.assertEqual(loaded.invoke("doanh thu")[0].id, "b")
        manifest = read_manifest(self.folder / "idx", "bm25-table")
        self.assertEqual(manifest["document_count"], 2)
        self.assertEqual(manifest["identity"], self.identity)

    def test_refuses_index_from_another_snapshot(self) -> None:
        save_bm25(_retriever(), self.folder, "bm25-table", self.identity)
        self.db.write_bytes(b"rebuilt corpus")
        current = sqlite_source_identity(self.db, chunk_types=("table",))

        with self.assertRaises(ValueError):
            load_bm25(self.folder, "bm25-table", expected_identity=current)

    def test_missing_manifest_and_document_count_mismatch(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_bm25(self.folder, "absent")
        save_bm25(_retriever(), self.folder, "bm25-table", self.identity)
        manifest_path = self.folder / "bm25-table.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["document_count"] = 3
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_bm25(self.folder, "bm25-table")

    def test_rejects_invalid_k_and_unknown_manifest_version(self) -> None:
        save_bm25(_retriever(), self.folder, "bm25-table", self.identity)
        with self.assertRaises(ValueError):
            load_bm25(self.folder, "bm25-table", k=0)
        manifest_path = self.folder / "bm25-table.json"
        manifest_path.write_text(json.dumps({"manifest_version": 99}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_bm25(self.folder, "bm25-table")

    def test_names_and_identity_cover_the_selection(self) -> None:
        self.assertEqual(index_name(("table", "page")), "bm25-table-page")
        self.assertEqual(index_name(("table",), 100), "bm25-table-limit100")
        self.assertNotEqual(self.identity, sqlite_source_identity(self.db, chunk_types=("table", "page")))
        self.assertNotEqual(self.identity, sqlite_source_identity(self.db, chunk_types=("table",), limit=5))
        self.assertEqual(list(self.folder.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

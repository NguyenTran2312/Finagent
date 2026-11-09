import errno
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from src.indexing import chroma
from src.indexing.chroma import _copy_cache, load_or_build_chroma


class RecordingStore:
    def __init__(self, **kwargs):
        self.records = {}

    def get(self, **kwargs):
        return {"ids": list(self.records)}

    def add_texts(self, *, texts, metadatas, ids):
        for identifier, text, metadata in zip(ids, texts, metadatas, strict=True):
            self.records[identifier] = (text, metadata)


class ChromaCacheCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.documents = [Document(page_content="Report page", metadata={
            "chunk_id": "doc:page:0", "table_key": None,
        })]
        self.embeddings = SimpleNamespace(
            model_name="test-model", _client=SimpleNamespace(max_seq_length=8),
            encode_kwargs={"normalize_embeddings": True},
        )
        # Reproduce the persisted fingerprint format before the ID fix.
        legacy_configuration = {
            "model": "test-model", "max_seq_length": 8,
            "normalize_embeddings": True, "distance": "cosine",
            "version": chroma.CHROMA_INDEX_VERSION,
        }
        digest = hashlib.sha256(json.dumps(legacy_configuration, sort_keys=True).encode("utf-8"))
        digest.update(json.dumps({
            "text": "Report page",
            "metadata": {"chunk_id": "doc:page:0", "table_key": None},
        }, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
        self.legacy_root = self.root / digest.hexdigest()
        self.legacy_root.mkdir()
        (self.legacy_root / "chroma.sqlite3").touch()
        self.legacy_records = {"None": ("Report page", {"chunk_id": "doc:page:0"})}
        self.caches = {str(self.legacy_root): self.legacy_records}

        def open_client(path):
            return SimpleNamespace(
                records=self.caches.setdefault(path, {}), get_max_batch_size=lambda: 256,
            )

        def open_store(client, **kwargs):
            store = RecordingStore()
            store.records = client.records
            return store

        modules = {
            "chromadb": SimpleNamespace(PersistentClient=open_client),
            "langchain_chroma": SimpleNamespace(Chroma=open_store),
        }
        for patcher in (
            patch.dict(os.environ, {"VIFINQA_CHROMA_CACHE_DIR": ""}),
            patch.dict(sys.modules, modules),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_rebuilds_in_separate_cache_after_legacy_id_change(self) -> None:
        try:
            store = load_or_build_chroma(self.documents, self.embeddings, cache_root=self.root)
        except ValueError as error:
            self.fail(f"Legacy cache prevented rebuilding: {error}")
        self.assertEqual(store.records, {
            "doc:page:0": ("Report page", {"chunk_id": "doc:page:0"}),
        })
        self.assertEqual(self.legacy_records, {
            "None": ("Report page", {"chunk_id": "doc:page:0"}),
        })
        self.assertTrue((self.legacy_root / "chroma.sqlite3").is_file())
        with patch.object(RecordingStore, "add_texts", side_effect=AssertionError("Reindexed existing data")):
            reused = load_or_build_chroma(self.documents, self.embeddings, cache_root=self.root)
        self.assertEqual(reused.records, store.records)

    def test_load_only_reports_missing_current_cache_without_rebuilding(self) -> None:
        try:
            with self.assertRaises(FileNotFoundError):
                load_or_build_chroma(self.documents, self.embeddings, cache_root=self.root, load_only=True)
        except ValueError as error:
            self.fail(f"An incompatible legacy cache was opened: {error}")
        self.assertEqual(list(self.caches), [str(self.legacy_root)])


class ChromaCacheCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        with closing(sqlite3.connect(self.source / "chroma.sqlite3")) as connection:
            connection.execute("CREATE TABLE snapshot (id INTEGER PRIMARY KEY)")
            connection.commit()
        self.destination = self.root / "loaded" / "snapshot"

    def test_reuses_snapshot_published_during_copy(self) -> None:
        real_rename = Path.rename

        def publish_competing_snapshot(staging, destination):
            shutil.copytree(self.source, destination)
            return real_rename(staging, destination)

        with patch.object(Path, "rename", publish_competing_snapshot):
            try:
                _copy_cache(self.source, self.destination)
            except OSError as error:
                self.fail(f"A complete competing snapshot was rejected: {error}")
        self.assertEqual(
            (self.destination / "chroma.sqlite3").read_bytes(),
            (self.source / "chroma.sqlite3").read_bytes(),
        )
        self.assertEqual(list(self.destination.parent.glob(".*.copying")), [])

    def test_reuses_snapshot_after_posix_nonempty_directory_error(self) -> None:
        def publish_competing_snapshot(staging, destination):
            shutil.copytree(self.source, destination)
            raise OSError(errno.ENOTEMPTY, "Destination was published concurrently")

        with patch.object(Path, "rename", publish_competing_snapshot):
            try:
                _copy_cache(self.source, self.destination)
            except OSError as error:
                self.fail(f"A complete competing snapshot was rejected: {error}")
        self.assertTrue((self.destination / "chroma.sqlite3").is_file())
        self.assertEqual(list(self.destination.parent.glob(".*.copying")), [])

    def test_rejects_incomplete_competing_destination(self) -> None:
        real_rename = Path.rename

        def publish_incomplete_snapshot(staging, destination):
            destination.mkdir()
            (destination / "unrelated.txt").write_text("Not a snapshot", encoding="utf-8")
            return real_rename(staging, destination)

        with patch.object(Path, "rename", publish_incomplete_snapshot):
            with self.assertRaises(OSError):
                _copy_cache(self.source, self.destination)
        self.assertFalse((self.destination / "chroma.sqlite3").exists())
        self.assertEqual(list(self.destination.parent.glob(".*.copying")), [])

    def test_propagates_permission_errors_even_with_competing_snapshot(self) -> None:
        def deny_rename(staging, destination):
            shutil.copytree(self.source, destination)
            raise PermissionError(errno.EACCES, "Rename denied")

        with patch.object(Path, "rename", deny_rename):
            with self.assertRaises(PermissionError):
                _copy_cache(self.source, self.destination)
        self.assertEqual(list(self.destination.parent.glob(".*.copying")), [])


class ChromaDocumentIdentityTests(unittest.TestCase):
    def test_indexes_page_chunks_with_distinct_chunk_ids(self) -> None:
        documents = [
            Document(id="doc:page:0", page_content="First page", metadata={
                "chunk_id": "doc:page:0", "table_key": None,
            }),
            Document(id="doc:page:1", page_content="Second page", metadata={
                "chunk_id": "doc:page:1", "table_key": None,
            }),
        ]
        client = SimpleNamespace(get_max_batch_size=lambda: 256)
        modules = {
            "chromadb": SimpleNamespace(PersistentClient=lambda **kwargs: client),
            "langchain_chroma": SimpleNamespace(Chroma=RecordingStore),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(os.environ, {"VIFINQA_CHROMA_CACHE_DIR": ""}), patch.dict(sys.modules, modules):
                try:
                    store = load_or_build_chroma(documents, SimpleNamespace(), cache_root=Path(temporary_directory))
                except ValueError as error:
                    self.fail(f"Valid page chunks were rejected: {error}")
        self.assertEqual(store.records, {
            "doc:page:0": ("First page", {"chunk_id": "doc:page:0"}),
            "doc:page:1": ("Second page", {"chunk_id": "doc:page:1"}),
        })

    def test_preserves_legacy_table_ids_and_non_null_fallbacks(self) -> None:
        documents = [
            Document(page_content="Table", metadata={"table_key": "legacy|table_1", "chunk_id": "new-id"}),
            Document(page_content="No key", metadata={}),
            Document(page_content="Null keys", metadata={"table_key": None, "chunk_id": None}),
        ]
        client = SimpleNamespace(get_max_batch_size=lambda: 256)
        modules = {
            "chromadb": SimpleNamespace(PersistentClient=lambda **kwargs: client),
            "langchain_chroma": SimpleNamespace(Chroma=RecordingStore),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(os.environ, {"VIFINQA_CHROMA_CACHE_DIR": ""}), patch.dict(sys.modules, modules):
                store = load_or_build_chroma(documents, SimpleNamespace(), cache_root=Path(temporary_directory))
        self.assertEqual(list(store.records), ["legacy|table_1", "1", "2"])


if __name__ == "__main__":
    unittest.main()

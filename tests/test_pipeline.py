import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.preprocess import pipeline
from src.preprocess.pipeline import process_dataset


class DatasetPipelineTests(unittest.TestCase):
    def test_imports_from_project_root_without_pythonpath(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import src.preprocess.pipeline"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_processes_pages_and_tables_and_writes_a_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_path = (
                root
                / "financial_statements"
                / "AAA"
                / "2024"
                / "report"
                / "AAA_financial_statements_2024_consolidated_extracted.txt"
            )
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                "===== PAGE 3 =====\nBáo cáo tài chính\n"
                "<table><tr><th>Chỉ tiêu</th><th>2024</th></tr>"
                "<tr><td>Doanh thu</td><td>100</td></tr></table>\n",
                encoding="utf-8",
            )

            summary = process_dataset(root, root / "outputs")
            output_chunks = [
                json.loads(line)
                for line in (root / "outputs" / "chunks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            report = json.loads(
                (root / "outputs" / "preprocess_report.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(summary["documents"], 1)
            self.assertEqual(summary["page_chunks"], 1)
            self.assertEqual(summary["table_chunks"], 1)
            self.assertEqual(summary["total_chunks"], 2)
            self.assertEqual(
                [chunk["metadata"]["chunk_type"] for chunk in output_chunks],
                ["page", "table"],
            )
            self.assertEqual(output_chunks[0]["metadata"]["page_number"], 3)
            self.assertEqual(output_chunks[1]["metadata"]["ticker"], "AAA")
            self.assertEqual(report["summary"], summary)

            report_path.write_text("===== PAGE 3 =====\nUpdated content", encoding="utf-8")
            updated_summary = process_dataset(root, root / "outputs")
            updated_chunks = [
                json.loads(line)
                for line in (root / "outputs" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            updated_report = json.loads(
                (root / "outputs" / "preprocess_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated_summary["total_chunks"], 1)
            self.assertEqual(updated_chunks[0]["page_content"], "Updated content")
            self.assertEqual(len(updated_chunks), 1)
            self.assertEqual(updated_report["summary"], updated_summary)
            self.assertEqual(list((root / "outputs").glob(".*.tmp")), [])

    def test_publication_failure_restores_previous_outputs(self) -> None:
        for existing_outputs in (False, True):
            for failure_target in ("chunks.jsonl", "preprocess_report.json"):
                with self.subTest(existing_outputs=existing_outputs, target=failure_target):
                    with tempfile.TemporaryDirectory() as temporary_directory:
                        root = Path(temporary_directory)
                        reports = root / "financial_statements"
                        reports.mkdir()
                        (reports / "AAA_financial_statements_2024_extracted.txt").write_text(
                            "===== PAGE 1 =====\nNew content", encoding="utf-8",
                        )
                        output = root / "outputs"
                        output.mkdir()
                        previous = {
                            "chunks.jsonl": b"previous chunks\n",
                            "preprocess_report.json": b'{"previous": true}\n',
                        }
                        if existing_outputs:
                            for filename, content in previous.items():
                                (output / filename).write_bytes(content)
                        real_replace = os.replace

                        def fail_publication(source, target):
                            if target.name == failure_target and "backup" not in source.name:
                                raise PermissionError("Simulated publication failure")
                            return real_replace(source, target)

                        with patch.object(pipeline.os, "replace", side_effect=fail_publication):
                            with self.assertRaisesRegex(PermissionError, "publication failure"):
                                process_dataset(root, output)

                        for filename, content in previous.items():
                            if existing_outputs:
                                self.assertEqual((output / filename).read_bytes(), content)
                            else:
                                self.assertFalse((output / filename).exists())
                        self.assertEqual(list(output.glob(".*.tmp")), [])

    def test_failed_restore_keeps_backup_for_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = root / "financial_statements"
            reports.mkdir()
            (reports / "AAA_financial_statements_2024_extracted.txt").write_text(
                "===== PAGE 1 =====\nNew content", encoding="utf-8",
            )
            output = root / "outputs"
            output.mkdir()
            (output / "chunks.jsonl").write_bytes(b"previous chunks\n")
            (output / "preprocess_report.json").write_bytes(b"previous report\n")
            real_replace = os.replace

            def fail_publication_and_restore(source, target):
                if target.name == "preprocess_report.json" or "backup" in source.name:
                    raise PermissionError("Simulated publication/restore failure")
                return real_replace(source, target)

            with patch.object(pipeline.os, "replace", side_effect=fail_publication_and_restore):
                with self.assertRaisesRegex(RuntimeError, "previous chunks retained at") as error:
                    process_dataset(root, output)

            backups = list(output.glob(".chunks-backup-*.tmp"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"previous chunks\n")
            self.assertIn(str(backups[0]), str(error.exception))
            self.assertEqual((output / "preprocess_report.json").read_bytes(), b"previous report\n")
            self.assertEqual(list(output.glob(".report-*.tmp")), [])
            self.assertEqual(list(output.glob(".chunks-*.tmp")), backups)


if __name__ == "__main__":
    unittest.main()

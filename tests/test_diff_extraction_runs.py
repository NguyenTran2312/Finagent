"""Labels of the run diff script (scripts/diff_extraction_runs.py), loaded by path."""

import contextlib
import csv
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diff_extraction_runs.py"
_spec = importlib.util.spec_from_file_location("diff_extraction_runs", SCRIPT)
diff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diff)


def row(outcome="Found", value="1", chunk="D:table:1"):
    return {"outcome": outcome, "value": value, "chunk_id": chunk}


class DiffLabelTests(unittest.TestCase):
    def test_labels(self):
        cases = [
            (row(), row(), None),
            (row(), row(value="2"), "value_changed"),
            (row(), row(chunk="D:table:2"), "source_changed"),
            (row("Ambiguous", "", ""), row(), "ambiguous_to_found"),
            (row(), row("NotFound", "", ""), "found_to_other"),
            (row("NotFound", "", ""), row(), "other_to_found"),
            (row("NotFound", "", ""), row("Ambiguous", "", ""), "other"),
        ]
        for base, new, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(diff.label(base, new), expected)

    def test_key_in_one_run_only_is_other(self):
        self.assertEqual(diff.label(None, row()), "other")
        self.assertEqual(diff.label(row(), None), "other")


class RankHistogramTests(unittest.TestCase):
    def test_rank_bins(self):
        cases = [
            ("retrieved", "1", "1"),
            ("retrieved", "5", "2-5"),
            ("retrieved", "10", "6-10"),
            ("retrieved", "11", ">10"),
            ("retrieved", "", "unranked"),  # §6.1: a statement table only the rule added
            ("retrieved", None, "unranked"),
            ("full_scan", "", None),
            ("", "", None),
        ]
        for step, rank, expected in cases:
            with self.subTest(step=step, rank=rank):
                self.assertEqual(diff.rank_bin(step, rank), expected)

    def test_unranked_is_printed_after_over_ten(self):
        fields = ["template", "ticker", "year", "scope", "leaf", "outcome", "value", "chunk_id",
                  "retrieval_step", "retrieval_rank"]
        rows = [["corporate", "VJC", "2023", "consolidated", f"leaf{i}", "Found", "1", "D:table:1", step, rank]
                for i, (step, rank) in enumerate([("retrieved", "1"), ("retrieved", "12"), ("retrieved", ""),
                                                  ("retrieved", ""), ("full_scan", "")])]
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name in ("base.csv", "new.csv"):
                path = Path(tmp) / name
                with path.open("w", encoding="utf-8", newline="") as fh:
                    csv.writer(fh).writerows([fields, *rows])
                paths.append(path)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                diff.main([str(paths[0]), str(paths[1]), "--out", str(Path(tmp) / "d.csv")])
        self.assertIn("NEW retrieval_rank: 1 1, 2-5 0, 6-10 0, >10 1, unranked 2", out.getvalue())


if __name__ == "__main__":
    unittest.main()

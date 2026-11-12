"""Labels of the run diff script (scripts/diff_extraction_runs.py), loaded by path."""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()

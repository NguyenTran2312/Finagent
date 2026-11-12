"""Compare two leaf_coverage.py CSV runs (e.g. --retriever none vs bm25).

    python -I -X utf8 scripts/diff_extraction_runs.py BASE.csv NEW.csv [--out DIFF.csv]

Rows are keyed by (template, ticker, year, scope, leaf). Every key whose outcome, value or
source chunk differs is written to --out (default: NEW's stem + ``.diff.csv``) with a label:
value_changed, source_changed, ambiguous_to_found, found_to_other, other_to_found or other.
A key present in only one run is labelled ``other`` and has outcome ``absent`` on the
missing side. Printed: label counts, NEW's retrieval_step counts and retrieval_rank
histogram (1, 2-5, 6-10), and p50 / p95 of ``seconds`` for each file.

Reads and writes only the CSV files given.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

KEY_FIELDS = ("template", "ticker", "year", "scope", "leaf")
OUT_FIELDS = ("label", *KEY_FIELDS, "base_outcome", "new_outcome", "base_value", "new_value",
              "base_chunk_id", "new_chunk_id", "new_retrieval_step", "new_retrieval_rank")
ABSENT = "absent"


def label(base: dict | None, new: dict | None) -> str | None:
    """Label of one key's change between two runs; None when unchanged."""
    if base is None or new is None:
        return "other"
    if all(base.get(f, "") == new.get(f, "") for f in ("outcome", "value", "chunk_id")):
        return None
    both_found = base.get("outcome") == "Found" and new.get("outcome") == "Found"
    if both_found and base.get("value", "") != new.get("value", ""):
        return "value_changed"
    if both_found and base.get("chunk_id", "") != new.get("chunk_id", ""):
        return "source_changed"
    if base.get("outcome") == "Ambiguous" and new.get("outcome") == "Found":
        return "ambiguous_to_found"
    if base.get("outcome") == "Found":
        return "found_to_other"
    if new.get("outcome") == "Found":
        return "other_to_found"
    return "other"


def load(path: Path) -> dict[tuple, dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {tuple(r[f] for f in KEY_FIELDS): r for r in csv.DictReader(fh)}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def seconds_summary(rows) -> str:
    values = []
    for r in rows:
        try:
            if r.get("seconds", "") != "":
                values.append(float(r["seconds"]))
        except ValueError:
            continue
    if not values:
        return "no values"
    return f"p50 {percentile(values, 0.5):.4f}s, p95 {percentile(values, 0.95):.4f}s (n={len(values)})"


def rank_bin(text: str) -> str | None:
    try:
        rank = int(text)
    except ValueError:
        return None
    return "1" if rank == 1 else "2-5" if rank <= 5 else "6-10" if rank <= 10 else ">10"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    for path in (args.base, args.new):
        if not path.is_file():
            parser.error(f"CSV not found: {path}")
    out = args.out or args.new.with_name(args.new.stem + ".diff.csv")

    base, new = load(args.base), load(args.new)
    counts: Counter = Counter()
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OUT_FIELDS)
        for key in sorted(set(base) | set(new)):
            b, n = base.get(key), new.get(key)
            lab = label(b, n)
            if lab is None:
                continue
            counts[lab] += 1
            b, n = b or {"outcome": ABSENT}, n or {"outcome": ABSENT}
            writer.writerow((lab, *key, b.get("outcome", ""), n.get("outcome", ""),
                             b.get("value", ""), n.get("value", ""),
                             b.get("chunk_id", ""), n.get("chunk_id", ""),
                             n.get("retrieval_step", ""), n.get("retrieval_rank", "")))

    print(f"# Run diff: {args.base.name} -> {args.new.name}\n")
    print(f"Keys: base {len(base)}, new {len(new)}; changed {sum(counts.values())} -> {out}")
    for lab, count in counts.most_common():
        print(f"- {lab}: {count}")
    steps = Counter(r.get("retrieval_step", "") for r in new.values() if r.get("retrieval_step", ""))
    print("\nNEW retrieval_step: " + (", ".join(f"{k} {v}" for k, v in steps.most_common()) or "none"))
    bins = Counter(b for r in new.values() if (b := rank_bin(r.get("retrieval_rank", ""))))
    print("NEW retrieval_rank: " + ", ".join(f"{b} {bins[b]}" for b in ("1", "2-5", "6-10", ">10")))
    print(f"\nseconds BASE: {seconds_summary(base.values())}")
    print(f"seconds NEW:  {seconds_summary(new.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

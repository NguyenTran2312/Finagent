"""Manual, read-only coverage run of leaf extraction over the local corpus copy.

Not imported by tests. Run from the repository root:

    python -I -X utf8 scripts/leaf_coverage.py [--db PATH] [--leaf ID ...] [--years 2015-2025] [--out CSV]
                                     [--retriever {none,bm25,hybrid}] [--bm25-dir DIR]
                                     [--bm25-name NAME] [--chroma-dir DIR] [--device DEVICE]

For every ticker whose template is corporate or bank, every report year in --years and every
scope that has a document for that year, each requested leaf (all leaves of the template by
default, or the --leaf ids that exist in it) is extracted with ``fallback=None``. One
request is a (ticker, year, scope, leaf) combination.

Printed (stdout; progress goes to stderr): per template and leaf, the count of each outcome
type; then per template the most common ``reason`` values and warning codes (the part of a
warning before its first ':'), the methods of the hits, and how many hits needed the anchor
rule. The anchor count is computed here, not by the extractor: a Found of a statement leaf
whose source table is tagged with a statement kind other than the leaf's own (an untagged,
``note`` or ``unknown`` table, or another statement) was read through the anchor rule.

--retriever (default none) lets the extractor rank the tables of each source with the real
BM25 / hybrid index before reading them; the report header prints it, and "Retrieval steps"
counts the Found facts read from the retrieved shortlist versus a full scan, followed by the
retrieval_error / retrieval_unknown_chunks warning totals.

--out writes one CSV row per request (template, ticker, year, scope, leaf, outcome, reason,
method, warnings, doc_id, anchor, value, chunk_id, retrieval_step, retrieval_rank, seconds).
value, chunk_id and the retrieval columns are empty unless the outcome is Found (and, for
the retrieval columns, carries a trace). seconds is the wall time of the extract call that
produced the row divided by its number of requests, 4 decimals. Nothing else is written.

The database is opened read-only (mode=ro) and is never written.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):  # the pickled BM25 imports preprocess.cleaner
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.calculation.metric_tree import LeafRequest, MetricTree  # noqa: E402
from src.extraction.extractor import LeafExtractor  # noqa: E402
from src.extraction.outcomes import Found  # noqa: E402
from src.extraction.report_store import SqliteReportStore  # noqa: E402
from src.extraction.templates import load_ticker_templates  # noqa: E402
from src.retrieval.table_retriever import add_retriever_arguments, retriever_from_args  # noqa: E402

DEFAULT_DB = REPO_ROOT / "data" / "databases" / "corpus.sqlite3"
TEMPLATES = ("corporate", "bank")
SCOPES = ("consolidated", "separate")  # the extractor accepts only these; other scopes are skipped
TOP_REASONS = 15
CSV_FIELDS = ("template", "ticker", "year", "scope", "leaf", "outcome", "reason", "method",
              "warnings", "doc_id", "anchor", "value", "chunk_id", "retrieval_step",
              "retrieval_rank", "seconds")


def parse_years(text: str) -> range:
    first, sep, last = text.partition("-")
    try:
        lo, hi = int(first), int(last if sep else first)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad --years {text!r} (use 2015-2025 or 2023)") from None
    if lo > hi:
        raise argparse.ArgumentTypeError(f"bad --years {text!r}: start after end")
    return range(lo, hi + 1)


def report_keys(db: Path, years: range) -> list[tuple[str, int, str]]:
    """Distinct (ticker, report_year, scope) of the documents table, sorted."""
    import sqlite3

    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT DISTINCT ticker, report_year, scope FROM documents").fetchall()
    finally:
        con.close()
    return sorted((t.strip().upper(), y, s) for t, y, s in rows if y in years)


class KindLookup:
    """chunk_id -> statement_kind of the tables of the last few documents (read through the store)."""

    def __init__(self, store: SqliteReportStore, keep: int = 4):
        self._store = store
        self._keep = keep
        self._cache: dict[str, dict[str, str | None]] = {}

    def kind(self, doc_id: str, chunk_id: str) -> str | None:
        if doc_id not in self._cache:
            with self._store.session() as session:
                self._cache[doc_id] = {t.chunk_id: t.statement_kind for t in session.tables(doc_id)}
            while len(self._cache) > self._keep:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[doc_id].get(chunk_id)


def describe(outcome) -> tuple[str, str, str, tuple[str, ...]]:
    """(outcome type, reason, method, warnings) of one outcome."""
    kind = type(outcome).__name__
    if isinstance(outcome, Found):
        return kind, "", outcome.extracted.method, outcome.extracted.warnings
    return kind, getattr(outcome, "reason", "") or "", "", tuple(getattr(outcome, "warnings", ()))


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--leaf", action="append", default=None, metavar="ID",
                        help="leaf id to extract (repeatable; default: every leaf)")
    parser.add_argument("--years", type=parse_years, default=range(2015, 2026), metavar="A-B")
    parser.add_argument("--out", type=Path, default=None, help="CSV of per-request outcomes")
    add_retriever_arguments(parser)
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error(f"database not found: {args.db} (pass --db explicitly when the corpus "
                     "is not under this checkout's data/databases/)")

    tree = MetricTree.load()
    templates = load_ticker_templates()
    leaves = {t: [leaf for leaf in tree.leaves(t) if args.leaf is None or leaf.id in args.leaf]
              for t in TEMPLATES}
    if args.leaf:
        known = {leaf.id for t in TEMPLATES for leaf in tree.leaves(t)}
        unknown = sorted(set(args.leaf) - known)
        if unknown:
            parser.error(f"unknown leaf id(s): {', '.join(unknown)}")
    store = SqliteReportStore(args.db)
    extractor = LeafExtractor(store, tree, templates, fallback=None,
                              retriever=retriever_from_args(args, args.db))
    kinds = KindLookup(store)

    keys, skipped = [], Counter()
    for k in report_keys(args.db, args.years):
        template = templates.get(k[0])
        if template is None:
            skipped["no_template"] += 1
        elif template not in TEMPLATES:
            skipped["template_other"] += 1
        elif k[2] == "unknown":
            skipped["unknown_scope"] += 1
        elif k[2] not in SCOPES:
            skipped["other_scope"] += 1
        else:
            keys.append(k)
    print(f"{len(keys)} (ticker, year, scope) reports; leaves: "
          + ", ".join(f"{t}={len(v)}" for t, v in leaves.items()), file=sys.stderr)

    outcomes: dict[tuple[str, str], Counter] = {}
    reasons: dict[str, Counter] = {t: Counter() for t in TEMPLATES}
    warnings: dict[str, Counter] = {t: Counter() for t in TEMPLATES}
    methods: dict[str, Counter] = {t: Counter() for t in TEMPLATES}
    anchor: dict[str, Counter] = {t: Counter() for t in TEMPLATES}  # "hits" / "statement_hits" / "anchor"
    steps: Counter = Counter()
    failures: Counter = Counter()
    out_fh = args.out.open("w", encoding="utf-8", newline="") if args.out else None
    writer = csv.writer(out_fh) if out_fh else None
    if writer:
        writer.writerow(CSV_FIELDS)
    started = time.monotonic()
    try:
        for n, (ticker, year, scope) in enumerate(keys, start=1):
            template = templates.get(ticker)
            by_id = {leaf.id: leaf for leaf in leaves[template]}
            requests = [LeafRequest(leaf.id, year, leaf.period_kind) for leaf in leaves[template]]
            if not requests:
                continue
            call_started = time.monotonic()
            try:
                result = extractor.extract(ticker, scope, requests)
            except Exception as exc:  # keep the run going; each failure is reported at the end
                failures[f"{type(exc).__name__}: {exc}"] += 1
                print(f"FAILED {ticker} {year} {scope}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            seconds = f"{(time.monotonic() - call_started) / len(requests):.4f}"
            for outcome in result.outcomes:
                leaf = by_id[outcome.request.metric_id]
                name, reason, method, warns = describe(outcome)
                outcomes.setdefault((template, leaf.id), Counter())[name] += 1
                if reason:
                    reasons[template][f"{name}:{reason}"] += 1
                for w in warns:
                    warnings[template][w.split(":", 1)[0]] += 1
                doc_id, via_anchor = "", ""
                value = chunk_id = step = rank = ""
                if isinstance(outcome, Found):
                    source = outcome.extracted.fact.source
                    value = str(outcome.extracted.fact.value.value)
                    chunk_id = source.chunk_id or ""
                    trace = outcome.extracted.retrieval
                    if trace is not None:
                        step, rank = trace.step, "" if trace.rank is None else str(trace.rank)
                        steps[step] += 1
                    doc_id = source.doc_id
                    methods[template][method] += 1
                    anchor[template]["hits"] += 1
                    if leaf.source == "statement":
                        anchor[template]["statement_hits"] += 1
                        table_kind = kinds.kind(source.doc_id, source.chunk_id)
                        via_anchor = str(table_kind != leaf.statement_kind)
                        anchor[template]["anchor"] += table_kind != leaf.statement_kind
                if writer:
                    writer.writerow((template, ticker, year, scope, leaf.id, name, reason, method,
                                     ";".join(warns), doc_id, via_anchor, value, chunk_id, step, rank,
                                     seconds))
            if out_fh:
                out_fh.flush()
            if n % 25 == 0 or n == len(keys):
                print(f"[{n}/{len(keys)}] {ticker} {year} {scope} "
                      f"({time.monotonic() - started:.0f}s)", file=sys.stderr)
    finally:
        if out_fh:
            out_fh.close()

    names = sorted({n for c in outcomes.values() for n in c})
    print(f"# Leaf coverage ({args.db.name}, years {args.years.start}-{args.years.stop - 1}, "
          f"fallback=None, retriever={args.retriever})\n")
    print(f"Reports processed: {len(keys)}; run time {time.monotonic() - started:.0f}s")
    print("Reports skipped: " + ", ".join(
        f"{r}={skipped[r]}" for r in ("unknown_scope", "other_scope", "template_other", "no_template"))
        + "\n")
    for template in TEMPLATES:
        total = Counter()
        print(f"## {template}\n")
        print("| leaf | requests | " + " | ".join(names) + " | found % |")
        print("|---|---|" + "---|" * len(names) + "---|")
        for leaf in leaves[template]:
            c = outcomes.get((template, leaf.id), Counter())
            total.update(c)
            n = sum(c.values())
            print(f"| {leaf.id} | {n} | " + " | ".join(str(c[x]) for x in names)
                  + f" | {percent(c['Found'], n)} |")
        n = sum(total.values())
        print(f"| **all** | {n} | " + " | ".join(str(total[x]) for x in names)
              + f" | {percent(total['Found'], n)} |\n")
        print(f"Top reasons ({template}):")
        for key, count in reasons[template].most_common(TOP_REASONS):
            print(f"- {key}: {count}")
        print(f"\nWarning codes ({template}):")
        for key, count in warnings[template].most_common(TOP_REASONS):
            print(f"- {key}: {count}")
        statement_hits = anchor[template]["statement_hits"]
        print(f"\nMethods of hits ({template}): "
              + (", ".join(f"{k} {v}" for k, v in methods[template].most_common()) or "none"))
        print(f"Hits whose source table kind differs from the leaf's, approximating anchor-rule use "
              f"({template}): {anchor[template]['anchor']} of {statement_hits} statement-leaf hits\n")
    print(f"Retrieval steps: retrieved {steps['retrieved']}, full_scan {steps['full_scan']}")
    for code in ("retrieval_error", "retrieval_unknown_chunks"):
        print(f"- {code}: {sum(w[code] for w in warnings.values())}")
    print()
    if failures:
        print("## Failures\n")
        for message, count in failures.most_common():
            print(f"- {count}x {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

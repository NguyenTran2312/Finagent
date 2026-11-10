"""Manual, read-only check of config/metrics codes and aliases against the corpus.

Not imported by tests. Run from the repository root:

    python -I -X utf8 scripts/verify_metric_codes.py [DB_PATH]

Corporate leaves: for a fixed sample of consolidated reports of VAS (Circular 200)
companies, every row of every table whose statement_kind equals the leaf's is read;
rows whose "Mã số" equals one of the leaf's codes are collected and the most common
folded label is printed next to the leaf's aliases. MATCH means alias lookup on that
label returns exactly the leaf (MetricTree.find_by_alias(label) == {leaf}). ACCEPTED is
a reviewed mismatch listed in ACCEPTED_MISMATCHES; it does not fail the run. Leaves
without codes (source: notes) are listed as SKIP.

Bank leaves: for a fixed sample of 10 bank reports, counts the reports in which a
row label is recognised as the leaf. Leaves with a parent must sit inside a section
whose header is recognised as the parent or as one of the parent's children (the
header repeats the child label, decision D2). A header is a roman numeral row or,
where OCR dropped the roman column, a row whose label the next row repeats. A parent
(section) leaf is found when such a header row exists.

The database is opened read-only (mode=ro) and is never written.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.calculation.metric_tree import DEFAULT_METRICS_DIR, MetricTree  # noqa: E402
from src.preprocess.cleaner import fold_text  # noqa: E402

DEFAULT_DB = Path("D:/finance_project/data/databases/corpus.sqlite3")
SEED = 20261110
CORPORATE_SAMPLE = 20
BANK_SAMPLE = 10
PASS_BANK = 8
BANK_YEARS = (2017, 2019, 2021, 2023, 2025)
# Tickers whose statements do not follow the Circular 200 corporate layout.
NON_CORPORATE = frozenset({"SSI", "MBS", "FTS", "EVF", "BVH"})
# Reviewed mismatches: leaf id -> folded top label (leading line numbers stripped).
# Code 61's usual label "Lợi nhuận sau thuế của công ty mẹ" is not an alias, because
# bare "công ty mẹ" means the separate report in this project (query_paraphrase).
ACCEPTED_MISMATCHES = {"profit_after_tax_parent_share": "loi nhuan sau thue cua cong ty me"}
_LEADING_NUMBERS = re.compile(r"^(?:\d+ )+")

_CODE = re.compile(r"^\d{2,3}[a-z]?$")
_CODE_PREFIX = re.compile(r"^(\d{2,3}[a-z]?)[.\s]+(\S.*)$")
_ROMAN = re.compile(r"^([IVXL]+)\.?$")
_ROMAN_PREFIX = re.compile(r"^([IVXL]+)[.\s]+(\S.*)$")
_LOCAL = re.compile(r"^(\d{1,2}|[a-z])[.)]?$")
_LETTERS = re.compile(r"[^\W\d_]")


def _cells(row) -> list[str]:
    return [str(c).strip() for c in row[:4]]


def corporate_code_label(row) -> tuple[str | None, str | None]:
    """Code and label of a Circular 200 row (code before or after the label, or merged)."""
    cells = _cells(row)
    code = next((c for c in cells[:3] if _CODE.match(c)), None)
    if code is None and cells:
        m = _CODE_PREFIX.match(cells[0])
        if m:
            return m.group(1), m.group(2)
        return None, None
    label = next((c for c in cells[:3] if c != code and _LETTERS.search(c)), None)
    return code, label


def bank_code_label(row) -> tuple[str | None, str | None]:
    """('roman'|'local'|None, label) for a bank row."""
    cells = _cells(row)
    if not cells:
        return None, None
    first = cells[0]
    if _ROMAN.match(first):
        kind, rest = "roman", cells[1:]
    elif _LOCAL.match(first):
        kind, rest = "local", cells[1:]
    else:
        m = _ROMAN_PREFIX.match(first)
        if m:
            return "roman", m.group(2)
        kind, rest = None, cells
    label = next((c for c in rest[:2] if _LETTERS.search(c)), None)
    return kind, label


def leaf_ids(file_name: str) -> list[str]:
    entries = yaml.safe_load((DEFAULT_METRICS_DIR / file_name).read_text(encoding="utf-8"))
    return [e["id"] for e in entries]


def connect(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)


def load_tickers(csv_path: Path) -> tuple[list[str], list[str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    banks = [r[0] for r in rows if "Ngân hàng" in r[1]]
    corporates = [r[0] for r in rows if "Ngân hàng" not in r[1] and r[0] not in NON_CORPORATE]
    return banks, corporates


def doc_tables(cur, doc_id: str, kinds: tuple[str, ...] | None = None):
    sql = "SELECT statement_kind, rows_json FROM tables WHERE doc_id = ?"
    args: list = [doc_id]
    if kinds:
        sql += f" AND statement_kind IN ({','.join('?' * len(kinds))})"
        args += list(kinds)
    for kind, rows_json in cur.execute(sql + " ORDER BY table_id", args).fetchall():
        yield kind, json.loads(rows_json)


def check_corporate(cur, tree: MetricTree, corporates: list[str]) -> int:
    marks = ",".join("?" * len(corporates))
    docs = [r[0] for r in cur.execute(
        f"SELECT doc_id FROM documents WHERE scope = 'consolidated' AND ticker IN ({marks}) "
        "ORDER BY doc_id", corporates)]
    sample = random.Random(SEED).sample(docs, min(CORPORATE_SAMPLE, len(docs)))
    print(f"== corporate: {len(sample)} consolidated reports (seed {SEED})")
    print("   " + ", ".join(d.replace("_financial_statements_", " ").replace("_extracted", "")
                            for d in sample))
    leaves = [tree.leaf(i, "corporate") for i in leaf_ids("corporate_leaves.yaml")]
    kinds = tuple(sorted({leaf.statement_kind.value for leaf in leaves if leaf.codes}))
    labels: dict[tuple[str, str], Counter] = {}
    docs_with: dict[tuple[str, str], set] = {}
    for doc in sample:
        for kind, rows in doc_tables(cur, doc, kinds):
            for row in rows:
                code, label = corporate_code_label(row)
                if code and label:
                    labels.setdefault((kind, code), Counter())[fold_text(label)] += 1
                    docs_with.setdefault((kind, code), set()).add(doc)
    failures = 0
    for leaf in leaves:
        if not leaf.codes:
            print(f"{'SKIP':8} {leaf.id:30} {leaf.statement_kind.value:16} (source {leaf.source}, "
                  "no statement code)")
            continue
        for code in leaf.codes:
            key = (leaf.statement_kind.value, code)
            counter = labels.get(key, Counter())
            top = counter.most_common(1)
            top_label, top_n = top[0] if top else ("<none>", 0)
            if tree.find_by_alias(top_label, "corporate") == {leaf.id}:
                status = "MATCH"
            elif ACCEPTED_MISMATCHES.get(leaf.id) == _LEADING_NUMBERS.sub("", top_label):
                status = "ACCEPTED"
            else:
                status = "MISMATCH"
            ok = status != "MISMATCH"
            failures += not ok
            print(f"{status:8} {leaf.id:30} {key[0]:16} {code:>4} "
                  f"docs={len(docs_with.get(key, ())):2}/{len(sample)} "
                  f"top={top_label!r} ({top_n}/{sum(counter.values())})")
            if not ok:
                print(f"         aliases={list(leaf.aliases)}")
                print(f"         next={counter.most_common(4)[1:]}")
    return failures


def bank_sample(cur, banks: list[str]) -> list[str]:
    out = []
    for ticker in banks:
        years = [r[0] for r in cur.execute(
            "SELECT report_year FROM documents WHERE ticker = ? AND scope = 'consolidated'", (ticker,))]
        if not years:
            continue
        target = BANK_YEARS[len(out) % len(BANK_YEARS)]
        year = min(years, key=lambda y: (abs(y - target), y))
        out.append(cur.execute(
            "SELECT doc_id FROM documents WHERE ticker = ? AND scope = 'consolidated' "
            "AND report_year = ? ORDER BY doc_id LIMIT 1", (ticker, year)).fetchone()[0])
        if len(out) == BANK_SAMPLE:
            break
    return out


def check_bank(cur, tree: MetricTree, banks: list[str]) -> int:
    sample = bank_sample(cur, banks)
    print(f"\n== bank: {len(sample)} consolidated reports")
    print("   " + ", ".join(d.replace("_financial_statements_", " ").replace("_extracted", "")
                            for d in sample))
    leaves = [tree.leaf(i, "bank") for i in leaf_ids("bank_leaves.yaml")]
    children: dict[str, set[str]] = {}
    for leaf in leaves:
        if leaf.parent:
            children.setdefault(leaf.parent, set()).add(leaf.id)
    found: dict[str, set] = {leaf.id: set() for leaf in leaves}
    strict: dict[str, set] = {leaf.id: set() for leaf in leaves}
    for doc in sample:
        for kind, rows in doc_tables(cur, doc):
            parsed = [bank_code_label(row) for row in rows]
            folded = [fold_text(label) if label else "" for _, label in parsed]
            section: set[str] = set()
            for idx, (code_kind, label) in enumerate(parsed):
                if not label:
                    continue
                hits = tree.find_by_alias(label, "bank")
                # A section header is a roman numeral row or, when OCR dropped the
                # roman column, a row whose label the next row repeats.
                header = code_kind == "roman" or (
                    idx + 1 < len(folded) and folded[idx] == folded[idx + 1])
                if header:
                    section = hits
                for leaf in leaves:
                    if leaf.id in children:  # section leaf: a header naming it or a child
                        if header and (leaf.id in hits or hits & children[leaf.id]):
                            found[leaf.id].add(doc)
                            if kind == leaf.statement_kind.value:
                                strict[leaf.id].add(doc)
                        continue
                    if leaf.id not in hits:
                        continue
                    if leaf.parent and header:
                        continue  # the header itself is the section, not the child line
                    if leaf.parent and not (leaf.parent in section
                                            or section & children[leaf.parent]):
                        continue
                    found[leaf.id].add(doc)
                    if kind == leaf.statement_kind.value:
                        strict[leaf.id].add(doc)
    failures = 0
    for leaf in leaves:
        n = len(found[leaf.id])
        ok = n >= PASS_BANK
        failures += not ok
        missing = [d.split("_")[0] for d in sample if d not in found[leaf.id]]
        print(f"{'OK' if ok else 'LOW':4} {leaf.id:38} found={n:2}/{len(sample)} "
              f"in_{leaf.statement_kind.value}_tables={len(strict[leaf.id]):2}"
              + (f" missing={missing}" if missing else ""))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", nargs="?", type=Path, default=DEFAULT_DB)
    parser.add_argument("--code-stock", type=Path, default=None,
                        help="ticker CSV (default: <db>/../../code_stock.csv)")
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error(f"database not found: {args.db}")
    code_stock = args.code_stock or args.db.resolve().parents[1] / "code_stock.csv"
    tree = MetricTree.load()
    banks, corporates = load_tickers(code_stock)
    con = connect(args.db)
    try:
        cur = con.cursor()
        corp_fail = check_corporate(cur, tree, corporates)
        bank_fail = check_bank(cur, tree, banks)
    finally:
        con.close()
    print(f"\nsummary: corporate mismatches={corp_fail}, bank leaves below "
          f"{PASS_BANK}/{BANK_SAMPLE}={bank_fail}")
    return 1 if corp_fail or bank_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

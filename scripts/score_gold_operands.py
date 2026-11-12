"""Score the leaf extractor against gold-question operands (read-only corpus; no gold code is executed).

Manual script, not imported by tests. Run from the repository root:

    python -I -X utf8 scripts/score_gold_operands.py GOLD_JSONL OUT_CSV [--db PATH] [--llm MODULE]
                                     [--retriever {none,bm25,hybrid}] [--bm25-dir DIR]
                                     [--bm25-name NAME] [--chroma-dir DIR] [--device DEVICE]

Each distinct (entity, leaf, period, scope) operand is one request. Gold values are normalised to
VND (the operand's ``unit``); a Found value is correct when it is within the source's printed
scale of the gold value. Sign-only differences (expense written as (x) vs x) are counted apart.
The gold ``calculation_program`` field is never read.

--llm MODULE builds the LLM cell locator from MODULE.build_client() (real calls cost money; off by
default). --retriever (default none) ranks the tables of each source with the BM25 / hybrid index
before reading; the report then adds Found counts via retrieved / full_scan and the rank histogram.
The CSV appends chunk_id, retrieval_step and retrieval_rank (empty unless Found with a trace).
The database is opened read-only by the store and is never written.
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib
import json
import sys
from decimal import Decimal
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
# Gold metric name -> leaf id (checked against the operands' source row labels).
ALIAS = {
    "revenue": "net_revenue", "net_income": "profit_after_tax", "equity": "owners_equity",
    "total_liabilities": "liabilities", "interest": "interest_expense", "gross_profit": "gross_profit_reported",
    "inventory": "inventories", "selling_expense": "selling_expenses", "admin_expense": "admin_expenses",
    "pbt": "profit_before_tax", "cogs": "cost_of_goods_sold", "cash": "cash_and_equivalents",
    "cash_and_cash_equivalents": "cash_and_equivalents",
}
SCALE = {"VND": Decimal(1), "million VND": Decimal(10) ** 6, "million_VND": Decimal(10) ** 6,
         "thousand VND": Decimal(1000)}


def rank_bin(rank: int) -> str:
    return "1" if rank == 1 else "2-5" if rank <= 5 else "6-10" if rank <= 10 else ">10"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gold", type=Path, help="gold questions JSONL")
    parser.add_argument("out", type=Path, help="CSV of per-operand verdicts")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--llm", default=None, metavar="MODULE",
                        help="module with build_client() for the LLM cell locator (costs money)")
    add_retriever_arguments(parser)
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error(f"database not found: {args.db} (pass --db explicitly when the corpus "
                     "is not under this checkout's data/databases/)")
    gold_path, out_path, llm_module = args.gold, args.out, args.llm
    tree, templates = MetricTree.load(), load_ticker_templates()
    recs = [json.loads(line) for line in open(gold_path, encoding="utf-8") if line.strip()]

    skipped = collections.Counter()
    gold: dict[tuple, dict] = {}
    for r in recs:
        for o in r["operands"]:
            entity, scope, unit = str(o.get("entity", "")).upper(), o.get("scope"), o.get("unit")
            template = templates.get(entity)
            leaf_id = ALIAS.get(o.get("metric"), o.get("metric"))
            if template not in ("corporate", "bank"):
                skipped[f"template:{template}"] += 1
                continue
            if scope not in ("consolidated", "separate"):
                skipped["no_scope"] += 1
                continue
            if unit not in SCALE:
                skipped[f"unit:{unit}"] += 1
                continue
            try:
                leaf = tree.leaf(leaf_id, template)
            except KeyError:
                skipped[f"not_leaf:{o.get('metric')}"] += 1
                continue
            key = (entity, scope, leaf.id, int(o["period"]))
            # Some operands give the value already in VND while ``unit`` names the table's scale: accept both.
            raw_value = Decimal(str(o["value"]))
            value = raw_value * SCALE[unit]
            row = o["source_locator"].get("row", "")
            g = gold.setdefault(key, {"value": value, "leaf": leaf, "qids": set(), "row": row, "alt": set()})
            g["qids"].add(r["id"])
            g["alt"].update({value, raw_value})
            if o.get("metric") == "interest" and "trong" not in row.lower():
                g["uncertain"] = True

    locator = None
    if llm_module:
        from src.extraction.llm_fallback import LlmCellLocator, structured_pointer_adapter
        mod = importlib.import_module(llm_module)
        locator = LlmCellLocator(structured_pointer_adapter(mod.build_client()),
                                 llm_errors=(TimeoutError, ConnectionError, OSError) + tuple(getattr(mod, "LLM_ERRORS", ())))
    ex = LeafExtractor(SqliteReportStore(args.db), tree, templates, fallback=locator,
                       retriever=retriever_from_args(args, args.db))

    by_call = collections.defaultdict(list)
    for key in gold:
        by_call[key[:2]].append(key)
    results = []
    for (entity, scope), keys in sorted(by_call.items()):
        reqs = [LeafRequest(gold[k]["leaf"].id, k[3], gold[k]["leaf"].period_kind) for k in keys]
        outcomes = ex.extract(entity, scope, reqs).outcomes
        for k, out in zip(keys, outcomes):
            g = gold[k]
            kind = type(out).__name__
            reason = getattr(out, "reason", "") or ""
            got = method = raw = chunk_id = step = rank = ""
            if isinstance(out, Found):
                fact = out.extracted.fact
                got, method, raw = fact.value.value, out.extracted.method, fact.source.raw_text
                chunk_id = fact.source.chunk_id or ""
                trace = out.extracted.retrieval
                if trace is not None:
                    step, rank = trace.step, "" if trace.rank is None else str(trace.rank)
                tol = max(fact.source.scale, Decimal(1))
                golds = {g["value"], *g.get("alt", ())}
                if any(abs(got - v) <= tol for v in golds):
                    verdict = "correct"
                elif any(abs(abs(got) - abs(v)) <= tol for v in golds):
                    verdict = "sign_only"
                else:
                    verdict = "WRONG"
            else:
                verdict = f"miss:{kind}" + (f":{reason}" if reason else "")
            if g.get("uncertain") and verdict != "correct":
                verdict = "mapping_uncertain:" + verdict
            results.append((k, g, verdict, got, method, raw, chunk_id, step, rank))

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "scope", "leaf", "year", "verdict", "gold_vnd", "got_vnd", "method", "raw", "gold_row", "qids",
                    "chunk_id", "retrieval_step", "retrieval_rank"])
        for k, g, verdict, got, method, raw, chunk_id, step, rank in results:
            w.writerow([*k[:2], k[2], k[3], verdict, g["value"], got, method, raw, g["row"], ";".join(map(str, sorted(g["qids"]))),
                        chunk_id, step, rank])

    n = len(results)
    v = collections.Counter(r[2] for r in results)
    print(f"# Gold operand scoring ({args.db.name}, retriever={args.retriever}, llm={llm_module or 'none'})")
    print(f"gold questions: {len(recs)}; operands skipped: {sum(skipped.values())} {dict(skipped)}")
    print(f"distinct operand requests scored: {n}")
    for name, c in v.most_common():
        print(f"  {c:4}  {100 * c / n:5.1f}%  {name}")
    print("\nper leaf (correct / requests, WRONG):")
    per = collections.defaultdict(collections.Counter)
    for k, _, verdict, *_ in results:
        per[k[2]][verdict] += 1
    for leaf, c in sorted(per.items(), key=lambda kv: -sum(kv[1].values())):
        t = sum(c.values())
        print(f"  {leaf:28} {c['correct']:3}/{t:<3} ({100 * c['correct'] / t:5.1f}%)  wrong={c['WRONG']} sign_only={c['sign_only']}")
    q_ok = collections.Counter()
    q_all = collections.Counter()
    status = {}
    for k, g, verdict, *_ in results:
        for q in g["qids"]:
            status.setdefault(q, []).append(verdict in ("correct", "sign_only"))
    full = sum(all(s) for s in status.values())
    print(f"\nquestions with every scored operand correct: {full} of {len(status)} questions that have scored operands")

    if args.retriever != "none":
        steps = collections.Counter(r[7] for r in results if r[7])
        bins = collections.Counter(rank_bin(int(r[8])) for r in results if r[8])
        print(f"\nRetrieval steps: retrieved {steps['retrieved']}, full_scan {steps['full_scan']}")
        print("Retrieval rank: " + ", ".join(f"{b} {bins[b]}" for b in ("1", "2-5", "6-10", ">10")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

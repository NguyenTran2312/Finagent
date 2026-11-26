"""Run Advanced RAG or Agentic RAG over a question file and write one JSON record per question.

Manual script (real LLM calls cost money). Run from the repository root:

    python -X utf8 scripts/run_experiment.py --system {advanced_rag,agentic_rag}
        --questions QUESTIONS_JSONL --out OUT_JSONL --chroma-dir DIR
        [--db PATH] [--bm25-dir DIR] [--model NAME] [--ids 1,2,3] [--limit N] [--resume]
        [--input-price-per-mtok USD --output-price-per-mtok USD]

Only the ``id``, ``question`` and optional ``qa_history`` fields of each input line are read;
any other field (e.g. a gold answer) is ignored and never reaches the system. Records are
appended and flushed one by one; ``--resume`` skips ids already in OUT_JSONL, otherwise an
existing non-empty OUT_JSONL is refused. An exception the agent does not turn into a run is
written as status ``error`` with reason ``exception:<ClassName>`` (traceback on stderr).
Defaults for --db / --bm25-dir / --chroma-dir come from FINAGENT_DB, FINAGENT_BM25_DIR and
FINAGENT_CHROMA_DIR, as in app/demo.py. OPENAI_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):  # the pickled BM25 imports preprocess.cleaner
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.agent.experiment import SYSTEMS, Pricing, error_record, open_runner  # noqa: E402
from src.agent.llm import DEFAULT_MODEL  # noqa: E402


def _env_path(name: str, default: Path | None) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else default


def read_questions(path: Path) -> list[dict]:
    """``[{"id", "question", "qa_history"}]``; every other input field is dropped here."""
    out, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"{path}:{number}: each line needs an 'id'")
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{number}: 'question' must be a non-empty string")
            history = record.get("qa_history", [])
            if not isinstance(history, list):
                raise ValueError(f"{path}:{number}: 'qa_history' must be a list")
            key = str(record["id"])
            if key in seen:
                raise ValueError(f"{path}:{number}: duplicate id {record['id']!r}")
            seen.add(key)
            out.append({"id": record["id"], "question": question, "qa_history": history})
    return out


def _done_ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as handle:
        return {str(json.loads(line)["id"]) for line in handle if line.strip()}


def _price(text: str | None) -> Decimal | None:
    if text is None:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # absent when stdout is redirected to a StringIO
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path,
                        default=_env_path("FINAGENT_DB", REPO_ROOT / "data" / "databases" / "corpus.sqlite3"))
    parser.add_argument("--bm25-dir", type=Path,
                        default=_env_path("FINAGENT_BM25_DIR", REPO_ROOT / "data" / "indexes" / "bm25"))
    parser.add_argument("--chroma-dir", type=Path, default=_env_path("FINAGENT_CHROMA_DIR", None))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ids", default=None, help="comma-separated ids to run (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="run at most N questions")
    parser.add_argument("--resume", action="store_true", help="skip ids already in --out")
    parser.add_argument("--input-price-per-mtok", type=_price, default=None, metavar="USD")
    parser.add_argument("--output-price-per-mtok", type=_price, default=None, metavar="USD")
    args = parser.parse_args(argv)

    if not args.questions.is_file():
        parser.error(f"questions file not found: {args.questions}")
    if not args.db.is_file():
        parser.error(f"database not found: {args.db} (pass --db or set FINAGENT_DB)")
    if not args.bm25_dir.is_dir():
        parser.error(f"BM25 index directory not found: {args.bm25_dir}")
    if args.chroma_dir is None or not args.chroma_dir.is_dir():
        parser.error("--chroma-dir (or FINAGENT_CHROMA_DIR) must name the Chroma index directory: "
                     "both systems use hybrid_rerank retrieval")
    if not args.out.parent.is_dir():
        parser.error(f"output directory does not exist: {args.out.parent}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if (args.input_price_per_mtok is None) != (args.output_price_per_mtok is None):
        parser.error("give both --input-price-per-mtok and --output-price-per-mtok, or neither")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set")

    try:
        questions = read_questions(args.questions)
    except (ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.ids is not None:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        unknown = wanted - {str(q["id"]) for q in questions}
        if unknown:
            parser.error(f"ids not in the questions file: {sorted(unknown)}")
        questions = [q for q in questions if str(q["id"]) in wanted]
    done: set[str] = set()
    if args.out.exists() and args.out.stat().st_size > 0:
        if not args.resume:
            parser.error(f"{args.out} is not empty; pass --resume to continue it")
        done = _done_ids(args.out)
    questions = [q for q in questions if str(q["id"]) not in done]
    if args.limit is not None:
        questions = questions[:args.limit]

    pricing = (None if args.input_price_per_mtok is None
               else Pricing(args.input_price_per_mtok, args.output_price_per_mtok))
    runner = open_runner(args.system, db=args.db, bm25_dir=args.bm25_dir, chroma_dir=args.chroma_dir,
                         model=args.model, pricing=pricing)
    print(f"{args.system}: {len(questions)} questions ({len(done)} already done) → {args.out}")
    counts: dict[str, int] = {}
    with open(args.out, "a", encoding="utf-8") as out:
        for item in questions:
            started = time.perf_counter()
            try:
                record = runner.run(item["id"], item["question"], item["qa_history"])
            except Exception as error:  # written as an error record; the message is not kept
                print(f"id {item['id']}: {type(error).__name__}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                record = error_record(question_id=item["id"], question=item["question"],
                                      system=args.system, config=runner.config, model=runner.model,
                                      error=error, seconds=time.perf_counter() - started)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            key = record["status"] if record["reason"] is None else f"{record['status']}:{record['reason']}"
            counts[key] = counts.get(key, 0) + 1
            print(f"id {item['id']}: {key}")
    print("summary:", json.dumps(dict(sorted(counts.items())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

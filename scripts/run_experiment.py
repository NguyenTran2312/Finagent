"""Run Advanced RAG or Agentic RAG over a question file and write one JSON record per question.

Manual script (real LLM calls cost money). Run from the repository root:

    python -X utf8 scripts/run_experiment.py --system {advanced_rag,agentic_rag}
        --questions QUESTIONS_JSONL --out OUT_JSONL --chroma-dir DIR
        [--db PATH] [--bm25-dir DIR] [--model NAME] [--temperature T] [--seed N]
        [--ids 1,2,3] [--limit N] [--resume]
        [--input-price-per-mtok USD --output-price-per-mtok USD]

Only the ``id``, ``question`` and optional ``qa_history`` fields of each input line are read;
any other field (e.g. a gold answer) is ignored and never reaches the system.

The run's manifest (system, config, LLM parameters, git commit, question-file hash, corpus and
index identities, retrieval parameters, model ids) is written to ``OUT_JSONL.manifest.json``
and every record carries its ``manifest_id``. Records are appended, flushed and fsynced one by
one. An existing non-empty OUT_JSONL is refused unless ``--resume`` is given; resuming requires
the same manifest and that every record carries its id, and skips ids already written. A final
line left incomplete by an interruption is removed (reported on stderr); any other malformed
line is an error. An exception the agent does not turn into a run is written as status
``error`` with reason ``exception:<ClassName>`` and the calls made before it (traceback on
stderr). Defaults for --db / --bm25-dir / --chroma-dir come from FINAGENT_DB,
FINAGENT_BM25_DIR and FINAGENT_CHROMA_DIR, as in app/demo.py. OPENAI_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):  # the pickled BM25 imports preprocess.cleaner
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.agent.experiment import SYSTEMS, Pricing, manifest_id, open_runner, run_manifest  # noqa: E402
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


def manifest_path(out: Path) -> Path:
    return out.with_name(out.name + ".manifest.json")


def existing_records(out: Path, expected_manifest_id: str) -> tuple[set[str], int]:
    """Ids written in ``out`` and the length of its complete part.

    Only a final line without its newline (an interrupted write) is excluded; a complete line
    that is not JSON, has no id, repeats an id or carries another manifest id is an error.
    """
    data = out.read_bytes()
    complete = data.rfind(b"\n") + 1  # 0 when no line is complete
    ids: set[str] = set()
    for number, line in enumerate(data[:complete].decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"{out}:{number}: not a JSON record") from None
        if not isinstance(record, dict) or "id" not in record:
            raise ValueError(f"{out}:{number}: record has no id")
        if record.get("manifest_id") != expected_manifest_id:
            raise ValueError(f"{out}:{number}: record belongs to another run "
                             f"(manifest_id {record.get('manifest_id')!r})")
        key = str(record["id"])
        if key in ids:
            raise ValueError(f"{out}:{number}: duplicate id {record['id']!r}")
        ids.add(key)
    return ids, complete


def _manifest_differences(old: object, new: dict) -> list[str]:
    if not isinstance(old, dict):
        return ["<whole manifest>"]
    return sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))


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
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
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
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature sent to the provider (default: provider default)")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling seed sent to the provider (default: none)")
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
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
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

    pricing = (None if args.input_price_per_mtok is None
               else Pricing(args.input_price_per_mtok, args.output_price_per_mtok))
    manifest = run_manifest(args.system, model=args.model, temperature=args.temperature, seed=args.seed,
                            pricing=pricing, questions=args.questions, db=args.db,
                            bm25_dir=args.bm25_dir, chroma_dir=args.chroma_dir, repo_root=REPO_ROOT)
    run_id = manifest_id(manifest)
    sidecar = manifest_path(args.out)

    done: set[str] = set()
    if args.out.exists() and args.out.stat().st_size > 0:
        if not args.resume:
            parser.error(f"{args.out} is not empty; pass --resume to continue it")
        if not sidecar.is_file():
            parser.error(f"cannot resume: {sidecar} is missing")
        differences = _manifest_differences(json.loads(sidecar.read_text(encoding="utf-8")), manifest)
        if differences:
            parser.error(f"cannot resume: the run manifest differs in {differences}; "
                         "write to a new --out instead")
        try:
            done, complete = existing_records(args.out, run_id)
        except ValueError as error:
            parser.error(f"cannot resume: {error}")
        dropped = args.out.stat().st_size - complete
        if dropped:
            with open(args.out, "r+b") as handle:
                handle.truncate(complete)
            print(f"removed an incomplete final line ({dropped} bytes) from {args.out}", file=sys.stderr)
    else:
        sidecar.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    questions = [q for q in questions if str(q["id"]) not in done]
    if args.limit is not None:
        questions = questions[:args.limit]

    runner = open_runner(args.system, db=args.db, bm25_dir=args.bm25_dir, chroma_dir=args.chroma_dir,
                         model=args.model, temperature=args.temperature, seed=args.seed,
                         pricing=pricing, manifest_id=run_id)
    print(f"{args.system}: {len(questions)} questions ({len(done)} already done) → {args.out} "
          f"[manifest {run_id[:12]}]")
    counts: dict[str, int] = {}
    with open(args.out, "a", encoding="utf-8") as out:
        for item in questions:
            record = runner.run(item["id"], item["question"], item["qa_history"])
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            key = record["status"] if record["reason"] is None else f"{record['status']}:{record['reason']}"
            counts[key] = counts.get(key, 0) + 1
            print(f"id {item['id']}: {key}")
    print("summary:", json.dumps(dict(sorted(counts.items())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

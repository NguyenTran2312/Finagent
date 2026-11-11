"""Manual smoke run of the LLM cell-pointer fallback on one real request. Never run by tests.

Run from the repository root:

    python -I -X utf8 scripts/smoke_llm_fallback.py --model-module MODULE --ticker T --year Y --leaf ID
        [--scope consolidated] [--db PATH]

MODULE is a module the user supplies (importable from the repository root or PYTHONPATH) that
exposes ``build_client()``: it returns a LangChain-style chat model, i.e. an object with
``with_structured_output``; the module owns the model choice and any credentials. The client is
wrapped with ``structured_pointer_adapter``.

The default ``llm_errors`` of ``LlmCellLocator`` (TimeoutError, ConnectionError, OSError) do not
cover provider exceptions such as openai or httpx ones, so such an error would propagate out
of the extraction. The module may therefore define ``LLM_ERRORS``, a tuple of Exception
subclasses; when present it is passed as ``llm_errors`` (added to the defaults).

The request is extracted twice with the same store: once without a fallback, then with
``LlmCellLocator``; both outcomes are printed so the fallback's effect is visible. The
fallback only runs when the deterministic search ends in ``NotFound``, so pick a leaf and
report where it does. The database is opened read-only (mode=ro) and is never written. A real
call to the user's model is made, which may cost money.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calculation.metric_tree import LeafRequest, MetricTree  # noqa: E402
from src.extraction.extractor import LeafExtractor  # noqa: E402
from src.extraction.llm_fallback import LlmCellLocator, structured_pointer_adapter  # noqa: E402
from src.extraction.report_store import SqliteReportStore  # noqa: E402
from src.extraction.templates import load_ticker_templates  # noqa: E402

DEFAULT_DB = REPO_ROOT / "data" / "databases" / "corpus.sqlite3"
DEFAULT_LLM_ERRORS = (TimeoutError, ConnectionError, OSError)


def load_model_module(name: str):
    module = importlib.import_module(name)
    build = getattr(module, "build_client", None)
    if not callable(build):
        raise SystemExit(f"module {name!r} must expose build_client()")
    errors = getattr(module, "LLM_ERRORS", ())
    if not isinstance(errors, tuple) or not all(isinstance(e, type) and issubclass(e, Exception) for e in errors):
        raise SystemExit(f"{name}.LLM_ERRORS must be a tuple of Exception subclasses")
    return build, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-module", required=True, help="module exposing build_client()")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--leaf", required=True, help="leaf id")
    parser.add_argument("--scope", default="consolidated", choices=("consolidated", "separate"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error(f"database not found: {args.db} (pass --db explicitly when the corpus "
                     "is not under this checkout's data/databases/)")

    build_client, extra_errors = load_model_module(args.model_module)
    tree = MetricTree.load()
    templates = load_ticker_templates()
    template = templates.get(args.ticker)
    if template not in ("corporate", "bank"):
        parser.error(f"ticker {args.ticker!r} has template {template!r}; need corporate or bank")
    try:
        leaf = tree.leaf(args.leaf, template)
    except KeyError as exc:
        parser.error(str(exc.args[0]))
    request = LeafRequest(leaf.id, args.year, leaf.period_kind)

    try:
        client = build_client()
    except Exception as exc:
        raise SystemExit(f"build_client failed: {type(exc).__name__}") from None
    store = SqliteReportStore(args.db)
    plain = LeafExtractor(store, tree, templates).extract(args.ticker, args.scope, [request])
    print("deterministic:", plain.outcomes[0])
    locator = LlmCellLocator(
        structured_pointer_adapter(client),
        llm_errors=DEFAULT_LLM_ERRORS + extra_errors,
    )
    try:
        with_llm = LeafExtractor(store, tree, templates, fallback=locator).extract(
            args.ticker, args.scope, [request])
    except Exception as exc:
        raise SystemExit(f"llm extract failed: {type(exc).__name__}") from None
    print("with fallback:", with_llm.outcomes[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

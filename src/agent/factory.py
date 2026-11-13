"""Wires the real components into a ReasoningAgent (spec §6). Heavy modules load lazily."""

from __future__ import annotations

import threading
from pathlib import Path

from config.settings import PROJECT_ROOT
from src.agent.agent import ReasoningAgent
from src.agent.llm import CountingLlm, LlmUnavailable
from src.agent.types import AgentConfig
from src.calculation.metric_tree import MetricTree
from src.extraction.extractor import CellLocator, LeafExtractor
from src.extraction.report_store import SqliteReportStore
from src.extraction.templates import load_ticker_templates
from src.query_paraphrase.entities import EntityRegistry
from src.query_paraphrase.paraphraser import QueryParaphraser
from src.query_paraphrase.schemas import QueryPlan


# Opened retrievers are shared by every agent: a new AgentConfig (slider, toggle) must not reload
# BGE-M3. Keyed by the arguments that identify the index; guarded so sessions open it once.
_RETRIEVERS: dict[tuple, object] = {}
_RETRIEVERS_LOCK = threading.Lock()


def _shared_retriever(db: Path, bm25_dir: Path, chroma_dir: Path | None, mode: str):
    key = (str(db), str(bm25_dir), None if chroma_dir is None else str(chroma_dir), mode)
    with _RETRIEVERS_LOCK:
        if key not in _RETRIEVERS:
            from src.retrieval.table_retriever import open_table_retriever

            _RETRIEVERS[key] = open_table_retriever(db, bm25_dir, chroma_dir=chroma_dir, mode=mode)
        return _RETRIEVERS[key]


def reset_retriever_cache() -> None:
    with _RETRIEVERS_LOCK:
        _RETRIEVERS.clear()


def build_agent(
    config: AgentConfig,
    *,
    llm: CountingLlm,
    db: Path,
    bm25_dir: Path,
    chroma_dir: Path | None,
) -> ReasoningAgent:
    store = SqliteReportStore(db)
    tree = MetricTree.load()
    templates = load_ticker_templates()
    registry = EntityRegistry.from_csv(PROJECT_ROOT / "data" / "code_stock.csv")
    paraphraser = QueryParaphraser(lambda prompt: llm(prompt, QueryPlan), registry=registry,
                                   provider_errors=(LlmUnavailable,))
    def retriever():
        if config.retriever_mode == "none":
            return None
        return _shared_retriever(db, bm25_dir, chroma_dir, config.retriever_mode)

    def extractor_factory(locator: CellLocator | None) -> LeafExtractor:
        return LeafExtractor(store, tree, templates, fallback=locator, retriever=retriever())

    return ReasoningAgent(paraphraser, tree, templates, extractor_factory, llm, config)

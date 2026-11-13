"""Wires the real components into a ReasoningAgent (spec §6). Heavy modules load lazily."""

from __future__ import annotations

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
    retriever_cache: list[object] = []  # opened once, on first use

    def retriever():
        if config.retriever_mode == "none":
            return None
        if not retriever_cache:
            from src.retrieval.table_retriever import open_table_retriever

            retriever_cache.append(open_table_retriever(
                db, bm25_dir, chroma_dir=chroma_dir, mode=config.retriever_mode))
        return retriever_cache[0]

    def extractor_factory(locator: CellLocator | None) -> LeafExtractor:
        return LeafExtractor(store, tree, templates, fallback=locator, retriever=retriever())

    return ReasoningAgent(paraphraser, tree, templates, extractor_factory, llm, config)

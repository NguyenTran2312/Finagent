"""Gather: turns data needs into leaf requests for the extractor and logs every leaf outcome (spec §4.2)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import get_args

from src.agent.types import DataNeed, LeafLog, Scope
from src.calculation.facts import Fact, FactSet
from src.calculation.metric_tree import LeafDef, LeafRequest, MetricTree, Template
from src.extraction.extractor import CellLocator, LeafExtractor, LocatorResult
from src.extraction.outcomes import Found, Outcome
from src.extraction.table_view import SourceRole, TableView
from src.extraction.templates import TickerTemplates

_CALCULATION_TEMPLATES = frozenset(get_args(Template))
_Key = tuple[str, str, int, str]  # (leaf_id, ticker, year, scope)


class CappedLocator:
    """Delegates to ``inner`` for the first ``cap`` calls, then skips without calling it."""

    def __init__(self, inner: CellLocator, cap: int):
        self._inner = inner
        self._cap = cap
        self.calls = 0
        self.skipped: set[tuple[str, int]] = set()

    def locate(
        self, leaf: LeafDef, request: LeafRequest, role: SourceRole, views: Sequence[TableView]
    ) -> LocatorResult:
        if self.calls >= self._cap:
            self.skipped.add((leaf.id, request.year))
            return LocatorResult(None, "locator_skipped", ())
        self.calls += 1
        return self._inner.locate(leaf, request, role, views)


class Gatherer:
    def __init__(
        self,
        tree: MetricTree,
        templates: TickerTemplates,
        extractor_factory: Callable[[CellLocator | None], LeafExtractor],
        locator: CellLocator | None,
        max_locator_calls: int,
        default_scope: Scope,
    ):
        self._tree = tree
        self._templates = templates
        self._capped = CappedLocator(locator, max_locator_calls) if locator else None
        self._extractor = extractor_factory(self._capped)
        self._default_scope = default_scope
        self._facts: list[Fact] = []
        self._logs: list[LeafLog] = []
        self._by_key: dict[_Key, LeafLog] = {}
        self.facts = FactSet(())

    @property
    def log(self) -> tuple[LeafLog, ...]:
        return tuple(self._logs)

    def requested(self, need: DataNeed) -> bool:
        keys = self._need_keys(need)
        return bool(keys) and all(k in self._by_key for k in keys)

    def failures(self, need: DataNeed) -> tuple[LeafLog, ...]:
        logs = (self._by_key.get(k) for k in self._need_keys(need))
        return tuple(l for l in logs if l is not None and l.outcome != "Found")

    def unmet(self, need: DataNeed) -> tuple[LeafLog, ...]:
        """The need's failure logs, or () when every year it covers has a fully Found alternative.

        Over-fetched alternatives that failed are harmless once another alternative is complete.
        """
        ticker, scope = _ticker(need), need.scope or self._default_scope
        found = {k for k, log in self._by_key.items() if log.outcome == "Found"}
        if all(any({(r.metric_id, ticker, r.year, scope) for r in alt} <= found for alt in alts)
               for alts in self._alternatives(need)):
            return ()
        return self.failures(need)

    def gather(self, needs: Iterable[DataNeed]) -> tuple[LeafLog, ...]:
        groups: dict[tuple[str, str], list[LeafRequest]] = {}
        for need in needs:
            ticker, scope = _ticker(need), need.scope or self._default_scope
            for request in self._requests(need):
                key = (request.metric_id, ticker, request.year, scope)
                if key not in self._by_key and request not in groups.get((ticker, scope), ()):
                    groups.setdefault((ticker, scope), []).append(request)
        added: list[LeafLog] = []
        for (ticker, scope), requests in groups.items():
            added.extend(self._extract(ticker, scope, requests))
        if added:
            self.facts = FactSet(self._facts)
        return tuple(added)

    # --- internals ------------------------------------------------------------
    def _template(self, need: DataNeed):
        return self._templates.get(_ticker(need))

    def _years(self, need: DataNeed) -> tuple[int, ...]:
        return (need.year, need.year - 1) if need.helper == "avg_balance" else (need.year,)

    def _resolvable(self, need: DataNeed) -> bool:
        template = self._template(need)
        return template in _CALCULATION_TEMPLATES and self._tree.applies(need.node_id, template)

    def _alternatives(self, need: DataNeed) -> list[list[frozenset[LeafRequest]]]:
        """Per year the need covers, its leaf-set alternatives (one minimal request when unresolvable)."""
        if self._resolvable(need):
            template = self._template(need)
            return [self._tree.required_leaves(need.node_id, template, y) for y in self._years(need)]
        return [[frozenset({LeafRequest(need.node_id, y, "flow")})] for y in self._years(need)]

    def _requests(self, need: DataNeed) -> list[LeafRequest]:
        """Leaf requests of one need; a minimal request when the extractor must report the failure."""
        if self._resolvable(need):
            leaves = {r for alts in self._alternatives(need) for alt in alts for r in alt}
            return sorted(leaves, key=lambda r: (r.year, r.metric_id, r.period_kind))
        # UnknownTicker / UnsupportedTemplate / NotApplicable come from the extractor itself.
        return [LeafRequest(need.node_id, y, "flow") for y in self._years(need)]

    def _need_keys(self, need: DataNeed) -> list[_Key]:
        ticker, scope = _ticker(need), need.scope or self._default_scope
        return [(r.metric_id, ticker, r.year, scope) for r in self._requests(need)]

    def _extract(self, ticker: str, scope: Scope, requests: list[LeafRequest]) -> list[LeafLog]:
        # CappedLocator.skipped is keyed (leaf, year) without a ticker: attribute only the keys this call adds.
        before = set(self._capped.skipped) if self._capped else set()
        if self._capped:
            self._capped.skipped.clear()
        try:
            result = self._extractor.extract(ticker, scope, requests)
        finally:
            skipped = set(self._capped.skipped) if self._capped else set()
            if self._capped:
                self._capped.skipped |= before
        logs = []
        for outcome in result.outcomes:
            log = _leaf_log(outcome, ticker, scope, skipped)
            self._logs.append(log)
            self._by_key[(log.leaf_id, ticker, log.year, scope)] = log
            if isinstance(outcome, Found):
                self._facts.append(outcome.extracted.fact)
            logs.append(log)
        return logs


def _ticker(need: DataNeed) -> str:
    return need.ticker.strip().upper()


def _leaf_log(outcome: Outcome, ticker: str, scope: Scope, skipped: set[tuple[str, int]]) -> LeafLog:
    request = outcome.request
    name = type(outcome).__name__
    reason = getattr(outcome, "reason", None)
    tool = "extractor"
    value = chunk_id = doc_id = page = row_label = None
    if isinstance(outcome, Found):
        fact = outcome.extracted.fact
        value, source = fact.value.value, fact.source
        chunk_id, doc_id, page, row_label = source.chunk_id, source.doc_id, source.page, source.row_label
        if outcome.extracted.method == "llm_located":
            tool = "locator"
    elif name == "NotFound" and (request.metric_id, request.year) in skipped:
        tool = "locator_skipped"
    return LeafLog(request.metric_id, ticker, request.year, scope, name, reason, value, chunk_id, doc_id, page,
                   row_label, tool)

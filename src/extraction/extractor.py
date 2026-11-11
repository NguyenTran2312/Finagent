"""Leaf extraction orchestration: validation, templates, report sources and scope variants (spec §5).

Per request: the primary report (year y, current column) and, when it gives ``ReportMissing`` or
``NotFound``, the next report (year y+1, comparative column). Each document of a source is read
on its own (§5.5); ``_combine_variants`` and ``_combine_sources`` are pure functions over those
per-document outcomes (§5.4, §5.3).

``NotFound`` reasons set here: ``no_row`` (no matching row in any candidate table),
``section_unresolved`` (the same for a section or child leaf, §6.4) and ``no_period_column``
(rows matched but none has a column for the period, §6.5). A locator may set others.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, get_args

from src.calculation.facts import Fact, FactSet, Scope
from src.calculation.metric_tree import LeafDef, LeafRequest, MetricTree, Template
from src.extraction.matching import RowHit, find_rows
from src.extraction.outcomes import (
    Ambiguous,
    BlankValue,
    Candidate,
    CellLocation,
    Conflict,
    ExtractedFact,
    Found,
    Method,
    NotApplicable,
    NotFound,
    Outcome,
    ReportMissing,
    UnitUnknown,
    UnknownTicker,
    UnsupportedTemplate,
)
from src.extraction.report_store import ReportDoc, ReportSession, ReportStore
from src.extraction.table_view import CellRead, MalformedTable, SourceRole, TableView
from src.extraction.templates import TickerTemplates

_SCOPES = frozenset(get_args(Scope))
_SUPPORTED_TEMPLATES = frozenset(get_args(Template))
NO_ROW = "no_row"
_VARIANT_KIND = {Ambiguous: "ambiguous", BlankValue: "blank_value", UnitUnknown: "unit_unknown"}


@dataclass(frozen=True)
class LocatorResult:
    """A fallback locator's answer; ``read.status == "ok"`` whenever ``read`` is set."""

    read: CellRead | None
    reason: str | None
    warnings: tuple[str, ...]


class CellLocator(Protocol):
    def locate(
        self, leaf: LeafDef, request: LeafRequest, role: SourceRole, views: Sequence[TableView]
    ) -> LocatorResult: ...


@dataclass(frozen=True)
class ExtractionResult:
    ticker: str
    scope: Scope
    outcomes: tuple[Outcome, ...]  # one per distinct request, request order

    @property
    def facts(self) -> tuple[ExtractedFact, ...]:
        return tuple(o.extracted for o in self.outcomes if isinstance(o, Found))

    def fact_set(self) -> FactSet:
        """May raise ``FactConflict``."""
        return FactSet(f.fact for f in self.facts)


@dataclass(frozen=True)
class _Doc:
    """One document's tables, loaded and viewed once per call."""

    doc: ReportDoc
    views: tuple[TableView, ...]
    skipped: tuple[str, ...]
    warnings: tuple[str, ...]


class _Call:
    """The session and per-call caches of one ``extract`` call; nothing outlives the call."""

    def __init__(self, session: ReportSession, ticker: str):
        self.ticker = ticker
        self._session = session
        self._docs: dict[int, list[ReportDoc]] = {}
        self._loaded: dict[str, _Doc] = {}

    def documents(self, year: int, scope: str) -> tuple[tuple[_Doc, ...], tuple[str, ...]]:
        """Documents of ``scope`` for ``year``, and the sorted scopes of all documents of that year."""
        if year not in self._docs:
            self._docs[year] = self._session.documents(self.ticker, year)
        docs = self._docs[year]
        return tuple(self._load(d) for d in docs if d.scope == scope), tuple(sorted({d.scope for d in docs}))

    def _load(self, doc: ReportDoc) -> _Doc:
        if doc.doc_id not in self._loaded:
            views, skipped, warnings = [], [], []
            for record in self._session.tables(doc.doc_id):
                try:
                    views.append(TableView(record, doc.report_year))
                except MalformedTable as exc:
                    code = "table_without_chunk" if exc.reason == "no_chunk" else "skipped_table"
                    skipped.append(exc.table_key)
                    warnings.append(f"{code}:{exc.table_key}")
            self._loaded[doc.doc_id] = _Doc(doc, tuple(views), tuple(skipped), tuple(warnings))
        return self._loaded[doc.doc_id]


@dataclass(frozen=True)
class _Job:
    """One request with a leaf: what every outcome needs."""

    call: _Call
    leaf: LeafDef
    template_leaves: tuple[LeafDef, ...]
    request: LeafRequest
    scope: Scope

    @property
    def keys(self) -> dict:
        return {"ticker": self.call.ticker, "scope": self.scope, "request": self.request}


@dataclass(frozen=True)
class _HitRead:
    """A matched row with its period column(s) read: one read, or every column's read when they differ."""

    hit: RowHit
    col: int
    reads: tuple[CellRead, ...]

    @property
    def resolved(self) -> bool:
        return len(self.reads) == 1


class LeafExtractor:
    def __init__(
        self,
        store: ReportStore,
        tree: MetricTree,
        templates: TickerTemplates,
        fallback: CellLocator | None = None,
    ):
        self._store = store
        self._tree = tree
        self._templates = templates
        self._fallback = fallback

    def extract(self, ticker: str, scope: Scope, requests: Sequence[LeafRequest]) -> ExtractionResult:
        ticker, template, distinct, leaves = self._validate(ticker, scope, requests)
        with self._store.session() as session:  # E1: exactly one session per call
            call = _Call(session, ticker)
            outcomes = tuple(self._outcome(call, template, leaves.get(r), scope, r) for r in distinct)
        return ExtractionResult(ticker, scope, outcomes)

    # --- validation (§5.1, R4; no store access) ------------------------------
    def _validate(self, ticker, scope, requests):
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a str")
        if not isinstance(scope, str) or scope not in _SCOPES:
            raise ValueError(f"scope must be one of {sorted(_SCOPES)}")
        requests = list(requests)
        for request in requests:
            if not isinstance(request, LeafRequest):
                raise TypeError(f"requests must hold LeafRequest items, got {type(request).__name__}")
        ticker = ticker.strip().upper()
        template = self._templates.get(ticker)
        distinct = tuple(dict.fromkeys(requests))
        leaves: dict[LeafRequest, LeafDef] = {}
        if template in _SUPPORTED_TEMPLATES:
            for request in distinct:
                try:
                    leaf = self._tree.leaf(request.metric_id, template)
                except KeyError:
                    continue  # NotApplicable
                if leaf.period_kind != request.period_kind:
                    raise ValueError(
                        f"period_kind {request.period_kind!r} of {request.metric_id} differs from the leaf's "
                        f"{leaf.period_kind!r}"
                    )
                leaves[request] = leaf
        return ticker, template, distinct, leaves

    # --- one request (§5.2, §5.3, §5.6) ---------------------------------------
    def _outcome(self, call: _Call, template, leaf: LeafDef | None, scope: Scope, request: LeafRequest) -> Outcome:
        keys = {"ticker": call.ticker, "scope": scope, "request": request}
        if template is None:
            return UnknownTicker(**keys)
        if template not in _SUPPORTED_TEMPLATES:
            return UnsupportedTemplate(**keys, template=template)
        if leaf is None:
            return NotApplicable(**keys, template=template)
        job = _Job(call, leaf, self._tree.leaves(template), request, scope)
        primary_docs, outcome = self._source(job, request.year, "primary")
        next_docs: tuple[_Doc, ...] = ()
        if isinstance(outcome, (ReportMissing, NotFound)):
            next_docs, fallback = self._source(job, request.year + 1, "fallback")
            outcome = _combine_sources(outcome, fallback)
        if isinstance(outcome, NotFound) and self._fallback is not None and (primary_docs or next_docs):
            role: SourceRole = "primary" if primary_docs else "fallback"
            outcome = self._locate(job, primary_docs or next_docs, role, outcome)
        return outcome

    def _source(self, job: _Job, year: int, role: SourceRole) -> tuple[tuple[_Doc, ...], Outcome]:
        docs, scopes = job.call.documents(year, job.scope)
        if not docs:
            return (), ReportMissing(**job.keys, available_scopes=scopes)
        return docs, _combine_variants(job, [(d.doc, _in_document(job, d, role)) for d in docs])

    def _locate(self, job: _Job, docs: tuple[_Doc, ...], role: SourceRole, deterministic: NotFound) -> Outcome:
        """§5.6: one locator call per document of the source the deterministic search used."""
        per_doc: list[tuple[ReportDoc, Outcome]] = []
        for d in docs:
            result = self._fallback.locate(job.leaf, job.request, role, d.views)
            if result.read is None:
                per_doc.append((d.doc, replace(
                    deterministic,
                    reason=result.reason or deterministic.reason,
                    searched_doc_ids=(d.doc.doc_id,),
                    warnings=result.warnings,
                )))
                continue
            if result.read.status != "ok":
                raise ValueError(f"CellLocator returned a read with status {result.read.status!r}")
            restated = role == "fallback" and _restated(d.views, result.read.location)
            warnings = result.warnings + d.warnings + _role_warnings(d.doc, role)
            per_doc.append((d.doc, _found(job, result.read, "llm_located", warnings, restated)))
        if not any(isinstance(o, Found) for _, o in per_doc):
            located = [o for _, o in sorted(per_doc, key=lambda p: p[0].doc_id)]
            reason = next((o.reason for o in located if o.reason != deterministic.reason), deterministic.reason)
            return replace(deterministic, reason=reason,
                           warnings=_unique(deterministic.warnings, *(o.warnings for o in located)))
        outcome = _combine_variants(job, per_doc)
        return _with_warnings(outcome, ("primary_report_missing",)) if role == "fallback" else outcome


# --- one document (§5.5, §6.5) ------------------------------------------------
def _in_document(job: _Job, d: _Doc, role: SourceRole) -> Outcome:
    hits = find_rows(job.leaf, d.views, job.template_leaves)
    hit_reads = []
    for hit in hits:
        cols = hit.view.period_columns(job.request.year, job.request.period_kind, role)
        if cols:  # a row without a column for this period holds no value for it
            reads = [hit.view.read(hit.row_idx, c, job.leaf.dimension) for c in cols]
            hit_reads.append(_HitRead(hit, cols[0], (reads[0],) if _same(reads) else tuple(reads)))
    if not hit_reads:
        if hits:
            reason = "no_period_column"
        elif job.leaf.parent is not None or any(other.parent == job.leaf.id for other in job.template_leaves):
            reason = "section_unresolved"
        else:
            reason = NO_ROW
        return NotFound(**job.keys, reason=reason, searched_doc_ids=(d.doc.doc_id,),
                        skipped_tables=d.skipped, warnings=d.warnings)
    first = hit_reads[0]
    if len(hit_reads) == 1 and not first.resolved:
        return Ambiguous(**job.keys, reason="multiple_period_columns", candidates=_candidates(first.reads))
    if len(hit_reads) > 1 and not (
        all(h.resolved for h in hit_reads) and _same([h.reads[0] for h in hit_reads])
    ):
        return Ambiguous(**job.keys, reason="multiple_rows",
                         candidates=_candidates(r for h in hit_reads for r in h.reads))
    return _read_outcome(job, d, role, first)


def _read_outcome(job: _Job, d: _Doc, role: SourceRole, hit_read: _HitRead) -> Outcome:
    read = hit_read.reads[0]
    if read.status == "blank":
        return BlankValue(**job.keys, location=read.location)
    if read.status == "unit":
        return UnitUnknown(**job.keys, reason=read.reason, location=read.location)
    if read.status == "unparsed":
        return Ambiguous(**job.keys, reason=read.reason, candidates=_candidates((read,)))
    hit = hit_read.hit
    restated = role == "fallback" and hit.view.restatement_marker(hit_read.col)
    return _found(job, read, hit.method, hit.warnings + d.warnings + _role_warnings(d.doc, role), restated)


def _found(job: _Job, read: CellRead, method: Method, warnings: tuple[str, ...], restated: bool) -> Found:
    fact = Fact(
        metric_id=job.request.metric_id,
        ticker=job.call.ticker,
        fiscal_year=job.request.year,
        scope=job.scope,
        period_kind=job.request.period_kind,
        value=read.value,
        source=read.source,
        restated=restated,
    )
    return Found(**job.keys, extracted=ExtractedFact(fact, method, _unique(warnings)))


def _role_warnings(doc: ReportDoc, role: SourceRole) -> tuple[str, ...]:
    return (f"from_next_year_report:{doc.doc_id}",) if role == "fallback" else ()


def _restated(views: Sequence[TableView], location: CellLocation) -> bool:
    """Restatement marker of the located cell's column (header or table context, §5.3)."""
    for view in views:
        if view.record.chunk_id == location.chunk_id:
            return any(
                view.restatement_marker(c) for c in view.value_cols if view.column_header(c) == location.column_header
            )
    return False


def _same(reads: Sequence[CellRead]) -> bool:
    """All reads give the same result: equal values for ``ok`` reads, else equal status and reason."""
    return len({(r.status, r.value.value if r.status == "ok" else r.reason) for r in reads}) == 1


def _candidates(reads: Iterable[CellRead]) -> tuple[Candidate, ...]:
    return tuple(Candidate(r.value.value if r.status == "ok" else None, r.location) for r in reads)


def _unique(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(w for group in groups for w in group))


# --- combination (pure; §5.3, §5.4) -------------------------------------------
def _combine_variants(job: _Job, per_doc: list[tuple[ReportDoc, Outcome]]) -> Outcome:
    if len(per_doc) == 1:
        return per_doc[0][1]
    by_id = sorted(per_doc, key=lambda p: p[0].doc_id)
    found = [(doc, o) for doc, o in per_doc if isinstance(o, Found)]
    if not found:
        other = next((o for _, o in by_id if not isinstance(o, NotFound)), None)
        return other if other is not None else _merge_not_found([o for _, o in by_id])
    facts = [o.extracted.fact for _, o in found]
    if not _agree(facts):
        return Conflict(**job.keys, candidates=tuple(Candidate(f.value.value, _location(f)) for f in facts))
    _, kept = min(found, key=lambda p: (
        p[1].extracted.fact.source.scale, p[0].scope_variant is not None, p[0].scope_variant or 0))
    warnings = []
    for doc, o in by_id:
        if isinstance(o, NotFound):
            warnings.append(f"not_in_variant:{doc.doc_id}")
        elif not isinstance(o, Found):
            warnings.append(f"variant_outcome:{doc.doc_id}:{_VARIANT_KIND[type(o)]}")
    return _with_warnings(kept, warnings)


def _combine_sources(primary: Outcome, fallback: Outcome) -> Outcome:
    """§5.3 table; called only when the primary outcome is ``ReportMissing`` or ``NotFound``."""
    if isinstance(fallback, ReportMissing):
        return primary
    if isinstance(primary, NotFound):
        return _merge_not_found([primary, fallback]) if isinstance(fallback, NotFound) else fallback
    return _with_warnings(fallback, ("primary_report_missing",))


def _agree(facts: Sequence[Fact]) -> bool:
    """Every pair within the larger of its two printed scales (as ``FactSet``)."""
    return all(
        abs(a.value.value - b.value.value) <= max(a.source.scale, b.source.scale)
        for i, a in enumerate(facts)
        for b in facts[i + 1:]
    )


def _location(fact: Fact) -> CellLocation:
    s = fact.source
    return CellLocation(s.chunk_id, s.doc_id, s.page, s.row_label, s.column_header or "", s.raw_text)


def _merge_not_found(outcomes: Sequence[NotFound]) -> NotFound:
    """The first outcome's reason; doc ids, skipped tables and warnings of all, first-seen order."""
    return replace(
        outcomes[0],
        searched_doc_ids=_unique(*(o.searched_doc_ids for o in outcomes)),
        skipped_tables=_unique(*(o.skipped_tables for o in outcomes)),
        warnings=_unique(*(o.warnings for o in outcomes)),
    )


def _with_warnings(outcome: Outcome, warnings: Iterable[str]) -> Outcome:
    if isinstance(outcome, Found):
        extracted = outcome.extracted
        return replace(outcome, extracted=replace(extracted, warnings=_unique(extracted.warnings, warnings)))
    if isinstance(outcome, NotFound):
        return replace(outcome, warnings=_unique(outcome.warnings, warnings))
    return outcome

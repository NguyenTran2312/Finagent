# Retrieval-Guided Leaf Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the existing table retriever (BM25 + BGE-M3 dense, RRF) in front of the rule-based leaf extractor: for each request the extractor reads only the top-10 retrieved tables of the report and falls back to the whole report when they hold no row, with provenance on every fact and tools to run the evaluation gate.

**Architecture:** `src/extraction/retrieval.py` defines a standard-library `TableRetriever` protocol and trace types; `LeafExtractor(..., retriever=None)` calls it once per (report year, leaf), reads each document first through the ranked subset (step R) and then in full (step F) on `NotFound`. The adapter `src/retrieval/table_retriever.py` builds the V3 query, applies ticker / year / scope filters, runs `retrieve_refined` and Chroma `similarity_search`, and fuses them with `rrf_fuse`. Three manual scripts measure the change before it becomes a default.

**Tech Stack:** Python 3.14, stdlib, unittest; existing langchain BM25 (`rank-bm25`), `langchain_chroma` / `chromadb`, `langchain_huggingface` (BGE-M3), torch. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-12-retrieval-guided-extraction-design.md` (read it with this plan; § numbers below refer to it). Base design: `docs/superpowers/specs/2026-11-11-leaf-extraction-design.md`.

## Global Constraints

- Branch `feat/retrieval-extraction` in worktree `D:\finance_project\.worktrees\retrieval-extraction` (at 9f1468a). Commit per task on this branch only. Never push, merge or open a PR without explicit user authorization.
- Project rules: untracked `D:\finance_project\CLAUDE.md`, `src/CLAUDE.md` and package `CLAUDE.md` files; read them from `D:\finance_project`.
- `src/extraction` must not import retrieval, embedding, torch, chromadb, langchain or network code (§3.1). Only `src/retrieval/table_retriever.py` holds those, and imports torch / chromadb / langchain_chroma / embeddings lazily inside `open_table_retriever`.
- `retriever=None` keeps today's behaviour exactly: the existing 572 tests stay green unchanged.
- Unchanged (§3.1): `find_rows`, `TableView`, row matchers, unit / period rules, `ReportStore`, `retrieve_refined`, `rrf_fuse`, `hybrid.py`, the index builder, both index formats.
- Fixed values: `k = 10`, `depth = 20`, collection `vifinqa_tables`, RRF `c = RRF_CONSTANT` (60), modes `hybrid` | `bm25`, steps `retrieved` | `full_scan`.
- Warning strings, exact: `retrieval_error:<ExceptionClass>`, `retrieval_unknown_chunks:<count>`.
- Statement names appended to the query, exact: `balance_sheet` → `bảng cân đối kế toán`; `income_statement` → `báo cáo kết quả hoạt động kinh doanh`; `cash_flow` → `báo cáo lưu chuyển tiền tệ`.
- Unit tests: offline, deterministic; no model, GPU, network, real index or corpus. The corpus `D:/finance_project/data/databases/corpus.sqlite3` is opened only by the manual scripts, only read-only (`mode=ro`).
- No real LLM calls unless the user asks (they cost money). The gold `calculation_program` is never executed.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Test command (PowerShell, worktree root):
  `$env:PYTHONPATH="$PWD;$PWD\src"; $env:TEMP="$PWD\.tmp-tests"; $env:TMP=$env:TEMP; D:\finance_project\.venv\Scripts\python.exe -B -m unittest <target> -v`
  Full suite: `<target>` = `discover -s tests`; also `python -m compileall -q src config scripts`.

## Plan decisions (resolve spec gaps; binding for all tasks)

- P1. **No retrieval when a source has no document.** `rank` is called only after `call.documents(Y, scope)` returned at least one document; a `ReportMissing` source costs no query.
- P2. **Retrieval warnings attach to the source outcome** (after `_combine_variants`, via `_with_warnings`). They then travel through `_combine_sources` like any other `NotFound` / `Found` warnings today: a primary `NotFound`'s warnings are merged into a final `NotFound`, and dropped when the next-year source gives `Found` (as today's primary warnings are).
- P3. **"Unknown chunk" means: not a chunk of any table record of the source's documents**, including records that `TableView` skipped as malformed. `_Doc` gains `chunk_ids: frozenset[str]` (every non-None `record.chunk_id` loaded for the document).
- P4. **Ranking validation lives in `src/extraction/retrieval.py`** as `ranking_of(answer) -> dict[str, int]` (chunk_id → rank); the extractor calls it on every answer.
- P5. **The trace is set where the `Found` is produced** (`_read_document` for steps R/F, `_locate` for the LLM), with `dataclasses.replace` on `ExtractedFact`; variant and source combination keep whatever trace the kept `Found` carries.
- P6. **Adapter input errors vs backend errors.** `HybridTableRetriever.rank` builds the query and validates the filters (`normalize_metadata_filters`) before touching a backend, so their `ValueError` propagates; every exception raised inside the BM25 / Chroma calls is re-raised as `TableRetrievalError(...) from exc`.
- P7. **Script CLI options are shared**: `add_retriever_arguments(parser)` and `retriever_from_args(args, db_path)` live in `src/retrieval/table_retriever.py`. Scripts put both `<repo>` and `<repo>/src` on `sys.path` (the pickled BM25 references `preprocess.cleaner.tokens`).
- P8. **`seconds` in the coverage CSV** = wall time of the `extract` call that produced the row ÷ its number of requests (one call reads one report for all leaves), 4 decimals.
- P9. **Diff of runs with different request sets**: a key present in only one CSV is written with label `other` and outcome `absent` on the missing side; the script does not abort.

## Review Focus

- A script that loads the real BM25 pickle without `<repo>/src` on `sys.path` dies with `ModuleNotFoundError: preprocess` → Task 6 smoke run (one report, `--retriever bm25`) before any long run; P7.
- A ranked chunk of a malformed (skipped) table must not be reported as `retrieval_unknown_chunks` → Task 3 `test_ranked_malformed_table_is_not_unknown`.
- A year with no document must not trigger a retriever call (latency over 53k requests) → Task 2 `test_report_missing_year_is_not_retrieved`.
- `hybrid` with `fp16=False`, `device="cpu"` or no `chroma_dir` must fail before the 1 GB BM25 pickle is loaded → Task 5 `test_bad_arguments_fail_before_loading`.
- Two coverage CSVs with different request sets (a report that failed in one run) must still diff → Task 6 `test_key_in_one_run_only_is_other`.

---

### Task 1: Retrieval protocol, trace types and answer validation

**Files:**
- Create: `src/extraction/retrieval.py`
- Modify: `src/extraction/outcomes.py` (`ExtractedFact`)
- Test: `tests/test_extraction_retrieval.py` (new; class `RankingValidationTests`)

**Interfaces:**
- Produces (in `src/extraction/retrieval.py`, standard library only at runtime; `Scope` and `LeafDef` are imported under `typing.TYPE_CHECKING`):
  - `RankedTable(chunk_id: str, rank: int)` frozen dataclass.
  - `TableRetriever` Protocol: attribute `mode: str`; `rank(ticker: str, scope: Scope, report_year: int, leaf: LeafDef) -> tuple[RankedTable, ...]`.
  - `class TableRetrievalError(RuntimeError)`.
  - `RetrievalTrace(mode: str, step: Literal["retrieved", "full_scan"], rank: int | None)` frozen dataclass.
  - `ranking_of(answer: object) -> dict[str, int]` — chunk_id → rank; raises `ValueError` unless `answer` is a `tuple` whose items are all `RankedTable`, ranks are exactly `1..len(answer)` (any order) and chunk_ids are distinct. `()` → `{}`.
- Produces (in `outcomes.py`): `ExtractedFact.retrieval: RetrievalTrace | None = None` (last field, default None). `outcomes.py` imports `RetrievalTrace` from `src.extraction.retrieval`; `retrieval.py` must not import `outcomes.py`.

- [ ] **Step 1: Write the failing tests**

```python
class RankingValidationTests(unittest.TestCase):
    def test_valid_answer_maps_chunks_to_ranks(self):
        answer = (RankedTable("D:table:2", 2), RankedTable("D:table:1", 1))
        self.assertEqual(ranking_of(answer), {"D:table:2": 2, "D:table:1": 1})
        self.assertEqual(ranking_of(()), {})

    def test_invalid_answers_raise(self):
        bad = {
            "list": [RankedTable("a", 1)],
            "rank_zero": (RankedTable("a", 0),),
            "rank_gap": (RankedTable("a", 1), RankedTable("b", 3)),
            "duplicate_rank": (RankedTable("a", 1), RankedTable("b", 1)),
            "duplicate_chunk": (RankedTable("a", 1), RankedTable("a", 2)),
            "not_ranked_table": (("a", 1),),
        }
        for name, answer in bad.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    ranking_of(answer)

    def test_extracted_fact_has_no_trace_by_default(self):
        self.assertIsNone(dataclasses.fields(ExtractedFact)[-1].default)
        self.assertEqual(dataclasses.fields(ExtractedFact)[-1].name, "retrieval")
```

- [ ] **Step 2: Run to verify failure** — `<test command> tests.test_extraction_retrieval -v` → FAIL (`ModuleNotFoundError: src.extraction.retrieval`).
- [ ] **Step 3: Implement** `src/extraction/retrieval.py` and the `ExtractedFact` field as in Interfaces.
- [ ] **Step 4: Run** `tests.test_extraction_retrieval` → PASS; full suite → 572 + 3 OK.
- [ ] **Step 5: Commit** `feat(extraction): table retriever protocol and retrieval trace`.

---

### Task 2: Extractor steps R → F, per-source cache and trace

**Files:**
- Modify: `src/extraction/extractor.py` (`LeafExtractor.__init__`, `_Call`, `_Doc`, `_source`, `_in_document`, new `_read_document`)
- Test: `tests/test_extraction_retrieval.py` (class `RetrievalStepTests`)

**Interfaces:**
- Consumes: Task 1 types and `ranking_of`.
- Produces:
  - `LeafExtractor(store, tree, templates, fallback: CellLocator | None = None, retriever: TableRetriever | None = None)`.
  - `_Call(session, ticker, retriever: TableRetriever | None = None)` with `ranking(year: int, scope: Scope, leaf: LeafDef, docs: tuple[_Doc, ...]) -> _Ranking | None` — `None` when no retriever; otherwise cached per `(year, leaf.id)`.
  - `_Ranking(ranks: dict[str, int] | None, warnings: tuple[str, ...])` frozen dataclass; `ranks is None` means "retrieval failed, full scan" (Task 3 fills that path; in this task `ranks` is always set and `warnings` empty).
  - `_Doc.chunk_ids: frozenset[str]` (P3).
  - `_in_document(job, d, role, views: Sequence[TableView])` — `find_rows` runs on `views` instead of `d.views`; nothing else changes.
  - `_read_document(job, d, role, ranking: _Ranking | None, mode: str | None) -> Outcome` implementing §4.2 steps R and F; `_source` uses it for every document and applies `ranking.warnings` via `_with_warnings` to the combined outcome (P2).

Algorithm of `_read_document` (§4.2; the one part signatures do not fix):

```
if ranking is None: return _in_document(job, d, role, d.views)              # no retriever: today
if ranking.ranks:                                                           # step R
    views_r = [v for v in d.views if v.record.chunk_id in ranking.ranks]
    if views_r:
        o = _in_document(job, d, role, views_r)
        if not isinstance(o, NotFound):
            return _traced(o, mode, "retrieved", ranking.ranks[source chunk_id of o] if Found)
o = _in_document(job, d, role, d.views)                                     # step F
return _traced(o, mode, "full_scan", None)
```

`_traced` sets `ExtractedFact.retrieval` on a `Found` and returns any other outcome unchanged.

Test fixtures: import `is_rows`, `is_table`, `doc`, `store_of`, `tree`, `templates`, `INTEREST_2023` from `tests.test_extraction_extractor` (functions and constants only — never a `TestCase` class) and `record` from `tests.extraction_fixtures`. A fake:

```python
class FakeRetriever:
    mode = "fake"
    def __init__(self, answers=None, error=None):
        self.answers, self.error, self.calls = answers or {}, error, []
    def rank(self, ticker, scope, report_year, leaf):
        self.calls.append((ticker, scope, report_year, leaf.id))
        if self.error is not None:
            raise self.error
        return self.answers.get((report_year, leaf.id), ())

def ranked(*chunk_ids):
    return tuple(RankedTable(c, i) for i, c in enumerate(chunk_ids, 1))
```

Base store `VJC23C` (2023, consolidated): table 1 = `is_table("VJC23C", is_rows(2023))` (interest row, value `-1950254195864`), table 2 = `record("VJC23C", 2, 30, <a note table without interest row>, kind="note")`. Expected value `D("-1950254195864")`.

Test helpers (module level in the test file):
- `base_store()` → `store_of((doc("VJC23C", 2023), [table 1, table 2]))`.
- `extract_with(retriever, requests, store=None, fallback=None) -> Outcome` → first outcome of `LeafExtractor(store or base_store(), tree, templates, fallback=fallback, retriever=retriever).extract("VJC", "consolidated", requests)`.
- `one_with(retriever, store=None, request=INTEREST_2023, fallback=None) -> Outcome` → `extract_with(retriever, [request], store, fallback)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_value_in_ranked_table_is_retrieved(self):
    r = FakeRetriever({(2023, "interest_expense"): ranked("VJC23C:table:2", "VJC23C:table:1")})
    out = one_with(r)                       # extract VJC consolidated [INTEREST_2023]
    self.assertEqual(out.extracted.fact.value.value, D("-1950254195864"))
    self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 2))

def test_row_outside_ranked_tables_falls_back_to_full_scan(self):
    r = FakeRetriever({(2023, "interest_expense"): ranked("VJC23C:table:2")})
    self.assertEqual(one_with(r).extracted.retrieval, RetrievalTrace("fake", "full_scan", None))

def test_blank_cell_in_ranked_table_is_final(self):
    # table 1 interest cell blank (is_rows(2023, current="")), table 3 = an unranked IS table with a value
    # ranked: table 1 only  ->  BlankValue (no step F)
    self.assertIsInstance(out, BlankValue)

def test_empty_answer_reads_full_report(self):
    out = one_with(FakeRetriever())
    self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "full_scan", None))

def test_r1_hidden_conflict_is_accepted_risk(self):
    # table 3: second income_statement table, same code 23, value "(5.000.000.000)"
    self.assertIsInstance(one(store_r1), Ambiguous)                                   # no retriever
    out = one_with(FakeRetriever({(2023, "interest_expense"): ranked("VJC23C:table:3")}), store_r1)
    self.assertEqual(out.extracted.fact.value.value, D("-5000000000"))
    self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 1))

def test_next_year_source_retrieves_with_its_own_year(self):
    # VJC22C (2022) has no interest row (is_rows(2022, interest=False)); VJC23C as base; request 2022
    r = FakeRetriever()
    out = extract_with(r, [LeafRequest("interest_expense", 2022, "flow")])
    self.assertIsInstance(out, Found)
    self.assertEqual([c[2] for c in r.calls], [2022, 2023])

def test_one_rank_call_per_year_and_leaf(self):
    # two 2023 variants (VJC23C variant None, VJC23C1 variant 1, same tables) + VJC22C without the row;
    # requests 2022 and 2023: 2023 is the fallback of 2022 and the primary of 2023
    self.assertEqual(sorted(r.calls), [("VJC", "consolidated", 2022, "interest_expense"),
                                       ("VJC", "consolidated", 2023, "interest_expense")])

def test_variant_without_ranked_table_is_read_in_full(self):
    # ranked: only VJC23C:table:1; variant VJC23C1 has the same row in VJC23C1:table:1 (unranked)
    self.assertIsInstance(out, Found)
    self.assertFalse(any(w.startswith("not_in_variant") for w in out.extracted.warnings))

def test_report_missing_year_is_not_retrieved(self):
    # store with VJC23C only; request 2021 -> 2021 and 2022 have no document
    self.assertIsInstance(out, ReportMissing)
    self.assertEqual(r.calls, [])

def test_no_retriever_keeps_outcomes_and_has_no_trace(self):
    without = extract(...)                    # LeafExtractor(store, tree, templates)
    with_empty = extract(..., retriever=FakeRetriever())
    for a, b in zip(without.outcomes, with_empty.outcomes):
        if isinstance(a, Found):
            self.assertIsNone(a.extracted.retrieval)
            self.assertEqual(a, dataclasses.replace(b, extracted=dataclasses.replace(b.extracted, retrieval=None)))
        else:
            self.assertEqual(a, b)
    # run over default_store() with INTEREST_2023, NET_REVENUE_2023 and an ACB bank request
```

- [ ] **Step 2: Run** `tests.test_extraction_retrieval` → the new tests FAIL (`unexpected keyword argument 'retriever'`).
- [ ] **Step 3: Implement** the Interfaces above in `extractor.py`. `ranking` calls `ranking_of(retriever.rank(call.ticker, scope, year, leaf))`. `_source` calls `job.call.ranking(...)` only when `docs` is non-empty (P1).
- [ ] **Step 4: Run** `tests.test_extraction_retrieval` and the full suite → all PASS (existing extractor tests unchanged).
- [ ] **Step 5: Commit** `feat(extraction): read retrieved tables first, full report on miss`.

---

### Task 3: Retriever failures, unknown chunks and the LLM fallback views

**Files:**
- Modify: `src/extraction/extractor.py` (`_Call.ranking`, `_locate`)
- Test: `tests/test_extraction_retrieval.py` (classes `RetrievalFailureTests`, `RetrievalLocatorTests`)

**Interfaces:**
- Consumes: Task 2 `_Ranking`, `_Call.ranking`, `_Doc.chunk_ids`, `_traced`.
- Produces:
  - `_Call.ranking`: `TableRetrievalError` or `OSError` from `rank` → `_Ranking(None, (f"retrieval_error:{type(exc).__name__}",))`; any other exception propagates; `ranking_of`'s `ValueError` propagates. Valid answer: chunk_ids not in the union of `d.chunk_ids` over the source's docs are dropped from `ranks` and, when any were, `warnings = (f"retrieval_unknown_chunks:{n}",)`.
  - `_locate(job, docs, role, deterministic)`: for each document passes `views_r` (as in Task 2) when non-empty, else `d.views`. The ranking comes from `job.call.ranking(year, job.scope, job.leaf, docs)` (the same `docs`; a cache hit) with `year = request.year` for `primary`, `request.year + 1` for `fallback` (cached, no new call). A `Found` gets `retrieved` + `ranks[location.chunk_id]` when that chunk is ranked, else `full_scan`; no trace when no retriever.

- [ ] **Step 1: Write the failing tests**

```python
def test_retrieval_error_degrades_to_full_scan(self):
    for error, code in ((TableRetrievalError("x"), "TableRetrievalError"),
                        (FileNotFoundError("x"), "FileNotFoundError")):
        with self.subTest(code):
            out = one_with(FakeRetriever(error=error))
            self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "full_scan", None))
            self.assertIn(f"retrieval_error:{code}", out.extracted.warnings)

def test_programming_error_from_retriever_propagates(self):
    with self.assertRaises(KeyError):
        one_with(FakeRetriever(error=KeyError("x")))

def test_invalid_answer_raises(self):
    with self.assertRaises(ValueError):
        one_with(FakeRetriever({(2023, "interest_expense"): (RankedTable("VJC23C:table:1", 2),)}))

def test_unknown_chunk_is_ignored_with_warning(self):
    out = one_with(FakeRetriever({(2023, "interest_expense"): ranked("XYZ:table:9", "VJC23C:table:1")}))
    self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 2))
    self.assertIn("retrieval_unknown_chunks:1", out.extracted.warnings)

def test_ranked_malformed_table_is_not_unknown(self):
    # table 2 = is_table(...) with rows_json="{not json"; ranked: table 2, table 1
    self.assertFalse(any(w.startswith("retrieval_unknown_chunks") for w in out.extracted.warnings))

def test_not_found_carries_retrieval_warnings(self):
    # store where no table has the row; retriever raises TableRetrievalError
    self.assertIsInstance(out, NotFound)
    self.assertIn("retrieval_error:TableRetrievalError", out.warnings)

def test_locator_receives_only_ranked_views(self):
    # no deterministic row anywhere (is_rows(2023, interest=False)); StubLocator records views
    # ranked VJC23C:table:2 -> locator saw [table 2];  empty answer -> locator saw tables 1 and 2

def test_locator_found_in_ranked_table_is_retrieved(self):
    # StubLocator(read_label(<label of table 2's row>)) with ranked("VJC23C:table:2")
    self.assertEqual(out.extracted.method, "llm_located")
    self.assertEqual(out.extracted.retrieval, RetrievalTrace("fake", "retrieved", 1))
```

(`StubLocator` / `read_label` from `tests.test_extraction_extractor`; extend `StubLocator` locally to also record `[v.record.chunk_id for v in views]`.)

- [ ] **Step 2: Run** → new tests FAIL.
- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run** `tests.test_extraction_retrieval`, `tests.test_extraction_extractor`, `tests.test_extraction_llm_fallback`, then the full suite → PASS.
- [ ] **Step 5: Commit** `feat(extraction): retrieval failures, stale chunks and locator views`.

---

### Task 4: Adapter `HybridTableRetriever`

**Files:**
- Create: `src/retrieval/table_retriever.py`
- Test: `tests/test_table_retriever.py` (class `HybridTableRetrieverTests`)

**Interfaces:**
- Consumes: `retrieve_refined` (`src.main.refine_questions`), `rrf_fuse` (`src.retrieval.hybrid`), `normalize_metadata_filters` (`src.retrieval.filters`), `RRF_CONSTANT` (`config.settings`), Task 1 types.
- Produces:
  - `STATEMENT_NAMES: dict[StatementKind, str]` (the three Global-Constraint strings).
  - `table_query(leaf: LeafDef) -> str` — `leaf.label_vi`, plus `" " + STATEMENT_NAMES[kind]` when `leaf.source == "statement"` and the kind is in the dict.
  - `HybridTableRetriever(bm25, dense_store, *, mode: str = "hybrid", k: int = 10, depth: int = 20)`; `ValueError` when `mode` not in {`hybrid`, `bm25`}, `k < 1`, `depth < k` (non-int counts as bad), or `mode == "hybrid"` and `dense_store is None`. Attributes `mode`, `k`, `depth`.
  - `rank(ticker, scope, report_year, leaf) -> tuple[RankedTable, ...]`:
    - filters `{"tickers": [ticker.strip().upper()], "years": [report_year], "scopes": [scope]}` validated by `normalize_metadata_filters` before any backend call (P6);
    - Chroma where: `{"$and": [{"ticker": {"$in": [T]}}, {"report_year": {"$in": [Y]}}, {"scope": {"$in": scopes}}]}` with `scopes = ["separate", "parent"]` for `separate`, else `[scope]`;
    - `bm25`: `retrieve_refined(bm25, query, k=self.k, metadata_filters=filters)["documents"]`;
    - `hybrid`: BM25 with `k=self.depth`, `dense_store.similarity_search(query, k=self.depth, filter=where)`, then `rrf_fuse(lexical, dense, k=self.k, c=RRF_CONSTANT)`;
    - result `RankedTable(doc.metadata["chunk_id"], i)` for i from 1; exceptions inside the backend calls → `TableRetrievalError(f"{mode} retrieval failed: {type(exc).__name__}") from exc`.
- No torch / chromadb / langchain_huggingface import at module level (Task 5 imports them lazily).

Test fixtures: a tiny BM25 from `build_bm25` (`src.retrieval.bm25`) over ~6 `Document`s with metadata `{chunk_id, doc_id, table_id, ticker, report_year, scope}` (two tickers, two years, scopes consolidated / separate / parent) and Vietnamese text; `FakeDenseStore` recording `(query, k, filter)` and returning a fixed list or raising a given error. Leaves from `MetricTree.load().leaf(...)`: a balance-sheet, an income-statement, a cash-flow and a `notes` leaf.

- [ ] **Step 1: Write the failing tests**

```python
def test_query_appends_statement_name(self):
    self.assertEqual(table_query(leaf_is), f"{leaf_is.label_vi} báo cáo kết quả hoạt động kinh doanh")
    self.assertEqual(table_query(leaf_bs), f"{leaf_bs.label_vi} bảng cân đối kế toán")
    self.assertEqual(table_query(leaf_cf), f"{leaf_cf.label_vi} báo cáo lưu chuyển tiền tệ")
    self.assertEqual(table_query(leaf_note), leaf_note.label_vi)

def test_bm25_mode_filters_ranks_and_truncates(self):
    r = HybridTableRetriever(bm25, None, mode="bm25", k=2, depth=2)
    out = r.rank("vjc", "consolidated", 2023, leaf_is)
    self.assertEqual([t.rank for t in out], [1, 2])
    self.assertTrue(all(t.chunk_id.startswith("VJC23C:") for t in out))   # ticker upper-cased, year/scope filtered

def test_separate_scope_also_accepts_parent(self):
    store = FakeDenseStore([])
    HybridTableRetriever(bm25, store, k=2, depth=3).rank("VJC", "separate", 2023, leaf_is)
    query, k, where = store.calls[0]
    self.assertEqual(k, 3)
    self.assertIn({"scope": {"$in": ["separate", "parent"]}}, where["$and"])
    self.assertIn({"ticker": {"$in": ["VJC"]}}, where["$and"])
    self.assertIn({"report_year": {"$in": [2023]}}, where["$and"])
    # BM25 side: a parent-scope document of VJC 2023 can appear in the bm25-mode answer for "separate"

def test_hybrid_fuses_both_branches(self):
    # dense returns [doc B, doc A]; expected order equals [d.metadata["chunk_id"] for d in rrf_fuse(bm25_docs, [B, A], k=2, c=60)]

def test_backend_error_is_wrapped(self):
    store = FakeDenseStore(error=RuntimeError("chroma down"))
    with self.assertRaises(TableRetrievalError) as caught:
        HybridTableRetriever(bm25, store).rank("VJC", "consolidated", 2023, leaf_is)
    self.assertIsInstance(caught.exception.__cause__, RuntimeError)

def test_bad_scope_is_input_error(self):
    with self.assertRaises(ValueError) as caught:
        HybridTableRetriever(bm25, None, mode="bm25").rank("VJC", "nonsense", 2023, leaf_is)
    self.assertNotIsInstance(caught.exception, TableRetrievalError)

def test_constructor_validates(self):
    for kwargs in ({"mode": "dense"}, {"k": 0}, {"k": 10, "depth": 5}, {"mode": "hybrid"}):
        with self.subTest(**kwargs), self.assertRaises(ValueError):
            HybridTableRetriever(bm25, None, **kwargs)
```

- [ ] **Step 2: Run** `tests.test_table_retriever` → FAIL (module missing).
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** `tests.test_table_retriever` → PASS.
- [ ] **Step 5: Commit** `feat(retrieval): table retriever adapter for leaf extraction`.

---

### Task 5: `open_table_retriever` start-up checks and shared CLI options

**Files:**
- Modify: `src/retrieval/table_retriever.py`
- Test: `tests/test_table_retriever.py` (class `OpenTableRetrieverTests`)

**Interfaces:**
- Produces:
  - `open_table_retriever(db_path, bm25_dir, *, bm25_name="bm25-table", chroma_dir=None, mode="hybrid", device="cuda", fp16=True, k=10, depth=20) -> HybridTableRetriever`. Order (§5.4), each failure raises before the next:
    1. arguments: mode / k / depth as in Task 4; `hybrid` additionally needs `chroma_dir` (`ValueError`), `fp16 is True` (`ValueError`, message names the fp16 index) and `str(device).startswith("cuda")` (`ValueError`);
    2. `load_bm25(bm25_dir, bm25_name, expected_identity=sqlite_source_identity(db_path, chunk_types=("table",)))`;
    3. hybrid: `(Path(chroma_dir) / "chroma.sqlite3").is_file()` else `FileNotFoundError`; `client = _chroma_client(chroma_dir)`; `client.get_collection("vifinqa_tables").count() == len(bm25.docs)` else `ValueError` naming both counts;
    4. hybrid: `embeddings = _embeddings(device, fp16)` (wraps `get_embeddings(device=..., fp16=...)`), then `Chroma(client=client, collection_name="vifinqa_tables", embedding_function=embeddings, create_collection_if_not_exists=False)`.
  - Module-private seams patched by tests: `_chroma_client(path) -> client` (wraps `chromadb.PersistentClient(path=str(path))`), `_embeddings(device, fp16)`, `_chroma_store(client, embeddings)`.
  - `add_retriever_arguments(parser)` adds `--retriever {none,bm25,hybrid}` (default `none`), `--bm25-dir` (default `<repo>/data/indexes/bm25`), `--bm25-name` (default `bm25-table`), `--chroma-dir` (no default), `--device` (default `cuda`).
  - `retriever_from_args(args, db_path) -> HybridTableRetriever | None` — `None` for `none`, else `open_table_retriever(db_path, args.bm25_dir, bm25_name=args.bm25_name, chroma_dir=args.chroma_dir, mode=args.retriever, device=args.device)`.

Test fixtures: a temporary directory (`tempfile.TemporaryDirectory`) with a small file standing in for the SQLite corpus and a tiny BM25 saved by `save_bm25(..., identity=sqlite_source_identity(that_file, chunk_types=("table",)))`.

- [ ] **Step 1: Write the failing tests**

```python
def test_bm25_mode_opens_saved_index(self):
    r = open_table_retriever(db, idx_dir, bm25_name="bm25-table", mode="bm25")
    self.assertEqual((r.mode, r.k, r.depth), ("bm25", 10, 20))

def test_identity_mismatch_fails(self):
    db.write_bytes(b"changed corpus")          # size / mtime change
    with self.assertRaises(ValueError):
        open_table_retriever(db, idx_dir, mode="bm25")

def test_chroma_count_must_match_bm25(self):
    (chroma / "chroma.sqlite3").write_bytes(b"")
    client = FakeClient(count=len(bm25.docs) + 1)
    with patch.object(table_retriever, "_chroma_client", return_value=client), \
         patch.object(table_retriever, "_embeddings") as emb:
        with self.assertRaises(ValueError):
            open_table_retriever(db, idx_dir, chroma_dir=chroma)
    emb.assert_not_called()

def test_hybrid_opens_with_fake_backends(self):
    # count equal; _embeddings and _chroma_store patched -> r.mode == "hybrid"; _embeddings called with ("cuda", True)

def test_bad_arguments_fail_before_loading(self):
    with patch.object(table_retriever, "load_bm25") as load:
        for kwargs in ({"mode": "dense"}, {"k": 0}, {"depth": 5}, {"chroma_dir": None},
                       {"chroma_dir": chroma, "fp16": False}, {"chroma_dir": chroma, "device": "cpu"}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                open_table_retriever(db, idx_dir, **{"chroma_dir": chroma, **kwargs})
        load.assert_not_called()

def test_cli_none_builds_no_retriever(self):
    parser = argparse.ArgumentParser(); add_retriever_arguments(parser)
    self.assertIsNone(retriever_from_args(parser.parse_args([]), db))
```

(`load_bm25` and `sqlite_source_identity` are imported into `table_retriever`'s namespace so `patch.object` works.)

- [ ] **Step 2: Run** `tests.test_table_retriever` → new tests FAIL.
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** the full suite and `python -m compileall -q src config scripts` → PASS; also `python -B -c "import src.extraction.extractor, sys; assert not any(m.split('.')[0] in ('torch','chromadb','langchain_chroma','langchain_huggingface') for m in sys.modules)"` → exit 0.
- [ ] **Step 5: Commit** `feat(retrieval): open table retriever with start-up index checks`.

---

### Task 6: Coverage script with `--retriever` and run diff

**Files:**
- Modify: `scripts/leaf_coverage.py`
- Create: `scripts/diff_extraction_runs.py`
- Test: `tests/test_diff_extraction_runs.py` (loads the script by path with `importlib.util.spec_from_file_location`)

**Interfaces:**
- Consumes: Task 5 `add_retriever_arguments`, `retriever_from_args`; `ExtractedFact.retrieval`.
- `leaf_coverage.py`:
  - `sys.path` gains `<repo>` and `<repo>/src` (P7); `add_retriever_arguments(parser)`; `LeafExtractor(store, tree, templates, fallback=None, retriever=retriever_from_args(args, args.db))`.
  - `CSV_FIELDS` appends `value`, `chunk_id`, `retrieval_step`, `retrieval_rank`, `seconds` (empty strings when not a `Found` / no trace; `value` = `str(fact.value.value)`; `seconds` per P8).
  - Report header prints `retriever=<none|bm25|hybrid>`; after the per-template tables print `Retrieval steps: retrieved N, full_scan M` and the `retrieval_error` / `retrieval_unknown_chunks` warning totals (already counted by warning code).
  - Docstring updated with the new options and columns.
- `diff_extraction_runs.py BASE.csv NEW.csv [--out DIFF.csv]` (default `--out` = `NEW` stem + `.diff.csv`):
  - key `(template, ticker, year, scope, leaf)`; `label(base: dict | None, new: dict | None) -> str | None`, first rule that applies:
    1. one side missing → `other` (P9);
    2. same `outcome`, `value`, `chunk_id` → `None` (unchanged, not written);
    3. both `Found`, value differs → `value_changed`;
    4. both `Found`, chunk differs → `source_changed`;
    5. base `Ambiguous`, new `Found` → `ambiguous_to_found`;
    6. base `Found` → `found_to_other`;
    7. new `Found` → `other_to_found`;
    8. → `other`.
  - Output columns: `label, template, ticker, year, scope, leaf, base_outcome, new_outcome, base_value, new_value, base_chunk_id, new_chunk_id, new_retrieval_step, new_retrieval_rank`.
  - Prints per-label counts, NEW's `retrieval_step` counts and `retrieval_rank` histogram (1, 2–5, 6–10), and p50 / p95 of `seconds` for each file (skipping blank values).

- [ ] **Step 1: Write the failing tests** (`tests/test_diff_extraction_runs.py`)

```python
def row(outcome="Found", value="1", chunk="D:table:1"):
    return {"outcome": outcome, "value": value, "chunk_id": chunk}

def test_labels(self):
    cases = [
        (row(), row(), None),
        (row(), row(value="2"), "value_changed"),
        (row(), row(chunk="D:table:2"), "source_changed"),
        (row("Ambiguous", "", ""), row(), "ambiguous_to_found"),
        (row(), row("NotFound", "", ""), "found_to_other"),
        (row("NotFound", "", ""), row(), "other_to_found"),
        (row("NotFound", "", ""), row("Ambiguous", "", ""), "other"),
    ]
    for base, new, expected in cases:
        with self.subTest(expected=expected):
            self.assertEqual(diff.label(base, new), expected)

def test_key_in_one_run_only_is_other(self):
    self.assertEqual(diff.label(None, row()), "other")
    self.assertEqual(diff.label(row(), None), "other")
```

- [ ] **Step 2: Run** `tests.test_diff_extraction_runs` → FAIL (script missing).
- [ ] **Step 3: Implement** both scripts as in Interfaces.
- [ ] **Step 4: Run** the test → PASS; then the smoke runs (PowerShell, worktree root; indexes from the main checkout):

```
D:\finance_project\.venv\Scripts\python.exe -I -X utf8 scripts/leaf_coverage.py --db D:/finance_project/data/databases/corpus.sqlite3 --years 2023 --leaf interest_expense --retriever none --out .tmp-tests/smoke_none.csv
D:\finance_project\.venv\Scripts\python.exe -I -X utf8 scripts/leaf_coverage.py --db D:/finance_project/data/databases/corpus.sqlite3 --years 2023 --leaf interest_expense --retriever bm25 --bm25-dir D:/finance_project/data/indexes/bm25 --out .tmp-tests/smoke_bm25.csv
D:\finance_project\.venv\Scripts\python.exe -I -X utf8 scripts/diff_extraction_runs.py .tmp-tests/smoke_none.csv .tmp-tests/smoke_bm25.csv
```

Expected: both runs exit 0 with no `FAILED` line; the bm25 CSV has `retrieval_step` filled on every `Found`; the diff prints label counts.
- [ ] **Step 5: Commit** `feat(scripts): coverage runs with a table retriever and run diff`.

---

### Task 7: Gold operand scoring script

**Files:**
- Create: `scripts/score_gold_operands.py` (port of `C:/Users/ADMIN/AppData/Local/Temp/claude/7f48740d5b6406bb/52870aad-3464-486a-9f68-df681f98e476/scratchpad/score_gold.py`)

**Interfaces:**
- Consumes: Task 5 CLI helpers; `ExtractedFact.retrieval`.
- `score_gold_operands.py GOLD_JSONL OUT_CSV [--db PATH] [--llm MODULE] <retriever options>`:
  - argparse instead of `sys.argv`; no hard-coded worktree or DB path (`--db` default `<repo>/data/databases/corpus.sqlite3`, error if missing); `sys.path` per P7; `sys.stdout.reconfigure(encoding="utf-8")` inside `main`.
  - Keeps verbatim: the `ALIAS` and `SCALE` tables, operand filtering and skip counters, VND normalisation with the raw-value alternative, tolerance `max(source.scale, 1)`, verdicts `correct` / `sign_only` / `WRONG` / `miss:<Outcome>[:<reason>]`, the `mapping_uncertain:` prefix for `interest` operands whose row lacks "trong", the per-leaf and per-question summaries. Never reads or executes `calculation_program`.
  - `--llm MODULE` builds the locator exactly as the scratch script does (it is only used when the user asks; real calls cost money).
  - CSV appends `chunk_id`, `retrieval_step`, `retrieval_rank`; stdout adds `Found via retrieved / full_scan` counts and the rank histogram (1, 2–5, 6–10) when a retriever is set, and prints `retriever=<mode>` in the first line.
- No unit test (spec §7.1: manual). Verified by the reproduction below.

- [ ] **Step 1: Port** the script as in Interfaces.
- [ ] **Step 2: Reproduce the baseline** with `--retriever none` on the gold file the user supplied (`official_gold.jsonl`; ask the user for its path if it is not under `D:/finance_project/data/questions/`):
  `D:\finance_project\.venv\Scripts\python.exe -I -X utf8 scripts/score_gold_operands.py <GOLD_JSONL> .tmp-tests/gold_none.csv --db D:/finance_project/data/databases/corpus.sqlite3`
  Expected: `distinct operand requests scored: 663`, `642 correct`, `2 WRONG` (the dec9de9 baseline). Any other numbers → stop and report; do not commit.
- [ ] **Step 3: Run** `python -m compileall -q scripts` → exit 0.
- [ ] **Step 4: Commit** `feat(scripts): gold operand scoring with a table retriever`.

---

### Task 9: Per-report candidate cache in the adapter (spec §5.5, added 2026-11-12)

Run before Task 8 step 2. Gate step 1 measured hybrid at ~2.1 s per `rank` call (Chroma filtered search 1.7 s); see spec §5.5.

**Files:**
- Modify: `src/main/refine_questions.py` (extract `rank_indices`), `src/retrieval/table_retriever.py`
- Test: `tests/test_table_retriever.py` (class `CandidateCacheTests`), plus one test for `rank_indices` beside the existing `retrieve_refined` tests (grep `tests/` for `retrieve_refined`)

**Interfaces:**
- `rank_indices(retriever, question: str, indices: list[int], k: int) -> tuple[list[Document], list[float]]` in `src/main/refine_questions.py`: the body of `retrieve_refined` from `get_batch_scores` to the top-k slice, moved verbatim (`np.argsort(scores)[::-1]`, zero scores dropped, first `k`); `retrieve_refined` calls it and returns the same dict as today.
- `HybridTableRetriever` keeps its constructor and `rank` signature. Inside:
  - `_candidates(filters) -> _Candidates(indices: list[int], chunk_ids: list[str])`, cached per `(ticker, year, tuple(scopes))` in an LRU of `CANDIDATE_CACHE_SIZE = 32` (`collections.OrderedDict`); computed with `filter_document_indices(self.bm25.docs, filters)`.
  - BM25 list: `rank_indices(self.bm25, query, candidates.indices, k)` (`k = self.k` in bm25 mode, `self.depth` in hybrid).
  - Dense (hybrid): per key, `self.dense_store.get(ids=candidates.chunk_ids, include=["embeddings", "metadatas"])` once (cached with the candidates); query vector `self.dense_store.embeddings.embed_query(query)` cached per query text (plain dict); scores = `matrix @ vector` (numpy, imported lazily); top `depth` by score descending, ties by fetch order; each becomes `Document(page_content="", metadata=<fetched metadata>)`. `similarity_search` is no longer called.
  - `rrf_fuse(lexical, dense, k=self.k, c=RRF_CONSTANT)` and the `RankedTable` building unchanged; the `try → TableRetrievalError` covers `rank_indices`, `get`, `embed_query` and the dot product only.
  - An empty candidate set → `()` without touching the dense store.

- [ ] **Step 1: Write the failing tests**

```python
def test_rank_indices_matches_retrieve_refined(self):        # beside the retrieve_refined tests
    # same tiny BM25; indices = filter_document_indices(docs, filters)
    docs, scores = rank_indices(bm25, "doanh thu", indices, 3)
    self.assertEqual(docs, retrieve_refined(bm25, "doanh thu", k=3, metadata_filters=filters)["documents"])

class CandidateCacheTests(unittest.TestCase):
    def test_candidates_computed_once_per_report(self):
        # patch filter_document_indices in table_retriever's namespace with wraps=...
        r.rank("VJC", "consolidated", 2023, leaf_is); r.rank("VJC", "consolidated", 2023, leaf_bs)
        self.assertEqual(spy.call_count, 1)
        r.rank("VJC", "consolidated", 2022, leaf_is)
        self.assertEqual(spy.call_count, 2)

    def test_dense_vectors_fetched_once_and_query_embedded_once(self):
        # FakeDenseStore with get(ids, include) -> {"ids", "embeddings", "metadatas"} and
        # embeddings.embed_query(text) -> fixed unit vectors per text; both count calls
        r.rank("VJC", "consolidated", 2023, leaf_is); r.rank("VJC", "consolidated", 2023, leaf_is)
        self.assertEqual(store.get_calls, 1); self.assertEqual(store.embed_calls, 1)
        self.assertEqual(store.similarity_calls, 0)

    def test_dense_order_is_exact_cosine(self):
        # vectors chosen so cosine order is C, A, B; bm25 list empty for the query (no shared tokens)
        self.assertEqual([t.chunk_id for t in out], ["VJC23C:3", "VJC23C:1", "VJC23C:2"][:r.k])

    def test_hybrid_equals_rrf_of_cached_branches(self):
        # expected = rrf_fuse(rank_indices(...)[0], <exact dense list>, k=r.k, c=60) chunk ids

    def test_dense_get_error_is_wrapped(self):
        # get raises RuntimeError -> TableRetrievalError with __cause__

    def test_no_candidates_returns_empty_without_dense_call(self):
        self.assertEqual(r.rank("ZZZ", "consolidated", 2023, leaf_is), ())
        self.assertEqual(store.get_calls, 0)
```

Existing `HybridTableRetrieverTests` keep their assertions; adapt `FakeDenseStore` (add `get` / `embeddings`) so the where-clause test checks the ids passed to `get` instead of the Chroma filter.
- [ ] **Step 2: Run** → new tests FAIL.
- [ ] **Step 3: Implement** as in Interfaces. Update the module docstring (cache, exact dense search).
- [ ] **Step 4: Run** `tests.test_table_retriever`, the `retrieve_refined` test module, the full suite, compileall, the import-isolation check (Task 5 Step 4) → PASS.
- [ ] **Step 5: Commit** `perf(retrieval): cache per-report candidates and score dense vectors exactly`.
- [ ] **Step 6 (controller, real indexes):** equivalence script in the scratchpad — for every gold request (ticker, scope, year, leaf) compare the new `rank` with the old path (`retrieve_refined` + `similarity_search` + `rrf_fuse`, top 10): identical-list share, mean overlap; then re-score gold with `--retriever hybrid` (expect 641 correct / 2 WRONG or better) and time it.

---

### Task 8: Evaluation gate (§7.2) and record

**Files:**
- Modify: `docs/superpowers/specs/2026-11-12-retrieval-guided-extraction-design.md` §10 (results)
- Create: shared note `C:\Users\ADMIN\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\side-session-notes\local_e62b6974-8989-4969-9f55-319a57ac3857\retrieval-gate-2026-11-12.md`

Outputs (CSVs, logs) go to the session scratchpad, never into the repo. Index paths: `--bm25-dir D:/finance_project/data/indexes/bm25 --chroma-dir D:/finance_project/data/chroma/7121d2b324f27b82bfcee6d324e7e60d41779b72602cb74f745f5dee1fec8297`. Before long runs, confirm the Codex desktop app is not running (it hashes files under `data/` and starves the disk).

- [ ] **Step 1: Gold, three modes** — `score_gold_operands.py` with `--retriever none`, `bm25`, `hybrid` on the same commit. Record correct / WRONG / sign_only per mode, `Found` via retrieved vs full_scan, rank histogram. Any WRONG beyond the baseline 2 → list each (ticker, leaf, year, got, gold, chunk_id).
- [ ] **Step 2: Full corpus** — `leaf_coverage.py --out <scratch>/cov_none.csv` and `--retriever hybrid --out <scratch>/cov_hybrid.csv`, each in the background (2–3 h; 53,480 requests expected); then `diff_extraction_runs.py cov_none.csv cov_hybrid.csv`. If hybrid has any `value_changed`, `ambiguous_to_found` or `source_changed`, also run `--retriever bm25` and diff it.
- [ ] **Step 3: Review** — for every `value_changed`, `ambiguous_to_found` and `source_changed` row (random 30 per label, fixed seed 0, when a label has more than 30) open the report table and mark the new value right / wrong / unclear, with risk R1 or R2 when it applies.
- [ ] **Step 4: Record** — write the note (commands, counts, p50 / p95 seconds per mode, review table) and a short §10 summary in the spec; commit `docs: retrieval-guided extraction gate results`.
- [ ] **Step 5: Hand the decision to the user** (§7.2 step 4): enable by default, or cross-check / the §6 mitigation and re-run. Do not change any default before the user decides.

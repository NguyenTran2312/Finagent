# Leaf Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/extraction/`: turn the stored tables of a requested report into typed, fully sourced `Fact`s for the calculation core's metric-tree leaves, deterministically first, with a verified LLM cell-pointer fallback, and typed outcomes for everything that is not a clean single value.

**Architecture:** A read-only `ReportStore` loads a report's documents and tables. `TableView` gives each table a code column, normalised labels, period-matched value columns, a number locale and per-column units. `matching` selects candidate tables (statement kind, then a label-anchor rule) and matches rows by code, exact label or bank section block. `LeafExtractor` orchestrates report t / t+1, scope variants and outcomes; an injected `CellLocator` (the LLM fallback) may only point at a cell, which is then read and checked by the same deterministic code.

**Tech Stack:** Python 3.14, stdlib (`dataclasses`, `decimal`, `json`, `sqlite3`, `re`, `itertools`), pydantic (pinned), PyYAML 6.0.3 (pinned), psycopg (pinned, imported lazily), unittest.

**Spec:** `docs/superpowers/specs/2026-11-11-leaf-extraction-design.md` (read it with this plan; § numbers below refer to it). Contracts it builds on: `docs/superpowers/specs/2026-11-10-calculation-core-design.md` §8, §16.

## Global Constraints

- Branch `feat/leaf-extraction` in worktree `D:\finance_project\.worktrees\leaf-extraction`, cut from `feat/calculation-core` (PR #2, unmerged). Commit per task on this branch only, and only after the user has authorized commits for plan execution. Never push, merge or open a PR without explicit user authorization.
- Project rules live in the untracked `D:\finance_project\CLAUDE.md`, `src/CLAUDE.md` and package `CLAUDE.md` files; read them from `D:\finance_project`.
- Production modules import via `src.*` (e.g. `from src.preprocess.cleaner import fold_text`); `config.settings` stays as is.
- No new dependencies. `psycopg` is imported only inside `PostgresReportStore`.
- No network, LLM, GPU or database server in unit tests. `data/databases/corpus.sqlite3` is opened only by the manual scripts, only with `mode=ro`.
- All values are `Decimal`; money is stored in đồng (`Quantity(parsed × scale, dimension)`). Never read `TableAnalyzer.cell().unit_scale` or `tables.unit_scale` (silent 1.0 fallback).
- Never: substring-match row labels (`find_by_alias` is for question text only), treat a blank cell as 0, pick between differing values, substitute a scope, choose a column by position.
- Warnings and reasons are fixed-format strings `code` or `code:<id>`; exceptions carry class names only; raw LLM text never leaves `llm_fallback.py`.
- Do not modify retrieval, scoring, paraphrasing, `metric_catalog.py`, `query_pipeline.py`, `src/preprocess/cleaner.py`, `src/indexing/table_semantics.py`, the PostgreSQL schema or Chroma.
- Test command (PowerShell, worktree root):
  `$env:PYTHONPATH="$PWD;$PWD\src"; $env:TEMP="$PWD\.tmp-tests"; $env:TMP=$env:TEMP; D:\finance_project\.venv\Scripts\python.exe -B -m unittest <target> -v`

## Plan decisions (resolve spec gaps; binding for all tasks)

- E1. `ReportStore.session()` is a context manager yielding a `ReportSession` with `documents(ticker, year)` and `tables(doc_id)`; `LeafExtractor.extract` opens exactly one session per call. This realises spec §9 "one connection per extract call".
- E2. `TableRecord` carries `rows_json: str` (undecoded) and `chunk_id: str | None`. Decoding and shape checks happen in `TableView`, which raises `MalformedTable(table_key, reason)`; the extractor turns that into `skipped_tables` + warning `skipped_table:<table_key>` (`table_without_chunk:<table_key>` when `chunk_id` is None).
- E3. Code-like cells (§6.1) use one template-independent test: `^\d{1,3}$`, a roman numeral I–XXX (letters I, V, X only), or one ASCII letter. Corporate tables simply never contain roman numerals.
- E4. Tickers are upper-cased (`strip().upper()`) at every entry point: templates, store queries, `Fact.ticker`, outcomes.
- E5. When several rows/columns carry the same value, the representative is the first in (page, table_id, row_idx, col_idx) order.
- E6. `CellLocator` is a Protocol defined in `extractor.py`; `LlmCellLocator` (Task 7) implements it. With `fallback=None` the extractor never calls an LLM.
- E7. The LLM shortlist (§8.2) renders only the first `max_rows` data rows of each table; a pointer to a row that was not rendered is rejected (`llm_pointer_rejected:not_rendered`).

## Review Focus

1. A joined OCR header such as "31.12.2022Triệu VND" must give closing 2022 and VND ×10^6, not `UnitUnknown`. Pinned in Task 4 (`test_joined_ocr_header_unit_and_period`).
2. A cell with a footnote mark or two numbers ("1.234.567 (*)") must give `Ambiguous(number_unparsed)`, never a truncated number. Pinned in Task 4 (`test_footnote_cell_is_unparsed`) and Task 6 (`test_unparsed_cell_outcome`).
3. A lower-case ticker ("acb") must work end to end and produce `Fact.ticker == "ACB"`. Pinned in Task 6 (`test_lowercase_ticker`).
4. A table whose only value column has no year and no period words must not be used, even though it is the only candidate. Pinned in Task 4 (`test_sole_unlabelled_column_not_used`).
5. An LLM pointer into a long table beyond the rendered rows must be rejected, even if that cell holds a valid number. Pinned in Task 7 (`test_pointer_beyond_rendered_rows_rejected`).

---

### Task 1: Order-independent `FactSet` and `MetricTree.leaves`

**Files:**
- Modify: `src/calculation/facts.py` (`_resolve`, `FactSet.__init__`)
- Modify: `src/calculation/metric_tree.py` (add `leaves`)
- Test: `tests/test_calculation_facts.py`, `tests/test_calculation_metric_tree.py`
- Add (copy, uncommitted so far): `docs/superpowers/specs/2026-11-11-leaf-extraction-design.md`, this plan

**Interfaces:**
- Produces: `FactSet(facts)` with spec §7.3 semantics (same public API, same warning text `restated value used for <metric>/<ticker>/<year>/<scope>/<period>`); `MetricTree.leaves(template: Template) -> tuple[LeafDef, ...]` in YAML definition order.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calculation_facts.py (FactSetTests)
def test_order_independent_all_pairs(self):
    x = fact("1000", scale="1")
    y = fact("1800", scale="1000")
    z = fact("2500", scale="1000")       # |z - x| = 1500 > 1000
    for order in itertools.permutations([x, y, z]):
        with self.subTest(order=[f.value.value for f in order]):
            with self.assertRaises(FactConflict):
                FactSet(order)

def test_order_independent_kept_fact(self):
    a, b, c = fact("1000", scale="1000"), fact("1000", scale="1"), fact("1000.4", scale="1000")
    for order in itertools.permutations([a, b, c]):
        self.assertIs(FactSet(order).get(a.key), b)

def test_one_restated_among_three_wins(self):
    r = fact("9000", restated=True)
    for order in itertools.permutations([fact("1000"), fact("1000"), r]):
        s = FactSet(order)
        self.assertIs(s.get(r.key), r)
        self.assertEqual(len(s.warnings), 1)

# tests/test_calculation_metric_tree.py
def test_leaves_accessor(self):
    tree = MetricTree.load()
    ids = [leaf.id for leaf in tree.leaves("bank")]
    self.assertEqual(ids[0], "interest_and_similar_income")
    self.assertIn("customer_loans_gross", ids)
    self.assertTrue(all(leaf.template == "corporate" for leaf in tree.leaves("corporate")))
```

- [ ] **Step 2: Run to verify they fail**

Run: test command with `tests.test_calculation_facts tests.test_calculation_metric_tree`
Expected: `test_order_independent_all_pairs` FAILS for some permutations (no `FactConflict`); `test_leaves_accessor` errors with `AttributeError`.

- [ ] **Step 3: Implement**

`FactSet.__init__`: validate types, group facts by key preserving input order, then resolve each group per spec §7.3 (restated rules 1–2, all-pairs tolerance `max(a.source.scale, b.source.scale)`, finest scale kept, tie → first in input order, dimension mismatch → `FactConflict`). Remove the pairwise `_resolve`. `MetricTree.leaves`: return the leaves of `template` in insertion order of `self._leaves`.

- [ ] **Step 4: Run the full calculation suite**

Run: test command with `discover -s tests -p "test_calculation_*.py"`
Expected: all pass, including the existing `test_three_facts_pairwise_in_input_order`, `test_two_restated_within_tolerance_ok`, `test_restated_within_tolerance_still_wins`.

- [ ] **Step 5: Commit**

```bash
git add src/calculation/facts.py src/calculation/metric_tree.py tests/test_calculation_facts.py tests/test_calculation_metric_tree.py docs/superpowers/specs/2026-11-11-leaf-extraction-design.md docs/superpowers/plans/2026-11-11-leaf-extraction.md
git commit -m "fix(calculation): order-independent FactSet resolution; add MetricTree.leaves"
```

---

### Task 2: Ticker templates loader

**Files:**
- Create: `src/extraction/__init__.py` (docstring only), `src/extraction/templates.py`
- Test: `tests/test_extraction_templates.py`

**Interfaces:**
- Produces:
  - `ExtractionTemplate = Literal["corporate", "bank", "securities", "insurance"]`
  - `DEFAULT_TICKER_TEMPLATES: Path` = `<repo>/config/ticker_templates.yaml`
  - `class TemplateConfigError(ValueError)` with attributes `file`, `ticker`, `reason`; message `"<file>: <ticker>: <reason>"`
  - `class TickerTemplates`: `__init__(mapping: Mapping[str, ExtractionTemplate])`, `get(ticker: str) -> ExtractionTemplate | None`, `__contains__(ticker) -> bool`, `calculation_templates() -> dict[str, Template]` (corporate/bank only)
  - `load_ticker_templates(path: Path = DEFAULT_TICKER_TEMPLATES) -> TickerTemplates`

- [ ] **Step 1: Write the failing tests** (fixture YAML files written to a `tempfile.TemporaryDirectory`)

```python
def test_load_valid(self):
    t = load_ticker_templates(write("ACB: bank\nvjc: corporate\nFTS: securities\n"))
    self.assertEqual(t.get("acb"), "bank")
    self.assertEqual(t.get("VJC"), "corporate")
    self.assertIsNone(t.get("XXX"))
    self.assertEqual(t.calculation_templates(), {"ACB": "bank", "VJC": "corporate"})

def test_errors(self):
    cases = {"- ACB\n": "-", "ACB: retail\n": "ACB", "ACB: bank\nacb: bank\n": "ACB",
             "'  ': bank\n": "?", "1: bank\n": "?"}
    for text, ticker in cases.items():
        with self.subTest(text=text), self.assertRaises(TemplateConfigError) as ctx:
            load_ticker_templates(write(text))
        self.assertEqual(ctx.exception.ticker, ticker)

def test_missing_file(self):
    with self.assertRaises(TemplateConfigError):
        load_ticker_templates(Path(self.tmp) / "absent.yaml")
```

- [ ] **Step 2: Run to verify they fail** — `ModuleNotFoundError: src.extraction`.
- [ ] **Step 3: Implement `templates.py`** with `yaml.safe_load`; a YAML/Unicode error or missing file raises `TemplateConfigError(name, "-", reason)` carrying the exception class name only.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(extraction): ticker template config loader`.

---

### Task 3: Report store

**Files:**
- Create: `src/extraction/report_store.py`
- Create: `tests/extraction_fixtures.py` (shared helpers; extended in Task 4)
- Test: `tests/test_extraction_report_store.py`

**Interfaces:**
- Produces (`report_store.py`):
  - `@dataclass(frozen=True) ReportDoc(doc_id: str, ticker: str, report_year: int, scope: str, scope_variant: int | None)`
  - `@dataclass(frozen=True) TableRecord(table_key: str, chunk_id: str | None, doc_id: str, table_id: int, page: int, context: str, statement_kind: str | None, rows_json: str)`
  - `class ReportSession(Protocol)`: `documents(ticker: str, year: int) -> list[ReportDoc]` (all scopes, ordered by `scope`, `scope_variant` NULLS FIRST, `doc_id`); `tables(doc_id: str) -> list[TableRecord]` (ordered by `page`, `table_id`)
  - `class ReportStore(Protocol)`: `session() -> ContextManager[ReportSession]`
  - `class ReportStoreError(RuntimeError)`: `__init__(operation: str, error_class: str)`; attributes of the same names
  - `SqliteReportStore(path: str | Path)`; `PostgresReportStore(connection_params: dict | None = None, *, connect: Callable | None = None)` (`connect` defaults to `psycopg.connect`, imported inside `session()`)
- Produces (`tests/extraction_fixtures.py`): `InMemoryReportStore(docs: Sequence[ReportDoc], tables: Mapping[str, Sequence[TableRecord]])`; `record(doc_id: str, table_id: int, page: int, rows: list[list[str]], *, kind: str | None, context: str = "", chunk: bool = True) -> TableRecord` (`table_key=f"{doc_id}:{table_id}"`, `chunk_id=f"{doc_id}:table:{table_id}"` or None, `rows_json=json.dumps(rows, ensure_ascii=False)`)

SQL (one text; placeholders `?` for SQLite, `%s` for psycopg):

```sql
SELECT doc_id, ticker, report_year, scope, scope_variant FROM documents
WHERE ticker = ? AND report_year = ?
ORDER BY scope, scope_variant IS NOT NULL, scope_variant, doc_id;

SELECT t.table_key, c.chunk_id, t.doc_id, t.table_id, t.page, COALESCE(t.context, ''),
       t.statement_kind, t.rows_json
FROM tables t
LEFT JOIN chunks c ON c.table_key = t.table_key AND c.doc_id = t.doc_id AND c.chunk_type = 'table'
WHERE t.doc_id = ? ORDER BY t.page, t.table_id;
```

- [ ] **Step 1: Write the failing tests**

```python
# temporary SQLite file built with src.indexing.schema.SCHEMA; two docs (consolidated v1, v2),
# one separate doc, tables on pages 8 and 7, one table without a chunk row.
def test_documents_all_scopes_ordered(self):
    with SqliteReportStore(self.path).session() as s:
        docs = s.documents("acb", 2022)
    self.assertEqual([(d.scope, d.scope_variant) for d in docs],
                     [("consolidated", 1), ("consolidated", 2), ("separate", None)])

def test_tables_ordered_and_chunk_join(self):
    with SqliteReportStore(self.path).session() as s:
        tables = s.tables("D1")
    self.assertEqual([t.page for t in tables], [7, 8, 8])
    self.assertEqual(tables[0].chunk_id, "D1:table:2")
    self.assertIsNone(tables[2].chunk_id)

def test_sqlite_is_read_only(self):
    with SqliteReportStore(self.path).session() as s:
        with self.assertRaises(sqlite3.OperationalError):
            s._conn.execute("DELETE FROM documents")   # attribute name fixed: _conn

def test_store_error_carries_class_name_only(self):
    with self.assertRaises(ReportStoreError) as ctx:
        with SqliteReportStore(Path(self.tmp) / "absent.sqlite3").session() as s:
            s.documents("ACB", 2022)
    self.assertEqual(ctx.exception.error_class, "OperationalError")
    self.assertNotIn("absent", str(ctx.exception))

def test_postgres_session_read_only_and_closed(self):
    fake = FakeConnect()                       # records execute() SQL; close() flag
    with self.assertRaises(RuntimeError):
        with PostgresReportStore({}, connect=fake).session():
            raise RuntimeError("boom")
    self.assertIn("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY", fake.sql[0])
    self.assertTrue(fake.closed)
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** `report_store.py` and the fixture helpers. SQLite opens `sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)`; Postgres opens one connection, starts a transaction, runs the `SET TRANSACTION` statement, and closes in `finally`. Database exceptions (`sqlite3.Error`, `psycopg.Error`) become `ReportStoreError(operation, type(exc).__name__)` raised `from None`.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(extraction): read-only report store`.

---

### Task 4: Outcomes and `TableView`

**Files:**
- Create: `src/extraction/outcomes.py`, `src/extraction/table_view.py`
- Modify: `tests/extraction_fixtures.py` (table fixtures)
- Test: `tests/test_extraction_table_view.py`

**Interfaces:**
- Consumes: `TableRecord` (Task 3); `TableAnalyzer`, `period_info`, `PeriodRole`, `row_label`, `TOTAL_MARKERS` (`src.indexing.table_semantics`); `parse_financial_number`, `parse_financial_unit`, `fold_text`, `clean_text` (`src.preprocess.cleaner`); `FactSource`, `Fact`, `Scope` (`src.calculation.facts`); `Quantity`, `Dimension` (`src.calculation.units`); `LeafRequest` (`src.calculation.metric_tree`).
- Produces (`outcomes.py`, all `@dataclass(frozen=True)`):
  - `Method = Literal["code", "label", "section_position", "llm_located"]`
  - `CellLocation(chunk_id: str, doc_id: str, page: int, row_label: str, column_header: str, raw_text: str)`
  - `Candidate(value: Decimal | None, location: CellLocation)`
  - `ExtractedFact(fact: Fact, method: Method, warnings: tuple[str, ...])`
  - outcomes, each with leading fields `ticker: str, scope: Scope, request: LeafRequest`: `Found(extracted)`, `NotFound(reason: str, searched_doc_ids: tuple[str, ...], skipped_tables: tuple[str, ...], warnings: tuple[str, ...])`, `BlankValue(location)`, `Ambiguous(reason: str, candidates: tuple[Candidate, ...])`, `Conflict(candidates)`, `UnitUnknown(reason: str, location)`, `ReportMissing(available_scopes: tuple[str, ...])`, `NotApplicable(template: str)`, `UnsupportedTemplate(template: str)`, `UnknownTicker()`; `Outcome` = their `Union`.
- Produces (`table_view.py`):
  - `SourceRole = Literal["primary", "fallback"]`
  - `class MalformedTable(ValueError)`: `table_key`, `reason ∈ {"no_chunk", "json", "shape"}`
  - `normalize_label(text: str) -> str` (§6.1); `enumerator_level(code: str) -> Literal["top", "section", "item", "sub_item"] | None` (§6.4)
  - `header_unit(cells: Sequence[str]) -> ParsedFinancialUnit` (§6.7, nearest header row first)
  - `@dataclass(frozen=True) CellRead(status: Literal["ok", "blank", "unparsed", "unit"], location: CellLocation, reason: str | None = None, value: Quantity | None = None, source: FactSource | None = None)`
  - `class TableView`: `__init__(record: TableRecord, report_year: int)` (raises `MalformedTable`); attributes `record`, `report_year`, `code_col: int | None`, `data_rows: tuple[int, ...]`, `value_cols: tuple[int, ...]`, `locale: Literal["vi", "en", "mixed"]`; methods `label(row_idx) -> str`, `norm_label(row_idx) -> str`, `code(row_idx) -> str`, `column_header(col_idx) -> str`, `period_columns(year: int, period_kind: str, role: SourceRole) -> tuple[int, ...]`, `read(row_idx: int, col_idx: int, dimension: Dimension) -> CellRead`, `restatement_marker(col_idx) -> bool`, `tokens() -> frozenset[str]`

Period rules are spec §6.5's table. Unit rules are spec §6.7 (`header_unit` per header-row cell after removing dates, years and the listed period phrases; then `TableAnalyzer.table_unit`; then `parse_financial_unit(record.context)`; then dimension checks with reasons `missing`, `ambiguous`, `kind_mismatch`, `currency`). `read` returns `unparsed` with reason `number_locale` when `locale == "mixed"`, `number_unparsed` for `invalid`/`ambiguous` numbers; `blank` for `missing`. `FactSource.unit_text` is the header cell the unit came from (or the table header / context text), `scale` is `scale_multiplier`.

Fixtures to add to `tests/extraction_fixtures.py` (real values from the corpus where noted; others synthetic):
- `VJC_IS_2023`: the two real VJC header rows (`['Mã số', '', 'Thuyết minh', 'Năm tài chính kết thúc ngày 31 tháng 12 năm', …]`, `['Mã số', '', 'Thuyết minh', '2023 VND', '2022 VND']`); real rows `01`, `10`, `11` (`(55.819.871.829.489)`), `20`, `23` (`- Trong đó: Chi phí lãi vay`, `(1.950.254.195.864)`, `(1.383.641.245.573)`); synthetic rows `50` (`Tổng lợi nhuận kế toán trước thuế`, `606.103.607.021`, `(2.648.659.574.470)`) and `60`.
- `ACB_BS_2022`: real header rows (`'31.12.2022 Triệu VND'`, `'31.12.2021 Triệu VND'`), `A TÀI SẢN`, `I`, `III` with children `1`–`3`, plus synthetic decoy child `4 Cho vay khách hàng 1.000 900` inside III, real `VI Cho vay khách hàng 408.856.508`, `1 Cho vay khách hàng 413.706.244`, `2 Dự phòng rủi ro cho vay khách hàng (4.849.736)`, real `VIII` with `1`, and `'' TỔNG TÀI SẢN 607.875.185 527.769.944`.
- `ACB_IS_2022_AS_NOTE`: header `['', '', 'Thuyết minh', 'Năm 2022 Triệu VND', 'Năm 2021 Triệu VND']`, real rows `1` and `2`, synthetic `I Thu nhập lãi thuần 23.533.529`, `II Lãi thuần từ hoạt động dịch vụ`, `VIII Chi phí hoạt động`, `IX Lợi nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng`, `X Chi phí dự phòng rủi ro tín dụng`, `XI Tổng lợi nhuận trước thuế`, `XIII Lợi nhuận sau thuế`.
- `ACB_SEGMENT_NOTE`: the real page-98 segment rows (`Tài sản`, `Nợ phải trả`, `Kết quả kinh doanh bộ phận`).
- View builders: `vjc_view()` (doc `VJC23C`, table 1, page 12, `income_statement`, report year 2023), `acb_bs_view()` (doc `ACB22C`, table 2, page 7, `balance_sheet`, 2022), `acb_is_view(kind="note")` (doc `ACB22C`, table 5, page 10), `segment_view(kind="income_statement")` (doc `ACB22C`, table 40, page 98); `row_of(view, text) -> int` returns the first data row whose code or label equals `text`.

- [ ] **Step 1: Write the failing tests**

```python
def test_vjc_real_header_unit_and_periods(self):
    v = TableView(record("D", 1, 12, VJC_IS_2023, kind="income_statement"), 2023)
    self.assertEqual(v.code_col, 0)
    self.assertEqual(v.period_columns(2023, "flow", "primary"), (3,))
    self.assertEqual(v.period_columns(2022, "flow", "fallback"), (4,))
    r = v.read(row_of(v, "11"), 3, Dimension.MONEY)
    self.assertEqual((r.status, r.value.value, r.source.scale), ("ok", D("-55819871829489"), D(1)))

def test_joined_ocr_header_unit_and_period(self):        # Review Focus 1
    rows = [["", "", "Thuyết minh", "Tại ngày"], ["", "", "Thuyết minh", "31.12.2022Triệu VND"],
            ["I", "Tiền mặt, vàng bạc, đá quý", "4", "8.460.892"]]
    v = TableView(record("D", 1, 7, rows, kind="balance_sheet"), 2022)
    self.assertEqual(v.period_columns(2022, "closing", "primary"), (3,))
    self.assertEqual(v.read(2, 3, Dimension.MONEY).value.value, D("8460892000000"))

def test_header_unit_rules(self):
    self.assertEqual(header_unit(["Số cuối năm Triệu đồng"]).scale_multiplier, D("1E+6"))
    self.assertEqual(header_unit(["2023 VND", "Năm tài chính kết thúc ngày 31 tháng 12 năm"]).unit_status, "valid")
    self.assertEqual(header_unit(["Triệu VND", "VND"]).unit_status, "ambiguous")

def test_context_declaration_unit(self): ...       # no header unit, context "Đơn vị tính: triệu đồng" → ×10^6
def test_code_column_rules(self): ...               # "Mã số" header; content ≥3 and ≥80%; "Thuyết minh" excluded; two candidates → None
def test_normalize_label_and_levels(self):
    self.assertEqual(normalize_label("VI Cho vay khách hàng"), "cho vay khach hang")
    self.assertEqual([enumerator_level(c) for c in ("A", "VI", "X", "2", "b", "6.1")],
                     ["top", "section", "section", "item", "sub_item", None])
def test_locale(self): ...                           # vi / en ("1,234,567") / mixed
def test_blank_and_unit_problems(self): ...          # "-" → blank; "%" header on money → unit/kind_mismatch; "USD" → unit/currency; EPS at "Triệu VND" → unit/kind_mismatch
def test_footnote_cell_is_unparsed(self):            # Review Focus 2
    r = view_with_cell("1.234.567 (*)").read(2, 3, Dimension.MONEY)
    self.assertEqual((r.status, r.reason), ("unparsed", "number_unparsed"))
def test_sole_unlabelled_column_not_used(self):      # Review Focus 4
    rows = [["Chỉ tiêu", "Giá trị"], ["Doanh thu thuần", "1.000"]]
    v = TableView(record("D", 1, 3, rows, kind="note"), 2023)
    self.assertEqual(v.period_columns(2023, "flow", "primary"), ())
def test_restatement_marker(self): ...               # header "Năm 2022 (trình bày lại)" → True
def test_malformed(self): ...                        # chunk None → no_chunk; "not json" → json; [["a"], 1] → shape
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `outcomes.py` and `table_view.py`.** Build one `TableAnalyzer(rows, context=record.context, report_year=report_year)` per view; header cells per column come from `analyzer.header_indices` (nearest row first for `header_unit`, joined string for `column_header`). Row labels via `row_label(row, code_idx=code_col)`.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(extraction): table view with periods, numbers and units`.

---

### Task 5: Table selection and row matching

**Files:**
- Create: `src/extraction/matching.py`
- Test: `tests/test_extraction_matching.py`

**Interfaces:**
- Consumes: `TableView`, `normalize_label`, `enumerator_level` (Task 4); `LeafDef`, `MetricTree.leaves` (Task 1); `StatementKind`, `TOTAL_MARKERS`.
- Produces:
  - `label_keys(leaf: LeafDef) -> frozenset[str]` (folded `label_vi` + aliases)
  - `@dataclass(frozen=True) RowHit(view: TableView, row_idx: int, method: Method, warnings: tuple[str, ...])`
  - `anchor_kinds(view: TableView, template_leaves: Sequence[LeafDef]) -> frozenset[StatementKind]`
  - `find_rows(leaf: LeafDef, views: Sequence[TableView], template_leaves: Sequence[LeafDef]) -> list[RowHit]` — spec §6.2–§6.4: tagged statement tables first, anchor tables only if those give no hit; note leaves search `note`/`unknown` tables; hits ordered per E5.

- [ ] **Step 1: Write the failing tests** (tree = `MetricTree.load()`, `bank = tree.leaves("bank")`, `corp = tree.leaves("corporate")`)

```python
def test_corporate_codes(self):
    hits = find_rows(tree.leaf("interest_expense", "corporate"), [vjc_view()], corp)
    self.assertEqual([(h.method, h.view.code(h.row_idx)) for h in hits], [("code", "23")])

def test_code_compared_as_integer(self): ...          # code cell "1" matches leaf code "01"
def test_no_code_column_falls_back_to_label(self):    # method "label", warning "matched_by_label"
def test_bank_section_and_children(self):
    v = acb_bs_view()
    for leaf_id, value in [("customer_loans_section", "408.856.508"),
                           ("customer_loans_gross", "413.706.244"),
                           ("customer_loan_provision", "(4.849.736)")]:
        hits = find_rows(tree.leaf(leaf_id, "bank"), [v], bank)
        self.assertEqual(len(hits), 1, leaf_id)
        self.assertEqual(hits[0].method, "section_position")
        self.assertEqual(v.read(hits[0].row_idx, 3, Dimension.MONEY).location.raw_text, value)
        # the decoy "4 Cho vay khách hàng" under III is never hit

def test_section_without_enumerators(self): ...       # code column dropped: X, X, provision pattern accepted; other shape → []
def test_total_assets_two_labels(self): ...           # "TỔNG TÀI SẢN" + "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU" → 2 hits
def test_no_substring_matching(self): ...             # "Dự phòng giảm giá hàng tồn kho" never hits inventories
def test_anchor_rule_bank_income(self):
    views = [segment_view(kind="income_statement"), acb_is_view(kind="note")]
    hits = find_rows(tree.leaf("net_interest_income", "bank"), views, bank)
    self.assertEqual([h.view.record.table_id for h in hits], [acb_is_view().record.table_id])
def test_anchor_threshold_and_cross_kind(self): ...   # 2 bank IS labels → no anchor; corporate cash flow with its 2 labels anchors; an "unknown" income statement with codes 02, 20 does not anchor as cash_flow; a table anchoring two kinds → none
def test_note_leaf_exact_label(self): ...             # accrued_interest_payable in a note table
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `matching.py`.** Section blocks per §6.4 using `enumerator_level(view.code(row))`; block ends at the next `top`/`section` row or a row whose normalised label starts with a folded `TOTAL_MARKERS` entry. Children of a section = `[l for l in template_leaves if l.parent == section.id]`.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(extraction): candidate tables and row matching`.

---

### Task 6: `LeafExtractor`

**Files:**
- Create: `src/extraction/extractor.py`
- Test: `tests/test_extraction_extractor.py`

**Interfaces:**
- Consumes: Tasks 2–5; `MetricTree.leaf`, `LeafRequest`, `FactSet`, `Fact`.
- Produces:
  - `@dataclass(frozen=True) LocatorResult(read: CellRead | None, reason: str | None, warnings: tuple[str, ...])` — `read.status == "ok"` whenever `read` is set
  - `class CellLocator(Protocol)`: `locate(leaf: LeafDef, request: LeafRequest, role: SourceRole, views: Sequence[TableView]) -> LocatorResult`
  - `@dataclass(frozen=True) ExtractionResult(ticker: str, scope: Scope, outcomes: tuple[Outcome, ...])` with property `facts -> tuple[ExtractedFact, ...]` and method `fact_set() -> FactSet`
  - `class LeafExtractor`: `__init__(store: ReportStore, tree: MetricTree, templates: TickerTemplates, fallback: CellLocator | None = None)`; `extract(ticker: str, scope: Scope, requests: Sequence[LeafRequest]) -> ExtractionResult`

Behaviour is spec §5 exactly: validation (§5.1), template outcomes (§5.2), the primary/fallback combination table (§5.3), restatement flag and `from_next_year_report:<doc_id>` warning, variant combination (§5.4) with `not_in_variant:<doc_id>` and `variant_outcome:<doc_id>:<kind>`, multi-row equality (§5.5, `Ambiguous(reason="multiple_rows")`), period-column ambiguity (`multiple_period_columns`, `no_period_column`), the LLM step (§5.6) and `MalformedTable` handling (E2). The fact's `fiscal_year` is the request year; `scope` and `ticker` are the call's (E4).

- [ ] **Step 1: Write the failing tests** (`InMemoryReportStore` + fixtures; `templates = TickerTemplates({"VJC": "corporate", "ACB": "bank", "FTS": "securities"})`)

```python
def test_vjc_found_with_full_source(self):
    out = extract("VJC", "consolidated", [LeafRequest("interest_expense", 2023, "flow")]).outcomes[0]
    self.assertIsInstance(out, Found)
    f = out.extracted.fact
    self.assertEqual(f.value.value, D("-1950254195864"))
    self.assertEqual((f.source.chunk_id, f.source.page, f.source.raw_text),
                     ("VJC23C:table:1", 12, "(1.950.254.195.864)"))
    self.assertEqual(out.extracted.method, "code")

def test_primary_missing_uses_next_year(self): ...    # only 2024 report: 2023 flow from prior column, warning from_next_year_report:<doc>, restated False
def test_restatement_marker_sets_restated(self): ...
def test_closing_from_opening_column(self): ...       # 2021 closing from 2022 report "Số đầu năm"
def test_report_missing_lists_scopes(self):            # only separate + unknown docs → ReportMissing(("separate", "unknown"))
def test_combination_table(self): ...                  # each row of spec §5.3 table, incl. primary_report_missing warning
def test_primary_ambiguous_does_not_fall_back(self): ...
def test_variants_agree_disagree_onesided(self): ...   # finer scale kept; Conflict; not_in_variant warning
def test_multiple_rows_and_columns(self): ...          # equal → one fact from first page; different → Ambiguous(multiple_rows / multiple_period_columns)
def test_blank_unit_unparsed_outcomes(self): ...       # BlankValue, UnitUnknown(missing), Ambiguous(number_unparsed)
def test_unparsed_cell_outcome(self): ...              # Review Focus 2, end to end
def test_template_outcomes(self): ...                  # FTS → UnsupportedTemplate; "ZZZ" → UnknownTicker; gross_revenue for ACB → NotApplicable
def test_lowercase_ticker(self):                       # Review Focus 3
    out = extract("acb", "consolidated", [LeafRequest("total_assets", 2022, "closing")]).outcomes[0]
    self.assertEqual(out.extracted.fact.ticker, "ACB")
def test_validation_errors(self): ...                  # scope "parent" → ValueError; tuple request → TypeError; period mismatch → ValueError; store never opened
def test_request_order_and_dedup(self): ...            # duplicates merged, order kept; fact_set() builds FactSet
def test_malformed_table_reported(self): ...           # skipped_tables + skipped_table:<key>; table_without_chunk:<key>
def test_one_session_per_call(self): ...               # InMemoryReportStore counts session() entries == 1
def test_fallback_called_only_for_not_found(self):     # stub CellLocator records calls
    ...  # NotFound → called once per document set; Ambiguous / ReportMissing → not called; stub read → Found(method="llm_located")
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `extractor.py`.** Within `extract`, cache `TableView`s per `doc_id` (dict local to the call). The per-document step returns one of: hits resolved to a single `CellRead`, or an outcome; the source and variant combination are pure functions over those results so they can be tested through `extract` alone.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(extraction): leaf extractor with typed outcomes`.

---

### Task 7: LLM cell-pointer fallback

**Files:**
- Create: `src/extraction/llm_fallback.py`
- Test: `tests/test_extraction_llm_fallback.py`

**Interfaces:**
- Consumes: `CellLocator`, `LocatorResult` (Task 6); `TableView`, `CellRead`, `SourceRole` (Task 4); `label_keys` (Task 5).
- Produces:
  - `CellPointer(BaseModel)`: `table_ref: str`, `row_idx: int (ge=0)`, `col_idx: int (ge=0)`; `extra="forbid"`
  - `PointerAnswer(BaseModel)`: `status: Literal["found", "not_present"]`, `pointer: CellPointer | None`; validator: pointer required iff `found`; `extra="forbid"`
  - `structured_pointer_adapter(client) -> Callable[[str], PointerAnswer | None]` (paraphraser pattern; `TypeError` if `client` lacks `with_structured_output`)
  - `shortlist(leaf: LeafDef, views: Sequence[TableView], max_tables: int) -> list[TableView]` (§8.2 ranking)
  - `build_prompt(leaf: LeafDef, request: LeafRequest, tables: Sequence[TableView], *, max_rows: int, max_chars: int) -> tuple[str, dict[str, tuple[TableView, int]]]` (prompt text, `table_ref → (view, rendered_rows)`)
  - `class LlmCellLocator`: `__init__(llm: Callable[[str], PointerAnswer | None], *, llm_errors: tuple[type[Exception], ...] = (TimeoutError, ConnectionError, OSError), max_tables: int = 5, max_rows: int = 60, max_chars: int = 12000)`; `locate(...)` per the `CellLocator` Protocol

Prompt copy (Vietnamese, exact frame; the tables follow as `T1`…`Tn` grids of `row_idx | col_idx: text`):

```text
Bạn là trợ lý đọc báo cáo tài chính. Hãy tìm ô chứa giá trị của chỉ tiêu dưới đây.
Chỉ tiêu: {label_vi} (tên khác: {aliases})
Loại báo cáo: {statement_kind}. Năm tài chính: {year}. Kỳ: {"trong năm" | "số dư cuối năm"}.
Chỉ trả về vị trí ô (table_ref, row_idx, col_idx) hoặc status "not_present". Không được trả về con số.
```

Verifier checks are spec §8.3 1–5 plus E7; rejection warnings `llm_pointer_rejected:<check>` with `<check>` ∈ `unknown_table`, `out_of_bounds`, `not_rendered`, `not_data_row`, `not_value_column`, `period`, `number`, `unit`, `label`. Failure handling is spec §8.4.

- [ ] **Step 1: Write the failing tests** (`FakeClient.with_structured_output` returns an object whose `invoke` returns a scripted mapping or raises)

```python
def test_valid_pointer_found(self):
    loc = LlmCellLocator(lambda p: PointerAnswer(status="found", pointer=CellPointer(table_ref="T1", row_idx=row23, col_idx=3)))
    res = loc.locate(leaf("interest_expense"), LeafRequest("interest_expense", 2023, "flow"), "primary", [vjc_view()])
    self.assertEqual(res.read.value.value, D("-1950254195864"))

def test_rejections(self):
    for pointer, check in [(("T9", 2, 3), "unknown_table"), (("T1", 999, 3), "out_of_bounds"),
                           (("T1", 0, 3), "not_data_row"), (("T1", row23, 0), "not_value_column"),
                           (("T1", row23, 4), "period"), (("T1", row_revenue, 3), "label")]:
        with self.subTest(check=check):
            res = locator_returning(pointer).locate(...)
            self.assertIsNone(res.read)
            self.assertEqual(res.reason, "llm_pointer_rejected")
            self.assertIn(f"llm_pointer_rejected:{check}", res.warnings)

def test_pointer_beyond_rendered_rows_rejected(self):   # Review Focus 5, E7
    view = long_table_view(rows=80)          # valid leaf row at index 70
    res = LlmCellLocator(returning(("T1", 70, 3)), max_rows=60).locate(..., [view])
    self.assertIn("llm_pointer_rejected:not_rendered", res.warnings)

def test_failures(self): ...        # llm_errors exception → llm_fallback_error:TimeoutError; ValueError propagates; None → llm_fallback_unparsed; not_present → no read, no warning
def test_adapter(self): ...         # parsing_error → None; parsed returned; raw never returned; missing with_structured_output → TypeError
def test_shortlist_ranking(self): ...   # token overlap, +2 statement kind, ties by page then table_id, top max_tables
def test_prompt_budget(self): ...       # max_chars drops lowest-ranked tables whole; prompt contains "Không được trả về con số"
def test_extractor_integration(self): ...   # LeafExtractor(fallback=LlmCellLocator(fake)) → Found(method="llm_located") with full FactSource
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `llm_fallback.py`.**
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(extraction): verified LLM cell-pointer fallback`.

---

### Task 8: Ticker template config and lint

**Files:**
- Create: `scripts/lint_ticker_templates.py`, `config/ticker_templates.yaml`
- Test: `tests/test_extraction_templates.py` (add real-file test)

**Interfaces:**
- Consumes: `load_ticker_templates` (Task 2), `SqliteReportStore` (Task 3).
- Produces: `config/ticker_templates.yaml` covering every ticker in `documents`; CLI `python scripts/lint_ticker_templates.py [--db PATH] [--draft]` (default DB `data/databases/corpus.sqlite3`, opened read-only). `--draft` prints YAML (`ticker: template  # form codes seen: …`), sorted by ticker; without it, prints one line per mismatch or unclassified ticker and exits 1 if any.

Form-code rules (folded first-page context and table contexts of each document): `tctd` → bank; `ctck` → securities; `dnbh` or `bao hiem` form markers → insurance; `b 0[1-9] dn` → corporate; none → unclassified.

- [ ] **Step 1: Write the failing test** — `test_real_config_loads`: `load_ticker_templates()` succeeds, `get("ACB") == "bank"`, `get("VJC") == "corporate"`.
- [ ] **Step 2: Run to verify it fails** (file missing).
- [ ] **Step 3: Implement the script; run `--draft` against the local corpus copy; write the draft to `config/ticker_templates.yaml`.**
- [ ] **Step 4: STOP — show the user the draft (and every unclassified or multi-form ticker) and wait for their review.** Apply their corrections. This file is hand-reviewed by decision (spec §2.4); do not continue without approval.
- [ ] **Step 5: Run the lint (expect exit 0 after review) and the test.** Report the lint output separately from unit tests.
- [ ] **Step 6: Commit** — `feat(extraction): reviewed ticker template config and lint`.

---

### Task 9: Coverage and smoke scripts; final verification

**Files:**
- Create: `scripts/leaf_coverage.py`, `scripts/smoke_llm_fallback.py`
- Modify: `docs/superpowers/specs/2026-11-11-leaf-extraction-design.md` (status line → "approved; implemented on feat/leaf-extraction")

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `python scripts/leaf_coverage.py [--db PATH] [--leaf ID ...] [--years 2015-2025] [--out CSV]`: for every ticker with template corporate/bank, every year and scope with a document, extracts each requested leaf (all leaves by default) with `fallback=None`; prints per leaf and template the count of each outcome type and each `reason`/warning code, plus how many hits needed the anchor rule; optional CSV of per-request outcomes. Read-only.
  - `python scripts/smoke_llm_fallback.py --model-module MODULE --ticker T --year Y --leaf ID`: imports a user-supplied module exposing `build_client()`, wraps it with `structured_pointer_adapter`, runs one extraction with `LlmCellLocator`, prints the outcome. Never run by the test suite.

- [ ] **Step 1: Implement both scripts** (argument parsing, read-only store, no writes outside `--out`).
- [ ] **Step 2: Run `leaf_coverage.py` on the local corpus copy** and save the summary to the shared notes folder `C:\Users\ADMIN\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\side-session-notes\local_aaa8793d-d71d-45b8-b3ca-614765233b20\leaf-coverage-<date>.md`. Report it as a manual corpus check, not a test.
- [ ] **Step 3: Run the full suite and compile check**

Run: test command with `discover -s tests -v`, then `D:\finance_project\.venv\Scripts\python.exe -m compileall -q src config scripts`
Expected: all tests pass (report any pre-existing failures outside `test_extraction_*` / `test_calculation_*` separately); compileall prints nothing.

- [ ] **Step 4: Inspect the branch diff** (`git diff feat/calculation-core...HEAD --stat`): only files listed in this plan changed.
- [ ] **Step 5: Commit** — `feat(extraction): coverage and smoke scripts`.
- [ ] **Step 6: Report** files changed, tests and manual checks actually run (PostgreSQL store against the live database and the LLM smoke run are listed as not run unless the user ran them), coverage numbers, and remaining risks from spec §12.

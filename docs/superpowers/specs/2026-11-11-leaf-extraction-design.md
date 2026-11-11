# Leaf Extraction (Statement Line Items → Sourced Facts) — Design

Date: 2026-11-11
Status: approved; implemented on feat/leaf-extraction
Sub-project: 2 of 4 of "answering and serving"
Builds on: `docs/superpowers/specs/2026-11-10-calculation-core-design.md` (branch
`feat/calculation-core`, PR #2), especially §6, §8 and §16.

## 1. Context

The calculation core (sub-project 1) evaluates metrics from a `FactSet` of sourced leaf facts.
It does not produce those facts. This sub-project turns the stored tables of the requested
report into typed `Fact`s, deterministically first, with an LLM fallback that may only point
at a table cell whose value is then read and checked by code.

Inputs come from the metric tree: `tree.required_leaves(node_id, template, year)` returns
`LeafRequest(metric_id, year, period_kind)` sets. Sub-project 3 (reasoning agent) calls this
sub-project with those requests plus ticker and scope, and passes the resulting `FactSet` to
the sandbox and gates.

Corpus facts this design is built on (read-only probes of the local copy
`data/databases/corpus.sqlite3`, 2026-11-11):

| Finding | Consequence |
|---|---|
| One chunk per table: `chunk_id = "<doc_id>:table:<table_id>"`. | A table is the unit of evidence and citation. |
| Units are usually in column headers ("2023 VND", "31.12.2022 Triệu VND"); `table_unit_json` is often `unknown` and `tables.unit_scale` then stores 1.0. `TableAnalyzer.cell().unit_scale` also falls back to 1.0, and `TableAnalyzer.cell().unit` is `unknown` for joined headers such as "Năm tài chính kết thúc ngày 31 tháng 12 năm \| 2023 VND". | Per-header-cell unit parsing after removing dates and period phrases (§6.7); never the 1.0 fallback. |
| 140 of ~1,911 statement documents have no table tagged `income_statement`. ACB 2022: the bank income statement (page 10) is tagged `note`; the only `income_statement` table is a segment note (page 98). | `statement_kind` alone cannot select tables (§6.2 anchor rule). |
| Bank statements number lines with roman numerals and repeated local numbers (ACB 2022 `VI Cho vay khách hàng` → `1 Cho vay khách hàng` → `2 Dự phòng rủi ro cho vay khách hàng`). | Section blocks (§6.3). |
| Per ticker-year: both scopes 929, consolidated only 22, separate only 19. | Typed `ReportMissing`, no scope substitution. |
| 20 ticker/year/scope pairs have two documents (`scope_variant` 1, 2); 38 documents have scope `unknown`, 7 `aggregated`. | §5.4. |
| No ticker → template mapping exists. | `config/ticker_templates.yaml` (§4). |
| Only 113 of the 360 single-lookup questions (ids 1–360) name a metric-tree alias. | Coverage grows by adding leaves as data (§2, decision 1); measured by `leaf_coverage.py`. |

## 2. Agreed decisions

1. **Tree leaves only.** Extraction serves `LeafRequest`s for leaves defined in
   `config/metrics/*_leaves.yaml`. Ad-hoc note items not in the tree are out of scope. Adding a
   leaf is a data change, never a code change: the extractor reads only `LeafDef` fields
   (`codes`, `label_vi`, `aliases`, `parent`, `statement_kind`, `period_kind`, `source`,
   `dimension`); there are no per-leaf branches.
2. **Missing report → typed outcome only.** No automatic scope substitution. `ReportMissing`
   lists the scopes that do exist; sub-project 3 decides whether to re-request.
3. **Scope variants are all sources.** Two documents for one (ticker, year, scope) are both
   read; agreement merges, disagreement is a `Conflict`. `unknown` / `aggregated` documents
   never satisfy a request but appear in `available_scopes`.
4. **Ticker → template from config.** `config/ticker_templates.yaml` is hand-reviewed and the
   single source of truth; a read-only lint checks it against printed form codes.
5. **LLM fallback ships now, with the client injected** (paraphraser pattern). The LLM returns
   a cell pointer, never a number; the value is read and checked deterministically.
6. **Report t first, t+1 as fallback** for a year-t value.
7. **Report-direct evidence (approach A).** Leaf extraction reads the report's tables through a
   read-only `ReportStore`; it does not use the hybrid retriever, reranker or Chroma. Retrieval
   remains the evidence path for free-text answering in sub-project 3.

Rejected approaches: (B) one hybrid retrieval + rerank per leaf request — up to ~84 rerank
passes for a 7-company × 3-year × 4-leaf question, top-k can miss the statement table, results
depend on index state; (C) extract only from `PipelineResult` evidence — subqueries about a
derived metric do not retrieve the leaf rows its formula needs.

## 3. Components

New package `src/extraction/`. Production modules import via `src.*`. No retrieval, embedding,
GPU or network dependency; database access only through `ReportStore`.

| Module | Responsibility |
|---|---|
| `templates.py` | Load and validate `config/ticker_templates.yaml` → `TickerTemplates`. |
| `report_store.py` | `ReportDoc`, `TableRecord`, `ReportStore` protocol, `PostgresReportStore`, `SqliteReportStore`, `ReportStoreError`. |
| `table_view.py` | `TableView`: code column, normalised row labels, value columns with period and unit, table number locale. |
| `matching.py` | Candidate-table selection (kind + anchor rule), row matchers (code, exact label, section block, note label). |
| `outcomes.py` | `ExtractedFact`, `Candidate`, `CellLocation` and the outcome types (§7). |
| `extractor.py` | `LeafExtractor`, `ExtractionResult`; orchestration (§5). |
| `llm_fallback.py` | Prompt, `CellPointer` / `PointerAnswer` schemas, `structured_pointer_adapter`, shortlist and verifier (§8). |

Scripts (manual, read-only; never part of the unit-test suite):

| Script | Purpose |
|---|---|
| `scripts/leaf_coverage.py` | Run extraction for every leaf over every report of the local corpus copy; count outcomes and reasons per leaf and template. |
| `scripts/lint_ticker_templates.py` | Compare `config/ticker_templates.yaml` with form codes printed in each report; with `--draft`, print a first draft of the file for manual review. |
| `scripts/smoke_llm_fallback.py` | Run the fallback on a few real tables with a model the user configures. |

Changes to the calculation core: `src/calculation/facts.py` conflict resolution becomes
order-independent (§7.3, the §16 contract); `MetricTree` gains the read-only accessor
`leaves(template) -> tuple[LeafDef, ...]` (definition order), needed by the anchor rule and
section children, because leaves are otherwise private. `Fact` and `FactSource` are unchanged.

Unchanged: retrieval and its scoring, paraphrasing, `metric_catalog.py`, `query_pipeline.py`,
the PostgreSQL schema, Chroma index identity. New dependencies: none (psycopg, pydantic and
PyYAML are already pinned).

## 4. Ticker templates (`templates.py`, `config/ticker_templates.yaml`)

```yaml
# ticker: template      (template ∈ corporate | bank | securities | insurance)
ACB: bank
VJC: corporate
FTS: securities
```

- Load fails with `TemplateConfigError(file, ticker, reason)` on: a non-mapping top level, an
  unknown template value, a blank or non-string ticker, a duplicate ticker after upper-casing.
  There is no partial load.
- `TickerTemplates.get(ticker) -> str | None` (ticker upper-cased).
- `TickerTemplates.calculation_templates() -> dict[str, Template]` returns only `corporate` /
  `bank` entries, for `ExecutionContext.ticker_templates` in sub-project 3.
- The first version of the file is drafted by `lint_ticker_templates.py --draft` from form
  codes ("B02/TCTD", "B05/TCTD" → bank; "B02-DN", "B 02 – DN" → corporate; "B02-CTCK" →
  securities; insurance forms → insurance) and reviewed by hand before it is committed.

## 5. Data flow (`extractor.py`)

```python
extractor = LeafExtractor(store, tree, templates, fallback=None)
result = extractor.extract(ticker, scope, requests)   # -> ExtractionResult
```

### 5.1 Validation (raises before any store access)

- `scope` not in {`consolidated`, `separate`} → `ValueError`.
- An item of `requests` that is not a `LeafRequest` → `TypeError`.
- A request whose `period_kind` differs from the leaf definition's → `ValueError`.
- Requests with equal `(metric_id, year, period_kind)` are merged; outcomes are keyed by the
  distinct request, in first-seen order.

### 5.2 Template

- Ticker not in the config → every request gets `UnknownTicker`.
- Template `securities` or `insurance` → every request gets `UnsupportedTemplate`.
- `tree.leaf(metric_id, template)` missing → `NotApplicable` for that request.

### 5.3 Source order for a year-y request

1. **Primary:** report (ticker, y, scope), current-period column (§6.5).
2. **Fallback:** report (ticker, y+1, scope), comparative column (§6.5).

The fallback runs only when the primary outcome is `ReportMissing` or `NotFound`. A primary
`Ambiguous`, `Conflict`, `BlankValue` or `UnitUnknown` is final. Combining the two sources:

| Primary | Fallback | Outcome |
|---|---|---|
| `ReportMissing` | `ReportMissing` | `ReportMissing` (scopes of year y) |
| `ReportMissing` | `NotFound` | `NotFound`, warning `primary_report_missing` |
| `NotFound` | `ReportMissing` | `NotFound` (primary) |
| `NotFound` | `NotFound` | `NotFound`, `searched_doc_ids` from both |
| `ReportMissing` / `NotFound` | any other outcome | the fallback outcome (plus `primary_report_missing` when applicable) |

A fact from the fallback
carries the warning `from_next_year_report:<doc_id>`, and `restated=True` only when its column
header or the table context contains "trình bày lại", "điều chỉnh lại" or "restated" (folded
comparison); otherwise `restated=False`.

### 5.4 Report resolution

`store.documents(ticker, y)` is filtered to `scope`.

- No document → this source is `ReportMissing(available_scopes)`, where `available_scopes` is
  the sorted set of `scope` values of all documents for (ticker, y), including `unknown`,
  `aggregated`, `explanations`, `letters` (combined with the other source per §5.3).
- One or more documents (scope variants): each is extracted independently (§6). Then:
  - all documents that found the leaf agree within `max(scale)` → one fact, the finer-scale one
    (tie: lowest `scope_variant`, `None` first);
  - only some documents found the leaf → that fact, with warning
    `not_in_variant:<doc_id>` for each document without it;
  - values differ beyond tolerance → `Conflict(candidates)`;
  - if any document yields `Ambiguous`, `BlankValue` or `UnitUnknown` and no other document
    yields `Found`, that outcome is returned (in document-id order, first wins); if another
    document yields `Found`, the `Found` is returned with warning `variant_outcome:<doc_id>:<kind>`.

### 5.5 Inside one document

Pick candidate tables (§6.2) → match rows (§6.3, §6.4) → choose the period column (§6.5) →
parse the number (§6.6) → resolve the unit (§6.7) → build the fact (§6.8). Several matching
rows across the candidate tables: all with equal value → one fact sourced from the first table
in page order (ties: `table_id`); otherwise `Ambiguous(reason="multiple_rows")`.

### 5.6 LLM fallback

Runs only when the final deterministic outcome of §5.3 is `NotFound` and a document exists
(the primary documents if any, otherwise the fallback documents), and only if `fallback` was
injected. See §8.

### 5.7 Result

```python
@dataclass(frozen=True)
class ExtractionResult:
    ticker: str
    scope: Scope
    outcomes: tuple[Outcome, ...]          # one per distinct request, request order
    @property
    def facts(self) -> tuple[ExtractedFact, ...]: ...   # from Found outcomes
    def fact_set(self) -> FactSet: ...     # FactSet(f.fact for f in facts); may raise FactConflict
```

Within one `extract` call each document's tables are loaded once and each `TableView` is built
once. Nothing is cached across calls.

## 6. Deterministic matching (`table_view.py`, `matching.py`)

### 6.1 Table structure (`TableView`)

Built from `TableRecord.rows` with `TableAnalyzer(rows, context=record.context,
report_year=doc.report_year)`.

- **Header rows:** `TableAnalyzer.header_indices`. Data rows are all other rows.
- **Code column:**
  1. a column whose header text (folded) contains `ma so` is the code column;
  2. otherwise a column qualifies when its data cells contain ≥ 3 code-like values and ≥ 80% of
     its non-empty data cells are code-like. Code-like: `^\d{1,3}$` (corporate) or a bank
     enumerator: a roman numeral I–XXX (letters I, V, X only), `^\d{1,2}$`, or a single letter;
  3. a column whose header contains `thuyet minh` never qualifies (note numbers "29", "6.1");
  4. two or more qualifying columns → the table has no code column.
- **Row label:** `row_label(row, code_idx=code_column)`.
- **Normalised label:** `fold_text(label)` with one leading enumerator token removed (roman
  numeral I–XXX, 1–2 digits, or a single letter). No other rewriting.
- **Value columns:** columns other than the code and `thuyet minh` columns with at least one
  data cell that parses as a number.
- **Number locale:** `vi` if the table's value cells include at least one cell that parses only
  under `vi` and none that parses only under `en`; `en` for the reverse; `mixed` if both
  occur; `vi` if neither occurs (all cells unambiguous).

### 6.2 Candidate tables in one document

- **Statement leaf** (`source: statement`):
  1. tables whose `statement_kind` equals the leaf's;
  2. if those yield no matching row, the **anchor rule**: tables tagged `note` or `unknown`
     whose normalised row labels exactly equal the folded `label_vi` or an alias of
     ≥ min(3, n) distinct statement leaves of the same template and `statement_kind`, where n
     is the number of such leaves (corporate cash flow has 2). Anchors use labels, never codes:
     codes repeat across statements (02 and 20 occur in both the income statement and the cash
     flow statement). A table that anchors for two statement kinds is not a candidate for
     either. Rows inside an anchored table are then matched with the normal matchers.
- **Note leaf** (`source: notes`): tables tagged `note` or `unknown`.
- A table whose `rows` are not a list of lists of strings is skipped and listed in
  `skipped_tables` on the outcome, with warning `skipped_table:<chunk_id>`.

### 6.3 Row matching: corporate

- **Coded leaf:** the code cell equals one of `leaf.codes`, compared as integers ("01" = "1").
  Method `code`.
- **Table without a code column:** exact equality of the normalised label with the folded
  `label_vi` or any folded alias. Method `label`, warning `matched_by_label`.
- **Note leaf:** exact normalised-label equality. Method `label`.

### 6.4 Row matching: bank

- **Leaf without `parent`:** exact normalised-label equality with folded `label_vi` or any
  folded alias. Never substring matching (calculation-core spec §16). Method `label`.
- **Enumerator levels** (code cell, case-sensitive): *top* = one uppercase letter other than
  I, V, X (A, B, …); *section* = an uppercase roman numeral I–XXX; *item* = 1–2 digits;
  *sub-item* = one lowercase letter. Item and sub-item rows belong to the enclosing section.
- **Section block:** a section-level row followed by the rows up to (not including) the next
  top- or section-level row, or the next row whose normalised label starts with a total marker
  (`TOTAL_MARKERS` of `table_semantics`).
- **Section leaf** (a leaf that is the `parent` of others): the section-level row whose own
  normalised label exactly equals a folded label or alias of the section leaf or of one of its
  children, and whose block contains an exact alias match for at least one of its children
  (ACB: "VI Cho vay khách hàng"; section III "Tiền gửi và cho vay các TCTD khác" does not
  qualify even if a child label occurs in its block). Two qualifying rows with different values
  → `Ambiguous(reason="multiple_rows")`. Method `section_position`.
- **Child leaf** (has `parent`): matched by exact alias equality only inside a block of its
  parent section. Method `section_position`.
- **No code column (enumerators lost in OCR):** a section is recognised only by the pattern
  "row with normalised label X, immediately followed by a row with the same label X, followed
  within 3 rows by an exact match of another child of the section". The first X row is the
  section, the second is the matching child. Anything else for a section or child leaf →
  `NotFound(reason="section_unresolved")`.

### 6.5 Period column

`period_info(column_header, report_year=R)` per value column (R = `ReportDoc.report_year`).

| Source | `flow` | `closing` |
|---|---|---|
| primary (y = R) | role CURRENT, or explicit year y with `period_type != "start"` | role CLOSING or CURRENT, or explicit year y with `period_type == "end"` |
| fallback (y = R−1) | role PRIOR, or explicit year y with `period_type != "start"` | role OPENING or PRIOR, or explicit year y with `period_type == "end"` |

- A column with role UNKNOWN and no year never matches; columns are never chosen by position.
- Several matching columns with equal parsed values → one (the leftmost). Different values →
  `Ambiguous(reason="multiple_period_columns")`. None → `NotFound(reason="no_period_column")`.

### 6.6 Number

- `parse_financial_number(raw, locale=table_locale)`; locale `mixed` →
  `Ambiguous(reason="number_locale")`.
- Status `missing` (empty, "-", "–", "n/a") → `BlankValue`. Never 0.
- Status `invalid` or `ambiguous` → `Ambiguous(reason="number_unparsed")` with the cell.
- The sign is kept as printed (parentheses or minus → negative).

### 6.7 Unit

- **Why not `TableAnalyzer.cell().unit` directly:** it parses the joined multi-row header, and
  `parse_financial_unit` rejects a bare currency surrounded by other words. Probe on real
  headers (2026-11-11): VJC 2023 "Năm tài chính kết thúc ngày 31 tháng 12 năm | 2023 VND" →
  `unknown`; "Số cuối năm VND" and "31/12/2023 VND" → `unknown`. The cleaner is shared with
  indexing and is not changed here.
- **Column unit (`header_unit`):** the column's header cells are taken one header row at a
  time, nearest row first. From each cell, dates (`dd.mm.yyyy`, `dd/mm/yyyy`, `dd-mm-yyyy`,
  "ngày d tháng m [năm yyyy]"), four-digit years and the period phrases (`số cuối năm`,
  `số đầu năm`, `số cuối kỳ`, `số đầu kỳ`, `cuối năm`, `đầu năm`, `năm nay`, `năm trước`,
  `kỳ này`, `kỳ trước`, `tại ngày`, `vào ngày`, `tại`, `ngày`, `cho năm tài chính kết thúc`,
  `năm tài chính kết thúc`, `cho năm`, `năm`) are removed; dates may omit the leading "ngày"
  ("31 tháng 12 năm 2025"); an empty remainder is skipped; the remainder is parsed with
  `parse_financial_unit("", column_header=remainder)`. The first `valid` unit is the column
  unit; a later cell with a different `valid` unit, or an `ambiguous` cell, makes the unit
  `ambiguous`. Probe results after stripping: "2023 VND" → VND ×1; "Số cuối năm Triệu đồng" →
  VND ×10^6; "31.12.2022Triệu VND" → VND ×10^6.
- **No column unit:** then `TableAnalyzer.table_unit` (header text of the whole table), then the
  context unit (`context_unit(record.context)`):
  - a declaration such as "Đơn vị tính: triệu đồng", parsed by `parse_financial_unit`;
  - otherwise a unit of at most three words that ends the context right after its last date
    or year, with "(tiếp theo)" ignored. This is the Big-4 layout where the unit is a line of
    its own above the table: "… vào ngày 31 tháng 12 năm 2016 VND", "… năm 2017 Ngàn VND".
  - `source_text` is the declaration or the trailing unit, not the whole context.
- **Corpus spellings (headers and context):** before parsing,
  - an OCR-mixed "Đơn vị tinh" is read as "Đơn vị tính";
  - "đồng Việt Nam" or "Việt Nam đồng" is read as VND;
  - a "VND" glued to the word before it ("Số cuối nămVND") is split off.
- **Measured effect:** on 2026-11-11 this was measured over the 586 corporate reports that had
  `UnitUnknown(missing)`. 6,415 of 9,120 such requests became Found. No Found outcome changed,
  and no report's scales disagreed.
- Required: `unit_status == "valid"` and
  - `dimension: money` → `unit_kind == "monetary"` and `currency == "VND"`;
  - `dimension: money_per_share` → `unit_kind == "monetary"`, `currency == "VND"`,
    `scale_multiplier == 1`.
- Otherwise `UnitUnknown(reason ∈ {missing, ambiguous, kind_mismatch, currency})`.
- `unit_scale` (the 1.0 fallback) is never read.

### 6.8 Fact construction

- `value = Quantity(parsed × scale_multiplier, leaf.dimension)` (money in đồng).
- `FactSource(chunk_id=record.chunk_id, doc_id, page=record.page, row_label=<original label
  text>, raw_text=<raw cell text>, unit_text=<unit.source_text, or the column header when the
  unit came from it>, scale=unit.scale_multiplier, column_header=<header>)`.
- `Fact(metric_id, ticker, fiscal_year=y, scope, period_kind, value, source, restated)`.
- `ExtractedFact(fact, method, warnings)`.

### 6.9 Continuation tables (v1 decision)

A table with no header rows does not inherit units or periods from the previous page's table;
its cells give `UnitUnknown` or `NotFound(no_period_column)`. `leaf_coverage.py` reports how
often this occurs; inheritance is a later, separately approved change.

## 7. Outcomes and errors (`outcomes.py`)

### 7.1 Outcome types

All are frozen dataclasses with `ticker`, `scope`, `request: LeafRequest`.

| Outcome | When | Payload |
|---|---|---|
| `Found` | one value, all checks passed | `extracted: ExtractedFact` |
| `NotFound` | no row / no period column / section unresolved / no candidate table / LLM pointer rejected | `reason`, `searched_doc_ids`, `skipped_tables`, `warnings` |
| `BlankValue` | matched cell is empty or a dash | `location: CellLocation` |
| `Ambiguous` | several values in one document (rows, period columns, number locale, unparsed number) | `reason`, `candidates: tuple[Candidate, ...]` |
| `Conflict` | scope-variant documents disagree | `candidates` (one per document) |
| `UnitUnknown` | unit missing, ambiguous, wrong kind or currency | `reason`, `location` |
| `ReportMissing` | no document for (ticker, y, scope) nor (ticker, y+1, scope) (§5.3 table) | `available_scopes: tuple[str, ...]` |
| `NotApplicable` | metric has no leaf for the template | `template` |
| `UnsupportedTemplate` | ticker template is `securities` or `insurance` | `template` |
| `UnknownTicker` | ticker not in `ticker_templates.yaml` | — |

`CellLocation(chunk_id, doc_id, page, row_label, column_header, raw_text)`;
`Candidate(value: Decimal | None, location: CellLocation)`;
`ExtractedFact(fact: Fact, method: Literal["code", "label", "section_position",
"llm_located"], warnings: tuple[str, ...])`.

### 7.2 Raised errors

- Caller errors (§5.1): `ValueError` / `TypeError`, before any store access.
- Configuration: `TemplateConfigError` at load; `TreeError` from the metric tree as today.
- Store failure (connection, query): `ReportStoreError(operation, error_class)`; propagates,
  carries the exception class name only (convention of `multi_query`).
- Warnings and reasons are fixed-format strings (a code plus ids). Raw LLM text and database
  exception messages never appear in outcomes, warnings or logs.

### 7.3 `FactSet` conflict resolution (calculation core, §16 contract)

`FactSet.__init__` groups facts by key first, then per key:

1. exactly one fact with `restated=True` → it wins; warning `restated value used for <key>`;
2. two or more restated facts → they must pairwise agree within `max(scale)` of each pair,
   else `FactConflict`; the finest-scale restated fact is kept (tie: first in input order);
3. no restated fact → all pairs must agree within `max(scale)` of each pair, else
   `FactConflict`; the finest-scale fact is kept (tie: first in input order);
4. a dimension mismatch inside a key → `FactConflict`.

The result is independent of input order except for exact ties, which keep the first. The
existing `test_calculation_facts.py` cases keep passing.

## 8. LLM fallback (`llm_fallback.py`)

### 8.1 Interface

```python
class CellPointer(BaseModel):          # extra="forbid"
    table_ref: str                     # "T1".."T5"
    row_idx: int                       # >= 0
    col_idx: int                       # >= 0

class PointerAnswer(BaseModel):        # extra="forbid"
    status: Literal["found", "not_present"]
    pointer: CellPointer | None        # required iff status == "found"

def structured_pointer_adapter(client) -> Callable[[str], PointerAnswer | None]: ...

class LlmCellLocator:
    def __init__(self, llm: Callable[[str], PointerAnswer | None], *,
                 llm_errors: tuple[type[Exception], ...] = (TimeoutError, ConnectionError, OSError),
                 max_tables: int = 5, max_rows: int = 60, max_chars: int = 12000): ...
```

`structured_pointer_adapter` follows `paraphraser.structured_llm_adapter`: it calls
`client.with_structured_output(PointerAnswer, include_raw=True)` once, returns `parsed` only
when `parsing_error` is None, else `None`; it imports nothing from langchain and never returns
raw text. `LeafExtractor(fallback=LlmCellLocator(...))` enables the fallback.

### 8.2 Shortlist and prompt

- Candidate tables of the document (statement kind first, then all `note`/`unknown` tables),
  ranked by the number of distinct folded tokens shared with the leaf's `label_vi` and aliases,
  +2 when `statement_kind` equals the leaf's; ties by page, then `table_id`. Top `max_tables`.
- Each table is rendered as an indexed grid (`row_idx`, `col_idx`, cell text), at most
  `max_rows` rows; the prompt is cut to `max_chars` by dropping lowest-ranked tables whole.
- The Vietnamese prompt states the leaf label and aliases, statement kind, fiscal year,
  `period_kind` (flow: "trong năm"; closing: "số dư cuối năm"), and that the answer is a cell
  pointer or `not_present`, never a number.

### 8.3 Verifier

A pointer becomes `Found(method="llm_located")` only if all hold:

1. `table_ref` names a shortlisted table; `row_idx`, `col_idx` are in bounds;
2. the row is a data row; the column is a value column (not code, label or `thuyet minh`);
3. the column passes the §6.5 period rule for this source;
4. the number parses (§6.6) and the unit is valid for the leaf (§6.7);
5. the row's normalised label shares at least one token of length ≥ 3 with the folded
   `label_vi` or an alias.

### 8.4 Failure handling

At most one LLM call per (request, document); no retry.

| Event | Outcome |
|---|---|
| exception in `llm_errors` | `NotFound`, warning `llm_fallback_error:<ExceptionClass>` |
| other exception | propagates |
| adapter returned `None` | `NotFound`, warning `llm_fallback_unparsed` |
| `not_present` | `NotFound` (deterministic reason kept) |
| verifier rejects | `NotFound(reason="llm_pointer_rejected")`, warning `llm_pointer_rejected:<check>` |
| blank or ambiguous cell at the pointer | `NotFound(reason="llm_pointer_rejected")` |

## 9. Report store (`report_store.py`)

```python
@dataclass(frozen=True)
class ReportDoc:
    doc_id: str; ticker: str; report_year: int; scope: str; scope_variant: int | None

@dataclass(frozen=True)
class TableRecord:
    chunk_id: str; doc_id: str; table_id: int; page: int
    context: str; statement_kind: str | None; rows: object   # validated by TableView

class ReportStore(Protocol):
    def documents(self, ticker: str, year: int) -> list[ReportDoc]: ...
    def tables(self, doc_id: str) -> list[TableRecord]: ...    # page, table_id order
```

- SQL: `documents` by `(ticker, report_year)`; `tables` joined to `chunks` on
  `(table_key, doc_id)` with `chunk_type = 'table'` for `chunk_id`. A table without a chunk is
  skipped with warning `table_without_chunk:<table_key>`.
- `PostgresReportStore(connection_params=None)`: one connection per `extract` call, a
  `REPEATABLE READ, READ ONLY` transaction (as `bm25.load_corpus_documents`); closed in
  `finally`.
- `SqliteReportStore(path)`: `sqlite3.connect("file:<path>?mode=ro", uri=True)`; for scripts
  and tests only.
- `rows_json` is decoded with `json.loads`; a decode error marks the table as skipped (§6.2).

## 10. Testing

unittest, offline and deterministic (`tests/test_extraction_*.py`). Fixtures are small
synthetic tables modelled on real ones, served by an in-memory `ReportStore`.

- Templates: real config file loads; fixture files with unknown template, duplicate ticker,
  blank ticker fail with `TemplateConfigError`; `calculation_templates()` drops
  securities/insurance.
- TableView: code column by header, by content, `thuyet minh` exclusion, two candidates → none;
  label normalisation; locale `vi` / `en` / `mixed`.
- Corporate (VJC 2023 income statement shape): codes 10, 11, 23, 50, 60; parentheses negative;
  VND scale 1; code "1" vs "01"; table without code column → `label` + warning.
- Bank (ACB 2022 shapes): section VI block (section, gross child, provision child); a gross
  label outside the block is not matched; `TỔNG TÀI SẢN` and `TỔNG NGUỒN VỐN` equal → one fact,
  different → `Ambiguous`; enumerator column lost (pattern accepted, other shapes →
  `section_unresolved`); triệu VND scale 10^6.
- Anchor rule: income statement tagged `note` found; segment-note decoy tagged
  `income_statement` not used; a note table with 2 label anchors (of ≥ 3 leaves) is not a
  candidate; corporate cash flow tagged `unknown` anchors with its 2 labels; an income
  statement tagged `unknown` with codes 02 and 20 does not anchor as a cash flow statement.
- Enumerator levels: `I`, `V`, `X` are sections, `A`/`B` are top level, `a`/`b` stay inside
  the section block.
- Periods: current vs prior columns; closing via "Số cuối năm" and explicit "31.12.2022";
  fallback via "Số đầu năm" / "01.01.R"; UNKNOWN column never used; two period columns with
  different values → `Ambiguous`.
- Header units: the real VJC two-row header gives VND ×1; "Số cuối năm Triệu đồng" gives
  ×10^6; two header rows with different valid units → `UnitUnknown(ambiguous)`; no column
  unit but "Đơn vị tính: triệu đồng" in the context → ×10^6.
- Numbers and units: dash → `BlankValue`; unknown unit → `UnitUnknown(missing)`; percent unit
  on a money leaf → `kind_mismatch`; USD → `currency`; EPS at scale ≠ 1 → `kind_mismatch`.
- Reports: primary missing → t+1 fallback with warning; restatement marker → `restated=True`;
  both missing → `ReportMissing` with `available_scopes` incl. `unknown`; scope variants agree,
  disagree (`Conflict`), one-sided (warning); primary `Ambiguous` does not fall back.
- Templates at runtime: securities → `UnsupportedTemplate`; unlisted ticker → `UnknownTicker`;
  corporate-only leaf for a bank → `NotApplicable`; `period_kind` mismatch → `ValueError`.
- Result: request order, duplicate requests merged, `fact_set()` builds a `FactSet`; malformed
  `rows_json` → skipped and reported.
- FactSet: every permutation of three facts of one key gives the same kept fact or the same
  `FactConflict`; one restated among three; two restated disagreeing.
- LLM fallback with a fake client: valid pointer → `llm_located`; out of bounds; header cell;
  code column; wrong period; no shared label token; unparsed (`None`); `not_present`; an
  `llm_errors` exception; a non-listed exception propagates; fallback not called for
  `Ambiguous` / `ReportMissing`; adapter returns `None` on `parsing_error`.
- Store: `SqliteReportStore` against a temporary database built from
  `src/indexing/schema.SCHEMA`; read-only mode rejects writes; tables ordered by page and
  `table_id`; chunk join.

Not unit-tested (reported separately): `PostgresReportStore` against the live database,
`leaf_coverage.py`, `lint_ticker_templates.py`, `smoke_llm_fallback.py` with a real model.

Verification: `python -m unittest discover -s tests -v` and
`python -m compileall -q src config scripts`.

## 11. Out of scope

Wiring extraction into `query_pipeline` or the reasoning loop (sub-project 3); ad-hoc note
items not in the tree; header inheritance for continuation tables; securities and insurance
templates; LLM model choice and live-provider verification; any change to retrieval, scoring,
paraphrasing, the PostgreSQL schema or Chroma.

## 12. Risks and open items

- Coverage of note leaves and bank labels depends on exact label equality; `leaf_coverage.py`
  measures it per leaf before a new leaf is trusted.
- Continuation tables without headers yield `UnitUnknown` / `NotFound` (§6.9).
- The anchor threshold (3 leaves) is a judgement call; the coverage script reports how many
  documents rely on it.
- Restatement detection relies on explicit markers; an unmarked restatement read from report
  t+1 is an ordinary fact (only used when report t lacks the value).
- `period_info` regexes do not recognise every header form (e.g. "31 tháng 12" without a
  year); such columns are UNKNOWN and never used, which can produce `no_period_column`.
- The first `ticker_templates.yaml` is drafted from OCR form codes and must be reviewed by hand.
- The LLM prompt and shortlist are not evaluated until sub-project 3 builds a labelled dev set.

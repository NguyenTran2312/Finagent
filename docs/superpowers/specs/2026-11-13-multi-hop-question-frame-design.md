# Multi-hop Questions and Year Ranges: Question Frame — Design

**Status:** approved in conversation, 2026-11-13.
**Branch:** `feat/reasoning-agent`.
**Builds on:** `2026-11-13-reasoning-agent-demo-design.md` (the reasoning agent).

## 1. Context

On the 100 gold questions with `gpt-6-luna`, after the paraphraser fixes:
- 54 / 56 direct questions get a valid plan.
- Only **5 / 42 multi-hop questions** do.

The calculation side can already answer these questions:
- The metric tree has the ratios the questions use: `roe`, `roa`, `liabilities_to_equity`, `quick_ratio`, `current_ratio`, `cfo_margin`, `cfo_to_profit_after_tax`, `cfo_to_current_liabilities`, `days_inventory`, `interest_coverage`, `asset_turnover`, `gross_margin`, `net_margin`, `selling_and_admin_expenses`, `long_term_assets_share` and others.
- The helpers include `filter_by`, `median`, `argmax`, `argmin`, `count_if`, `mean`, `growth_pct` and `pct_point_change`.
- The solver declares its own `needs` and gathers missing data between attempts.

The bottleneck is the plan.
- **`QueryPlan` allows only one (ticker, year, metric) per subquery, and at most 8 subqueries.** A typical multi-hop question is "9 tickers × 2 years × 3 metrics, filter by a condition, pick the extreme": about 54 leaves. The model cannot express it.
- **The validator then checks the text of every subquery.** The failures seen are `search_text_conflict`, `missing_year`, `missing_ticker`, `too_many_subqueries` and `duplicate_subquery`.
- **"CFO" is flagged as an unknown ticker.** This alone sends 8 questions to `needs_clarification`.

## 2. Agreed decisions

| # | Decision |
|---|---|
| D1 | Success means multi-hop questions reach the solver with a valid plan. End-to-end accuracy is measured in the evaluation spec, not here. |
| D2 | Approach A: an optional compact `frame` on `QueryPlan`. The validator expands it into ordinary leaf subqueries, so every existing consumer is unchanged. |
| D3 | Ownership ratios in subsidiaries (#97, #106) are out of scope. They need an investee-name field and a notes extraction path. |
| D4 | Metric mapping stays strict: exact node, or an accepted derived-metric proposal, or `metric_unmapped`. The solver never invents a formula for an unmapped metric. |
| D5 | The per-question LLM cap stays at 13 calls. |

## 3. Schema (`src/query_paraphrase/schemas.py`)

New contracts, using the same strict `_Contract` base:

```python
class FrameEntity(_Contract):
    ticker: NonEmptyStr | None
    company_name: NonEmptyStr | None

class FrameMetric(_Contract):
    metric: NonEmptyStr | None        # catalog id, as in Subquery.metric
    metric_text: NonEmptyStr | None   # wording from the question

class QuestionFrame(_Contract):
    entities: Annotated[list[FrameEntity], Field(min_length=1)]
    years: Annotated[list[Year], Field(min_length=1)]
    scope: ReportScope | None
    metrics: Annotated[list[FrameMetric], Field(min_length=1)]
    selection: NonEmptyStr            # the filter / selection / aggregation, in words
```

Changes to `QueryPlan`:
- **New field:** `frame: QuestionFrame | None = None`.
- **`subqueries` loses `min_length=1`.** A model validator requires at least one subquery when `frame` is None.
- **Exclusive forms:** a plan with a `frame` must have `subqueries == []`. This prevents mixing the two forms.

Entity and metric entries need at least one of their two fields. The validator rejects an entry with both fields null (`empty_frame_entry`).

## 4. Validator (`src/query_paraphrase/validator.py`)

### 4.1 Frame checks

These run instead of the per-subquery checks, before expansion.

- **Entities:**
  - `company_name` fills `ticker` through the registry, with the same rule as `_fill_tickers`.
  - Every ticker or company named in the question must be in `entities`. Otherwise: `missing_ticker`.
  - No entity may be absent from the question. Otherwise: `invented_ticker`.
  - An unresolved name gives `company_name_mismatch`.
- **Years:**
  - Every year in the question must be in `years`. Otherwise: `missing_year`.
  - A range must be fully expanded. Otherwise: `incomplete_year_range`. Range forms, after `fold_text`:
    - `A–B` / `A-B` (en dash or hyphen);
    - `tu (nam) A den (nam) B`;
    - `giai doan A(-|–| den )B`;
    - `tu (nam) A sang (nam) B`.
    The range is `A..B` inclusive, with `A < B` and `B − A ≤ 15`.
  - No year outside the question's years and ranges. Otherwise: `invented_year`.
  - The year before a range, needed for "so với năm trước", is not part of the frame. The solver requests it through `needs`.
- **Metrics:** the existing catalog normalisation (`metric_not_normalized`) and the `invented_metric_text` check, applied to each `FrameMetric`.
- **Scope:** the existing scope rule ("công ty mẹ X" → separate), applied to `frame.scope`.
- **Operation:** the existing `_operation_issues` intents. The "aggregate needs at least two operands" rule counts expanded leaves.

### 4.2 Expansion

After the frame checks pass, the validator returns a plan whose `subqueries` are generated:
- **Order:** entity, then year, then metric.
- **Id:** `"{ticker}-{year}-m{k}"`, where `k` is the 1-based metric index.
- **`question` / `search_query`:** `"{metric_text or metric} {ticker} {year}"`.
- **Other fields:** `scope` comes from the frame; `period`, `unit` and `document_types` are left empty.

The frame is kept on the returned plan. Generated subqueries skip `search_text_conflict` and `duplicate_subquery`, since they are consistent by construction.

**Leaf cap:** `ParaphraseSettings.max_frame_leaves`, default 160, with env override `FINAGENT_PARAPHRASE_MAX_FRAME_LEAVES`. More leaves gives `too_many_frame_leaves`, status `invalid`. `max_subqueries` (8) still applies to plans without a frame.

### 4.3 Acronyms

Add CFO, ROE, ROA, EBIT, EBITDA, LNST, LNTT, SG&A and D/E to `FINANCE_ACRONYMS` (`src/query_paraphrase/entities.py`). The test pins that none of them is a ticker in `data/code_stock.csv`.

## 5. Prompt (`src/query_paraphrase/prompts.py`)

Add one mode rule.
- **Use `frame` (with `subqueries: []`)** when the question covers a group of tickers, a year range, or a filter / selection / ranking / count over several values.
- **Otherwise use subqueries** as today.

Add one short frame example: a group of 3 tickers, 2 years, 2 metrics and a selection sentence. Repair feedback works unchanged, because issue ids for frame problems are `None` or name the entity / year.

## 6. Agent

### 6.1 Metric mapping (`src/agent/metrics.py`)

Pending subqueries are de-duplicated by `(metric, metric_text, template)` before the LLM prompt.
- Each distinct key is listed once, under its first subquery id.
- The choice is applied to every subquery with the same key.
- The catalog and alias paths are unchanged.
- The exact-match prompt rule from ed60aee and the derived-metric proposal path stay.

Unmapped subqueries are reported once per distinct metric text in the `metric_unmapped` refusal.

### 6.2 Gather

No code change. The agent prefetches every expanded leaf, as it does today. Deterministic extraction is cheap. The LLM locator stays capped by `max_locator_calls` (6). Missing data still ends as `facts_missing`.

### 6.3 Solver prompt (`src/agent/solver.py`)

When `plan.frame` is set, the "Subqueries" block is replaced by a frame block:
- tickers;
- years;
- scope;
- each metric text → node id;
- the `selection` sentence;
- one line saying the solver may add `needs` for the year before a range.

Everything else is unchanged: instructions, gates and repairs.

### 6.4 Demo (`src/agent/present.py`)

`plan_data` adds a `frame` entry with tickers, years, scope, metrics and selection when the frame is present. The page shows that entry instead of the leaf table. The leaf list stays in the event data.

## 7. Testing

Offline unit tests (no model, GPU or network):
- **Schema:** frame-only plan accepted; neither form rejected; both forms rejected; empty entry rejected.
- **Range detection:** each range form, en dash vs hyphen, the `B − A ≤ 15` bound, and a range plus a separate year.
- **Validator:**
  - missing / invented ticker;
  - incomplete range;
  - invented year;
  - company-name fill;
  - leaf cap;
  - expansion order, ids and fields;
  - aggregate counted on leaves;
  - direct plans unchanged (existing tests stay green).
- **Acronyms:** none of the new acronyms is in the CSV; "CFO" no longer gives `unknown_entity`.
- **Mapping de-duplication:** one prompt line per distinct key; the choice is applied to all leaves; templates are kept apart.
- **Solver prompt:** the frame block is present for a frame plan and the subquery block is unchanged otherwise.
- **Agent:** a scripted-LLM run of one frame question (2 tickers × 2 years, `argmax`) ends `answered`.

## 8. Done criteria

1. The offline suite is green.
2. **Live paraphrase eval** (`gpt-6-luna`, with the agent's one repair round):
   - **≥ 25 / 42 multi-hop plans valid** (5 today);
   - **direct questions ≥ 52 / 56 valid** (no regression).
   - About 150 LLM calls. Run only with the user's go-ahead.
3. **Live agent runs** on #440 (single-company year range), #469 (group filter + mean) and #457 (count). Results are reported; no accuracy threshold.

## 9. Out of scope

- Subsidiary / investee ownership questions (#97, #106).
- What-if scenarios ("nếu chi phí lãi vay tăng 20%"). The solver may handle some, but they are not targeted.
- Batch accuracy evaluation and thresholds.
- Changes to the calculation core, extractor or retriever.

## 10. Risks

- **Prefetch size.** 160 leaves of a derived metric can need about 3 base leaves each. Extraction is SQLite-only, but a cold run of a large frame may take tens of seconds. This is reported in the step timings.
- **Metric coverage.** Questions with ad-hoc ratios ("tỷ số dồn tích", "giá trị tài sản ròng chịu áp lực thanh lý") depend on an accepted derived-metric proposal. If the proposal is rejected, the agent refuses with `metric_unmapped`, by D4.
- **Selection is free text.** The validator cannot check that `selection` matches the question. A wrong selection is caught only by the solver's gates or not at all. This is the same exposure as `operation_description` today.

# Calculation Core (Metric Tree, Helper Library, Sandbox, Gates) — Design

Date: 2026-11-10
Status: draft for review
Sub-project: 1 of 4 of "answering and serving"

## 1. Context

Finagent retrieves evidence (BM25 + BGE-M3 dense + RRF + cross-encoder) and, since the
query-paraphrasing work, returns evidence grouped per subquery (`build_query_pipeline` →
`PipelineResult`). It does not yet compute or answer anything.

ViFinQA (1,012 Vietnamese questions, `data/questions/questions.jsonl`) needs much more than
lookups. A keyword classification of all questions (approximate counts):

| Need | Count | Example id |
|---|---|---|
| Single lookup | ~360 (mostly ids 1–360) | 1 |
| Ratio / share / percent | ~306 | 443 |
| Difference / change | ~147 | 760 |
| Growth % between years (1 CAGR) | ~75 | 606, 370 |
| Average, incl. average balances | ~119 / 47 | 856, 398 |
| Argmax / argmin ("cao nhất", "năm nào") | ~240 | 363, 1000 |
| Conditional filter ("xét các DN có …") | ~134 | 401 |
| Median of a group | 39 | 443 |
| Count ("có bao nhiêu") | 21 | 965 |
| Scenario ("nếu … giảm 10%") | 9 | 435 |
| Percentage points ("điểm phần trăm") | 23 | 393 |

Requested units: tỷ, triệu, nghìn tỷ, trăm tỷ đồng, %, điểm phần trăm, lần, vòng, ngày,
đồng/cổ phiếu. Up to 7 companies × several years per question. Templates: ~719 corporate
questions, ~214 bank, ~11 securities. ViFinQA ships no answers, programs or gold evidence.

References used: FinanceReasoning (ACL 2025, arXiv 2506.05828: a 3,133-function library with
documented units/constraints; formula misuse and calculation errors dominate hard-question
failures; its evaluator uses 0.2% relative tolerance but normalizes units loosely and runs
code with unrestricted `exec`), FinAgent-RAG (arXiv 2605.05409, withdrawn; idea only:
decompose → program-of-thought → self-verifier → refine, max 3 iterations), FinSage
(arXiv 2504.14493 and its repo: per-subquery answer + merge).

## 2. Agreed decisions

1. The LLM reasons and writes Python; it never does arithmetic that reaches the user
   unchecked. Generated code runs in a sandbox and the result must pass deterministic gates,
   including agreement with the LLM's own claimed answer.
2. Metrics form a dependency tree. Derived metrics are roots/inner nodes; leaves are line
   items directly retrievable from the statements or notes. Retrieval only targets leaves.
3. The LLM may propose a decomposition for a metric missing from the tree; proposals are
   validated by code before use and logged for review.
4. Version 1 covers the corporate (Circular 200) and bank statement templates.
5. Representation is hybrid ("approach C"): leaves as data, derived nodes as declarative
   entries with a safe formula language, helpers in Python.

## 3. Overall decomposition (4 sub-projects)

1. **Calculation core** (this spec): metric tree, units, formula language, facts, helper
   library, sandboxed executor, gates. Pure Python; no LLM, GPU, network or database.
2. **Leaf extraction**: retrieved table/page chunks → typed `Fact`s, deterministic first
   (`TableAnalyzer`: row by statement code/alias, column by period, unit scale), LLM only as
   fallback, with a check that the value occurs in the source.
3. **Reasoning agent**: LLM selects or proposes tree nodes, requests leaves, writes
   `solution()`, runs the sandbox and gates, repairs up to N times, then answers in Vietnamese
   with citations or refuses. Owns the LLM choice, prompts and the hand-labelled dev set.
4. **Serving**: HTTP API, chat web UI, sessions/history.

Reranker fine-tuning (pairwise/DPO-style, hard negatives) follows after these.

Each sub-project has its own spec → plan → implementation cycle.

## 4. Components (`src/calculation/`)

| Module | Responsibility |
|---|---|
| `units.py` | `Dimension`, `Quantity`, dimension algebra, VN scales, display forms |
| `metric_tree.py` | `LeafDef`, `DerivedDef`, loading and validating `config/metrics/*.yaml`, leaf expansion |
| `expressions.py` | Safe formula language: parse, dimension inference, evaluation |
| `proposals.py` | Validation of LLM-proposed derived nodes |
| `facts.py` | `FactSource`, `Fact`, `FactSet` (keyed lookup, conflict rejection) |
| `helpers.py` | Functions available to generated code; records a trace |
| `sandbox.py` | AST whitelist check and isolated execution of `solution()` |
| `gates.py` | Post-execution checks producing a `GateReport`; `compare_answers` |

Data files in `config/metrics/`: `corporate_leaves.yaml`, `bank_leaves.yaml`, `derived.yaml`.

Reuse, not duplication:
- `src/preprocess/cleaner.py`: `parse_vn_decimal`, `parse_financial_unit`,
  `parse_requested_unit`, `unit_conversion`, `fold_text`.
- `src/indexing/table_semantics.py`: `StatementKind`.
- Node ids reuse the English ids of `src/query_paraphrase/metric_catalog.py`
  (`net_revenue`, `profit_before_tax`, ...). The catalog itself is not changed here.

Imports follow the repository convention: production modules import via `src.*`.

Unchanged: retrieval, paraphrasing, PostgreSQL schema, Chroma index identity, public APIs.
New dependencies: none (PyYAML is already pinned).

## 5. Units (`units.py`)

- `Dimension`: `money`, `ratio`, `money_per_share`, `shares`, `days`, `count`.
- `Quantity(value: Decimal, dimension: Dimension)`, frozen. Money is stored in đồng.
  Floats are never used for values.
- Algebra (anything else raises `UnitError`):
  - money ± money → money; ratio ± ratio → ratio; days ± days → days
  - money ÷ money → ratio; money ÷ shares → money_per_share; ratio ÷ ratio → ratio
  - money × ratio → money; ratio × ratio → ratio; any × dimensionless constant → same
  - days: only a node declared `dimension: days` may produce days, and its formula must have
    the form `<ratio expression> * 365`
- Display forms: money at a scale (đồng, nghìn, triệu, tỷ, trăm tỷ, nghìn tỷ), `percent`,
  `percentage_points` (only for a difference of two ratios), `times`, `turns`, `days`,
  `dong_per_share`. The requested display form comes from `parse_requested_unit(question)`
  plus explicit markers ("điểm phần trăm", "lần", "vòng", "ngày").
- Rounding happens once, at the end, to the requested unit and precision (2 decimals in the
  display unit when the question states none). No intermediate rounding.

## 6. Metric tree (`metric_tree.py`, `config/metrics/`)

### 6.1 Leaf entries (per template file)

```yaml
- id: profit_before_tax
  label_vi: Tổng lợi nhuận kế toán trước thuế
  aliases: [lntt, lợi nhuận trước thuế, tổng lợi nhuận kế toán trước thuế]
  statement_kind: income_statement
  codes: ["50"]
  period_kind: flow            # flow | closing
  source: statement            # statement | notes
  dimension: money
```

Statement codes are matched against the code column found by `TableAnalyzer`. Codes in this
spec are illustrative (Circular 200); every code in the data files must be verified against
the corpus during implementation. Bank leaves (Circular 49 layout) may rely on labels where
codes are absent; such leaves set `codes: []` and need at least two aliases. Note-sourced
leaves (e.g. board remuneration, subsidiary voting ratios) set `source: notes`.

### 6.2 Derived entries (`derived.yaml`)

```yaml
- id: interest_coverage
  label_vi: Hệ số khả năng thanh toán lãi vay
  aliases: [hệ số khả năng thanh toán lãi vay, icr]
  templates: [corporate]
  dimension: ratio
  display: times
  formulas:
    - (profit_before_tax + abs(interest_expense)) / abs(interest_expense)
  expected_range: [-1000, 1000]

- id: gross_profit
  templates: [corporate]
  dimension: money
  formulas:
    - gross_profit_reported              # leaf, code 20
    - net_revenue - abs(cost_of_goods_sold)
```

- `formulas` is ordered. The first formula whose leaves are all available is the value; the
  remaining computable formulas are cross-checks (§9).
- A derived node may be used by other derived nodes.
- `expected_range` is optional, in the node's dimension (ratios as plain ratios).

### 6.3 Load-time validation

Loading fails with `TreeError(file, node_id, reason)` on: unknown reference, cycle, formula
syntax error, dimension mismatch, duplicate id, an alias claimed by two nodes of the same
template, a derived node referencing a leaf of a template it does not list, a leaf with
neither codes nor two aliases. There is no partial load.

### 6.4 Expansion

`tree.required_leaves(node_id, template, year)` returns, per formula alternative, the set of
`LeafRequest(metric_id, year, period_kind)` with period offsets resolved
(`total_assets[t-1]` at year 2024 → `total_assets`, 2023, closing).

### 6.5 Version 1 coverage

Driven by the ViFinQA questions: roughly 40–60 derived metrics (margins, growth-based
metrics, D/E, liabilities/equity, current and quick ratio, interest coverage, CFO/LNST,
CFO/current liabilities, CFO margin, asset turnover on average assets, days inventory,
fixed-asset depreciation ratio, provision ratios, loan/deposit style bank ratios) and the
leaves they need, plus common lookup leaves. The exact list is produced in the plan from the
question set.

## 7. Formula language (`expressions.py`)

- Grammar: node ids, numeric literals, `+ - * /`, unary `-`, parentheses, period offsets
  `id[t-k]` (k ≥ 0, integer), and the single function `abs(...)`.
- Parsed with `ast.parse(text, mode="eval")` and checked node by node against a whitelist.
  Rejected: any other call, attributes, `**`, comparisons, boolean operators, subscripts other
  than `[t-k]`, names not in the tree, string or complex constants.
- Dimensions are inferred and checked before evaluation.
- Division by zero raises `CalculationError`; results are never `inf`, `nan` or a silent 0.

## 8. Facts (`facts.py`)

- `FactSource(chunk_id, doc_id, page, row_label, column_header, raw_text, unit_text)`; all
  fields required except `column_header` for page-text facts.
- `Fact(metric_id, ticker, fiscal_year, scope, period_kind, value: Quantity, source,
  restated: bool = False)`. `scope` ∈ {consolidated, separate}. A fact cannot be built without
  a source.
- Key: `(metric_id, ticker, fiscal_year, scope, period_kind)`.
- `FactSet` raises `FactConflict` on two different values for one key unless exactly one is
  marked `restated=True` (the restated one wins and a warning is recorded).
- Opening balance of year t is the closing balance of t−1; there is no separate opening key.
  If both the year-t report's prior column and the year-(t−1) report give it, they must agree
  within tolerance (§9) or the gates report an error.
- Signs are kept as printed (parentheses or minus → negative). Formulas use `abs()` where a
  magnitude is meant.

## 9. Helper library (`helpers.py`)

Built per execution by `build_helpers(tree, facts, context, trace)`; every call is appended
to the trace.

- Data access (the only way to reach data):
  - `fact(metric_id, ticker, year, period="flow"|"closing", scope=None)`
  - `metric(node_id, ticker, year, scope=None)`: evaluates a tree node from the facts
  - `scope=None` uses `context.default_scope`.
- Calculations: `growth_pct(new, old)`, `cagr(end, start, years)`,
  `avg_balance(metric_id, ticker, year)`, `pct_point_change(new_ratio, old_ratio)`, `mean`,
  `median`, `argmax(mapping)`, `argmin(mapping)` (return the winning key, ties → error),
  `filter_by(mapping, op, threshold)` (returns the sub-mapping whose values satisfy
  `value op threshold`, `op` ∈ {"<", "<=", ">", ">=", "=="}), `count_if(values, op, threshold)`.
  Comprehensions may be used for compound conditions.
- Result: `answer(value, unit)` where `value` is a `Quantity`, year, ticker, count or boolean.
- Each helper has a docstring with inputs, units and constraints (FinanceReasoning style);
  these docstrings are what sub-project 3 shows the LLM.

`ExecutionContext(default_scope, ticker_templates, question_literals, requested_unit)` is
passed in by the caller.

## 10. Sandbox (`sandbox.py`)

- `check_code(code) -> list[SyntaxRejected]`: the module must contain exactly one
  `def solution():` and nothing else at top level.
  - Allowed: assignment, augmented assignment, `if`, `for`, list/dict/generator
    comprehensions, `return`, arithmetic (`+ - * /`), comparisons, `and/or/not`, conditional
    expressions, list/tuple/dict literals, indexing, calls to helpers and to `len`, `min`,
    `max`, `sum`, `abs`, `sorted`, `range`, `enumerate`, `zip`, `list`, `dict`.
  - Forbidden: `import`, attribute access, any name containing `__`, `while`, `lambda`, `try`,
    `with`, `global`/`nonlocal`, `class`, nested `def`, `**`, `eval`, `exec`, `open`,
    `getattr`, `round`, f-strings and string formatting, starred calls.
- `run(code, tree, facts, context, timeout=2.0) -> ExecutionResult | ExecutionFailure`.
  - Runs in a separate process (`multiprocessing`, spawn) with `__builtins__` replaced by the
    whitelist and only helpers in scope. The timeout counts from when `solution()` starts.
  - Output is size-capped. On Linux an address-space limit is applied; on Windows no memory
    cap is enforced (documented risk; the timeout and the AST whitelist still apply).
  - `ExecutionResult(answer, trace)`;
    `ExecutionFailure(kind ∈ {rejected, timeout, calculation_error, runtime_error}, detail)`.
    `detail` holds the rejected node and line, or the exception class name only.

## 11. Gates (`gates.py`)

`run_gates(code, result, tree, facts, context, claimed=None) -> GateReport(passed, issues)`;
each `GateIssue(code, severity ∈ {error, warning}, message, line | trace_step)`.

| Gate | Check | Severity |
|---|---|---|
| provenance | Every numeric literal is an allowed constant (0, 1, 2, 4, 12, 100, 365), a number stated in the question (`context.question_literals`, e.g. "20 tỷ", "1,5 lần", "10%"), or a year. Every fact in the trace exists and has a source. | error |
| formula | If the question or plan names a tree node, the trace contains `metric(that id)`. | error |
| unit_type | Result dimension and kind match the request: money vs ratio vs percentage points vs days vs times; year / ticker / count / boolean for "năm nào", "công ty nào", "có bao nhiêu", yes/no. | error |
| sanity | Within the node's `expected_range`; money magnitude plausible (< 10^16 đồng). | warning |
| cross_check | Reported vs derived alternatives agree; opening = prior closing. Tolerance: one unit of the source's printed scale. | error (warning if a restatement is marked) |
| agreement | `compare_answers(claimed, executed, requested_unit)`: ≤ 0.2% relative after rounding to the requested precision (2 decimals in the display unit when the question states none); exact for year, ticker, count, boolean; both zero → equal. Only runs when `claimed` is given. | error |

`question_literals(question)` (in `gates.py`) extracts the numbers stated in the question
using `parse_vn_decimal`, including percentages as both `10` and `0.10`.

## 12. Error handling

- Load-time tree errors stop startup with file, node and reason.
- All runtime failures are typed values; raw exception text and LLM output never reach users.
- Not allowed anywhere: choosing between conflicting facts silently, treating missing as 0,
  intermediate rounding, defaulting an ambiguous source unit (fact creation fails instead).

## 13. Testing

unittest, offline and deterministic (`tests/test_calculation_*.py`):

- Units: algebra and `UnitError`, all VN scales, percent vs percentage points, Decimal
  exactness.
- Expressions: valid formulas; every forbidden construct rejected; dimension errors before
  evaluation; division by zero → `CalculationError`.
- Tree: the real `config/metrics/` files pass a lint test; fixture trees with cycles, unknown
  leaves, dimension mismatches and duplicate aliases fail with the expected `TreeError`;
  alternative formulas and cross-check pairing.
- Proposals: valid accepted; redefinition and wrong-template leaves rejected.
- Facts: source required, conflicts, restatement, opening = prior closing, signs.
- Sandbox escape suite: `import os`, `__import__`, `().__class__.__bases__`, `getattr`,
  `open`, `eval`, `while True`, huge comprehension (timeout), unknown names, missing or extra
  top-level definitions.
- Golden programs: ~15 hand-written `solution()`s over synthetic facts, modelled on question
  ids 1, 760, 606, 976, 363, 443, 435, 393, 370, 1000, 965 and others; each matches a
  hand-computed answer and passes all gates.
- Gates: invented literal, hand-rolled named metric, ratio answer to a money question,
  failing cross-check, claim off by > 0.2%.

Verification: `python -m unittest discover -s tests -v` and
`python -m compileall -q src config`.

## 14. Out of scope

Leaf extraction from evidence (sub-project 2); LLM, prompts, repair loop, answer text, dev set
(sub-project 3); API, UI, sessions (sub-project 4); reranker fine-tuning; securities and
insurance templates; any change to retrieval, paraphrasing, PostgreSQL or Chroma.

## 15. Risks and open items

- Statement codes and bank labels must be verified against the corpus; OCR may drop or
  misplace the code column (sub-project 2 handles fallback to labels).
- No memory cap for the sandbox on Windows.
- Process spawn on Windows adds latency per execution (measured in the plan; one process per
  attempt, not per helper call).
- `metric_catalog.py` and the tree share ids but are separate until a later change generates
  the catalog from the tree.
- End-to-end accuracy cannot be measured until sub-project 3 builds a labelled dev set.

## 16. Known limits and contracts for sub-projects 2–4

Implemented behaviour at the end of sub-project 1; later sub-projects must respect or fix it.

**Alias matching (sub-project 2).**
- `MetricTree.find_by_alias` is for QUESTION text only: it matches aliases as whole-word
  substrings, longest first. Statement row labels must be matched by statement code or by
  exact equality of folded labels (`fold_text`). Substring matching misroutes real labels:
  "Dự phòng giảm giá hàng tồn kho" → `inventories`, "Tài sản ngắn hạn khác" →
  `current_assets`, "Nợ phải trả ngắn hạn khác" → `current_liabilities`, bank "Chi phí hoạt
  động dịch vụ" → `operating_expenses`.
- Bank leaves inside a section (`parent`) are located by position under the section header
  (plan decision D2), not by their own aliases.
- Known question-alias collisions, pinned in `tests/test_calculation_metric_data.py`: Q157
  "tổng nguồn vốn đầu tư xây dựng cơ bản" → `total_assets`; Q118 "dự phòng chung …" →
  `customer_loans_gross`.

**Generated-code constraints (sub-project 3 prompts).**
- No attribute access at all (`d.items()`, `xs.append(...)`); iterate with `for k in d` and
  build lists with comprehensions.
- Numeric literals are exact `Decimal`s, but `int / int` from builtins/helpers (e.g.
  `count_if(...) / len(xs)`) yields a float and fails closed; multiply by a literal first
  (e.g. `count_if(...) * 100 / len(xs)`).
- Thresholds go through `percent(n)` / `money(n, scale)`; for "1 nghìn tỷ" write
  `money(1, "nghin_ty")`, not `money(1000, "ty")` (1000 is not a question literal).
- Strings are allowed as plain values (ticker lists); f-strings are forbidden. The bare name
  `_` is allowed; every other `_`-prefixed name is rejected.

**Gate limits (sub-project 3).**
- Provenance residual bypasses: arithmetic on allowed constants (`1 - 1/4`), `len("...")`,
  and year-range integers used as amounts are not flagged.
- The formula gate checks that `metric()` / `fact()` steps exist, not data flow: a program
  can call `metric()` and return a hand-rolled value.
- `requested_unit` is None for ~49 of ~940 quantity questions, so the unit check is skipped.
- `compare_answers` requires identical units (no ty vs nghìn tỷ conversion).
- A restated fact used only by a cross-check alternative still yields a cross-check error.
- A duration phrasing "bao nhiêu năm" (e.g. payback period) would classify as `count`.
- An English-format "1.5" in a question is not a question literal.

**Facts (sub-project 2).** D1 conflict resolution (R7) is pairwise in input order and is
order-dependent for 3+ facts of one key; it must compare each new fact against all prior
facts before sub-project 2 merges facts from several sources.

**Sandbox and display (sub-project 4).**
- No memory cap and no OS-level isolation on Windows; serving should run the child in a Job
  Object with a memory limit. The POSIX `RLIMIT_AS` path is untested.
- `round_display` can return `-0.00`; the display layer must normalise it.

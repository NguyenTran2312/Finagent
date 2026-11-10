# Calculation Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/calculation/`: a validated metric dependency tree (leaves = statement line items), exact units, a safe formula language, sourced facts, a helper library, a sandbox that runs LLM-written `solution()` code, and deterministic verification gates.

**Architecture:** Leaves and derived metrics are YAML data in `config/metrics/`; derived formulas use a whitelisted `ast` subset and are dimension-checked at load time. Generated code is AST-checked, numeric literals are rewritten to `Decimal`, and the code runs in a spawned process with only helpers in scope. Every helper call is traced, and gates check the trace and the result.

**Tech Stack:** Python 3.14, stdlib (`ast`, `decimal`, `multiprocessing`, `dataclasses`, `enum`), PyYAML 6.0.3 (already pinned), unittest.

**Spec:** `docs/superpowers/specs/2026-11-10-calculation-core-design.md`

## Global Constraints

- Pure Python. No LLM, GPU, network or database access in `src/calculation/` or its tests. Task 5's code-verification step reads `data/databases/corpus.sqlite3` read-only and is a manual check, not a unit test.
- No new dependencies. PyYAML 6.0.3 is already in `requirements.txt`.
- Production modules import via `src.*` (e.g. `from src.preprocess.cleaner import fold_text`); `config.settings` stays as is.
- All values are `Decimal`. Floats never enter `Quantity`. Money is stored in đồng.
- Rounding happens once, at the end: requested precision, or 2 decimals in the display unit when the question states none. `ROUND_HALF_UP`.
- No silent fallback. Never choose between conflicting facts silently, never treat missing as 0, never default an ambiguous unit, never return `inf`/`nan`.
- Raw exception text never leaves the sandbox; failures carry the exception class name only.
- Do not modify retrieval, paraphrasing, `metric_catalog.py`, the PostgreSQL schema or Chroma.
- Git: work in worktree `.worktrees/calculation-core` on branch `feat/calculation-core` cut from `main`. Commit per task on that branch only. Never push, merge or open a PR without explicit user authorization.
- Test command (PowerShell, from the worktree root):
  `$env:PYTHONPATH="$PWD;$PWD\src"; $env:TEMP="$PWD\.tmp-tests"; $env:TMP=$env:TEMP; D:\finance_project\.venv\Scripts\python.exe -B -m unittest <target> -v`

## Plan decisions (resolve spec gaps; binding for all tasks)

- D1. `FactSource` gains `scale: Decimal`, the multiplier the value was parsed with (1 for VND, 10^6 for triệu VND). The tolerance "one unit of the printed scale" is `max(scale)` of the two facts compared. Two facts with the same key whose values differ by no more than that tolerance are **not** a conflict; the finer-scale fact is kept.
- D2. Bank line codes are roman numerals plus locally repeated numbers (ACB 2022: `VI Cho vay khách hàng` → `1 Cho vay khách hàng`). Bank leaves use `codes: []`, at least two aliases, and an optional `parent` (the id of the section leaf). `parent` must name a leaf in the same template.
- D3. In generated code, every numeric literal is rewritten before execution to `_num("<source text>")`, which returns `Decimal(<source text>)`. The provenance gate reads literals from the original AST.
- D4. The generated code builds thresholds with the helpers `money(value, scale)` and `percent(value)`, because it cannot construct `Quantity` directly. Comparisons between a `Quantity` and a bare number are allowed only for `RATIO` and `COUNT`.
- D5. `answer(value, unit)`: `unit` ∈ {"dong", "nghin", "trieu", "ty", "tram_ty", "nghin_ty", "percent", "percentage_points", "times", "days", "money_per_share", "year", "ticker", "count", "boolean"}. The unit determines `Answer.kind`.
- D6. `Formula` pickles by its text and re-parses on load, so the tree can cross the process boundary.
- D7. Sandbox timeout (2.0 s) counts from a "started" message the child sends just before calling `solution()`. The child gets a separate startup budget of 20 s, because spawning on Windows is slow.
- D8. Cross-check tolerance between derived alternatives is the sum of `source.scale` over the facts the two alternatives used.
- D9. Spec §6.5 estimated 40–60 derived metrics. This plan ships the 20 tabled in Task 5, which cover every golden program. Further nodes arrive through validated proposals (Task 7), which are reviewed and promoted in sub-project 3. Persisting accepted proposals to a log file is also sub-project 3's job; `ProposalResult` carries everything needed.
- D10. Opening-balance consistency (spec §8) is enforced when the `FactSet` is built, via the D1 conflict rule, because opening(t) and closing(t−1) share one key. The `cross_check` gate covers derived alternatives only.

## Review Focus

1. Costs printed in parentheses (`(1.950.254.195.864)` for interest expense, VJC 2023). Ratios must use `abs()` and give a positive ICR. Pinned in Task 6 (`test_icr_with_negative_printed_interest`).
2. An opening balance that comes from two reports in different scales (VND vs triệu VND) must not raise a false conflict. Pinned in Task 4 (`test_same_key_within_scale_tolerance_is_not_conflict`).
3. A question with no stated unit ("năm nào", "có bao nhiêu", plain "bao nhiêu?") must not crash, and must get the right answer kind and the default precision. Pinned in Task 9 (`test_requested_kind_without_unit`).
4. A tie in `argmax` / `argmin` must raise `CalculationError`, never pick arbitrarily. Pinned in Task 6 (`test_argmax_tie_raises`).
5. A bank ticker asked for a corporate-only metric (e.g. `gross_margin` for ACB) must raise a typed error instead of evaluating with the wrong leaves. Pinned in Task 6 (`test_metric_wrong_template_raises`) and Task 7 (`test_proposal_wrong_template_leaf_rejected`).

---

### Task 1: Worktree, package and units

**Files:**
- Create: `src/calculation/__init__.py` (empty docstring module)
- Create: `src/calculation/units.py`
- Create: `tests/test_calculation_units.py`
- Copy: `docs/superpowers/specs/2026-11-10-calculation-core-design.md` and this plan from the main checkout into the worktree

**Interfaces:**
- Produces:
  - `class Dimension(StrEnum)`: `MONEY="money"`, `RATIO="ratio"`, `MONEY_PER_SHARE="money_per_share"`, `SHARES="shares"`, `DAYS="days"`, `COUNT="count"`
  - `class UnitError(ValueError)`, `class CalculationError(ArithmeticError)`
  - `@dataclass(frozen=True, slots=True) class Quantity: value: Decimal; dimension: Dimension`. It supports `+ - * /`, unary `-`, `abs()` and ordering comparisons, and accepts `int`/`Decimal` operands as dimensionless (treated as `RATIO`).
  - `MONEY_SCALES: dict[str, Decimal]` = {"dong": 1, "nghin": 10^3, "trieu": 10^6, "ty": 10^9, "tram_ty": 10^11, "nghin_ty": 10^12}
  - `@dataclass(frozen=True) class DisplayUnit: kind: Literal["money","percent","percentage_points","times","days","money_per_share","count","shares"]; scale: Decimal = Decimal(1)`
  - `def display_unit(name: str) -> DisplayUnit`, mapping the D5 unit strings (money names → kind "money" with the scale)
  - `def to_display(q: Quantity, unit: DisplayUnit) -> Decimal`
  - `def round_display(value: Decimal, places: int = 2) -> Decimal`

- [ ] **Step 1: Create the worktree**

```bash
git -C D:/finance_project worktree add .worktrees/calculation-core -b feat/calculation-core main
```
Copy the spec and plan files into the same paths in the worktree. Expected: `git -C .worktrees/calculation-core status` lists the two docs as untracked.

- [ ] **Step 2: Write the failing tests** in `tests/test_calculation_units.py`

```python
from decimal import Decimal as D
from src.calculation.units import (Quantity, Dimension as Dm, UnitError, CalculationError,
                                   display_unit, to_display, round_display)
M = lambda v: Quantity(D(v), Dm.MONEY)
R = lambda v: Quantity(D(v), Dm.RATIO)

def test_money_algebra():
    assert M("5") + M("3") == M("8") and (M("6") / M("3")) == R("2")
    assert (M("10") / Quantity(D("2"), Dm.SHARES)).dimension is Dm.MONEY_PER_SHARE
    assert M("10") * R("0.5") == M("5.0") and M("10") * 2 == M("20")
def test_invalid_algebra_raises():      # each in assertRaises(UnitError)
    M("1") + R("1"); M("1") * M("1"); R("1") - Quantity(D(1), Dm.DAYS)
def test_division_by_zero():            # assertRaises(CalculationError)
    M("1") / M("0")
def test_rejects_float_and_nonfinite():  # TypeError for 0.1; UnitError for D("NaN"), D("Infinity")
def test_compare_with_number_only_ratio_count():
    assert R("1.6") > D("1.5"); with assertRaises(UnitError): M("5") > 3
def test_display_money_scales():
    assert to_display(M("1234567000000"), display_unit("ty")) == D("1234.567")
    assert to_display(M("2500000000000"), display_unit("nghin_ty")) == D("2.5")
def test_display_percent_and_points():
    assert to_display(R("0.1234"), display_unit("percent")) == D("12.34")
    assert to_display(R("-0.015"), display_unit("percentage_points")) == D("-1.5")
    with assertRaises(UnitError): to_display(M("1"), display_unit("percent"))
def test_display_money_accepts_money_per_share():
    assert to_display(Quantity(D("1500"), Dm.MONEY_PER_SHARE), display_unit("nghin")) == D("1.5")
def test_round_half_up():
    assert round_display(D("2.345")) == D("2.35") and round_display(D("-2.345")) == D("-2.35")
def test_decimal_exact():
    assert M("0.1") + M("0.2") == M("0.3")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: the test command with target `tests.test_calculation_units`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'src.calculation'`

- [ ] **Step 4: Implement `src/calculation/units.py`** to the interface above

Use a dimension table for `*` and `/`; any pair not in the table raises `UnitError`:
- `/`: MONEY/MONEY→RATIO; MONEY/SHARES→MONEY_PER_SHARE; X/X→RATIO for every X; X/RATIO→X
- `*`: RATIO×X→X; X×RATIO→X

Days: only `expressions.py` (Task 2) produces `DAYS` (via the `* 365` form). Ordering comparisons need equal dimensions; a bare number compares only with `RATIO`/`COUNT` (D4). `to_display` multiplies by 100 for percent and percentage points, divides money by the scale, and passes days, times, count and shares through. `to_display` rejects every other dimension/kind pair with `UnitError`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: the test command with target `tests.test_calculation_units`. Expected: all OK.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-11-10-calculation-core-design.md docs/superpowers/plans/2026-11-10-calculation-core.md src/calculation/__init__.py src/calculation/units.py tests/test_calculation_units.py
git commit -m "feat(calculation): add exact units and dimension algebra"
```

---

### Task 2: Formula language

**Files:**
- Create: `src/calculation/expressions.py`
- Create: `tests/test_calculation_expressions.py`

**Interfaces:**
- Consumes: `Quantity`, `Dimension`, `UnitError`, `CalculationError` (Task 1)
- Produces:
  - `class FormulaError(ValueError)`
  - `@dataclass(frozen=True) class Ref: node_id: str; offset: int` (`offset` ≥ 0; `x[t-1]` → offset 1; a bare `x` → offset 0)
  - `class Formula`: attributes `text: str`, `refs: frozenset[Ref]`, `is_days_form: bool`; pickles by text (D6)
  - `def parse_formula(text: str) -> Formula`
  - `def infer_dimension(formula: Formula, dimension_of: Callable[[str], Dimension]) -> Dimension`
  - `def evaluate(formula: Formula, resolve: Callable[[Ref], Quantity]) -> Quantity`

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_refs_and_offsets():
    f = parse_formula("net_revenue / ((total_assets + total_assets[t-1]) / 2)")
    assert f.refs == {Ref("net_revenue", 0), Ref("total_assets", 0), Ref("total_assets", 1)}
def test_rejected_constructs():   # each raises FormulaError
    for bad in ["x ** 2", "max(a, b)", "a.b", "a < b", "a and b", "x[t+1]", "x[0]", "x[t-1.5]",
                "'s'", "lambda: 1", "__import__('os')", "a if b else c", "1j"]: ...
def test_abs_is_only_call():
    assert parse_formula("abs(interest_expense)").refs == {Ref("interest_expense", 0)}
def test_infer_dimension_and_mismatch():
    dims = {"a": Dm.MONEY, "b": Dm.MONEY, "r": Dm.RATIO}.__getitem__
    assert infer_dimension(parse_formula("a / b"), dims) is Dm.RATIO
    with assertRaises(UnitError): infer_dimension(parse_formula("a + r"), dims)
def test_days_form():
    f = parse_formula("((inv + inv[t-1]) / 2) / abs(cogs) * 365")
    assert f.is_days_form and infer_dimension(f, {"inv": Dm.MONEY, "cogs": Dm.MONEY}.__getitem__) is Dm.DAYS
    assert not parse_formula("a * 365 + 1").is_days_form
def test_evaluate_exact():
    vals = {Ref("a", 0): M("10"), Ref("b", 0): M("4")}
    assert evaluate(parse_formula("(a - b) / b"), vals.__getitem__) == R("1.5")
def test_evaluate_division_by_zero():  # CalculationError for "a / b" with b = 0
def test_pickle_roundtrip():
    f = parse_formula("a / b"); g = pickle.loads(pickle.dumps(f)); assert g.text == f.text and g.refs == f.refs
```

- [ ] **Step 2: Run the tests to verify they fail**

Target: `tests.test_calculation_expressions`. Expected: ERROR, module missing.

- [ ] **Step 3: Implement `src/calculation/expressions.py`**

Parse with `ast.parse(text, mode="eval")`. Walk the tree with an explicit whitelist:
- `Expression`, `BinOp` (`Add`/`Sub`/`Mult`/`Div` only), `UnaryOp` (`USub`)
- `Constant` (int or finite numeric only, never bool or str)
- `Name` (no `__`)
- `Call` only when `func` is the `Name` `abs` with exactly one positional argument
- `Subscript` only of the form `Name[t - <int ≥ 1>]`

Any other node raises `FormulaError` with the node type. `is_days_form` is true when the top-level node is `BinOp(Mult)` whose right operand is `Constant(365)`. Constants evaluate as `Decimal(str(value))`.

- [ ] **Step 4: Run the tests to verify they pass.** Same target, all OK.

- [ ] **Step 5: Commit**

```bash
git add src/calculation/expressions.py tests/test_calculation_expressions.py
git commit -m "feat(calculation): add whitelisted formula language"
```

---

### Task 3: Metric tree loader and validation

**Files:**
- Create: `src/calculation/metric_tree.py`
- Create: `tests/test_calculation_metric_tree.py`

**Interfaces:**
- Consumes: `parse_formula`, `infer_dimension`, `Formula`, `Ref` (Task 2); `Dimension` (Task 1); `StatementKind` from `src.indexing.table_semantics`; `fold_text` from `src.preprocess.cleaner`
- Produces:
  - `Template = Literal["corporate", "bank"]`; `TEMPLATES = ("corporate", "bank")`
  - `@dataclass(frozen=True) class LeafDef: id: str; template: Template; label_vi: str; aliases: tuple[str, ...]; statement_kind: StatementKind; codes: tuple[str, ...]; period_kind: Literal["flow","closing"]; source: Literal["statement","notes"]; dimension: Dimension; parent: str | None = None`
  - `@dataclass(frozen=True) class DerivedDef: id: str; label_vi: str; aliases: tuple[str, ...]; templates: tuple[Template, ...]; dimension: Dimension; display: str | None; formulas: tuple[Formula, ...]; expected_range: tuple[Decimal, Decimal] | None`
  - `class TreeError(ValueError)` with attributes `file`, `node_id`, `reason`; `str()` = `f"{file}: {node_id}: {reason}"`
  - `@dataclass(frozen=True) class LeafRequest: metric_id: str; year: int; period_kind: str`
  - `DEFAULT_METRICS_DIR: Path` = project root / "config" / "metrics"
  - `class MetricTree` with:
    - `load(directory: Path = DEFAULT_METRICS_DIR) -> MetricTree` (classmethod)
    - `from_entries(leaves: Mapping[Template, list[dict]], derived: list[dict], source: str = "<memory>") -> MetricTree` (classmethod)
    - `leaf(id, template) -> LeafDef`
    - `derived(id) -> DerivedDef`
    - `applies(id, template) -> bool`
    - `kind(id) -> Literal["leaf","derived"]`
    - `dimension_of(id, template) -> Dimension`
    - `required_leaves(node_id, template, year) -> list[frozenset[LeafRequest]]` (one set per formula alternative, in order; a leaf node returns `[frozenset({itself})]`)
    - `find_by_alias(text, template) -> set[str]` (folded, longest-alias-first, same approach as `metric_catalog.find_metrics`)
    - `with_derived(d: DerivedDef) -> MetricTree` (returns a new validated tree; used by proposals)

- [ ] **Step 1: Write the failing tests** using small `from_entries` fixtures. Leaves `a`, `b` (money, flow) and `c` (money, closing) in corporate:

```python
def test_valid_tree_and_expansion():
    t = tree(derived=[{"id": "r", "templates": ["corporate"], "dimension": "ratio",
                       "formulas": ["a / ((c + c[t-1]) / 2)"]}])
    assert t.required_leaves("r", "corporate", 2024) == [frozenset({
        LeafRequest("a", 2024, "flow"), LeafRequest("c", 2024, "closing"), LeafRequest("c", 2023, "closing")})]
def test_derived_of_derived_expands_to_leaves(): ...
def test_alternatives_in_order(): formulas ["g_rep", "a - b"] → two sets, reported first
def test_tree_errors():   # each assertRaises(TreeError) and the reason contains the substring
    unknown ref → "unknown reference"; r = r + a → "cycle"; "a +" → "formula syntax";
    ratio node with formula "a" (money) → "dimension mismatch"; duplicate id → "duplicate id";
    alias claimed twice in one template → "duplicate alias";
    bank leaf with codes [] and one alias → "needs codes or two aliases";
    days node without the * 365 form → "days form";
    derived lists template "bank" but references a corporate-only leaf → "unknown reference";
    leaf parent not in template → "unknown parent"
def test_error_names_file_and_node(): str(err) starts with "<memory>: r: "
def test_find_by_alias_longest_first():
    aliases "lợi nhuận sau thuế" (a) and "lợi nhuận sau thuế thuộc cổ đông công ty mẹ" (b)
    → find_by_alias("LNST thuộc cổ đông công ty mẹ ... lợi nhuận sau thuế thuộc cổ đông công ty mẹ", "corporate") == {"b"}
def test_with_derived_revalidates(): adding a cyclic DerivedDef raises TreeError; the original tree is unchanged
```

- [ ] **Step 2: Run the tests to verify they fail.** Target `tests.test_calculation_metric_tree`. Expected: ERROR, module missing.

- [ ] **Step 3: Implement `src/calculation/metric_tree.py`.** `load` reads `corporate_leaves.yaml`, `bank_leaves.yaml` and `derived.yaml` with `yaml.safe_load`; the file name goes into `TreeError.file`. `expected_range` values become `Decimal(str(v))`. Validation order:
  1. per-entry schema
  2. ids
  3. aliases
  4. references per listed template
  5. cycles (DFS)
  6. dimensions (`infer_dimension` with `dimension_of(id, template)` for each listed template)
  7. days form

  There is no partial tree on error.

- [ ] **Step 4: Run the tests to verify they pass.**

- [ ] **Step 5: Commit**

```bash
git add src/calculation/metric_tree.py tests/test_calculation_metric_tree.py
git commit -m "feat(calculation): add validated metric dependency tree"
```

---

### Task 4: Facts

**Files:**
- Create: `src/calculation/facts.py`
- Create: `tests/test_calculation_facts.py`

**Interfaces:**
- Consumes: `Quantity` (Task 1)
- Produces:
  - `Scope = Literal["consolidated","separate"]`, `PeriodKind = Literal["flow","closing"]`
  - `@dataclass(frozen=True) class FactSource: chunk_id: str; doc_id: str; page: int; row_label: str; raw_text: str; unit_text: str; scale: Decimal; column_header: str | None = None`. All `str` fields except `column_header` must be non-empty, `page >= 0`, `scale > 0`; otherwise `ValueError`.
  - `@dataclass(frozen=True) class FactKey: metric_id: str; ticker: str; fiscal_year: int; scope: Scope; period_kind: PeriodKind`
  - `@dataclass(frozen=True) class Fact: metric_id; ticker; fiscal_year; scope; period_kind; value: Quantity; source: FactSource; restated: bool = False`, with property `key -> FactKey`. `source` must be a `FactSource` (`TypeError` otherwise).
  - `class FactConflict(ValueError)`, `class MissingFact(LookupError)`
  - `class FactSet` with:
    - `__init__(facts: Iterable[Fact])`
    - `get(key: FactKey) -> Fact` (raises `MissingFact`)
    - `find(metric_id, ticker, year, scope, period_kind) -> Fact | None`
    - `opening(metric_id, ticker, year, scope) -> Fact` (the closing fact of `year-1`)
    - `warnings: tuple[str, ...]`
    - `__iter__`, `__len__`

- [ ] **Step 1: Write the failing tests**

```python
def test_source_required(): Fact(..., source=None) → TypeError; FactSource(chunk_id="", ...) → ValueError
def test_get_and_missing(): get(unknown key) → MissingFact
def test_conflict_rejected(): two facts, same key, values differ by more than the tolerance → FactConflict
def test_same_key_within_scale_tolerance_is_not_conflict():
    a = fact(value=M("1234567000000"), scale=D("1000000"))   # triệu report
    b = fact(value=M("1234567123456"), scale=D("1"))         # VND report
    s = FactSet([a, b]); assert s.get(a.key).value == b.value   # finer scale kept (D1)
def test_restated_wins_with_warning(): one restated → restated value kept, warnings non-empty
def test_two_restated_conflict(): both restated and differing → FactConflict
def test_opening_is_prior_closing(): opening("total_assets","VJC",2024,"consolidated") is the 2023 closing fact
def test_sign_preserved(): a fact with value M("-1950254195864") keeps its negative value
```

- [ ] **Step 2: Run the tests to verify they fail.** Target `tests.test_calculation_facts`.
- [ ] **Step 3: Implement `src/calculation/facts.py`** per the interface and D1.
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit**

```bash
git add src/calculation/facts.py tests/test_calculation_facts.py
git commit -m "feat(calculation): add sourced facts with conflict rules"
```

---

### Task 5: Metric data files (v1: corporate + bank)

**Files:**
- Create: `config/metrics/corporate_leaves.yaml`, `config/metrics/bank_leaves.yaml`, `config/metrics/derived.yaml`
- Create: `tests/test_calculation_metric_data.py`
- Create: `scripts/verify_metric_codes.py` (manual, read-only corpus check; not imported by tests)

**Interfaces:**
- Consumes: `MetricTree.load` (Task 3)
- Produces: the node ids below, which Tasks 6–9 and the golden programs use

**Corporate leaves.** Statement codes per Circular 200. The income statement codes were verified on the VJC 2023 consolidated report (table 6); the balance sheet codes 100–153 on table 3. Every other code must be confirmed in Step 4.

| id | statement | code | period |
|---|---|---|---|
| gross_revenue | income | 01 | flow |
| net_revenue | income | 10 | flow |
| cost_of_goods_sold | income | 11 | flow |
| gross_profit_reported | income | 20 | flow |
| financial_income | income | 21 | flow |
| financial_expenses | income | 22 | flow |
| interest_expense | income | 23 | flow |
| selling_expenses | income | 25 | flow |
| admin_expenses | income | 26 | flow |
| operating_profit | income | 30 | flow |
| other_income | income | 31 | flow |
| other_expenses | income | 32 | flow |
| profit_before_tax | income | 50 | flow |
| profit_after_tax | income | 60 | flow |
| profit_after_tax_parent_share | income | 61 | flow |
| basic_eps (money_per_share) | income | 70 | flow |
| current_assets | balance | 100 | closing |
| cash_and_equivalents | balance | 110 | closing |
| inventories | balance | 140 | closing |
| long_term_assets | balance | 200 | closing |
| construction_in_progress | balance | 242 | closing |
| total_assets | balance | 270 | closing |
| liabilities | balance | 300 | closing |
| current_liabilities | balance | 310 | closing |
| short_term_borrowings | balance | 320 | closing |
| long_term_borrowings | balance | 338 | closing |
| owners_equity | balance | 400 | closing |
| cfo | cash_flow | 20 | flow |
| depreciation | cash_flow | 02 | flow |

**Bank leaves.** `codes: []` with aliases and `parent` (D2):
`interest_and_similar_income`, `interest_and_similar_expenses`, `net_interest_income`, `net_fee_income`, `operating_expenses`, `credit_risk_provision_expense`, `profit_before_tax`, `profit_after_tax`, `total_assets`, `customer_loans_section` (VI), `customer_loans_gross` (parent `customer_loans_section`), `customer_loan_provision` (parent `customer_loans_section`), `deposits_at_other_credit_institutions`, `customer_deposits`, `liabilities`, `owners_equity`.

**Derived (v1).** Formulas are exact:

| id | templates | dimension/display | formulas (in order) |
|---|---|---|---|
| gross_profit | corporate | money | `gross_profit_reported`; `net_revenue - abs(cost_of_goods_sold)` |
| gross_margin | corporate | ratio/percent | `gross_profit / net_revenue` |
| net_margin | corporate | ratio/percent | `profit_after_tax / net_revenue` |
| ebit | corporate | money | `profit_before_tax + abs(interest_expense)` |
| interest_coverage | corporate | ratio/times | `ebit / abs(interest_expense)` |
| liabilities_to_equity | corporate, bank | ratio/times | `liabilities / owners_equity` |
| borrowings_to_equity | corporate | ratio/times | `(short_term_borrowings + long_term_borrowings) / owners_equity` |
| current_ratio | corporate | ratio/times | `current_assets / current_liabilities` |
| quick_ratio | corporate | ratio/times | `(current_assets - inventories) / current_liabilities` |
| cfo_to_current_liabilities | corporate | ratio/times | `cfo / current_liabilities` |
| cfo_to_profit_after_tax | corporate | ratio/times | `cfo / profit_after_tax` |
| cfo_margin | corporate | ratio/percent | `cfo / net_revenue` |
| asset_turnover | corporate | ratio/times | `net_revenue / ((total_assets + total_assets[t-1]) / 2)` |
| days_inventory | corporate | days | `((inventories + inventories[t-1]) / 2) / abs(cost_of_goods_sold) * 365` |
| selling_and_admin_expenses | corporate | money | `abs(selling_expenses) + abs(admin_expenses)` |
| long_term_assets_share | corporate | ratio/percent | `long_term_assets / total_assets` |
| roe | corporate, bank | ratio/percent | `profit_after_tax / ((owners_equity + owners_equity[t-1]) / 2)` |
| roa | corporate, bank | ratio/percent | `profit_after_tax / ((total_assets + total_assets[t-1]) / 2)` |
| loan_to_deposit | bank | ratio/percent | `customer_loans_gross / customer_deposits` |
| loan_provision_ratio | bank | ratio/percent | `abs(customer_loan_provision) / customer_loans_gross` |

- [ ] **Step 1: Write the failing data lint test**

```python
def test_real_tree_loads(): MetricTree.load() succeeds
def test_required_nodes_present():
    t = MetricTree.load()
    for i in ["interest_coverage","quick_ratio","days_inventory","asset_turnover","roe","loan_to_deposit"]: t.derived(i)
    assert t.applies("gross_margin","corporate") and not t.applies("gross_margin","bank")
def test_every_corporate_statement_leaf_has_code(): every corporate leaf with source "statement" has non-empty codes
def test_bank_leaves_have_two_aliases(): every bank leaf with codes () has len(aliases) >= 2
def test_catalog_ids_reused(): every id in METRIC_CATALOG that names a statement item ("net_revenue","profit_before_tax","profit_after_tax","profit_after_tax_parent_share") exists in the tree
```

- [ ] **Step 2: Run the tests to verify they fail.** Target `tests.test_calculation_metric_data`. Expected: FAIL, metrics files missing.
- [ ] **Step 3: Write the three YAML files** exactly as tabled. Each leaf needs `label_vi` and at least the label plus common short forms as aliases (e.g. `lntt`, `lnst`, `chi phí lãi vay`, `hàng tồn kho`, `tổng tài sản`, `vốn chủ sở hữu`, `nợ phải trả`, `nợ ngắn hạn`, `lưu chuyển tiền thuần từ hoạt động kinh doanh`).
- [ ] **Step 4: Verify codes against the corpus (manual).** Write `scripts/verify_metric_codes.py`. It:
  1. opens `data/databases/corpus.sqlite3` read-only (`file:...?mode=ro`)
  2. for each corporate leaf, samples 20 consolidated tables of its `statement_kind`
  3. prints, per code, the most common folded row label next to the leaf's aliases
  4. for each bank leaf, prints how many of 10 bank reports contain any alias

  Run: `D:\finance_project\.venv\Scripts\python.exe -I -X utf8 scripts/verify_metric_codes.py`. Expected: every corporate code's top label matches the leaf; every bank leaf is found in at least 8/10 reports. Fix any mismatched code or alias in the YAML and record the output summary in the commit message body.
- [ ] **Step 5: Run the lint tests to verify they pass.**
- [ ] **Step 6: Commit**

```bash
git add config/metrics tests/test_calculation_metric_data.py scripts/verify_metric_codes.py
git commit -m "feat(calculation): add v1 corporate and bank metric tree data"
```

---

### Task 6: Helper library and trace

**Files:**
- Create: `src/calculation/helpers.py`
- Create: `tests/test_calculation_helpers.py`

**Interfaces:**
- Consumes: `MetricTree`, `Template` (Task 3); `FactSet`, `FactKey`, `Scope` (Task 4); `evaluate`, `Ref` (Task 2); `Quantity`, `Dimension`, `display_unit`, `DisplayUnit`, `CalculationError`, `UnitError`, `MONEY_SCALES` (Task 1)
- Produces:
  - `AnswerKind = Literal["quantity","year","ticker","count","boolean"]`
  - `@dataclass(frozen=True) class Answer: kind: AnswerKind; value: Quantity | int | str | bool; unit: str` (the D5 name)
  - `@dataclass(frozen=True) class ExecutionContext: default_scope: Scope; ticker_templates: Mapping[str, Template]; question_literals: frozenset[Decimal]; requested_unit: str | None; requested_kind: AnswerKind; named_metrics: frozenset[str] = frozenset()`
  - `@dataclass(frozen=True) class TraceStep: helper: str; args: tuple; result: str; fact_keys: tuple[FactKey, ...]; detail: Mapping[str, str] = field(default_factory=dict)`
  - `class Trace` with `steps: list[TraceStep]`, `facts_used() -> tuple[FactKey, ...]`, `metrics_used() -> frozenset[str]`, `cross_checks() -> list[TraceStep]`
  - `class MetricUnavailable(LookupError)`: the node does not apply to the ticker's template, or no alternative has all its facts
  - `def build_helpers(tree, facts, context, trace) -> dict[str, Callable]` with exactly these names:
    - `fact(metric_id, ticker, year, period="flow", scope=None) -> Quantity`
    - `metric(node_id, ticker, year, scope=None) -> Quantity`
    - `avg_balance(metric_id, ticker, year, scope=None) -> Quantity`
    - `growth_pct(new, old) -> Quantity` (RATIO, `(new-old)/abs(old)`)
    - `cagr(end, start, years: int) -> Quantity` (RATIO, needs `end, start > 0`, `years ≥ 1`, 28-digit Decimal context)
    - `pct_point_change(new_ratio, old_ratio) -> Quantity`
    - `mean(values)`, `median(values)` (even count → mean of the middle two)
    - `argmax(mapping)`, `argmin(mapping)` (return the key; a tie or an empty mapping raises `CalculationError`)
    - `filter_by(mapping, op, threshold) -> dict`, `count_if(values, op, threshold) -> int` (`op` ∈ {"<","<=",">",">=","=="}, anything else raises `ValueError`)
    - `money(value, scale: str) -> Quantity`, `percent(value) -> Quantity`
    - `answer(value, unit: str) -> Answer`
  - `metric()` evaluates the first alternative whose facts all exist. It then evaluates every other computable alternative and appends a `TraceStep(helper="cross_check", detail={"node","primary","alternative","tolerance"})` (tolerance per D8). A template mismatch raises `MetricUnavailable`.

- [ ] **Step 1: Write the failing tests** with an in-memory tree (Task 3 fixtures plus `interest_coverage`/`gross_profit` shaped like Task 5) and VJC-2023-shaped facts:

```python
def test_icr_with_negative_printed_interest():
    # profit_before_tax 606.103.607.021; interest_expense -1.950.254.195.864 (printed in parentheses)
    q = h["metric"]("interest_coverage", "VJC", 2023)
    assert q.dimension is Dm.RATIO and round_display(q.value, 4) == D("1.3108")
def test_trace_records_facts_and_metrics(): trace.metrics_used() == {"interest_coverage","ebit"}; facts_used() has 2 keys
def test_alternative_cross_check_recorded(): gross_profit with both alternatives available → one cross_check step
def test_metric_wrong_template_raises(): metric("gross_margin","ACB",2022) with ACB → bank → MetricUnavailable
def test_missing_fact_raises(): MissingFact propagates from fact(...)
def test_avg_balance(): (closing 2024 + closing 2023)/2
def test_growth_and_cagr(): growth_pct(M(120),M(100)) == R("0.2"); round(cagr(M(121),M(100),2).value,6) == D("0.100000")
def test_cagr_rejects_nonpositive(): CalculationError
def test_median_even_and_odd()
def test_argmax_tie_raises()
def test_filter_and_count():
    filter_by({"A": R("0.4"), "B": R("0.8")}, ">", percent(50)) == {"B": R("0.8")}
    count_if([R("1.2"), R("1.6")], "<", D("1.5")) == 1
def test_money_threshold(): money(20, "ty") == M("20000000000")
def test_answer_kinds(): answer(2023,"year").kind == "year"; answer(q,"ty").kind == "quantity"; answer(1,"bogus") → ValueError
```

- [ ] **Step 2: Run the tests to verify they fail.** Target `tests.test_calculation_helpers`.
- [ ] **Step 3: Implement `src/calculation/helpers.py`.** Each helper wraps its body to append one `TraceStep`. `metric()` resolves `Ref`s recursively: a leaf goes through `FactSet.find` with the leaf's `period_kind` and `year - offset`; a derived ref goes through `metric()`. The scope is `scope or context.default_scope`, and the template is `context.ticker_templates[ticker]` (an unknown ticker raises `MetricUnavailable`).
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit**

```bash
git add src/calculation/helpers.py tests/test_calculation_helpers.py
git commit -m "feat(calculation): add traced helper library for generated code"
```

---

### Task 7: Proposal validation

**Files:**
- Create: `src/calculation/proposals.py`
- Create: `tests/test_calculation_proposals.py`

**Interfaces:**
- Consumes: `MetricTree.with_derived`, `DerivedDef`, `TreeError` (Task 3); `parse_formula` (Task 2)
- Produces:
  - `@dataclass(frozen=True) class ProposalResult: accepted: bool; tree: MetricTree | None; issues: tuple[str, ...]; proposal: Mapping[str, object]`
  - `def validate_proposal(tree: MetricTree, proposal: Mapping[str, object], template: Template) -> ProposalResult`. The proposal has the `derived.yaml` entry shape. It is rejected if: the id already exists ("redefines existing node"); `template` is not in its `templates`; any `TreeError` occurs when added (the reason is copied in); or an alias collides with an existing node. On acceptance, `tree` is the extended tree. The function never raises for bad proposals.

- [ ] **Step 1: Write the failing tests**

```python
def test_valid_proposal_accepted(): {"id":"cash_to_assets","templates":["corporate"],"dimension":"ratio","formulas":["cash_and_equivalents / total_assets"]} → accepted, tree.applies(...)
def test_redefinition_rejected(): id "gross_margin" → issues contains "redefines existing node"
def test_proposal_wrong_template_leaf_rejected(): template bank, formula uses net_revenue → not accepted, "unknown reference"
def test_bad_formula_rejected_not_raised(): "a ** 2" → accepted False, "formula syntax"
def test_original_tree_unchanged()
```

- [ ] **Step 2: Run the tests to verify they fail.** Target `tests.test_calculation_proposals`.
- [ ] **Step 3: Implement `validate_proposal`.**
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit**

```bash
git add src/calculation/proposals.py tests/test_calculation_proposals.py
git commit -m "feat(calculation): validate LLM-proposed metric decompositions"
```

---

### Task 8: Sandbox

**Files:**
- Create: `src/calculation/sandbox.py`
- Create: `tests/test_calculation_sandbox.py`

**Interfaces:**
- Consumes: `build_helpers`, `Trace`, `Answer`, `ExecutionContext` (Task 6); `MetricTree` (Task 3); `FactSet` (Task 4); `CalculationError`, `UnitError` (Task 1)
- Produces:
  - `@dataclass(frozen=True) class SyntaxRejected: line: int; node: str; reason: str`
  - `def check_code(code: str) -> list[SyntaxRejected]`
  - `@dataclass(frozen=True) class ExecutionResult: answer: Answer; trace: Trace`
  - `@dataclass(frozen=True) class ExecutionFailure: kind: Literal["rejected","timeout","calculation_error","runtime_error"]; detail: str; rejections: tuple[SyntaxRejected, ...] = ()`
  - `ALLOWED_BUILTINS = ("len","min","max","sum","abs","sorted","range","enumerate","zip","list","dict")`
  - `def run(code, tree, facts, context, *, timeout: float = 2.0, startup_timeout: float = 20.0) -> ExecutionResult | ExecutionFailure`

- [ ] **Step 1: Write the failing tests**

```python
ESCAPES = {
 "import": "import os\ndef solution(): return 1",
 "dunder_import": "def solution(): return __import__('os')",
 "class_chain": "def solution(): return ().__class__.__bases__",
 "getattr": "def solution(): return getattr(1, 'real')",
 "open": "def solution(): return open('x')",
 "eval": "def solution(): return eval('1')",
 "while": "def solution():\n    while True: pass",
 "lambda": "def solution(): return (lambda: 1)()",
 "pow": "def solution(): return 10 ** 10 ** 10",
 "try": "def solution():\n    try: return 1\n    except: return 2",
 "two_defs": "def solution(): return 1\ndef other(): return 2",
 "no_solution": "x = 1",
 "fstring": "def solution(): return f'{1}'",
 "round": "def solution(): return round(1)",
 "nested_def": "def solution():\n    def g(): return 1\n    return g()",
}
def test_escapes_rejected_statically(): every ESCAPES value → check_code non-empty and run(...).kind == "rejected"
def test_unknown_name_runtime_error(): "def solution(): return foo" → kind "runtime_error", detail == "NameError"
def test_timeout(): "def solution():\n    return sum(i for i in range(1000000000000))" with timeout=1.0 → kind "timeout"
def test_calculation_error_kind(): division of money by money(0,"dong") → kind "calculation_error"
def test_literals_are_decimal(): "def solution(): return answer(money(0.1,'dong') + money(0.2,'dong'), 'dong')" → value == M("0.3")  (exact, not 0.30000000000000004)
def test_success_returns_answer_and_trace(): metric("interest_coverage", ...) → ExecutionResult, trace.metrics_used() includes it
def test_detail_has_no_exception_text(): a raised ValueError("secret") → detail == "ValueError"
```

- [ ] **Step 2: Run the tests to verify they fail.** Target `tests.test_calculation_sandbox`.
- [ ] **Step 3: Implement `src/calculation/sandbox.py`.**
  - `check_code` whitelists AST nodes per spec §10, with one top-level `FunctionDef` named `solution` and no arguments. It rejects `Attribute`, any `Name`/arg containing `__`, `Pow`, `While`, `Lambda`, `Try`, `With`, `Global`, `Nonlocal`, `ClassDef`, `JoinedStr`, `FormattedValue`, `Starred` in calls, nested `FunctionDef`, and calls to names outside helpers ∪ `ALLOWED_BUILTINS`.
  - `run`:
    1. Calls `check_code` and returns `rejected` on any issue.
    2. Rewrites numeric constants to `_num("<source segment>")` (D3) with an `ast.NodeTransformer`.
    3. Compiles and starts a `multiprocessing.get_context("spawn").Process`, passing code text, tree, facts and context.
    4. In the child: build helpers and execute with `{"__builtins__": {name: builtins[name] for name in ALLOWED_BUILTINS}, **helpers, "_num": _num}`, send `("started",)`, call `solution()`, and send `("ok", answer, trace)` or `("error", kind, class_name)`. `CalculationError` maps to `calculation_error`; everything else maps to `runtime_error`.
    5. The parent waits for `started` within `startup_timeout`, then for the result within `timeout` (D7), and on expiry calls `terminate()`, then `join(1)`, then `kill()`.
  - The returned trace is capped at 10,000 steps; overflowing becomes `runtime_error` "TraceOverflow". A non-`Answer` return becomes `runtime_error` "InvalidReturn".

  Document in the module docstring that no memory cap is enforced on Windows.
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit**

```bash
git add src/calculation/sandbox.py tests/test_calculation_sandbox.py
git commit -m "feat(calculation): add AST-whitelisted spawn sandbox"
```

---

### Task 9: Gates and golden programs

**Files:**
- Create: `src/calculation/gates.py`
- Create: `tests/test_calculation_gates.py`
- Create: `tests/test_calculation_golden.py`
- Modify: `src/calculation/__init__.py`: re-export the public names listed in the Interfaces blocks of Tasks 1–9

**Interfaces:**
- Consumes: everything above
- Produces:
  - `ALLOWED_CONSTANTS = frozenset(Decimal(x) for x in (0, 1, 2, 4, 12, 100, 365))`
  - `@dataclass(frozen=True) class GateIssue: code: Literal["provenance","formula","unit_type","sanity","cross_check","agreement"]; severity: Literal["error","warning"]; message: str; line: int | None = None; trace_step: int | None = None`
  - `@dataclass(frozen=True) class GateReport: issues: tuple[GateIssue, ...]` with property `passed` (no error issues)
  - `def question_literals(question: str) -> frozenset[Decimal]`: numbers in the question (VN format via `parse_vn_decimal(..., locale="vi")`), plus percentages as both `n` and `n/100`, plus a number followed by a money scale word as `n × scale` too
  - `def requested_answer_kind(question: str) -> AnswerKind`: "năm nào" → year; "công ty nào"/"doanh nghiệp nào"/"ngân hàng nào" → ticker; "có bao nhiêu" + (doanh nghiệp|công ty|đơn vị|năm|ngân hàng) → count; otherwise quantity
  - `def requested_unit(question: str) -> str | None`: the D5 name, or None. "điểm phần trăm" → percentage_points (checked before %); "%"/"phần trăm" → percent; "lần"/"vòng" → times; "ngày" only after "bao nhiêu" → days; money via `parse_requested_unit` mapped onto `MONEY_SCALES` names
  - `def compare_answers(claimed: Answer, executed: Answer) -> bool`
  - `def run_gates(code: str, result: ExecutionResult, tree: MetricTree, facts: FactSet, context: ExecutionContext, claimed: Answer | None = None) -> GateReport`
  - Gate rules are exactly spec §11 with: literals read from the original AST; allowed years are integers 1900–2100; `formula` checks `context.named_metrics ⊆ trace.metrics_used()`; `unit_type` compares `answer.kind` with `context.requested_kind` and `answer.unit` with `context.requested_unit` when given (money names and "money_per_share" are mutually compatible because display converts; "percent" ≠ "percentage_points"); `sanity` uses the node `expected_range` for the last `metric` step and `abs(money) < 10^16`; `cross_check` reads `trace.cross_checks()` (D10); `agreement` rounds both values in the claimed unit to 2 places (or the claim's own precision when it has more) and allows ≤ 0.2% relative; year/ticker/count/boolean must be equal; both zero → equal.

- [ ] **Step 1: Write the failing gate tests** (`tests/test_calculation_gates.py`)

```python
def test_question_literals():
    ql = question_literals("có bao nhiêu DN có chi phí lớn hơn 20 tỷ đồng và ICR dưới 1,5 lần, giảm 10%")
    assert {D(20), D("20000000000"), D("1.5"), D(10), D("0.10")} <= ql
def test_requested_kind_without_unit():
    assert requested_answer_kind("Năm nào có số dư lớn nhất?") == "year" and requested_unit("Năm nào có số dư lớn nhất?") is None
    assert requested_answer_kind("có bao nhiêu doanh nghiệp có ...?") == "count"
def test_requested_unit_variants(): "bao nhiêu điểm phần trăm" → "percentage_points"; "bao nhiêu %" → "percent";
    "bao nhiêu lần" → "times"; "bao nhiêu tỷ đồng" → "ty"; "bao nhiêu nghìn tỷ đồng" → "nghin_ty"
def test_invented_literal_fails_provenance(): code uses 0.37 not in question → error "provenance"
def test_year_literal_allowed()
def test_hand_rolled_named_metric_fails_formula(): named_metrics={"interest_coverage"}, code divides facts itself → error "formula"
def test_ratio_answer_to_money_question_fails_unit(): requested "ty", answer unit "percent" → error "unit_type"
def test_cross_check_mismatch_error(): reported gross profit off by > tolerance → error "cross_check"
def test_sanity_out_of_range_warning_only(): passed stays True, one warning
def test_agreement(): claimed 1.31 vs executed 1.3108 times → passes; claimed 1.35 → error "agreement"; years equal → pass
```

- [ ] **Step 2: Run them to verify they fail.** Target `tests.test_calculation_gates`.
- [ ] **Step 3: Implement `src/calculation/gates.py`.**
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Write the golden tests** (`tests/test_calculation_golden.py`). Each test builds synthetic facts with `FactSource` (chunk ids like `"syn:<id>"`) and a hand-computed expected value written in a comment. It runs `sandbox.run`, asserts the answer, and asserts `run_gates(...).passed`. Shapes (ViFinQA ids):
  1. lookup (1)
  2. difference between two companies (760)
  3. growth 2019→2023 (606)
  4. mean of four companies' shares (976)
  5. argmax year by `liabilities_to_equity`, then `interest_coverage` of that year (363)
  6. median filter on `liabilities_to_equity`, then argmax of revenue growth, then `gross_margin` (443)
  7. scenario with revenue −10% and a `count_if` of ICR < 1.5 (435)
  8. `pct_point_change` of `gross_margin` (393)
  9. `cagr` selection then `net_margin` (370)
  10. "năm nào" argmax (1000)
  11. "có bao nhiêu" with a `money(20,"ty")` threshold (965)
  12. bank `loan_to_deposit` (bank template)
  13. `days_inventory` with average balances

- [ ] **Step 6: Run the golden tests.** Target `tests.test_calculation_golden`. Expected: 13 OK.
- [ ] **Step 7: Full verification.** Run `python -B -m unittest discover -s tests -v` (expected: all previous tests plus the new ones OK, PostgreSQL skips unchanged), then `python -B -m compileall -q src config` and `git diff --check main...HEAD`. Expected: clean.
- [ ] **Step 8: Commit**

```bash
git add src/calculation/gates.py src/calculation/__init__.py tests/test_calculation_gates.py tests/test_calculation_golden.py
git commit -m "feat(calculation): add verification gates and golden programs"
```

# Reasoning Agent Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Vietnamese question goes through a bounded agent. The agent paraphrases the question, maps metrics to the tree, extracts facts, has the LLM write `solution()`, runs the sandbox and gates, and returns a template answer with sources (or a refusal). A Streamlit page shows every step.

**Architecture:** New package `src/agent/`. Each file has one job: llm, types, metrics, gather, solver, answer, agent, present and factory. The package composes the existing `QueryParaphraser`, `LeafExtractor` (+ `LlmCellLocator`) and calculation core, and changes none of them. `app/demo.py` is a thin Streamlit page over `present.py`.

**Tech Stack:**
- Python 3.13, `unittest`, pydantic v2;
- `langchain_openai.ChatOpenAI` (`gpt-6-luna`);
- Streamlit (new dependency).

**Spec:** `docs/superpowers/specs/2026-11-13-reasoning-agent-demo-design.md` (approved; §4.3 amended in f86ba2e: declared data needs).

## Global Constraints

**Tests**
- Unit tests are offline and deterministic: no network, GPU or real LLM. Every LLM in tests is the scripted fake from Task 1.
- Test command (PowerShell, worktree root):

  ```powershell
  $env:PYTHONPATH="$PWD;$PWD\src"; $env:TEMP="$PWD\.tmp-tests"; $env:TMP=$env:TEMP; D:\finance_project\.venv\Scripts\python.exe -B -m unittest discover -s tests
  ```

  Git Bash breaks the `;` separator.
- The sandbox spawns processes. Probe scripts that call `sandbox.run` must be files with an `if __name__ == "__main__":` guard, never stdin.

**Imports and data**
- Importing any `src.agent.*` module must not import `streamlit`, `torch`, `chromadb` or `langchain_openai`. These load lazily inside functions.
- The corpus is read-only.
- Never print or log `OPENAI_API_KEY`.
- Real LLM calls happen only in Task 9, and only on the user's go-ahead.

**Unchanged code**
- No changes to `src/calculation`, `src/extraction`, `src/query_paraphrase` or `src/retrieval`.

**Agent behaviour (from the spec)**
- Model `gpt-6-luna`, timeout 60 s, `max_retries=1`.
- `AgentConfig` defaults:
  - `retriever_mode="none"`
  - `locator_enabled=True`
  - `max_repairs=3`
  - `max_gather_rounds=2`
  - `max_locator_calls=6`
  - `default_scope="consolidated"`
- Statuses: `answered`, `clarify`, `refused`, `error`.
- Refusal codes: `plan_invalid`, `metric_unmapped`, `facts_missing`, `gates_failed`. Error code: `llm_unavailable`.
- Answer text is built by code only. The LLM never writes the final sentence.

**Git**
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Use `git commit -m`/`-F -`, never a PowerShell `Set-Content` file (it adds a BOM).
- No push. No branch switch, stash or reset.

## Review Focus

1. **Follow-up turn.** `qa_history` reaches the paraphraser unchanged, and a `needs_clarification` plan becomes status `clarify` with the paraphraser's issue messages (Task 7).
2. **Company without a report in the requested scope** (default `consolidated`, only `separate` exists). The extractor's `ReportMissing` surfaces as `facts_missing`, and the refusal bullet names the scope (Tasks 3, 6).
3. **Draft with empty `needs` whose code calls `fact()`.** After `MissingFact`, the next prompt says the data was not declared. The attempt counts as a repair (Task 5).
4. **Provider failure in the middle of solve.** `LlmUnavailable` on attempt 2 gives status `error`, reason `llm_unavailable`, no answer, and the events up to that point are kept (Task 7).
5. **Many companies and the locator cap.** Seven `NotFound` leaves with the cap at 6 → exactly 6 locator calls, and the seventh log entry has `tool="locator_skipped"` (Task 4).

---

### Task 1: Types, LLM port and scripted fake

**Files:**
- Create: `src/agent/__init__.py` (docstring only), `src/agent/types.py`, `src/agent/llm.py`, `tests/agent_fixtures.py`
- Test: `tests/test_agent_llm.py`

**Interfaces (Produces):**
- `src/agent/types.py`:
  - `Scope = Literal["consolidated", "separate"]`
  - `@dataclass(frozen=True) DataNeed(node_id: str, ticker: str, year: int, scope: Scope | None = None, helper: Literal["fact", "metric", "avg_balance"] = "metric")`
  - `@dataclass(frozen=True) LeafLog`, with fields:
    - `leaf_id: str`, `ticker: str`, `year: int`, `scope: Scope`;
    - `outcome: str` (the outcome class name, e.g. `"Found"`, `"NotFound"`);
    - `reason: str | None`, `value: Decimal | None`, `chunk_id: str | None`, `doc_id: str | None`, `page: int | None`, `row_label: str | None`;
    - `tool: Literal["extractor", "locator", "locator_skipped"]`.
  - `AgentConfig`, `StepEvent` and `AgentRun`, exactly as spec §4. `AgentRun.sources: tuple[FactSource, ...]`.
- `src/agent/llm.py`:
  - `class LlmUnavailable(RuntimeError)`: the message is the provider exception's class name only.
  - `StructuredLlm = Callable[[str, type[BaseModel]], BaseModel]`
  - `class CountingLlm` with `__init__(self, inner: StructuredLlm)`, `__call__(prompt, schema) -> BaseModel` and `calls: int`. It counts every call, including failed ones.
  - `def openai_llm(model: str = "gpt-6-luna", *, timeout: float = 60, max_retries: int = 1, client=None) -> StructuredLlm`:
    - it imports `langchain_openai` lazily, only when `client is None`;
    - each call is `client.with_structured_output(schema).invoke(prompt)`;
    - any `Exception` from the provider → `LlmUnavailable(type(exc).__name__)`;
    - a result that is not an instance of `schema` → `LlmUnavailable("MalformedOutput")`.
- `tests/agent_fixtures.py`:
  - `class ScriptedLlm`, with `__init__(self, *replies)`. A reply is a `BaseModel`, or an `Exception` instance to raise.
  - `__call__(prompt, schema)` pops the next reply, asserts `isinstance(reply, schema)`, and records `prompts: list[str]`.
  - It raises `AssertionError("unexpected LLM call")` when no replies are left.

- [ ] **Step 1: Write the failing tests** (`tests/test_agent_llm.py`):
  - `test_counting_llm_counts_success_and_failure`: `CountingLlm(ScriptedLlm(M(x=1), LlmUnavailable("Timeout")))` → `calls == 2` after both calls; the second call raises.
  - `test_openai_llm_maps_provider_error`: a fake client whose `with_structured_output(...).invoke` raises `ConnectionError("secret")` → `LlmUnavailable` with `str(e) == "ConnectionError"`.
  - `test_openai_llm_rejects_wrong_type`: the fake returns a dict → `LlmUnavailable("MalformedOutput")`.
  - `test_scripted_llm_fails_on_unexpected_call`.
  - `test_dataclasses_are_frozen`: assigning to `DataNeed(...)` raises `FrozenInstanceError`.
- [ ] **Step 2: Run** → FAIL (modules missing).
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** `tests.test_agent_llm` → PASS.
- [ ] **Step 5: Commit** `feat(agent): types, structured LLM port and OpenAI adapter`.

### Task 2: Metric mapping

**Files:**
- Create: `src/agent/metrics.py`
- Test: `tests/test_agent_metrics.py`

**Interfaces:**
- Consumes:
  - `QueryPlan` / `Subquery` (`src/query_paraphrase/schemas.py`);
  - `MetricTree.find_by_alias(text, template) -> set[str]`, `MetricTree.applies`, `MetricTree.with_derived`;
  - `validate_proposal(tree, proposal, template) -> ProposalResult`;
  - `TickerTemplates.get(ticker)`;
  - `StructuredLlm`.
- Produces:
  - `CATALOG_TO_TREE: dict[str, str]`:
    - `revenue` and `net_revenue` → `net_revenue`
    - `profit_before_tax` → `profit_before_tax`
    - `profit_after_tax` and `consolidated_profit_after_tax` → `profit_after_tax`
    - `profit_after_tax_parent_share` → `profit_after_tax_parent_share`
  - `@dataclass(frozen=True) MetricMapping(subquery_id: str, node_id: str | None, how: Literal["catalog", "alias", "llm", "proposal", "unmapped"])`
  - pydantic models:
    - `MetricChoice(subquery_id: str, node_id: str | None = None, proposal: dict | None = None)`
    - `MetricChoices(choices: list[MetricChoice])`
  - `def map_metrics(plan: QueryPlan, tree: MetricTree, templates: TickerTemplates, llm: StructuredLlm) -> tuple[tuple[MetricMapping, ...], MetricTree]`:
    - It returns the possibly extended tree.
    - The template is `templates.get(sq.ticker)`, else `"corporate"` when the ticker is None or unknown.
    - The order is spec §4.1. **At most one LLM call**, covering all unresolved subqueries.
    - The LLM prompt lists `node_id — label_vi` for every leaf and derived node that `applies` to the template. Each mapped id must `applies(node_id, template)`, else the subquery is `unmapped`.

- [ ] **Step 1: Write the failing tests.** Build plans with `sq`/`plan` from `tests/test_query_validation.py`, a `TickerTemplates({"VJC": "corporate", "ACB": "bank"})` and the real tree.
  - `test_catalog_map_targets_exist_in_tree`: every value in `CATALOG_TO_TREE` passes `tree.applies(v, "corporate")`.
  - `test_catalog_metric_maps_without_llm`: `metric="revenue"` → `("q1", "net_revenue", "catalog")`; `ScriptedLlm()` gets no call.
  - `test_metric_text_alias_maps_without_llm`: `metric_text="chi phí lãi vay"` → `interest_expense`, `how="alias"`.
  - `test_unresolved_texts_use_one_llm_call`: two subqueries with unknown texts → one call returning both choices.
  - `test_llm_node_must_apply_to_template`: the LLM picks `loan_to_deposit` for VJC (corporate) → `unmapped`.
  - `test_valid_proposal_extends_tree` and `test_rejected_proposal_is_unmapped`. Use a proposal shaped like the existing passing / failing cases in `tests/test_calculation_proposals.py`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(agent): map paraphrased metrics to metric tree nodes`.

### Task 3: Gather

**Files:**
- Create: `src/agent/gather.py`
- Test: `tests/test_agent_gather.py`

**Interfaces:**
- Consumes:
  - `LeafExtractor.extract(ticker, scope, requests) -> ExtractionResult` and the outcome classes in `src/extraction/outcomes.py`;
  - `CellLocator` / `LocatorResult` (`src/extraction/extractor.py`);
  - `MetricTree.required_leaves(node_id, template, year)`;
  - `FactSet`, and `DataNeed` / `LeafLog` from Task 1.
- Produces:
  - `class CappedLocator` (implements `CellLocator`):
    - `__init__(self, inner: CellLocator, cap: int)`, with `calls: int` and `skipped: set[tuple[str, int]]` of `(leaf_id, year)`.
    - Past the cap, it returns `LocatorResult(None, "locator_skipped", ())` without calling `inner`.
  - `class Gatherer`:
    - `__init__(self, tree, templates: TickerTemplates, extractor_factory: Callable[[CellLocator | None], LeafExtractor], locator: CellLocator | None, max_locator_calls: int, default_scope: Scope)`;
    - `gather(self, needs: Iterable[DataNeed]) -> tuple[LeafLog, ...]` returns the log entries added by this call;
    - `facts: FactSet`, built from every `Found` so far;
    - `log: tuple[LeafLog, ...]`;
    - `requested(self, need: DataNeed) -> bool`;
    - `failures(self, need: DataNeed) -> tuple[LeafLog, ...]` returns the logs of that need's leaves whose outcome is not `Found`.
  - Rules:
    - The scope is `need.scope or default_scope`.
    - `helper == "avg_balance"` also requests `year - 1`.
    - The leaves are the union of `required_leaves(...)` alternatives.
    - There is one `extract` per `(ticker, scope)` per call, and a `(leaf, ticker, year, scope)` is never requested twice.
    - An unknown ticker or template still calls `extract`, which yields `UnknownTicker` / `UnsupportedTemplate`, logged per leaf with `leaf_id` = the need's node id.
    - `tool` is `"locator"` when `Found.extracted.method == "llm_located"`; `"locator_skipped"` when `(leaf, year)` is in `CappedLocator.skipped` and the outcome is `NotFound`; else `"extractor"`.

- [ ] **Step 1: Write the failing tests.** Use `default_store()`, `store_of`, `doc`, `is_table`, `is_rows` and `StubLocator` from `tests/test_extraction_extractor.py`, and `extractor_factory = lambda loc: LeafExtractor(store, tree, templates, fallback=loc)`.
  - `test_found_leaf_enters_factset`: `DataNeed("interest_expense", "VJC", 2023)` → `facts.get(FactKey(...))` value `-1950254195864`; log `tool="extractor"`.
  - `test_derived_need_requests_all_alternative_leaves`: `gross_margin` → logs for `net_revenue`, `cost_of_goods_sold`, `gross_profit_reported`.
  - `test_missing_scope_report_is_logged` (Review Focus 2): `DataNeed(..., scope="separate")` on a store with only consolidated → outcome `ReportMissing`, `failures(...)` non-empty.
  - `test_no_leaf_requested_twice`: gathering the same need twice → the second call returns `()` and the store counts one extraction.
  - `test_avg_balance_requests_prior_year`.
  - `test_locator_cap_marks_skipped` (Review Focus 5): 7 needs whose leaves are `NotFound`, with a `StubLocator` that finds nothing and `max_locator_calls=6` → stub called 6 times, one log with `tool="locator_skipped"`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(agent): gather facts for data needs through the extractor`.

### Task 4: Answer text

**Files:**
- Create: `src/agent/answer.py`
- Test: `tests/test_agent_answer.py`

**Interfaces (Produces):**
- `def format_value(answer: Answer) -> str`. Quantities use `to_display(answer.value, display_unit(answer.unit))` and `round_display(.., 2)`, with Vietnamese digits (`.` thousands, `,` decimals). `-0,00` becomes `0,00`. The unit labels are:

  | Unit | Label |
  |---|---|
  | `dong` | đồng |
  | `nghin` | nghìn đồng |
  | `trieu` | triệu đồng |
  | `ty` | tỷ đồng |
  | `tram_ty` | trăm tỷ đồng |
  | `nghin_ty` | nghìn tỷ đồng |
  | `percent` | % |
  | `percentage_points` | điểm phần trăm |
  | `times` | lần |
  | `days` | ngày |
  | `money_per_share` | đồng/cổ phiếu |

  For non-quantities:
  - year → `str(value)`;
  - ticker → the value;
  - count → `str(value)`;
  - boolean → `Có` / `Không`.
- `def answer_text(answer: Answer, sources: Sequence[FactSource], description: str | None) -> str`. The first line is `f"{description}: {value}."` when `description` is set, else `f"Trả lời: {value}."`. Then `Nguồn:` and one `- {row_label} — {doc_id}, trang {page}` line per source, de-duplicated, in order.
- `def refusal_text(code: str, details: Sequence[str]) -> str`. Fixed first sentences:
  - `plan_invalid` → "Không phân tích được câu hỏi."
  - `metric_unmapped` → "Không xác định được chỉ tiêu cần tính."
  - `facts_missing` → "Không đủ số liệu để trả lời."
  - `gates_failed` → "Không tạo được phép tính hợp lệ sau nhiều lần thử."
  - `clarify` → "Câu hỏi cần làm rõ thêm."
  - `llm_unavailable` → "Không gọi được mô hình ngôn ngữ, vui lòng thử lại."

  Each detail follows as a `- ` bullet.
- `def leaf_detail(log: LeafLog) -> str` → `f"{leaf_id} {ticker} {year} ({'hợp nhất' if scope == 'consolidated' else 'riêng'}): {outcome}" + (f" — {reason}" if reason else "")`.

- [ ] **Step 1: Write the failing tests:**
  - `test_money_in_ty`: −1,950,254,195,864 VND, unit `ty` → `"-1.950,25 tỷ đồng"`;
  - `test_percent`;
  - `test_negative_zero_normalised`;
  - `test_boolean_and_year`;
  - `test_sources_deduplicated`;
  - `test_each_refusal_code_has_text`;
  - `test_leaf_detail_names_scope` (Review Focus 2).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(agent): Vietnamese answer and refusal templates`.

### Task 5: Solver

**Files:**
- Create: `src/agent/solver.py`
- Test: `tests/test_agent_solver.py`

**Interfaces:**
- Consumes:
  - `sandbox.run`, `ExecutionResult`, `ExecutionFailure`;
  - `run_gates(code, result, tree, facts, context, claimed) -> GateReport` (`.passed`, `.issues`);
  - `ExecutionContext`, `Answer`, `Quantity`, `display_unit`, `MONEY_SCALES`;
  - `Gatherer` (Task 3), `StructuredLlm`, `AgentConfig`.
- Produces:
  - pydantic models:
    - `NeedModel(node_id: str, ticker: str, year: int, scope: Literal["consolidated", "separate"] | None = None, helper: Literal["fact", "metric", "avg_balance"] = "metric")`
    - `SolutionDraft(code: str, claimed_value: str, claimed_unit: str, needs: list[NeedModel])`
  - `@dataclass(frozen=True) Attempt(number: int, code: str, needs: tuple[DataNeed, ...], gathered: tuple[LeafLog, ...], problems: tuple[str, ...], outcome: Literal["passed", "repair", "facts_missing"])`
  - `@dataclass(frozen=True) SolveResult(status: Literal["passed", "facts_missing", "gates_failed"], result: ExecutionResult | None, attempts: tuple[Attempt, ...], failed: tuple[LeafLog, ...])`
  - `def claimed_answer(draft: SolutionDraft) -> Answer | None`. It inverts `to_display`:
    - `kind` comes from the unit (`year` / `ticker` / `count` / `boolean`, else `quantity`);
    - money scales: × `MONEY_SCALES[unit]`, `Dimension.MONEY`;
    - `percent` and `percentage_points`: ÷ 100, `Dimension.RATIO`;
    - the other quantity units mirror `to_display`.

    It returns `None` when `claimed_value` does not parse; the gates then see `claimed=None`.
  - `def solve(question: str, plan: QueryPlan, mappings: Sequence[MetricMapping], gatherer: Gatherer, tree: MetricTree, context: ExecutionContext, llm: StructuredLlm, config: AgentConfig, on_attempt: Callable[[Attempt], None]) -> SolveResult`
  - `def build_context(question: str, mappings, templates: TickerTemplates, default_scope: Scope) -> ExecutionContext`, using `question_literals`, `requested_unit`, `requested_answer_kind`, `templates.calculation_templates()` and `named_metrics` = the mapped ids.
- Loop (spec §4.3 as amended):
  - Attempts are at most `1 + config.max_repairs`.
  - Before running each draft, gather its new needs while gather rounds remain (`config.max_gather_rounds`). A need with a node id unknown to the tree → problem `unknown node id: X`, and the code is not run.
  - On `ExecutionFailure` with detail `MissingFact` / `MetricUnavailable`:
    - if any declared need has `failures` → status `facts_missing` (with those logs);
    - otherwise → problem `"code used data not declared in needs"`.
  - Other failures and the gates' error issues become problems.
  - The prompt contains:
    - the question;
    - `subquery id → node id`;
    - the helper names with their docstrings (`inspect.getdoc` of `build_helpers(...)` values);
    - the code constraints from the calculation-core spec §16 as a fixed text block in `solver.py`;
    - the fact inventory lines `metric_id ticker year scope = value`;
    - on a repair, the last code and its problems.

- [ ] **Step 1: Write the failing tests.** Use the real sandbox, gates and tree, and the Task 3 gatherer on `default_store()`.
  - `test_claimed_answer_round_trips_every_quantity_unit`: for each quantity unit, `to_display(claimed_answer(draft).value, display_unit(u)) == Decimal(v)`.
  - `test_good_draft_passes_first_attempt`: a draft with `answer(fact("interest_expense", "VJC", 2023), "ty")`, claimed `-1950.25`/`ty` and needs `[interest_expense VJC 2023]` → `passed`, one attempt.
  - `test_bad_then_good_draft`: a syntax-rejected draft, then a good one → `passed`, two attempts; the second prompt contains the rejection.
  - `test_gate_errors_every_attempt_fail`: 4 drafts whose claim disagrees → `gates_failed`, 4 attempts.
  - `test_new_need_gathers_before_running`: the first draft declares a need not yet gathered → the store is extracted, then success. It is not counted as a repair.
  - `test_failed_declared_need_is_facts_missing`: a need for 2030 → `ReportMissing` → `facts_missing`, `failed` non-empty.
  - `test_undeclared_data_is_repair` (Review Focus 3): `needs=[]` but the code calls `fact(...)` for an ungathered key → the next prompt contains `"code used data not declared in needs"`.
  - `test_unknown_node_in_needs_is_repair_without_running`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(agent): solver loop with declared data needs, sandbox and gates`.

### Task 6: Present

**Files:**
- Create: `src/agent/present.py`
- Test: `tests/test_agent_present.py`

**Interfaces (Produces):** `def rows(event: StepEvent) -> list[dict[str, str]]`, one table per event kind. Column names are Vietnamese and fixed:

| Event kind | Columns |
|---|---|
| plan | `Câu hỏi con`, `Mã`, `Năm`, `Phạm vi`, `Chỉ tiêu`, `Cụm từ` |
| metrics | `Câu hỏi con`, `Node`, `Cách` |
| gather | `Leaf`, `Mã`, `Năm`, `Phạm vi`, `Kết quả`, `Giá trị`, `Nguồn`, `Công cụ` |
| solve | `Lần`, `Kết quả`, `Vấn đề` |
| answer | `Trạng thái`, `Lý do` |

The `StepEvent.data` shapes are defined here as module constants. The event producers (Task 7) must use these key names.

Also `def footer(run: AgentRun) -> str` → `f"{run.llm_calls} lần gọi LLM · {run.seconds:.1f} giây"`.

- [ ] **Step 1: Write the failing tests:** one test per event kind with a hand-built `StepEvent`, checking the column names and one cell; `test_gather_failure_row_shows_reason` (Review Focus 2); `test_footer`.
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Commit** `feat(agent): display rows for agent events`.

### Task 7: Agent orchestration

**Files:**
- Create: `src/agent/agent.py`
- Test: `tests/test_agent_agent.py`

**Interfaces:**
- Consumes: Tasks 1–6, plus `QueryParaphraser(llm, registry=..., provider_errors=(LlmUnavailable,))` built by the caller.
- Produces: `class ReasoningAgent` with:
  - `__init__(self, paraphraser, tree: MetricTree, templates: TickerTemplates, extractor_factory: Callable[[CellLocator | None], LeafExtractor], llm: CountingLlm, config: AgentConfig = AgentConfig())`;
  - `run(self, question: str, qa_history: Sequence[QATurn | Mapping] = (), on_event: Callable[[StepEvent], None] | None = None) -> AgentRun`.
- Flow (spec §5):
  1. Paraphrase:
     - `needs_clarification` → `clarify`, with the issue messages as details;
     - `invalid` → `refused/plan_invalid`;
     - `unavailable` → `error/llm_unavailable`.
  2. A subquery without `ticker` or `year` → `refused/plan_invalid`.
  3. Map metrics. Any `unmapped` → `refused/metric_unmapped`.
  4. Create a `Gatherer`. Its locator is `LlmCellLocator(lambda p: llm(p, PointerAnswer), llm_errors=(LlmUnavailable,))` when `config.locator_enabled`, else `None`.
  5. Gather the needs `DataNeed(node_id, sq.ticker, sq.year, sq.scope)`.
  6. Solve, then answer: `answered` with `answer_text(...)` and sources from `result.trace.facts_used()` through `gatherer.facts.get`. `facts_missing` details come from `leaf_detail`.
- `LlmUnavailable` from any step → `error/llm_unavailable`, keeping the events so far.
- `llm_calls` = the `CountingLlm.calls` delta for this run; `seconds` is wall time.
- Events go in this order: plan, metrics, gather (one per gather call, including rounds inside solve), solve (one per attempt), answer. Each one is passed to `on_event` as it happens.

- [ ] **Step 1: Write the failing tests** (end-to-end with `ScriptedLlm`, the real paraphraser and validator on the `sq`/`plan` builders, and `default_store()`):
  - `test_lookup_answered`: "Chi phí lãi vay năm 2023 của VJC là bao nhiêu tỷ đồng?" → `answered`; text starts `Trả lời: -1.950,25 tỷ đồng.`; one source; `llm_calls == 2`; event kinds `[plan, metrics, gather, solve, answer]`.
  - `test_history_forwarded_and_clarify` (Review Focus 1): the scripted plan makes the validator return `needs_clarification` → `clarify`, and the paraphraser prompt contains the history question.
  - `test_invalid_plan_refused`, `test_unmapped_metric_refused`, `test_facts_missing_refused`.
  - `test_llm_failure_mid_solve_is_error` (Review Focus 4): the solve reply 2 raises `LlmUnavailable` → `error`, `answer is None`, and the events include the first solve attempt.
  - `test_on_event_called_in_order`.
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Commit** `feat(agent): reasoning agent orchestration with events`.

### Task 8: Factory, Streamlit page and isolation

**Files:**
- Create: `src/agent/factory.py`, `app/demo.py`
- Modify: `requirements.txt` (add `streamlit`)
- Test: `tests/test_agent_factory.py`

**Interfaces:**
- `def build_agent(config: AgentConfig, *, llm: CountingLlm, db: Path, bm25_dir: Path, chroma_dir: Path | None) -> ReasoningAgent`:
  - It wires `SqliteReportStore(db)`, `MetricTree.load()`, `load_ticker_templates()`, `EntityRegistry.from_csv(PROJECT_ROOT / "data" / "code_stock.csv")` and `QueryParaphraser(lambda p: llm(p, QueryPlan), registry=..., provider_errors=(LlmUnavailable,))`.
  - `extractor_factory` builds `LeafExtractor(store, tree, templates, fallback=loc, retriever=r)`. `r` is `None` for mode `none`, else `open_table_retriever(db, bm25_dir, chroma_dir=chroma_dir, mode=config.retriever_mode)`, opened once and lazily.
- Environment defaults (read in `app/demo.py` only):
  - `FINAGENT_DB` = `data/databases/corpus.sqlite3`
  - `FINAGENT_BM25_DIR` = `data/indexes/bm25`
  - `FINAGENT_CHROMA_DIR` unset → hybrid is disabled in the sidebar.
- `app/demo.py`, as in spec §6:
  - `st.cache_resource` holds the agent per `AgentConfig`, and `openai_llm()` is built only if `OPENAI_API_KEY` is set, else an error banner.
  - Chat history lives in `st.session_state["history"]` as a list of `{"question", "answer"}`.
  - The status placeholders are updated through `on_event`.
  - One `st.expander` per event shows `st.dataframe(rows(event))`; solve expanders also show `st.code(code)`. The footer goes below.

- [ ] **Step 1: Write the failing tests:**
  - `test_build_agent_without_retriever_touches_no_index`: `bm25_dir` and `chroma_dir` point to non-existent paths and the mode is `none` → construction succeeds.
  - `test_agent_package_imports_are_light`: in a subprocess (`python -c`), import every `src.agent` module, then assert none of `streamlit`, `torch`, `chromadb`, `langchain_openai` is in `sys.modules`.
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS. Also run `python -m compileall -q src config scripts app` and the full suite.
- [ ] **Step 5 (controller asks the user):** install `streamlit` into `D:\finance_project\.venv` only after the user's approval. Then check that `streamlit run app/demo.py --server.headless true` starts and serves HTTP 200 on its port, and stop it.
- [ ] **Step 6: Commit** `feat(agent): agent factory and Streamlit demo page`.

### Task 9: Real-model smoke (controller, user approval required)

- [ ] **Step 1:** Ask the user before spending money. Run about 6 gold questions through `build_agent` with `openai_llm()` from a scratchpad script (file with a `__main__` guard). Pick one of each:
  - `direct_extraction`;
  - `simple_calculation`;
  - `multi_year_comparison`;
  - `cross_company_comparison`;
  - `multi_hop_conditional`;
  - a follow-up turn on the first one.
- [ ] **Step 2:** For each run, record the status, answer vs `gold_answer_numeric`, LLM calls and seconds in the shared note `C:\Users\ADMIN\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\side-session-notes\local_e62b6974-8989-4969-9f55-319a57ac3857\agent-smoke-2026-11-13.md`. Never execute the gold `calculation_program`.
- [ ] **Step 3:** Hand the page to the user: `streamlit run app/demo.py`.

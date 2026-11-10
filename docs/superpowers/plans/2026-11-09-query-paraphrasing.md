# Query Paraphrasing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validated Vietnamese financial query decomposition and contextual rewriting before existing hybrid retrieval, returning evidence per subquery.

**Architecture:** Inject a callable LLM into QueryParaphraser; validate its structured plan before invoking retrieval. A separate multi-query orchestrator builds individual physical filters and preserves the existing BM25/Dense/RRF/reranker algorithms and legacy call forms.

**Tech Stack:** Python, Pydantic v2, unittest/mocks, existing LangChain RunnableLambda, in-memory BM25. No additional dependencies or external services for basic tests.

**Spec:** `docs/superpowers/specs/2026-11-09-query-paraphrasing-design.md` (approved 2026-11-09).

## Global Constraints

- Use `src/query_paraphrase/`, preserving the existing illustrative field names.
- max_subqueries=8, max_history_turns=5; question and each history field maximum 8,000 characters; year 1900–2199.
- default_report_scope=consolidated, notes_fallback=true; scope unknown in LLM output remains null.
- No corpus/database/indexing changes; no rewrite of BM25, Dense, RRF or reranker.
- Keep string and question/filter_question retrieval APIs and their defaults.
- No inferred current year/company; no alias guessing or accounting metric merging.
- Nullable LLM fields required explicitly; extra fields forbidden; strict primitive types.
- LLM callable/client injected; no API key, GPU or network required for basic tests.
- Invalid/ambiguous output cannot reach retrieval. Semantic checks are explicitly bounded.
- New production modules and new tests import project code only via `src.*` (`config.settings` unchanged); boundaries accept mappings and re-validate foreign Pydantic instances via `model_dump()`, never rely on `isinstance` across import paths.
- Implement sequentially, RED → GREEN → REFACTOR; review each task before its dependent task.

## Review Focus

1. Ticker-looking finance abbreviations (LNST, LNTT) and company names containing another ticker must not become spurious entities (Task 1/3 tests).
2. Explicit ticker/year pairs must not silently become a Cartesian product or get swapped (Task 3 tests).
3. Long history truncation must retain original turn indexes; inherited values from discarded/unrelated/answer-only turns must fail (Task 3/4 tests).
4. Primary retrieval errors must not trigger an unfiltered fallback; partial supplemental failure must remain visible (Task 5 tests).
5. Empty/invalid structured filters, mutable model inputs and conflicting legacy/new filter arguments must fail safely or preserve documented semantics (Task 2/3 tests).

## Verification amendments (2026-11-10, against commit 5e2fd24)

Plan re-verified against current code; approved design unchanged. Owning tasks below carry the tests.

- R1 (Global, Task 2/5): `hybrid.py` imports `src.main.refine_questions` while existing tests import bare `main.*`/`retrieval.*`; one module can load under two names. Use `src.*` imports for new code/tests; extend `test_retrieval_modules_import_from_project_root` to `src.retrieval.filters`, `src.retrieval.multi_query`, `src.main.query_pipeline`.
- R2 (Task 5): all-empty filters make BM25 accept every document and Hybrid send `where=None`. `retrieve_subqueries` must reject empty scopes before any invoke.
- R3 (Task 2): BM25 upper-cases metadata tickers but Chroma `$in` is exact. `MetadataFilters` normalizes tickers with `strip().upper()` and rejects blanks.
- R4 (Task 3/5): `refine_question` maps "công ty mẹ" to separate. Validator reuses only its years (`default_scope=None`), never its scopes; new flow never calls `refine_question`.
- R5 (Task 5): retrieval errors caught are an injected tuple, default `(TimeoutError, ConnectionError, OSError)`; ValueError/KeyError/TypeError propagate.
- R6 (Task 5): worst case ≈ max_subqueries × (primary + notes/letters) sequential hybrid+rerank passes; document the bound and assert call count.
- R7 (Task 3): BaseModel input is always re-validated via `QueryPlan.model_validate(raw.model_dump(), strict=True)`.
- Unverified: strict provider structured output may reject `context_refs` dict/defaulted fields (no live API check); a CSV ticker absent from the corpus yields `empty`.

## File responsibilities and dependencies

- `src/query_paraphrase/schemas.py`: strict LLM and result models, QATurn, SemanticReview (settings live in `config/settings.py`).
- `src/query_paraphrase/metric_catalog.py`: explicit canonical accounting aliases, specific-first matching.
- `src/query_paraphrase/entities.py`: CSV registry and verified injected aliases; no fuzzy matching.
- `config/settings.py`: lazy constructor-time paraphrasing config from environment, no secrets/provider defaults.
- `src/retrieval/filters.py`: validated physical filters independent of paraphrasing.
- `src/main/refine_questions.py`, `src/retrieval/hybrid.py`: minimal optional metadata_filters routing.
- `src/query_paraphrase/validator.py`: deterministic plan/context checks and clear failure results.
- `src/query_paraphrase/prompts.py`, `paraphraser.py`, `__init__.py`: prompt, injectable LLM/service and exports.
- `src/retrieval/multi_query.py`: policy, per-subquery retrieval and evidence statuses.
- `src/main/query_pipeline.py`: Runnable entrypoint combining paraphraser and orchestrator.
- Tests mirror these responsibilities; `docs/query-paraphrasing.md` documents use and limitations.

Dependency order: Task 1 → Task 2 (independent contract, executed sequentially) → Task 3 (uses 1) → Task 4 (uses 1/3) → Task 5 (uses 1/2/4). Final review and regression follow all tasks.

## Execution setup (before product changes)

Updated 2026-11-10: skeleton files are tracked since `5e2fd24`; `.worktrees/` is already ignored (`9c6d577`). A prior run left branch `codex/query-paraphrasing` (base 176e47e) in `.worktrees/query-paraphrasing` with an uncommitted Task 1 RED test; leave it untouched.

- [ ] Create branch `feat/query-paraphrasing-impl` from `feat/query-paraphrasing` (5e2fd24) in `.worktrees/query-paraphrasing-v2` via using-git-worktrees. Do not carry over the main checkout's uncommitted `prompts.py` edit.
- [ ] Copy `tests/test_query_contracts.py` from `.worktrees/query-paraphrasing` and change its `query_paraphrase.*` imports to `src.query_paraphrase.*`. Do not copy .venv, caches, corpus or credentials.
- [ ] Run baseline from worktree using `D:\finance_project\.venv\Scripts\python.exe -B -m unittest discover -s tests -v`, with PYTHONPATH set to worktree root plus its src and TEMP/TMP set to a writable `<worktree>\.tmp-tests`. Expect ~64 tests, 3 PostgreSQL skips, plus test_query_contracts RED. Record counts, skips, failures and runtime. Fix no baseline failure without systematic-debugging; separate unrelated existing failures from feature regressions.
- [ ] Use explicit worktree cwd for every worker and review. Commit only owned files; no force pushes, merges or deployment.

### Task 1: Strict contracts, metric catalog and company registry

**Files:** Complete `src/query_paraphrase/schemas.py`; create `metric_catalog.py`, `entities.py`; modify `config/settings.py`; `tests/test_query_contracts.py` (ported RED file, see setup).

**Consumes:** approved schema, local CSV, existing fold_text/clean_text helpers.

**Produces interfaces:**

- `ParaphraseSettings` with approved defaults and bounds; `load_paraphrase_settings(**overrides) -> ParaphraseSettings` resolves environment at call time. Use FINAGENT_PARAPHRASE_MAX_SUBQUERIES, MAX_HISTORY_TURNS, MAX_INPUT_CHARS, DEFAULT_REPORT_SCOPE and NOTES_FALLBACK (all with FINAGENT_PARAPHRASE_ prefix).
- `QATurn(question, answer=None)`, `Subquery`, `QueryRelationship(kind, subquery_ids, description)` with kind in lookup|compare|aggregate|calculate, `QueryPlan`, `ValidationIssue`, `ParaphraseResult`, `SemanticReview(status, reason)`.
- `find_metrics(text: str) -> set[str]`, `normalize_metric(text: str) -> str | None`, `METRIC_CATALOG`.
- `EntityRegistry.from_csv(path, aliases=None)`, `.resolve(name: str) -> set[str]`, `.find_mentions(text: str) -> set[str]`, `.company_name(ticker: str) -> str | None`, `.unknown_ticker_candidates(text: str) -> set[str]`. CSV header is `Mã CK,Tên công ty`. Aliases map alias to one ticker or a list; collisions remain sets. Unknown uppercase ticker candidates must be inspectable by validator without treating finance acronyms as companies.

- [ ] Write unittest cases asserting all required nullable fields missing cause ValidationError, empty/extra fields and bool/string year fail, valid null metadata passes; settings load defaults 8/5/8000/consolidated/true and reject malformed env values.
- [ ] Write catalog tests asserting LNST→profit_after_tax, LNTT→profit_before_tax, net revenue distinct from revenue, parent share/consolidated profits distinct, longest metric suppresses contained generic matches, unknown metric→None.
- [ ] Write registry tests for actual VJC/full name, explicit case-insensitive ticker, unknown ticker, ambiguous injected alias, invalid alias target, duplicate CSV ticker and LNST/LNTT exclusion. Names such as CTCP Viễn thông FPT must not also imply ticker FPT when matching FOX's full name.
- [ ] Run `python -B -m unittest discover -s tests -p test_query_contracts.py -v`; record expected RED from missing contracts.
- [ ] Implement the interfaces and models minimally; keep registry CSV loaded once per constructed service, not once per subquery.
- [ ] Repeat the command for GREEN, inspect changes, remove duplication without changing behavior, rerun.
- [ ] Obtain fresh spec review and quality review, resolve confirmed findings, commit owned files.

### Task 2: Per-request structured retrieval filters

**Files:** Create `src/retrieval/filters.py`, `tests/test_metadata_filters.py`; modify `src/main/refine_questions.py`, `src/retrieval/hybrid.py`; extend `tests/test_hybrid_reranker.py` only if necessary.

**Produces interfaces:** `MetadataFilters(tickers: list[str], years: list[int], scopes: list[str])`; `retrieve_refined(..., metadata_filters: MetadataFilters | dict | None = None)`; Hybrid mapping accepts metadata_filters. Model serializes using existing plural filter keys. Parent normalized to separate. Reject filter_question together with metadata_filters.

- [ ] Write tests with VJC 2022/2023 and HPG fixtures: invoke search text mentioning all years but explicit 2022 filter, assert lexical candidates and recorded Chroma where contain only VJC/2022/consolidated. Both branches use identical restrictions.
- [ ] Add invalid type/year/unknown scope/extra field and conflicting input tests; verify no Dense call on invalid input. Verify empty dict is rejected as incomplete filter mapping; explicit empty lists mean unrestricted only for their fields. Verify no mutation and parent alias compatibility. Lowercase/padded tickers (`" vjc "`) normalize to `VJC` identically for BM25 and Chroma where; blank tickers fail (R3). Extend the import-from-root subprocess test with `src.retrieval.filters` (R1).
- [ ] Pin legacy string/filter_question calls and reranker mapping forwarding, keeping current result types and default consolidated.
- [ ] Run new tests for RED, then implement only structured filter selection/normalization and forwarding; preserve BM25 scoring/RRF unchanged.
- [ ] Run `python -B -m unittest discover -s tests -p test_metadata_filters.py -v`, existing test_refine_questions.py and test_hybrid_reranker.py for GREEN; inspect/refactor minimally and rerun.
- [ ] Fresh spec review, quality review, fixes and commit owned files.

### Task 3: Deterministic validation and history provenance

**Files:** Complete `src/query_paraphrase/validator.py`; create `tests/test_query_validation.py`.

**Consumes:** Task 1 models/catalog/registry.

**Produces interfaces:** `validate_plan(raw, question: str, qa_history: list[QATurn] | None = None, *, registry: EntityRegistry, settings: ParaphraseSettings) -> ParaphraseResult`; input helper `prepare_history(qa_history, settings) -> list[tuple[int, QATurn]]` retains original indexes. Helpers may live in validator.py rather than introduce a separate framework.

- [ ] Write tests for simple lookup valid, multiple years/metrics/entities valid, duplicate queries/IDs, unknown relation IDs, max 8 configurable, original_question mismatch, malformed JSON/markdown/extra fields and model instances modified after construction.
- [ ] Add preservation tests for omitted/invented ticker/year/recognized metric/scope/period/unit, metadata contradicting question/search text, distinct metric variants, unknown metric_text, unsupported operation and missing comparison/calculation intent. Structured null must fail when an explicit recognized constraint is lost.
- [ ] Add paired VJC-2022/HPG-2023 tests rejecting swapped bindings and Cartesian multiplication; verify the VJC-2022/2023 comparison permits exactly the two requested needs. For ambiguous supported binding/context return needs_clarification rather than guessing.
- [ ] Test unknown/ambiguous entity input blocks retrieval even when model supplies a real different ticker; missing company without any mention can remain null. Verify full-name ticker resolution and metadata consistency, no fuzzy aliases.
- [ ] Test follow-up "còn năm 2023?" inherits missing entity/metric only with valid context_refs; current year overrides history year; unrelated/answer-only/ambiguous/out-of-window context fails. Test original indexes after retaining last five turns and no scope separate inferred from parent-share profit.
- [ ] Test "LNST thuộc cổ đông công ty mẹ VJC 2023" keeps scope=None and is valid (R4); a mutated or adapter-returned QueryPlan instance is re-validated via `model_dump()` (R7).
- [ ] Run test_query_validation.py for RED, implement bounded extraction/comparison using catalog/registry and only the years from `refine_question(..., default_scope=None)`; never use its scopes. Do not claim full semantic proof. Separate structural invalid, context/entity clarification, and semantic-not-run status; return plan=None on nonvalid.
- [ ] Run all Task 1/3 tests for GREEN, simplify repeated checks, rerun; obtain both reviews and commit.

### Task 4: Vietnamese prompt and injectable paraphrasing service

**Files:** Replace unused FinSage contents in `prompts.py`; complete `paraphraser.py`, `__init__.py`, `tests/test_query_paraphrasing.py`.

**Consumes:** Task 1/3 contracts and validation.

**Produces interfaces:** `build_paraphrase_prompt(question, history, *, settings) -> str`; `QueryParaphraser(llm, *, registry=None, settings=None, semantic_reviewer=None).paraphrase(question, qa_history=None) -> ParaphraseResult`; `structured_llm_adapter(client) -> callable` wraps client.with_structured_output(QueryPlan) without loading client dependencies eagerly. Semantic reviewer callable accepts question/history/plan and returns SemanticReview; failed/unknown prevent retrieval.

- [ ] Write mocked tests asserting prompt contains Vietnamese rewriting/schema/catalog guidance and delimited user data, no Lotus/2024 defaults or three-line English output. A single simple need remains one subquery in fixture, comparisons retain independent needs.
- [ ] Assert history trimming/original refs and input limits before LLM invocation, original history not mutated; answer content is explicitly unverified. Passing supplied dicts/Pydantic QATurns follows the same normalization path.
- [ ] Test one callable invocation, dict/JSON/QueryPlan responses routed through validator, wrong types/schema→invalid, TimeoutError/connection/provider exceptions→unavailable with sanitized diagnostics; do not mask programmer/config errors. Define provider failures by an explicit injected exception tuple with defaults TimeoutError/ConnectionError, allowing caller to add provider classes.
- [ ] Test semantic reviewer passed/failed/unknown and unavailable reviewer result; no implicit second LLM call. Assert adapter invokes structured client and still validates returned model locally.
- [ ] Run test_query_paraphrasing.py for RED; implement prompt/service, no retry or output repair; keep provider and model selection external.
- [ ] Run Task 1/3/4 tests for GREEN; review/refactor/rerun, fresh spec and quality reviews, commit.

### Task 5: Grouped evidence orchestration and end-to-end integration

**Files:** Complete `src/retrieval/multi_query.py`, `src/main/query_pipeline.py`, `tests/test_multi_query.py`; create `docs/query-paraphrasing.md`.

**Consumes:** ParaphraseResult, settings, Subquery; Task 2 retriever mapping.

**Produces interfaces:** `RetrievalPolicy(default_report_scope='consolidated', notes_fallback=True)`; `retrieve_subqueries(result: ParaphraseResult, retriever, *, policy=None, retrieval_errors: tuple[type[Exception], ...] = (TimeoutError, ConnectionError, OSError)) -> list[EvidenceGroup]`; `build_query_pipeline(paraphraser, retriever, *, policy=None)` returns RunnableLambda. Define EvidenceGroup/PipelineResult as small dataclasses or Pydantic models locally; expose status, subquery, filters, primary_evidence, supplemental_evidence, warnings and verification labels. No new retrieval algorithm.

- [ ] Write tests asserting invalid/needs_clarification/unavailable results produce zero calls; two VJC years produce two exact mappings without filter_question; each group preserves its own Documents/scores and original source metadata.
- [ ] Add explicit consolidated/separate and default-policy tests; requested scope remains None when policy supplies default. Notes go only to supplemental with scope_verified=false/report_scope=None, never primary. Explicit letters and combined document types remain distinct; notes-only scoped result is partial.
- [ ] Test one explanations fallback on empty primary, no fallback after primary errors, no widening ticker/year/scope. Supplemental errors remain visible, continue other groups, overall partial/error rules; all empty→empty. Period/unit requests retain metadata and period_verified=false warning.
- [ ] Test empty scopes produce zero retriever calls (R2); only injected `retrieval_errors` become group errors while ValueError/KeyError/TypeError propagate (R5); `refine_question` is never called in the new flow (R4); total retriever calls ≤ max_subqueries × passes (R6). Extend the import-from-root subprocess test with `src.retrieval.multi_query` and `src.main.query_pipeline` (R1).
- [ ] Add pipeline integration with real in-memory BM25 plus fake filtered Dense store and fake encoder. Check actual results and recorded per-subquery where, RRF/rerank scores, string/history entrypoints, multiple metrics/entities and retained relationship metadata.
- [ ] Run test_multi_query.py for RED; implement sequential orchestration and result grouping; catch only documented retrieval runtime failures and preserve diagnostics without credentials/raw provider errors. Pass exact mappings through existing reranker.
- [ ] Run new integrations and full local unittest discovery for GREEN. Document complete usage including injected callable and LangChain adapter, supported entity aliases/context forms, environment options, safe failures, corpus period/scope limitations and untested live API.
- [ ] Fresh spec review, quality review, fix confirmed issues, rerun affected tests, commit owned files.

## Final verification and delivery

- [ ] Use requesting-code-review for a fresh whole-branch read-only review against approved spec and production engineering rules; report actionable confirmed defects separately from unverified risks. No edits by reviewers.
- [ ] For any bug/failure invoke systematic-debugging, reproduce and add an appropriate failing regression before minimal fix; re-review material fixes.
- [ ] Use verification-before-completion: run full `python -B -m unittest discover -s tests -v` in the worktree with explicit PYTHONPATH, then `git diff --check` and inspect branch status/history. Record total, skipped and real failures. Do not claim live API/GPU/PostgreSQL integration ran when skipped.
- [ ] Deliver architecture summary, changed file paths, actual test results, review verdict and remaining semantic/API/corpus limitations. Keep branch/worktree for inspection; no merge/push without user request.

## Plan self-review

Coverage: spec sections 1–3 constrain all tasks; 4–7 covered by Tasks 1/3/4;
8 by Task 2; 9–10 by Task 5; 11 by all test steps/docs/config; 12 by setup and review
gates. Interfaces use one contract source (Task 1) and one physical filter model
(Task 2). Five review-focus inputs have explicit owning tests. No unresolved
implementation decisions require changing the approved feature scope.

# Reasoning Agent, Demo First (Sub-project 3) — Design

Date: 2026-11-13
Status: draft for review
Sub-project: 3 of 4 of "answering and serving" (see `2026-11-10-calculation-core-design.md` §3)
Branch: `feat/reasoning-agent`, cut from `c34969a` on `feat/retrieval-extraction`. The branch
contains the paraphraser (main, PR #1), the calculation core (SP1), leaf extraction (SP2) and the
optimised table retriever. It does not contain the dropped §6.1 work (Task 10).

## 1. Context

Finagent can already do three things:
- paraphrase a Vietnamese question into a validated `QueryPlan` (`QueryParaphraser`);
- extract sourced leaf facts from the reports (`LeafExtractor`);
- run LLM-written `solution()` code in a sandbox behind deterministic gates (calculation core).

Nothing connects these pieces yet, and nothing answers a question. This sub-project connects
them into an agent with this architecture:

```
User → Agent/Planner → Retrieval + Knowledge + Memory + Tools → LLM → Verification/Action
```

| Layer | Finagent part |
|---|---|
| Planner | `QueryParaphraser` (subqueries, history, entities) + metric mapping to the metric tree |
| Tools | `LeafExtractor` (+ `LlmCellLocator`, optional table retriever) |
| Knowledge | metric tree, ticker templates |
| Memory | `qa_history` (`QATurn`s) of the session |
| LLM | writes `solution()` and claims its answer |
| Verification / Action | sandbox, gates, bounded repair, template answer or refusal |

## 2. Agreed decisions

1. **Demo first.** A working web demo is the first deliverable. Accuracy on the 100-question
   gold file (`official_gold.jsonl`) is the next spec.
2. **One model.** `gpt-6-luna` serves every LLM step, through `langchain_openai.ChatOpenAI`
   (timeout 60 s, `max_retries=1`). The model is injected, so tests use fakes and a later
   change of model touches no agent code.
3. **Orchestration "C".** The skeleton is fixed (plan → metrics → gather → solve → answer),
   plus one bounded loop: missing facts send the agent back to gather.
4. **Refuse rather than guess.** The agent refuses when data is missing or the gates still
   fail after the repairs, and it states the reason.
5. **Reuse `QueryParaphraser`.** Its 6-entry metric catalog is mapped to tree ids, and
   `metric_text` covers every other metric.
6. **Template answers.** The final Vietnamese sentence is built by code, never by the LLM, so
   the LLM cannot alter a number.
7. **Simple web page.** Streamlit, kept thin: all logic lives in `src/agent/`, which never
   imports Streamlit.
8. **Table retrieval only.** The SQLite indexes hold table chunks only. Page chunks are not
   indexed, so questions that need prose are refused.

## 3. Components

New package `src/agent/`; each file has one job.

| File | Responsibility | LLM calls |
|---|---|---|
| `llm.py` | Structured-output LLM port (`Callable[[str, type[BaseModel]], BaseModel]`) and the OpenAI adapter. Provider errors are mapped to `LlmUnavailable`. | — |
| `metrics.py` | Map each subquery's `metric` / `metric_text` to a tree node id. | 0–1 per question |
| `gather.py` | Turn needs into `LeafRequest`s, call `LeafExtractor`, build a `FactSet` plus a per-leaf outcome log. | locator only |
| `solver.py` | LLM writes `solution()` and claims an answer; sandbox, gates, repairs. | 1–4 |
| `answer.py` | Vietnamese answer or refusal text from templates. | 0 |
| `agent.py` | Orchestration, the bounded loop, events, caps → `AgentRun`. | — |
| `present.py` | Pure functions: `AgentRun` / `StepEvent` → display rows for the UI. | — |
| `app/demo.py` | Streamlit page (thin). | — |

Dependency added: `streamlit` in `requirements.txt`. Installing it needs the user's approval.

## 4. Contracts

```python
@dataclass(frozen=True)
class AgentConfig:
    retriever_mode: Literal["none", "bm25", "hybrid"] = "none"
    locator_enabled: bool = True
    max_repairs: int = 3          # code attempts = 1 + max_repairs
    max_gather_rounds: int = 2    # extra gather rounds after the first
    max_locator_calls: int = 6    # per question
    default_scope: Scope = "consolidated"

@dataclass(frozen=True)
class StepEvent:
    kind: Literal["plan", "metrics", "gather", "solve", "answer"]
    title: str                    # Vietnamese, shown as the block header
    data: Mapping[str, object]    # JSON-like; present.py knows each kind's shape

@dataclass(frozen=True)
class AgentRun:
    status: Literal["answered", "clarify", "refused", "error"]
    text: str                     # answer, clarification request, refusal or error message
    reason: str | None            # refusal / error code, None when answered
    answer: Answer | None
    sources: tuple[FactSource, ...]   # facts used by the accepted program
    events: tuple[StepEvent, ...]
    llm_calls: int
    seconds: float

class ReasoningAgent:
    def __init__(self, paraphraser, tree, templates, extractor_factory, llm, config): ...
    def run(self, question: str, qa_history: Sequence[QATurn] = (),
            on_event: Callable[[StepEvent], None] | None = None) -> AgentRun: ...
```

`extractor_factory(locator) -> LeafExtractor` lets the agent wrap the locator in a per-question
counter. The store and the optional retriever are bound by the caller.

### 4.1 Metric mapping (`metrics.py`)

For each subquery, in order:
1. If `metric` is set (catalog id), use a fixed map to tree ids:
   - `revenue` and `net_revenue` → `net_revenue`
   - `profit_before_tax` → `profit_before_tax`
   - `profit_after_tax` → `profit_after_tax`
   - `consolidated_profit_after_tax` → `profit_after_tax`
   - `profit_after_tax_parent_share` → `profit_after_tax_parent_share`

   All targets exist in the tree at `c34969a`. A test pins the map against the tree.
2. If `metric_text` is set, `tree.find_by_alias(metric_text, template)`. Exactly one id → use it.
3. Otherwise, one LLM call for all remaining subqueries together. The prompt lists the tree's
   node ids with Vietnamese labels for the template. The output schema is
   `{subquery_id: node_id | {"proposal": ...} | null}`. A proposal is accepted only if
   `validate_proposal` passes, and it is then added with `tree.with_derived`.
4. A subquery that is still unmapped → refusal `metric_unmapped`.

The event `metrics` records, per subquery, the id and how it was found: `catalog`, `alias`,
`llm` or `proposal`.

### 4.2 Gather (`gather.py`)

- A **need** is `(node_id, ticker, year, scope)`, one per mapped subquery. The scope comes from
  `subquery.scope`, else `config.default_scope`.
- The template comes from `TickerTemplates`. An unknown ticker or unsupported template → that
  need fails with the extractor's own outcome.
- The leaves are the union of all sets from `tree.required_leaves(node_id, template, year)`.
  This over-fetches, matching how the core evaluates alternatives.
- There is one `extract(ticker, scope, requests)` call per `(ticker, scope)`. `Found` facts go
  into the `FactSet`. Every other outcome is logged as `(leaf, ticker, year, scope, outcome type,
  reason)`.
- **Locator cap.** The locator passed to the extractor is wrapped. After `max_locator_calls`
  calls in this question, the wrapper returns "skipped" (the extractor's `NotFound` stands). The
  log marks those leaves `locator_skipped`.
- A gather round may add needs. A leaf already requested in this question is never requested
  again.

### 4.3 Solve (`solver.py`)

- **LLM output schema:** `SolutionDraft(code: str, claimed_value: str, claimed_unit: str,
  needs: list[DataNeed])`.
- **Prompt:**
  - the question;
  - the plan's subqueries with their mapped ids;
  - the helper docstrings (calculation core §9);
  - the generated-code constraints (calculation core §16);
  - the inventory of available facts (`metric_id, ticker, year, scope, value`);
  - on a repair, the previous code and the problems found.
- **Context.** `ExecutionContext` is built with:
  - `default_scope`;
  - `ticker_templates`;
  - `question_literals(question)`;
  - `requested_unit(question)` and `requested_answer_kind(question)`;
  - `named_metrics` = the mapped ids.
- **Each attempt:**
  1. `sandbox.run(code, tree, facts, context)`.
  2. Then `run_gates(code, result, tree, facts, context, claimed=Answer(...))`.
  3. Problems for the repair prompt:
     - `ExecutionFailure` (its rejections or detail);
     - `GateReport` issues of severity error.
- **Declared data needs (amended 2026-11-13 while planning).** The sandbox reports only the
  exception class name: `ExecutionFailure("runtime_error", "MissingFact")` or
  `"MetricUnavailable"` (calculation core §10). The missing key therefore cannot be read from a
  failure. Instead:
  - `SolutionDraft` also carries `needs: list[DataNeed]`, with
    `DataNeed(node_id, ticker, year, scope | None)`: every `fact()` / `metric()` /
    `avg_balance()` the code will call.
  - **Before running a draft**, the agent compares its needs with the needs gathered so far.
    - New needs whose node ids exist in the tree → one gather round (at most
      `max_gather_rounds` per question). This does not consume a repair. `avg_balance` needs also
      fetch the prior year.
    - A need with an unknown node id → a problem for repair, without running the code.
  - **After a draft fails with `MissingFact` or `MetricUnavailable`:**
    - If any declared need failed in gather (logged outcome other than `Found`) → refusal
      `facts_missing`, citing those outcomes.
    - Otherwise the code used data it did not declare → a problem for repair: "code used data not
      declared in needs".
- **Exhausted repairs.** After `1 + max_repairs` attempts without a passing report → refusal
  `gates_failed`, with the last problems.

### 4.4 Answer (`answer.py`)

- **Answered.** A Vietnamese sentence from the `Answer`:
  - the value in the requested unit through the core's display functions;
  - `-0.00` normalised;
  - the original question's subject (from the plan);
  - then "Nguồn:" with one line per source fact: report `doc_id`, page and row label.
- **Refused and clarify.** One fixed sentence per code: `plan_invalid`, `metric_unmapped`,
  `facts_missing`, `gates_failed`, `clarify`. Details are appended as bullets, for example
  "LNST VJC 2022 (hợp nhất): Ambiguous — 2 bảng cho 2 giá trị".
- **Error.** "Không gọi được mô hình ngôn ngữ, vui lòng thử lại" plus the error class.

## 5. Flow and statuses

1. **plan:** `paraphraser.paraphrase(question, qa_history)`.
   - `needs_clarification` → status `clarify`.
   - `invalid` → one repair call `paraphrase(question, qa_history, feedback=[...])` (issue messages,
     prefixed "q1: " when the issue has a subquery id);
     its result is handled by these same rules, and a second `invalid` → `refused/plan_invalid`.
     Only the final result emits the plan event.
   - `unavailable` → `error`.
2. **metrics:** §4.1.
3. **gather:** §4.2.
4. **solve:** §4.3. It may loop back to gather, at most `max_gather_rounds` times.
5. **answer:** §4.4.

`LlmUnavailable` at any step ends the run with status `error`; no answer is guessed. Every step
emits one `StepEvent`, and solve emits one per attempt. `on_event` is called synchronously, so
the UI can render progressively.

**Per-question cap.** At most 2 paraphrase (1 + 1 repair) + 1 mapping + 6 locator + 4 code calls = 13 LLM calls
(the semantic reviewer is off). The usual cost is 2–3.

## 6. Web demo (`app/demo.py`)

- **Sidebar:**
  - retriever mode `none` (default) / `bm25` / `hybrid`;
  - locator on/off;
  - max repairs;
  - "Hội thoại mới" (clears the history).
- **Main area:** chat (`st.chat_message`). Each answer is followed by one expander per event,
  rendered from `present.py` rows:
  - Plan: the subquery table and the operation;
  - Metrics: the mapping table;
  - Gather: the leaf table with value, source and tool; failures highlighted;
  - Solve: the code and the gate issues per attempt;
  - Answer: the text and the sources;
  - a footer with the LLM calls and seconds.
- **History:** kept in `st.session_state` as `QATurn`s, appending the answered or clarified
  turn. It is not persisted (SP4).
- **Startup (`st.cache_resource`):**
  - `SqliteReportStore` (read-only corpus);
  - `MetricTree.load()`, `load_ticker_templates()`;
  - `EntityRegistry` from `data/code_stock.csv`;
  - the OpenAI adapter. A missing `OPENAI_API_KEY` shows an error banner, not a crash.
  - The table retriever opens only when a mode other than `none` is selected (`hybrid` loads
    BGE-M3 on the GPU).
  - Paths come from the environment variables `FINAGENT_DB`, `FINAGENT_BM25_DIR` and
    `FINAGENT_CHROMA_DIR`, with defaults under `data/`.
- **Run:** `streamlit run app/demo.py`.

## 7. Testing

Unit tests are offline and deterministic, with no network, GPU or real LLM. The LLM fake
returns scripted pydantic objects in order and fails the test on an unexpected call. The rest is
real:
- the tree, sandbox and gates;
- the paraphraser with its real validator, using the scripted LLM;
- the extractor on the in-memory store fixtures of the SP2 tests.

| Module | Tests |
|---|---|
| `metrics.py` | catalog map; alias hit with no LLM call; LLM pick; accepted proposal; rejected proposal → `metric_unmapped` |
| `gather.py` | Found → FactSet; each other outcome logged with its reason; the locator cap gives `locator_skipped`; no leaf is requested twice |
| `solver.py` | bad code then good; gate errors on every attempt → `gates_failed`; a draft with a new declared need → one gather round, then success; MissingFact with a failed declared need → `facts_missing`; MissingFact with all declared needs found → repair with "data not declared"; an unknown node id in needs → repair without running |
| `answer.py` | number display per unit, sources, each refusal text |
| `agent.py` | end-to-end with fakes: event order, `answered` / `clarify` / `refused` / `error`, `qa_history` forwarded, `llm_calls` counted |
| `present.py` | rows for each event kind |

Verification commands:
- `python -m unittest discover -s tests`
- `python -m compileall -q src config scripts app`
- an import-isolation check: `src.agent` loads neither streamlit, torch nor chromadb.

## 8. Done criteria

1. The tests, compileall and the isolation check pass.
2. **Smoke run with the real model** (costs money, so it runs only on the user's go-ahead). Run
   about 6 gold questions:
   - one direct lookup;
   - one calculation;
   - one multi-year comparison;
   - one multi-company comparison;
   - one conditional filter;
   - one follow-up turn.

   Record the outcomes, LLM calls and seconds in a shared note.
3. The user opens the page and asks questions.

## 9. Out of scope

- Batch evaluation on the 100 gold questions and accuracy targets (next spec).
- A free ReAct-style tool loop.
- Prose evidence (page chunks are not indexed).
- An HTTP API, persisted sessions and authentication (SP4).
- Changes to the paraphraser, calculation core, extractor or retriever, beyond what §4
  explicitly calls.

## 10. Risks and open items

- The paraphraser's catalog and the metric tree are separate. The fixed map in §4.1 is pinned by a
  test and must change with either side.
- **Declared needs can be incomplete.** The LLM may call data it did not declare. That case
  costs a repair attempt and is reported as such. The sandbox deliberately hides exception
  messages, so the agent cannot tell which key was missing.
- **Many-company questions.** 40 gold questions use 9 or more operands. With the locator capped
  at 6, some refusals will be `facts_missing` where more locator calls might have helped. This
  is measured in the next spec.
- **Retriever modes are off by default.** The first retrieval gate found new wrong values in
  both modes (retrieval spec §10). `bm25` / `hybrid` are offered for demonstration only.
- **No memory cap for the sandbox on Windows** (calculation core §16).

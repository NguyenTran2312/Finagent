# Advanced RAG baseline

Advanced RAG is the non-agentic baseline for the Agentic RAG experiment on ViFinQA. Both systems run
the same `ReasoningAgent` pipeline (`src/agent/agent.py`) with the same models, indexes, prompts,
retrieval parameters, extraction, calculation sandbox, verification gates and answer formatting.
They differ only in three configuration fields that turn adaptation to intermediate results on or off.

## Pipeline (shared)

| Stage | Component |
|---|---|
| Query reformulation and decomposition | `QueryParaphraser` (`src/query_paraphrase/`) — one LLM call, deterministic validation |
| Financial metric mapping | `map_metrics` (`src/agent/metrics.py`) — catalog, alias, then at most one LLM call |
| Metadata filtering | `HybridTableRetriever` candidates filtered by ticker, scope, report year |
| Hybrid retrieval, RRF, cross-encoder rerank | `retriever_mode="hybrid_rerank"` (`src/retrieval/table_retriever.py`) |
| Financial data extraction | `Gatherer` → `LeafExtractor` (retrieved tables first, full report on miss) |
| LLM cell localization | `LlmCellLocator` (one call per leaf, verified, capped at 6 per question) |
| Numerical reasoning | solver LLM writes a `solution()` program (`src/agent/solver.py`) |
| Deterministic calculation | AST-whitelisted sandbox (`src/calculation/sandbox.py`), exact units |
| Verification gates | `run_gates` (`src/calculation/gates.py`) |
| Answer and citations | `answer_text` (`src/agent/answer.py`), code-built Vietnamese text |

## The experimental variable

`src/agent/experiment.py` defines both configurations. Every `AgentConfig` field except these three
is identical (a test enforces it):

| Field | Advanced RAG | Agentic RAG | Mechanism |
|---|---|---|---|
| `max_plan_repairs` | 0 | 1 | re-plan with the validator's problems as feedback |
| `max_gather_rounds` | 0 | 2 | gather the data a solution draft declares |
| `max_repairs` | 0 | 3 | regenerate the solution with sandbox/gate problems as feedback |

Shared: `retriever_mode="hybrid_rerank"`, `locator_enabled=True`, `max_locator_calls=6`,
`default_scope="consolidated"`.

With all three at 0, Advanced RAG:

1. plans once; an invalid plan is refused (`plan_invalid`), a plan needing clarification returns
   `clarify`;
2. gathers data once, from the plan: for each subquery, every leaf of every formula alternative of
   its mapped metric, including prior-year balances (`owners_equity[t-1]` in ROE, ROA, asset
   turnover, inventory days) via `MetricTree.required_leaves`;
3. asks for one solution program and runs it once in the sandbox and through the gates. A failure
   is final: `gates_failed` (sandbox or gate problems kept in the refusal), `facts_missing` (a
   declared leaf was not found), or `data_not_gathered` (the program needs data outside the
   initial collection, such as a year or metric not in the plan).

Infrastructure retries stay the same for both systems: the OpenAI client is built with
`max_retries=1`, and the cell locator, mapping and paraphrase steps make no semantic retries.

## Running

The script needs `OPENAI_API_KEY`. Real LLM calls cost money.

```powershell
python -X utf8 scripts/run_experiment.py --system advanced_rag --questions data/questions/questions.jsonl --out outputs/advanced_rag.jsonl --db D:/finance_project/data/databases/corpus.sqlite3 --bm25-dir D:/finance_project/data/indexes/bm25 --chroma-dir D:/finance_project/data/chroma/<index-sha>
```

Use `--system agentic_rag` for the comparison run. Useful options:

- `--ids 1,2,3` runs only those questions; `--limit N` runs at most N.
- `--resume` continues an interrupted run.
- `--model NAME` sets the model (default `gpt-6-luna`).
- `--input-price-per-mtok USD --output-price-per-mtok USD` adds a cost estimate.

`--db`, `--bm25-dir` and `--chroma-dir` default to `FINAGENT_DB`, `FINAGENT_BM25_DIR` and
`FINAGENT_CHROMA_DIR`.

The script reads only `id`, `question` and optional `qa_history` from each input line. Other
fields, gold answers included, never reach the system.

From Python:

```python
from src.agent.experiment import open_runner
runner = open_runner("advanced_rag", db=..., bm25_dir=..., chroma_dir=...)
record = runner.run(question_id=1, question="...")
```

## Output record (one JSON line per question)

| Field | Content |
|---|---|
| `status`, `reason` | `answered` / `refused` / `clarify` / `error`; reason code (`plan_invalid`, `metric_unmapped`, `facts_missing`, `gates_failed`, `data_not_gathered`, `clarify`, `llm_unavailable`, `exception:<Class>`) |
| `text` | final Vietnamese answer or refusal, with `Nguồn:` citations |
| `answer` | kind, exact value (string), unit, display string |
| `sources` | facts used by the accepted program: chunk id, doc id, page, row label |
| `evidence` | every leaf outcome gathered (Found/NotFound/...), with value, chunk, page, tool |
| `calculation` | last solution program and the number of attempts |
| `verification` | `passed` and per-attempt outcome and problems |
| `telemetry` | `seconds`, `llm_calls`, `llm_calls_by_stage` (QueryPlan, MetricChoices, PointerAnswer, SolutionDraft), `retrieval_calls` (retriever `rank` calls), `solver_gather_rounds`, `tokens` |

`tokens` holds provider-reported usage, collected by a LangChain callback. It is `null` when no
response reported usage. `complete: false` means some responses lacked usage or a call raised;
the totals are then a lower bound and no cost is estimated. There is no built-in price list: a
cost appears only when you pass prices.

## Fairness assessment and known differences

Shared by both systems (by construction): the LLM client and model, embedding model, reranker,
corpus and indexes (one shared retriever instance), the retrieval parameters per call (`k`,
`depth`), metric tree, extraction, locator and its cap, sandbox, gates, prompts, answer templates
and refusal policy. Until the first intermediate result that Agentic RAG adapts to (an invalid
plan, a draft with new needs, a failed draft), both systems send byte-identical prompts. A test
checks this for a lookup question.

Known differences and limitations:

- The solver prompt says "Data not listed in needs is not fetched". In Advanced RAG, needs beyond
  the initial collection are never fetched. The prompt was left identical on purpose, so that
  first drafts are comparable. A draft that relies on fetching is refused as `data_not_gathered`.
- Questions whose operands the plan does not name are refused by Advanced RAG. An example is an
  average balance over a leaf metric through `avg_balance()` that no metric formula derives. This
  is the expected effect of removing dynamic gathering.
- The `gates_failed` refusal text says "after several attempts" in both systems, including
  Advanced RAG's single attempt. The shared template was not forked.
- `data_not_gathered` is a new reason code. It also applies to Agentic RAG when its gather rounds
  are exhausted, a case that was previously reported as `gates_failed`. The repair feedback the
  LLM sees is unchanged.
- No temperature or seed is set. Both systems use the provider default, so repeated runs may
  differ.
- Latency includes warm caches: the retriever's per-report candidate cache and the Chroma snapshot
  are shared within a process. Run each system in its own process, in the same order, for
  comparable timings.
- Branch base: `feat/reasoning-agent` does not contain commit `0463a84` from
  `feat/retrieval-extraction`, which always reads statement tables of the leaf's kind in step R.
  Both systems share this extraction behaviour, so the comparison is unaffected. Absolute scores
  may differ from a build that includes it.

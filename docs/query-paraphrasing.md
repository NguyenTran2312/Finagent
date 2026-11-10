# Query paraphrasing and grouped evidence retrieval

Pipeline: `question (+ qa_history)` -> `QueryParaphraser` (one LLM call, deterministic
validation) -> `ParaphraseResult` -> `retrieve_subqueries` (existing hybrid + reranker, one
explicit-filter call per subquery) -> `PipelineResult` with evidence grouped per subquery.

Modules: `src/query_paraphrase/` (contracts, prompt, validator, service),
`src/retrieval/multi_query.py` (grouped retrieval), `src/main/query_pipeline.py` (entrypoint).
`refine_question` is never called in this flow; filters are explicit `metadata_filters`.

## Usage

The LLM is injected; this package never picks a provider, model, key or timeout.

```python
from src.query_paraphrase import QueryParaphraser
from src.main.query_pipeline import build_query_pipeline

def llm(prompt: str):            # returns JSON text, a mapping or a QueryPlan-like model
    return my_client_call(prompt)

paraphraser = QueryParaphraser(llm)
pipeline = build_query_pipeline(paraphraser, reranked_retriever)   # hybrid + reranker built by you
result = pipeline.invoke("So sanh doanh thu VJC nam 2022 va 2023")
result = pipeline.invoke({"question": "Con nam 2023 thi sao?",
                          "qa_history": [{"question": "Doanh thu VJC 2022?", "answer": "..."}]})
```

With a LangChain chat model, use `structured_llm_adapter`. The caller creates and configures
the client (provider, model, timeouts, retries):

```python
from src.query_paraphrase import QueryParaphraser, structured_llm_adapter

client = make_chat_model(model="...", timeout=30)      # caller's choice
paraphraser = QueryParaphraser(structured_llm_adapter(client))
```

The adapter calls `client.with_structured_output(QueryPlan, include_raw=True)` once and
invokes it per prompt. It passes on `parsed` only when `parsing_error` is None and `parsed`
is set; any parse failure becomes `invalid` (`unsupported_output`) instead of raising, and
the raw model text never reaches the result. The parsed object is still strictly
re-validated.

`reranked_retriever` is `build_reranked_retriever(build_hybrid_retriever(bm25, dense_store,
k=..., candidate_k=...), cross_encoder, k=...)`. Loading BM25, embeddings, Chroma and the
cross encoder is the caller's job; the pipeline only calls `retriever.invoke(mapping)`.

### Semantic reviewer (optional)

`QueryParaphraser(llm, semantic_reviewer=fn)`. It runs only for a plan that already passed
validation and is called with keyword arguments `question`, `qa_history` and `plan`.
`qa_history` is a list of `(original_index, QATurn)` pairs for the turns retained after
truncation; `original_index` is the index that `context_refs` use. It must return a
`SemanticReview(status="passed"|"failed"|"unknown", reason=...)` (or an equivalent mapping).
`failed` -> `invalid`; `unknown` or a malformed review -> `needs_clarification`.
The reviewer receives a deep copy of the plan; mutating it never changes the plan used for
retrieval.
Note the asymmetry: `validate_plan` receives the raw history, the reviewer receives the
prepared pairs.

### Supported entity and follow-up forms

- Entities: a ticker (`VJC`), the full company name exactly as in `data/code_stock.csv`, or
  aliases injected through `EntityRegistry.from_csv(path, aliases={...})`. Matching is
  accent/case-folded and exact; there is no fuzzy matching. Unknown or ambiguous entities
  give `needs_clarification`.
- Follow-ups: a leading "con ...", "... thi sao", "cong ty/doanh nghiep/chi tieu do|nay",
  "nam do". A metric is inherited only after "chi tieu do/nay" (or a bare follow-up that names
  nothing else). Inherited fields must be backed by `context_refs` pointing at history
  indexes; other implicit references are not inherited and ask for clarification.
- A follow-up subquery that leaves `ticker` or `year` null while the current question
  states none and a retained history question states one gives `needs_clarification`
  (`unresolved_follow_up`): the value must be stated or inherited through `context_refs`,
  never dropped (which would widen retrieval to every company or year).
- A ticker written as a word in another case ("vjc", "gas") supports a plan that picks
  that ticker, but when no subquery has a ticker it gives `needs_clarification`
  (`ambiguous_entity_mention`).
- Document types: `financial_statements` is always allowed. `explanations` requires
  "thuyet minh" and `letters` (explanatory letters) requires "thu giai trinh" in the
  accent-folded question; a subquery requesting either without its cue is `invalid`
  (`invented_document_type`), and a cue that no subquery requests is `invalid`
  (`missing_document_type`).

## Configuration

Read by `load_paraphrase_settings()` (defaults < environment < overrides) when the objects
are built, not per request: `QueryParaphraser` resolves `settings=None` at construction and
`build_query_pipeline` resolves `policy=None` once when the pipeline is built; a direct
`retrieve_subqueries(..., policy=None)` call resolves it for that call. Later environment
changes do not affect an existing paraphraser or pipeline.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FINAGENT_PARAPHRASE_MAX_SUBQUERIES` | 8 | Maximum subqueries in a plan |
| `FINAGENT_PARAPHRASE_MAX_HISTORY_TURNS` | 5 | Most recent history turns kept |
| `FINAGENT_PARAPHRASE_MAX_INPUT_CHARS` | 8000 | Maximum characters per text input |
| `FINAGENT_PARAPHRASE_DEFAULT_REPORT_SCOPE` | consolidated | `consolidated` or `separate` |
| `FINAGENT_PARAPHRASE_NOTES_FALLBACK` | true | `true/false/1/0`; explanations fallback |

Invalid values raise `ValueError` rather than being corrected.

## Statuses and safe failures

`ParaphraseResult.status`:

- `valid`: plan passed deterministic validation (and the reviewer, when configured).
- `needs_clarification`: ambiguous or unresolved input (unknown entity, unresolved follow-up
  including a dropped history ticker/year (`unresolved_follow_up`), lowercase ticker word
  with no ticker chosen (`ambiguous_entity_mention`), ambiguous binding, unknown semantic
  review). No retrieval happens.
- `invalid`: malformed or inconsistent model output (including adapter parse failures and
  unsupported document types: `invented_document_type`, `missing_document_type`), or a
  failed semantic review.
- `unavailable`: LLM or reviewer raised one of the configured provider errors
  (default `TimeoutError`, `ConnectionError`); only the exception class name is kept.

There is no retry, repair or fallback. Bad input (`ValueError`) raises before any LLM call.
Any non-valid result gives `retrieval_status="not_run"` and zero retriever calls.

`PipelineResult(paraphrase, evidence_groups, retrieval_status)`; relationships remain on
`paraphrase.plan.relationships`. `retrieval_status`: `not_run` (paraphrase not valid), `ok`
(all groups ok), `error` (all groups error), `empty` (all groups empty), otherwise `partial`.

Retrieval failures: exceptions that are instances of `retrieval_errors` (default
`TimeoutError`, `ConnectionError`, `OSError`) are recorded per pass as the class name only
(never `str(exc)`), and other groups continue. Any other exception (`ValueError`,
`KeyError`, `TypeError`, ...) propagates as a programming/contract error. Transport errors
that do not subclass the builtin defaults (for example from `httpx`, `chromadb` or `redis`)
must be added to `retrieval_errors` (retriever) or `provider_errors` (LLM/reviewer);
otherwise they abort the whole request.

## Retrieval policy and passes

`RetrievalPolicy(default_report_scope="consolidated", notes_fallback=True)`; `None` resolves
from `load_paraphrase_settings()`. Per subquery, sequentially and in plan order, every call
is exactly `retriever.invoke({"question": search_query, "metadata_filters": {...}})` with
`tickers=[ticker]` and `years=[year]` (empty list when absent), identical in every pass:

1. Primary (financial statements): runs when `document_types` is empty or contains
   `financial_statements`; scope is the subquery scope or the policy default.
2. Explanations: scope `explanations`; runs when requested. When `document_types` is empty,
   `notes_fallback` is true and primary returned no documents without error, it runs once as
   a fallback. Never after a primary error.
3. Letters: scope `letters`; only when requested.

All filters for all subqueries are built and validated before the first retriever call.
Filters are never widened. Documents are returned as the retriever produced them; nothing is
merged, deduplicated or re-ranked across subqueries. Notes and letters appear only in
`EvidenceGroup.supplemental` (`scope_verified=False`, `report_scope=None`), never in
`primary_evidence`.

Group status: `error` when every executed pass errored; `partial` when evidence exists
together with a pass error, when only fallback notes were found, when notes/letters are
requested with an explicit consolidated/separate scope, or when some passes errored and none
returned evidence; `empty` with no evidence and no error; otherwise `ok`.
`requested_scope` is the subquery scope (None when the policy supplied it);
`effective_scope`/`scope_source` (`requested` or `policy_default`) describe the primary pass.
Requested period/unit set `period_verified`/`unit_verified` to `False` with a warning but do
not downgrade the status.

## Limitations

- Period (quarter, half-year) and unit are carried and reported but not verified or filtered
  against corpus metadata.
- The scope of explanations and letters cannot be verified; an explicit report scope
  combined with them yields `partial`.
- A ticker present in the CSV registry but absent from the corpus yields an empty group,
  not an error.
- The validator is strict and asks for clarification in doubtful cases, e.g. a single year
  placed between two companies.
- Metrics outside the catalog are not recognized: a question about such a metric may keep
  `metric` null (only `metric_text` is checked when the model sets it).
- Trade names absent from `data/code_stock.csv` and from injected aliases (e.g. "Vinamilk")
  are not recognized; a plan that leaves `ticker` null for them retrieves without a ticker
  restriction. Inject aliases to cover them.
- History fields are validated (shape, length) for every supplied turn before the history is
  trimmed to `max_history_turns`, so an invalid old turn raises `ValueError`.
- Worst case is sequential: `max_subqueries` x up to 3 hybrid + rerank calls per plan
  (default 8 x 3 = 24), with no parallelism or caching.
- Not tested: any live LLM provider, strict structured-output compatibility of the
  `QueryPlan` schema with a given provider (only that the JSON schema builds), the real
  PostgreSQL corpus, ChromaDB index, embedding model or GPU reranker. Unit and integration
  tests use in-memory BM25 with fake dense store, encoder and LLM.

# Retrieval-Guided Leaf Extraction — Design

Date: 2026-11-12
Status: draft for review
Sub-project: 2 of 4 (leaf extraction), amendment
Builds on: `docs/superpowers/specs/2026-11-11-leaf-extraction-design.md` (branch
`feat/leaf-extraction` @ dec9de9) and the SQLite index builder (branch `feat/sqlite-indexing` @
6cd4e61). Work happens on `feat/retrieval-extraction` (cut from dec9de9, with 6cd4e61 merged).
Replaces decision 7 of the leaf-extraction design ("Report-direct evidence, approach A").

## 1. Context

Leaf extraction today reads every table of the requested report and matches rows by rule
(leaf design §5, §6). The project's stated architecture is retrieval-augmented: a retriever
selects evidence that later steps read. This amendment puts the existing table retriever
(BM25 + BGE-M3 dense, reciprocal rank fusion) in front of the rule-based reader: for each leaf
request, the retriever ranks the report's tables and the extractor reads only the top k,
falling back to the whole report when those tables do not contain the row.

Measured before this design (2026-11-12; shared note `retrieval-recall-gold-2026-11-12.md`):

- Requests: the 638 gold operand requests (ticker, scope, leaf, year) that leaf extraction at
  dec9de9 answered correctly. Truth = the table chunk the extractor read the value from.
- Filters ticker / report_year / scope leave a median of 74 candidate tables (min 31, max 196);
  a random ranking gives R@5 = 6.9 %.
- Query wording matters. "V1" (`label_vi` only) and "V3" (`label_vi` + Vietnamese statement
  name, §5.1):

| Query | Retriever | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| V1 | BM25 | 44.7 | 93.4 | 98.0 | 99.7 |
| V1 | dense | 8.2 | 40.3 | 59.9 | 77.3 |
| V1 | hybrid | 27.1 | 70.5 | 86.8 | 95.8 |
| V3 | BM25 | 42.8 | 93.6 | 99.7 | 100 |
| V3 | dense | 29.6 | 75.1 | 88.9 | 96.7 |
| **V3** | **hybrid** | **53.4** | **96.2** | **99.2** | **99.8** |

- With V1, dense ranks income-statement tables poorly (R@5 26.9 %) and pulls hybrid below BM25;
  V3 fixes this. V3 hybrid misses at k = 5 concentrate in `profit_after_tax` (6/45),
  `interest_expense` (5/16) and `profit_before_tax` (4/22), rows that recur in many notes.
- Caveats: only requests the extractor already got right are measured; exact-chunk truth is a
  lower bound (another table may hold the same value); 9 of the 638 values came from the
  next-year report, which the real flow retrieves with its own year filter (§4.2).

Indexes available on the local machine (built by `src/main/build_indexes.py`, 6cd4e61):

| Index | Path | Content |
|---|---|---|
| BM25 | `data/indexes/bm25/bm25-table.pkl` + `.json` manifest | 146,246 table chunks |
| Chroma | `data/chroma/7121d2b324f27b82bfcee6d324e7e60d41779b72602cb74f745f5dee1fec8297/` | 146,246 BGE-M3 fp16 vectors, collection `vifinqa_tables` |

Page chunks are not indexed and are out of scope (§9).

## 2. Agreed decisions

1. **The retriever filters the tables the extractor reads** (top k), with a full-report scan
   when they yield no row. It is enabled by default only after the evaluation gate (§7) shows
   no new wrong value.
2. **Hybrid BM25 + dense (RRF) by default; mode `bm25` selectable** for machines without a GPU
   and for tests. The gate evaluates both.
3. **Injected, optional dependency.** `LeafExtractor(..., retriever=None)`; `None` keeps today's
   behaviour exactly, so existing tests, scripts and callers are unchanged.
4. **No retrieval algorithm changes.** The adapter reuses `retrieve_refined`, Chroma
   `similarity_search` and `rrf_fuse` as they are; no cross-encoder rerank.
5. **k = 10** tables per (request, source), from the measured R@10 of 99.2 % (hybrid V3).

Rejected:

- **Cross-check** (always also scan the whole report; on disagreement keep the full-scan
  outcome): never worse than today, but the retriever then changes no result.
- **Provenance only** (scan as today, record the retrieved rank of the source table): safe, but
  the retriever stays outside the decision path.
- **Wrapper** `RetrievingExtractor` over a filtered `ReportStore`: duplicates the source-order,
  variant and fallback logic of `extractor.py` and splits provenance across two layers.
- **Retrieval inside `ReportStore`**: mixes SQL access with ranking and model loading.

The leaf design's original objections to per-leaf retrieval (§2 there) are addressed as
follows: there is no rerank pass (decision 4); a top-k miss falls back to the full report
(§4.2); index state is checked at start-up (§5.4) and recorded on every fact (§3.2).

## 3. Components and interfaces

### 3.1 Modules

| Module | Change |
|---|---|
| `src/extraction/retrieval.py` (new) | `TableRetriever` protocol, `RankedTable`, `RetrievalTrace`, `TableRetrievalError`. Standard library only. |
| `src/extraction/outcomes.py` | `ExtractedFact` gains `retrieval: RetrievalTrace \| None = None`. |
| `src/extraction/extractor.py` | `LeafExtractor(..., retriever=None)`; per-document R → F reading (§4). |
| `src/retrieval/table_retriever.py` (new) | `HybridTableRetriever` and `open_table_retriever(...)` (§5). Holds the torch / Chroma / BM25 dependencies. |
| `scripts/leaf_coverage.py` | `--retriever {none,bm25,hybrid}`, index-path options, extra CSV columns (§7.1). |
| `scripts/diff_extraction_runs.py` (new) | Compare two coverage CSVs (§7.1). |
| `scripts/score_gold_operands.py` (new) | Gold-operand scoring with `--retriever`, ported from the scratch script used for the 96.8 % figure (§7.1). |

`src/extraction` keeps its rule from the leaf design §3: no retrieval, embedding, GPU or network
import. It only defines the protocol; the adapter lives under `src/retrieval`.

Unchanged: `find_rows`, `TableView`, row matchers, unit and period rules, `ReportStore`,
`retrieve_refined`, `rrf_fuse`, `hybrid.py`, the index builder and both index formats.
(*Amended 2026-11-12:* `retrieve_refined`'s scoring loop moves into `rank_indices` in the same
module, behaviour unchanged, so the adapter can score cached candidates, §5.5.)

### 3.2 Types

```python
@dataclass(frozen=True)
class RankedTable:
    chunk_id: str          # "<doc_id>:table:<table_id>", as in TableRecord.chunk_id
    rank: int              # 1-based, unique within one answer

class TableRetriever(Protocol):
    mode: str              # "hybrid" | "bm25"; copied into RetrievalTrace
    def rank(self, ticker: str, scope: Scope, report_year: int,
             leaf: LeafDef) -> tuple[RankedTable, ...]: ...   # best first, ranks 1..n

class TableRetrievalError(RuntimeError):
    """Backend failure (index, model, device). The extractor degrades to a full scan."""

@dataclass(frozen=True)
class RetrievalTrace:
    mode: str                              # retriever.mode
    step: Literal["retrieved", "full_scan"]
    rank: int | None                       # rank of the source table; None for full_scan
```

`ExtractedFact.retrieval` is `None` when no retriever is injected. With a retriever, every
`Found` carries a trace: `retrieved` when the value came from a top-k table (step R or an LLM
pointer into a top-k table), `full_scan` when it came from step F.

## 4. Data flow (`extractor.py`)

### 4.1 Unchanged parts

Validation (§5.1 of the leaf design), templates (§5.2), source order and source combination
(§5.3), scope-variant combination (§5.4) and the result type (§5.7) do not change.

### 4.2 Reading one document of one source

For a request with leaf L and a source with report year Y (Y = y for the primary source, y + 1
for the next-year source):

1. **Retrieve once per source.** `ranked = retriever.rank(ticker, scope, Y, L)`, cached in the
   call object per `(Y, L.id)`; the result covers every scope-variant document of the source.
2. **Step R.** For each document D of the source: when `ranked` is non-empty, `views_R` = D's
   views whose `record.chunk_id` is in `ranked`, **plus** (§6.1, a statement leaf only) every view
   of D whose `record.statement_kind` equals `L.statement_kind`. The views keep D's order. When
   `ranked` is empty, `views_R` is empty. If `views_R` is non-empty, run today's `_in_document`
   with `views_R` instead of all views. A `Found`, `Ambiguous`, `BlankValue` or `UnitUnknown` is
   D's outcome. A `Found` gets `RetrievalTrace(mode, "retrieved", rank)`, where `rank` is the
   source table's rank, or `None` when only the §6.1 rule added that table.
3. **Step F.** If `views_R` is empty or step R gave `NotFound`, run today's `_in_document` with
   all of D's views. A `Found` gets `RetrievalTrace(mode, "full_scan", None)`.
4. Per-document outcomes are combined across variants and sources exactly as today.

Steps R and F run per document, so a variant document without a retrieved table is still read
in full and the variant rules (`not_in_variant`, `Conflict`) keep their meaning. The next-year
source is reached only when the primary source's combined outcome is `ReportMissing` or
`NotFound`, as today; it runs its own retrieval with `report_year = y + 1`.

### 4.3 LLM fallback

Runs under the same condition as today (final outcome `NotFound`, a document exists, a
`CellLocator` injected). For each document it receives that document's `views_R` when
non-empty, otherwise all of the document's views (today's behaviour). The locator's own shortlist (§8.2 of
the leaf design) then ranks within that set. A verified pointer gives `Found(method=
"llm_located")`. Its trace is `retrieved` when the pointed table is in that document's `views_R`,
with the table's rank (`None` when §6.1 added it), and `full_scan` otherwise.

### 4.4 Retriever answers and failures

- `TableRetrievalError` or `OSError` from `rank`: the source is read as if no retriever were
  injected (step F only) and the request's outcome gets warning
  `retrieval_error:<ExceptionClass>` through the existing `_with_warnings` (so on `Found` and
  `NotFound`; the other outcome types carry no warnings, as today). Other exceptions propagate
  (programming errors stay visible).
- Invalid answer (not a tuple of `RankedTable`, ranks not exactly 1..n, duplicate `chunk_id`):
  `ValueError` (programming error in the adapter). The extractor does not know `k`; the
  adapter enforces it (§5.2).
- A ranked `chunk_id` that matches no table of the source's documents is ignored; when any
  are ignored the outcome gets warning `retrieval_unknown_chunks:<count>` the same way (a stale
  index).
- An empty answer is valid: every document goes straight to step F.

## 5. Adapter (`src/retrieval/table_retriever.py`)

### 5.1 Query

`query = leaf.label_vi` plus, for `source: statement` leaves, the Vietnamese statement name:

| `statement_kind` | Appended text |
|---|---|
| `balance_sheet` | bảng cân đối kế toán |
| `income_statement` | báo cáo kết quả hoạt động kinh doanh |
| `cash_flow` | báo cáo lưu chuyển tiền tệ |

`source: notes` leaves, and any other kind, use `label_vi` alone. Aliases are not added (V2
measured no better than V1). The query is built from config only; no LLM, no user question.

### 5.2 Filters and ranking

- Filters: ticker (upper-cased), `report_year = Y`, scope; a `separate` request also accepts
  scope `parent`, as `hybrid.py` does. Only table chunks are indexed, so no chunk-type filter.
- `hybrid`: BM25 through `retrieve_refined(bm25, query, k=depth, metadata_filters=...)` and dense
  through `store.similarity_search(query, k=depth, filter=<Chroma where>)`, `depth = 20` each,
  fused by `rrf_fuse(..., k=k, c=RRF_CONSTANT)`; the first `k = 10` are returned.
  *Amended 2026-11-12:* both branches now search a cached per-report candidate set (§5.5);
  scoring and fusion are unchanged.
- `bm25`: the first `k` of the BM25 list.
- Ranks are positions in the returned list. `chunk_id` comes from document metadata.
- Backend exceptions (Chroma, torch, BM25 scoring) are re-raised as `TableRetrievalError` with
  the original as `__cause__`. Input errors (`ValueError` from `normalize_metadata_filters`)
  propagate unchanged.

### 5.3 Construction

```python
retriever = open_table_retriever(
    db_path="data/databases/corpus.sqlite3",     # for the BM25 identity check only
    bm25_dir="data/indexes/bm25", bm25_name="bm25-table",
    chroma_dir="data/chroma/7121d2b3…",          # the fingerprint directory; hybrid only
    mode="hybrid", device="cuda", fp16=True, k=10, depth=20,
)
extractor = LeafExtractor(store, tree, templates, retriever=retriever)
```

`HybridTableRetriever(bm25, dense_store, *, mode, k, depth)` is the plain constructor used by
tests; `open_table_retriever` loads the real indexes.

### 5.4 Start-up checks (fail fast, before any request)

1. `load_bm25(..., expected_identity=sqlite_source_identity(db_path, chunk_types=("table",)))`:
   the BM25 index must come from the current corpus file.
2. Hybrid only: open the Chroma directory directly (`chromadb.PersistentClient(path)`, langchain
   `Chroma(collection_name="vifinqa_tables", create_collection_if_not_exists=False)`), not
   through `load_or_build_dense_sqlite` (that reloads 146k chunks to compute the fingerprint,
   about 3 minutes). The collection count must equal the BM25 document count.
3. Hybrid only: embeddings via `get_embeddings(device=..., fp16=...)`; `fp16=True` is required to
   match how the index was built (the Chroma fingerprint records fp16).
4. `k >= 1`, `depth >= k`, `mode` in {`hybrid`, `bm25`}; otherwise `ValueError`.

Any failed check raises before the extractor is built; there is no silent fallback to
another mode or to no retriever.

*Amended 2026-11-12 (final review):* step 2 opens a snapshot copy of the Chroma directory
(`chroma_snapshot`, at `<chroma_dir parent>/loaded/<sha>`, copied once), never the source
directory: chromadb applies migrations and replays its log on open, which could rewrite the
index.

### 5.5 Per-report candidate cache (amended 2026-11-12, after gate measurements)

Measured on the real indexes (25 `rank` calls): Chroma `similarity_search` with the where-filter
1.73 s median (21 s max) against 0.02 s unfiltered; BM25 `retrieve_refined` 0.28 s (Python scan of
146k metadata); query embedding 0.13 s; Chroma `get(ids=...)` of one report's ~74 vectors 0.04 s.
A full-corpus hybrid run would take over 30 h. Chroma ids equal `chunk_id`.

- **Candidates** per key `(ticker, report_year, scope)`: `filter_document_indices(bm25.docs, filters)`
  once; the key's chunk_ids are those documents' `chunk_id`. Kept in an LRU of 32 keys (the
  extractor reads one report and its next-year report at a time).
- **BM25 branch:** scoring of the cached indices by `rank_indices(retriever, question, indices, k)`,
  extracted unchanged from `retrieve_refined` (same scores, ordering and zero-score drop);
  `retrieve_refined` calls it, so its behaviour is identical.
- **Dense branch:** the key's vectors and metadata are fetched once with
  `store.get(ids=chunk_ids, include=["embeddings", "metadatas"])`; the query vector comes from
  `store.embeddings.embed_query(query)`, cached per query text (queries repeat across reports).
  Scores are dot products of the normalised vectors (cosine, the index's space); the top `depth`
  by score form the dense list. This is an exact search over the filtered set where Chroma ran a
  filtered HNSW search; lists can differ only at near-ties.
- Fusion, `k`, `depth`, ranks and error wrapping (`get`, `embed_query`, BM25 scoring →
  `TableRetrievalError`) are as in §5.2.
- **Equivalence check before the gate:** for the gold requests, the new lists are compared with
  the old path (`retrieve_refined` + `similarity_search` + `rrf_fuse`); the share of identical
  top-k lists is reported and gold is re-scored with `hybrid`.

Decision 4 still holds for scoring: no new model, weights or fusion; only the search over the
filtered set is exact and cached.

## 6. Risks of filtering

| Risk | Mechanism | How it shows | Measured by |
|---|---|---|---|
| R1 hidden conflict | Two rows match with different values; the table holding the second is outside the top k. Today: `Ambiguous(multiple_rows)`. With R: a confident `Found`. Real case: DCM total assets, balance sheet 8,717 bn vs a subsidiary note's "Tổng tài sản" 50.7 bn. | `Ambiguous` → `Found`, or a changed value | §7 step 2 |
| R2 tier substitution | §6.2 of the leaf design reads tables tagged with the leaf's statement kind first and anchored `note`/`unknown` tables only when those give no row. If the top k holds no tagged table but an anchored one, step R reads the anchored table where today's scan would read the tagged one. | changed value or changed source table | §7 step 2 (source `chunk_id` compared too) |
| R3 stale index | The corpus changed after indexing. | start-up failure (§5.4) or `retrieval_unknown_chunks` | start-up check |
| R4 latency | BM25 filtering scans document metadata in Python (0.28 s per query measured on the gate); Chroma's filtered search 1.7 s median. Addressed by §5.5. | per-request time | §7 step 2 (p50 / p95 reported) |

### 6.1 Statement-table rule (adopted 2026-11-13 after the first gate)

The first gate (§10) found R1/R2 wrong values in both modes. Statement leaves therefore always add the document's views tagged with the leaf's statement kind to `views_R` (§4.2 step 2).

Step R's tier 1 (`find_rows`: tagged tables first) then sees the same tagged tables as a full scan. The retriever narrows only the anchored `note`/`unknown` tier and the tables of note leaves (`source == "notes"`), which the rule leaves unchanged.

The cost is that fewer `Ambiguous → Found` gains survive when the conflict lies between tagged tables. The rule does not help when a bank statement is tagged `note`. Rank histograms count rows with step `retrieved` and no rank as `unranked`.

## 7. Evaluation gate (before enabling by default)

### 7.1 Tooling

- `scripts/leaf_coverage.py --retriever {none,bm25,hybrid} [--bm25-dir --chroma-dir --device]`;
  `--out` CSV gains `value`, `chunk_id`, `retrieval_step`, `retrieval_rank`, `seconds`.
- `scripts/diff_extraction_runs.py BASE.csv NEW.csv` prints counts and writes a CSV of every
  request whose outcome type, value or source `chunk_id` differs, labelled `value_changed`,
  `ambiguous_to_found`, `found_to_other`, `other_to_found`, `source_changed`, `other`.
- `scripts/score_gold_operands.py GOLD_JSONL OUT_CSV [--retriever ...]`: the gold scoring used
  for the 96.8 % figure (operand → leaf alias table, VND normalisation, scale tolerance,
  `sign_only`, `mapping_uncertain`), with the gold `calculation_program` never executed.

All three are manual, read-only on the corpus, and outside the unit-test suite.

### 7.2 Steps

1. **Gold.** Score with `none`, `bm25` and `hybrid` on the same commit. Baseline at dec9de9:
   663 requests, 642 correct, 2 wrong. Report correct / wrong per mode, the share of `Found`
   via `retrieved` vs `full_scan`, and the rank distribution of retrieved sources.
2. **Full corpus.** `leaf_coverage.py` over all 53,480 requests with `none` and `hybrid`
   (2–3 h in the background), then `diff_extraction_runs.py`. Same for `bm25` if hybrid shows
   changes.
3. **Review.** Inspect every `value_changed`, `ambiguous_to_found` and `source_changed`
   request (a random sample of 30 per label when a label has more than 30), against the
   report table.
4. **Decision (by the user).** No new wrong value → `retriever` becomes the default in callers
   (sub-project 3 wiring and scripts). Any new wrong value → switch to cross-check or adopt the
   §6 mitigation, then re-run the gate.

Results go to a shared note and to §10 of this document.

## 8. Testing

unittest, offline, deterministic; no model, GPU or network. New files
`tests/test_extraction_retrieval.py` and `tests/test_table_retriever.py`.

Extractor with a fake `TableRetriever` (in-memory store fixtures from the leaf design tests):

- value in a top-k table → `Found`, trace `retrieved` with that table's rank;
- top k without the row → step F finds it → trace `full_scan`;
- top k holds a matching row with a blank / unparsable cell → that outcome, no step F;
- empty answer → step F only;
- `TableRetrievalError` → full scan, warning `retrieval_error:TableRetrievalError`;
  `KeyError` from the fake propagates;
- invalid answers (rank 0, rank gap, duplicate rank, duplicate chunk, a non-`RankedTable`
  item) → `ValueError`;
- unknown `chunk_id` → ignored, warning `retrieval_unknown_chunks:1`;
- R1 fixture: two tables with conflicting rows, only one ranked → `Found` from the ranked one
  (documents the accepted risk); the same fixture without a retriever → `Ambiguous`;
- next-year source: the fake records `report_year = y + 1` for the second call; cache: one
  `rank` call per (year, leaf) even for duplicate requests and two variant documents;
- scope variants: one variant has ranked tables, the other none → the other is read in full;
- LLM locator (fake) receives only the ranked views; with no ranked view it receives all;
- `retriever=None` → outcomes identical to today and `ExtractedFact.retrieval is None` (run
  over the existing extractor fixtures).

Adapter (`HybridTableRetriever` with a tiny in-memory BM25 built by `build_bm25` and a fake
dense store):

- V3 query text per statement kind; `notes` leaf uses `label_vi` only;
- filters: ticker upper-cased, `report_year`, `separate` → `{separate, parent}` in the Chroma
  where-clause and in the BM25 filter;
- `bm25` mode ranks and truncates to k; `hybrid` passes `depth` to both branches and fuses
  with `rrf_fuse`;
- backend exception in the fake store → `TableRetrievalError` with `__cause__`;
- `open_table_retriever` start-up failures with temporary files: BM25 manifest identity
  mismatch, Chroma count ≠ BM25 count (fake client), bad `k` / `depth` / `mode`.

Verification: `python -m unittest discover -s tests` and `python -m compileall -q src config
scripts`.

## 9. Out of scope

Page chunks and their dense index (and the known `rrf_fuse` deduplication by
`(doc_id, table_id)`, which would merge all page chunks of a document); cross-encoder rerank;
wiring into `query_pipeline` / the reasoning agent (sub-project 3); changes to retrieval
scoring or index formats; re-indexing.

## 10. Open items

- **Gate results (§7), 2026-11-13, at 60d4239. Verdict: do not enable by default.** Both
  `bm25` and `hybrid` add new wrong values (step 4). The user chose to record the results only.
  The §6 mitigation and cross-check stay open.
  - **Gold** (663 requests):
    - `none`: 642 correct / 2 wrong.
    - `bm25`: 642 / 2, no outcome change.
    - `hybrid`: 641 / 2. QNS admin_expenses 2023 went Found → `Ambiguous(multiple_period_columns)`.
  - **Speed** (§5.5): gold hybrid time fell from 1,662 s to 271 s. 90 % of top-10 lists are identical to Chroma's filtered HNSW and the mean overlap is 99.1 %. Outcomes are unchanged.
  - **Full corpus** (53,480 requests), compared with `none` (2,558 s):

    | | `bm25` | `hybrid` |
    |---|---|---|
    | Run time | 2,231 s | 2,318 s |
    | Retrieved / full scan | 43,121 / 1,547 | 42,848 / 1,828 |
    | Requests changed | 51 | 82 |
    | `ambiguous_to_found` | 31: all checked, 28 correct, **3 wrong** | 45: sample of 30, 27 correct, **3 wrong** |
    | `value_changed` | 0 | 1, **wrong** (base correct) |
    | `source_changed`, all same value | 6 | 11 |
    | `found_to_other` (no wrong value) | 7 | 15 |
    | `other_to_found` | 0 | 2, correct |

    `retrieval_error` and `retrieval_unknown_chunks` were 0 in both runs.
  - **Wrong values:**
    - `bm25`: HHV 2024 consolidated `construction_in_progress` took the note sub-line 9.34 bn instead of the total 11.08 bn (BS 242). DPM 2015 consolidated `cost_of_goods_sold` and `financial_expenses` were read from the separate income statement inside the consolidated document (t6 instead of t65).
    - `hybrid`: the same HHV case; DPM 2015 `profit_before_tax` (same separate-statement cause); GAS 2021 separate `other_expenses` from a BCC note; GVR 2023 `retained_earnings`, where a note table's 01/01/2023 figure replaced the balance sheet's 31/12/2023 figure.
    - Every case is R1/R2: the top k held a non-statement table, or a statement of the wrong scope, while the right statement fell outside the top k.
  - **Lost `Found`:** the step R ambiguity is final, so a note table with several period columns (QNS, EIB), or a bank table with no unit (BID), stops the read before the full scan. These cases produce no wrong value, only lost coverage.
  - **Next step (adopted 2026-11-13 as §6.1; second gate below):** the §6 mitigation. Always add the tables tagged with the leaf's statement kind. Reading every case, it would turn all 7 wrong values into the statement value or, for DPM (both statements tagged), back to `Ambiguous`. That is not measured. Rerun §7 for both modes. Bank statements sometimes tagged `note` limit this.
- If indexes are rebuilt under `data/`, note that `data/chroma/` and `data/indexes/` are
  ignored only on branches containing 6cd4e61; tools that snapshot untracked files of the main
  checkout (the Codex desktop app hashed them during the first build) can slow indexing badly.

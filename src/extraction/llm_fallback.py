"""LLM cell-pointer fallback (spec §8): the LLM only points at a table cell; code reads and verifies it.

``LlmCellLocator`` implements the extractor's ``CellLocator`` (E6). Per call it shortlists the
document's tables (§8.2), renders the first ``max_rows`` data rows of each (E7) into one prompt,
asks the injected ``llm`` once and verifies the answer (§8.3). The LLM's own text never leaves
this module: outcomes carry only fixed-format reasons and warnings (§7.2), and nothing is logged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.calculation.metric_tree import LeafDef, LeafRequest
from src.extraction.extractor import LocatorResult
from src.extraction.matching import label_keys
from src.extraction.table_view import SourceRole, TableView
from src.indexing.table_semantics import StatementKind
from src.preprocess.cleaner import clean_text

REJECTED = "llm_pointer_rejected"
_UNTAGGED = frozenset({StatementKind.NOTE, StatementKind.UNKNOWN})
_PERIOD_TEXT = {"flow": "trong năm", "closing": "số dư cuối năm"}
_FRAME = (
    "Bạn là trợ lý đọc báo cáo tài chính. Hãy tìm ô chứa giá trị của chỉ tiêu dưới đây.\n"
    "Chỉ tiêu: {label_vi} (tên khác: {aliases})\n"
    "Loại báo cáo: {statement_kind}. Năm tài chính: {year}. Kỳ: {period}.\n"
    'Chỉ trả về vị trí ô (table_ref, row_idx, col_idx) hoặc status "not_present". Không được trả về con số.\n'
)
_MIN_LABEL_TOKEN = 3  # §8.3 check 5


class CellPointer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_ref: str
    row_idx: int = Field(ge=0)
    col_idx: int = Field(ge=0)


class PointerAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["found", "not_present"]
    pointer: CellPointer | None

    @model_validator(mode="after")
    def _pointer_iff_found(self) -> PointerAnswer:
        if (self.status == "found") != (self.pointer is not None):
            raise ValueError("pointer is required when status is found and forbidden otherwise")
        return self


def structured_pointer_adapter(client) -> Callable[[str], PointerAnswer | None]:
    """Adapt a LangChain-style chat model to the ``llm(prompt)`` callable of ``LlmCellLocator``.

    Calls ``client.with_structured_output(PointerAnswer, include_raw=True)`` once, here; nothing
    from langchain is imported. Each call returns a ``PointerAnswer`` only when ``parsing_error``
    is None and ``parsed`` validates as one: ``parsed`` (a ``PointerAnswer``, another model or a
    mapping) is re-validated strictly, so a string index or an extra field (e.g. a number) gives
    None, which the locator reports as ``llm_fallback_unparsed``. The raw text is never returned.
    """
    binder = getattr(client, "with_structured_output", None)
    if not callable(binder):
        raise TypeError("client must provide with_structured_output")
    structured = binder(PointerAnswer, include_raw=True)

    def invoke(prompt: str) -> PointerAnswer | None:
        response = structured.invoke(prompt)
        if not isinstance(response, Mapping) or response.get("parsing_error") is not None:
            return None
        parsed = response.get("parsed")
        if isinstance(parsed, BaseModel):
            parsed = parsed.model_dump()
        if not isinstance(parsed, Mapping):
            return None
        try:
            return PointerAnswer.model_validate(dict(parsed), strict=True)
        except ValidationError:
            return None

    return invoke


# --- shortlist and prompt (§8.2, E7) --------------------------------------------
def _leaf_tokens(leaf: LeafDef) -> frozenset[str]:
    return frozenset(t for key in label_keys(leaf) for t in key.split())


def shortlist(leaf: LeafDef, views: Sequence[TableView], max_tables: int) -> list[TableView]:
    """Tables of the leaf's statement kind and ``note``/``unknown``/untagged tables, best first.

    Score: distinct folded label tokens shared with the leaf's ``label_vi`` and aliases, +2 for the
    leaf's statement kind; ties by page, then ``table_id``. At most ``max_tables``.
    """
    tokens = _leaf_tokens(leaf)
    scored = []
    for view in views:
        kind = view.record.statement_kind
        same_kind = kind == leaf.statement_kind
        if not (same_kind or kind is None or kind in _UNTAGGED):
            continue
        score = len(view.tokens() & tokens) + (2 if same_kind else 0)
        scored.append((-score, view.record.page, view.record.table_id, view))
    scored.sort(key=lambda s: s[:3])
    return [s[3] for s in scored[:max_tables]]


def _grid(view: TableView) -> list[list[str]]:
    """The table's rows; ``TableView`` already checked that ``rows_json`` decodes to a grid of strings."""
    return json.loads(view.record.rows_json)


def _rendered_rows(view: TableView, n_rows: int, max_rows: int) -> int:
    """Rows ``0 .. n-1`` are rendered: everything up to the ``max_rows``-th data row (E7)."""
    data = view.data_rows
    return n_rows if len(data) <= max_rows else data[max_rows]


def _render(ref: str, grid: list[list[str]], rendered: int) -> str:
    lines = [ref]
    for r in range(rendered):
        cells = (f" | {c}: {text}" for c, text in ((c, clean_text(cell)) for c, cell in enumerate(grid[r])) if text)
        lines.append(f"{r}{''.join(cells)}")
    return "\n".join(lines)


def build_prompt(
    leaf: LeafDef, request: LeafRequest, tables: Sequence[TableView], *, max_rows: int, max_chars: int
) -> tuple[str, dict[str, tuple[TableView, int]]]:
    """The prompt and ``table_ref -> (view, rendered_rows)`` for the tables that fit.

    ``tables`` are ranked best first; ``T1``.. follow that order. Rows with index below
    ``rendered_rows`` are in the prompt. Lowest-ranked tables are dropped whole until the prompt
    is at most ``max_chars`` long; when not even the first fits, no table is rendered.
    """
    frame = _FRAME.format(
        label_vi=leaf.label_vi,
        aliases=", ".join(leaf.aliases),
        statement_kind=str(leaf.statement_kind),
        year=request.year,
        period=_PERIOD_TEXT[request.period_kind],
    )
    blocks: list[str] = []
    refs: dict[str, tuple[TableView, int]] = {}
    for i, view in enumerate(tables, start=1):
        grid = _grid(view)
        rendered = _rendered_rows(view, len(grid), max_rows)
        refs[f"T{i}"] = (view, rendered)
        blocks.append(_render(f"T{i}", grid, rendered))
    while blocks and len(_join(frame, blocks)) > max_chars:
        blocks.pop()
        refs.pop(f"T{len(blocks) + 1}")
    return _join(frame, blocks), refs


def _join(frame: str, blocks: Sequence[str]) -> str:
    return "\n\n".join((frame, *blocks))


# --- locator (§8.3, §8.4) -------------------------------------------------------
class LlmCellLocator:
    def __init__(
        self,
        llm: Callable[[str], PointerAnswer | None],
        *,
        llm_errors: tuple[type[Exception], ...] = (TimeoutError, ConnectionError, OSError),
        max_tables: int = 5,
        max_rows: int = 60,
        max_chars: int = 12000,
    ):
        if not callable(llm):
            raise TypeError("llm must be callable")
        if not isinstance(llm_errors, tuple) or not all(
            isinstance(e, type) and issubclass(e, Exception) for e in llm_errors
        ):
            raise TypeError("llm_errors must be a tuple of Exception subclasses")
        for name, value in (("max_tables", max_tables), ("max_rows", max_rows), ("max_chars", max_chars)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        self._llm = llm
        self._llm_errors = llm_errors
        self._max_tables = max_tables
        self._max_rows = max_rows
        self._max_chars = max_chars

    def locate(
        self, leaf: LeafDef, request: LeafRequest, role: SourceRole, views: Sequence[TableView]
    ) -> LocatorResult:
        """At most one LLM call; no retry. No table fits the prompt → no call, nothing found."""
        tables = shortlist(leaf, views, self._max_tables)
        prompt, refs = build_prompt(leaf, request, tables, max_rows=self._max_rows, max_chars=self._max_chars)
        if not refs:
            return LocatorResult(None, None, ())
        try:
            answer = self._llm(prompt)
        except self._llm_errors as exc:
            return LocatorResult(None, None, (f"llm_fallback_error:{type(exc).__name__}",))
        if not isinstance(answer, PointerAnswer):  # None, or a callable breaking its contract
            return LocatorResult(None, None, ("llm_fallback_unparsed",))
        if answer.status == "not_present":
            return LocatorResult(None, None, ())
        return _verify(leaf, request, role, refs, answer.pointer)


def _reject(check: str) -> LocatorResult:
    return LocatorResult(None, REJECTED, (f"{REJECTED}:{check}",))


def _verify(
    leaf: LeafDef,
    request: LeafRequest,
    role: SourceRole,
    refs: Mapping[str, tuple[TableView, int]],
    pointer: CellPointer,
) -> LocatorResult:
    """§8.3 checks 1-5 in order, plus E7 (``not_rendered``); the first failing check is reported."""
    if pointer.table_ref not in refs:
        return _reject("unknown_table")
    view, rendered = refs[pointer.table_ref]
    grid = _grid(view)
    row, col = pointer.row_idx, pointer.col_idx
    if row >= len(grid) or col >= max((len(r) for r in grid), default=0):
        return _reject("out_of_bounds")
    if row >= rendered:
        return _reject("not_rendered")
    if row not in view.data_rows:
        return _reject("not_data_row")
    if col not in view.value_cols:
        return _reject("not_value_column")
    period_cols = view.period_columns(request.year, request.period_kind, role)
    if col not in period_cols:
        return _reject("period")
    reads = [view.read(row, c, leaf.dimension) for c in period_cols]
    # Never pick between differing values: every period column of the row must read the same.
    if len({(r.status, r.value.value if r.status == "ok" else r.reason) for r in reads}) > 1:
        return _reject("period")
    read = reads[period_cols.index(col)]
    if read.status == "unit":
        return _reject("unit")
    if read.status != "ok":  # blank or unparsed
        return _reject("number")
    row_tokens = {t for t in view.norm_label(row).split() if len(t) >= _MIN_LABEL_TOKEN}
    if not row_tokens & _leaf_tokens(leaf):
        return _reject("label")
    return LocatorResult(read, None, ())

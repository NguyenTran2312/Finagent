"""Candidate tables and row matching for one leaf inside one document (spec §6.2-§6.4)."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from src.calculation.metric_tree import LeafDef
from src.calculation.units import Dimension
from src.extraction.outcomes import Method
from src.extraction.table_view import TableView, enumerator_level
from src.indexing.table_semantics import TOTAL_MARKERS, StatementKind
from src.preprocess.cleaner import fold_text

_UNTAGGED = frozenset({StatementKind.NOTE, StatementKind.UNKNOWN})
_DIGITS = re.compile(r"\d+")
# §6.4 no-code pattern: another child must follow the second X row within this many rows.
_PATTERN_WINDOW = 3


@dataclass(frozen=True)
class RowHit:
    view: TableView
    row_idx: int
    method: Method
    warnings: tuple[str, ...] = ()


@cache  # leaves are frozen config objects; matching re-reads their keys per table
def label_keys(leaf: LeafDef) -> frozenset[str]:
    """Folded ``label_vi`` and aliases of ``leaf``; rows match only by exact equality."""
    return frozenset(key for key in (fold_text(t) for t in (leaf.label_vi, *leaf.aliases)) if key)


def anchor_kinds(view: TableView, template_leaves: Sequence[LeafDef]) -> frozenset[StatementKind]:
    """Statement kinds whose labels anchor ``view`` (§6.2 anchor rule; labels only, never codes).

    A kind anchors when the table's normalised row labels exactly equal a label or alias of
    at least min(3, n) distinct statement leaves of that kind, n being the number of such leaves,
    through at least min(3, n) distinct labels (leaves sharing a label do not count twice).
    """
    labels = {view.norm_label(r) for r in _label_rows(view)}
    by_kind: dict[StatementKind, list[LeafDef]] = defaultdict(list)
    for leaf in template_leaves:
        if leaf.source == "statement" and leaf.statement_kind not in _UNTAGGED:
            by_kind[leaf.statement_kind].append(leaf)
    anchored = set()
    for kind, leaves in by_kind.items():
        need = min(3, len(leaves))
        matched = [label_keys(leaf) & labels for leaf in leaves]
        if sum(map(bool, matched)) >= need and len(frozenset().union(*matched)) >= need:
            anchored.add(kind)
    return frozenset(anchored)


def find_rows(leaf: LeafDef, views: Sequence[TableView], template_leaves: Sequence[LeafDef]) -> list[RowHit]:
    """Rows of one document's tables that hold ``leaf``, ordered by (page, table_id, row_idx) (E5).

    Statement leaves search tables tagged with their kind first and, only when those give no hit,
    ``note``/``unknown`` tables that anchor for exactly that kind. Note leaves search
    ``note``/``unknown`` tables. A table without a ``statement_kind`` counts as ``unknown``.
    """
    untagged = [v for v in views if v.record.statement_kind is None or v.record.statement_kind in _UNTAGGED]
    if leaf.source == "notes":
        return _ordered(hit for v in untagged for hit in _match(leaf, v, template_leaves))
    tagged = [v for v in views if v.record.statement_kind == leaf.statement_kind]
    hits = _ordered(hit for v in tagged for hit in _match(leaf, v, template_leaves))
    if hits:
        return hits
    anchored = [v for v in untagged if anchor_kinds(v, template_leaves) == {leaf.statement_kind}]
    return _ordered(hit for v in anchored for hit in _match(leaf, v, template_leaves))


def _ordered(hits) -> list[RowHit]:
    unique = {(h.view.record.page, h.view.record.table_id, h.row_idx): h for h in hits}
    return [unique[key] for key in sorted(unique)]


def _match(leaf: LeafDef, view: TableView, template_leaves: Sequence[LeafDef]) -> list[RowHit]:
    if leaf.source == "notes":
        return _by_label(leaf, view, ())
    if leaf.template == "bank":
        if leaf.parent is not None:
            return _section_rows(_leaf(leaf.parent, template_leaves), view, template_leaves, child=leaf)
        if _children(leaf, template_leaves):
            return _section_rows(leaf, view, template_leaves, child=None)
        return _by_label(leaf, view, ())
    if leaf.codes and view.code_col is not None:
        wanted = {_code_key(c) for c in leaf.codes}
        return [RowHit(view, r, "code") for r in view.data_rows if view.code(r) and _code_key(view.code(r)) in wanted]
    return _by_label(leaf, view, ("matched_by_label",))


def _code_key(code: str) -> int | str:
    """Codes compare as integers ("01" == "1")."""
    return int(code) if _DIGITS.fullmatch(code) else code


def _is_heading(view: TableView, row_idx: int) -> bool:
    """A top-level row (code A, B, ...) whose value cells are all empty, e.g. "B NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU"."""
    return enumerator_level(view.code(row_idx)) == "top" and all(
        not view.read(row_idx, col, Dimension.MONEY).location.raw_text for col in view.value_cols
    )


def _label_rows(view: TableView) -> list[int]:
    """Data rows that may match by label: headings never do."""
    return [r for r in view.data_rows if not _is_heading(view, r)]


def _by_label(leaf: LeafDef, view: TableView, warnings: tuple[str, ...]) -> list[RowHit]:
    keys = label_keys(leaf)
    return [RowHit(view, r, "label", warnings) for r in _label_rows(view) if view.norm_label(r) in keys]


def _leaf(leaf_id: str, template_leaves: Sequence[LeafDef]) -> LeafDef:
    for leaf in template_leaves:
        if leaf.id == leaf_id:
            return leaf
    raise ValueError(f"parent leaf {leaf_id!r} not in template_leaves")


def _children(section: LeafDef, template_leaves: Sequence[LeafDef]) -> list[LeafDef]:
    return [leaf for leaf in template_leaves if leaf.parent == section.id]


def _is_total(label: str) -> bool:
    return any(label == marker or label.startswith(f"{marker} ") for marker in TOTAL_MARKERS)


def _section_rows(
    section: LeafDef, view: TableView, template_leaves: Sequence[LeafDef], *, child: LeafDef | None
) -> list[RowHit]:
    """Section-leaf rows (``child`` is None) or rows of ``child`` inside its section's blocks (§6.4, R2)."""
    children = _children(section, template_leaves)
    child_keys = [label_keys(c) for c in children]
    blocks = _blocks(section, view, child_keys) if view.code_col is not None else _pattern_blocks(view, child_keys)
    if child is None:
        return [RowHit(view, head, "section_position") for head, _ in blocks]
    keys = label_keys(child)
    return [
        RowHit(view, r, "section_position") for _, body in blocks for r in body if view.norm_label(r) in keys
    ]


def _blocks(
    section: LeafDef, view: TableView, child_keys: list[frozenset[str]]
) -> list[tuple[int, tuple[int, ...]]]:
    """Qualifying (section row, rows after it in its block) pairs, using the enumerator column."""
    all_child_keys = frozenset().union(*child_keys)
    own_keys = label_keys(section) | all_child_keys
    rows = view.data_rows
    found = []
    for pos, head in enumerate(rows):
        if enumerator_level(view.code(head)) != "section" or view.norm_label(head) not in own_keys:
            continue
        body = []
        for r in rows[pos + 1:]:
            if enumerator_level(view.code(r)) in ("top", "section") or _is_total(view.norm_label(r)):
                break
            body.append(r)
        if any(view.norm_label(r) in all_child_keys for r in body):
            found.append((head, tuple(body)))
    return found


def _pattern_blocks(view: TableView, child_keys: list[frozenset[str]]) -> list[tuple[int, tuple[int, ...]]]:
    """No enumerator column: rows X, X, then another child within 3 rows (§6.4).

    The first X row is the section; the body is the second X row and the 3 rows after it.
    """
    rows = view.data_rows
    found = []
    for pos in range(len(rows) - 1):
        label = view.norm_label(rows[pos])
        if not label or view.norm_label(rows[pos + 1]) != label:
            continue
        first = next((i for i, keys in enumerate(child_keys) if label in keys), None)
        if first is None:
            continue
        body = rows[pos + 1:pos + 2 + _PATTERN_WINDOW]
        if any(
            view.norm_label(r) in keys
            for r in body[1:]
            for i, keys in enumerate(child_keys)
            if i != first
        ):
            found.append((rows[pos], tuple(body)))
    return found

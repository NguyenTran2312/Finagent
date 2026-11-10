"""Validated metric dependency tree: leaves read from statements, derived nodes built by formulas."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import yaml

from src.calculation.expressions import Formula, FormulaError, infer_dimension, parse_formula
from src.calculation.units import Dimension, UnitError
from src.indexing.table_semantics import StatementKind
from src.preprocess.cleaner import fold_text

Template = Literal["corporate", "bank"]
TEMPLATES: tuple[Template, ...] = ("corporate", "bank")

DEFAULT_METRICS_DIR = Path(__file__).resolve().parents[2] / "config" / "metrics"

_RESERVED_IDS = frozenset({"t", "abs"})
_MAX_OFFSET = 10
_LEAF_FILES: dict[str, Template] = {
    "corporate_leaves.yaml": "corporate",
    "bank_leaves.yaml": "bank",
}
_DERIVED_FILE = "derived.yaml"
_LEAF_REQUIRED = {"id", "label_vi", "aliases", "statement_kind", "codes", "period_kind",
                  "source", "dimension"}
_LEAF_OPTIONAL = {"parent", "template"}
_DERIVED_REQUIRED = {"id", "label_vi", "aliases", "templates", "dimension", "formulas"}
_DERIVED_OPTIONAL = {"display", "expected_range"}


class TreeError(ValueError):
    """Invalid metric tree; carries the file, node id and reason."""

    def __init__(self, file: str, node_id: str, reason: str):
        super().__init__(f"{file}: {node_id}: {reason}")
        self.file = file
        self.node_id = node_id
        self.reason = reason


@dataclass(frozen=True)
class LeafDef:
    id: str
    template: Template
    label_vi: str
    aliases: tuple[str, ...]
    statement_kind: StatementKind
    codes: tuple[str, ...]
    period_kind: Literal["flow", "closing"]
    source: Literal["statement", "notes"]
    dimension: Dimension
    parent: str | None = None


@dataclass(frozen=True)
class DerivedDef:
    id: str
    label_vi: str
    aliases: tuple[str, ...]
    templates: tuple[Template, ...]
    dimension: Dimension
    display: str | None
    formulas: tuple[Formula, ...]
    expected_range: tuple[Decimal, Decimal] | None


@dataclass(frozen=True)
class LeafRequest:
    metric_id: str
    year: int
    period_kind: str


def _str_list(value: Any, file: str, node_id: str, key: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise TreeError(file, node_id, f"{key} must be a list of non-empty strings")
    if not value and not allow_empty:
        raise TreeError(file, node_id, f"{key} must not be empty")
    return tuple(value)


def _enum(cls: type, value: Any, file: str, node_id: str, key: str):
    try:
        return cls(value)
    except ValueError:
        raise TreeError(file, node_id, f"invalid {key} {value!r}") from None


def _check_keys(entry: Mapping, required: set[str], optional: set[str], file: str) -> str:
    node_id = entry.get("id")
    node_id = node_id if isinstance(node_id, str) and node_id else "?"
    missing = required - entry.keys()
    if missing:
        raise TreeError(file, node_id, f"missing key {sorted(missing)[0]}")
    extra = entry.keys() - required - optional
    if extra:
        raise TreeError(file, node_id, f"unknown key {sorted(map(str, extra))[0]}")
    if node_id == "?":
        raise TreeError(file, node_id, "id must be a non-empty string")
    if not isinstance(entry["label_vi"], str) or not entry["label_vi"].strip():
        raise TreeError(file, node_id, "label_vi must be a non-empty string")
    return node_id


def _parse_leaf(entry: Any, template: Template, file: str) -> LeafDef:
    if not isinstance(entry, Mapping):
        raise TreeError(file, "?", "entry must be a mapping")
    node_id = _check_keys(entry, _LEAF_REQUIRED, _LEAF_OPTIONAL, file)
    if "template" in entry and entry["template"] != template:
        raise TreeError(file, node_id, f"template mismatch: {entry['template']!r} in {template} file")
    aliases = _str_list(entry["aliases"], file, node_id, "aliases", allow_empty=False)
    codes = _str_list(entry["codes"], file, node_id, "codes", allow_empty=True)
    if template == "bank" and not codes and len(aliases) < 2:
        raise TreeError(file, node_id, "needs codes or two aliases")
    if entry["period_kind"] not in ("flow", "closing"):
        raise TreeError(file, node_id, f"invalid period_kind {entry['period_kind']!r}")
    if entry["source"] not in ("statement", "notes"):
        raise TreeError(file, node_id, f"invalid source {entry['source']!r}")
    parent = entry.get("parent")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise TreeError(file, node_id, "parent must be a non-empty string")
    return LeafDef(
        id=node_id, template=template, label_vi=entry["label_vi"], aliases=aliases,
        statement_kind=_enum(StatementKind, entry["statement_kind"], file, node_id, "statement_kind"),
        codes=codes, period_kind=entry["period_kind"], source=entry["source"],
        dimension=_enum(Dimension, entry["dimension"], file, node_id, "dimension"), parent=parent,
    )


def _parse_range(value: Any, file: str, node_id: str) -> tuple[Decimal, Decimal] | None:
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float, str)) for v in value)):
        raise TreeError(file, node_id, "expected_range must be [low, high]")
    try:
        low, high = Decimal(str(value[0])), Decimal(str(value[1]))
    except InvalidOperation:
        raise TreeError(file, node_id, "expected_range must be numeric") from None
    if not (low.is_finite() and high.is_finite() and low <= high):
        raise TreeError(file, node_id, "expected_range must satisfy low <= high")
    return low, high


def _parse_derived(entry: Any, file: str) -> DerivedDef:
    if not isinstance(entry, Mapping):
        raise TreeError(file, "?", "entry must be a mapping")
    node_id = _check_keys(entry, _DERIVED_REQUIRED, _DERIVED_OPTIONAL, file)
    aliases = _str_list(entry["aliases"], file, node_id, "aliases", allow_empty=True)
    templates = _str_list(entry["templates"], file, node_id, "templates", allow_empty=False)
    if any(t not in TEMPLATES for t in templates) or len(set(templates)) != len(templates):
        raise TreeError(file, node_id, f"templates must be distinct values of {TEMPLATES}")
    texts = _str_list(entry["formulas"], file, node_id, "formulas", allow_empty=False)
    formulas = []
    for text in texts:
        try:
            formulas.append(parse_formula(text))
        except FormulaError as exc:
            raise TreeError(file, node_id, f"formula syntax: {text!r}: {exc}") from None
    display = entry.get("display")
    if display is not None and not isinstance(display, str):
        raise TreeError(file, node_id, "display must be a string")
    return DerivedDef(
        id=node_id, label_vi=entry["label_vi"], aliases=aliases, templates=templates,  # type: ignore[arg-type]
        dimension=_enum(Dimension, entry["dimension"], file, node_id, "dimension"),
        display=display, formulas=tuple(formulas),
        expected_range=_parse_range(entry.get("expected_range"), file, node_id),
    )


def _read_yaml_list(path: Path) -> list:
    if not path.is_file():
        raise TreeError(path.name, "-", "missing file")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise TreeError(path.name, "-", f"invalid yaml ({type(exc).__name__})") from None
    if not isinstance(data, list):
        raise TreeError(path.name, "-", "top level must be a list of entries")
    return data


class MetricTree:
    """Immutable, fully validated tree; construct with `load` or `from_entries`."""

    def __init__(self, leaves: list[LeafDef], derived: list[DerivedDef], files: Mapping[tuple, str]):
        self._files = dict(files)
        self._validate(leaves, derived)
        self._leaves = {(d.template, d.id): d for d in leaves}
        self._derived = {d.id: d for d in derived}
        self._alias_index: dict[Template, tuple[dict[str, str], re.Pattern | None]] = {}
        for template in TEMPLATES:
            by_alias = {fold_text(a): n.id for n in self._nodes(template) for a in n.aliases}
            pattern = re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(k) for k in sorted(by_alias, key=len, reverse=True))
                + r")(?!\w)") if by_alias else None
            self._alias_index[template] = (by_alias, pattern)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_entries(cls, leaves: Mapping[Template, list[dict]], derived: list[dict],
                     source: str = "<memory>") -> MetricTree:
        defs: list[LeafDef] = []
        for template, entries in leaves.items():
            if template not in TEMPLATES:
                raise TreeError(source, "-", f"unknown template {template!r}")
            defs += [_parse_leaf(e, template, source) for e in entries]
        ddefs = [_parse_derived(e, source) for e in derived]
        files = {("leaf", d.template, d.id): source for d in defs}
        files.update({("derived", d.id): source for d in ddefs})
        return cls(defs, ddefs, files)

    @classmethod
    def load(cls, directory: Path = DEFAULT_METRICS_DIR) -> MetricTree:
        directory = Path(directory)
        defs: list[LeafDef] = []
        files: dict[tuple, str] = {}
        for name, template in _LEAF_FILES.items():
            for entry in _read_yaml_list(directory / name):
                leaf = _parse_leaf(entry, template, name)
                defs.append(leaf)
                files[("leaf", template, leaf.id)] = name
        ddefs = []
        for entry in _read_yaml_list(directory / _DERIVED_FILE):
            node = _parse_derived(entry, _DERIVED_FILE)
            ddefs.append(node)
            files[("derived", node.id)] = _DERIVED_FILE
        return cls(defs, ddefs, files)

    def with_derived(self, d: DerivedDef) -> MetricTree:
        files = dict(self._files)
        files[("derived", d.id)] = "<proposal>"
        return MetricTree(list(self._leaves.values()), [*self._derived.values(), d], files)

    # ---- validation ---------------------------------------------------
    def _file(self, *key: str) -> str:
        return self._files.get(key, "<memory>")

    def _validate(self, leaves: list[LeafDef], derived: list[DerivedDef]) -> None:
        f = self._file
        # ids
        seen_leaf: set[tuple[str, str]] = set()
        seen_derived: set[str] = set()
        leaf_ids = {d.id for d in leaves}
        for d in leaves:
            if d.id in _RESERVED_IDS:
                raise TreeError(f("leaf", d.template, d.id), d.id, "reserved id")
            if (d.template, d.id) in seen_leaf:
                raise TreeError(f("leaf", d.template, d.id), d.id, "duplicate id")
            seen_leaf.add((d.template, d.id))
        for n in derived:
            if n.id in _RESERVED_IDS:
                raise TreeError(f("derived", n.id), n.id, "reserved id")
            if n.id in seen_derived or n.id in leaf_ids:
                raise TreeError(f("derived", n.id), n.id, "duplicate id")
            seen_derived.add(n.id)
        # aliases (one owner per folded alias within a template)
        for template in TEMPLATES:
            owners: dict[str, str] = {}
            nodes = [(f("leaf", template, d.id), d) for d in leaves if d.template == template]
            nodes += [(f("derived", n.id), n) for n in derived if template in n.templates]
            for file, node in nodes:
                for alias in node.aliases:
                    key = fold_text(alias)
                    if not key or key in owners:
                        raise TreeError(file, node.id, f"duplicate alias {alias!r}")
                    owners[key] = node.id
        # references
        dim_ok = {(d.template, d.id) for d in leaves}
        for d in leaves:
            if d.parent is not None and (d.parent == d.id or (d.template, d.parent) not in dim_ok):
                raise TreeError(f("leaf", d.template, d.id), d.id, f"unknown parent {d.parent!r}")
        by_id = {n.id: n for n in derived}
        for n in derived:
            for template in n.templates:
                for formula in n.formulas:
                    for ref in formula.refs:
                        if ref.offset > _MAX_OFFSET:
                            raise TreeError(f("derived", n.id), n.id,
                                            f"offset too large: {ref.node_id}[t-{ref.offset}]")
                        target = by_id.get(ref.node_id)
                        if not ((template, ref.node_id) in dim_ok
                                or (target is not None and template in target.templates)):
                            raise TreeError(f("derived", n.id), n.id,
                                            f"unknown reference {ref.node_id!r} for template {template}")
        # cycles
        state: dict[str, int] = {}

        def visit(node_id: str, path: list[str]) -> None:
            if state.get(node_id) == 2:
                return
            if state.get(node_id) == 1:
                raise TreeError(f("derived", node_id), node_id, "cycle: " + " -> ".join([*path, node_id]))
            state[node_id] = 1
            for formula in by_id[node_id].formulas:
                for ref in formula.refs:
                    if ref.node_id in by_id:
                        visit(ref.node_id, [*path, node_id])
            state[node_id] = 2

        for n in derived:
            visit(n.id, [])
        # dimensions, then days form
        declared = {(d.template, d.id): d.dimension for d in leaves}
        for n in derived:
            for template in n.templates:
                declared[(template, n.id)] = n.dimension
        for n in derived:
            for template in n.templates:
                for formula in n.formulas:
                    if n.dimension is Dimension.DAYS and not formula.is_days_form:
                        raise TreeError(f("derived", n.id), n.id,
                                        f"days form required (<ratio> * 365): {formula.text!r}")
                    try:
                        inferred = infer_dimension(formula, lambda r, t=template: declared[(t, r)])
                    except UnitError as exc:
                        raise TreeError(f("derived", n.id), n.id,
                                        f"dimension mismatch: {formula.text!r}: {exc}") from None
                    if inferred is not n.dimension:
                        raise TreeError(f("derived", n.id), n.id,
                                        f"dimension mismatch: declared {n.dimension}, "
                                        f"{formula.text!r} gives {inferred}")

    # ---- queries ------------------------------------------------------
    def _nodes(self, template: Template):
        yield from (d for (t, _), d in self._leaves.items() if t == template)
        yield from (n for n in self._derived.values() if template in n.templates)

    def leaf(self, id: str, template: Template) -> LeafDef:
        try:
            return self._leaves[(template, id)]
        except KeyError:
            raise KeyError(f"no leaf {id!r} for template {template}") from None

    def derived(self, id: str) -> DerivedDef:
        try:
            return self._derived[id]
        except KeyError:
            raise KeyError(f"no derived metric {id!r}") from None

    def applies(self, id: str, template: Template) -> bool:
        if (template, id) in self._leaves:
            return True
        node = self._derived.get(id)
        return node is not None and template in node.templates

    def kind(self, id: str) -> Literal["leaf", "derived"]:
        if id in self._derived:
            return "derived"
        if any(i == id for (_, i) in self._leaves):
            return "leaf"
        raise KeyError(f"unknown metric {id!r}")

    def dimension_of(self, id: str, template: Template) -> Dimension:
        if (template, id) in self._leaves:
            return self._leaves[(template, id)].dimension
        if self.applies(id, template):
            return self._derived[id].dimension
        raise KeyError(f"metric {id!r} does not apply to template {template}")

    def required_leaves(self, node_id: str, template: Template, year: int) -> list[frozenset[LeafRequest]]:
        """One leaf set per formula alternative; nested derived nodes use their first formula."""
        if not self.applies(node_id, template):
            raise KeyError(f"metric {node_id!r} does not apply to template {template}")

        def expand(formula: Formula, base_year: int) -> frozenset[LeafRequest]:
            out: set[LeafRequest] = set()
            for ref in formula.refs:
                out |= expand_node(ref.node_id, base_year - ref.offset)
            return frozenset(out)

        def expand_node(nid: str, y: int) -> frozenset[LeafRequest]:
            leaf = self._leaves.get((template, nid))
            if leaf is not None:
                return frozenset({LeafRequest(nid, y, leaf.period_kind)})
            return expand(self._derived[nid].formulas[0], y)

        leaf = self._leaves.get((template, node_id))
        if leaf is not None:
            return [frozenset({LeafRequest(node_id, year, leaf.period_kind)})]
        return [expand(f, year) for f in self._derived[node_id].formulas]

    def find_by_alias(self, text: str, template: Template) -> set[str]:
        by_alias, pattern = self._alias_index[template]
        if pattern is None:
            return set()
        return {by_alias[m.group()] for m in pattern.finditer(fold_text(text))}

"""Validation of LLM-proposed derived metrics before they are used."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from src.calculation.metric_tree import TEMPLATES, MetricTree, Template, TreeError, _parse_derived

_FILE = "<proposal>"


@dataclass(frozen=True)
class ProposalResult:
    accepted: bool
    tree: MetricTree | None
    issues: tuple[str, ...]
    proposal: Mapping[str, object]


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _snapshot(proposal) -> Mapping[str, object]:
    if not isinstance(proposal, Mapping):
        return MappingProxyType({})
    try:
        return _freeze(copy.deepcopy(dict(proposal)))
    except Exception:  # un-copyable junk inside a bad proposal; the record stays empty
        return MappingProxyType({})


def validate_proposal(tree: MetricTree, proposal: Mapping[str, object], template: Template) -> ProposalResult:
    """Check a proposed derived metric; never raises for a bad proposal."""
    if not isinstance(tree, MetricTree):
        raise TypeError("tree must be a MetricTree")
    record = _snapshot(proposal)

    def reject(*issues: str) -> ProposalResult:
        return ProposalResult(False, None, tuple(issues), record)

    if template not in TEMPLATES:
        return reject(f"unknown template {template!r}")
    try:
        node = _parse_derived(proposal, _FILE)
    except TreeError as exc:
        return reject(exc.reason)

    issues: list[str] = []
    try:
        tree.kind(node.id)
        issues.append(f"redefines existing node {node.id!r}")
    except KeyError:
        pass
    if template not in node.templates:
        issues.append(f"template not listed: {template} not in {list(node.templates)}")
    for t in node.templates:
        for alias in node.aliases:
            owners = tree.find_by_alias(alias, t)
            if owners:
                issues.append(f"alias collision: {alias!r} already used by {sorted(owners)[0]!r} ({t})")
    if issues:
        return reject(*issues)
    try:
        extended = tree.with_derived(node)
    except TreeError as exc:
        reason = exc.reason
        return reject(reason if "alias" not in reason else f"alias collision: {reason}")
    return ProposalResult(True, extended, (), record)

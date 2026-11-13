"""Map paraphrased subquery metrics to metric-tree node ids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from src.agent.llm import StructuredLlm
from src.calculation.metric_tree import MetricTree, Template
from src.calculation.proposals import validate_proposal
from src.extraction.templates import TickerTemplates
from src.query_paraphrase.schemas import QueryPlan

CATALOG_TO_TREE: dict[str, str] = {
    "revenue": "net_revenue",
    "net_revenue": "net_revenue",
    "profit_before_tax": "profit_before_tax",
    "profit_after_tax": "profit_after_tax",
    "consolidated_profit_after_tax": "profit_after_tax",
    "profit_after_tax_parent_share": "profit_after_tax_parent_share",
}


@dataclass(frozen=True)
class MetricMapping:
    subquery_id: str
    node_id: str | None
    how: Literal["catalog", "alias", "llm", "proposal", "unmapped"]


class MetricChoice(BaseModel):
    subquery_id: str
    node_id: str | None = None
    proposal: dict | None = None


class MetricChoices(BaseModel):
    choices: list[MetricChoice]


def _template_for(templates: TickerTemplates, ticker: str | None) -> Template:
    found = templates.get(ticker) if ticker else None
    return found if found in ("corporate", "bank") else "corporate"


def _node_listing(tree: MetricTree, template: Template) -> list[str]:
    lines = [f"{leaf.id} — {leaf.label_vi}" for leaf in tree.leaves(template)]
    lines += [f"{n.id} — {n.label_vi}" for n in tree._derived.values() if template in n.templates]
    return lines


def _prompt(tree: MetricTree, pending: list[tuple]) -> str:
    out = ["Map each subquery to one metric node id, or propose a new derived metric, or null.",
           "Reply as choices: subquery_id plus node_id (an id listed for its template) "
           "or proposal (id, label_vi, aliases, templates, dimension, formulas)."]
    for template in ("corporate", "bank"):
        if any(t == template for _, t in pending):
            out.append(f"\nNodes for template {template} (node_id — label_vi):")
            out += _node_listing(tree, template)
    out.append("\nSubqueries:")
    for sq, template in pending:
        out.append(f"- {sq.id} [{template}]: {sq.metric_text or sq.metric or sq.question}")
    return "\n".join(out)


def map_metrics(plan: QueryPlan, tree: MetricTree, templates: TickerTemplates,
                llm: StructuredLlm) -> tuple[tuple[MetricMapping, ...], MetricTree]:
    results: dict[str, MetricMapping] = {}
    pending: list[tuple] = []  # (subquery, template)
    for sq in plan.subqueries:
        template = _template_for(templates, sq.ticker)
        target = CATALOG_TO_TREE.get(sq.metric) if sq.metric else None
        if target is not None and tree.applies(target, template):
            results[sq.id] = MetricMapping(sq.id, target, "catalog")
            continue
        if sq.metric_text:
            ids = tree.find_by_alias(sq.metric_text, template)
            if len(ids) == 1:
                (node_id,) = ids
                if tree.applies(node_id, template):
                    results[sq.id] = MetricMapping(sq.id, node_id, "alias")
                    continue
        pending.append((sq, template))

    if pending:
        reply = llm(_prompt(tree, pending), MetricChoices)
        choices = {c.subquery_id: c for c in reply.choices}
        for sq, template in pending:
            choice = choices.get(sq.id)
            mapping = MetricMapping(sq.id, None, "unmapped")
            if choice is not None and choice.node_id is not None:
                if tree.applies(choice.node_id, template):
                    mapping = MetricMapping(sq.id, choice.node_id, "llm")
            elif choice is not None and choice.proposal is not None:
                checked = validate_proposal(tree, choice.proposal, template)
                if checked.accepted:
                    tree = checked.tree
                    mapping = MetricMapping(sq.id, str(choice.proposal["id"]), "proposal")
            results[sq.id] = mapping
    return tuple(results[sq.id] for sq in plan.subqueries), tree

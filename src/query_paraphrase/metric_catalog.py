"""Canonical financial metric catalog with explicit, non-merging aliases."""
import re
from dataclasses import dataclass

from src.preprocess.cleaner import fold_text


@dataclass(frozen=True)
class MetricDefinition:
    id: str
    label_vi: str
    aliases: tuple[str, ...]


METRIC_CATALOG: dict[str, MetricDefinition] = {m.id: m for m in (
    MetricDefinition("revenue", "Doanh thu", ("doanh thu",)),
    MetricDefinition("net_revenue", "Doanh thu thuần", ("doanh thu thuần",)),
    MetricDefinition("profit_before_tax", "Lợi nhuận trước thuế",
                     ("lntt", "lợi nhuận trước thuế")),
    MetricDefinition("profit_after_tax", "Lợi nhuận sau thuế",
                     ("lnst", "lợi nhuận sau thuế")),
    MetricDefinition("consolidated_profit_after_tax", "Lợi nhuận sau thuế hợp nhất",
                     ("lnst hợp nhất", "lợi nhuận sau thuế hợp nhất")),
    MetricDefinition("profit_after_tax_parent_share",
                     "Lợi nhuận sau thuế thuộc cổ đông công ty mẹ",
                     ("lnst thuộc cổ đông công ty mẹ",
                      "lợi nhuận sau thuế thuộc cổ đông công ty mẹ",
                      "lợi nhuận sau thuế của cổ đông công ty mẹ")),
)}

_ALIAS_TO_ID: dict[str, str] = {}
for _metric in METRIC_CATALOG.values():
    for _alias in _metric.aliases:
        _key = fold_text(_alias)
        if _ALIAS_TO_ID.setdefault(_key, _metric.id) != _metric.id:
            raise ValueError(f"metric alias {_alias!r} is declared for several metrics")

# Longest alias first so the most specific phrase wins at any position.
_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(k) for k in sorted(_ALIAS_TO_ID, key=len, reverse=True))
    + r")(?!\w)")


def find_metrics(text: str) -> set[str]:
    """Return canonical IDs of catalog metrics mentioned; specific phrases hide contained ones."""
    return {_ALIAS_TO_ID[m.group()] for m in _PATTERN.finditer(fold_text(text))}


def normalize_metric(text: str) -> str | None:
    """Return the canonical ID only when exactly one catalog metric is mentioned."""
    found = find_metrics(text)
    return next(iter(found)) if len(found) == 1 else None

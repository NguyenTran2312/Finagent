"""Ticker -> statement template config for leaf extraction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

import yaml

from src.calculation.metric_tree import Template

ExtractionTemplate = Literal["corporate", "bank", "securities", "insurance"]

DEFAULT_TICKER_TEMPLATES = Path(__file__).resolve().parents[2] / "config" / "ticker_templates.yaml"

_EXTRACTION_TEMPLATES = frozenset(get_args(ExtractionTemplate))
_CALCULATION_TEMPLATES = frozenset(get_args(Template))


class TemplateConfigError(ValueError):
    """Invalid ticker template config; carries the file, ticker and reason."""

    def __init__(self, file: str, ticker: str, reason: str):
        super().__init__(f"{file}: {ticker}: {reason}")
        self.file = file
        self.ticker = ticker
        self.reason = reason


def _normalize(ticker: object) -> str | None:
    if not isinstance(ticker, str):
        return None
    return ticker.strip().upper() or None


class _DuplicateKey(Exception):
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.ticker = ticker


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects repeated mapping keys (PyYAML keeps the last silently)."""

    def construct_mapping(self, node, deep=False):
        seen: set[str] = set()
        for key_node, _ in node.value:
            if isinstance(key_node, yaml.ScalarNode):
                key = _normalize(self.construct_object(key_node, deep=True))
                if key is not None:
                    if key in seen:
                        raise _DuplicateKey(key)
                    seen.add(key)
        return super().construct_mapping(node, deep=deep)


class TickerTemplates:
    """Read-only ticker -> template lookup; tickers are upper-cased."""

    def __init__(self, mapping: Mapping[str, ExtractionTemplate]):
        self._mapping = MappingProxyType({_normalize(k) or str(k): v for k, v in mapping.items()})

    def get(self, ticker: str) -> ExtractionTemplate | None:
        key = _normalize(ticker)
        return None if key is None else self._mapping.get(key)

    def __contains__(self, ticker: object) -> bool:
        key = _normalize(ticker)
        return key is not None and key in self._mapping

    def calculation_templates(self) -> dict[str, Template]:
        return {k: v for k, v in self._mapping.items() if v in _CALCULATION_TEMPLATES}


def load_ticker_templates(path: Path = DEFAULT_TICKER_TEMPLATES) -> TickerTemplates:
    name = path.name
    if not path.is_file():
        raise TemplateConfigError(name, "-", "missing file")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # noqa: S506 (SafeLoader subclass)
    except _DuplicateKey as dup:
        raise TemplateConfigError(name, dup.ticker, "duplicate ticker") from None
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise TemplateConfigError(name, "-", f"invalid yaml ({type(exc).__name__})") from None
    except OSError as exc:
        raise TemplateConfigError(name, "-", f"unreadable file ({type(exc).__name__})") from None
    if not isinstance(data, Mapping):
        raise TemplateConfigError(name, "-", "top level must be a mapping of ticker to template")
    result: dict[str, ExtractionTemplate] = {}
    for raw_ticker, value in data.items():
        ticker = _normalize(raw_ticker)
        if ticker is None:
            raise TemplateConfigError(name, "?", "ticker must be a non-blank string")
        if not isinstance(value, str) or value not in _EXTRACTION_TEMPLATES:
            raise TemplateConfigError(name, ticker, "unknown template")
        if ticker in result:
            raise TemplateConfigError(name, ticker, "duplicate ticker")
        result[ticker] = value
    return TickerTemplates(result)

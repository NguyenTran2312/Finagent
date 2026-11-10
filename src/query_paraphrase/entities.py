"""Company registry built from the local ticker CSV."""
import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.preprocess.cleaner import clean_text, fold_text

CSV_HEADER = ["Mã CK", "Tên công ty"]
_TICKER_CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{2,3}(?![A-Za-z0-9])")
# Finance acronyms that look like tickers but are not companies.
FINANCE_ACRONYMS = frozenset({
    "LNST", "LNTT", "BCTC", "CTCP", "TMCP", "TNDN", "ROE", "ROA", "EPS", "VND", "USD",
    "GDP", "YOY", "QOQ", "TTM", "VAT", "IFRS", "VAS", "HOSE", "HNX", "UPCOM",
})

Aliases = Mapping[str, str | Sequence[str]]


class EntityRegistry:
    def __init__(self, companies: Mapping[str, str], aliases: Aliases | None = None):
        self._names = dict(companies)
        self._keys: dict[str, set[str]] = {}
        for ticker, name in self._names.items():
            self._add(ticker, ticker)
            self._add(name, ticker)
        for alias, targets in (aliases or {}).items():
            if isinstance(targets, str):
                targets = [targets]
            if not isinstance(targets, Sequence) or not targets:
                raise ValueError(f"alias {alias!r} needs one or more ticker targets")
            for target in targets:
                normalized = clean_text(target).upper() if isinstance(target, str) else None
                if normalized not in self._names:
                    raise ValueError(f"alias {alias!r} targets unknown ticker {target!r}")
                self._add(alias, normalized)
        # Longest key first so the most specific name wins at any position.
        self._pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(k) for k in sorted(self._keys, key=len, reverse=True))
            + r")(?!\w)")

    def _add(self, key: object, ticker: str) -> None:
        folded = fold_text(key)
        if not folded:
            raise ValueError(f"empty registry key for ticker {ticker!r}")
        self._keys.setdefault(folded, set()).add(ticker)

    @classmethod
    def from_csv(cls, path: str | Path, aliases: Aliases | None = None) -> "EntityRegistry":
        companies: dict[str, str] = {}
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if next(reader, None) != CSV_HEADER:
                raise ValueError(f"{path}: header must be exactly {','.join(CSV_HEADER)}")
            for line, row in enumerate(reader, start=2):
                if len(row) != 2:
                    raise ValueError(f"{path}:{line}: expected 2 columns")
                ticker, name = clean_text(row[0]).upper(), clean_text(row[1])
                if not ticker or not name:
                    raise ValueError(f"{path}:{line}: empty ticker or company name")
                if ticker in companies:
                    raise ValueError(f"{path}:{line}: duplicate ticker {ticker}")
                companies[ticker] = name
        return cls(companies, aliases)

    def resolve(self, name: str) -> set[str]:
        return set(self._keys.get(fold_text(name), ()))

    def find_mentions(self, text: str) -> set[str]:
        found: set[str] = set()
        for match in self._pattern.finditer(fold_text(text)):
            found |= self._keys[match.group()]
        return found

    def company_name(self, ticker: str) -> str | None:
        return self._names.get(clean_text(ticker).upper())

    def unknown_ticker_candidates(self, text: str) -> set[str]:
        return {c for c in _TICKER_CANDIDATE_RE.findall(clean_text(text))
                if c not in self._names and c not in FINANCE_ACRONYMS}

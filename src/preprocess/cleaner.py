"""Vietnamese text and financial-number normalization helpers."""

import html
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from typing import Literal


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PLAIN_NUMBER_RE = re.compile(r"^\d+$")
_VN_GROUPED_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$")
_EN_GROUPED_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_DECIMAL_RE = re.compile(r"^\d+[.,]\d{1,2}$")
_SPACE_GROUPED_RE = re.compile(r"^\d{1,3}(?:\s\d{3})+$")
_MINUS_TRANSLATION = str.maketrans({ch: "-" for ch in "−–—‒﹣－‐‑"})
_JOINED_YEAR_RE = re.compile(r"(?:19|20|21)[0-9]{2}(?![0-9])")


def clean_text(value: object) -> str:
    text = html.unescape("" if value is None else str(value))
    return _SPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def fold_text(value: object) -> str:
    text = unicodedata.normalize("NFC", clean_text(value))
    # Repair joined OCR words for search without altering stored text or numeric cells.
    text = "".join((" " if index and (
        (text[index - 1].islower() and char.isupper())
        or (text[index - 1].isalpha() and _JOINED_YEAR_RE.match(text, index))
        or (text[index - 1].isdigit() and char.isalpha())
        or (text[index - 1].isupper() and char.isupper()
            and index + 1 < len(text) and text[index + 1].islower())
    ) else "") + char for index, char in enumerate(text))
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def tokens(value: object) -> tuple[str, ...]:
    return tuple(fold_text(value).split())


NumberLocale = Literal["vi", "en", "auto"]
UnitKind = Literal["monetary", "percentage", "ratio", "count", "unknown"]


@dataclass(frozen=True, slots=True)
class ParsedFinancialUnit:
    currency: str | None
    scale_multiplier: Decimal | None
    unit_kind: UnitKind
    unit_status: Literal["valid", "ambiguous", "unknown"]
    source_text: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedFinancialNumber:
    raw_value: str | None
    value: Decimal | None
    status: Literal["valid", "ambiguous", "missing", "invalid"]
    number_locale: NumberLocale
    unit_kind: UnitKind
    currency: str | None
    scale: Decimal | None
    is_percent: bool
    reason: str | None = None
    unit_status: str = "unknown"


@dataclass(frozen=True, slots=True)
class UnitConversion:
    status: Literal["compatible", "currency_conversion_required", "incompatible", "unknown"]
    multiplier: Decimal | None
    reason: str | None = None


def _number_candidate(raw: str, locale: str) -> str | None:
    decimal, grouping = (",", ".") if locale == "vi" else (".", ",")
    if raw.count(decimal) > 1:
        return None
    integer, separator, fraction = raw.partition(decimal)
    if separator and not re.fullmatch(r"[0-9]+", fraction):
        return None
    if grouping in integer and " " in integer:
        return None
    group = grouping if grouping in integer else " "
    if group in integer:
        # Leading zero is decimal evidence, not a thousands group (e.g. 0.125).
        if not re.fullmatch(r"[1-9][0-9]{0,2}(?:" + re.escape(group) + r"[0-9]{3})+", integer):
            return None
        integer = integer.replace(group, "")
    elif not re.fullmatch(r"[0-9]+", integer):
        return None
    return integer + ("." + fraction if separator else "")


def parse_financial_number(
    value: object, *, locale: NumberLocale = "auto", context: Mapping | None = None,
    unit: ParsedFinancialUnit | None = None,
) -> ParsedFinancialNumber:
    """Parse exact display values; valid syntax is never evidence verification.

    Only an explicit number_locale/decimal_separator in trusted context resolves
    ambiguity. Currency, report language and arbitrary prose do not imply locale.
    Percentages retain their displayed number; units never multiply the stored value.
    """
    if locale not in ("vi", "en", "auto"):
        raise ValueError("number locale must be vi, en or auto")
    raw_value = None if value is None else str(value)
    raw = clean_text(value)
    missing = not raw or raw.casefold() in {"-", "–", "—", "−", "n/a", "na", "nil"}
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1].strip()
    raw = raw.translate(_MINUS_TRANSLATION)
    bad_sign = negative and raw.startswith(("-", "+"))
    if raw.startswith(("-", "+")):
        negative = negative or raw.startswith("-")
        raw = raw[1:].strip()
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1].strip()
    if unit is None:
        unit = parse_financial_unit("%" if percent else "")

    def result(status, parsed=None, reason=None, resolved_locale=locale):
        return ParsedFinancialNumber(
            raw_value, parsed, status, resolved_locale,
            "percentage" if percent else unit.unit_kind,
            None if percent else unit.currency, Decimal(1) if percent else unit.scale_multiplier,
            percent, reason, "valid" if percent else unit.unit_status,
        )

    if missing:
        return result("missing", reason="Missing financial cell")
    if bad_sign:
        return result("invalid", reason="Parentheses and an explicit sign cannot be combined")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("Trusted number context must be a mapping")
    context = context or {}
    contextual = context.get("number_locale")
    decimal_separator = context.get("decimal_separator")
    if contextual not in (None, "vi", "en", "auto") or decimal_separator not in (None, ",", "."):
        raise ValueError("Invalid trusted number locale context")
    by_separator = {",": "vi", ".": "en"}.get(decimal_separator)
    hints = {hint for hint in (locale, contextual, by_separator) if hint not in (None, "auto")}
    if len(hints) > 1:
        return result("ambiguous", reason="Explicit locale conflicts with trusted context")
    selected = next(iter(hints), "auto")
    candidates = {name: normalized for name in (("vi", "en") if selected == "auto" else (selected,))
                  if (normalized := _number_candidate(raw, name)) is not None}
    if not candidates:
        return result("invalid", reason="Malformed financial number for the selected locale")
    values = {Decimal(normalized) for normalized in candidates.values()}
    if len(values) > 1:
        return result("ambiguous", reason="Separator can denote either grouping or decimals")
    name, normalized = next(iter(candidates.items()))
    number = Decimal(("-" if negative else "") + normalized)
    resolved = selected if selected != "auto" else name if len(candidates) == 1 else "auto"
    return result("valid", number, resolved_locale=resolved)


def _legacy_normalized_number(value: object) -> str | None:
    raw = clean_text(value)
    if not raw or raw.casefold() in {"-", "–", "—", "n/a", "na", "nil"}:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1].strip()
    raw = raw.translate(_MINUS_TRANSLATION).strip()
    if raw.startswith(("+", "-")):
        negative = negative or raw.startswith("-")
        raw = raw[1:].strip()
    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1].strip()
    if not any(pattern.fullmatch(raw) for pattern in (
        _PLAIN_NUMBER_RE, _VN_GROUPED_RE, _EN_GROUPED_RE, _DECIMAL_RE, _SPACE_GROUPED_RE
    )):
        return None
    if is_percent and "," in raw and "." not in raw:
        raw = raw.replace(",", ".")
    elif _SPACE_GROUPED_RE.fullmatch(raw):
        raw = raw.replace(" ", "")
    elif _EN_GROUPED_RE.fullmatch(raw):
        raw = raw.replace(",", "")
    elif _VN_GROUPED_RE.fullmatch(raw):
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        raw = raw.replace(",", ".")
    return "-" + raw if negative else raw


def _normalized_number(value: object, *, locale: NumberLocale | None = None,
                       context: Mapping | None = None) -> str | None:
    if locale is None and context is None:
        return _legacy_normalized_number(value)
    parsed = parse_financial_number(value, locale=locale or "auto", context=context)
    return str(parsed.value) if parsed.status == "valid" else None


def parse_vn_decimal(value: object, *, locale: NumberLocale | None = None,
                     context: Mapping | None = None) -> Decimal | None:
    """Legacy cell parser; pass locale='auto' or use the structured API for evidence."""
    raw = _normalized_number(value, locale=locale, context=context)
    return Decimal(raw) if raw is not None else None


def parse_vn_number(value: object, *, locale: NumberLocale | None = None,
                    context: Mapping | None = None) -> float | None:
    """Legacy float wrapper for table detection; not proof of an unambiguous value."""
    raw = _normalized_number(value, locale=locale, context=context)
    if raw is None:
        return None
    number = float(raw)
    return number if math.isfinite(number) else None


_SCALES = {"nghin ty": "1e12", "ngan ty": "1e12", "tram ty": "1e11", "ty": "1e9",
           "trieu": "1e6", "nghin": "1e3", "ngan": "1e3", "thousand": "1e3",
           "million": "1e6", "billion": "1e9", "trillion": "1e12"}
_EN_SCALE = r"(?:thousands?|millions?|billions?|trillions?)"
_SCALE = r"(?:nghin ty|ngan ty|tram ty|trieu|nghin|ngan|ty|" + _EN_SCALE + r")"
_CURRENCY = r"(?:do la my|us dollars?|vnd|usd|dong)"
_UNIT_RE = re.compile(r"\b(?:(?P<prefix>" + _SCALE + r")\s+)?(?P<currency>" + _CURRENCY
                      + r")(?:\s+(?P<suffix>" + _EN_SCALE + r"))?\b")
_DECLARATION = re.compile(r"(?:đơn\s*vị(?:\s*tính)?|don\s*vi(?:\s*tinh)?|units?(?:\s+of\s+measurement)?|currency)\s*:\s*([^\n;]+)", re.I)
_NOTE = re.compile(r"(?:ghi\s*chú|ghi\s*chu|chú\s*thích|chu\s*thich|note|thuyết\s*minh)\b", re.I)


def _unit_word(match: re.Match) -> str:
    word = match.group()
    # Accent folding must not turn động/đông/đóng into the currency đồng.
    return "noncurrency" if fold_text(word) == "dong" and word.casefold() not in ("đồng", "dong") else word


def _unit_from_text(text: object, *, trusted: bool) -> ParsedFinancialUnit:
    original = "" if text is None else str(text)
    raw = unicodedata.normalize("NFC", html.unescape(original))
    declarations = list(_DECLARATION.finditer(raw))
    if declarations:
        # A second declaration in an explanatory note is not a table declaration.
        note = _NOTE.search(raw)
        declarations = [match for match in declarations if note is None or match.start() < note.start()]
        raw = " | ".join(match.group(1) for match in declarations)
        trusted = True
    raw = _NOTE.split(raw, maxsplit=1)[0]
    raw = re.sub(r"(?i)(đồng|dong)\s*\(\s*vnd\s*\)", r"\1", raw)
    raw = re.sub(r"\b[^\W\d_]+\b", _unit_word, raw)
    folded = fold_text(raw)
    matches = list(_UNIT_RE.finditer(folded))
    found = set()
    for match in matches:
        currency = "USD" if match.group("currency") in ("usd", "do la my", "us dollar", "us dollars") else "VND"
        prefix, suffix = match.group("prefix"), match.group("suffix")
        if prefix and suffix:
            return ParsedFinancialUnit(None, None, "unknown", "ambiguous", original, "Two multipliers on one currency")
        scale = (prefix or suffix or "").rstrip("s")
        found.add(("monetary", currency, Decimal(_SCALES[scale]) if scale else Decimal(1)))
    kind = None
    if "%" in raw or re.search(r"\b(?:phan tram|percent|percentage)\b", folded):
        kind = "percentage"
    elif re.search(r"\b(?:lan|times|ratio)\b", folded):
        kind = "ratio"
    elif re.search(r"\b(?:co phieu|shares|nguoi|people|count)\b", folded):
        kind = "count"
    if kind:
        found.add((kind, None, Decimal(1)))
    # Plain labels are accepted; ordinary report/body sentences are not scanned.
    residual = _UNIT_RE.sub("", folded).strip()
    residual = re.sub(r"\b(?:phan tram|percent|percentage|lan|times|ratio|co phieu|shares|nguoi|people|count)\b", "", residual)
    residual = re.sub(r"\b(?:or|and|hoac|va)\b", "", residual).strip()
    bare_currency = bool(matches) and all(not match.group("prefix") and not match.group("suffix") for match in matches)
    bare_residual = re.sub(r"\b(?:nam truoc|nam nay|cuoi nam|dau nam)\b|\b(?:19|20|21)[0-9]{2}\b", "", residual).strip()
    if bare_currency and bare_residual:
        # Unsupported multipliers such as 'hundred USD' must not become USD x1.
        if trusted:
            currencies = {item[1] for item in found}
            return ParsedFinancialUnit(next(iter(currencies)) if len(currencies) == 1 else None,
                                       None, "monetary", "unknown", original,
                                       "Currency is explicit but the unit/scale label is not recognized")
        found.clear()
    if not trusted and residual:
        found.clear()
    if not found:
        return ParsedFinancialUnit(None, None, "unknown", "unknown", original, "No explicit supported unit")
    if len(found) != 1:
        kinds = {item[0] for item in found}
        currencies = {item[1] for item in found}
        return ParsedFinancialUnit(next(iter(currencies)) if len(currencies) == 1 else None,
                                   None, next(iter(kinds)) if len(kinds) == 1 else "unknown",
                                   "ambiguous", original, "Conflicting explicit units")
    unit_kind, currency, scale = next(iter(found))
    return ParsedFinancialUnit(currency, scale, unit_kind, "valid", original)


def parse_financial_unit(
    source_text: object = "", *, column_header: object | None = None,
    table_header: object | None = None, trusted_metadata: object | None = None,
) -> ParsedFinancialUnit:
    """Prefer trusted metadata, then column/table headers, then a declared unit.

    Conflicts at a higher priority are retained, never rescued by a lower tier.
    Pass only actual headers/metadata in the trusted arguments, not table bodies.
    """
    for text in (trusted_metadata, column_header, table_header):
        if text is not None:
            parsed = text if isinstance(text, ParsedFinancialUnit) else _unit_from_text(text, trusted=True)
            if parsed.unit_status != "unknown" or parsed.currency is not None:
                return parsed
    return _unit_from_text(source_text, trusted=False)


def parse_requested_unit(question: object) -> ParsedFinancialUnit:
    raw = "" if question is None else str(question)
    # These explicit target markers exclude a source unit mentioned in the question.
    parts = re.split(r"\b(?:bao\s+nhiêu|bao\s+nhieu|sang|into)\b", raw, flags=re.I)
    return _unit_from_text(parts[-1], trusted=True)


def unit_conversion(source: ParsedFinancialUnit, requested: ParsedFinancialUnit) -> UnitConversion:
    """Describe scale-only conversion; never supply or invent an exchange rate."""
    if source.unit_status != "valid" or requested.unit_status != "valid":
        return UnitConversion("unknown", None, "Both source and requested units must be explicit")
    if any(not isinstance(unit.scale_multiplier, Decimal) or not unit.scale_multiplier.is_finite()
           or unit.scale_multiplier <= 0 or (unit.unit_kind == "monetary" and unit.currency is None)
           for unit in (source, requested)):
        return UnitConversion("unknown", None, "Missing or invalid currency/scale metadata")
    if source.unit_kind != requested.unit_kind:
        return UnitConversion("incompatible", None, "Different unit kinds require an explicit operation")
    if source.currency != requested.currency:
        return UnitConversion("currency_conversion_required", None, "An external exchange rate is required")
    if source.unit_kind == "count":
        dimensions = [tuple(index for index, pattern in enumerate((r"\b(?:co phieu|shares)\b",
                       r"\b(?:nguoi|people)\b", r"\bcount\b")) if re.search(pattern, fold_text(unit.source_text)))
                      for unit in (source, requested)]
        if any(len(dimension) != 1 for dimension in dimensions):
            return UnitConversion("unknown", None, "Count denomination must be explicit")
        if dimensions[0] != dimensions[1]:
            return UnitConversion("incompatible", None, "Different count denominations")
    with localcontext(Context(prec=50)):
        factor = source.scale_multiplier / requested.scale_multiplier
    return UnitConversion("compatible", factor)


def english_vnd_scale(text: object) -> float | None:
    parsed = parse_financial_unit(text)
    if parsed.unit_status == "valid" and parsed.currency == "VND" and re.search(r"\b" + _EN_SCALE + r"\b", fold_text(text)):
        return float(parsed.scale_multiplier)
    return None


def source_scale(text: object) -> float:
    """Legacy multiplier-only wrapper; 1.0 fallback does not verify an unknown unit."""
    parsed = parse_financial_unit(text)
    return float(parsed.scale_multiplier) if parsed.unit_status == "valid" else 1.0


def requested_scale(question: str) -> float:
    """Legacy multiplier-only wrapper; use unit_conversion to check currencies."""
    parsed = parse_requested_unit(question)
    return float(parsed.scale_multiplier) if parsed.unit_status == "valid" else 1.0
"""Deterministic, bounded validation of LLM query plans before retrieval.

A ``valid`` result only means the plan passed the code checks below: strict schema and
limits, explicit question constraints (registry tickers, years from refine_question,
catalog metrics, scope, period, unit and operation keywords), ticker/year pairs when the
question layout makes them verifiable, and history provenance. It is not proof of
semantic correctness, so semantic_status stays "not_run". The phrase lists are small on
purpose; wording outside them is not recognized and is left to the optional reviewer.
"""
import re
import unicodedata
from collections.abc import Mapping

from pydantic import BaseModel, ValidationError

from config.settings import ParaphraseSettings
from src.main.refine_questions import refine_question
from src.preprocess.cleaner import clean_text, fold_text
from src.query_paraphrase.entities import EntityRegistry
from src.query_paraphrase.metric_catalog import METRIC_CATALOG, find_metrics
from src.query_paraphrase.schemas import ParaphraseResult, QATurn, QueryPlan, ValidationIssue

INVALID, CLARIFY = "invalid", "needs_clarification"
_FIELDS = ("ticker", "year", "metric", "scope", "period", "unit")
_AMBIGUOUS = "ambiguous"


def _compile(pairs):
    return tuple((label, re.compile(pattern)) for label, pattern in pairs)


# Folded text with catalog metric phrases removed first, so the metric
# "LNST thuộc cổ đông công ty mẹ" never implies the separate report scope.
_METRIC_PHRASES = re.compile(r"\b(?:" + "|".join(re.escape(alias) for alias in sorted(
    {fold_text(a) for m in METRIC_CATALOG.values() for a in m.aliases}, key=len, reverse=True))
    + r")\b")
_SCOPES = _compile((("consolidated", r"\bhop nhat\b"),
                    ("separate", r"\b(?:rieng le|bao cao rieng|cong ty me)\b")))
# Folded text. Labels are compared as sets; no equivalence (e.g. 6 tháng = H1) is inferred.
_PERIODS = _compile((("q1", r"\b(?:quy (?:1|i)|q1)\b"), ("q2", r"\b(?:quy (?:2|ii)|q2)\b"),
                     ("q3", r"\b(?:quy (?:3|iii)|q3)\b"), ("q4", r"\b(?:quy (?:4|iv)|q4)\b"),
                     ("h1", r"\bh1\b"), ("h2", r"\bh2\b"), ("6m", r"\b6 thang\b"),
                     ("9m", r"\b9 thang\b"), ("half_year", r"\bban nien\b"),
                     ("cumulative", r"\bluy ke\b")))


# The lists below match casefolded NFC text with diacritics, as whole words, because
# folding would merge e.g. "đồng" (unit) with "động" and "cộng" with "công".
def _words(pattern: str) -> str:
    return r"(?<!\w)(?:" + pattern + r")(?!\w)"


_UNITS = _compile((("billion_vnd", _words("t[ỷỉ] đồng")), ("million_vnd", _words("triệu đồng")),
                   ("thousand_vnd", _words("(?:nghìn|ngàn) đồng")),
                   ("vnd", _words("vn[dđ]|(?:đơn vị|bằng|theo|bao nhiêu) đồng|^đồng$")),
                   ("percent", "%|" + _words("phần trăm")), ("usd", _words("usd"))))
_OPERATIONS = _compile((("compare", _words("so sánh")), ("aggregate", _words("tổng|cộng")),
                        ("calculate", _words("chênh lệch|tăng trưởng|t[ỷỉ] lệ|thay đổi"))))
# Line items and names containing "tổng"/"cộng" that do not ask for an aggregation.
_NOT_OPERATIONS = re.compile(_words("tổng (?:cộng )?(?:tài sản|nguồn vốn)|tổng nợ|tổng công ty"
                                    "|tổng giám đốc|cộng đồng|cộng (?:hòa|hoà)"))
# Supported follow-up forms: leading "còn ...", "... thì sao", "công ty/doanh nghiệp/
# chỉ tiêu đó|này", "năm đó". Other implicit references are not treated as inheritance.
_FOLLOW_UP = re.compile(r"^\W*còn(?!\w)|"
                        + _words("thì sao|(?:công ty|doanh nghiệp|chỉ tiêu) (?:đó|này)|năm đó"))
_YEAR = re.compile(r"\b(?:19|20|21)\d{2}\b")


def _years_in(text: str) -> set[int]:
    return {int(year) for year in _YEAR.findall(text)}


def _nfc(text: str | None) -> str:
    return unicodedata.normalize("NFC", clean_text(text))


def _norm(text: str | None) -> str:
    return _nfc(text).casefold()


def _labels(text: str, patterns) -> set[str]:
    return {label for label, pattern in patterns if pattern.search(text)}


def _contains(text: str, phrase: str) -> bool:
    """Verbatim (case/Unicode-insensitive) whole-word occurrence of phrase in text."""
    return re.search(_words(re.escape(_norm(phrase))), _norm(text)) is not None


def _issue(severity, code, message, subquery_id=None):
    return severity, code, message, subquery_id


def _result(issues, plan=None) -> ParaphraseResult:
    if not issues:
        return ParaphraseResult(status="valid", plan=plan, issues=[])
    status = INVALID if any(issue[0] == INVALID for issue in issues) else CLARIFY
    return ParaphraseResult(status=status, plan=None, issues=[
        ValidationIssue(code=code, message=message, subquery_id=sid)
        for _, code, message, sid in issues])


def _summary(error: ValidationError) -> str:
    parts = [f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in error.errors()[:5]]
    return f"{error.error_count()} schema error(s): " + "; ".join(parts)


def _check_text(value, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")


def prepare_history(qa_history, settings: ParaphraseSettings) -> list[tuple[int, QATurn]]:
    """Validate history and keep the last max_history_turns as (original_index, QATurn).

    Accepts None or a list/tuple of QATurn-like models or mappings; input is not mutated.
    Malformed turns or fields over max_input_chars raise ValueError (nothing is truncated).
    """
    if qa_history is None:
        return []
    if not isinstance(qa_history, (list, tuple)):
        raise ValueError("qa_history must be a list of turns")
    turns = []
    for index, item in enumerate(qa_history):
        if isinstance(item, BaseModel):
            data = item.model_dump()
        elif isinstance(item, Mapping):
            data = dict(item)
        else:
            raise ValueError(f"qa_history[{index}] must be a QATurn or a mapping")
        for field in ("question", "answer"):
            if isinstance(data.get(field), str) and len(data[field]) > settings.max_input_chars:
                raise ValueError(f"qa_history[{index}].{field} exceeds "
                                 f"{settings.max_input_chars} characters")
        try:
            turns.append((index, QATurn.model_validate(data, strict=True)))
        except ValidationError as error:
            raise ValueError(f"qa_history[{index}] is invalid: {_summary(error)}") from error
    return turns[-settings.max_history_turns:] if settings.max_history_turns else []


def _parse(raw):
    """Strictly parse JSON text, a mapping or any BaseModel (always re-validated)."""
    try:
        if isinstance(raw, str):
            return QueryPlan.model_validate_json(raw, strict=True), []
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(warnings=False)
        elif isinstance(raw, Mapping):
            raw = dict(raw)
        else:
            return None, [_issue(INVALID, "unsupported_output",
                                 f"unsupported LLM output type {type(raw).__name__}")]
        return QueryPlan.model_validate(raw, strict=True), []
    except ValidationError as error:
        malformed = any(e["type"] == "json_invalid" for e in error.errors())
        return None, [_issue(INVALID, "malformed_json" if malformed else "schema_invalid",
                             _summary(error))]


def _mask_weak_tickers(text: str, registry: EntityRegistry) -> str:
    """Blank words that merely fold to a ticker ("gas", "hụt"); literal tickers stay."""
    def keep(match):
        word = match.group()
        ticker = fold_text(word).upper()
        return " " if word != ticker and registry.company_name(ticker) else word
    return re.sub(r"\w+", keep, text)


def _facts(text: str, registry: EntityRegistry) -> dict[str, set]:
    """Constraints recognized in text. "ticker" forces; "ticker_support" only supports."""
    folded = fold_text(text)
    return {"ticker": registry.find_mentions(_mask_weak_tickers(text, registry)),
            "ticker_support": registry.find_mentions(text),
            "year": set(refine_question(text, set(), default_scope=None)["years"]),
            "metric": find_metrics(text),
            "scope": _labels(_METRIC_PHRASES.sub(" ", folded), _SCOPES),
            "period": _labels(folded, _PERIODS), "unit": _labels(_norm(text), _UNITS)}


def _unknown_entities(text: str, registry: EntityRegistry) -> set[str]:
    names = {word for ticker in registry.find_mentions(text)
             for word in re.findall(r"[A-Z0-9]+", registry.company_name(ticker))}
    return {candidate for candidate in registry.unknown_ticker_candidates(text)
            if candidate not in names and not registry.resolve(candidate)}


def _values(sq) -> dict[str, set]:
    """Recognized labels of a subquery's metadata (empty set when null/unrecognized)."""
    return {"ticker": {sq.ticker} if sq.ticker else set(),
            "year": {sq.year} if sq.year is not None else set(),
            "metric": {sq.metric} if sq.metric else set(),
            "scope": {sq.scope} if sq.scope else set(),
            "period": _labels(fold_text(sq.period), _PERIODS) if sq.period else set(),
            "unit": _labels(_norm(sq.unit), _UNITS) if sq.unit else set()}


def _pairs(folded: str, spans, years: set[int]):
    """Exact (ticker, year) pairs when every entity is directly followed by its own years.

    Returns None when pairing is not verifiable (coverage checks still apply) and
    _AMBIGUOUS when several year groups attach to entities in another layout.
    """
    mentions = []
    for start, end, tickers in spans:
        previous = mentions[-1] if mentions else None
        if previous and previous[2] == tickers and not _years_in(folded[previous[1]:start]):
            mentions[-1] = (previous[0], end, tickers)  # "VJC (CTCP Hàng không Vietjet)"
        else:
            mentions.append((start, end, tickers))
    if len({t for _, _, tickers in mentions for t in tickers}) < 2 or len(years) < 2:
        return None
    bounds = [start for start, _, _ in mentions[1:]] + [len(folded)]
    lead = _years_in(folded[:mentions[0][0]])
    slots = [_years_in(folded[end:bound]) for (_, end, _), bound in zip(mentions, bounds)]
    if lead.union(*slots) != years:  # year ranges etc.: layout not verifiable
        return None
    if not lead and all(slots):
        return {(t, y) for (_, _, tickers), ys in zip(mentions, slots)
                for t in tickers for y in ys}
    return _AMBIGUOUS if sum(1 for slot in (lead, *slots) if slot) > 1 else None


def _structure_issues(plan: QueryPlan, question: str, settings: ParaphraseSettings):
    issues = []
    if _nfc(plan.original_question) != _nfc(question):
        issues.append(_issue(INVALID, "original_question_mismatch",
                             "original_question differs from the input question"))
    if len(plan.subqueries) > settings.max_subqueries:
        return issues + [_issue(INVALID, "too_many_subqueries", f"{len(plan.subqueries)} "
                                f"subqueries exceed the limit of {settings.max_subqueries}")]
    seen = {}
    for sq in plan.subqueries:
        key = (fold_text(sq.search_query), sq.ticker, sq.year, fold_text(sq.period), sq.scope,
               sq.metric, fold_text(sq.metric_text), _norm(sq.unit),
               tuple(sorted(sq.document_types)))
        if key in seen:
            issues.append(_issue(INVALID, "duplicate_subquery",
                                 f"repeats the search and constraints of {seen[key]}", sq.id))
        seen.setdefault(key, sq.id)
    for relation in plan.relationships:
        ids = relation.subquery_ids
        needs_two = relation.kind in ("compare", "aggregate")
        if len(set(ids)) != len(ids) or (needs_two and len(ids) < 2):
            issues.append(_issue(INVALID, "invalid_relationship",
                                 f"{relation.kind} relationship needs distinct operands"))
    return issues


def _input_issues(question: str, facts, registry: EntityRegistry):
    """Question-side blockers: unknown/ambiguous entities and unverifiable bindings."""
    issues = [_issue(CLARIFY, "unknown_entity", f"{c!r} is not a known ticker or company")
              for c in sorted(_unknown_entities(question, registry))]
    masked = _mask_weak_tickers(question, registry)
    spans = registry.mention_spans(masked)
    issues += [_issue(CLARIFY, "ambiguous_entity", "a mention matches several companies: "
                      + ", ".join(sorted(tickers)))
               for tickers in sorted({t for _, _, t in spans if len(t) > 1}, key=sorted)]
    if issues:
        return None, issues
    pairs = _pairs(fold_text(masked), spans, facts["year"])
    if pairs == _AMBIGUOUS:
        return None, [_issue(CLARIFY, "ambiguous_binding",
                             "cannot tell which years belong to which company")]
    return pairs, []


def _subquery_issues(plan, question, facts, history, registry):
    follow_up = bool(_FOLLOW_UP.search(_norm(question)))
    turn_facts, inherited, issues = {}, {}, []
    for sq in plan.subqueries:
        def add(severity, code, message, sid=sq.id):
            issues.append(_issue(severity, code, message, sid))
        values, sources = _values(sq), {}
        present = {f: getattr(sq, f) is not None for f in ("ticker", "company_name", "year",
                                                              "scope", "period", "unit")}
        present["metric"] = sq.metric is not None or sq.metric_text is not None
        for field, index in sq.context_refs.items():
            key = "ticker" if field == "company_name" else field
            if not follow_up:
                add(CLARIFY, "unsupported_context", f"{field} is inherited but the question "
                    "has no supported follow-up reference")
            elif index not in history:
                add(INVALID, "context_ref_out_of_window", f"{field} refers to turn {index} "
                    "outside the retained history")
            elif facts[key]:
                add(INVALID, "context_overrides_question",
                    f"{field} is stated in the current question")
            elif not present[field]:
                add(INVALID, "context_ref_unused", f"{field} refers to history but is null")
            else:
                if index not in turn_facts:
                    turn_facts[index] = _facts(history[index].question, registry)
                sources[field] = index
                continue
            values.pop(key, None)  # rejected reference: do not also judge the value

        def supported(field, labels, raw):
            index = sources.get("metric" if field == "metric_text" else field)
            src, text = facts, question
            if index is not None:
                src, text = turn_facts[index], history[index].question
            key = "ticker_support" if field in ("ticker", "company_name") else field
            if labels and labels <= src[key] or not labels and _contains(text, raw):
                if index is not None and labels:
                    fact = key.removesuffix("_support")
                    inherited.setdefault((fact, index), set()).update(labels)
                return
            if index is None:
                add(INVALID, f"invented_{field}", f"{field} {raw!r} is not in the question")
            else:
                add(INVALID, "context_value_not_in_history",
                    f"{field} {raw!r} is not in the question of turn {index}")

        if "ticker" in values and sq.ticker is not None:
            if registry.company_name(sq.ticker) is None or sq.ticker != sq.ticker.strip().upper():
                add(INVALID, "unknown_ticker", f"{sq.ticker!r} is not a registry ticker")
            else:
                supported("ticker", {sq.ticker}, sq.ticker)
        if sq.company_name is not None:
            if registry.resolve(sq.company_name) != {sq.ticker}:
                add(INVALID, "company_name_mismatch", "company_name does not resolve to ticker")
            elif "company_name" in sources:
                supported("company_name", {sq.ticker}, sq.company_name)
        if "metric" in values and sq.metric is not None:
            if sq.metric not in METRIC_CATALOG:
                add(INVALID, "unknown_metric", f"{sq.metric!r} is not a catalog metric ID")
            else:
                supported("metric", {sq.metric}, sq.metric)
            if sq.metric_text and find_metrics(sq.metric_text) not in (set(), {sq.metric}):
                add(INVALID, "metric_mismatch", "metric_text names a different catalog metric")
        elif "metric" in values and sq.metric_text is not None:
            if find_metrics(sq.metric_text):
                add(INVALID, "metric_not_normalized", "catalog metric left in metric_text only")
            else:
                supported("metric_text", set(), sq.metric_text)
        for field in ("year", "scope", "period", "unit"):
            if field in values and present[field]:
                supported(field, values[field], getattr(sq, field))
        for field in ("ticker", "year", "metric", "scope", "period"):
            if field in values and not present[field] and facts[field]:
                add(INVALID, f"missing_{field}", f"{field} stated in the question is null")
        issues += _search_text_issues(sq, registry)

    for (key, index), taken in inherited.items():
        for j in history:
            if j > index and j not in turn_facts:
                turn_facts[j] = _facts(history[j].question, registry)
        later = set().union(*(turn_facts[j][key] for j in history if j > index))
        stated = turn_facts[index][key]
        if (len(stated) > 1 and not stated <= taken) or later - taken:
            issues.append(_issue(CLARIFY, "ambiguous_context", f"{key} inherited from turn "
                                 f"{index} has several plausible antecedents in history"))
    return issues


def _search_text_issues(sq, registry):
    """Subquery question/search text must not name constraints its metadata lacks."""
    own = _values(sq)
    issues = []
    for name in ("question", "search_query"):
        text = getattr(sq, name)
        found = _facts(text, registry)
        extra = [f for f in ("ticker", "year", "metric", "scope", "period") if found[f] - own[f]]
        if _unknown_entities(text, registry):
            extra.append("unknown entity")
        if extra:
            issues.append(_issue(INVALID, "search_text_conflict", f"{name} mentions "
                                 f"{', '.join(extra)} absent from its metadata", sq.id))
    return issues


def _coverage_issues(plan: QueryPlan, facts, pairs):
    covered = {field: set() for field in _FIELDS}
    for sq in plan.subqueries:
        for field, labels in _values(sq).items():
            covered[field] |= labels
    covered["unit"] |= _labels(_norm(plan.operation_description or ""), _UNITS)
    issues = [_issue(INVALID, f"missing_{field}", f"question {field} not covered by any "
                     f"subquery: {sorted(map(str, facts[field] - covered[field]))}")
              for field in _FIELDS if facts[field] - covered[field]]
    if pairs is not None:
        got = {(sq.ticker, sq.year) for sq in plan.subqueries}
        if got != pairs:
            issues.append(_issue(INVALID, "pairing_mismatch", "ticker/year pairs "
                                 f"{sorted(got, key=str)} differ from {sorted(pairs, key=str)}"))
    return issues


def _operation_issues(plan: QueryPlan, question: str):
    intents = _labels(_NOT_OPERATIONS.sub(" ", _norm(question)), _OPERATIONS)
    issues = []
    if intents and plan.operation not in intents:
        issues.append(_issue(INVALID, "operation_mismatch", f"question asks for "
                             f"{'/'.join(sorted(intents))}, plan says {plan.operation}"))
    if plan.operation == "lookup" and any(r.kind != "lookup" for r in plan.relationships):
        issues.append(_issue(INVALID, "operation_mismatch",
                             "lookup plan carries compare/aggregate/calculate relationships"))
    return issues


def validate_plan(raw, question: str, qa_history: list[QATurn] | None = None, *,
                  registry: EntityRegistry, settings: ParaphraseSettings) -> ParaphraseResult:
    """Gate a raw LLM plan; only a ``valid`` result carries a plan that may reach retrieval.

    Caller errors (question/history) raise ValueError. Bad model output is ``invalid``;
    unknown/ambiguous entities, bindings or context are ``needs_clarification``.
    """
    _check_text(question, "question", settings.max_input_chars)
    history = dict(prepare_history(qa_history, settings))
    plan, issues = _parse(raw)
    if plan is None:
        return _result(issues)
    if issues := _structure_issues(plan, question, settings):
        return _result(issues)
    facts = _facts(question, registry)
    pairs, issues = _input_issues(question, facts, registry)
    if issues:
        return _result(issues)
    issues = (_subquery_issues(plan, question, facts, history, registry)
              + _coverage_issues(plan, facts, pairs) + _operation_issues(plan, question))
    return _result(issues, plan)

"""Vietnamese answer and refusal text, built by code only (the LLM never writes it)."""
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from src.agent.types import LeafLog
from src.calculation import Answer, FactSource
from src.calculation.units import display_unit, round_display, to_display

_UNIT_LABELS = {
    "dong": "đồng",
    "nghin": "nghìn đồng",
    "trieu": "triệu đồng",
    "ty": "tỷ đồng",
    "tram_ty": "trăm tỷ đồng",
    "nghin_ty": "nghìn tỷ đồng",
    "percent": "%",
    "percentage_points": "điểm phần trăm",
    "times": "lần",
    "days": "ngày",
    "money_per_share": "đồng/cổ phiếu",
}

_REFUSALS = {
    "plan_invalid": "Không phân tích được câu hỏi.",
    "metric_unmapped": "Không xác định được chỉ tiêu cần tính.",
    "facts_missing": "Không đủ số liệu để trả lời.",
    "gates_failed": "Không tạo được phép tính hợp lệ sau nhiều lần thử.",
    "clarify": "Câu hỏi cần làm rõ thêm.",
    "llm_unavailable": "Không gọi được mô hình ngôn ngữ, vui lòng thử lại.",
}


def _vietnamese_number(value: Decimal) -> str:
    text = f"{round_display(value, 2):,.2f}"
    text = text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return "0,00" if text == "-0,00" else text


def format_value(answer: Answer) -> str:
    if answer.kind == "quantity":
        number = to_display(answer.value, display_unit(answer.unit))
        return f"{_vietnamese_number(number)} {_UNIT_LABELS[answer.unit]}"
    if answer.kind == "boolean":
        return "Có" if answer.value else "Không"
    return str(answer.value)  # year, ticker, count


def answer_text(answer: Answer, sources: Sequence[FactSource], description: str | None) -> str:
    value = format_value(answer)
    lines = [f"{description}: {value}." if description else f"Trả lời: {value}."]
    seen: set[str] = set()
    source_lines = []
    for source in sources:
        line = f"- {source.row_label} — {source.doc_id}, trang {source.page}"
        if line not in seen:
            seen.add(line)
            source_lines.append(line)
    if source_lines:
        lines.append("Nguồn:")
        lines.extend(source_lines)
    return "\n".join(lines)


def refusal_text(code: str, details: Sequence[str]) -> str:
    return "\n".join([_REFUSALS[code], *(f"- {detail}" for detail in details)])


def leaf_detail(log: LeafLog) -> str:
    scope = "hợp nhất" if log.scope == "consolidated" else "riêng"
    text = f"{log.leaf_id} {log.ticker} {log.year} ({scope}): {log.outcome}"
    return text + (f" — {log.reason}" if log.reason else "")

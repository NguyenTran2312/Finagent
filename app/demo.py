"""Streamlit chat page over the reasoning agent. Run: streamlit run app/demo.py"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import streamlit as st  # noqa: E402

from src.agent import present  # noqa: E402
from src.agent.types import AgentConfig, AgentRun, StepEvent  # noqa: E402

DB = Path(os.environ.get("FINAGENT_DB", ROOT / "data" / "databases" / "corpus.sqlite3"))
BM25_DIR = Path(os.environ.get("FINAGENT_BM25_DIR", ROOT / "data" / "indexes" / "bm25"))
CHROMA_DIR = Path(os.environ["FINAGENT_CHROMA_DIR"]) if os.environ.get("FINAGENT_CHROMA_DIR") else None


@st.cache_resource(show_spinner="Đang khởi tạo agent...")
def get_agent(config: AgentConfig):
    # Heavy imports stay here so the page renders (and shows the banner) without them.
    from src.agent.factory import build_agent
    from src.agent.llm import CountingLlm, openai_llm

    return build_agent(config, llm=CountingLlm(openai_llm()), db=DB, bm25_dir=BM25_DIR,
                       chroma_dir=CHROMA_DIR)


def render_event(event: StepEvent) -> None:
    with st.expander(event.title):
        table = present.rows(event)
        if table:
            st.dataframe(_styled(table) if event.kind == "gather" else table, hide_index=True)
        if event.kind == "solve":
            st.code(str(event.data.get(present.CODE) or ""), language="python")
        if event.kind == "answer":
            for source in event.data.get(present.SOURCES) or []:  # type: ignore[union-attr]
                st.markdown(f"- {source}")


def _styled(table: list[dict[str, str]]):
    import pandas as pd

    frame = pd.DataFrame(table)
    failed = frame["Kết quả"].str.split(":").str[0] != "Found"
    return frame.style.apply(
        lambda row: ["background-color: rgba(255, 75, 75, 0.25)" if failed[row.name] else ""] * len(row),
        axis=1)


def render_turn(events, footer: str | None) -> None:
    for event in events:
        render_event(event)
    if footer:
        st.caption(footer)


def sidebar() -> AgentConfig:
    with st.sidebar:
        st.header("Cấu hình")
        modes = ["none", "bm25", "hybrid", "hybrid_rerank"] if CHROMA_DIR is not None else ["none", "bm25"]
        mode = st.selectbox("Bộ truy hồi bảng", modes, index=0)
        if CHROMA_DIR is None:
            st.caption("Hybrid tắt: chưa đặt FINAGENT_CHROMA_DIR.")
        locator = st.toggle("Định vị ô bằng LLM", value=True)
        repairs = st.slider("Số lần sửa mã tối đa", 0, 5, 3)
        if st.button("Hội thoại mới"):
            st.session_state["history"] = []
            st.session_state["messages"] = []
    return AgentConfig(retriever_mode=mode, locator_enabled=locator, max_repairs=repairs)


def main() -> None:
    st.set_page_config(page_title="Hỏi đáp báo cáo tài chính", layout="wide")
    st.title("Hỏi đáp báo cáo tài chính")
    st.session_state.setdefault("history", [])  # [{"question", "answer"}] fed to the agent
    st.session_state.setdefault("messages", [])  # display records
    config = sidebar()

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Chưa đặt biến môi trường OPENAI_API_KEY; không thể trả lời câu hỏi.")
        return

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            render_turn(message.get("events", ()), message.get("footer"))

    question = st.chat_input("Nhập câu hỏi")
    if not question:
        return
    st.session_state["messages"].append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()
        events_box = st.container()

        def on_event(event: StepEvent) -> None:
            status.caption(f"Đang xử lý: {event.title}")
            with events_box:
                render_event(event)

        try:
            run: AgentRun = get_agent(config).run(
                question, list(st.session_state["history"]), on_event=on_event)
        except Exception as exc:  # never show the message: it could carry secrets
            logging.getLogger(__name__).exception("agent run failed")  # server-side log only
            status.empty()
            error = f"Đã xảy ra lỗi ({type(exc).__name__}). Vui lòng thử lại."
            st.error(error)
            # Keep the question answered after a rerun; it is not added to the agent history.
            st.session_state["messages"].append({"role": "assistant", "text": error})
            return
        status.empty()
        st.markdown(run.text)
        footer = present.footer(run)
        st.caption(footer)

    # Events were already rendered live; keep them for later reruns.
    st.session_state["messages"].append(
        {"role": "assistant", "text": run.text, "events": run.events, "footer": footer})
    if run.status in ("answered", "clarify"):
        st.session_state["history"].append({"question": question, "answer": run.text})


main()

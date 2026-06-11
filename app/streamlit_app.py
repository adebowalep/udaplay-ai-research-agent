"""
UdaPlay — Interactive AI Research Agent for the Video Game Industry
Streamlit UI: question input, tool trace, confidence, source type, citations,
long-term memory indicator, structured report, conversation history.
"""
from __future__ import annotations

import json
import sys
import pathlib

import streamlit as st

# Allow running from the repo root without `pip install -e .`
_repo_root = pathlib.Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UdaPlay — Video Game Research Agent",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Source-type display helpers
# ---------------------------------------------------------------------------
_SOURCE_LABEL = {
    "local_rag":  ("💾", "Local RAG",  "green"),
    "web_search": ("🌐", "Web Search",  "blue"),
    "memory":     ("🧠", "Memory",     "orange"),
    "unknown":    ("❓", "Unknown",    "gray"),
}

_CONF_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def _source_badge(source_type: str) -> str:
    icon, label, colour = _SOURCE_LABEL.get(source_type, _SOURCE_LABEL["unknown"])
    return f":{colour}[{icon} **{label}**]"


# ---------------------------------------------------------------------------
# Backend loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading environment & vector database…")
def _load_backend():
    from dotenv import load_dotenv
    load_dotenv()

    from udaplay.config import Settings
    settings = Settings()
    settings.validate()

    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    client = chromadb.PersistentClient(path=settings.chroma_path)
    ef = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return settings, collection


@st.cache_resource(show_spinner="Initialising agent…")
def _load_agent(_collection, _settings):
    from udaplay.long_term_memory import LongTermMemory
    from udaplay.tools import make_game_tools
    from udaplay.agents import Agent

    ltm = LongTermMemory(_settings.long_term_memory_file)

    (
        retrieve, evaluate, web_search,
        search_mem, save_mem, summarize, fmt_report,
    ) = make_game_tools(_collection, settings=_settings, memory=ltm)

    INSTRUCTIONS = (
        "You are UdaPlay, an expert AI research agent specialising in the video game industry.\n\n"
        "Follow this workflow for EVERY question:\n"
        "1. Call `search_memory` to check long-term memory for previous research.\n"
        "2. Call `retrieve_game` to search the local vector database.\n"
        "3. Call `evaluate_retrieval` to judge the quality of local results.\n"
        "4. Decide the source:\n"
        "   - If `evaluate_retrieval` returns high/medium confidence: answer from local RAG.\n"
        "   - If `search_memory` had a strong match AND local RAG is weak: answer from memory.\n"
        "   - If both are insufficient: call `game_web_search`.\n"
        "5. Optionally call `summarize_game_profile` when the user asks for details about a specific game.\n"
        "6. Always call `format_report` as the final step with the complete answer.\n"
        "7. If the answer is useful (confidence medium or high), call `save_memory`.\n\n"
        "Always cite your sources. Be explicit about whether the answer came from "
        "Local RAG, Memory, or Web Search."
    )

    agent = Agent(
        model_name=_settings.llm_model,
        instructions=INSTRUCTIONS,
        tools=[retrieve, evaluate, web_search, search_mem, save_mem, summarize, fmt_report],
        temperature=0.0,
    )
    return agent, ltm


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Run parser
# ---------------------------------------------------------------------------

def _parse_run(run) -> dict:
    """Extract tool trace, report, confidence, source_type, and answer from a Run."""
    final = run.get_final_state()
    messages = final.get("messages", []) if final else []

    trace: list[dict] = []
    answer = ""
    report: dict = {}
    memory_used = False
    answer_saved = False
    confidence = "unknown"
    source_type = "unknown"
    sources: list[str] = []

    for msg in messages:
        # Capture tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                trace.append({"tool": tc.function.name, "args": args})

        # Parse tool results
        msg_name = getattr(msg, "name", "") or ""
        msg_content = getattr(msg, "content", "") or ""

        if isinstance(msg_content, str) and msg_content.startswith('"'):
            # Unwrap double-encoded JSON strings
            try:
                msg_content = json.loads(msg_content)
            except Exception:
                pass

        if msg_name == "search_memory":
            try:
                data = json.loads(msg_content) if isinstance(msg_content, str) else msg_content
                if data.get("found"):
                    memory_used = True
            except Exception:
                pass

        if msg_name == "save_memory":
            if "saved successfully" in str(msg_content).lower():
                answer_saved = True

        if msg_name == "format_report":
            try:
                raw = msg_content if isinstance(msg_content, str) else json.dumps(msg_content)
                report = json.loads(raw)
                confidence  = report.get("confidence", confidence)
                source_type = report.get("source_type", source_type)
                sources     = report.get("sources", [])
                answer      = report.get("answer", "")
                memory_used = report.get("memory_used", memory_used)
            except Exception:
                pass

        # Capture final AI text (no tool calls)
        if (type(msg).__name__ == "AIMessage"
                and msg_content
                and not getattr(msg, "tool_calls", None)):
            answer = msg_content

    # Fallback answer from last message
    if not answer and messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        if content:
            answer = content

    return {
        "answer":       answer,
        "trace":        trace,
        "confidence":   confidence,
        "source_type":  source_type,
        "sources":      sources,
        "report":       report,
        "memory_used":  memory_used,
        "answer_saved": answer_saved,
    }


# ---------------------------------------------------------------------------
# Render a single result card
# ---------------------------------------------------------------------------

def _render_result(parsed: dict) -> None:
    st.write(parsed["answer"])

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        conf = parsed["confidence"]
        st.markdown(f"**Confidence:** {_CONF_ICON.get(conf, '⚪')} {conf}")
    with c2:
        st.markdown(f"**Source:** {_source_badge(parsed['source_type'])}")
    with c3:
        flags = []
        if parsed.get("memory_used"):
            flags.append("🧠 memory used")
        if parsed.get("answer_saved"):
            flags.append("💾 saved to memory")
        if flags:
            st.markdown(" · ".join(flags))

    if parsed.get("sources"):
        st.markdown(
            "**Citations:** "
            + " · ".join(
                f"[{s.split('/')[-1] or s}]({s})" if s.startswith("http") else f"`{s}`"
                for s in parsed["sources"]
            )
        )

    cols = st.columns(2)
    with cols[0]:
        with st.expander("🔍 Tool trace"):
            for i, step in enumerate(parsed["trace"], 1):
                arg_str = json.dumps(step["args"], indent=2) if step["args"] else "{}"
                st.code(f"{i}. {step['tool']}({arg_str})", language="json")

    with cols[1]:
        if parsed.get("report"):
            with st.expander("📋 Structured report (JSON)"):
                st.json(parsed["report"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎮 UdaPlay")
    st.caption("AI Research Agent · Video Game Industry")
    st.divider()

    st.markdown("**Workflow**")
    st.markdown(
        "1. 🧠 Search long-term memory\n"
        "2. 💾 Retrieve from local RAG\n"
        "3. ⚖️  Evaluate retrieval quality\n"
        "4. 🌐 Web search if needed\n"
        "5. 📋 Format structured report\n"
        "6. 💾 Save useful answers"
    )
    st.divider()

    with st.expander("Example questions"):
        examples = [
            "When was Pokémon Gold and Silver released?",
            "Who developed FIFA 21?",
            "Tell me about God of War Ragnarök",
            "What platforms did Super Mario 64 launch on?",
            "What is Rockstar Games working on?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["prefill"] = ex

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id}`")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🎮 UdaPlay — Video Game Research Agent")
st.caption(
    "Ask anything about the video game industry. "
    "UdaPlay checks long-term memory first, then local RAG, then live web search."
)

# Load backend
try:
    settings, collection = _load_backend()
    agent, ltm = _load_agent(collection, settings)
    backend_ok = True
except Exception as e:
    st.error(
        f"**Backend error:** {e}\n\n"
        "Check that `.env` contains `OPENAI_API_KEY` and that ChromaDB is populated "
        "(run `notebooks/01_rag_pipeline.ipynb` first)."
    )
    backend_ok = False
    agent = ltm = None

# Show memory entry count
if backend_ok and ltm:
    n = ltm.count()
    st.caption(f"🧠 Long-term memory: **{n}** saved entr{'y' if n == 1 else 'ies'}")

# Conversation history
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        _render_result(entry)

# Input
prefill = st.session_state.pop("prefill", "")
question = st.chat_input("Ask about any video game…", disabled=not backend_ok)

if question or prefill:
    q = question or prefill

    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            try:
                run    = agent.invoke(q, session_id=st.session_state.session_id)
                parsed = _parse_run(run)
            except Exception as e:
                parsed = {
                    "answer":       f"An error occurred: {e}",
                    "trace":        [],
                    "confidence":   "unknown",
                    "source_type":  "unknown",
                    "sources":      [],
                    "report":       {},
                    "memory_used":  False,
                    "answer_saved": False,
                }

        _render_result(parsed)
        st.session_state.history.append({"question": q, **parsed})

# ---------------------------------------------------------------------------
# Memory viewer (sidebar or bottom of page)
# ---------------------------------------------------------------------------
if backend_ok and ltm and ltm.count() > 0:
    st.divider()
    with st.expander(f"🧠 Long-term memory ({ltm.count()} entries)"):
        st.caption(
            "Saved answers from previous research sessions. "
            "Entries are stored locally in `data/memory/long_term_memory.json`."
        )
        entries = ltm.all_entries()
        for i, e in enumerate(reversed(entries), 1):
            icon, label, _ = _SOURCE_LABEL.get(e.get("source_type", ""), _SOURCE_LABEL["unknown"])
            conf = e.get("confidence", "?")
            ts   = e.get("timestamp", "")[:19].replace("T", " ")
            with st.expander(f"{i}. {e.get('question', '')[:80]} — {icon} {label} · {ts}"):
                st.markdown(f"**Answer:** {e.get('answer', '')[:500]}")
                if e.get("sources"):
                    st.markdown("**Sources:** " + ", ".join(e["sources"]))
                st.markdown(
                    f"**Confidence:** {_CONF_ICON.get(conf, '⚪')} {conf}  ·  "
                    f"**Tags:** {', '.join(e.get('tags', []))}"
                )

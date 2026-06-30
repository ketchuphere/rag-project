"""
Main application entrypoint – Streamlit UI.
Orchestrates PDF ingestion, vector indexing, and LangGraph query execution.

Run with:
    streamlit run app/main.py

Future improvements:
  - Decouple LangGraph execution from Streamlit by offloading graph runs to a
    Celery/Redis task queue, preventing UI blocking during Deep Research loops.
  - Replace Streamlit with a FastAPI backend + the existing React/Vite frontend
    for a production-grade, scalable deployment.
  - Add per-session persistent storage (SQLite / Postgres) to retain chat
    history and indexed documents across browser refreshes.
  - Implement streaming answer tokens via st.write_stream() so users see
    partial responses as they are generated.
"""

import re
import uuid
from html import escape

import streamlit as st

from app.services.ingestion.pdf_loader import load_pdfs
from app.services.chunking.text_splitter import split_documents
from app.services.vector_store.chroma_store import build_vectorstore, delete_collection
from app.services.generation.llm_factory import get_llm
from app.services.agents.graph import run_rag_graph, run_deep_research_graph
from app.utils.templates import bot_template, css, user_template


SMALL_TALK: dict[str, str] = {
    "hi": "Hi! Upload and process your PDFs, then ask me anything about them.",
    "hello": "Hello! I can help you search and summarise your uploaded PDFs.",
    "hey": "Hey! Send me a question about your documents whenever you are ready.",
    "thanks": "You are welcome!",
    "thank you": "You are welcome!",
    "bye": "Goodbye! Come back whenever you want to explore another PDF.",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s]", "", text).strip().lower()


def is_small_talk(question: str) -> bool:
    norm = _normalise(question)
    return norm in SMALL_TALK or any(norm.startswith(f"{p} ") for p in SMALL_TALK)


def get_small_talk_response(question: str) -> str:
    norm = _normalise(question)
    for phrase, reply in SMALL_TALK.items():
        if norm == phrase or norm.startswith(f"{phrase} "):
            return reply
    return "Hi! Upload and process your PDFs, then ask a question about them."


def render_message(role: str, message: str, trace_data=None, sources=None):
    template = user_template if role == "user" else bot_template
    safe_msg = escape(message).replace("\n", "<br>")
    st.write(template.replace("{{MSG}}", safe_msg), unsafe_allow_html=True)

    if role == "bot":
        if sources and isinstance(sources, list) and sources and sources[0] != "No sources available":
            st.markdown("**Sources:**")
            cols = st.columns(min(len(sources), 3))
            for idx, src in enumerate(sources):
                with cols[idx % 3]:
                    st.info(src, icon="📄")
        if trace_data:
            with st.expander("Workflow Trace", expanded=False):
                st.markdown(f"**Rewritten Query:** `{trace_data.get('rewritten_question')}`")
                st.markdown(f"**Documents Retrieved:** `{trace_data.get('relevant_docs_count')}`")
                st.markdown(f"**Relevance Score:** `{trace_data.get('relevance_score')}`")
                st.markdown(f"**Web Search Triggered:** `{'Yes' if trace_data.get('web_results_count', 0) > 0 else 'No'}`")


def add_message(role: str, message: str, trace_data=None, sources=None):
    st.session_state.chat_history.append({
        "role": role, "message": message,
        "trace_data": trace_data, "sources": sources,
    })


def initialise_session():
    defaults = {
        "conversation": None,
        "chat_history": [],
        "graph_memory": [],
        "file_stats": [],
        "chunk_count": 0,
        "index_status": "No documents indexed",
        "last_relevance_score": 0.0,
        "web_search_count": 0,
        "deep_research_mode": False,
        "current_provider": "Gemini",
        "fallback_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []
    if not isinstance(st.session_state.graph_memory, list):
        st.session_state.graph_memory = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = f"pdf_collection_{st.session_state.session_id}"


def answer_from_documents(question: str) -> dict:
    vectorstore = st.session_state.conversation["vectorstore"]
    llm = st.session_state.conversation["llm"]
    runner = run_deep_research_graph if st.session_state.get("deep_research_mode") else run_rag_graph
    final_state = runner(
        question=question,
        vectorstore=vectorstore,
        llm=llm,
        chat_history=st.session_state.chat_history,
        conversation_memory=st.session_state.graph_memory,
    )
    st.session_state.graph_memory = final_state["conversation_memory"]
    return final_state


def handle_input(question: str):
    add_message("user", question)

    if is_small_talk(question):
        add_message("bot", get_small_talk_response(question))
        return

    if st.session_state.conversation is None:
        add_message("bot", "Please upload and process your PDFs first, then I can answer questions.")
        return

    is_deep = st.session_state.get("deep_research_mode", False)
    with st.spinner("Deep Researching..." if is_deep else "Searching your PDFs..."):
        final_state = answer_from_documents(question)

    raw_answer = final_state.get("answer", "")
    clean_answer = re.split(
        r"\n\s*(?:4\.\s*)?Sources\s*:?\s*", raw_answer, maxsplit=1, flags=re.IGNORECASE
    )[0].strip()

    trace_data = {
        "rewritten_question": final_state.get("rewritten_question", ""),
        "relevant_docs_count": len(final_state.get("relevant_docs", [])),
        "relevance_score": final_state.get("relevance_score", 0.0),
        "web_results_count": len(final_state.get("web_results", [])),
    }
    st.session_state.last_relevance_score = trace_data["relevance_score"]
    if trace_data["web_results_count"] > 0:
        st.session_state.web_search_count += 1
    st.session_state.current_provider = final_state["llm"].current_provider
    st.session_state.fallback_count = final_state["llm"].fallback_count

    add_message("bot", clean_answer, trace_data=trace_data, sources=final_state.get("sources", []))


def reset_documents():
    conv = st.session_state.get("conversation")
    vs = conv.get("vectorstore") if conv else None
    if vs and hasattr(vs, "_client"):
        delete_collection(vs._client, st.session_state.collection_name)
    st.session_state.conversation = None
    st.session_state.graph_memory = []
    st.session_state.file_stats = []
    st.session_state.chunk_count = 0
    st.session_state.index_status = "No documents indexed"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Agentic Research Assistant", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    initialise_session()
    st.header("Agentic Research Assistant :books:")

    for item in st.session_state.chat_history:
        render_message(item["role"], item["message"], item.get("trace_data"), item.get("sources"))

    question = st.chat_input("Ask a question about your documents")
    if question:
        handle_input(question)
        st.rerun()

    with st.sidebar:
        st.title("Agent Status")
        st.toggle("Deep Research Mode", key="deep_research_mode")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🤖 Current LLM", st.session_state.current_provider)
            st.metric("Memory Entries", len(st.session_state.graph_memory))
            st.metric("Docs Indexed", len(st.session_state.file_stats))
            st.metric("Web Searches", st.session_state.web_search_count)
        with col2:
            st.metric("🔁 Fallbacks", st.session_state.fallback_count)
            st.metric("Current Mode", "Deep Research" if st.session_state.deep_research_mode else "Chat")
            st.metric("Chunks Indexed", st.session_state.chunk_count)
            st.metric("Last Relevance", f"{st.session_state.last_relevance_score:.2f}")

        st.divider()
        st.subheader("Document Management")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'",
            accept_multiple_files=True, type=["pdf"],
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Upload at least one PDF first.")
            else:
                with st.spinner("Processing PDFs..."):
                    documents, file_stats = load_pdfs(pdf_docs)
                    chunks = split_documents(documents)
                    if not chunks:
                        st.warning("No readable text was found in the uploaded PDFs.")
                        return
                    vectorstore = build_vectorstore(chunks, st.session_state.collection_name)
                    st.session_state.conversation = {"llm": get_llm(), "vectorstore": vectorstore}
                    st.session_state.file_stats = file_stats
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.index_status = "Documents indexed and ready"
                st.success("Documents processed successfully.")

        if st.button("Clear documents / reset index"):
            reset_documents()
            st.success("Document index cleared.")

        st.caption(st.session_state.index_status)
        if st.session_state.file_stats:
            st.write("Indexed files")
            for f in st.session_state.file_stats:
                st.write(f"- {f['name']} ({f['extracted_pages']}/{f['pages']} pages with text)")
            st.metric("Chunks indexed", st.session_state.chunk_count)


if __name__ == "__main__":
    main()

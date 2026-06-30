"""
Agents service – LangGraph node implementations.
Each function is a pure state-in / state-out node for the RAG or Deep Research graph.

Future improvements:
  - Decompose this module into one file per agent (memory_agent.py,
    research_agent.py, etc.) as the graph grows beyond ~5 nodes.
  - Add structured tool-calling so agents can invoke external APIs
    (Wolfram Alpha, code interpreters, database queries) mid-graph.
  - Introduce a supervisor agent that dynamically selects which sub-agent
    to invoke next, replacing the current static conditional routing.
  - Stream intermediate node outputs to the UI via Server-Sent Events (SSE)
    so users see progress during long Deep Research loops.
"""

import json
import re
from hashlib import sha256
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.documents import Document

from app.config.state import RAGState
from app.config.settings import (
    MAX_MEMORY_MESSAGES,
    TAVILY_SEARCH_URL,
    TAVILY_MAX_RESULTS,
    DEEP_RESEARCH_WEB_RESULTS,
    UNKNOWN_ANSWER,
    MIN_RESEARCH_CONFIDENCE,
    MAX_RESEARCH_ITERATIONS,
)
from app.services.retrieval.retriever import retrieve, retrieve_for_research
from app.services.evaluation.grader import grade_relevance, should_web_search, verify_evidence



def _trim(messages: list[dict]) -> list[dict]:
    return messages[-MAX_MEMORY_MESSAGES:]


def _append(messages: list[dict], role: str, message: str) -> list[dict]:
    if messages and messages[-1].get("role") == role and messages[-1].get("message") == message:
        return _trim(messages)
    return _trim([*messages, {"role": role, "message": message}])


def _format_memory(messages: list[dict]) -> str:
    if not messages:
        return "No previous conversation."
    return "\n".join(f"{m['role'].title()}: {m['message']}" for m in _trim(messages))



def _fingerprint(text: str) -> str:
    return sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def _dedup_sources(sources: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in sources:
        norm = re.sub(r"\s+", " ", s.strip().lower())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(s)
    return out



def _pdf_items(docs: list[Document]) -> list[dict]:
    items = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "?")
        chunk = doc.metadata.get("chunk_index", i)
        label = f"{source}, page {page}"
        items.append({
            "type": "pdf",
            "dedup_key": _fingerprint(doc.page_content),
            "source": label,
            "content": f"[PDF {i}: {label}, chunk {chunk}]\n{doc.page_content}",
        })
    return items


def _web_items(results: list[dict]) -> list[dict]:
    items = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "Untitled result"
        url = r.get("url") or ""
        content = r.get("content") or ""
        score = float(r.get("score", 0.0) or 0.0)
        items.append({
            "type": "web",
            "dedup_key": _fingerprint(url or f"{title} {content}"),
            "source": f"{title} ({url})" if url else title,
            "content": (
                f"[Web {i}: {title}]\n"
                f"URL: {url or 'No URL'}\n"
                f"Relevance: {score:.2f}\n"
                f"Summary: {content or 'No summary available.'}"
            ),
        })
    return items


def _fuse(pdf_docs: list[Document], web_results: list[dict]) -> tuple[str, list[str], list[dict]]:
    items = [*_pdf_items(pdf_docs), *_web_items(web_results)]
    seen: set[str] = set()
    blocks, sources, evidence = [], [], []
    for item in items:
        if item["dedup_key"] in seen:
            continue
        seen.add(item["dedup_key"])
        blocks.append(item["content"])
        sources.append(item["source"])
        evidence.append({"type": item["type"], "source": item["source"], "content": item["content"]})
    context = "\n\n".join(blocks) or "No relevant PDF chunks or web search results were found."
    return context, _dedup_sources(sources), evidence



def _tavily_search(query: str, max_results: int) -> list[dict]:
    from app.config.settings import TAVILY_API_KEY
    if not TAVILY_API_KEY:
        raise ValueError("Missing TAVILY_API_KEY environment variable.")
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": max_results,
    }
    req = Request(
        TAVILY_SEARCH_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode())
    results = data.get("results", [])
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""),
         "content": r.get("content", ""), "score": float(r.get("score", 0.0) or 0.0)}
        for r in results
    ]



def _strip_sources_section(text: str) -> str:
    return re.split(r"\n\s*(?:4\.\s*)?Sources\s*:?\s*", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _format_answer(answer: str, sources: list[str]) -> str:
    clean = _strip_sources_section(answer) or UNKNOWN_ANSWER
    deduped = _dedup_sources(sources) or ["No sources available"]
    source_lines = "\n".join(f"* {s}" for s in deduped)
    return f"{clean}\n\nSources:\n\n{source_lines}"


def _format_report(report: str, sources: list[str]) -> str:
    clean = _strip_sources_section(report) or (
        "1. Executive Summary\n\nInsufficient evidence.\n\n"
        "2. Key Findings\n\n* No findings.\n\n"
        "3. Detailed Analysis\n\nNot enough context."
    )
    deduped = _dedup_sources(sources) or ["No sources available"]
    source_lines = "\n".join(f"* {s}" for s in deduped)
    return f"{clean}\n\n4. Sources\n\n{source_lines}"



def initialize_memory(state: RAGState) -> RAGState:
    memory = state["conversation_memory"] or state["chat_history"]
    memory = _append(memory, "user", state["question"])
    return {**state, "conversation_memory": memory, "memory_context": _format_memory(memory)}


def rewrite_query(state: RAGState) -> RAGState:
    prompt = f"""
Rewrite the user's question into a concise search query for retrieving relevant PDF chunks.
Use the recent chat only to resolve references like "it", "that", or "the previous topic".
Do not answer the question. Return only the rewritten search query.

Recent chat:
{state["memory_context"]}

User question:
{state["question"]}
"""
    response = state["llm"].invoke(prompt)
    rewritten = response.content.strip() or state["question"]
    return {**state, "rewritten_question": rewritten}


def retrieve_documents(state: RAGState) -> RAGState:
    query = state["rewritten_question"] or state["question"]
    candidates, relevant, sources = retrieve(state["vectorstore"], query)
    return {**state, "candidate_docs": candidates, "relevant_docs": relevant, "sources": sources}


def grade_retrieved_documents(state: RAGState) -> RAGState:
    score = grade_relevance(
        state["llm"], state["question"], state["rewritten_question"], state["relevant_docs"]
    )
    return {**state, "relevance_score": score}


def route_after_grading(state: RAGState) -> str:
    return "web_search" if should_web_search(state["relevance_score"]) else "context_fusion"


def web_search(state: RAGState) -> RAGState:
    query = state["rewritten_question"] or state["question"]
    try:
        results = _tavily_search(query, TAVILY_MAX_RESULTS)
    except Exception as exc:
        return {
            **state,
            "web_results": [],
            "answer": (
                f"Retrieved PDF relevance score: {state['relevance_score']:.2f}.\n\n"
                f"Tavily web search is not available: {exc}"
            ),
        }
    return {**state, "web_results": results}


def context_fusion(state: RAGState) -> RAGState:
    context, sources, _ = _fuse(state["relevant_docs"], state["web_results"])
    return {**state, "final_context": context, "context_sources": sources}


def generate_answer(state: RAGState) -> RAGState:
    prompt = f"""
You are a helpful question-answering assistant.
Answer the user's question using only the provided fused context.
If the answer is not present in the context, say: "{UNKNOWN_ANSWER}"
Include short source citations when you use context facts.

Recent chat:
{state["memory_context"]}

Question:
{state["question"]}

Fused context:
{state["final_context"]}
"""
    response = state["llm"].invoke(prompt)
    answer = _format_answer(response.content, state["context_sources"])
    return {**state, "answer": answer}


def update_memory(state: RAGState) -> RAGState:
    memory = _append(state["conversation_memory"], "assistant", state["answer"])
    return {**state, "conversation_memory": memory, "memory_context": _format_memory(memory)}



def research_agent(state: RAGState) -> RAGState:
    # Create plan on first iteration
    if not state["research_plan"]:
        plan_prompt = f"""
Create a concise research plan for answering the user's question.
Break the work into focused search and evidence-gathering steps.
Return the plan as short bullets.

Recent chat:
{state["memory_context"]}

Research question:
{state["question"]}
"""
        plan_resp = state["llm"].invoke(plan_prompt)
        state = {**state, "research_plan": plan_resp.content.strip()}

    query = state["rewritten_question"] or state["question"]
    candidates, relevant, sources = retrieve_for_research(state["vectorstore"], query)
    state = {**state, "candidate_docs": candidates, "relevant_docs": relevant, "sources": sources}

    try:
        web_results = _tavily_search(query, DEEP_RESEARCH_WEB_RESULTS)
    except Exception:
        web_results = []

    context, ctx_sources, evidence = _fuse(relevant, web_results)
    return {
        **state,
        "web_results": web_results,
        "research_evidence": evidence,
        "final_context": context,
        "context_sources": ctx_sources,
        "research_iterations": state["research_iterations"] + 1,
    }


def verification_agent(state: RAGState) -> RAGState:
    if not state["research_evidence"]:
        return {
            **state,
            "confidence_score": 0.0,
            "consistency_passed": False,
            "verification_report": "No evidence was gathered for verification.",
        }
    result = verify_evidence(
        state["llm"],
        state["question"],
        state["research_plan"],
        state["final_context"],
        state["context_sources"],
    )
    return {
        **state,
        "confidence_score": result["confidence_score"],
        "consistency_passed": result["consistency_passed"],
        "verification_report": result["verification_report"],
    }


def route_after_verification(state: RAGState) -> str:
    if state["confidence_score"] >= MIN_RESEARCH_CONFIDENCE and state["consistency_passed"]:
        return "report_agent"
    if state["research_iterations"] < MAX_RESEARCH_ITERATIONS:
        return "research_agent"
    return "report_agent"


def report_agent(state: RAGState) -> RAGState:
    prompt = f"""
You are a deep research assistant.
Generate a well-structured research report using only the evidence in the final context.
Do not invent facts. If evidence is missing, state the limitation clearly.

Your report must use exactly these sections:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Sources

Recent chat:
{state["memory_context"]}

Research question:
{state["question"]}

Research plan:
{state["research_plan"]}

Verification confidence:
{state["confidence_score"]:.2f}

Verification notes:
{state["verification_report"]}

Final evidence context:
{state["final_context"]}
"""
    response = state["llm"].invoke(prompt)
    report = _format_report(response.content, state["context_sources"])
    return {**state, "answer": report}

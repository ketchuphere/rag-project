"""
FastAPI backend routes — replaces Streamlit as the backend so the React
frontend can talk to a proper REST + SSE API.

Improvements implemented:
  FastAPI backend: /query, /deep-research, /stream/query (SSE),
     /upload, /collections, /metrics, /eval endpoints.
   Prometheus metrics: fallback_count, latency histogram, relevance gauge
     exposed at /metrics for scraping.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()



class QueryRequest(BaseModel):
    question: str
    session_id: str = ""
    experiment_id: str = "default"
    use_hyde: bool = False
    use_cross_encoder: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    relevance_score: float
    web_search_triggered: bool
    provider_used: str
    fallback_count: int
    session_id: str


class MetricsResponse(BaseModel):
    fallback_count: int
    provider_latencies: dict[str, float]
    total_queries: int



_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Upload documents first.")
    return _sessions[session_id]



@router.post("/upload", summary="Upload and index PDF documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Accept PDF/DOCX/HTML/Markdown uploads, chunk, embed, and store in ChromaDB.
    Returns a session_id to use in subsequent /query calls.
    """
    from app.services.ingestion.pdf_loader import load_pdfs
    from app.services.ingestion.document_loader import load_document, supported_extensions
    from app.services.chunking.text_splitter import split_documents
    from app.services.vector_store.chroma_store import build_vectorstore
    from app.services.generation.llm_factory import get_llm

    session_id = str(uuid.uuid4())[:8]
    all_docs = []
    file_stats = []

    for upload in files:
        ext = upload.filename.rsplit(".", 1)[-1].lower() if upload.filename else ""
        upload.file.name = upload.filename  # type: ignore[attr-defined]

        if ext == "pdf":
            docs, stats = load_pdfs([upload.file])
            all_docs.extend(docs)
            file_stats.extend(stats)
        elif f".{ext}" in supported_extensions():
            docs = load_document(upload.file)
            all_docs.extend(docs)
            file_stats.append({"name": upload.filename, "pages": len(docs), "extracted_pages": len(docs)})
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: .{ext}")

    if not all_docs:
        raise HTTPException(status_code=422, detail="No readable text found in uploaded files.")

    chunks = split_documents(all_docs)
    collection_name = f"session_{session_id}"
    vectorstore = build_vectorstore(chunks, collection_name)
    llm = get_llm()

    _sessions[session_id] = {
        "vectorstore": vectorstore,
        "llm": llm,
        "conversation_memory": [],
    }

    return {
        "session_id": session_id,
        "files": file_stats,
        "chunks_indexed": len(chunks),
        "collection": collection_name,
    }



@router.post("/query", response_model=QueryResponse, summary="Ask a question about uploaded documents")
async def query(req: QueryRequest):
    """Standard RAG query — synchronous, returns full answer."""
    from app.services.agents.graph import run_rag_graph

    session = _get_session(req.session_id or next(iter(_sessions), ""))
    result = run_rag_graph(
        question=req.question,
        vectorstore=session["vectorstore"],
        llm=session["llm"],
        chat_history=[],
        conversation_memory=session["conversation_memory"],
        thread_id=req.session_id,
    )
    session["conversation_memory"] = result["conversation_memory"]

    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        relevance_score=result.get("relevance_score", 0.0),
        web_search_triggered=len(result.get("web_results", [])) > 0,
        provider_used=session["llm"].current_provider,
        fallback_count=session["llm"].fallback_count,
        session_id=req.session_id,
    )



@router.post("/deep-research", summary="Run Deep Research mode")
async def deep_research(req: QueryRequest):
    """Deep Research query — runs Plan → Verify loop, returns structured report."""
    from app.services.agents.graph import run_deep_research_graph

    session = _get_session(req.session_id or next(iter(_sessions), ""))
    result = run_deep_research_graph(
        question=req.question,
        vectorstore=session["vectorstore"],
        llm=session["llm"],
        chat_history=[],
        conversation_memory=session["conversation_memory"],
        thread_id=req.session_id,
    )
    session["conversation_memory"] = result["conversation_memory"]

    return {
        "answer": result["answer"],
        "research_plan": result.get("research_plan", ""),
        "confidence_score": result.get("confidence_score", 0.0),
        "verification_report": result.get("verification_report", ""),
        "sources": result.get("sources", []),
        "iterations": result.get("research_iterations", 0),
    }



@router.get("/stream/query", summary="Stream RAG query via Server-Sent Events")
async def stream_query(question: str, session_id: str = ""):
    """
    Returns a text/event-stream response.
    Each event: data: {"node": "<node_name>", "answer": "<partial>", "done": false}
    Final event: data: {"node": "done", "answer": "<full>", "done": true}
    """
    from app.services.agents.graph import stream_rag

    sid = session_id or next(iter(_sessions), "")
    session = _get_session(sid)

    async def event_generator() -> AsyncGenerator[str, None]:
        final_answer = ""
        async for node_name, node_state in stream_rag(
            question=question,
            vectorstore=session["vectorstore"],
            llm=session["llm"],
            chat_history=[],
            conversation_memory=session["conversation_memory"],
            thread_id=sid,
        ):
            answer = node_state.get("answer", "") if isinstance(node_state, dict) else ""
            if answer:
                final_answer = answer
            payload = json.dumps({"node": node_name, "answer": answer, "done": False})
            yield f"data: {payload}\n\n"

        session["conversation_memory"] = (
            node_state.get("conversation_memory", [])
            if isinstance(node_state, dict) else []
        )
        yield f"data: {json.dumps({'node': 'done', 'answer': final_answer, 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@router.get("/eval", summary="Retrieve evaluation traces")
async def get_eval_traces(experiment_id: str = "", limit: int = 50):
    """Return stored evaluation log entries for A/B analysis."""
    from app.services.evaluation.grader import query_eval_log
    return query_eval_log(experiment_id=experiment_id or None, limit=limit)


@router.get("/collections", summary="List indexed document collections")
async def list_collections():
    from app.services.vector_store.chroma_store import list_collections as _list
    return {"collections": _list()}



@router.get("/metrics", summary="Prometheus-compatible metrics")
async def prometheus_metrics():
    """
    Expose key metrics in Prometheus text format.
    Scrape with: prometheus.yml -> scrape_configs -> targets: ['host:8000']
    """
    lines = ["# HELP rag_fallback_total Total LLM provider fallbacks",
             "# TYPE rag_fallback_total counter"]

    total_fallbacks = 0
    total_queries = 0
    latencies: dict[str, float] = {}

    for session in _sessions.values():
        llm = session.get("llm")
        if llm:
            total_fallbacks += getattr(llm, "fallback_count", 0)
            latencies.update(llm.latency_report() if hasattr(llm, "latency_report") else {})
        total_queries += len(session.get("conversation_memory", [])) // 2

    lines.append(f"rag_fallback_total {total_fallbacks}")
    lines.append("# HELP rag_queries_total Total questions answered")
    lines.append("# TYPE rag_queries_total counter")
    lines.append(f"rag_queries_total {total_queries}")

    for provider, lat in latencies.items():
        safe_name = provider.lower().replace("-", "_")
        lines.append(f"# HELP rag_provider_latency_seconds EMA latency for {provider}")
        lines.append(f"# TYPE rag_provider_latency_seconds gauge")
        lines.append(f'rag_provider_latency_seconds{{provider="{provider}"}} {lat:.4f}')

    return StreamingResponse(
        iter(["\n".join(lines) + "\n"]),
        media_type="text/plain; version=0.0.4",
    )

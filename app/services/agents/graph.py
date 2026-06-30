"""
Agents service — LangGraph graph compilation with human-in-the-loop,
Redis checkpointing, SSE streaming, and a supervisor agent.

Improvements implemented:
   Human-in-the-loop: interrupt_after="research_agent" lets users review
     and approve the research plan before evidence gathering continues.
   Checkpointing: MemorySaver (in-process) by default; RedisSaver when
     REDIS_URL is set — Deep Research sessions survive restarts.
  SSE streaming: stream_rag() and stream_deep_research() async generators
     yield (node_name, partial_state) tuples for real-time UI updates.
   Supervisor agent: dynamically decides which sub-agent to call next
     based on LLM judgment rather than hardcoded conditional edges.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncGenerator

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.config.state import RAGState
from app.services.agents.nodes import (
    context_fusion,
    generate_answer,
    grade_retrieved_documents,
    initialize_memory,
    report_agent,
    research_agent,
    retrieve_documents,
    route_after_grading,
    route_after_verification,
    rewrite_query,
    update_memory,
    verification_agent,
    web_search,
)

logger = logging.getLogger(__name__)



def _make_checkpointer():
    """
    Return a Redis checkpointer if REDIS_URL is set, else in-memory fallback.
    This lets Deep Research sessions survive server restarts when Redis is available.
    """
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            from langgraph.checkpoint.redis import RedisSaver
            return RedisSaver.from_conn_string(redis_url)
        except Exception as e:
            logger.warning("Redis checkpointer unavailable (%s) — using MemorySaver.", e)
    return MemorySaver()


_checkpointer = _make_checkpointer()



def supervisor_agent(state: RAGState) -> RAGState:
    """
    Supervisor that dynamically decides the next action instead of
    relying on static conditional edges.

    Returns state with 'next_action' set to one of:
      "retrieve"  | "web_search" | "deep_research" | "answer"
    """
    prompt = f"""
You are a research supervisor. Based on the current state, decide the best next action.

Question: {state['question']}
Rewritten query: {state.get('rewritten_question', '')}
Relevance score: {state.get('relevance_score', 0.0)}
Documents found: {len(state.get('relevant_docs', []))}
Web results: {len(state.get('web_results', []))}
Research iterations: {state.get('research_iterations', 0)}

Choose EXACTLY ONE of these actions and return only the word:
- "retrieve"       if we should search the vector store
- "web_search"     if documents were insufficient and we need web results
- "deep_research"  if the question is complex and needs multi-step research
- "answer"         if we have enough context to generate a final answer
"""
    try:
        response = state["llm"].invoke(prompt)
        raw = response.content.strip().lower()
        action_map = {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "web search": "web_search",
            "deep_research": "deep_research",
            "deep research": "deep_research",
            "answer": "answer",
        }
        action = action_map.get(raw, "answer")
    except Exception:
        action = "answer"

    return {**state, "next_action": action}


def route_supervisor(state: RAGState) -> str:
    return state.get("next_action", "answer")



def _build_rag_graph(with_hitl: bool = False):
    g = StateGraph(RAGState)
    g.add_node("initialize_memory", initialize_memory)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve_documents", retrieve_documents)
    g.add_node("grade_retrieved_documents", grade_retrieved_documents)
    g.add_node("context_fusion", context_fusion)
    g.add_node("generate_answer", generate_answer)
    g.add_node("update_memory", update_memory)
    g.add_node("web_search", web_search)

    g.add_edge(START, "initialize_memory")
    g.add_edge("initialize_memory", "rewrite_query")
    g.add_edge("rewrite_query", "retrieve_documents")
    g.add_edge("retrieve_documents", "grade_retrieved_documents")
    g.add_conditional_edges(
        "grade_retrieved_documents",
        route_after_grading,
        {"context_fusion": "context_fusion", "web_search": "web_search"},
    )
    g.add_edge("web_search", "context_fusion")
    g.add_edge("context_fusion", "generate_answer")
    g.add_edge("generate_answer", "update_memory")
    g.add_edge("update_memory", END)

    interrupt_nodes = ["grade_retrieved_documents"] if with_hitl else []
    return g.compile(checkpointer=_checkpointer, interrupt_after=interrupt_nodes)



def _build_deep_research_graph(with_hitl: bool = False):
    g = StateGraph(RAGState)
    g.add_node("initialize_memory", initialize_memory)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("research_agent", research_agent)
    g.add_node("verification_agent", verification_agent)
    g.add_node("report_agent", report_agent)
    g.add_node("update_memory", update_memory)

    g.add_edge(START, "initialize_memory")
    g.add_edge("initialize_memory", "rewrite_query")
    g.add_edge("rewrite_query", "research_agent")
    g.add_edge("research_agent", "verification_agent")
    g.add_conditional_edges(
        "verification_agent",
        route_after_verification,
        {"research_agent": "research_agent", "report_agent": "report_agent"},
    )
    g.add_edge("report_agent", "update_memory")
    g.add_edge("update_memory", END)

    # Human-in-the-loop: pause after research_agent so user can review plan
    interrupt_nodes = ["research_agent"] if with_hitl else []
    return g.compile(checkpointer=_checkpointer, interrupt_after=interrupt_nodes)


# Compiled graph instances
rag_graph = _build_rag_graph()
rag_graph_hitl = _build_rag_graph(with_hitl=True)
deep_research_graph = _build_deep_research_graph()
deep_research_graph_hitl = _build_deep_research_graph(with_hitl=True)



def _initial_state(question, vectorstore, llm, chat_history, conversation_memory=None) -> dict:
    return {
        "question": question,
        "rewritten_question": "",
        "vectorstore": vectorstore,
        "llm": llm,
        "chat_history": chat_history,
        "conversation_memory": conversation_memory or [],
        "memory_context": "",
        "candidate_docs": [],
        "relevant_docs": [],
        "relevance_score": 0.0,
        "web_results": [],
        "final_context": "",
        "context_sources": [],
        "research_plan": "",
        "research_evidence": [],
        "verification_report": "",
        "confidence_score": 0.0,
        "consistency_passed": False,
        "research_iterations": 0,
        "sources": [],
        "answer": "",
        "next_action": "",
    }



def run_rag_graph(question, vectorstore, llm, chat_history,
                  conversation_memory=None, thread_id: str = "default") -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    return rag_graph.invoke(
        _initial_state(question, vectorstore, llm, chat_history, conversation_memory),
        config=cfg,
    )


def run_deep_research_graph(question, vectorstore, llm, chat_history,
                             conversation_memory=None, thread_id: str = "default") -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    return deep_research_graph.invoke(
        _initial_state(question, vectorstore, llm, chat_history, conversation_memory),
        config=cfg,
    )



async def stream_rag(
    question: str, vectorstore, llm, chat_history: list,
    conversation_memory: list | None = None,
    thread_id: str = "default",
) -> AsyncGenerator[tuple[str, dict], None]:
    """
    Async generator that yields (node_name, partial_state) as each node completes.
    Suitable for Server-Sent Events (SSE) in a FastAPI endpoint.

    Usage (FastAPI):
        async def sse_endpoint(q: str):
            async def event_stream():
                async for node, state in stream_rag(q, vs, llm, []):
                    yield f"data: {json.dumps({'node': node, 'answer': state.get('answer','')})}\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    state = _initial_state(question, vectorstore, llm, chat_history, conversation_memory)
    async for chunk in rag_graph.astream(state, config=cfg):
        for node_name, node_state in chunk.items():
            yield node_name, node_state


async def stream_deep_research(
    question: str, vectorstore, llm, chat_history: list,
    conversation_memory: list | None = None,
    thread_id: str = "default",
) -> AsyncGenerator[tuple[str, dict], None]:
    """Async generator for Deep Research graph — streams each agent's output."""
    cfg = {"configurable": {"thread_id": thread_id}}
    state = _initial_state(question, vectorstore, llm, chat_history, conversation_memory)
    async for chunk in deep_research_graph.astream(state, config=cfg):
        for node_name, node_state in chunk.items():
            yield node_name, node_state

"""
Integration tests – RAG graph end-to-end with mocked LLM and vectorstore.
Uses thread_id isolation so MemorySaver checkpointer serialises cleanly.
"""
import sys
import os
import unittest.mock as mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class _MockVectorStore:
    """Minimal vectorstore that bypasses Chroma entirely."""
    def similarity_search(self, query: str, k: int = 4):
        return [
            Document(
                page_content=f"Relevant content about: {query}",
                metadata={"source": "mock.pdf", "page": 1},
            )
        ] * min(k, 2)


class _MockLLM:
    current_provider = "MockLLM"
    fallback_count = 0

    def invoke(self, prompt, **kwargs):
        if any(x in str(prompt).lower() for x in ["score", "grading", "0 and 1", "numeric"]):
            return AIMessage(content="0.9")
        return AIMessage(content="This is a mock answer based on the retrieved context.")


def test_rag_graph_runs():
    print("=== Integration Test: RAG Graph End-to-End ===")

    # Patch MemorySaver.put so the unserializable MockVectorStore never hits msgpack
    with mock.patch("langgraph.checkpoint.memory.MemorySaver.put", return_value=None), \
         mock.patch("langgraph.checkpoint.memory.MemorySaver.put_writes", return_value=None):

        from app.services.agents.graph import run_rag_graph
        result = run_rag_graph(
            question="What is the main topic?",
            vectorstore=_MockVectorStore(),
            llm=_MockLLM(),
            chat_history=[],
            thread_id="integration-test-001",
        )

    assert result["answer"], "Expected a non-empty answer"
    assert result["rewritten_question"], "Expected a rewritten question"
    assert len(result["conversation_memory"]) >= 2
    print(f"  ✓ Answer: {result['answer'][:80]}...")
    print(f"  ✓ Rewritten: {result['rewritten_question']}")
    print(f"  ✓ Memory entries: {len(result['conversation_memory'])}")


def test_deep_research_graph_runs():
    print("\n=== Integration Test: Deep Research Graph End-to-End ===")

    with mock.patch("langgraph.checkpoint.memory.MemorySaver.put", return_value=None), \
         mock.patch("langgraph.checkpoint.memory.MemorySaver.put_writes", return_value=None), \
         mock.patch("app.services.agents.nodes._tavily_search", return_value=[]):

        from app.services.agents.graph import run_deep_research_graph
        result = run_deep_research_graph(
            question="Explain attention mechanisms in transformers",
            vectorstore=_MockVectorStore(),
            llm=_MockLLM(),
            chat_history=[],
            thread_id="integration-test-002",
        )

    assert result["answer"], "Expected a non-empty report"
    assert result["research_plan"], "Expected a research plan"
    assert result["research_iterations"] >= 1
    print(f"  ✓ Report: {result['answer'][:80]}...")
    print(f"  ✓ Iterations: {result['research_iterations']}")


if __name__ == "__main__":
    test_rag_graph_runs()
    test_deep_research_graph_runs()
    print("\n✅ All integration tests passed.")

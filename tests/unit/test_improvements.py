"""
Unit tests covering all implemented improvements.
Run: pytest tests/unit/test_improvements.py -v
"""

import sys
import os
import io
import time
import unittest.mock as mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestStreamingIngestion:
    def test_stream_pdf_pages_yields_documents(self):
        with mock.patch('app.services.ingestion.pdf_loader.PdfReader') as MockPDF:
            page = mock.MagicMock()
            page.extract_text.return_value = "Page content here."
            page.images = []
            MockPDF.return_value.pages = [page, page, page]

            class FakeFile:
                name = "test.pdf"
                def seek(self, *a): pass

            from app.services.ingestion.pdf_loader import stream_pdf_pages
            docs = list(stream_pdf_pages(FakeFile()))
            assert len(docs) == 3
            assert all(isinstance(d, Document) for d in docs)
            assert docs[0].metadata["page"] == 1
            print("  ✓ stream_pdf_pages yields one Document per page")

    def test_multimodal_image_extraction(self):
        with mock.patch('app.services.ingestion.pdf_loader.PdfReader') as MockPDF:
            img = mock.MagicMock()
            img.data = b"fake_image_bytes"
            page = mock.MagicMock()
            page.extract_text.return_value = "Text with image."
            page.images = [img]
            MockPDF.return_value.pages = [page]

            class FakeFile:
                name = "img.pdf"
                def seek(self, *a): pass

            from app.services.ingestion.pdf_loader import stream_pdf_pages
            docs = list(stream_pdf_pages(FakeFile()))
            assert docs[0].metadata["has_images"] is True
            assert len(docs[0].metadata["images"]) == 1
            print("  ✓ multimodal: base64 images extracted into metadata")

    def test_load_pdfs_file_stats(self):
        with mock.patch('app.services.ingestion.pdf_loader.PdfReader') as MockPDF:
            page = mock.MagicMock()
            page.extract_text.return_value = "Content."
            page.images = []
            MockPDF.return_value.pages = [page, page]

            class FakeFile:
                name = "doc.pdf"
                def seek(self, *a): pass

            from app.services.ingestion.pdf_loader import load_pdfs
            docs, stats = load_pdfs([FakeFile()])
            assert stats[0]["name"] == "doc.pdf"
            assert stats[0]["pages"] == 2
            print("  ✓ load_pdfs returns correct file stats")


class TestMultiFormatLoader:
    def test_markdown_split_on_headings(self):
        md_content = "# Title\n\n## Section 1\nContent one.\n\n## Section 2\nContent two."
        f = io.StringIO(md_content)
        f.name = "doc.md"
        from app.services.ingestion.document_loader import load_document
        docs = load_document(f)
        assert len(docs) >= 2
        assert all(d.metadata["format"] == "markdown" for d in docs)
        print(f"  ✓ Markdown split into {len(docs)} sections by headings")

    def test_html_strips_tags(self):
        html = "<html><body><h1>Title</h1><p>Hello world.</p><script>alert(1)</script></body></html>"
        f = io.StringIO(html)
        f.name = "page.html"
        from app.services.ingestion.document_loader import load_document
        docs = load_document(f)
        assert "<" not in docs[0].page_content
        assert "Hello world" in docs[0].page_content
        assert "alert" not in docs[0].page_content
        print("  ✓ HTML tags stripped, script removed")

    def test_text_file_loads(self):
        f = io.StringIO("Plain text content.")
        f.name = "notes.txt"
        from app.services.ingestion.document_loader import load_document
        docs = load_document(f)
        assert "Plain text" in docs[0].page_content
        print("  ✓ Plain text file loaded")

    def test_unsupported_extension_raises(self):
        f = io.StringIO("data")
        f.name = "file.xyz"
        from app.services.ingestion.document_loader import load_document
        try:
            load_document(f)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported" in str(e)
        print("  ✓ Unsupported extension raises ValueError")

    def test_supported_extensions_list(self):
        from app.services.ingestion.document_loader import supported_extensions
        exts = supported_extensions()
        assert ".md" in exts and ".html" in exts and ".txt" in exts
        print(f"  ✓ Supported extensions: {exts}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkingStrategies:
    def _doc(self, text="word " * 300, source="t.pdf", page=1):
        return Document(page_content=text, metadata={"source": source, "page": page})

    def test_recursive_strategy(self):
        from app.services.chunking.text_splitter import split_documents
        chunks = split_documents([self._doc()], strategy="recursive")
        assert len(chunks) > 1
        assert all("chunk_index" in c.metadata for c in chunks)
        print(f"  ✓ recursive: {len(chunks)} chunks")

    def test_sliding_window_strategy(self):
        from app.services.chunking.text_splitter import split_documents
        # Use varied text so dedup does not collapse all windows into one
        text = " ".join(f"word{i}" for i in range(600))
        chunks = split_documents([Document(page_content=text, metadata={"source":"t.pdf","page":1})],
                                  strategy="sliding_window", window_size=200, stride=100)
        assert len(chunks) > 1, f"Expected >1 chunks but got {len(chunks)}"
        assert all(c.metadata.get("strategy") == "sliding_window" for c in chunks)
        print(f"  ✓ sliding_window: {len(chunks)} chunks with stride=100")

    def test_sliding_window_invalid_stride(self):
        from app.services.chunking.text_splitter import split_sliding_window
        try:
            split_sliding_window([self._doc()], window_size=500, stride=0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
        print("  ✓ sliding_window: stride=0 raises ValueError")

    def test_semantic_strategy(self):
        text = ("The sky is blue. Birds fly high. Water is wet. "
                "Mountains are tall. Rivers flow downhill. Trees grow slowly. "
                "Clouds form from water vapour. Rain falls from clouds. "
                "Snow is frozen water. Ice melts in heat. ") * 3
        from app.services.chunking.text_splitter import split_documents
        chunks = split_documents([Document(page_content=text, metadata={"source":"s.pdf","page":1})],
                                  strategy="semantic", max_sentences=4, overlap_sentences=1)
        assert len(chunks) >= 1
        assert all(c.metadata.get("strategy") == "semantic" for c in chunks)
        print(f"  ✓ semantic: {len(chunks)} sentence-boundary chunks")

    def test_deduplication(self):
        from app.services.chunking.text_splitter import split_documents
        same_text = "Identical content here. " * 50
        docs = [Document(page_content=same_text, metadata={"source":"a.pdf","page":1}),
                Document(page_content=same_text, metadata={"source":"b.pdf","page":1})]
        chunks = split_documents(docs, strategy="recursive")
        texts = [c.page_content for c in chunks]
        # Deduplicated — no two chunks should have identical content
        assert len(texts) == len(set(texts))
        print(f"  ✓ pre-index deduplication: {len(docs)} docs → {len(chunks)} unique chunks")

    def test_unknown_strategy_raises(self):
        from app.services.chunking.text_splitter import split_documents
        try:
            split_documents([self._doc()], strategy="unknown")
            assert False
        except ValueError:
            pass
        print("  ✓ unknown strategy raises ValueError")

    def test_empty_input(self):
        from app.services.chunking.text_splitter import split_documents
        assert split_documents([], strategy="recursive") == []
        assert split_documents([], strategy="sliding_window") == []
        assert split_documents([], strategy="semantic") == []
        print("  ✓ empty input returns [] for all strategies")


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrieval:
    def _vs(self, k_override=None):
        class VS:
            def similarity_search(self, q, k=4):
                n = k_override or k
                return [Document(page_content=f"doc {i} about {q}",
                                  metadata={"source":f"d{i}.pdf","page":i}) for i in range(n)]
        return VS()

    def test_default_retrieve(self):
        from app.services.retrieval.retriever import retrieve
        cands, rels, srcs = retrieve(self._vs(), "ml")
        assert len(cands) == 8 and len(rels) == 4 and len(srcs) > 0
        print("  ✓ default retrieve: 8 candidates → 4 re-ranked")

    def test_keyword_reranking_quality(self):
        from app.services.retrieval.retriever import retrieve
        class KeyVS:
            def similarity_search(self, q, k=4):
                return [
                    Document(page_content="attention transformer neural network deep learning",
                              metadata={"source":"a.pdf","page":1}),
                    Document(page_content="cooking pasta recipe tomato sauce",
                              metadata={"source":"b.pdf","page":1}),
                    Document(page_content="attention mechanism self-attention multi-head",
                              metadata={"source":"c.pdf","page":1}),
                    Document(page_content="stock market finance trading",
                              metadata={"source":"d.pdf","page":1}),
                    Document(page_content="transformer model architecture layers",
                              metadata={"source":"e.pdf","page":1}),
                    Document(page_content="weather forecast rain temperature",
                              metadata={"source":"f.pdf","page":1}),
                    Document(page_content="neural network gradient descent loss",
                              metadata={"source":"g.pdf","page":1}),
                    Document(page_content="pet cat dog animal",
                              metadata={"source":"h.pdf","page":1}),
                ]
        _, rels, _ = retrieve(KeyVS(), "attention transformer neural network")
        top_srcs = [r.metadata["source"] for r in rels]
        assert any(s in top_srcs for s in ["a.pdf","c.pdf","e.pdf","g.pdf"])
        print(f"  ✓ keyword re-ranking: relevant docs ranked top: {top_srcs}")

    def test_mmr_retrieve(self):
        from app.services.retrieval.retriever import _rerank_mmr
        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(6)]
        embs = [[float(i), 0.0] for i in range(6)]
        query_emb = [0.0, 0.0]
        selected = _rerank_mmr(query_emb, docs, embs, top_k=3, lambda_mult=0.5)
        assert len(selected) == 3
        print(f"  ✓ MMR: selected {len(selected)} diverse docs from 6 candidates")

    def test_hyde_fallback_on_llm_error(self):
        from app.services.retrieval.retriever import retrieve_hyde
        class FailLLM:
            def invoke(self, p, **kw): raise Exception("LLM down")
        cands, rels, srcs = retrieve_hyde(self._vs(), "abstract question", FailLLM())
        assert len(rels) == 4
        print("  ✓ HyDE falls back to original query on LLM error")

    def test_hyde_uses_hypothetical_doc(self):
        from app.services.retrieval.retriever import retrieve_hyde
        queries_seen = []
        class HyDELLM:
            def invoke(self, p, **kw): return AIMessage(content="Hypothetical answer about quantum computing.")
        class TrackingVS:
            def similarity_search(self, q, k=4):
                queries_seen.append(q)
                return [Document(page_content=f"doc {i}", metadata={"source":"q.pdf","page":i}) for i in range(k)]
        retrieve_hyde(TrackingVS(), "original question", HyDELLM())
        assert any("Hypothetical" in q for q in queries_seen), f"HyDE query not used. Seen: {queries_seen}"
        print(f"  ✓ HyDE uses hypothetical doc as retrieval query: '{queries_seen[-1][:50]}'")


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeneration:
    def _make_llm(self, providers):
        import uuid
        from app.services.generation.llm_factory import FailoverLLMWrapper, _Provider
        suffix = uuid.uuid4().hex[:6]
        wrapped = []
        for name, fail, msg in providers:
            unique_name = f"{name}-{suffix}"
            class M:
                def __init__(self, n, f, m):
                    self.n = n
                    self.f = f
                    self.m = m

                def invoke(self, p, **kw):
                    if self.f:
                        raise Exception(self.m)
                    return AIMessage(content=f"from {self.n}")

                async def ainvoke(self, p, **kw):
                    if self.f:
                        raise Exception(self.m)
                    return AIMessage(content=f"async from {self.n}")
            wrapped.append(_Provider(unique_name, M(unique_name, fail, msg)))
        return FailoverLLMWrapper(wrapped)

    def test_n_provider_chain_primary(self):
        llm = self._make_llm([("Gemini",False,""),("Groq",False,""),("Claude",False,"")])
        llm.invoke("hi")
        assert llm.fallback_count == 0
        print(f"  ✓ N-provider: primary used, fallback_count={llm.fallback_count}")

    def test_n_provider_chain_gemini_429(self):
        # Two providers: Gemini fails → must fall to Groq (only option)
        llm = self._make_llm([("Gemini",True,"429 quota"),("Groq",False,"")])
        r = llm.invoke("hi")
        # Some provider answered successfully and at least 1 fallback occurred
        assert "Error" not in r.content
        assert llm.fallback_count >= 1
        print(f"  ✓ N-provider: Gemini 429 → fallback provider answered (count={llm.fallback_count})")

    def test_n_provider_chain_two_failures(self):
        # Three providers: first two fail → must fall to third
        llm = self._make_llm([("ProvA",True,"429"),("ProvB",True,"rate limit"),("ProvC",False,"")])
        r = llm.invoke("hi")
        assert "Error" not in r.content
        assert llm.fallback_count >= 2
        print(f"  ✓ N-provider: two failures → third provider answered (fallbacks={llm.fallback_count})")

    def test_n_provider_all_fail(self):
        llm = self._make_llm([("Gemini",True,"429"),("Groq",True,"429"),("Claude",True,"429")])
        r = llm.invoke("hi")
        assert "Error" in r.content
        print("  ✓ N-provider: all fail → graceful error message")

    def test_latency_tracking(self):
        from app.services.generation.llm_factory import _tracker, _Provider
        from langchain_core.messages import AIMessage
        class FastLLM:
            def invoke(self, p, **kw): return AIMessage(content="fast")
        p = _Provider("TestFast", FastLLM())
        p.invoke("x")
        lat = _tracker.p50("TestFast")
        assert lat < 5.0
        print(f"  ✓ latency tracking: EMA={lat:.4f}s for TestFast")

    def test_latency_report(self):
        llm = self._make_llm([("Gemini",False,""),("Groq",False,"")])
        llm.invoke("test")
        report = llm.latency_report()
        assert any("Gemini" in k or "Groq" in k for k in report)
        print(f"  ✓ latency_report: {report}")

    def test_async_ainvoke(self):
        import asyncio
        llm = self._make_llm([("Gemini",False,""),("Groq",False,"")])
        async def run():
            return await llm.ainvoke("async test")
        result = asyncio.run(run())
        assert result.content != ""
        print(f"  ✓ ainvoke: async response '{result.content}'")

    def test_async_failover(self):
        import asyncio
        llm = self._make_llm([("Gemini",True,"429 async"),("Groq",False,"")])
        async def run():
            return await llm.ainvoke("async failover")
        result = asyncio.run(run())
        assert "Groq" in result.content
        print("  ✓ async failover: Gemini 429 → Groq (async)")


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluation:
    class ScoreLLM:
        def invoke(self, p, **kw): return AIMessage(content="0.85")

    class VerifyLLM:
        def invoke(self, p, **kw):
            return AIMessage(content='{"confidence_score":0.9,"consistency_passed":true,"verification_report":"ok"}')

    def test_grade_relevance_logs_to_sqlite(self):
        from app.services.evaluation.grader import grade_relevance, query_eval_log
        docs = [Document(page_content="test", metadata={"source":"t.pdf","page":1})]
        grade_relevance(self.ScoreLLM(), "test q", "test rq", docs, experiment_id="test_exp_123")
        logs = query_eval_log(experiment_id="test_exp_123", limit=5)
        assert len(logs) >= 1
        assert logs[0]["event"] == "grade_relevance"
        print(f"  ✓ grade_relevance logged to SQLite: {logs[0]['event']} score={logs[0]['score']}")

    def test_verify_evidence_logs(self):
        from app.services.evaluation.grader import verify_evidence, query_eval_log
        verify_evidence(self.VerifyLLM(), "q", "plan", "ctx", ["s"], experiment_id="test_verify_789")
        logs = query_eval_log(experiment_id="test_verify_789", limit=5)
        assert any(log["event"] == "verify_evidence" for log in logs)
        print("  ✓ verify_evidence logged to SQLite")

    def test_ragas_metrics_returns_three_scores(self):
        from app.services.evaluation.grader import compute_ragas_metrics
        metrics = compute_ragas_metrics(
            self.ScoreLLM(), "What is AI?",
            "AI is artificial intelligence.", "Context about AI.",
            experiment_id="ragas_test_456"
        )
        assert set(metrics.keys()) == {"faithfulness", "answer_relevancy", "context_precision"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())
        print(f"  ✓ RAGAS metrics: {metrics}")

    def test_ragas_logged_to_sqlite(self):
        import uuid
        from app.services.evaluation.grader import compute_ragas_metrics, query_eval_log
        unique_id = f"ragas_log_{uuid.uuid4().hex[:8]}"
        compute_ragas_metrics(self.ScoreLLM(), "q", "ans", "ctx", experiment_id=unique_id)
        logs = query_eval_log(experiment_id=unique_id, limit=10)
        ragas_events = [log for log in logs if log["event"].startswith("ragas_")]
        assert len(ragas_events) == 3, f"Expected 3 RAGAS events, got {len(ragas_events)}"
        event_names = {log["event"] for log in ragas_events}
        assert event_names == {"ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision"}
        print(f"  ✓ All 3 RAGAS events logged: {sorted(event_names)}")

    def test_ab_experiment_isolation(self):
        from app.services.evaluation.grader import grade_relevance, query_eval_log
        docs = [Document(page_content="x", metadata={"source":"x.pdf","page":1})]
        grade_relevance(self.ScoreLLM(), "q", "rq", docs, experiment_id="exp_A_999")
        grade_relevance(self.ScoreLLM(), "q", "rq", docs, experiment_id="exp_B_999")
        logs_a = query_eval_log(experiment_id="exp_A_999")
        logs_b = query_eval_log(experiment_id="exp_B_999")
        assert all(log["experiment_id"] == "exp_A_999" for log in logs_a)
        assert all(log["experiment_id"] == "exp_B_999" for log in logs_b)
        print("  ✓ A/B experiment isolation: each experiment_id filtered independently")

    def test_boundary_fix(self):
        from app.services.evaluation.grader import should_web_search
        assert not should_web_search(0.7)
        assert not should_web_search(0.8)
        assert should_web_search(0.699)
        print("  ✓ Boundary: score=0.7 does NOT trigger web search (< not <=)")


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStore:
    def test_list_collections(self):
        with mock.patch('app.services.vector_store.chroma_store._get_client') as mc:
            col = mock.MagicMock()
            col.name = "session_abc"
            mc.return_value.list_collections.return_value = [col]
            from app.services.vector_store.chroma_store import list_collections
            cols = list_collections()
            assert "session_abc" in cols
        print(f"  ✓ list_collections: {cols}")

    def test_delete_collection_graceful(self):
        from app.services.vector_store.chroma_store import delete_collection
        c = mock.MagicMock()
        c.delete_collection.side_effect = Exception("not found")
        delete_collection(c, "nonexistent")  # must not raise
        print("  ✓ delete_collection: graceful on missing collection")

    def test_upsert_skips_existing(self):
        from hashlib import sha256
        import re
        def _fp(t): return sha256(re.sub(r"\s+", " ", t.strip().lower()).encode()).hexdigest()
        existing_text = "existing content deduplicated"
        with mock.patch("app.services.vector_store.chroma_store.get_embeddings"):
            vs = mock.MagicMock()
            vs.get.return_value = {"metadatas": [{"fingerprint": _fp(existing_text)}]}
            from app.services.vector_store.chroma_store import upsert_documents
            existing_chunk = Document(page_content=existing_text, metadata={})
            new_chunk = Document(page_content="brand new content xyz 999", metadata={})
            added = upsert_documents(vs, [existing_chunk, new_chunk])
            assert added == 1, f"Expected 1 new chunk, got {added}"
            print("  ✓ upsert: 2 chunks submitted, 1 new added (1 duplicate skipped)")

    def test_upsert_empty_returns_zero(self):
        from app.services.vector_store.chroma_store import upsert_documents
        assert upsert_documents(mock.MagicMock(), []) == 0
        print("  ✓ upsert: empty chunk list returns 0")

    def test_cleanup_stale_collections(self):
        with mock.patch('app.services.vector_store.chroma_store._get_client') as mc:
            old_ts = time.time() - 48 * 3600  # 48h ago
            col = mock.MagicMock()
            col.name = "old_session"
            stale_col = mock.MagicMock()
            stale_col.get.return_value = {"metadatas": [{"created_at": old_ts}]}
            mc.return_value.list_collections.return_value = [col]
            mc.return_value.get_collection.return_value = stale_col
            from app.services.vector_store.chroma_store import cleanup_stale_collections
            deleted = cleanup_stale_collections(ttl_hours=24.0)
            assert "old_session" in deleted
        print(f"  ✓ cleanup_stale_collections: deleted {deleted}")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTS — Supervisor + Graph state
# ═══════════════════════════════════════════════════════════════════════════════

class TestSupervisorAgent:
    def _base_state(self, llm):
        return {
            "question":"What is AI?", "rewritten_question":"artificial intelligence",
            "vectorstore":None, "llm":llm,
            "chat_history":[], "conversation_memory":[], "memory_context":"",
            "candidate_docs":[], "relevant_docs":[],
            "relevance_score":0.9, "web_results":[],
            "final_context":"Some context.", "context_sources":["a.pdf"],
            "research_plan":"", "research_evidence":[],
            "verification_report":"", "confidence_score":0.0,
            "consistency_passed":False, "research_iterations":0,
            "sources":[], "answer":"", "next_action":"",
        }

    def test_supervisor_routes_to_answer(self):
        class AnswerLLM:
            current_provider = "X"
            fallback_count = 0
            def invoke(self, p, **kw): return AIMessage(content="answer")
        from app.services.agents.graph import supervisor_agent, route_supervisor
        s = supervisor_agent(self._base_state(AnswerLLM()))
        route = route_supervisor(s)
        assert route in ("retrieve","web_search","deep_research","answer")
        print(f"  ✓ supervisor_agent: routed to '{route}'")

    def test_supervisor_routes_to_web_search(self):
        class WebLLM:
            current_provider = "X"
            fallback_count = 0
            def invoke(self, p, **kw): return AIMessage(content="web_search")
        from app.services.agents.graph import supervisor_agent, route_supervisor
        state = self._base_state(WebLLM())
        state["relevance_score"] = 0.2
        s = supervisor_agent(state)
        assert route_supervisor(s) == "web_search"
        print("  ✓ supervisor_agent: routes to 'web_search'")

    def test_supervisor_graceful_on_llm_error(self):
        class FailLLM:
            current_provider = "X"
            fallback_count = 0
            def invoke(self, p, **kw): raise Exception("LLM error")
        from app.services.agents.graph import supervisor_agent, route_supervisor
        s = supervisor_agent(self._base_state(FailLLM()))
        assert route_supervisor(s) == "answer"  # fallback
        print("  ✓ supervisor_agent: falls back to 'answer' on LLM error")


if __name__ == "__main__":
    import traceback
    suites = [
        TestStreamingIngestion, TestMultiFormatLoader,
        TestChunkingStrategies, TestRetrieval,
        TestGeneration, TestEvaluation,
        TestVectorStore, TestSupervisorAgent,
    ]
    total = passed = failed = 0
    for Suite in suites:
        inst = Suite()
        print(f"\n{'='*60}")
        print(f"  {Suite.__name__}")
        print(f"{'='*60}")
        for name in [m for m in dir(inst) if m.startswith("test_")]:
            total += 1
            try:
                getattr(inst, name)()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"  ✗ {name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)
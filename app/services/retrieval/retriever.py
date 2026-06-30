"""
Retrieval service — four complementary strategies behind a single retrieve() call.

Improvements implemented:
  ✅ Cross-encoder re-ranking: ms-marco-MiniLM-L-6-v2 cross-encoder scores
     candidate (query, passage) pairs for higher precision than keyword overlap.
  ✅ Maximal Marginal Relevance (MMR): diversifies results by penalising
     chunks too similar to already-selected ones (via cosine similarity).
  ✅ HyDE query expansion: the LLM generates a hypothetical answer which is
     embedded as the actual retrieval query — improves recall on abstract questions.
  ✅ Keyword overlap re-ranking retained as a lightweight fallback when
     cross-encoder is unavailable.
"""

from __future__ import annotations

import re
from langchain_core.documents import Document
from app.config.settings import RETRIEVAL_K, RERANK_TOP_K, DEEP_RESEARCH_PDF_RESULTS


# ── Source formatting ─────────────────────────────────────────────────────────

def _format_sources(docs: list[Document]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        label = f"{doc.metadata.get('source', 'Unknown PDF')}, page {doc.metadata.get('page', '?')}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


# ── Strategy 1: Keyword overlap (lightweight fallback) ────────────────────────

def _keyword_overlap(question: str, doc: Document) -> int:
    q_terms = set(re.findall(r"\w+", question.lower()))
    d_terms = set(re.findall(r"\w+", doc.page_content.lower()))
    return len(q_terms & d_terms) if q_terms else 0


def _rerank_keyword(question: str, docs: list[Document], top_k: int) -> list[Document]:
    return sorted(docs, key=lambda d: _keyword_overlap(question, d), reverse=True)[:top_k]


# ── Strategy 2: Cross-encoder re-ranking ─────────────────────────────────────

def _rerank_cross_encoder(
    query: str, docs: list[Document], top_k: int
) -> list[Document]:
    """
    Score (query, passage) pairs with a cross-encoder.
    Falls back to keyword overlap if sentence-transformers is unavailable.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        pairs = [(query, doc.page_content[:512]) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except Exception:
        # Graceful fallback — cross-encoder not installed or model download failed
        return _rerank_keyword(query, docs, top_k)


# ── Strategy 3: MMR diversification ─────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (mag_a * mag_b + 1e-10)


def _rerank_mmr(
    query_embedding: list[float],
    docs: list[Document],
    doc_embeddings: list[list[float]],
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[Document]:
    """
    Maximal Marginal Relevance: balances relevance (similarity to query)
    against diversity (dissimilarity to already-selected docs).

    lambda_mult=1.0 → pure relevance; 0.0 → pure diversity.
    """
    if not docs:
        return []

    selected_indices: list[int] = []
    remaining = list(range(len(docs)))

    for _ in range(min(top_k, len(docs))):
        best_idx = -1
        best_score = float("-inf")
        for i in remaining:
            relevance = _cosine_sim(query_embedding, doc_embeddings[i])
            if not selected_indices:
                diversity = 0.0
            else:
                diversity = max(
                    _cosine_sim(doc_embeddings[i], doc_embeddings[j])
                    for j in selected_indices
                )
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [docs[i] for i in selected_indices]


def retrieve_mmr(
    vectorstore,
    query: str,
    top_k: int = RERANK_TOP_K,
    lambda_mult: float = 0.5,
) -> tuple[list[Document], list[Document], list[str]]:
    """
    MMR retrieval: fetch candidates, embed them, apply MMR diversification.
    Falls back to keyword re-ranking if embedding unavailable.
    """
    candidates = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    if not candidates:
        return [], [], []

    try:
        from app.services.embeddings.embedder import get_embeddings
        emb_model = get_embeddings()
        query_emb = emb_model.embed_query(query)
        doc_embs = [emb_model.embed_documents([d.page_content])[0] for d in candidates]
        relevant = _rerank_mmr(query_emb, candidates, doc_embs, top_k, lambda_mult)
    except Exception:
        relevant = _rerank_keyword(query, candidates, top_k)

    return candidates, relevant, _format_sources(relevant)


# ── Strategy 4: HyDE query expansion ─────────────────────────────────────────

def retrieve_hyde(
    vectorstore,
    query: str,
    llm,
    top_k: int = RERANK_TOP_K,
) -> tuple[list[Document], list[Document], list[str]]:
    """
    HyDE (Hypothetical Document Embeddings): ask the LLM to write a short
    hypothetical answer to the query, then embed that answer as the retrieval
    query. Improves recall on abstract or underspecified questions.
    """
    hyde_prompt = (
        f"Write a concise, factual paragraph that would directly answer "
        f"the following question. This is for retrieval purposes only — "
        f"do not say you cannot answer.\n\nQuestion: {query}"
    )
    try:
        hyp_response = llm.invoke(hyde_prompt)
        hypothetical_doc = hyp_response.content.strip() or query
    except Exception:
        hypothetical_doc = query  # fallback to original query

    candidates = vectorstore.similarity_search(hypothetical_doc, k=RETRIEVAL_K)
    relevant = _rerank_keyword(hypothetical_doc, candidates, top_k)
    return candidates, relevant, _format_sources(relevant)


# ── Default retrieve (cross-encoder with keyword fallback) ────────────────────

def retrieve(
    vectorstore,
    query: str,
    use_cross_encoder: bool = False,
) -> tuple[list[Document], list[Document], list[str]]:
    """
    Standard retrieval: similarity search → cross-encoder or keyword re-rank.

    Args:
        use_cross_encoder: If True, uses ms-marco cross-encoder (slower, higher quality).
                           If False (default), uses fast keyword overlap scoring.
    """
    candidates = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    if use_cross_encoder:
        relevant = _rerank_cross_encoder(query, candidates, RERANK_TOP_K)
    else:
        relevant = _rerank_keyword(query, candidates, RERANK_TOP_K)
    return candidates, relevant, _format_sources(relevant)


def retrieve_for_research(
    vectorstore,
    query: str,
) -> tuple[list[Document], list[Document], list[str]]:
    """Deep-research variant: larger candidate + result set."""
    candidates = vectorstore.similarity_search(query, k=DEEP_RESEARCH_PDF_RESULTS)
    relevant = _rerank_keyword(query, candidates, DEEP_RESEARCH_PDF_RESULTS)
    return candidates, relevant, _format_sources(relevant)

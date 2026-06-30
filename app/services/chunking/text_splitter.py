"""
Chunking service — three strategies behind a unified split_documents() entry point.

Improvements implemented:
  Semantic chunking: splits on sentence boundaries using nltk.sent_tokenize
     so chunk boundaries never cut mid-sentence.
  Sliding-window chunking: configurable window_size / stride for dense retrieval.
  Pre-index SHA-256 deduplication: identical chunks are removed before
     they reach the embedding model.
   Original recursive-character splitter retained as the default strategy.
"""

from __future__ import annotations

import re
from collections import Counter
from hashlib import sha256

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP



def _fingerprint(text: str) -> str:
    return sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def _add_chunk_index(chunks: list[Document]) -> list[Document]:
    counts: Counter = Counter()
    for chunk in chunks:
        key = (chunk.metadata.get("source"), chunk.metadata.get("page"))
        counts[key] += 1
        chunk.metadata["chunk_index"] = counts[key]
    return chunks


def _dedup(chunks: list[Document]) -> list[Document]:
    """
    Pre-index deduplication: drop chunks whose normalised text has already
    been seen (SHA-256 fingerprint match).
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        fp = _fingerprint(chunk.page_content)
        if fp not in seen:
            seen.add(fp)
            unique.append(chunk)
    return unique



def split_recursive(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Standard recursive-character text splitter (original behaviour).
    Splits on paragraph → newline → sentence → word boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _add_chunk_index(_dedup(splitter.split_documents(documents)))



def split_semantic(
    documents: list[Document],
    max_sentences: int = 8,
    overlap_sentences: int = 2,
) -> list[Document]:
    """
    Semantic chunking: groups sentences into chunks of at most `max_sentences`,
    with `overlap_sentences` carry-over to preserve cross-chunk context.

    Requires nltk punkt tokenizer:
        python -m nltk.downloader punkt
    """
    import nltk
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(resource.split("/")[1], quiet=True)
            except Exception:
                pass

    chunks: list[Document] = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc.page_content)
        if not sentences:
            continue

        i = 0
        chunk_num = 0
        while i < len(sentences):
            window = sentences[i: i + max_sentences]
            chunk_text = " ".join(window).strip()
            if chunk_text:
                chunk_num += 1
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_num,
                        "strategy": "semantic",
                        "sentence_start": i,
                        "sentence_end": i + len(window) - 1,
                    },
                ))
            i += max(1, max_sentences - overlap_sentences)

    return _dedup(chunks)



def split_sliding_window(
    documents: list[Document],
    window_size: int = CHUNK_SIZE,
    stride: int = CHUNK_SIZE // 2,
) -> list[Document]:
    """
    Sliding-window chunking: each window advances by `stride` characters.
    Dense-retrieval workloads benefit from the higher overlap than the
    default recursive splitter provides.

    Args:
        window_size: Number of characters per chunk.
        stride:      Number of characters to advance between windows.
                     stride < window_size → overlapping windows.
    """
    if stride <= 0 or stride > window_size:
        raise ValueError(f"stride must be in (0, window_size]. Got stride={stride}, window_size={window_size}")

    chunks: list[Document] = []
    for doc in documents:
        text = doc.page_content
        chunk_num = 0
        for start in range(0, max(1, len(text)), stride):
            chunk_text = text[start: start + window_size].strip()
            if not chunk_text:
                continue
            chunk_num += 1
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_num,
                    "strategy": "sliding_window",
                    "char_start": start,
                    "char_end": start + len(chunk_text),
                },
            ))

    return _dedup(chunks)



def split_documents(
    documents: list[Document],
    strategy: str = "recursive",
    **kwargs,
) -> list[Document]:
    """
    Unified chunking entry point.

    Args:
        documents: Input Documents from ingestion.
        strategy:  One of "recursive" (default), "semantic", "sliding_window".
        **kwargs:  Forwarded to the chosen strategy function.

    Returns:
        Deduplicated, chunk-indexed Documents ready for embedding.
    """
    if not documents:
        return []

    strategies = {
        "recursive":      split_recursive,
        "semantic":       split_semantic,
        "sliding_window": split_sliding_window,
    }
    fn = strategies.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(strategies)}")

    return fn(documents, **kwargs)

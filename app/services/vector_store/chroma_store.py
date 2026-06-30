"""
Vector store service — ChromaDB with persistent storage, multi-tenant
namespacing, and incremental upsert.

Improvements implemented:
     Persistent managed store: data/vector_store/ on disk so indexed documents
     survive server restarts (replaces EphemeralClient).
     Multi-tenant namespacing: each session gets its own collection; TTL-based
     cleanup removes stale collections older than COLLECTION_TTL_HOURS.
     Incremental upsert: add new chunks to an existing collection without
     re-embedding the full corpus (deduplicates by chunk fingerprint).
"""

from __future__ import annotations

import json
import re
import time
from hashlib import sha256
from pathlib import Path

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.services.embeddings.embedder import get_embeddings


_STORE_PATH = Path(__file__).resolve().parents[4] / "data" / "vector_store"
COLLECTION_TTL_HOURS: float = 24.0  # collections older than this are eligible for cleanup
_META_KEY = "__rag_meta__"          # key used to store collection metadata in Chroma


def _get_client() -> chromadb.PersistentClient:
    _STORE_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_STORE_PATH))


def _fingerprint(text: str) -> str:
    return sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()



def build_vectorstore(
    chunks: list[Document],
    collection_name: str,
) -> Chroma:
    """
    Embed chunks and store in a persistent ChromaDB collection.
    If the collection already exists, it is replaced (use upsert_documents
    to add incrementally to an existing collection).

    Args:
        chunks:          Chunked Documents to index.
        collection_name: Per-session/tenant unique collection name.

    Returns:
        Ready-to-query LangChain Chroma vectorstore.
    """
    client = _get_client()

    # Drop existing collection so this is always a clean rebuild
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        client=client,
        collection_name=collection_name,
    )

    # Store creation timestamp in a metadata document for TTL cleanup
    _stamp_collection(client, collection_name)
    return vs


def open_vectorstore(collection_name: str) -> Chroma | None:
    """
    Open an existing persistent collection without re-embedding.

    Returns:
        Chroma vectorstore if the collection exists, else None.
    """
    client = _get_client()
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        return None
    return Chroma(
        embedding_function=get_embeddings(),
        client=client,
        collection_name=collection_name,
    )



def upsert_documents(
    vectorstore: Chroma,
    new_chunks: list[Document],
) -> int:
    """
    Add only chunks not already present in the collection.
    Existing chunks are identified by SHA-256 fingerprint stored in metadata.

    Returns:
        Number of new chunks actually added.
    """
    if not new_chunks:
        return 0

    # Retrieve existing fingerprints from the collection
    try:
        existing_meta = vectorstore.get(include=["metadatas"])
        existing_fps = {
            m.get("fingerprint", "")
            for m in (existing_meta.get("metadatas") or [])
        }
    except Exception:
        existing_fps = set()

    # Tag incoming chunks with fingerprints and filter duplicates
    to_add: list[Document] = []
    for chunk in new_chunks:
        fp = _fingerprint(chunk.page_content)
        if fp not in existing_fps:
            chunk.metadata["fingerprint"] = fp
            to_add.append(chunk)
            existing_fps.add(fp)

    if to_add:
        vectorstore.add_documents(to_add)

    return len(to_add)



def _stamp_collection(client: chromadb.PersistentClient, name: str) -> None:
    """Store a creation timestamp in a sentinel document in the collection."""
    try:
        col = client.get_collection(name)
        col.upsert(
            ids=[_META_KEY],
            documents=["__meta__"],
            metadatas=[{"created_at": time.time(), "collection": name}],
        )
    except Exception:
        pass


def cleanup_stale_collections(ttl_hours: float = COLLECTION_TTL_HOURS) -> list[str]:
    """
    Delete collections whose sentinel timestamp is older than ttl_hours.

    Returns:
        List of collection names that were deleted.
    """
    client = _get_client()
    cutoff = time.time() - ttl_hours * 3600
    deleted: list[str] = []

    for col_info in client.list_collections():
        name = col_info.name
        try:
            col = client.get_collection(name)
            result = col.get(ids=[_META_KEY], include=["metadatas"])
            metas = result.get("metadatas") or []
            if metas and metas[0].get("created_at", time.time()) < cutoff:
                client.delete_collection(name)
                deleted.append(name)
        except Exception:
            pass

    return deleted


def delete_collection(client, collection_name: str) -> None:
    """Delete a collection by name. Silently ignores missing collections."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def list_collections() -> list[str]:
    """Return names of all existing persistent collections."""
    try:
        return [c.name for c in _get_client().list_collections()]
    except Exception:
        return []

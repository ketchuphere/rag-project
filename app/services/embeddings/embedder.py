"""
Embeddings service – HuggingFace sentence-transformer wrapper.
Returns a cached embedding model instance used by the vector store.

Future improvements:
  - Support OpenAI / Cohere / Voyage embeddings via an adapter interface
    so the embedding provider can be swapped without touching other services.
  - Add hybrid dense+sparse embeddings (fastembed + SPLADE) for better recall
    on keyword-heavy technical documents.
  - Cache embedding results to disk (SQLite / Redis) to avoid re-embedding
    unchanged chunks on restart.
"""

from functools import lru_cache
from app.config.settings import EMBEDDING_MODEL

try:
    from langchain_huggingface import HuggingFaceEmbeddings  #preferred (non-deprecated)
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  #fallback


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFace embedding model.
    The model is loaded once per process and reused across all requests.

    Uses langchain_huggingface (preferred) with fallback to langchain_community.
    If the model cannot be downloaded (e.g. no internet), raises OSError with
    a clear message directing the user to pre-download the model.
    """
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except OSError as e:
        raise OSError(
            f"Could not load embedding model '{EMBEDDING_MODEL}'. "
            "Ensure you have internet access on first run so the model can be downloaded, "
            "or pre-download it with: "
            f"python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{EMBEDDING_MODEL}')\".\n"
            f"Original error: {e}"
        ) from e

"""Centralised application settings — all tuneable constants live here."""

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str    = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY: str      = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY: str    = os.getenv("TAVILY_API_KEY", "")
REDIS_URL: str         = os.getenv("REDIS_URL", "")

GEMINI_MODEL: str  = "gemini-2.5-flash"
GROQ_MODEL: str    = "llama-3.3-70b-versatile"
CLAUDE_MODEL: str  = "claude-haiku-4-5-20251001"
LLM_TEMPERATURE: float = 0.2

EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"

CHUNK_SIZE: int    = 1000
CHUNK_OVERLAP: int = 200

RETRIEVAL_K: int   = 8
RERANK_TOP_K: int  = 4
RELEVANCE_THRESHOLD: float = 0.7

DEEP_RESEARCH_WEB_RESULTS: int  = 8
DEEP_RESEARCH_PDF_RESULTS: int  = 10
MAX_RESEARCH_ITERATIONS: int    = 2
MIN_RESEARCH_CONFIDENCE: float  = 0.7

MAX_MEMORY_MESSAGES: int = 12

TAVILY_SEARCH_URL: str  = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS: int = 5

COLLECTION_TTL_HOURS: float = 24.0

UNKNOWN_ANSWER: str = "I don't know from the uploaded documents."

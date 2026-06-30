<div align="center">

# 🤖 Agentic Research Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-State_Machines-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4.svg)](https://deepmind.google/technologies/gemini/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_Fallback-F55036.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An autonomous, multi-agent AI research assistant built with LangGraph.**  
*Seamlessly fuses PDF vector retrieval, live web search, and cross-provider LLM failover.*

</div>

---

## 📖 Overview

The **Agentic Research Assistant** is an advanced Retrieval-Augmented Generation (RAG) system. Unlike standard linear chatbots, it uses **LangGraph** to construct a dynamic, multi-agent state machine. It evaluates its own retrieval quality (Corrective RAG), searches the live web via Tavily when local PDFs are insufficient, and features an autonomous **Deep Research Mode** that verifies its own evidence before generating a structured report.

To ensure uptime, the system employs a custom **FailoverLLMWrapper** that intercepts API rate limits (429s) from Google Gemini and instantly reroutes queries to Groq's Llama-3.3 model without disrupting the pipeline.

---

## 📂 Project Structure

```
rag-project/
├── .env                          # API keys (never commit)
├── .env.example                  # Environment variable template
├── .gitignore
├── Dockerfile                    # Container build instructions
├── docker-compose.yml            # Multi-container setup
├── requirements.txt              # Python dependencies
├── README.md
│
├── app/
│   ├── main.py                   # Streamlit entrypoint
│   ├── config/
│   │   ├── settings.py           # All constants and env vars
│   │   └── state.py              # RAGState TypedDict schema
│   ├── routes/                   # Future FastAPI route handlers
│   ├── services/
│   │   ├── ingestion/
│   │   │   └── pdf_loader.py     # Load and extract PDF pages
│   │   ├── chunking/
│   │   │   └── text_splitter.py  # Recursive character chunking
│   │   ├── embeddings/
│   │   │   └── embedder.py       # HuggingFace embedding model
│   │   ├── vector_store/
│   │   │   └── chroma_store.py   # ChromaDB build & delete
│   │   ├── retrieval/
│   │   │   └── retriever.py      # Similarity search + re-ranking
│   │   ├── generation/
│   │   │   └── llm_factory.py    # FailoverLLMWrapper (Gemini → Groq)
│   │   ├── evaluation/
│   │   │   └── grader.py         # Relevance grading + evidence verification
│   │   └── agents/
│   │       ├── nodes.py          # All LangGraph node functions
│   │       └── graph.py          # Graph compilation + public runners
│   └── utils/
│       └── templates.py          # Streamlit HTML/CSS chat templates
│
├── data/
│   ├── raw/                      # Original uploaded PDFs
│   ├── processed/                # Cleaned / pre-processed text
│   └── vector_store/             # Persistent ChromaDB files (gitignored)
│
├── prompts/
│   └── templates/
│       └── rag_prompts.py        # Centralised prompt strings
│
├── tests/
│   ├── unit/
│   │   ├── test_failover.py      # FailoverLLMWrapper unit tests
│   │   └── test_chunking.py      # Text splitter unit tests
│   └── integration/
│       └── test_rag_graph.py     # End-to-end RAG graph integration test
│
├── frontend/                     # React + Vite + TypeScript UI
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api.ts
│       ├── types.ts
│       └── styles.css
│
├── logs/                         # Runtime logs (gitignored)
├── docs/                         # Architecture diagrams, ADRs
└── scripts/
    ├── ingest.py                 # CLI bulk PDF ingestion
    └── run_tests.sh              # Run all tests
```

---

## ✨ Key Features

- **Agentic Routing (CRAG):** Evaluates retrieved document relevance; routes to web search when local data is insufficient (score < 0.7).
- **Deep Research Mode:** Autonomous Plan → Search → Verify → Report loop that re-runs evidence gathering up to 2× before generating a structured report.
- **Cross-Provider Failover:** `FailoverLLMWrapper` silently intercepts Gemini 429 errors and reroutes to Groq Llama-3.3-70b with zero downtime.
- **Hybrid Context Fusion:** Merges PDF chunks and Tavily web results, deduplicated via SHA-256 fingerprinting.
- **Conversational Memory:** Sliding-window chat history (last 12 messages) injected into every prompt for contextual follow-ups.
- **Observability UI:** Streamlit sidebar shows workflow traces, fallback metrics, chunk counts, and source cards.

---

## 🏗️ Architecture

Two compiled LangGraph state machines share a unified `RAGState` TypedDict.

### Standard RAG Graph

```
USER QUERY
   │
   ▼
[ Initialize Memory ]
   │
   ▼
[ Rewrite Query ]
   │
   ▼
[ Retrieve PDFs ]  →  similarity_search (k=8) + keyword re-rank (top 4)
   │
   ▼
[ Grade Relevance ]  →  LLM scores 0–1
   │
   ├─ score ≥ 0.7 ──────────────────────────┐
   │                                         ▼
   └─ score < 0.7 → [ Web Search (Tavily) ] → [ Context Fusion ]
                                                       │
                                                       ▼
                                              [ Generate Answer ]
                                                       │
                                                       ▼
                                              [ Update Memory ] → END
```

### Deep Research Graph

```
[ Initialize Memory ] → [ Rewrite Query ] → [ Research Agent ]
                                                    │
                                                    ▼
                                         [ Verification Agent ]
                                           │               │
                               confidence < 0.7      confidence ≥ 0.7
                               iterations < 2          OR max reached
                                    │                       │
                                    └──── loops back        ▼
                                                    [ Report Agent ]
                                                            │
                                                    [ Update Memory ] → END
```

### LLM Failover

```
state["llm"].invoke(prompt)
   │
   ├─► Try: Gemini 2.5 Flash
   │
   └─► Except (429 / quota) ─► fallback_count++ ─► Groq Llama-3.3-70b
                                                          │
                                                   Except (any) ─► Error AIMessage
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9+
- Node.js 18+ (for the React frontend)
- API keys: Google Gemini, Groq, Tavily

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/rag-project.git
cd rag-project
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Fill in your GOOGLE_API_KEY, GROQ_API_KEY, TAVILY_API_KEY
```

### 3. Run the Streamlit App

```bash
streamlit run app/main.py
```

### 4. Run the React Frontend (optional)

```bash
cd frontend
cp .env.example .env        # set VITE_API_BASE_URL
npm install
npm run dev
```

### 5. Run with Docker

```bash
docker-compose up --build
```

### 6. Run Tests

```bash
bash scripts/run_tests.sh
# or individually:
python tests/unit/test_failover.py
python tests/unit/test_chunking.py
python tests/integration/test_rag_graph.py
```

---

## 🔮 Future Improvements

The following improvements are planned. Each is also documented inline in the relevant service module.

### 🔁 Ingestion (`services/ingestion/`)
- **Multi-format loaders:** Support DOCX, HTML, and Markdown via pluggable loader adapters alongside PDFs.
- **Streaming ingestion:** Process large PDFs page-by-page without loading the full file into memory.
- **Multimodal extraction:** Extract and store embedded images for vision-capable retrieval pipelines.

### ✂️ Chunking (`services/chunking/`)
- **Semantic chunking:** Split on sentence/paragraph boundaries using spaCy or NLTK instead of fixed character counts, improving retrieval coherence.
- **Sliding-window strategy:** Configurable stride for dense-retrieval workloads.
- **Pre-index deduplication:** SHA-256 fingerprint chunks before indexing to avoid redundant embeddings.

### 🔍 Retrieval (`services/retrieval/`)
- **Cross-encoder re-ranking:** Replace keyword overlap scoring with ms-marco-MiniLM-L-6-v2 or FlashRank for significantly higher retrieval precision.
- **Maximal Marginal Relevance (MMR):** Diversify retrieved chunks to reduce redundancy in the context window.
- **HyDE query expansion:** Generate a hypothetical answer and embed it as a retrieval query to improve recall on abstract questions.

### 🧠 Generation (`services/generation/`)
- **Multi-provider chain:** Extend `FailoverLLMWrapper` to N fallbacks (Gemini → Groq → Anthropic Claude).
- **Async invoke:** Add `ainvoke()` support so parallel LangGraph branches can run concurrently.
- **Latency-aware routing:** Track per-provider P50 latency and auto-select the fastest currently available model.
- **LiteLLM integration:** Unify all providers behind LiteLLM so new models can be added via config, not code.

### 📊 Evaluation (`services/evaluation/`)
- **RAGAS metrics:** Add faithfulness, answer relevancy, and context precision as optional post-generation quality gates.
- **Evaluation logging:** Persist relevance scores and verification results to SQLite for offline pipeline benchmarking.
- **A/B testing hooks:** Compare retrieval strategies using stored evaluation traces.

### 🤖 Agents & Graph (`services/agents/`)
- **Human-in-the-loop:** Add an interrupt node so users can approve or redirect the research plan before evidence gathering starts.
- **Checkpointing:** Persist compiled graph state to Redis so Deep Research sessions survive server restarts.
- **SSE streaming:** Expose graph execution as an async generator to stream intermediate node outputs to the UI.
- **Supervisor agent:** Dynamically select which sub-agent to invoke next, replacing the current static conditional routing.

### 💾 Vector Store (`services/vector_store/`)
- **Persistent managed store:** Migrate from ephemeral ChromaDB to Pinecone, Weaviate, or Qdrant for cross-restart persistence.
- **Multi-tenant namespacing:** One collection namespace per user/session with automatic TTL-based cleanup.
- **Incremental upsert:** Add new PDFs to an existing index without re-embedding the full corpus.

### 🖥️ Infrastructure
- **Async task queue:** Decouple LangGraph execution from Streamlit using Celery + Redis to prevent UI blocking during Deep Research.
- **FastAPI backend:** Replace Streamlit's backend role with a proper FastAPI service that the React frontend calls directly.
- **CI/CD pipeline:** GitHub Actions workflow that runs tests, lints, and builds the Docker image on every PR.
- **Prometheus metrics:** Expose fallback counts, latency, and retrieval scores as Prometheus metrics for production monitoring.

---

## 📄 License

This project is licensed under the MIT License.

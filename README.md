# TechMart Multi-Agent AI Customer Support Assistant

A multi-agent customer support system that uses **Retrieval-Augmented Generation (RAG)** to
route customer queries to specialized AI agents (Billing, Technical, Product, Complaint, FAQ),
retrieve relevant company documents, and generate grounded responses.

This implementation follows the assignment's architecture exactly (Intent Detection → Agent
Router → Specialized Agents → RAG → Response Aggregator) but makes two practical substitutions
so it **runs immediately, offline, with zero cost**, before you wire up paid APIs:

| Spec calls for | This build uses by default | Why |
|---|---|---|
| sentence-transformers embeddings + FAISS | **TF-IDF + cosine similarity** (scikit-learn) | No model download required — works with no internet access |
| OpenAI / Gemini / Llama | **Mock LLM client** that echoes retrieved context | Lets you test routing + RAG before spending API credits |
| MongoDB / PostgreSQL | **SQLite** (file-based) | Zero setup, same schema logic — swap later in 10 minutes |

Everything is built behind clean interfaces (`VectorStore`, `BaseLLMClient`) so upgrading to
the "real" stack is a config change, not a rewrite. See **Upgrading to production** below.

## Quick Start

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env            # defaults to LLM_PROVIDER=mock — no key needed to start

# 3. Run the backend
uvicorn backend.main:app --reload --port 8000

# 4. Open the frontend
# Just open frontend/index.html in your browser (it talks to localhost:8000)
```

Test it's alive: `curl http://localhost:8000/health`

Try a query:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I paid yesterday but Premium is still locked."}'
```
This query correctly routes to **both** the Billing and Technical agents, since it touches
payment and access issues at once — that's the multi-agent routing example from the spec.

## Architecture

```
Customer → frontend/index.html
              │
              ▼
        POST /chat  (backend/main.py)
              │
              ▼
     Intent Detection Agent  (agents/intent_detector.py)
              │
              ▼
        Agent Router  (agents/router.py)
              │
   ┌──────────┼──────────┬───────────┬──────────┐
   ▼          ▼          ▼           ▼          ▼
Billing   Technical   Product    Complaint     FAQ
  │          │          │           │           │
  └──────────┴──────────┴───────────┴───────────┘
              │
              ▼
     RAG Pipeline (rag/pipeline.py)
              │
              ▼
     TF-IDF Vector Store (vectorstore/store.py)
              │
              ▼
     knowledge_base/*.txt  (TechMart FAQ, refund, shipping, warranty, pricing)
              │
              ▼
     Response Aggregator (router.py: _aggregate)
              │
              ▼
        Final answer + conversation saved to SQLite
```

## Project Structure

```
customer-support-ai/
├── frontend/
│   └── index.html            # Self-contained chat UI, no build step
├── backend/
│   ├── main.py                # FastAPI app + /chat, /health, /sessions endpoints
│   ├── agents/
│   │   ├── intent_detector.py # Keyword-based + optional LLM-based classifiers
│   │   ├── base.py            # Shared agent behavior (prompt building, RAG call)
│   │   ├── specialized.py     # Billing, Technical, Product, Complaint, FAQ agents
│   │   └── router.py          # Orchestrator: routes + aggregates multi-agent replies
│   ├── rag/
│   │   ├── loader.py          # Document loading + chunking
│   │   └── pipeline.py        # Ties loader + vector store together
│   ├── vectorstore/
│   │   └── store.py           # TfidfVectorStore (default) + SentenceTransformerVectorStore (optional)
│   ├── models/
│   │   └── llm.py             # LLM client abstraction: mock / openai / gemini / groq
│   ├── database/
│   │   └── memory.py          # SQLite-backed conversation history
│   └── api/
│       └── schemas.py         # Pydantic request/response models
├── knowledge_base/             # TechMart Electronics sample company documents
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## Upgrading to production

**Real LLM:** set `LLM_PROVIDER=openai` (or `gemini` / `groq`) in `.env`, add the matching API
key, and uncomment the relevant line in `requirements.txt`. No other code changes needed.

**Dense embeddings:** uncomment `sentence-transformers` and `faiss-cpu` in `requirements.txt`,
then in `backend/rag/pipeline.py` change:
```python
from backend.vectorstore.store import SentenceTransformerVectorStore
self.store = store or SentenceTransformerVectorStore()
```

**MongoDB/Postgres:** replace the SQLite calls in `backend/database/memory.py` with your driver
of choice (`pymongo` or `psycopg2`/`SQLAlchemy`) — the function signatures (`save_message`,
`get_history`) are the contract the rest of the app relies on, so nothing else needs to change.

**Deployment:** the included `Dockerfile` builds the backend for Railway/Render. Push
`frontend/index.html` to Vercel/Netlify as a static site and set `window.TECHMART_API_BASE` in
a small inline `<script>` tag to your deployed backend URL before the main script runs.

## Extending the knowledge base

Drop any `.txt` or `.pdf` file into `knowledge_base/` and restart the server — it's
auto-ingested and chunked on startup (`rag/pipeline.py`). No re-indexing step required.

## How this maps to the evaluation criteria

| Component | Where |
|---|---|
| Frontend Design | `frontend/index.html` — full chat UI, typing indicator, intent tags, suggestion chips |
| Backend APIs | `backend/main.py` — REST endpoints for chat, health, session history |
| Multi-Agent Architecture | `backend/agents/` — intent detection, 5 specialized agents, router, aggregator |
| RAG Implementation | `backend/rag/`, `backend/vectorstore/` — chunking, retrieval, grounded prompts |
| LLM Integration | `backend/models/llm.py` — pluggable OpenAI/Gemini/Groq clients |
| Database Design | `backend/database/memory.py` — session-based conversation history |
| Documentation & Deployment | this README, `Dockerfile`, `.env.example` |

## Suggested next steps for your submission

1. Swap in a real LLM key and re-test the four sample queries in the frontend's suggestion chips.
2. Write a short project report covering: problem statement, architecture, and 3-5 example
   conversations with screenshots.
3. Record the demo video: show a query that triggers a *single* agent, then one that triggers
   *multiple* agents (like the Premium billing/technical example above) to demonstrate routing.
4. Deploy: backend → Railway/Render, frontend → Vercel, and paste both URLs into your report.

"""
Multi-Agent AI Customer Support Assistant — Backend Entry Point

Run with:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    POST /chat          -> send a customer message, get a routed multi-agent response
    GET  /health         -> service + knowledge base status
    GET  /sessions        -> list active conversation session IDs
    GET  /sessions/{id}/history -> full history for a session
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.router import AgentRouter
from backend.api.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.database import memory
from backend.models.llm import get_llm_client
from backend.rag.pipeline import rag_pipeline

app = FastAPI(
    title="TechMart Multi-Agent Customer Support API",
    description="Multi-agent AI customer support assistant using RAG and LLMs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = get_llm_client()
router = AgentRouter(llm_client=llm_client, rag_pipeline=rag_pipeline)


@app.on_event("startup")
def startup():
    memory.init_db()
    chunk_count = rag_pipeline.ingest()
    print(f"Knowledge base ingested: {chunk_count} chunks indexed.")


@app.get("/health", response_model=HealthResponse)
def health():
    chunk_count = len(rag_pipeline.store.chunks) if rag_pipeline.store.chunks else 0
    return HealthResponse(status="ok", knowledge_base_chunks=chunk_count)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id or memory.new_session_id()
    history = memory.get_history(session_id)

    result = router.route(request.message, history=history)

    memory.save_message(session_id, "user", request.message)
    memory.save_message(session_id, "assistant", result["final_answer"], intents=",".join(result["intents"]))

    return ChatResponse(
        session_id=session_id,
        intents=result["intents"],
        agent_responses=result["agent_responses"],
        final_answer=result["final_answer"],
    )


@app.get("/sessions")
def list_sessions():
    return {"sessions": memory.get_all_sessions()}


@app.get("/sessions/{session_id}/history")
def session_history(session_id: str):
    return {"session_id": session_id, "history": memory.get_history(session_id, limit=100)}

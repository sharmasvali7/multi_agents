from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    agent: str
    answer: str
    used_context: bool


class ChatResponse(BaseModel):
    session_id: str
    intents: List[str]
    agent_responses: List[AgentResponse]
    final_answer: str


class HealthResponse(BaseModel):
    status: str
    knowledge_base_chunks: int

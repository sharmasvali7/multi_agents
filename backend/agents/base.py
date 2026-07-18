"""Base class shared by all specialized support agents."""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    name: str = "base"
    system_prompt: str = "You are a helpful customer support agent."

    def __init__(self, llm_client, rag_pipeline):
        self.llm_client = llm_client
        self.rag_pipeline = rag_pipeline

    def build_prompt(self, query: str, context: str, history: str = "") -> str:
        return (
            f"{self.system_prompt}\n\n"
            f"Relevant company documents:\n{context or 'No relevant documents found.'}\n\n"
            f"Conversation so far:\n{history or '(none)'}\n\n"
            f"Customer message: {query}\n\n"
            "Answer using ONLY the information in the documents above where relevant. "
            "If the documents don't cover the question, say so honestly and offer to "
            "escalate to a human agent. Keep the answer concise and friendly."
        )

    def respond(self, query: str, history: str = "") -> dict:
        context = self.rag_pipeline.retrieve_context(query, top_k=4)
        prompt = self.build_prompt(query, context, history)
        answer = self.llm_client.complete(prompt)
        return {
            "agent": self.name,
            "answer": answer,
            "used_context": bool(context),
        }

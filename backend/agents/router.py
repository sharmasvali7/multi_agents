"""
Agent Router (the central orchestrator).

Given a customer query:
1. Detect intent(s) via the Intent Detection Agent
2. Invoke each matched specialized agent, each doing its own RAG retrieval
3. Aggregate responses into a single, coherent reply
"""
from typing import List

from backend.agents.intent_detector import KeywordIntentDetector
from backend.agents.specialized import AGENT_REGISTRY


class AgentRouter:
    def __init__(self, llm_client, rag_pipeline, intent_detector=None):
        self.llm_client = llm_client
        self.rag_pipeline = rag_pipeline
        self.intent_detector = intent_detector or KeywordIntentDetector()
        self.agents = {
            name: cls(llm_client, rag_pipeline) for name, cls in AGENT_REGISTRY.items()
        }

    def route(self, query: str, history: str = "") -> dict:
        intents = self.intent_detector.detect(query)
        agent_responses = []

        for intent in intents:
            agent = self.agents.get(intent, self.agents["faq"])
            result = agent.respond(query, history=history)
            agent_responses.append(result)

        final_answer = self._aggregate(query, agent_responses)

        return {
            "query": query,
            "intents": intents,
            "agent_responses": agent_responses,
            "final_answer": final_answer,
        }

    def _aggregate(self, query: str, agent_responses: List[dict]) -> str:
        """
        Combine multiple agent answers into one reply. If only one agent
        responded, its answer is used directly. Otherwise, ask the LLM to
        merge them into a single coherent message.
        """
        if len(agent_responses) == 1:
            return agent_responses[0]["answer"]

        combined = "\n\n".join(
            f"[{r['agent'].upper()} AGENT]: {r['answer']}" for r in agent_responses
        )
        merge_prompt = (
            "The following are draft answers from different specialist support agents "
            f"responding to the same customer query: \"{query}\".\n\n{combined}\n\n"
            "Combine these into a single, well-organized, non-repetitive reply to the "
            "customer. Keep all relevant information, remove redundancy."
        )
        return self.llm_client.complete(merge_prompt)

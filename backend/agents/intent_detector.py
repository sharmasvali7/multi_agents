"""
Intent Detection Agent.

Classifies an incoming customer query into one or more of:
billing, technical, product, complaint, faq

Two implementations are provided:
- KeywordIntentDetector: rule-based, zero-dependency, works offline. Used by default.
- LLMIntentDetector: asks the configured LLM to classify intent as JSON. Use once
  an LLM API key is configured (see backend/models/llm.py).

Both return a list of intents because a single query can span domains
(e.g. "I paid yesterday but Premium is still locked" -> billing + technical).
"""
import json
import re
from typing import List

INTENTS = ["billing", "technical", "product", "complaint", "faq"]

KEYWORDS = {
    "billing": [
        "pay", "payment", "charge", "charged", "invoice", "subscription",
        "refund", "billing", "premium", "renew", "credit card", "upi",
        "emi", "price", "overcharged", "wallet",
    ],
    "technical": [
        "login", "log in", "password", "reset", "install", "installation",
        "error", "bug", "crash", "not working", "unlock", "locked", "app",
        "sync", "connect", "bluetooth", "update", "freeze", "slow",
    ],
    "product": [
        "feature", "specification", "specs", "compare", "comparison",
        "available", "availability", "stock", "price of", "warranty",
        "buy", "model", "difference between",
    ],
    "complaint": [
        "complaint", "complain", "disappointed", "angry", "unacceptable",
        "worst", "terrible", "escalate", "manager", "frustrated", "refuse",
        "again and again", "never resolved",
    ],
    "faq": [
        "hours", "contact", "policy", "how do i", "how can i", "what is",
        "shipping", "delivery time", "return policy", "account", "sign up",
    ],
}


class KeywordIntentDetector:
    def detect(self, query: str) -> List[str]:
        text = query.lower()
        matched = []
        for intent, keywords in KEYWORDS.items():
            if any(kw in text for kw in keywords):
                matched.append(intent)
        return matched or ["faq"]  # default to FAQ agent if nothing matches


class LLMIntentDetector:
    """
    LLM-based classifier for higher accuracy than keyword matching.
    Requires an LLM client (see backend/models/llm.py).
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def detect(self, query: str) -> List[str]:
        prompt = (
            "Classify the customer support query into one or more of these intents: "
            f"{', '.join(INTENTS)}. Respond ONLY with a JSON array of intent strings, "
            f'e.g. ["billing", "technical"]. Query: "{query}"'
        )
        response_text = self.llm_client.complete(prompt)
        try:
            match = re.search(r"\[.*\]", response_text, re.DOTALL)
            intents = json.loads(match.group(0)) if match else []
            intents = [i for i in intents if i in INTENTS]
            return intents or ["faq"]
        except (json.JSONDecodeError, AttributeError):
            return ["faq"]

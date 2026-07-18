"""
LLM client abstraction. Pick a provider by setting LLM_PROVIDER in your .env file.

Supported: "openai", "gemini", "groq" (for Llama 3), "mock" (offline, no API key,
useful for testing the agent/routing/RAG logic without any network access or cost).
"""
import os


class BaseLLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content


class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(model)

    def complete(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class GroqClient(BaseLLMClient):
    """For running Llama 3 via Groq's fast inference API."""

    def __init__(self, model: str = "llama3-70b-8192", api_key: str = None):
        from groq import Groq

        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content


class MockLLMClient(BaseLLMClient):
    """
    Offline stand-in that just surfaces the retrieved context directly.
    Lets you run and test the full pipeline (routing, RAG retrieval, API)
    with zero API keys and zero internet access, before wiring up a real LLM.
    """

    def complete(self, prompt: str) -> str:
        # Pull out the "Relevant company documents" section for a readable stub answer
        marker = "Relevant company documents:\n"
        if marker in prompt:
            after = prompt.split(marker, 1)[1]
            context = after.split("\n\nConversation so far:")[0].strip()
            if context and context != "No relevant documents found.":
                snippet = context.split("---")[0].strip()
                return f"[MOCK LLM — offline test mode]\nBased on our records:\n{snippet}"
        return (
            "[MOCK LLM — offline test mode] I couldn't find a specific document for "
            "that, but I can escalate this to a human agent if you'd like."
        )


def get_llm_client() -> BaseLLMClient:
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    if provider == "openai":
        return OpenAIClient()
    if provider == "gemini":
        return GeminiClient()
    if provider == "groq":
        return GroqClient()
    return MockLLMClient()

"""
RAG pipeline: ingest the knowledge base once at startup, then retrieve
relevant chunks for each incoming query.
"""
import os
from typing import List, Tuple

from backend.rag.loader import Chunk, build_chunks
from backend.vectorstore.store import TfidfVectorStore, VectorStore

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base")


class RAGPipeline:
    def __init__(self, knowledge_base_dir: str = KNOWLEDGE_BASE_DIR, store: VectorStore = None):
        self.knowledge_base_dir = knowledge_base_dir
        self.store = store or TfidfVectorStore()
        self._ready = False

    def ingest(self) -> int:
        """Load, chunk, and index all documents in the knowledge base. Returns chunk count."""
        chunks = build_chunks(self.knowledge_base_dir)
        self.store.build(chunks)
        self._ready = True
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if not self._ready:
            self.ingest()
        return self.store.search(query, top_k=top_k)

    def retrieve_context(self, query: str, top_k: int = 4) -> str:
        """Return retrieved chunks formatted as a single context string for an LLM prompt."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""
        parts = []
        for chunk, score in results:
            parts.append(f"[Source: {chunk.source}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)


# Singleton used by the FastAPI app
rag_pipeline = RAGPipeline()

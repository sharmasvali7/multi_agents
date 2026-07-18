"""
Vector store for the RAG pipeline.

Default backend: TF-IDF + cosine similarity via scikit-learn. This requires
no downloaded models and works fully offline, which makes local development
and grading straightforward.

To upgrade to dense embeddings (recommended for production), swap
`TfidfVectorStore` for `SentenceTransformerVectorStore`, which uses
sentence-transformers/all-MiniLM-L6-v2 + FAISS. Both implement the same
`VectorStore` interface, so the rest of the app (rag/pipeline.py) does not
need to change.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.rag.loader import Chunk


class VectorStore(ABC):
    @abstractmethod
    def build(self, chunks: List[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


class TfidfVectorStore(VectorStore):
    """Lightweight, dependency-free (beyond scikit-learn) semantic-ish retriever."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=20000
        )
        self.chunks: List[Chunk] = []
        self.matrix = None

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.matrix is None or not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(zip(self.chunks, sims), key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k] if score > 0]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "chunks": self.chunks, "matrix": self.matrix}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.chunks = data["chunks"]
        self.matrix = data["matrix"]


class SentenceTransformerVectorStore(VectorStore):
    """
    Optional upgrade path: dense embeddings + FAISS.
    Requires: pip install sentence-transformers faiss-cpu
    Not used by default so the project runs without internet access to
    download model weights; enable it once you have connectivity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._index = None
        self.chunks: List[Chunk] = []

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, chunks: List[Chunk]) -> None:
        import faiss
        import numpy as np

        model = self._load_model()
        self.chunks = chunks
        embeddings = model.encode([c.text for c in chunks], normalize_embeddings=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(np.array(embeddings, dtype="float32"))

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        import numpy as np

        if self._index is None:
            return []
        model = self._load_model()
        query_vec = model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.array(query_vec, dtype="float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        import faiss

        faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str) -> None:
        import faiss

        self._index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

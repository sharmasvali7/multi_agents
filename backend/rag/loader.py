"""
Document loading and chunking for the RAG pipeline.

Supports .txt and .pdf files in the knowledge_base/ directory.
Splits documents into overlapping chunks suitable for embedding and retrieval.
"""
import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    """Extract text from a PDF using pypdf. Falls back gracefully if unavailable."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required to read PDF files. Install with: pip install pypdf"
        ) from e
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(knowledge_base_dir: str) -> List[dict]:
    """Load all .txt and .pdf files from the knowledge base directory."""
    docs = []
    for filename in sorted(os.listdir(knowledge_base_dir)):
        path = os.path.join(knowledge_base_dir, filename)
        if filename.lower().endswith(".txt"):
            text = _read_txt(path)
        elif filename.lower().endswith(".pdf"):
            text = _read_pdf(path)
        else:
            continue
        docs.append({"filename": filename, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks by character count, breaking on
    paragraph/sentence boundaries where possible so chunks stay coherent.
    """
    # Normalize whitespace but keep paragraph breaks
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # Paragraph itself is too long; split by sentence
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= chunk_size:
                        current = f"{current} {sent}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para
    if current:
        chunks.append(current)

    # Add overlap between consecutive chunks for better retrieval continuity
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail} {chunks[i]}".strip())
        return overlapped

    return chunks


def build_chunks(knowledge_base_dir: str, chunk_size: int = 600, overlap: int = 100) -> List[Chunk]:
    """Load documents from disk and split them into Chunk objects ready for embedding."""
    all_chunks: List[Chunk] = []
    for doc in load_documents(knowledge_base_dir):
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    text=piece,
                    source=doc["filename"],
                    chunk_id=f"{doc['filename']}::chunk{i}",
                )
            )
    return all_chunks

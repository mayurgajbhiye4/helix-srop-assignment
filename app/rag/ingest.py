"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Reads markdown files, chunks them, embeds, and writes to the vector store.
"""
import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

from app.agents.tools.search_docs import COLLECTION_NAME, EMBEDDING_MODEL
from app.settings import settings

EMBEDDING_BATCH_SIZE = 64


def _get_collection() -> Any:
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_documents(chunks: list[str]) -> list[list[float]]:
    if not chunks:
        return []
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to ingest docs")

    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=chunks,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [list(embedding.values) for embedding in result.embeddings]


def _chunk_id(source_path: str, chunk_index: int) -> str:
    raw = f"{source_path}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text using heading-aware chunking (Strategy C).
    
    Splits on ## and ### headings, then sub-chunks large sections by sentence.
    This preserves document structure and heading context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # Split on markdown headings (## or ###)
    sections = re.split(r'\n(?=#{2,3} )', text)
    chunks: list[str] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If section fits within chunk_size, keep it as-is
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # Sub-chunk large sections using sentence-aware splitting
            sub_chunks = _chunk_by_sentences(section, chunk_size, overlap)
            chunks.extend(sub_chunks)

    return [c for c in chunks if c.strip()]


def _chunk_by_sentences(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text on sentence boundaries. If a sentence is longer than chunk_size,
    split it character-wise to enforce the size limit.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        # If single sentence exceeds chunk_size, split it further
        if len(sentence) > chunk_size:
            # Save any accumulated sentences first
            if current:
                chunk_str = " ".join(current)
                if chunk_str.strip():
                    chunks.append(chunk_str)
                current = []

            # Split long sentence into character-based chunks
            for i in range(0, len(sentence), chunk_size):
                piece = sentence[i : i + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            continue

        # Try to add sentence to current chunk
        candidate = " ".join(current + [sentence])
        if len(candidate) <= chunk_size:
            # Fits, add it
            current.append(sentence)
        else:
            # Doesn't fit, save current and start new
            if current:
                chunk_str = " ".join(current)
                if chunk_str.strip():
                    chunks.append(chunk_str)
            current = [sentence]

    # Save final chunk
    if current:
        chunk_str = " ".join(current)
        if chunk_str.strip():
            chunks.append(chunk_str)

    return chunks


def extract_metadata(file_path: Path, text: str) -> dict:
    """
    Extract metadata from a markdown file's frontmatter.

    Expected frontmatter format:
        ---
        title: Deploy Keys
        product_area: security
        tags: [keys, secrets]
        ---

    Returns a dict suitable for vector store metadata filtering.
    """
    metadata: dict[str, str | int | float | bool] = {"source": file_path.name}

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return metadata

    end = normalized.find("\n---\n", 4)
    if end == -1:
        return metadata

    for line in normalized[4:end].splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue

        if value.startswith("[") and value.endswith("]"):
            metadata[key] = ",".join(
                item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()
            )
        else:
            metadata[key] = value.strip("\"'")

    return metadata


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.

    Design considerations:
    - Generate a stable chunk_id (e.g. sha256(file + chunk_index)) for deduplication.
    - Run embeddings in batches to avoid rate limiting.
    - Print progress so the user can see what's happening.
    """
    md_files = sorted(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")
    collection = _get_collection()

    for file_path in md_files:
        source_path = file_path.relative_to(docs_path).as_posix()
        text = file_path.read_text(encoding="utf-8")
        metadata = extract_metadata(file_path, text)
        chunks = chunk_markdown(text, chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")
        if not chunks:
            continue

        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_chunks = chunks[start : start + EMBEDDING_BATCH_SIZE]
            batch_ids = [
                _chunk_id(source_path, start + index)
                for index in range(len(batch_chunks))
            ]
            batch_metadata = [
                metadata | {"chunk_index": start + index, "chunk_id": chunk_id}
                for index, chunk_id in enumerate(batch_ids)
            ]
            embeddings = await asyncio.to_thread(_embed_documents, batch_chunks)
            await asyncio.to_thread(
                collection.upsert,
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_chunks,
                metadatas=batch_metadata,
            )

    print("Ingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()

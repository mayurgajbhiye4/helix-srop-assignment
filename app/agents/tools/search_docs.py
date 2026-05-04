"""
search_docs tool — used by KnowledgeAgent.

Queries the vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.

"""
import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.settings import settings

COLLECTION_NAME = "helix_docs"
EMBEDDING_MODEL = "gemini-embedding-001"


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


@lru_cache(maxsize=1)
def _get_collection() -> Any:
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_query(query: str) -> list[float]:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to search docs")

    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(result.embeddings[0].values)


def _score_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - distance))


def _search_docs_sync(query: str, k: int, product_area: str | None) -> list[DocChunk]:
    if not query.strip() or k <= 0:
        return []

    results = _get_collection().query(
        query_embeddings=[_embed_query(query)],
        n_results=k,
        where={"product_area": product_area} if product_area else None,
        include=["documents", "metadatas", "distances"],
    )

    chunks = [
        DocChunk(
            chunk_id=chunk_id,
            score=round(_score_from_distance(distance), 4),
            content=document or "",
            metadata=metadata or {},
        )
        for chunk_id, distance, document, metadata in zip(
            results.get("ids", [[]])[0],
            results.get("distances", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            strict=False,
        )
    ]
    return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the vector store for top-k relevant chunks.

    Args:
        query: natural language query from the user
        k: number of chunks to return
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        List of DocChunk ordered by descending similarity score.

    Design considerations:
    - How do you embed the query? Same model as at ingest time.
    - Do you apply a score threshold to filter low-quality results?
    - How do you format chunks for the agent? Include chunk_id so agent can cite.
    """
    chunks = await asyncio.to_thread(_search_docs_sync, query, k, product_area)

    if not chunks:
        return "No relevant documentation found."

    body = "\n\n".join(
        f"[score={chunk.score}] (source: {chunk.metadata.get('source', 'unknown')})\n{chunk.content}"
        for chunk in chunks
    )
    chunk_ids = ",".join(chunk.chunk_id for chunk in chunks)
    return f"{body}\n\n<!-- chunk_ids: {chunk_ids} -->"
"""
Unit tests for RAG retrieval and chunking.
"""
import pytest


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids(monkeypatch):
    """search_docs must return chunk IDs and scores in [0, 1]."""
    import app.agents.tools.search_docs as search_docs_module

    chunk_ids = ["a" * 64, "b" * 64, "c" * 64]

    class FakeCollection:
        def query(self, **kwargs):
            assert kwargs["query_embeddings"] == [[1.0, 0.0, 0.0]]
            assert kwargs["n_results"] == 3
            assert kwargs["where"] is None
            assert kwargs["include"] == ["documents", "metadatas", "distances"]
            return {
                "ids": [chunk_ids],
                "distances": [[0.0, 0.2, 1.2]],
                "documents": [
                    [
                        "Rotate a deploy key by creating a replacement and revoking the old key.",
                        "Deploy keys can be scoped to specific repositories.",
                        "API authentication uses bearer tokens.",
                    ]
                ],
                "metadatas": [
                    [
                        {"chunk_id": chunk_ids[0], "product_area": "security"},
                        {"chunk_id": chunk_ids[1], "product_area": "security"},
                        {"chunk_id": chunk_ids[2], "product_area": "api"},
                    ]
                ],
            }

    monkeypatch.setattr(search_docs_module, "_embed_query", lambda query: [1.0, 0.0, 0.0])
    monkeypatch.setattr(search_docs_module, "_get_collection", lambda: FakeCollection())
    results = await search_docs_module.search_docs("rotate deploy key", k=3)

    assert len(results) > 0
    assert all(result.chunk_id for result in results)
    assert all(0.0 <= result.score <= 1.0 for result in results)
    assert results == sorted(results, key=lambda result: result.score, reverse=True)


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    from app.rag.ingest import chunk_markdown

    text = (
        "# Header\n\n"
        "Some content about deploy keys and rotation.\n\n"
        "## Section 2\n\n"
        "More content here with enough text to force multiple chunks."
    )
    chunks = chunk_markdown(text, chunk_size=60, overlap=10)

    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 60 for chunk in chunks)

# Helix SROP — Mayur Gajbhiye

## Setup

### For pytest (clean clone, no API key needed)

```bash
git clone https://github.com/mayurgajbhiye4/helix-srop-assignment
cd helix-srop-assignment
uv sync
uv run pytest -q
```

### For manual testing with live LLM
```bash
cp .env.example .env  # fill in GOOGLE_API_KEY
uv run python -m app.rag.ingest --path docs/
uv run uvicorn app.main:app --reload
```

## Quick Test

```bash
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .
```

## Architecture

┌─────────────────────────────────────────────────────────────┐
│ FastAPI REST API │
│ POST /v1/sessions | POST /v1/chat | GET /v1/traces │
└──────────────┬──────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ SROP Pipeline (pipeline.py) │
│ 1. Load SessionState from DB │
│ 2. Build ADK root agent with state context │
│ 3. Run user message through orchestrator │
│ 4. Extract routed_to + tool calls from ADK events │
│ 5. Persist updated state + trace to SQLite │
│ 6. Return reply + metadata │
└──────────────┬──────────────────────────────────────────────┘
│
┌───────┴────────────────────────────┐
│ │
▼ ▼
┌──────────────────────┐ ┌──────────────────────┐
│ Root Orchestrator │ │ SessionState (DB) │
│ (ADK LlmAgent) │ │ - user_id │
│ Routes intent │ │ - plan_tier │
│ via AgentTool │ │ - last_agent │
└──────────────────────┘ │ - turn_count │
│ └──────────────────────┘
│
┌───┼───┬──────────────┐
│ │ │ │
▼ ▼ ▼ ▼
┌──────────┐ ┌──────────────┐ ┌────────────────┐
│Knowledge │ │ Account │ │ Escalation │
│Agent │ │ Agent │ │ Agent │
│[ADK] │ │ [ADK] │ │ [ADK] │
└──────────┘ └──────────────┘ └────────────────┘
│ │ │
▼ ▼ ▼
┌──────────┐ ┌──────────────┐ ┌────────────────┐
│search_ │ │get_recent_ │ │create_ │
│docs() │ │builds() │ │ticket() │
│ │ │get_account │ │ │
│[RAG] │ │_status() │ │[Mocks] │
└──────────┘ └──────────────┘ └────────────────┘
│
▼
┌──────────────────────────────┐
│ Chroma Vector Store │
│ (PersistentClient) │
│ - 10 markdown docs chunked │
│ - Cosine similarity (HNSW) │
│ - Google embeddings │
└──────────────────────────────┘

## Design Decisions

### State persistence (which pattern and why)
I used Pattern 3 from the ADK guide: store only compact `SessionState` in the
database and inject it into the root agent instruction each turn. This keeps the
context window small, avoids replaying full chat history, and still survives
process restarts because `user_id`, `plan_tier`, `last_agent`, and `turn_count`
are persisted in `sessions.state`.

### Chunking strategy
I used **hybrid heading-aware + sentence-aware chunking (Strategy B/C combined)** because the Helix docs are structured in Markdown with clear section headings. The strategy splits on `##` and `###` headings first (preserving structural context), then sub-chunks large sections using sentence boundaries to maintain coherence within the 512-character chunk size. This balances heading context preservation with retrieval granularity.

### Vector store choice
I chose **Chroma** because it offers persistent local storage (no external dependencies), built-in support for similarity search with cosine distance and HNSW indexing for performance, and integration with Google's embedding API. The schema naturally stores chunk IDs, scores, and metadata (product_area, source file) needed for citation tracking.

## Known Limitations

- No idempotency guards on ticket creation (E1 extension not implemented)
- Escalation agent is basic; no priority-based routing or SLA tracking
- No reranking step to improve retrieval quality
- Vector store schema doesn't support advanced filtering by product_area or recency
- No eval harness to measure RAG quality or agent routing accuracy
- Streaming SSE only partially implemented (response model doesn't adapt)


## What I'd Do With More Time

- **E1: Idempotency**: Implement request deduplication using a unique `request_id` header + hash-based storage to ensure duplicate chat messages return the same response
- **E2: Escalation agent completion**: Add priority-based ticket routing, SLA assignment based on plan_tier, and automatic escalation workflows
- **E3: Streaming SSE completion**: Implement true streaming responses that emit token-by-token updates from the LLM using ADK event streaming
- **E4: Reranking**: Integrate a cross-encoder model (e.g., `ms-marco-MiniLM-L-6-v2`) to rerank top-k retrieved chunks before passing to the LLM
- **E7: Eval harness**: Build a test suite measuring intent classification accuracy, retrieval quality (precision@k), and end-to-end conversation correctness
- **Hybrid search**: Combine vector search with BM25 full-text search for better recall on technical queries
- **Chunk metadata filtering**: Filter by product_area, date, or category before ranking results
- **Guardrails hardening**: Add prompt injection detection and output sanitization
- **Token budgeting**: Dynamically adjust chunk count based on available context window

## Time Spent

| Phase | Time |
|-------|------|
| Setup + DB + FastAPI boilerplate | 60 min |
| RAG ingest + chunking strategy | 90 min |
| ADK agents + AgentTool pattern | 90 min |
| pipeline.py + state persistence | 180 min |
| Tests + error handling | 60 min |
| README + Architecture | 30 min |
|Bug Fixes              | 180 min |
| **Total** | 690 min (≈ 11.5 hrs) |

## Extensions Completed

- [No] E1: Idempotency
- [Partial] E2: Escalation agent
- [Partial] E3: Streaming SSE
- [No] E4: Reranking
- [Yes] E5: Guardrails
- [Yes] E6: Docker
- [No] E7: Eval harness
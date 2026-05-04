"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches the ADK root agent so tests don't hit the real LLM
- `seeded_db`: DB with a test user and session pre-created
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base
from app.db.session import get_db
from app.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# @pytest_asyncio.fixture
# async def db()-> AsyncSession:
#     async with TestSessionLocal() as session:
#         yield session

@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def db():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch the ADK pipeline so tests don't call the real LLM.
    """
    class MockADK:
        def __init__(self):
            self.calls = []
            self.called_agents = []

        async def run_turn(self, session_id, user_message, state):
            lowered = user_message.lower()

            if "rotate" in lowered or "deploy key" in lowered:
                routed_to = "knowledge"
                content = """To rotate a deploy key,
                             create a replacement key,
                             update consumers,
                             then revoke the old key.
                          """
                chunk_ids = ["a" * 64]
                tool_calls = [
                    {
                        "tool_name": "search_docs",
                        "args": {"query": user_message},
                        "result": {
                            "chunk_id": chunk_ids[0],
                            "title": "Deploy key rotation",
                        },
                    }
                ]
            elif "plan tier" in lowered:
                routed_to = "account"
                content = f"Your plan tier is {state.plan_tier}."
                chunk_ids = []
                tool_calls = [
                    {
                        "tool_name": "get_account_status",
                        "args": {"user_id": state.user_id},
                        "result": {"plan_tier": state.plan_tier},
                    }
                ]
            else:
                routed_to = "smalltalk"
                content = "I can help with Helix account and knowledge base questions."
                chunk_ids = []
                tool_calls = []

            self.calls.append(
                {
                    "session_id": session_id,
                    "message": user_message,
                    "routed_to": routed_to,
                    "state": state,
                }
            )
            self.called_agents.append(routed_to)
            return content, routed_to, tool_calls, chunk_ids

    mock = MockADK()
    monkeypatch.setattr("app.srop.pipeline._run_adk_turn", mock.run_turn)
    return mock

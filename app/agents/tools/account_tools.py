"""
Account tools — used by AccountAgent.

These tools query the DB for user-specific data.
Mock data is acceptable for the take-home; the integration matters.

TODO for candidate: implement these tools.
"""
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

_db_session: AsyncSession | None = None

def set_db_session(db: AsyncSession) -> None:
    """Set the database session for tool use."""
    global _db_session
    _db_session = db

@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: datetime
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


async def get_recent_builds(user_id: str, limit: int = 5) -> list[BuildSummary]:
    """
    Return the most recent builds for a user, newest first.

    For the take-home: returning mock/seeded data is fine.
    The key evaluation point is that this is wired as an ADK tool
    and the agent correctly invokes it when the user asks about builds.
    """
    builds = [
        BuildSummary(
            build_id=f"{user_id}-bld-1042",
            pipeline="deploy",
            status="failed",
            branch="main",
            started_at=datetime(2026, 5, 2, 9, 42, 0),
            duration_seconds=418,
        ),
        BuildSummary(
            build_id=f"{user_id}-bld-1041",
            pipeline="test",
            status="passed",
            branch="feature/rag-routing",
            started_at=datetime(2026, 5, 2, 8, 17, 0),
            duration_seconds=231,
        ),
        BuildSummary(
            build_id=f"{user_id}-bld-1040",
            pipeline="build",
            status="passed",
            branch="main",
            started_at=datetime(2026, 5, 1, 22, 8, 0),
            duration_seconds=189,
        ),
        BuildSummary(
            build_id=f"{user_id}-bld-1039",
            pipeline="deploy",
            status="cancelled",
            branch="release/2026-05",
            started_at=datetime(2026, 5, 1, 18, 31, 0),
            duration_seconds=73,
        ),
        BuildSummary(
            build_id=f"{user_id}-bld-1038",
            pipeline="test",
            status="passed",
            branch="fix/webhook-retry",
            started_at=datetime(2026, 5, 1, 15, 4, 0),
            duration_seconds=264,
        ),
    ]
    return builds[: max(0, limit)]


async def get_account_status(user_id: str) -> AccountStatus:
    plan_tier = "free"

    if _db_session is not None:
        from app.db.models import User
        user = await _db_session.get(User, user_id)
        if user is not None:
            plan_tier = user.plan_tier

    return AccountStatus(
        user_id=user_id,
        plan_tier=plan_tier,
        concurrent_builds_used=2,
        concurrent_builds_limit=10,
        storage_used_gb=37.4,
        storage_limit_gb=250.0,
    )

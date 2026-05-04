"""
Escalation tools — used by EscalationAgent.

Allows users to escalate issues by creating support tickets.
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session as DbSession
from app.db.models import Ticket
from app.srop.state import SessionState

logger = logging.getLogger(__name__)


@dataclass
class TicketCreated:
    ticket_id: str
    user_id: str
    summary: str
    priority: str
    status: str
    created_at: datetime


# Global db session context (set by the agent/route handler)
_db_session: AsyncSession | None = None


def set_db_session(db: AsyncSession) -> None:
    """Set the database session for tool use."""
    global _db_session
    _db_session = db


async def create_ticket(
    session_id: str,
    user_id: str,
    summary: str,
    priority: str = "medium",
) -> dict:
    """
    Create a support ticket for the user.

    Args:
        user_id: The user's ID (passed from session context)
        summary: Brief description of the issue
        priority: one of "low", "medium", "high", "critical"

    Returns:
        Dict with ticket_id, summary, priority, status, created_at

    This is the function exposed to the ADK agent tool calling.
    """
    if _db_session is None:
        raise RuntimeError("Database session not initialized for create_ticket")

    valid_priorities = {"low", "medium", "high", "critical"}
    if priority not in valid_priorities:
        raise ValueError(f"Priority must be one of {valid_priorities}, got: {priority}")

    ticket_id = f"ticket-{str(uuid.uuid4())[:8]}"
    now = datetime.utcnow()

    # Create and persist to database
    db_ticket = Ticket(
        ticket_id=ticket_id,
        session_id=session_id,
        user_id=user_id,
        summary=summary,
        priority=priority,
        status="open",
        created_at=now,
    )

    try:
        _db_session.add(db_ticket)
        await _db_session.commit()
        await _db_session.refresh(db_ticket)

        # write ticket_id back to session state
        db_session = await _db_session.get(DbSession, session_id)
        if db_session is not None:
            state = SessionState.from_db_dict(
                        db_session.state or {
                            "user_id": db_session.user_id,
                            "plan_tier": db_session.plan_tier,  # read from the users/sessions table
                        }
                    )
            state.latest_ticket_id = db_ticket.ticket_id
            db_session.state = state.to_db_dict()
            db_session.updated_at = datetime.utcnow()
            await _db_session.commit()

        logger.info(
            f"Ticket {db_ticket.ticket_id} created and linked to session {session_id}"
        )
        logger.info(
            f"Ticket created: {ticket_id} for user {user_id} with priority {priority}"
        )

    except Exception as e:
        await _db_session.rollback()
        logger.error(f"Failed to create ticket for user {user_id}: {str(e)}")
        raise

    return {
        "ticket_id": db_ticket.ticket_id,
        "user_id": db_ticket.user_id,
        "summary": db_ticket.summary,
        "priority": db_ticket.priority,
        "status": db_ticket.status,
        "created_at": db_ticket.created_at.isoformat(),
    }

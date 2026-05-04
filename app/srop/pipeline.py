"""
SROP entrypoint - called by the message route.

Pattern choice: Pattern 3 from docs/google-adk-guide.md.
Only the compact persisted SessionState is injected as runtime context, which
keeps the LLM context small, survives process restarts, and avoids rebuilding
full ADK message history on every turn.
"""
import asyncio
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.agents.tools import escalation_tools, account_tools
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message
from app.db.models import Session as DbSession
from app.settings import settings
from app.srop.state import SessionState


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


APP_NAME = "helix_srop"


def _context_instruction(state: SessionState) -> str:
    from app.agents.orchestrator import ROOT_INSTRUCTION

    latest_ticket_line = ""
    if state.latest_ticket_id:
        latest_ticket_line = f"- latest_ticket_id: {state.latest_ticket_id}\n"

    return f"""{ROOT_INSTRUCTION}

Current persisted session context:
- user_id: {state.user_id}
- plan_tier: {state.plan_tier}
- last_agent: {state.last_agent or "none"}
- turn_count: {state.turn_count}
{latest_ticket_line}
Use this context for follow-ups. If the user asks about their account, pass
user_id exactly as shown above to account tools.
"""


def _build_root_agent(state: SessionState) -> Any:
    from google.adk.agents import LlmAgent
    from google.adk.tools.agent_tool import AgentTool

    from app.agents.orchestrator import (
        ACCOUNT_INSTRUCTION,
        ESCALATION_INSTRUCTION,
        KNOWLEDGE_INSTRUCTION,
    )
    from app.agents.tools.account_tools import get_account_status, get_recent_builds
    from app.agents.tools.escalation_tools import create_ticket
    from app.agents.tools.search_docs import search_docs

    knowledge_agent = LlmAgent(
        name="knowledge_agent",
        model=settings.adk_model,
        instruction=KNOWLEDGE_INSTRUCTION,
        tools=[search_docs],
    )
    account_agent = LlmAgent(
        name="account_agent",
        model=settings.adk_model,
        instruction=ACCOUNT_INSTRUCTION,
        tools=[get_recent_builds, get_account_status],
    )
    escalation_agent = LlmAgent(
        name="escalation_agent",
        model=settings.adk_model,
        instruction=ESCALATION_INSTRUCTION,
        tools=[create_ticket],
    )
    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=_context_instruction(state),
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
            AgentTool(agent=escalation_agent)
        ],
    )


def _user_content(text: str) -> Any:
    try:
        from google.genai import types
    except ImportError:
        return {"role": "user", "parts": [{"text": text}]}

    return types.Content(role="user", parts=[types.Part(text=text)])


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    return str(value)


def _content_text(content: Any) -> str:
    parts = content.get("parts", []) if isinstance(content, dict) else getattr(content, "parts", [])
    texts: list[str] = []
    for part in parts or []:
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if text:
            texts.append(str(text))
    return "\n".join(texts).strip()


def _route_from_name(name: str | None) -> str | None:
    if not name:
        return None
    lowered = name.lower()
    if "knowledge" in lowered or "search_docs" in lowered:
        return "knowledge"
    if (
        "account" in lowered
        or "get_recent_builds" in lowered
        or "get_account_status" in lowered
    ):
        return "account"
    if "escalation" in lowered or "create_ticket" in lowered:
        return "escalation"
    if "smalltalk" in lowered:
        return "smalltalk"
    return None


def _event_calls(event: Any) -> list[dict[str, Any]]:
    if hasattr(event, "get_function_calls"):
        return [
            {
                "id": getattr(call, "id", None),
                "tool_name": getattr(call, "name", ""),
                "args": _jsonable(getattr(call, "args", {})),
                "result": None,
            }
            for call in event.get_function_calls()
        ]

    if getattr(event, "type", None) == "tool_call":
        return [
            {
                "id": getattr(event, "tool_call_id", None),
                "tool_name": getattr(event, "tool_name", ""),
                "args": _jsonable(getattr(event, "tool_args", {})),
                "result": None,
            }
        ]

    calls: list[dict[str, Any]] = []
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", []) if content is not None else []
    for part in parts or []:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            calls.append(
                {
                    "id": getattr(function_call, "id", None),
                    "tool_name": getattr(function_call, "name", ""),
                    "args": _jsonable(getattr(function_call, "args", {})),
                    "result": None,
                }
            )
    return calls


def _event_results(event: Any) -> list[dict[str, Any]]:
    if hasattr(event, "get_function_responses"):
        return [
            {
                "id": getattr(response, "id", None),
                "tool_name": getattr(response, "name", ""),
                "result": _jsonable(getattr(response, "response", None)),
            }
            for response in event.get_function_responses()
        ]

    if getattr(event, "type", None) == "tool_result":
        return [
            {
                "id": getattr(event, "tool_call_id", None),
                "tool_name": getattr(event, "tool_name", ""),
                "result": _jsonable(getattr(event, "tool_result", None)),
            }
        ]

    results: list[dict[str, Any]] = []
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", []) if content is not None else []
    for part in parts or []:
        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            results.append(
                {
                    "id": getattr(function_response, "id", None),
                    "tool_name": getattr(function_response, "name", ""),
                    "result": _jsonable(getattr(function_response, "response", None)),
                }
            )
    return results


def _attach_tool_result(tool_calls: list[dict[str, Any]], result: dict[str, Any]) -> None:
    for call in reversed(tool_calls):
        same_id = result.get("id") and call.get("id") == result.get("id")
        same_name = result.get("tool_name") and call.get("tool_name") == result.get("tool_name")
        if call.get("result") is None and (same_id or same_name):
            call["result"] = result.get("result")
            return
    tool_calls.append(
        {
            "tool_name": result.get("tool_name", ""),
            "args": {},
            "result": result.get("result"),
        }
    )


def _chunk_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        chunk_id = value.get("chunk_id")
        if isinstance(chunk_id, str):
            found.append(chunk_id)
        # Also extract ticket_id if present
        ticket_id = value.get("ticket_id")
        if isinstance(ticket_id, str) and ticket_id.startswith("ticket-"):
            found.append(f"[Ticket: {ticket_id}]")
        for item in value.values():
            found.extend(_chunk_ids(item))
    elif isinstance(value, list | tuple | set):
        for item in value:
            found.extend(_chunk_ids(item))
    elif isinstance(value, str):
        found.extend(re.findall(r"\bchunk_[a-f0-9]{16}\b", value))
        for match in re.finditer(r"<!-- chunk_ids: ([^>]+) -->", value):
            found.extend(id_.strip() for id_ in match.group(1).split(",") if id_.strip())
        # Extract ticket references from response text
        found.extend(re.findall(r"\bticket-[a-f0-9]{8}\b", value))
    return list(dict.fromkeys(found))


async def _run_adk_turn(
    session_id: str,
    user_message: str,
    state: SessionState,
) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    from google.adk.runners import InMemoryRunner

    agent = _build_root_agent(state)
    try:
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    except TypeError:
        runner = InMemoryRunner(agent=agent)

    session_service = getattr(runner, "session_service", None)
    if session_service is not None:
        try:
            await session_service.create_session(
                app_name=getattr(runner, "app_name", APP_NAME),
                user_id=state.user_id,
                session_id=session_id,
            )
        except TypeError:
            session = await session_service.create_session(
                app_name=getattr(runner, "app_name", APP_NAME),
                user_id=state.user_id,
            )
            session_id = getattr(session, "id", session_id)

    tool_calls: list[dict[str, Any]] = []
    routed_to: str | None = None
    final_text = ""

    async for event in runner.run_async(
        user_id=state.user_id,
        session_id=session_id,
        new_message=_user_content(user_message),
    ):
        for call in _event_calls(event):
            tool_calls.append(call)
            routed_to = _route_from_name(call.get("tool_name")) or routed_to

        for result in _event_results(event):
            _attach_tool_result(tool_calls, result)

        if event.is_final_response():
            final_text = _content_text(getattr(event, "content", None))
            routed_to = _route_from_name(getattr(event, "author", None)) or routed_to

    if routed_to is None:
        routed_to = "smalltalk"

    retrieved_chunk_ids: list[str] = []
    for call in tool_calls:
        retrieved_chunk_ids.extend(_chunk_ids(call.get("result")))
    retrieved_chunk_ids.extend(_chunk_ids(final_text))

    return final_text, routed_to, tool_calls, list(dict.fromkeys(retrieved_chunk_ids))


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    trace_id = str(uuid.uuid4())
    started = time.perf_counter()

    escalation_tools.set_db_session(db)
    account_tools.set_db_session(db)

    db_session = await db.get(DbSession, session_id)
    if db_session is None:
        raise SessionNotFoundError(f"Session {session_id} was not found")

    state = SessionState.from_db_dict(
            db_session.state or {
                "user_id": db_session.user_id,
                "plan_tier": db_session.plan_tier,  # read from the users/sessions table
            }
    )

    try:
        content, routed_to, tool_calls, retrieved_chunk_ids = await asyncio.wait_for(
            _run_adk_turn(session_id, user_message, state),
            timeout=settings.llm_timeout_seconds,
        )
    except TimeoutError as exc:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        ) from exc

    state.turn_count += 1
    state.last_agent = routed_to  # type: ignore[assignment]

    # Extract ticket ID from tool calls if escalation happened
    if routed_to == "escalation":
        for call in tool_calls:
            if call.get("tool_name") == "create_ticket":
                result = call.get("result")
                if result and isinstance(result, dict):
                    ticket_id = result.get("ticket_id")
                    if ticket_id:
                        state.latest_ticket_id = ticket_id

    db_session.state = state.to_db_dict()
    db_session.updated_at = datetime.utcnow()
    flag_modified(db_session, "state")

    latency_ms = round((time.perf_counter() - started) * 1000)
    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
        )
    )
    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=content,
            trace_id=trace_id,
        )
    )
    db.add(
        AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed_to,
            tool_calls=[
                {key: value for key, value in call.items() if key != "id"}
                for call in tool_calls
            ],
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
        )
    )
    await db.commit()

    return PipelineResult(content=content, routed_to=routed_to, trace_id=trace_id)

async def run_stream(
    session_id: str, user_message: str, db: AsyncSession
) -> AsyncGenerator[str, None]:
    """
    Async generator for SSE streaming.

    Yields:
        event: delta / data: {"chunk": "..."}   — one per text fragment
        event: done  / data: {reply, routed_to, trace_id}   — final frame
    """
    import json

    trace_id = str(uuid.uuid4())
    started = time.perf_counter()

    escalation_tools.set_db_session(db)
    account_tools.set_db_session(db)

    db_session = await db.get(DbSession, session_id)
    if db_session is None:
        raise SessionNotFoundError(f"Session {session_id} was not found")

    state = SessionState.from_db_dict(
            db_session.state or {
                "user_id": db_session.user_id,
                "plan_tier": db_session.plan_tier,  # read from the users/sessions table
            }
    )

    # ── stream ADK events ─────────────────────────────────────────────────────
    tool_calls: list[dict[str, Any]] = []
    routed_to: str | None = None
    full_reply_parts: list[str] = []

    from google.adk.runners import InMemoryRunner

    agent = _build_root_agent(state)
    try:
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    except TypeError:
        runner = InMemoryRunner(agent=agent)

    session_service = getattr(runner, "session_service", None)
    adk_session_id = session_id
    if session_service is not None:
        try:
            await session_service.create_session(
                app_name=getattr(runner, "app_name", APP_NAME),
                user_id=state.user_id,
                session_id=session_id,
            )
        except TypeError:
            adk_session = await session_service.create_session(
                app_name=getattr(runner, "app_name", APP_NAME),
                user_id=state.user_id,
            )
            adk_session_id = getattr(adk_session, "id", session_id)
    try:
        async for event in runner.run_async(
            user_id=state.user_id,
            session_id=adk_session_id,
            new_message=_user_content(user_message),
        ):
            # Tool call tracking (same logic as _run_adk_turn)
            for call in _event_calls(event):
                tool_calls.append(call)
                routed_to = _route_from_name(call.get("tool_name")) or routed_to
            for result in _event_results(event):
                _attach_tool_result(tool_calls, result)

            # Emit text fragments from every event that carries text
            chunk = _content_text(getattr(event, "content", None))
            if chunk:
                full_reply_parts.append(chunk)
                yield f"event: delta\ndata: {json.dumps({'chunk': chunk})}\n\n"

            if event.is_final_response():
                routed_to = _route_from_name(getattr(event, "author", None)) or routed_to

        if routed_to is None:
            routed_to = "smalltalk"

        content = "".join(full_reply_parts)

        # ── persist (identical to run()) ─────────────────────────────────────────
        retrieved_chunk_ids: list[str] = []
        for call in tool_calls:
            retrieved_chunk_ids.extend(_chunk_ids(call.get("result")))
        retrieved_chunk_ids.extend(_chunk_ids(content))
        retrieved_chunk_ids = list(dict.fromkeys(retrieved_chunk_ids))

        state.turn_count += 1
        state.last_agent = routed_to  # type: ignore[assignment]

        if routed_to == "escalation":
            for call in tool_calls:
                if call.get("tool_name") == "create_ticket":
                    result = call.get("result")
                    if result and isinstance(result, dict):
                        ticket_id = result.get("ticket_id")
                        if ticket_id:
                            state.latest_ticket_id = ticket_id

        db_session.state = state.to_db_dict()
        db_session.updated_at = datetime.utcnow()
        flag_modified(db_session, "state")

        latency_ms = round((time.perf_counter() - started) * 1000)
        db.add(Message(message_id=str(uuid.uuid4()), session_id=session_id, role="user", content=user_message, trace_id=trace_id))
        db.add(Message(message_id=str(uuid.uuid4()), session_id=session_id, role="assistant", content=content, trace_id=trace_id))
        db.add(AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed_to,
            tool_calls=[{k: v for k, v in c.items() if k != "id"} for c in tool_calls],
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
        ))
        await db.commit()

        # ── final frame ───────────────────────────────────────────────────────────
        yield f"event: done\ndata: {json.dumps({'reply': content, 'routed_to': routed_to, 'trace_id': trace_id})}\n\n"

    except Exception as exc:
        # Surface the error as an SSE frame so the client sees it
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        raise   # re-raise so FastAPI logs it

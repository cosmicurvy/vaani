"""Chat endpoint for Vaani.

All text messages are routed through CustomerSupportService so greetings,
order lookups, complaints, and knowledge questions follow the same intent
routing and conversation-state logic.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="auto", max_length=30)

    # The frontend must reuse this ID for every turn in the same conversation.
    # If omitted, a new conversation is created for this request.
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=36,
        max_length=36,
    )

    # Kept for compatibility with the current support-service interface.
    chat_history: str = Field(default="", max_length=12000)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    route: str = "unknown"
    awaiting: str | None = None
    supported: bool = False
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_count: int = 0


def _get_support_service(request: Request):
    support_service = getattr(request.app.state, "support_service", None)

    if support_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Customer support service is not configured. "
                "Initialize it during application startup."))

    return support_service


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Process one text message through the customer-support orchestrator."""
    support_service = _get_support_service(request)

    try:
        # Validate and normalize the session ID before passing it to the
        # support service, which uses UUID-based conversation state.
        session_id = str(uuid.UUID(payload.session_id))

        result = await run_in_threadpool(
            support_service.process,
            session_id,
            payload.question,
            payload.language,
            payload.chat_history,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id must be a valid UUID.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Vaani customer-support chat request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process the message right now.") from exc

    if not isinstance(result, dict) or not result.get("answer"):
        logger.error("CustomerSupportService returned an invalid response")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The customer-support service returned an invalid response.",
        )

    return ChatResponse(
        answer=str(result["answer"]),
        session_id=str(result.get("session_id") or session_id),
        route=str(result.get("route") or "unknown"),
        awaiting=result.get("awaiting"),
        supported=bool(result.get("supported", False)),
        citations=result.get("citations") or [],
        retrieved_count=int(result.get("retrieved_count", 0)),
    )


@router.delete("/{session_id}")
async def clear_chat_session(session_id: str, request: Request) -> dict[str, bool]:
    """Delete one conversation's in-memory LangGraph history and pending state."""
    support_service = _get_support_service(request)

    try:
        normalized_session_id = str(uuid.UUID(session_id))
        await run_in_threadpool(
            support_service.clear_session,
            normalized_session_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id must be a valid UUID.",
        ) from exc
    except Exception as exc:
        logger.exception("Could not clear Vaani conversation session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to clear the conversation right now.",
        ) from exc

    return {"cleared": True}

"""Voice endpoint for Vaani.

Audio is transcribed with Faster-Whisper, routed through
CustomerSupportService, and optionally synthesized with Gemini TTS.
"""
from __future__ import annotations

import base64
import logging
import re
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])

MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/webm",
    "audio/ogg",
    "application/octet-stream",
}


def _speakable(text: str) -> str:
    """Remove source citations and Markdown syntax before sending text to TTS."""
    text = re.sub(r"\s*\[S\d+\]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`#>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


class VoiceResponse(BaseModel):
    transcript: str
    answer: str
    language: str
    session_id: str
    route: str = "unknown"
    awaiting: str | None = None
    supported: bool = False
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_count: int = 0
    audio_base64: str | None = None
    audio_content_type: str | None = None


def _get_service(request: Request, service_name: str):
    service = getattr(request.app.state, service_name, None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{service_name.replace('_', ' ').capitalize()} is not configured.",
        )
    return service


@router.post("", response_model=VoiceResponse)
async def voice_chat(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form(default="auto", max_length=30),
    session_id: str = Form(default="", max_length=36),
    chat_history: str = Form(default="", max_length=12000),
) -> VoiceResponse:
    """Transcribe an audio message and process it through customer support."""
    try:
        try:
            session_id = str(uuid.UUID(session_id)) if session_id else str(uuid.uuid4())
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id must be a valid UUID.",
            ) from exc

        if file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported audio content type: {file.content_type}",
            )

        audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty.",
            )
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio file exceeds the 20 MB limit.",
            )

        stt_service = _get_service(request, "stt_service")
        support_service = _get_service(request, "support_service")
        tts_service = getattr(request.app.state, "tts_service", None)

        requested_language = language.strip().lower()
        stt_language = requested_language if requested_language in {"en", "hi"} else None

        try:
            transcript = await run_in_threadpool(
                stt_service.transcribe,
                audio_bytes,
                file.filename or "recording.wav",
                stt_language,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc) or "No speech could be recognized in the uploaded audio.",
            ) from exc

        if not isinstance(transcript, str) or not transcript.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No speech could be recognized in the uploaded audio.",
            )

        result = await run_in_threadpool(
            support_service.process,
            session_id,
            transcript.strip(),
            language,
            chat_history,
        )

        if not isinstance(result, dict) or not result.get("answer"):
            logger.error("CustomerSupportService returned an invalid voice response")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The customer-support service returned an invalid response.",
            )

        audio_b64 = None
        audio_type = None

        if tts_service is not None:
            try:
                synthesized = await run_in_threadpool(
                    tts_service.synthesize,
                    _speakable(str(result["answer"])),
                    language,
                )

                if synthesized:
                    if not isinstance(synthesized, (bytes, bytearray)):
                        raise TypeError("TTS service must return audio bytes.")

                    audio_b64 = base64.b64encode(synthesized).decode("ascii")
                    audio_type = getattr(tts_service, "content_type", "audio/wav")
                    logger.info("TTS audio generated: %s bytes", len(synthesized))
            except Exception:
                logger.exception("Vaani TTS failed; returning text-only response")

        return VoiceResponse(
            transcript=transcript.strip(),
            answer=str(result["answer"]),
            language=language,
            session_id=str(result.get("session_id") or session_id),
            route=str(result.get("route") or "unknown"),
            awaiting=result.get("awaiting"),
            supported=bool(result.get("supported", False)),
            citations=result.get("citations") or [],
            retrieved_count=int(result.get("retrieved_count", 0)),
            audio_base64=audio_b64,
            audio_content_type=audio_type,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Vaani voice request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process the audio request right now.",
        ) from exc
    finally:
        await file.close()
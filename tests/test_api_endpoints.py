import io
import uuid
import wave
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.voice import _speakable, router as voice_router
from backend.services.customer_support import CustomerSupportService

def _make_dummy_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 100)
    return buf.getvalue()


@pytest.fixture
def client(support_service: CustomerSupportService, mock_rag_service: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(voice_router, prefix="/api/v1")
    app.include_router(health_router, prefix="/api/v1")

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "What is the return policy?"

    mock_tts = MagicMock()
    mock_tts.content_type = "audio/wav"
    mock_tts.synthesize.return_value = _make_dummy_wav()

    app.state.support_service = support_service
    app.state.rag_service = mock_rag_service
    app.state.stt_service = mock_stt
    app.state.tts_service = mock_tts

    return TestClient(app)


def test_health_endpoints(client: TestClient):
    live = client.get("/api/v1/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"

    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ok"


def test_chat_and_delete_session_endpoints(client: TestClient):
    session_id = str(uuid.uuid4())

    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is the return policy?",
            "language": "en",
            "session_id": session_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "rag"
    assert data["supported"] is True
    assert data["session_id"] == session_id

    # Clear session
    del_response = client.delete(f"/api/v1/chat/{session_id}")
    assert del_response.status_code == 200
    assert del_response.json() == {"cleared": True}


def test_voice_endpoint_and_speakable_helper(client: TestClient):
    # Verify _speakable strips [S1] citations and Markdown
    cleaned = _speakable("Returns take **7 days** [S1] [S2].")
    assert "[S1]" not in cleaned
    assert "**" not in cleaned
    assert cleaned == "Returns take 7 days."

    wav_bytes = _make_dummy_wav()
    response = client.post(
        "/api/v1/voice",
        files={"file": ("question.wav", wav_bytes, "audio/wav")},
        data={"language": "en", "session_id": str(uuid.uuid4())},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "What is the return policy?"
    assert body["route"] == "rag"
    assert body["audio_base64"] is not None
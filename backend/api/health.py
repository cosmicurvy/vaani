"""Liveness and readiness endpoints."""
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])

class HealthResponse(BaseModel):
    status: str
    services: dict[str, bool]

@router.get("/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Confirms the API process is running; does not load or query models."""
    return HealthResponse(status="ok", services={"api": True})

@router.get("/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    """Reports whether required service objects were attached at startup."""
    services = {
        "support": getattr(request.app.state, "support_service", None) is not None,
        "rag": getattr(request.app.state, "rag_service", None) is not None,
        "stt": getattr(request.app.state, "stt_service", None) is not None,
        "tts": getattr(request.app.state, "tts_service", None) is not None,
    }
   
    ready = all(services.values())
    # ready = services["rag"] and services["stt"] and services["tts"]
    return HealthResponse(
        status="ok" if ready else "not_ready",
        services=services)
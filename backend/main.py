"""FastAPI application entry point.

This file registers the API routers and initializes the RAG, STT, TTS,
memory, and customer-support orchestration services during startup.
"""
from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.voice import router as voice_router
from backend.services.conversation_state import ConversationState
from backend.services.customer_repository import CustomerRepository
from backend.services.customer_support import CustomerSupportService
from backend.services.memory_service import LangGraphMemoryService
from backend.services.order_service import OrderService
from backend.services.rag_service import RAGService
from backend.services.stt_service import STTService
from backend.services.tts_service import TTSService
from backend.core.config import settings
from backend.core.logging import setup_logging

load_dotenv()

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = None
    app.state.stt_service = None
    app.state.tts_service = None
    app.state.memory_service = None
    app.state.support_service = None

    try:
        # 1. RAG service
        rag_service = RAGService()
        app.state.rag_service = rag_service

        # 2. Speech-to-text: Faster-Whisper
        if settings.ENABLE_STT:
            app.state.stt_service = STTService()

        # 3. Text-to-speech: Gemini
        if settings.ENABLE_TTS:
            app.state.tts_service = TTSService()
                    
        # 4. Customer support orchestration & memory services
        repository = CustomerRepository()
        state_manager = ConversationState()
        order_service = OrderService(repository)
        memory_service = LangGraphMemoryService()

        app.state.memory_service = memory_service
        app.state.support_service = CustomerSupportService(
            rag_service=rag_service,
            memory_service=memory_service,
            state_manager=state_manager,
            order_service=order_service)

        yield

    finally:
        app.state.rag_service = None
        app.state.stt_service = None
        app.state.tts_service = None
        app.state.memory_service = None
        app.state.support_service = None

app = FastAPI(
    title="Vaani API",
    description="Voice-based customer support with grounded RAG answers.",
    version="0.1.0",
    lifespan=lifespan)

# Configure explicit trusted frontend origins in deployment.
origins = [origin.strip() 
           for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
           if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"])

app.include_router(chat_router, prefix="/api/v1")
app.include_router(voice_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")

@app.get("/", tags=["root"])
async def root():
    return {
        "name": "Vaani API",
        "docs": "/docs",
        "health": "/api/v1/health/live"}
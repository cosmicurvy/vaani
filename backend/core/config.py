"""Centralized application configuration from environment variables."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings:
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # TTS Settings
    GOOGLE_TTS_MODEL: str = os.getenv("GOOGLE_TTS_MODEL", "gemini-3.1-flash-tts-preview")
    GOOGLE_TTS_VOICE: str = os.getenv("GOOGLE_TTS_VOICE", "Kore")

    # Feature Toggles
    ENABLE_STT: bool = os.getenv("ENABLE_STT", "true").lower() == "true"
    ENABLE_TTS: bool = os.getenv("ENABLE_TTS", "true").lower() == "true"

    # Whisper STT Settings
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # Paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    ORDERS_DB_PATH: Path = DATA_DIR / "voltkart_orders.db"
    SUPPORT_DB_PATH: Path = DATA_DIR / "vaani_support.db"
    QDRANT_PATH: Path = DATA_DIR / "qdrant"
    BM25_PATH: Path = DATA_DIR / "bm25_documents.pkl"
    
    # rag
    COLLECTION_NAME = "vaani_knowledge_base"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    RERANKER_MODEL =  "BAAI/bge-reranker-v2-m3"
    EMBEDDING_DEVICE = "cpu"


settings = Settings()
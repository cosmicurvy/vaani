"""Google Gemini text-to-speech service for Vaani.

Requires: pip install -U google-genai

Environment:
    GOOGLE_API_KEY=...
    GOOGLE_TTS_MODEL=gemini-3.1-flash-tts-preview
    GOOGLE_TTS_VOICE=Kore
"""
from __future__ import annotations

import io
import os
import re
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class TTSService:
    """Generate WAV audio from text using the Gemini Developer API."""

    def __init__(self, api_key: str | None = None, model: str | None = None, voice: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model or os.getenv("GOOGLE_TTS_MODEL", "gemini-3.1-flash-tts-preview")
        self.voice = voice or os.getenv("GOOGLE_TTS_VOICE", "Kore")
        self.content_type = "audio/wav"
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not configured. Add a Gemini API key "
                "from Google AI Studio to your backend .env file.")
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _clean_text_for_speech(text: str) -> str:
        """Strip RAG citations like [S1] and Markdown symbols before synthesis."""
        text = re.sub(r"\s*\[S\d+\]", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[*_`#>]+", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _build_prompt(cls, text: str, language: str) -> str:
        clean_text = cls._clean_text_for_speech(text)
        language = (language or "auto").strip().lower()
        instructions = {
            "hi": (
                "Read the following text aloud naturally in Hindi. "
                "Do not translate, summarize, or add words."
            ),
            "en": (
                "Read the following text aloud naturally in English. "
                "Do not translate, summarize, or add words."
            ),
            "hinglish": (
                "Read the following text aloud naturally in conversational "
                "Hinglish, preserving its Hindi-English code-switching. "
                "Do not translate, summarize, or add words."
            ),
        }
        instruction = instructions.get(
            language,
            "Read the following text aloud naturally in the language and "
            "pronunciation indicated by the text. Do not translate, "
            "summarize, or add words.",
        )
        return f"{instruction}\n\nText:\n{clean_text}"

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes) -> bytes:
        """Gemini TTS returns 24 kHz, mono, 16-bit PCM; package it as WAV."""
        if not pcm_data:
            raise RuntimeError("Gemini TTS returned empty audio data.")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(pcm_data)
        return buffer.getvalue()

    def synthesize(self, text: str, language: str = "auto") -> bytes:
        clean_text = self._clean_text_for_speech(text or "")
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=self._build_prompt(clean_text, language),
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.voice
                            )
                        )
                    ),
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Gemini TTS request failed ({type(exc).__name__}): {exc}"
            ) from exc

        try:
            for candidate in response.candidates or []:
                for part in candidate.content.parts or []:
                    inline_data = part.inline_data
                    if inline_data and inline_data.data:
                        return self._pcm_to_wav(inline_data.data)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "Gemini TTS returned an unexpected response structure."
            ) from exc

        raise RuntimeError(
            "Gemini TTS response did not contain audio. Check the selected "
            "model, API access, and response."
        )
"""Local speech-to-text using Faster-Whisper. No cloud API key required."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from backend.core.config import settings



class STTService:
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None, 
                 compute_type: Optional[str] = None) -> None:
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        # Model loads once when the service is initialized.
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type)

    def transcribe(self, audio_bytes: bytes, filename: str = "recording.wav",
                    language: Optional[str] = None) -> str:
        if not audio_bytes:
            raise ValueError("No audio data was provided.")

        suffix = Path(filename).suffix or ".audio"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            lang = language if language in {"en", "hi"} else None
            segments, _info = self.model.transcribe(
                temp_path,
                language=lang,
                vad_filter=True,
                beam_size=5)

            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            if not transcript:
                raise ValueError("No speech could be recognized in the audio.")
            return transcript
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
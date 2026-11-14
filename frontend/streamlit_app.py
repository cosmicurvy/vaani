"""Vaani Streamlit frontend."""

import base64
import os
import uuid

import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE_URL = os.getenv("VAANI_API_BASE_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")

CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
VOICE_ENDPOINT = f"{API_BASE_URL}/voice"

REQUEST_TIMEOUT = 180
MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB

st.set_page_config(page_title="Vaani | Voice Customer Support", page_icon="🎙️", layout="centered")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "last_audio_type" not in st.session_state:
    st.session_state.last_audio_type = "audio/wav"

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

if "autoplay_audio" not in st.session_state:
    st.session_state.autoplay_audio = False

if "tts_missing_on_last_voice" not in st.session_state:
    st.session_state.tts_missing_on_last_voice = False

if "recording_version" not in st.session_state:
    st.session_state.recording_version = 0


def add_message(role: str, content: str, citations: list | None = None) -> None:
    message = {"role": role, "content": content}
    if citations is not None:
        message["citations"] = citations
    st.session_state.messages.append(message)


def get_api_error(response: requests.Response) -> str:
    """Extract a useful FastAPI error message where possible."""
    try:
        body = response.json()
        detail = body.get("detail", body)
        return str(detail)
    except (ValueError, AttributeError):
        return response.text[:500] or "Unknown API error."


def call_chat_api(question: str, language: str) -> dict:
    """Call POST /chat using the current conversation session."""
    payload = {
        "question": question,
        "language": language,
        "session_id": st.session_state.session_id,
    }

    response = requests.post(
        CHAT_ENDPOINT,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"Chat API returned HTTP {response.status_code}: "
            f"{get_api_error(response)}"
        )

    return response.json()


def call_voice_api(audio_file, language: str) -> dict:
    """Upload a recorded audio file to POST /voice."""
    audio_bytes = audio_file.getvalue()

    if not audio_bytes:
        raise ValueError("The recording is empty. Please record again.")

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError("Recording exceeds the 20 MB upload limit.")

    filename = getattr(audio_file, "name", None) or "recording.wav"
    content_type = getattr(audio_file, "type", None) or "audio/wav"

    response = requests.post(
        VOICE_ENDPOINT,
        files={"file": (filename, audio_bytes, content_type)},
        data={
            "language": language,
            "session_id": st.session_state.session_id,
        },
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"Voice API returned HTTP {response.status_code}: "
            f"{get_api_error(response)}"
        )

    return response.json()


def play_once(audio_bytes: bytes, mime_type: str) -> None:
    """Play audio a single time using a JS Audio object.

    Unlike st.audio(autoplay=True), this leaves no <audio autoplay> element in
    the page, so the browser has nothing to restart later (for example when the
    microphone is activated for a new recording).
    """
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    components.html(
        f"""
        <script>
        const audio = new Audio("data:{mime_type};base64,{b64}");
        audio.play().catch(() => {{}});
        </script>
        """,
        height=0,
    )


def show_citations(citations: list) -> None:
    if not citations:
        return

    with st.expander("Sources"):
        for citation in citations:
            source = citation.get("source", "Unknown source")
            page = citation.get("page")
            faq_id = citation.get("faq_id")

            details = [source]
            if page is not None:
                details.append(f"Page {page}")
            if faq_id:
                details.append(f"FAQ: {faq_id}")

            st.markdown("- " + " · ".join(details))


# Header and sidebar controls
st.title("🎙️ Vaani")
st.caption(
    "Voice-based customer support chatbot, "
    "powered by retrieval-augmented generation."
)

with st.sidebar:
    st.header("Settings")

    language = st.selectbox(
        "Response language",
        options=["auto", "en", "hinglish"],
        format_func=lambda value: {
            "auto": "Automatic",
            "en": "English",
            "hinglish": "Hinglish",
        }[value],
    )

    st.caption(
        "Automatic mode lets Whisper detect the spoken language. "
        "English and Hinglish provide a transcription language hint."
    )

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        try:
            clear_response = requests.delete(
                f"{CHAT_ENDPOINT}/{st.session_state.session_id}",
                timeout=REQUEST_TIMEOUT,
            )
            if not clear_response.ok:
                st.warning(
                    "Could not clear server-side memory. "
                    "A new session will still be started."
                )
        except requests.RequestException:
            st.warning("Could not reach the API to clear server-side memory.")

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.last_audio = None
        st.session_state.last_audio_type = "audio/wav"
        st.session_state.last_transcript = ""
        st.session_state.autoplay_audio = False
        st.session_state.tts_missing_on_last_voice = False

        # The audio input widget key includes recording_version, so bumping it
        # gives a fresh widget and clears any stale recording.
        st.session_state.recording_version += 1

        st.rerun()

    st.caption("History is kept in this browser session.")


mode = st.radio(
    "Choose interaction mode",
    options=["Text chat", "Voice mode"],
    horizontal=True,
)

# Conversation display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_citations(message.get("citations") or [])


# Text mode
if mode == "Text chat":
    question = st.chat_input("Ask Vaani a question...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Checking on that.."):
                try:
                    result = call_chat_api(question, language)
                    answer = result.get("answer", "").strip()
                    st.session_state.session_id = (
                        result.get("session_id") or st.session_state.session_id
                    )

                    if not answer:
                        raise RuntimeError("The chat API returned an empty answer.")

                    citations = result.get("citations") or []
                    st.markdown(answer)
                    show_citations(citations)

                    # Save turn only after a successful response.
                    add_message("user", question)
                    add_message("assistant", answer, citations)

                except Exception as exc:
                    st.error(str(exc))


# Voice mode: record, then send
else:
    st.markdown("### Ask Vaani using your voice")
    st.write(
        "1. Record your question using the microphone below.\n"
        "2. Review the recording if needed.\n"
        "3. Press **Send recording** to get a spoken answer."
    )

    recording = st.audio_input(
        "Record your question",
        key=f"voice_recording_{st.session_state.recording_version}",
    )

    if recording is not None:
        st.audio(recording)

        if st.button(
            "Send recording",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Checking on that..."
                ):
                    result = call_voice_api(recording, language)

                st.session_state.session_id = (
                    result.get("session_id") or st.session_state.session_id
                )
                transcript = result.get("transcript", "").strip()
                answer = result.get("answer", "").strip()

                if not transcript:
                    raise RuntimeError(
                        "No transcript was returned. Please try recording again."
                    )

                if not answer:
                    raise RuntimeError("The voice API returned an empty answer.")

                add_message("user", transcript)
                add_message(
                    "assistant",
                    answer,
                    result.get("citations") or [],
                )

                st.session_state.last_transcript = transcript

                audio_b64 = result.get("audio_base64")
                audio_type = result.get("audio_content_type") or "audio/wav"

                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64, validate=True)

                    if not audio_bytes.startswith(b"RIFF"):
                        raise RuntimeError("The API returned invalid WAV audio.")

                    st.session_state.last_audio = audio_bytes
                    st.session_state.autoplay_audio = True
                    st.session_state.tts_missing_on_last_voice = False
                else:
                    st.session_state.last_audio = None
                    st.session_state.autoplay_audio = False
                    st.session_state.tts_missing_on_last_voice = True

                st.session_state.last_audio_type = audio_type

                # Create a fresh audio input widget
                st.session_state.recording_version += 1
                st.rerun()

            except Exception as exc:
                st.error(f"Voice request failed: {exc}")

    # Show the latest voice response after the rerun.
    if st.session_state.last_transcript:
        st.markdown("#### Recognized speech")
        st.write(st.session_state.last_transcript)

    if st.session_state.last_audio:
        st.markdown("#### Vaani's spoken response")

        if st.session_state.autoplay_audio:
            # Speak exactly once, on the run right after sending.
            play_once(
                st.session_state.last_audio,
                st.session_state.last_audio_type,
            )
            st.session_state.autoplay_audio = False

        # Normal player with NO autoplay. The user can press play to replay.
        st.audio(
            st.session_state.last_audio,
            format=st.session_state.last_audio_type,
        )
    elif st.session_state.tts_missing_on_last_voice:
        st.info(
            "The answer is available as text above, but no speech audio was returned. "
            "Check whether TTS is enabled and configured in FastAPI."
        )
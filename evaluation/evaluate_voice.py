"""Voice quality evaluation for Vaani: WER, STT/TTS Failure Rate, and End-to-End Latency."""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.conversation_state import ConversationState
from backend.services.customer_repository import CustomerRepository
from backend.services.customer_support import CustomerSupportService
from backend.services.memory_service import LangGraphMemoryService
from backend.services.order_service import OrderService
from backend.services.rag_service import RAGService
from backend.services.stt_service import STTService
from backend.services.tts_service import TTSService

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "evaluation_data" / "voice_eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"


def normalize_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split into word tokens for WER calculation."""
    text = (text or "").lower().strip()
    # Keep alphanumeric and Devanagari characters
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    return text.split()


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) = (S + D + I) / N using Levenshtein distance."""
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Dynamic programming matrix of size (len(ref)+1) x (len(hyp)+1)
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    edit_distance = dp[len(ref_words)][len(hyp_words)]
    return edit_distance / len(ref_words)


def percentile(values: list[float], pct: float) -> float:
    """Compute percentile (e.g. 50 or 95) from a list of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = (len(sorted_vals) - 1) * (pct / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = index - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def run_voice_evaluation() -> dict[str, Any]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as fp:
        samples = json.load(fp)

    print("Initializing Vaani services for Voice Evaluation...")
    stt_service = STTService()
    tts_service = TTSService()
    rag_service = RAGService()
    support_service = CustomerSupportService(
        rag_service=rag_service,
        memory_service=LangGraphMemoryService(),
        state_manager=ConversationState(),
        order_service=OrderService(CustomerRepository()),
    )

    total_samples = len(samples)
    stt_attempts = 0
    stt_failures = 0
    tts_attempts = 0
    tts_failures = 0

    wer_scores: list[float] = []
    stt_latencies_ms: list[float] = []
    support_latencies_ms: list[float] = []
    tts_latencies_ms: list[float] = []
    e2e_latencies_ms: list[float] = []

    detailed_results: list[dict[str, Any]] = []

    print(f"\nEvaluating {total_samples} voice samples...\n" + "-" * 60)

    for sample in samples:
        sample_id = sample["id"]
        language = sample.get("language", "auto")
        ground_truth = sample["ground_truth"]
        audio_path = sample.get("audio_path")

        record: dict[str, Any] = {
            "id": sample_id,
            "language": language,
            "ground_truth": ground_truth,
            "transcript": None,
            "wer": None,
            "stt_success": False,
            "tts_response_success": False,
            "stt_latency_ms": None,
            "support_latency_ms": None,
            "tts_latency_ms": None,
            "e2e_latency_ms": None,
            "error": None,
        }

        # Step 1: Obtain input audio (either from file or synthesized via TTS)
        input_audio_bytes: bytes | None = None
        if audio_path:
            path_obj = PROJECT_ROOT / audio_path
            input_audio_bytes = path_obj.read_bytes()
        else:
            try:
                input_audio_bytes = tts_service.synthesize(ground_truth, language=language)
                # Pause briefly to avoid hitting Gemini TTS free-tier rate limits
                time.sleep(1.0)
            except Exception as exc:
                record["error"] = f"Input audio generation failed: {exc}"
                detailed_results.append(record)
                print(f"[{sample_id}] Skipped (input TTS error): {exc}")
                continue

        # Step 2: Evaluate STT (Transcription + STT Latency + WER)
        stt_attempts += 1
        stt_lang_hint = language if language in {"en", "hi"} else None
        t0_stt = time.perf_counter()
        try:
            transcript = stt_service.transcribe(
                audio_bytes=input_audio_bytes,
                filename=f"{sample_id}.wav",
                language=stt_lang_hint,
            )
            stt_ms = (time.perf_counter() - t0_stt) * 1000.0
            stt_latencies_ms.append(stt_ms)

            wer = calculate_wer(ground_truth, transcript)
            wer_scores.append(wer)

            record["transcript"] = transcript
            record["wer"] = round(wer, 4)
            record["stt_success"] = True
            record["stt_latency_ms"] = round(stt_ms, 2)
        except Exception as exc:
            stt_failures += 1
            record["error"] = f"STT failed: {exc}"
            detailed_results.append(record)
            print(f"[{sample_id}] STT FAILED: {exc}")
            continue

        # Step 3: Evaluate CustomerSupportService / RAG Latency
        session_id = str(uuid.uuid4())
        t0_support = time.perf_counter()
        try:
            support_res = support_service.process(
                session_id=session_id,
                question=transcript,
                language=language,
            )
            support_ms = (time.perf_counter() - t0_support) * 1000.0
            support_latencies_ms.append(support_ms)
            record["support_latency_ms"] = round(support_ms, 2)
            answer_text = support_res["answer"]
        except Exception as exc:
            record["error"] = f"Support service failed: {exc}"
            detailed_results.append(record)
            print(f"[{sample_id}] SUPPORT FAILED: {exc}")
            continue

        # Step 4: Evaluate Output TTS (Synthesis + TTS Failure Rate + TTS Latency)
        tts_attempts += 1
        t0_tts = time.perf_counter()
        try:
            output_wav = tts_service.synthesize(answer_text, language=language)
            tts_ms = (time.perf_counter() - t0_tts) * 1000.0

            if not output_wav or not output_wav.startswith(b"RIFF"):
                raise RuntimeError("Invalid or empty WAV bytes returned")

            tts_latencies_ms.append(tts_ms)
            e2e_ms = stt_ms + support_ms + tts_ms
            e2e_latencies_ms.append(e2e_ms)

            record["tts_response_success"] = True
            record["tts_latency_ms"] = round(tts_ms, 2)
            record["e2e_latency_ms"] = round(e2e_ms, 2)
        except Exception as exc:
            tts_failures += 1
            record["error"] = f"Output TTS failed: {exc}"

        detailed_results.append(record)
        print(
            f"[{sample_id}] WER: {record['wer']:.2%} | "
            f"STT: {record['stt_latency_ms']} ms | "
            f"Support: {record['support_latency_ms']} ms | "
            f"TTS: {record['tts_latency_ms']} ms | "
            f"E2E: {record['e2e_latency_ms']} ms"
        )
        time.sleep(1.0)

    # Aggregate Summary Metrics
    summary = {
        "total_samples": total_samples,
        "wer_mean": round(statistics.mean(wer_scores), 4) if wer_scores else None,
        "wer_median": round(statistics.median(wer_scores), 4) if wer_scores else None,
        "stt_failure_rate": round(stt_failures / stt_attempts, 4) if stt_attempts else 0.0,
        "tts_failure_rate": round(tts_failures / tts_attempts, 4) if tts_attempts else 0.0,
        "latency_ms": {
            "stt_mean": round(statistics.mean(stt_latencies_ms), 2) if stt_latencies_ms else None,
            "stt_p95": round(percentile(stt_latencies_ms, 95), 2) if stt_latencies_ms else None,
            "support_mean": round(statistics.mean(support_latencies_ms), 2) if support_latencies_ms else None,
            "tts_mean": round(statistics.mean(tts_latencies_ms), 2) if tts_latencies_ms else None,
            "tts_p95": round(percentile(tts_latencies_ms, 95), 2) if tts_latencies_ms else None,
            "e2e_mean": round(statistics.mean(e2e_latencies_ms), 2) if e2e_latencies_ms else None,
            "e2e_p50": round(percentile(e2e_latencies_ms, 50), 2) if e2e_latencies_ms else None,
            "e2e_p95": round(percentile(e2e_latencies_ms, 95), 2) if e2e_latencies_ms else None,
        },
        "samples": detailed_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "voice_evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("VOICE QUALITY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mean Word Error Rate (WER) : {(summary['wer_mean'] or 0) * 100:.2f}%")
    print(f"STT Failure Rate           : {summary['stt_failure_rate'] * 100:.2f}% ({stt_failures}/{stt_attempts})")
    print(f"TTS Failure Rate           : {summary['tts_failure_rate'] * 100:.2f}% ({tts_failures}/{tts_attempts})")
    print(f"Avg STT Latency            : {summary['latency_ms']['stt_mean']} ms")
    print(f"Avg Support/RAG Latency    : {summary['latency_ms']['support_mean']} ms")
    print(f"Avg TTS Latency            : {summary['latency_ms']['tts_mean']} ms")
    print(f"End-to-End Latency (Mean)  : {summary['latency_ms']['e2e_mean']} ms")
    print(f"End-to-End Latency (P50)   : {summary['latency_ms']['e2e_p50']} ms")
    print(f"End-to-End Latency (P95)   : {summary['latency_ms']['e2e_p95']} ms")
    print(f"\nSaved full JSON report to  : {report_path}")

    return summary


if __name__ == "__main__":
    run_voice_evaluation()
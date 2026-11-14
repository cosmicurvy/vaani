"""RAG quality evaluation for Vaani using DeepEval and Groq as the LLM Judge.

Metrics evaluated:
1. FaithfulnessMetric
2. AnswerRelevancyMetric
3. ContextualRelevancyMetric
4. ContextualPrecisionMetric
5. ContextualRecallMetric
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from backend.services.rag_service import RAGService

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "evaluation_data" / "rag_eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"


def _extract_json_str(text: str) -> str:
    """Extract a raw JSON object/array from LLM output, stripping markdown fences."""
    text = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find outermost { ... } or [ ... ]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]
    return text


class GroqDeepEvalJudge(DeepEvalBaseLLM):
    """Custom DeepEval LLM Judge backed by ChatGroq supporting DeepEval's schema parameter."""

    def __init__(self, model_name: str | None = None) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing in your .env file.")

        self.model_name = model_name or os.getenv(
            "DEEPEVAL_GROQ_MODEL", "qwen/qwen3.8-27b"
        )
        self._client = ChatGroq(
            model=self.model_name,
            temperature=0,
            api_key=api_key,
            max_retries=3,
        )

    def load_model(self):
        return self._client

    def generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> str | BaseModel:
        if schema is not None:
            # Instruct Groq to return JSON matching DeepEval's requested Pydantic schema
            try:
                structured_llm = self._client.with_structured_output(
                    schema, method="json_mode"
                )
                json_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Respond ONLY with a valid JSON object matching the required schema."
                )
                return structured_llm.invoke(json_prompt)
            except Exception:
                # Fallback: invoke raw model, clean markdown fences, and parse into schema
                response = self._client.invoke(
                    f"{prompt}\n\nReturn ONLY raw valid JSON without markdown code blocks."
                )
                raw_text = str(getattr(response, "content", response))
                cleaned_json = _extract_json_str(raw_text)
                return schema.model_validate_json(cleaned_json)

        response = self._client.invoke(prompt)
        raw_text = str(getattr(response, "content", response))
        return _extract_json_str(raw_text)

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> str | BaseModel:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return f"Groq ({self.model_name})"


def run_rag_evaluation() -> dict[str, Any]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"RAG evaluation dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as fp:
        dataset = json.load(fp)

    print("Initializing Vaani RAGService and Groq DeepEval Judge...")
    rag_service = RAGService()
    judge = GroqDeepEvalJudge()

    threshold = 0.7
    metrics_map = {
        "faithfulness": FaithfulnessMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
        "contextual_precision": ContextualPrecisionMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
        "contextual_recall": ContextualRecallMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
    }

    score_buckets: dict[str, list[float]] = {name: [] for name in metrics_map}
    detailed_results: list[dict[str, Any]] = []

    print(f"\nEvaluating {len(dataset)} RAG test cases...\n" + "-" * 65)

    for item in dataset:
        case_id = item["id"]
        question = item["question"]
        expected_output = item["expected_output"]
        language = item.get("language", "auto")

        # 1. Retrieve context chunks directly from VaaniRetriever
        retrieved_docs = rag_service.retriever.search(question)
        retrieval_context = [doc.page_content for doc in retrieved_docs]

        # 2. Generate grounded answer from LLMService
        gen_result = rag_service.llm_service.generate(
            question=question,
            documents=retrieved_docs,
            language=language,
            chat_history="",
        )
        raw_answer = gen_result["answer"]
        # Strip [S1] citation tags for DeepEval so they aren't treated as unverified claims
        clean_actual_output = re.sub(r"\s*\[S\d+\]", "", raw_answer).strip()

        # 3. Build DeepEval LLMTestCase
        test_case = LLMTestCase(
            input=question,
            actual_output=clean_actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        case_scores: dict[str, Any] = {}
        for metric_name, metric in metrics_map.items():
            try:
                metric.measure(test_case)
                score = float(metric.score if metric.score is not None else 0.0)
                score_buckets[metric_name].append(score)
                case_scores[metric_name] = {
                    "score": round(score, 4),
                    "passed": bool(score >= threshold),
                    "reason": getattr(metric, "reason", ""),
                }
            except Exception as exc:
                print(f"  [!] {case_id} - {metric_name} error: {exc}")
                case_scores[metric_name] = {
                    "score": None,
                    "passed": False,
                    "error": str(exc),
                }
            # Pause briefly to stay within Groq free-tier TPM/RPM limits
            time.sleep(2.0)

        detailed_results.append(
            {
                "id": case_id,
                "language": language,
                "question": question,
                "expected_output": expected_output,
                "actual_output": raw_answer,
                "supported": gen_result["supported"],
                "citations": gen_result["citations"],
                "retrieved_chunks_count": len(retrieval_context),
                "metrics": case_scores,
            }
        )

        print(
            f"[{case_id}] "
            f"Faithfulness: {case_scores['faithfulness']['score']} | "
            f"AnsRel: {case_scores['answer_relevancy']['score']} | "
            f"CtxRel: {case_scores['contextual_relevancy']['score']} | "
            f"CtxPrec: {case_scores['contextual_precision']['score']} | "
            f"CtxRec: {case_scores['contextual_recall']['score']}"
        )

    # Compute overall average scores
    averages = {
        name: round(statistics.mean(scores), 4) if scores else None
        for name, scores in score_buckets.items()
    }

    summary = {
        "judge_model": judge.get_model_name(),
        "total_test_cases": len(dataset),
        "threshold": threshold,
        "average_scores": averages,
        "test_cases": detailed_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "rag_evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("DEEPEVAL RAG QUALITY SUMMARY")
    print("=" * 65)
    print(f"Faithfulness         : {averages['faithfulness']}")
    print(f"Answer Relevancy     : {averages['answer_relevancy']}")
    print(f"Contextual Relevancy : {averages['contextual_relevancy']}")
    print(f"Contextual Precision : {averages['contextual_precision']}")
    print(f"Contextual Recall    : {averages['contextual_recall']}")
    print(f"\nSaved full JSON report to: {report_path}")

    return summary


if __name__ == "__main__":
    run_rag_evaluation()
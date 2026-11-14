"""Multi-turn conversation and memory evaluation for Vaani using DeepEval 4.3.3.

Metrics evaluated:
1. TurnRelevancyMetric (Conversation Relevancy)
2. KnowledgeRetentionMetric
3. ConversationCompletenessMetric
4. RoleAdherenceMetric
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
import uuid
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
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TurnRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn

from backend.services.conversation_state import ConversationState
from backend.services.customer_repository import CustomerRepository
from backend.services.customer_support import CustomerSupportService
from backend.services.memory_service import LangGraphMemoryService
from backend.services.order_service import OrderService
from backend.services.rag_service import RAGService

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "evaluation_data" /"conversation_eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

VAANI_CHATBOT_ROLE = (
    "Vaani is a polite, grounded bilingual (English and Hinglish) customer-support "
    "assistant for VoltKart, an Indian electronics e-commerce store. Vaani helps "
    "customers track orders (after verifying their Order ID, customer name, and city), "
    "answers questions strictly based on VoltKart's return, refund, shipping, "
    "cancellation, warranty, and installation policies without inventing facts, "
    "and politely declines off-topic queries outside customer support."
)


def _extract_json_str(text: str) -> str:
    """Extract a raw JSON object from LLM output, stripping markdown fences."""
    text = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]
    return text


class GroqDeepEvalJudge(DeepEvalBaseLLM):
    """Custom DeepEval 4.3.3 LLM Judge backed by ChatGroq."""

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


def run_conversation_evaluation() -> dict[str, Any]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Conversation evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as fp:
        scenarios = json.load(fp)

    print("Initializing Vaani CustomerSupportService and Groq DeepEval Judge...")
    rag_service = RAGService()
    memory_service = LangGraphMemoryService()
    state_manager = ConversationState()
    order_service = OrderService(CustomerRepository())

    support_service = CustomerSupportService(
        rag_service=rag_service,
        memory_service=memory_service,
        state_manager=state_manager,
        order_service=order_service,
    )

    judge = GroqDeepEvalJudge()
    threshold = 0.7

    metrics_map = {
        "conversation_relevancy": TurnRelevancyMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
        "knowledge_retention": KnowledgeRetentionMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
        "conversation_completeness": ConversationCompletenessMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
        "role_adherence": RoleAdherenceMetric(
            threshold=threshold, model=judge, include_reason=True, async_mode=False
        ),
    }

    score_buckets: dict[str, list[float]] = {name: [] for name in metrics_map}
    detailed_results: list[dict[str, Any]] = []

    print(f"\nEvaluating {len(scenarios)} multi-turn conversations...\n" + "-" * 70)

    for scenario in scenarios:
        conv_id = scenario["id"]
        description = scenario.get("description", "")
        language = scenario.get("language", "auto")
        user_turns = scenario["user_turns"]

        session_id = str(uuid.uuid4())
        deepeval_turns: list[Turn] = []
        dialogue_log: list[dict[str, Any]] = []

        # 1. Simulate the multi-turn conversation through CustomerSupportService
        for turn_idx, user_message in enumerate(user_turns, start=1):
            result = support_service.process(
                session_id=session_id,
                question=user_message,
                language=language,
            )
            raw_answer = result["answer"]
            clean_answer = re.sub(r"\s*\[S\d+\]", "", raw_answer).strip()

            deepeval_turns.append(Turn(role="user", content=user_message))
            deepeval_turns.append(Turn(role="assistant", content=clean_answer))

            dialogue_log.append(
                {
                    "turn": turn_idx,
                    "user": user_message,
                    "assistant": raw_answer,
                    "route": result.get("route"),
                    "awaiting": result.get("awaiting"),
                }
            )
            time.sleep(1.0)

        # Clean up session state after finishing the conversation
        support_service.clear_session(session_id)

        # 2. Build DeepEval 4.3.3 ConversationalTestCase
        convo_test_case = ConversationalTestCase(
            turns=deepeval_turns,
            chatbot_role=VAANI_CHATBOT_ROLE,
        )

        # 3. Measure all 4 multi-turn metrics
        conv_scores: dict[str, Any] = {}
        for metric_name, metric in metrics_map.items():
            try:
                metric.measure(convo_test_case)
                score = float(metric.score if metric.score is not None else 0.0)
                score_buckets[metric_name].append(score)
                conv_scores[metric_name] = {
                    "score": round(score, 4),
                    "passed": bool(score >= threshold),
                    "reason": getattr(metric, "reason", ""),
                }
            except Exception as exc:
                print(f"  [!] {conv_id} - {metric_name} error: {exc}")
                conv_scores[metric_name] = {
                    "score": None,
                    "passed": False,
                    "error": str(exc),
                }
            time.sleep(2.0)

        detailed_results.append(
            {
                "id": conv_id,
                "description": description,
                "language": language,
                "num_user_turns": len(user_turns),
                "dialogue": dialogue_log,
                "metrics": conv_scores,
            }
        )

        print(
            f"[{conv_id}] "
            f"ConvRel: {conv_scores['conversation_relevancy']['score']} | "
            f"KnowRet: {conv_scores['knowledge_retention']['score']} | "
            f"Completeness: {conv_scores['conversation_completeness']['score']} | "
            f"RoleAdh: {conv_scores['role_adherence']['score']}"
        )

    averages = {
        name: round(statistics.mean(scores), 4) if scores else None
        for name, scores in score_buckets.items()
    }

    summary = {
        "judge_model": judge.get_model_name(),
        "deepeval_version": "4.3.3",
        "total_conversations": len(scenarios),
        "threshold": threshold,
        "average_scores": averages,
        "conversations": detailed_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "conversation_evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DEEPEVAL MULTI-TURN CONVERSATION & MEMORY SUMMARY")
    print("=" * 70)
    print(f"Conversation Relevancy    : {averages['conversation_relevancy']}")
    print(f"Knowledge Retention       : {averages['knowledge_retention']}")
    print(f"Conversation Completeness : {averages['conversation_completeness']}")
    print(f"Role Adherence            : {averages['role_adherence']}")
    print(f"\nSaved full JSON report to : {report_path}")

    return summary


if __name__ == "__main__":
    run_conversation_evaluation()
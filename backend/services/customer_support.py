from __future__ import annotations

import uuid

from backend.services.conversation_state import ConversationState
from backend.services.direct_response import DirectResponse
from backend.services.intent_classifier import IntentClassifier
from backend.services.order_service import OrderService

class CustomerSupportService:
    """Routes a message to direct handling, order lookup or existing RAG."""

    def __init__(self, rag_service, memory_service=None,
                  state_manager: ConversationState | None = None,
                order_service: OrderService | None = None):
        self.rag_service = rag_service
        self.memory_service = memory_service
        self.state_manager = state_manager or ConversationState()
        self.order_service = order_service or OrderService()
        self.classifier = IntentClassifier()

    @staticmethod
    def _response(answer, route, session_id, awaiting=None, supported=False, citations=None, retrieved_count=0):
        return {
            "answer": answer,
            "route": route,
            "session_id": session_id,
            "awaiting": awaiting,
            "supported": supported,
            "citations": citations or [],
            "retrieved_count": retrieved_count}

    def _save_turn(self, session_id, question, answer):
        if self.memory_service is None:
            return
        
        self.memory_service.save_turn(session_id, question, answer)
    
    def clear_session(self, session_id):
        """Clear LangGraph chat history and pending support workflow state."""
        session_id = str(uuid.UUID(str(session_id)))
        if self.memory_service is not None:
            self.memory_service.clear_session(session_id)
        self.state_manager.clear_state(session_id)

    def _history(self, session_id, fallback=""):
        """Load server-side LangGraph history; ignore client history fallback."""
        if self.memory_service is None:
            return ""
        
        return self.memory_service.get_history(session_id)

    def _finish(self, session_id, question_to_log, answer, route, **kwargs):
        # Verification messages are deliberately redacted by callers.
        self._save_turn(session_id, question_to_log, answer)
        return self._response(
            answer=answer,
            route=route,
            session_id=session_id,
            **kwargs)

    @staticmethod
    def _format_order(order: dict) -> str:
        lines = [
            f"Order ID: {order['order_id']}",
            f"Items: {order['items']}",
            f"Order status: {order['status'].replace('_', ' ').title()}",
            f"Order date: {order['placed_on']}",
        ]

        optional_fields = [
            ("Estimated delivery", "estimated_delivery"),
            ("Courier", "courier"),
            ("Tracking ID", "tracking_id"),
            ("Latest tracking scan", "last_scan"),
            ("Delivered on", "delivered_on"),
            ("Cancelled on", "cancelled_on"),
            ("Installation status", "installation_status"),
            ("Installation date", "installation_date"),
            ("Return status", "return_status"),
            ("Refund status", "refund_status"),
            ("Refund method", "refund_method"),
            ("Refund amount", "refund_amount_inr"),
            ("Refund initiated on", "refund_initiated_on"),
            ("Expected refund date", "refund_expected_by"),
        ]

        for label, key in optional_fields:
            value = order.get(key)
            if value is not None and value != "":
                if key in {"return_status", "refund_status", "installation_status"}:
                    value = str(value).replace("_", " ").title()
                if key == "refund_amount_inr":
                    value = f"₹{value}"
                lines.append(f"{label}: {value}")

        return "Here are the verified details for your order:\n\n" + "\n".join(lines)

    @staticmethod
    def _parse_name_and_city(message: str):
        # Ask the user to submit: "Aarav Sharma, Delhi"
        if "," not in message:
            return None, None
        name, city = message.rsplit(",", 1)
        name, city = name.strip(), city.strip()
        if not name or not city:
            return None, None
        return name, city

    def process(self, session_id, question, language="auto", chat_history=""):
        session_id = str(uuid.UUID(str(session_id)))
        question = (question or "").strip()

        if not question:
            return self._response(
                "Please enter a message.",
                "clarification",
                session_id)

        state = self.state_manager.get_state(session_id)
        pending = state.get("pending_action")
        context = state.get("context") or {}

        # 1. Explicit cancel keywords
        if pending and question.lower() in {"cancel", "stop", "never mind", "nevermind"}:
            self.state_manager.clear_state(session_id)
            answer = "No problem. I've cancelled the order lookup. How else can I help?"
            return self._finish(session_id, question, answer, "direct")

        # 2. Handle pending order ID step
        if pending == "awaiting_order_id":
            order_id = self.classifier.extract_order_id(question)

            if order_id:
                self.state_manager.set_state(
                    session_id,
                    "awaiting_order_verification",
                    {"order_id": order_id},
                )
                answer = (
                    "Thank you. To verify the order for this demo, "
                    "please enter the customer name and city separated "
                    "by a comma. For example: Aarav Sharma, Delhi"
                )
                return self._finish(
                    session_id,
                    question,
                    answer,
                    "order_lookup",
                    awaiting="customer_name_and_city",
                )

            # Check if the user changed the topic instead of providing an order ID
            check_result = self.classifier.classify(question)
            if check_result.intent in {"greeting", "farewell", "policy_question", "product_question",
                                       "complaint", "human_support", "unknown"}:
                self.state_manager.clear_state(session_id)
            else:
                answer = (
                    "Please enter your order ID in the format VK10001 "
                    "(or type 'cancel' to exit order lookup)."
                )
                return self._finish(
                    session_id,
                    question,
                    answer,
                    "order_lookup",
                    awaiting="order_id",
                )

        # 3. Handle pending Name, City verification step
        elif pending == "awaiting_order_verification":
            name, city = self._parse_name_and_city(question)

            if name and city:
                result = self.order_service.lookup(
                    order_id=context.get("order_id", ""),
                    customer_name=name,
                    city=city,
                )

                if not result["success"]:
                    return self._finish(
                        session_id,
                        "[customer verification details omitted]",
                        result["message"] + " (Or ask another question to exit order lookup.)",
                        "order_lookup",
                        awaiting="customer_name_and_city",
                    )

                self.state_manager.clear_state(session_id)
                answer = self._format_order(result["order"])

                return self._finish(
                    session_id,
                    "[customer verification completed]",
                    answer,
                    "order_lookup",
                )

            # No comma found: check if the user switched topics or said hi/bye
            check_result = self.classifier.classify(question)
            if (
                check_result.intent in {
                    "greeting",
                    "farewell",
                    "policy_question",
                    "product_question",
                    "complaint",
                    "human_support",
                    "unknown",
                }
                or "?" in question
                or len(question.split()) >= 5
            ):
                # User moved on to a new question or greeting/farewell; clear pending state!
                self.state_manager.clear_state(session_id)
            else:
                answer = (
                    "Please enter both details separated by a comma "
                    "(for example: Aarav Sharma, Delhi), or type 'cancel' to ask something else."
                )
                return self._finish(
                    session_id,
                    "[customer verification details omitted]",
                    answer,
                    "order_lookup",
                    awaiting="customer_name_and_city",
                )

        # 4. Normal intent classification & routing (runs immediately if state was cleared above)
        result = self.classifier.classify(question)
        intent = result.intent

        if intent in {"greeting", "farewell", "off_topic"}:
            answer = DirectResponse.generate(intent, question, language)
            return self._finish(session_id, question, answer, "direct")

        if intent == "order_status":
            order_id = result.order_id

            if not order_id:
                self.state_manager.set_state(
                    session_id,
                    "awaiting_order_id",
                    {},
                )
                answer = (
                    "Sure, I can help track your order. "
                    "Please share your order ID, for example VK10001."
                )
                return self._finish(
                    session_id,
                    question,
                    answer,
                    "order_lookup",
                    awaiting="order_id",
                )

            self.state_manager.set_state(
                session_id,
                "awaiting_order_verification",
                {"order_id": order_id},
            )
            answer = (
                "I found the order ID format. To verify the order for "
                "this demo, please enter the customer name and city "
                "separated by a comma. For example: Aarav Sharma, Delhi"
            )
            return self._finish(
                session_id,
                question,
                answer,
                "order_lookup",
                awaiting="customer_name_and_city",
            )

        if intent == "human_support":
            answer = DirectResponse.generate("human_support", question, language)
            return self._finish(session_id, question, answer, "human_support")

        if intent in {"policy_question", "product_question", "complaint", "unknown"}:
            if self.rag_service is None:
                return self._finish(
                    session_id,
                    question,
                    DirectResponse.generate("clarification", question, language),
                    "service_unavailable")

            history = self._history(session_id, fallback=chat_history)

            rag_result = self.rag_service.ask(question, language, history)

            if not isinstance(rag_result, dict):
                raise RuntimeError("RAG returned an invalid response.")

            answer = rag_result.get("answer")
            if not answer:
                raise RuntimeError("RAG returned an empty answer.")

            return self._finish(
                session_id,
                question,
                str(answer),
                "rag",
                supported=bool(rag_result.get("supported", False)),
                citations=rag_result.get("citations") or [],
                retrieved_count=int(rag_result.get("retrieved_count", 0)))

        answer = DirectResponse.generate("clarification", question, language)
        return self._finish(session_id, question, answer, "clarification")
"""LangGraph-backed, in-memory conversation history for Vaani.

This service uses LangGraph's InMemorySaver as the source of truth for
conversation messages. It does not create or connect to a database.

Important: InMemorySaver is process-local and non-durable. All histories are
lost when the FastAPI process exits or restarts.
"""
from __future__ import annotations

import logging
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

logger = logging.getLogger(__name__)

def _record_messages(_state: MessagesState) -> dict:
    """A minimal node that lets LangGraph checkpoint the updated message state."""
    return {}


class LangGraphMemoryService:
    """Session-scoped short-term conversation memory using LangGraph."""

    def __init__(self, max_messages: int = 20, max_history_chars: int = 12_000) -> None:
        self.max_messages = max(2, int(max_messages))
        self.max_history_chars = max(1_000, int(max_history_chars))

        self.checkpointer = InMemorySaver()

        builder = StateGraph(MessagesState)
        builder.add_node("record_messages", _record_messages)
        builder.add_edge(START, "record_messages")
        builder.add_edge("record_messages", END)

        self.graph = builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _thread_id(session_id: str) -> str:
        """Validate the API session ID and use it as LangGraph thread_id."""
        return str(UUID(str(session_id)))

    def _config(self, session_id: str) -> dict:
        return {
            "configurable": {
                "thread_id": self._thread_id(session_id),
            }}

    def get_messages(self, session_id: str) -> list:
        """Return the messages currently checkpointed for this conversation."""
        snapshot = self.graph.get_state(self._config(session_id))
        if not snapshot or not snapshot.values:
            return []
        return list(snapshot.values.get("messages", []))

    def get_history(self, session_id: str) -> str:
        """Return bounded history in the format expected by Vaani's RAG services."""
        messages = self.get_messages(session_id)[-self.max_messages:]
        lines: list[str] = []

        for message in messages:
            role = getattr(message, "type", "")
            if role == "human":
                label = "Customer"
            elif role == "ai":
                label = "Vaani"
            else:
                # Ignore tool/system messages; they are not customer dialogue.
                continue

            content = message.content
            if not isinstance(content, str):
                content = str(content)

            content = content.strip()
            if content:
                lines.append(f"{label}: {content}")

        return "\n".join(lines)[-self.max_history_chars:]

    def save_turn(self, session_id: str, question: str, answer: str) -> None:
        """Checkpoint one completed customer/assistant turn atomically."""
        question = (question or "").strip()
        answer = (answer or "").strip()

        if not question or not answer:
            raise ValueError("Both question and answer must be non-empty.")

        self.graph.invoke(
            {
                "messages": [
                    HumanMessage(content=question),
                    AIMessage(content=answer),
                ]
            },
            config=self._config(session_id),
        )

    def clear_session(self, session_id: str) -> None:
        """Delete all checkpoints belonging to one conversation thread."""
        thread_id = self._thread_id(session_id)
        self.checkpointer.delete_thread(thread_id)


# Compatibility alias in case another module imports MemoryService.
MemoryService = LangGraphMemoryService

"""Orchestrates retrieval and grounded LLM generation."""
from __future__ import annotations

from typing import Any

from backend.services.llm_service import LLMService
from backend.services.query_rewriter import QueryRewriter
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    """Text RAG service exposed to API routes.

    Expected retriever interface:
        VaaniRetriever.search(query: str) -> list[langchain_core.documents.Document]

    The retriever is imported lazily so service modules can be tested/imported
    without immediately loading embedding and reranker models.
    """

    def __init__(self, retriever=None, llm_service: LLMService | None = None,
                 query_rewriter: QueryRewriter | None = None) -> None:
        if retriever is None:
            try:
                from backend.rag.retriever import VaaniRetriever
            except Exception as exc:
                raise RuntimeError(
                    "Could not import VaaniRetriever from "
                    "backend.ingestion.retriever. Verify that the project "
                    "root is on PYTHONPATH and backend/ingestion contains "
                    "retriever.py and its required __init__.py files."
                ) from exc
            retriever = VaaniRetriever()

        self.retriever = retriever
        self.llm_service = llm_service or LLMService()

        self.rewrite_history_messages = 8
        self.rewrite_history_chars = 2000
        self.answer_history_messages = 12
        self.answer_history_chars = 4000

        self.query_rewriter = query_rewriter


    def _get_query_rewriter(self) -> QueryRewriter:
        if self.query_rewriter is None:
            self.query_rewriter = QueryRewriter(max_history_messages=self.rewrite_history_messages,
                                                max_history_chars=self.rewrite_history_chars)
            
        return self.query_rewriter
    

    def ask(self, question: str, language: str = "auto", chat_history: str = "") -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        
        original_question = question.strip()
        history = chat_history or ""

        # Bound history independently for final answer generation.
        bounded_answer_history = QueryRewriter.bound_history(history,
                                                             max_messages=self.answer_history_messages,
                                                             max_chars=self.answer_history_chars)

        # Rewrite only when conversation history exists.
        if history.strip():
            retrieval_query = self._get_query_rewriter().rewrite(question=original_question,
                                                                 history=history)
        else:
            retrieval_query = original_question

        # Retrieve using the standalone query.
        documents = self.retriever.search(retrieval_query)

        # Generate the answer using the original question and bounded history.
        return self.llm_service.generate(question=original_question,
                                         documents=documents,
                                         language=language or "auto",
                                         chat_history=bounded_answer_history)
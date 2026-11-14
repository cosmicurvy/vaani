from unittest.mock import MagicMock
from langchain_core.documents import Document
from backend.services.llm_service import GroundedAnswer, LLMService
from backend.services.memory_service import LangGraphMemoryService
from backend.services.query_rewriter import QueryRewriter
from backend.services.rag_service import RAGService

def test_langgraph_memory_service(memory_service: LangGraphMemoryService):
    session_id = "22222222-2222-2222-2222-222222222222"

    assert memory_service.get_history(session_id) == ""

    memory_service.save_turn(session_id, "I bought a TV.", "How can I help with your TV?")
    history = memory_service.get_history(session_id)
    assert "Customer: I bought a TV." in history
    assert "Vaani: How can I help with your TV?" in history

    memory_service.clear_session(session_id)
    assert memory_service.get_history(session_id) == ""


def test_query_rewriter_with_mock_llm():
    mock_llm = MagicMock()
    rewriter = QueryRewriter(llm=mock_llm)

    # Without history, no LLM call is needed
    assert rewriter.rewrite("What is the return policy?", history="") == "What is the return policy?"

    # Bound history helper
    history = "Customer: I bought a VoltKart TV.\nVaani: Sure, how can I help?"
    bounded = QueryRewriter.bound_history(history, max_messages=2, max_chars=500)
    assert "VoltKart TV" in bounded


def test_llm_service_grounding_and_abstention(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    # Mock ChatGroq before initializing LLMService
    mock_structured_llm = MagicMock()
    mock_chat_groq = MagicMock()
    mock_chat_groq.return_value.with_structured_output.return_value = mock_structured_llm
    monkeypatch.setattr("backend.services.llm_service.ChatGroq", mock_chat_groq)

    service = LLMService()
    docs = [
        Document(
            page_content="Refunds are processed within 5 business days.",
            metadata={"source": "refund_policy.pdf", "page": 0},
        )
    ]

    # Case 1: Empty documents -> immediate abstention
    empty_res = service.generate("When will I get my refund?", documents=[])
    assert not empty_res["supported"]
    assert empty_res["citations"] == []

    # Case 2: Supported answer with valid citation S1
    service.prompt = MagicMock()
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = GroundedAnswer(
        answer="Refunds take 5 business days.",
        supported=True,
        citation_ids=["S1"],
    )
    service.prompt.__or__.return_value = chain_mock

    supported_res = service.generate("When will I get my refund?", documents=docs)
    assert supported_res["supported"] is True
    assert "[S1]" in supported_res["answer"]
    assert len(supported_res["citations"]) == 1
    assert supported_res["citations"][0]["source"] == "refund_policy.pdf"

    # Case 3: Hallucinated citation ID (e.g., S99) -> fails closed to abstention
    chain_mock.invoke.return_value = GroundedAnswer(
        answer="Unverified claim.",
        supported=True,
        citation_ids=["S99"],
    )
    abstained_res = service.generate("When will I get my refund?", documents=docs)
    assert abstained_res["supported"] is False
    assert abstained_res["citations"] == []


def test_rag_service_orchestration():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        Document(page_content="Policy text", metadata={"source": "policy.pdf"})
    ]
    mock_llm_service = MagicMock()
    mock_llm_service.generate.return_value = {
        "answer": "Policy answer [S1]",
        "supported": True,
        "citations": [{"source_id": "S1", "source": "policy.pdf"}],
        "retrieved_count": 1,
    }
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "Rewritten standalone query"

    rag = RAGService(
        retriever=mock_retriever,
        llm_service=mock_llm_service,
        query_rewriter=mock_rewriter)

    result = rag.ask(
        question="What about warranty?",
        language="en",
        chat_history="Customer: I bought headphones.\nVaani: How can I help?")

    mock_rewriter.rewrite.assert_called_once()
    mock_retriever.search.assert_called_once_with("Rewritten standalone query")
    assert result["supported"] is True
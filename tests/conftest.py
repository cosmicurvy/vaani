import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.services.conversation_state import ConversationState
from backend.services.customer_repository import CustomerRepository
from backend.services.customer_support import CustomerSupportService
from backend.services.memory_service import LangGraphMemoryService
from backend.services.order_service import OrderService


@pytest.fixture
def sample_orders_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite orders database with one verified test order."""
    db_path = tmp_path / "voltkart_orders.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer_name TEXT,
                city TEXT,
                preferred_language TEXT,
                items TEXT,
                primary_sku TEXT,
                category TEXT,
                return_window_days INTEGER,
                change_of_mind_rule TEXT,
                placed_on TEXT,
                status TEXT,
                payment_method TEXT,
                total_inr REAL,
                shipped_on TEXT,
                estimated_delivery TEXT,
                delivered_on TEXT,
                cancelled_on TEXT,
                courier TEXT,
                tracking_id TEXT,
                last_scan TEXT,
                installation_status TEXT,
                installation_date TEXT,
                return_status TEXT,
                refund_status TEXT,
                refund_method TEXT,
                refund_amount_inr REAL,
                refund_initiated_on TEXT,
                refund_expected_by TEXT
            )
        """)
        conn.execute("""
            INSERT INTO orders VALUES (
                'VK10001', 'Aarav Sharma', 'Delhi', 'en',
                'VoltKart Wireless Earbuds', 'SKU-EB-01', 'Audio',
                7, 'replacement_only', '2026-10-25', 'in_transit',
                'UPI', 1999.0, '2026-10-26', '2026-10-30',
                NULL, NULL, 'BlueDart', 'BD998877', 'New Delhi Hub',
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
            )
        """)
    return db_path


@pytest.fixture
def state_manager(tmp_path: Path) -> ConversationState:
    return ConversationState(db_path=tmp_path / "vaani_support.db")


@pytest.fixture
def customer_repository(sample_orders_db: Path) -> CustomerRepository:
    return CustomerRepository(db_path=sample_orders_db)


@pytest.fixture
def order_service(customer_repository: CustomerRepository) -> OrderService:
    return OrderService(repository=customer_repository)


@pytest.fixture
def memory_service() -> LangGraphMemoryService:
    return LangGraphMemoryService(max_messages=10, max_history_chars=4000)


@pytest.fixture
def mock_rag_service() -> MagicMock:
    rag = MagicMock()
    rag.ask.return_value = {
        "answer": "You can return defective items within 7 days of delivery. [S1]",
        "supported": True,
        "citations": [
            {
                "source_id": "S1",
                "source": "return_policy.pdf",
                "page": 2,
                "faq_id": None,
                "category": "returns",
                "language": "en",
                "source_type": "policy",
            }
        ],
        "retrieved_count": 3,
    }
    return rag


@pytest.fixture
def support_service(
    mock_rag_service: MagicMock,
    memory_service: LangGraphMemoryService,
    state_manager: ConversationState,
    order_service: OrderService,
) -> CustomerSupportService:
    return CustomerSupportService(
        rag_service=mock_rag_service,
        memory_service=memory_service,
        state_manager=state_manager,
        order_service=order_service,
    )
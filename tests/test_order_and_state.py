from backend.services.conversation_state import ConversationState
from backend.services.order_service import OrderService


def test_conversation_state_lifecycle(state_manager: ConversationState):
    session_id = "11111111-1111-1111-1111-111111111111"

    # Default empty state
    initial = state_manager.get_state(session_id)
    assert initial == {"pending_action": None, "context": {}}

    # Set state
    state_manager.set_state(
        session_id,
        "awaiting_order_verification",
        {"order_id": "VK10001"},
    )
    updated = state_manager.get_state(session_id)
    assert updated["pending_action"] == "awaiting_order_verification"
    assert updated["context"] == {"order_id": "VK10001"}

    # Clear state (verifies single-element tuple SQLite binding)
    state_manager.clear_state(session_id)
    cleared = state_manager.get_state(session_id)
    assert cleared == {"pending_action": None, "context": {}}


def test_order_service_validation_and_lookup(order_service: OrderService):
    # Invalid order ID format
    invalid_id = order_service.lookup("INVALID123", "Aarav Sharma", "Delhi")
    assert not invalid_id["success"]

    # Missing customer name or city
    missing_city = order_service.lookup("VK10001", "Aarav Sharma", "")
    assert not missing_city["success"]

    # Wrong customer identity for existing order
    wrong_person = order_service.lookup("VK10001", "Urvashi", "Mumbai")
    assert not wrong_person["success"]
    assert "couldn't verify" in wrong_person["message"]

    # Verified match (case-insensitive & whitespace-tolerant)
    verified = order_service.lookup("vk10001", "  aarav sharma ", " delhi ")
    assert verified["success"]
    assert verified["order"]["order_id"] == "VK10001"
    assert verified["order"]["courier"] == "BlueDart"
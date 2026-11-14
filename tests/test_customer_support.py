import uuid
from backend.services.customer_support import CustomerSupportService

def test_full_order_lookup_workflow(support_service: CustomerSupportService):
    session_id = str(uuid.uuid4())

    # Step 1: User asks where their order is without an order ID
    r1 = support_service.process(session_id, "Where is my order?")
    assert r1["route"] == "order_lookup"
    assert r1["awaiting"] == "order_id"

    # Step 2: User provides order ID
    r2 = support_service.process(session_id, "It is VK10001")
    assert r2["route"] == "order_lookup"
    assert r2["awaiting"] == "customer_name_and_city"

    # Step 3: User provides verified Name, City
    r3 = support_service.process(session_id, "Aarav Sharma, Delhi")
    assert r3["route"] == "order_lookup"
    assert r3["awaiting"] is None
    assert "VK10001" in r3["answer"]
    assert "BlueDart" in r3["answer"]


def test_topic_switch_escapes_pending_order_verification(support_service: CustomerSupportService):
    session_id = str(uuid.uuid4())

    # Enter awaiting_order_verification state
    r1 = support_service.process(session_id, "Track my order VK10001")
    assert r1["awaiting"] == "customer_name_and_city"

    # Wrong verification details keep awaiting customer_name_and_city
    r2 = support_service.process(session_id, "Urvashi, Mumbai")
    assert r2["awaiting"] == "customer_name_and_city"

    # User suddenly switches topic to a policy question or says "Bye"
    r3 = support_service.process(
        session_id,
        "Kya TV installation ke paise dene honge?",
    )
    assert r3["route"] == "rag"
    assert r3["awaiting"] is None

    # Or if in verification and user says "Bye", it exits cleanly
    support_service.process(session_id, "Where is my order VK10001?")
    r4 = support_service.process(session_id, "Bye")
    assert r4["route"] == "direct"
    assert r4["awaiting"] is None
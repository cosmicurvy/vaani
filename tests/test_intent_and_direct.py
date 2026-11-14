from backend.services.direct_response import DirectResponse
from backend.services.intent_classifier import IntentClassifier


def test_extract_order_id():
    classifier = IntentClassifier()
    assert classifier.extract_order_id("Where is my order vk10001?") == "VK10001"
    assert classifier.extract_order_id("No order ID here") is None


def test_intent_classification_routes():
    classifier = IntentClassifier()

    assert classifier.classify("Hello!").intent == "greeting"
    assert classifier.classify("Namaste").intent == "greeting"
    assert classifier.classify("Goodbye").intent == "farewell"

    # Order status queries
    res_order = classifier.classify("Where is my order VK10001?")
    assert res_order.intent == "order_status"
    assert res_order.order_id == "VK10001"

    assert classifier.classify("Mera order kahan hai?").intent == "order_status"
    assert classifier.classify("VK10010").intent == "order_status"

    # Policy question that mentions an Order ID should route to policy_question
    res_policy_with_id = classifier.classify(
        "Can I return the earbuds from VK10010 because I changed my mind?"
    )
    assert res_policy_with_id.intent == "policy_question"
    assert res_policy_with_id.order_id == "VK10010"

    # Hinglish complaint & policy
    assert classifier.classify("Mera product toota hua mila").intent == "complaint"
    assert classifier.classify("Paise kab wapas milega?").intent == "policy_question"
    assert classifier.classify("Agent se baat karni hai").intent == "human_support"

    # Off-topic query
    assert classifier.classify("Who won the cricket match yesterday?").intent == "off_topic"


def test_direct_response_language_detection():
    assert DirectResponse.detect_language("Hello there", "auto") == "en"
    assert (
        DirectResponse.detect_language("Mera order kab milega mujhe", "auto")
        == "hinglish"
    )
    assert DirectResponse.detect_language("नमस्ते", "auto") == "hi"

    greeting_en = DirectResponse.generate("greeting", "Hello", "en")
    assert "Vaani" in greeting_en

    greeting_hinglish = DirectResponse.generate(
        "greeting", "Mera order kya hai", "hinglish"
    )
    assert "Namaste" in greeting_hinglish
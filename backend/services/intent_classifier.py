from __future__ import annotations

import re
from dataclasses import dataclass

@dataclass
class IntentResult:
    intent: str
    confidence: float
    order_id: str | None = None

class IntentClassifier:
    """Multilingual first-pass router for English and Hinglish."""
    ORDER_ID_PATTERN = re.compile(r"\bVK\d{5}\b", re.IGNORECASE)

    ORDER_PATTERNS = [
        r"\bwhere\s+is\s+my\s+order\b", r"\btrack\s+(?:my\s+)?(?:order|package|shipment)\b",
        r"\border\s+status\b", r"\bdelivery\s+status\b",
        r"\bwhen\s+will\s+(?:my\s+)?order\s+arrive\b",
        r"\bhas\s+my\s+order\s+shipped\b",
        r"\bmy\s+package\s+(?:hasn't|has\s+not)\s+arrived\b",
        r"\bi\s+(?:haven't|have\s+not)\s+received\s+my\s+order\b",
        r"\bcheck\s+my\s+shipment\b",
        r"\b(order|parcel|package)\s+(kahan|kahaan)\s+(hai|hain)\b",
        r"\bmera\s+order\b.*\b(status|kahan|kab|track)\b"]
    
    HUMAN_PATTERNS = [
        r"\bhuman\s+agent\b", r"\blive\s+agent\b", r"\breal\s+person\b",
        r"\btalk\s+to\s+someone\b", r"\bcustomer\s+care\s+executive\b",
        r"\b(agent|executive)\s+se\s+baat\b"]
    
    COMPLAINT_PATTERNS = [
        r"\bdamaged\b", r"\bbroken\b", r"\bdefective\b",
        r"\bwrong\s+(?:item|product)\b", r"\bmissing\s+(?:item|product)\b",
        r"\bcomplaint\b", r"\bpoor\s+service\b",
        r"\b(toot|tooti|toota|tuta|tuti)\b",
        r"\bkharab\s+(maal|product|item|nikla|nikli)\b",
        r"\bdamage\s+ho\s+gaya\b", r"\btuta\s+hua\b",
        r"\bgalat\s+(item|product|samaan)\b"]
    
    POLICY_PATTERNS = [
        r"\brefund\b", r"\breturn\s+policy\b", r"\breturn\s+window\b",
        r"\bexchange\s+policy\b", r"\bcancellation\s+policy\b",
        r"\bwarranty\b", r"\bshipping\s+policy\b", r"\bprivacy\s+policy\b",
        r"\bhow\s+(?:can|do)\s+i\s+return\b",
        r"\b(refund|return|exchange|replace|replacement)\b",
        r"\b(wapas|lautana|badalna|replace\s+karna)\b",
        r"\b(paise|paisa|money)\b.*\b(kab|wapas|milega|aayega)\b",
        r"\bkitne\s+din\b.*\b(wapas|return|refund|replace)\b",
        r"\b(wapas|return)\s+kar\s+sakta\b"]
    
    PRODUCT_PATTERNS = [
        r"\bproduct\s+(?:details|information|features)\b",
        r"\bproduct\s+specifications\b", r"\bwhat\s+are\s+the\s+features\b",
        r"\btell\s+me\s+about\s+(?:the\s+)?(?:product|item)\b",
        r"\b(product|item)\s+(ke|ki)\s+(features|details)\b"]
    
    # Positive indicators that an otherwise unfamiliar query belongs to
    # this customer-support assistant's scope.
    SUPPORT_SCOPE_PATTERNS = [
        r"\b(order|package|parcel|shipment|delivery|shipping)\b",
        r"\b(product|item|purchase|bought|buy|store|website)\b",
        r"\b(return|refund|exchange|replace|replacement|warranty)\b",
        r"\b(payment|paid|billing|invoice|transaction|receipt)\b",
        r"\b(cancel|cancellation|dispatch|tracking|track)\b",
        r"\b(customer|support|service|account|login|password)\b",
        r"\b(mera|meri|mere|mujhe|aapka)\b.*\b(order|samaan|product|parcel)\b",
        r"\b(order|product|samaan|parcel)\b.*\b(kahan|kab|kaise|kharab|toota|tooti)\b",
        r"\b(wapas|paise|paisa|khareeda|kharida|delivery|payment)\b",
    ]
    
    GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon",
                 "good evening", "namaste"}
    FAREWELLS = {"bye", "goodbye", "see you", "see you later", "take care",
                 "alvida"}

    @classmethod
    def extract_order_id(cls, message: str) -> str | None:
        match = cls.ORDER_ID_PATTERN.search(message or "")
        return match.group(0).upper() if match else None

    @staticmethod
    def _matches(patterns: list[str], text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def classify(self, message: str) -> IntentResult:
        text = (message or "").strip().lower()

        normalized_text = re.sub(r"[^\w\s]", "", text.lower()).strip()
        normalized_text = " ".join(normalized_text.split())

        if not text:
            return IntentResult("unknown", 0.0)

        order_id = self.extract_order_id(message)

        # 1. Explicit order tracking patterns always win
        if self._matches(self.ORDER_PATTERNS, text):
            return IntentResult("order_status", 0.99, order_id)

        # 2. Check specific support intents BEFORE assuming any message with an order_id is order tracking
        if self._matches(self.HUMAN_PATTERNS, text):
            return IntentResult("human_support", 0.95, order_id)
        if self._matches(self.COMPLAINT_PATTERNS, text):
            return IntentResult("complaint", 0.90, order_id)
        if normalized_text in self.GREETINGS:
            return IntentResult("greeting", 0.99)
        if normalized_text in self.FAREWELLS:
            return IntentResult("farewell", 0.99)
        if self._matches(self.POLICY_PATTERNS, text):
            return IntentResult("policy_question", 0.90, order_id)
        if self._matches(self.PRODUCT_PATTERNS, text):
            return IntentResult("product_question", 0.85, order_id)

        # 3. If the user just typed an order ID (e.g., "VK10010" or "check VK10010"), treat as order_status
        if order_id:
            return IntentResult("order_status", 0.95, order_id)

        # 4. Unfamiliar customer-support phrasing can still reach RAG
        if self._matches(self.SUPPORT_SCOPE_PATTERNS, text):
            return IntentResult("unknown", 0.30)

        # Clearly outside the customer-support domain
        return IntentResult("off_topic", 0.95)

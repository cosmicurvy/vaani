from __future__ import annotations

import random

class DirectResponse:
    RESPONSES = {
        "greeting": {
            "en": ["Hello! Welcome to Vaani. How can I help you today?"],
            "hinglish": ["Namaste! Vaani mein aapka swagat hai. Main aapki kya help kar sakti hoon?"],
        },
        "farewell": {
            "en": ["Thank you for contacting us. Have a great day!"],
            "hinglish": ["Vaani se baat karne ke liye thank you! Aapka din achha rahe."],
        },
        "off_topic": {
            "en": ["I'm Vaani, your customer support assistant. I can help with orders, deliveries, returns, refunds, products and company policies."],
            "hinglish": ["Main Vaani, aapki customer support assistant hoon. Main orders, delivery, returns, refunds, products aur company policies mein help kar sakti hoon."],
        },
        "complaint": {
            "en": ["I'm sorry to hear about this. I'll check the relevant return or replacement guidance for you."],
            "hinglish": ["Aapko hui pareshani ke liye sorry. Main aapke liye return ya replacement ki relevant policy check karti hoon."],
        },
        "human_support": {
            "en": ["I understand you'd like to speak with a human agent. This demo can record the request, but a live agent handoff has not been connected yet."],
            "hinglish": ["Samajh gayi. Aap human agent se baat karna chahte hain. Is demo mein live agent handoff abhi connect nahi hai."],
        },
        "clarification": {
            "en": ["I'm not sure I understood. Could you rephrase your question or tell me whether you need help with an order, delivery, return, refund or product?"],
            "hinglish": ["Mujhe aapka question poori tarah samajh nahi aaya. Kya aap ise thoda alag tareeke se bata sakte hain? Aapko order, delivery, return, refund ya product mein help chahiye?"],
        },
    }

    @staticmethod
    def detect_language(message: str = "", language: str = "auto") -> str:
        requested = (language or "auto").lower()
        if requested in {"hi", "hindi"}:
            return "hi"
        if requested in {"hinglish", "hi-en", "hindi-english"}:
            return "hinglish"
        if requested in {"en", "english"}:
            return "en"
        text = message or ""
        if any("\u0900" <= char <= "\u097f" for char in text):
            return "hi"
        words = set(text.lower().split())
        markers = {
            "mera", "meri", "mere", "mujhe", "kya", "kaise", "kitna",
            "kitne", "kab", "kahan", "kahaan", "hai", "hain", "ho",
            "karna", "sakta", "sakti", "chahiye", "wapas", "khareeda",
            "kharida", "tooti", "toota", "mila", "mili", "nahi", "nahin",
            "liye", "aap", "kripya",
        }
        return "hinglish" if len(words & markers) >= 2 else "en"

    @classmethod
    def generate(cls, intent: str, message: str = "", language: str = "auto") -> str:
        lang = cls.detect_language(message, language)
        options = cls.RESPONSES.get(intent, cls.RESPONSES["clarification"])
        return random.choice(options.get(lang) or options["en"])

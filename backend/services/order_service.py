from __future__ import annotations

import re
from backend.services.customer_repository import CustomerRepository

class OrderService:
    def __init__(self, repository: CustomerRepository | None = None):
        self.repository = repository or CustomerRepository()

    @staticmethod
    def valid_order_id(order_id: str) -> bool:
        return bool(re.fullmatch(r"VK\d{5}", order_id or "", re.IGNORECASE))

    def lookup(self, order_id: str, customer_name: str, city: str) -> dict:
        order_id = (order_id or "").strip().upper()
        customer_name = (customer_name or "").strip()
        city = (city or "").strip()

        if not self.valid_order_id(order_id):
            return {
                "success": False,
                "message": "That order ID format doesn't look right. "
                           "Please check the ID and try again."}

        if not customer_name or not city:
            return {
                "success": False,
                "message": "Please provide both the customer name and city.",
            }

        if len(customer_name) > 120 or len(city) > 100:
            return {
                "success": False,
                "message": "Please check the customer name and city.",
            }

        order = self.repository.get_verified_order(
            order_id=order_id,
            customer_name=customer_name,
            city=city)

        # Use the same message for an unknown ID and failed identity match;
        # don't reveal whether an order ID exists.
        if order is None:
            return {
                "success": False,
                "message": (
                    "I couldn't verify those details. Please check your "
                    "order ID, customer name and city."
                ),
            }

        return {"success": True, "order": order}

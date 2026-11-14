import os
import sqlite3
from pathlib import Path

class CustomerRepository:
    """Read-only repository for the supplied Voltkart orders database.

    The existing database is not modified. Conversation state should remain
    in the separate Vaani support/session database.
    """

    def __init__(self, db_path: str | Path | None = None):
        configured_path = (db_path)
        if configured_path:
            self.db_path = Path(configured_path).expanduser().resolve()
        else:
            self.db_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "voltkart_orders.db"
            )

        if not self.db_path.is_file():
            raise FileNotFoundError(
                f"Orders database not found: {self.db_path}. "
                "Set VAANI_ORDERS_DB or place voltkart_orders.db "
                "inside the project's data/ directory.")

    def connect(self):
        # mode=ro prevents accidental writes to the supplied source database.
        uri = f"{self.db_path.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def get_verified_order(self,order_id: str, customer_name: str, city: str) -> dict | None:
        query = """
            SELECT order_id, customer_name, city, preferred_language, items,
                primary_sku, category, return_window_days, change_of_mind_rule, placed_on,
                status, payment_method, total_inr, shipped_on, estimated_delivery, delivered_on,
                cancelled_on, courier, tracking_id, last_scan, installation_status,
                installation_date, return_status, refund_status, refund_method, refund_amount_inr,
                refund_initiated_on, refund_expected_by
            FROM orders
            WHERE order_id = ?
              AND LOWER(TRIM(customer_name)) = LOWER(TRIM(?))
              AND LOWER(TRIM(city)) = LOWER(TRIM(?))
            LIMIT 1
        """

        with self.connect() as conn:
            row = conn.execute(query,
                (order_id, customer_name, city)).fetchone()

        return dict(row) if row else None

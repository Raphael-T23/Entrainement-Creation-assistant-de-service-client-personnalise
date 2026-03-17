"""Database access layer for the customer service assistant.

Provides safe, parameterized queries to the SQLite database
containing users and orders information.
"""

import sqlite3
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "orders.db"

# Human-readable status translations (French)
STATUS_LABELS = {
    "invoiced": "facturée (en attente d'expédition)",
    "shipped": "expédiée (en cours de livraison)",
    "delivered": "livrée",
}


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a new SQLite connection with row factory enabled."""
    path = db_path or str(DEFAULT_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def get_user_by_email(email: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Retrieve a user record by email address."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT user_id, first_name, last_name, email, phone, address, city, zip_code "
            "FROM users WHERE email = ?",
            (email,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_orders_for_user(user_id: int, db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all orders for a given user_id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT order_id, status, date_purchase, date_shipped, date_delivered "
            "FROM orders WHERE user_id = ? ORDER BY date_purchase DESC",
            (user_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_order_by_id(
    order_id: int, user_id: int, db_path: Optional[str] = None
) -> Optional[dict]:
    """Retrieve a specific order, scoped to the authenticated user."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT order_id, status, date_purchase, date_shipped, date_delivered "
            "FROM orders WHERE order_id = ? AND user_id = ?",
            (order_id, user_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_orders_by_status(
    user_id: int, status: str, db_path: Optional[str] = None
) -> list[dict]:
    """Retrieve orders for a user filtered by status."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT order_id, status, date_purchase, date_shipped, date_delivered "
            "FROM orders WHERE user_id = ? AND status = ?",
            (user_id, status),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def format_status(status: str) -> str:
    """Convert an internal status code to a human-readable French label."""
    return STATUS_LABELS.get(status, status)


def format_order_summary(order: dict) -> str:
    """Format a single order dict into a human-readable summary string."""
    lines = [f"Commande n°{order['order_id']}"]
    lines.append(f"  Statut : {format_status(order['status'])}")
    lines.append(f"  Date d'achat : {order['date_purchase']}")
    if order.get("date_shipped"):
        lines.append(f"  Date d'expédition : {order['date_shipped']}")
    if order.get("date_delivered"):
        lines.append(f"  Date de livraison : {order['date_delivered']}")
    return "\n".join(lines)

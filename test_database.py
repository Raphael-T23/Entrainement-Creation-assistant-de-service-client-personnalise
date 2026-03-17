"""Tests for the database access layer."""

import sqlite3
import tempfile
import os

import pytest

from src.database import (
    format_order_summary,
    format_status,
    get_connection,
    get_order_by_id,
    get_orders_by_status,
    get_orders_for_user,
    get_user_by_email,
)


@pytest.fixture
def test_db():
    """Create a temporary test database with sample data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE users (
            "index" INTEGER,
            user_id INTEGER,
            first_name TEXT,
            last_name TEXT,
            joining_date TIMESTAMP,
            phone INTEGER,
            email TEXT,
            address TEXT,
            city TEXT,
            zip_code INTEGER
        )"""
    )
    conn.execute(
        """CREATE TABLE orders (
            "index" INTEGER,
            order_id INTEGER,
            user_id INTEGER,
            status TEXT,
            date_purchase TIMESTAMP,
            date_shipped TIMESTAMP,
            date_delivered TIMESTAMP
        )"""
    )
    # Insert test users
    conn.execute(
        'INSERT INTO users VALUES (0, 1, "Alice", "Dupont", "2024-01-01", '
        '612345678, "alice@test.com", "1 Rue Test", "Paris", 75001)'
    )
    conn.execute(
        'INSERT INTO users VALUES (1, 2, "Bob", "Martin", "2024-02-01", '
        '698765432, "bob@test.com", "2 Rue Test", "Lyon", 69001)'
    )
    # Insert test orders
    conn.execute(
        'INSERT INTO orders VALUES (0, 101, 1, "delivered", '
        '"2024-05-01 10:00:00", "2024-05-02 10:00:00", "2024-05-10 10:00:00")'
    )
    conn.execute(
        'INSERT INTO orders VALUES (1, 102, 1, "shipped", '
        '"2024-05-15 10:00:00", "2024-05-16 10:00:00", NULL)'
    )
    conn.execute(
        'INSERT INTO orders VALUES (2, 103, 1, "invoiced", '
        '"2024-05-20 10:00:00", NULL, NULL)'
    )
    conn.execute(
        'INSERT INTO orders VALUES (3, 201, 2, "shipped", '
        '"2024-05-10 10:00:00", "2024-05-11 10:00:00", NULL)'
    )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


class TestGetConnection:
    def test_returns_connection(self, test_db):
        conn = get_connection(test_db)
        assert conn is not None
        conn.close()

    def test_row_factory_enabled(self, test_db):
        conn = get_connection(test_db)
        assert conn.row_factory == sqlite3.Row
        conn.close()


class TestGetUserByEmail:
    def test_existing_user(self, test_db):
        user = get_user_by_email("alice@test.com", test_db)
        assert user is not None
        assert user["first_name"] == "Alice"
        assert user["last_name"] == "Dupont"
        assert user["user_id"] == 1

    def test_nonexistent_user(self, test_db):
        user = get_user_by_email("nobody@test.com", test_db)
        assert user is None

    def test_email_case_sensitivity(self, test_db):
        # SQLite is case-insensitive for ASCII by default in LIKE,
        # but = is case-sensitive
        user = get_user_by_email("ALICE@TEST.COM", test_db)
        assert user is None


class TestGetOrdersForUser:
    def test_user_with_orders(self, test_db):
        orders = get_orders_for_user(1, test_db)
        assert len(orders) == 3
        # Should be ordered by date_purchase DESC
        assert orders[0]["order_id"] == 103
        assert orders[1]["order_id"] == 102
        assert orders[2]["order_id"] == 101

    def test_user_with_no_orders(self, test_db):
        orders = get_orders_for_user(999, test_db)
        assert orders == []

    def test_different_user_orders(self, test_db):
        orders = get_orders_for_user(2, test_db)
        assert len(orders) == 1
        assert orders[0]["order_id"] == 201


class TestGetOrderById:
    def test_existing_order(self, test_db):
        order = get_order_by_id(101, 1, test_db)
        assert order is not None
        assert order["order_id"] == 101
        assert order["status"] == "delivered"

    def test_nonexistent_order(self, test_db):
        order = get_order_by_id(999, 1, test_db)
        assert order is None

    def test_order_belongs_to_different_user(self, test_db):
        """User 1 should not be able to access User 2's orders."""
        order = get_order_by_id(201, 1, test_db)
        assert order is None

    def test_cross_user_access_prevented(self, test_db):
        """User 2 should not be able to access User 1's orders."""
        order = get_order_by_id(101, 2, test_db)
        assert order is None


class TestGetOrdersByStatus:
    def test_filter_by_delivered(self, test_db):
        orders = get_orders_by_status(1, "delivered", test_db)
        assert len(orders) == 1
        assert orders[0]["order_id"] == 101

    def test_filter_by_shipped(self, test_db):
        orders = get_orders_by_status(1, "shipped", test_db)
        assert len(orders) == 1
        assert orders[0]["order_id"] == 102

    def test_filter_by_invoiced(self, test_db):
        orders = get_orders_by_status(1, "invoiced", test_db)
        assert len(orders) == 1
        assert orders[0]["order_id"] == 103

    def test_no_matching_status(self, test_db):
        orders = get_orders_by_status(2, "invoiced", test_db)
        assert orders == []


class TestFormatStatus:
    def test_invoiced(self):
        assert "facturée" in format_status("invoiced")

    def test_shipped(self):
        assert "expédiée" in format_status("shipped")

    def test_delivered(self):
        assert "livrée" in format_status("delivered")

    def test_unknown(self):
        assert format_status("unknown") == "unknown"


class TestFormatOrderSummary:
    def test_delivered_order(self):
        order = {
            "order_id": 101,
            "status": "delivered",
            "date_purchase": "2024-05-01 10:00:00",
            "date_shipped": "2024-05-02 10:00:00",
            "date_delivered": "2024-05-10 10:00:00",
        }
        summary = format_order_summary(order)
        assert "101" in summary
        assert "livrée" in summary
        assert "2024-05-01" in summary
        assert "2024-05-02" in summary
        assert "2024-05-10" in summary

    def test_invoiced_order(self):
        order = {
            "order_id": 103,
            "status": "invoiced",
            "date_purchase": "2024-05-20 10:00:00",
            "date_shipped": None,
            "date_delivered": None,
        }
        summary = format_order_summary(order)
        assert "103" in summary
        assert "facturée" in summary
        # Shipped date should not appear for an invoiced order
        assert "Date d'expédition" not in summary

    def test_shipped_order(self):
        order = {
            "order_id": 102,
            "status": "shipped",
            "date_purchase": "2024-05-15 10:00:00",
            "date_shipped": "2024-05-16 10:00:00",
            "date_delivered": None,
        }
        summary = format_order_summary(order)
        assert "102" in summary
        assert "expédiée" in summary
        # Delivery date should not appear for a shipped (not yet delivered) order
        assert "Date de livraison" not in summary

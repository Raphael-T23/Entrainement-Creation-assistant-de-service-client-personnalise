"""Tests for the main bot module."""

import json
import sqlite3
import tempfile
import os
from unittest.mock import MagicMock, patch

import pytest

from src.bot import CustomerServiceBot, OFF_TOPIC_RESPONSE, TOOLS


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
    conn.execute(
        'INSERT INTO users VALUES (0, 1, "Alice", "Dupont", "2024-01-01", '
        '612345678, "alice@test.com", "1 Rue Test", "Paris", 75001)'
    )
    conn.execute(
        'INSERT INTO users VALUES (1, 2, "Bob", "Martin", "2024-02-01", '
        '698765432, "bob@test.com", "2 Rue Test", "Lyon", 69001)'
    )
    conn.execute(
        'INSERT INTO orders VALUES (0, 101, 1, "delivered", '
        '"2024-05-01 10:00:00", "2024-05-02 10:00:00", "2024-05-10 10:00:00")'
    )
    conn.execute(
        'INSERT INTO orders VALUES (1, 102, 1, "shipped", '
        '"2024-05-15 10:00:00", "2024-05-16 10:00:00", NULL)'
    )
    conn.execute(
        'INSERT INTO orders VALUES (2, 201, 2, "invoiced", '
        '"2024-05-20 10:00:00", NULL, NULL)'
    )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


def _make_mock_client():
    """Create a mock OpenAI client."""
    return MagicMock()


def _make_tool_call(name, arguments, call_id="call_123"):
    """Create a mock tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_response(content=None, tool_calls=None):
    """Create a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": (
            [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
            if tool_calls
            else None
        ),
    }
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


class TestBotInitialization:
    def test_valid_user(self, test_db):
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        assert bot.user["first_name"] == "Alice"
        assert bot.user["user_id"] == 1

    def test_invalid_user(self, test_db):
        client = _make_mock_client()
        with pytest.raises(ValueError, match="non trouvé"):
            CustomerServiceBot(
                openai_client=client,
                user_email="nobody@test.com",
                routing_strategy="keywords",
                db_path=test_db,
            )

    def test_system_prompt_contains_user_info(self, test_db):
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        system_msg = bot.conversation[0]["content"]
        assert "Alice" in system_msg
        assert "Dupont" in system_msg
        assert "alice@test.com" in system_msg


class TestBotToolExecution:
    def test_get_all_orders(self, test_db):
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        result = bot._execute_tool_call("get_all_orders", {})
        assert "101" in result
        assert "102" in result
        assert "201" not in result  # Bob's order

    def test_get_order_details_existing(self, test_db):
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        result = bot._execute_tool_call("get_order_details", {"order_id": 101})
        assert "101" in result
        assert "livrée" in result

    def test_get_order_details_not_found(self, test_db):
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        result = bot._execute_tool_call("get_order_details", {"order_id": 999})
        assert "non trouvée" in result

    def test_get_order_details_other_user_order(self, test_db):
        """Alice should not see Bob's order."""
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        result = bot._execute_tool_call("get_order_details", {"order_id": 201})
        assert "non trouvée" in result

    def test_get_orders_by_status(self, test_db):
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        result = bot._execute_tool_call("get_orders_by_status", {"status": "shipped"})
        assert "102" in result

    def test_transfer_to_human(self, test_db):
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        result = bot._execute_tool_call(
            "transfer_to_human", {"reason": "Annulation demandée"}
        )
        assert "TRANSFERT" in result
        assert "agent humain" in result


class TestBotChat:
    def test_off_topic_query_blocked(self, test_db):
        """Off-topic queries should be blocked before reaching the LLM."""
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        response = bot.chat("Quel temps fait-il ?")
        assert response == OFF_TOPIC_RESPONSE
        # The LLM should not have been called
        client.chat.completions.create.assert_not_called()

    def test_customer_service_query_calls_llm(self, test_db):
        """Customer service queries should be forwarded to the LLM."""
        client = _make_mock_client()
        mock_response = _make_response(
            content="Votre commande n°101 a été livrée."
        )
        client.chat.completions.create.return_value = mock_response

        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        response = bot.chat("Où en est ma commande ?")
        assert response == "Votre commande n°101 a été livrée."
        client.chat.completions.create.assert_called_once()

    def test_tool_call_flow(self, test_db):
        """Test that tool calls are properly handled."""
        client = _make_mock_client()

        # First response: LLM wants to call a tool
        tool_call = _make_tool_call("get_all_orders", {})
        first_response = _make_response(tool_calls=[tool_call])

        # Second response: LLM gives final answer
        final_response = _make_response(
            content="Vous avez 2 commandes : n°102 (expédiée) et n°101 (livrée)."
        )

        client.chat.completions.create.side_effect = [
            first_response,
            final_response,
        ]

        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        response = bot.chat("Quelles sont mes commandes ?")
        assert "2 commandes" in response
        assert client.chat.completions.create.call_count == 2

    def test_reset_conversation(self, test_db):
        client = _make_mock_client()
        bot = CustomerServiceBot(
            openai_client=client,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        # Add a fake message
        bot.conversation.append({"role": "user", "content": "test"})
        assert len(bot.conversation) == 2

        bot.reset_conversation()
        assert len(bot.conversation) == 1
        assert bot.conversation[0]["role"] == "system"


class TestToolDefinitions:
    def test_tools_have_required_structure(self):
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_all_expected_tools_present(self):
        tool_names = {t["function"]["name"] for t in TOOLS}
        expected = {
            "get_all_orders",
            "get_order_details",
            "get_orders_by_status",
            "transfer_to_human",
        }
        assert tool_names == expected

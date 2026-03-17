"""Tests for the main bot module."""

import sqlite3
import tempfile
import os
from unittest.mock import MagicMock

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from src.bot import CustomerServiceBot, OFF_TOPIC_RESPONSE


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


def _make_mock_llm():
    """Create a mock LangChain ChatOpenAI that returns an empty response."""
    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    # Default: return an AIMessage with no tool calls
    default_response = AIMessage(content="")
    mock_llm_with_tools.invoke.return_value = default_response
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    return mock_llm, mock_llm_with_tools


@pytest.fixture
def alice_bot(test_db):
    """CustomerServiceBot fixture for Alice with a mocked LLM."""
    mock_llm, _ = _make_mock_llm()
    return CustomerServiceBot(
        llm=mock_llm,
        user_email="alice@test.com",
        routing_strategy="keywords",
        db_path=test_db,
    )


class TestBotInitialization:
    def test_valid_user(self, test_db):
        mock_llm, _ = _make_mock_llm()
        bot = CustomerServiceBot(
            llm=mock_llm,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        assert bot.user["first_name"] == "Alice"
        assert bot.user["user_id"] == 1

    def test_invalid_user(self, test_db):
        mock_llm, _ = _make_mock_llm()
        with pytest.raises(ValueError, match="non trouvé"):
            CustomerServiceBot(
                llm=mock_llm,
                user_email="nobody@test.com",
                routing_strategy="keywords",
                db_path=test_db,
            )

    def test_system_prompt_contains_user_info(self, test_db):
        mock_llm, _ = _make_mock_llm()
        bot = CustomerServiceBot(
            llm=mock_llm,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        assert "Alice" in bot.system_prompt
        assert "Dupont" in bot.system_prompt
        assert "alice@test.com" in bot.system_prompt


class TestBotToolExecution:
    def test_get_all_orders(self, alice_bot):
        result = alice_bot._execute_tool_call("get_all_orders", {})
        assert "101" in result
        assert "102" in result
        assert "201" not in result  # Bob's order

    def test_get_order_details_existing(self, alice_bot):
        result = alice_bot._execute_tool_call("get_order_details", {"order_id": 101})
        assert "101" in result
        assert "livrée" in result

    def test_get_order_details_not_found(self, alice_bot):
        result = alice_bot._execute_tool_call("get_order_details", {"order_id": 999})
        assert "non trouvée" in result

    def test_get_order_details_other_user_order(self, alice_bot):
        """Alice should not see Bob's order."""
        result = alice_bot._execute_tool_call("get_order_details", {"order_id": 201})
        assert "non trouvée" in result

    def test_get_orders_by_status(self, alice_bot):
        result = alice_bot._execute_tool_call(
            "get_orders_by_status", {"status": "shipped"}
        )
        assert "102" in result

    def test_transfer_to_human(self, alice_bot):
        result = alice_bot._execute_tool_call(
            "transfer_to_human", {"reason": "Annulation demandée"}
        )
        assert "TRANSFERT" in result
        assert "agent humain" in result


class TestBotChat:
    def test_off_topic_query_blocked(self, test_db):
        """Off-topic queries should be blocked before reaching the LLM."""
        mock_llm, mock_llm_with_tools = _make_mock_llm()
        bot = CustomerServiceBot(
            llm=mock_llm,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        response = bot.chat("Quel temps fait-il ?")
        assert response == OFF_TOPIC_RESPONSE
        # The LLM should not have been called
        mock_llm_with_tools.invoke.assert_not_called()

    def test_customer_service_query_calls_llm(self, test_db):
        """Customer service queries should be forwarded to the LLM."""
        mock_llm, mock_llm_with_tools = _make_mock_llm()
        mock_llm_with_tools.invoke.return_value = AIMessage(
            content="Votre commande n°101 a été livrée."
        )
        bot = CustomerServiceBot(
            llm=mock_llm,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        response = bot.chat("Où en est ma commande ?")
        assert response == "Votre commande n°101 a été livrée."
        mock_llm_with_tools.invoke.assert_called_once()

    def test_tool_call_flow(self, test_db):
        """Test that tool calls are properly handled in the multi-step loop."""
        mock_llm, mock_llm_with_tools = _make_mock_llm()

        # First response: LLM requests a tool
        first_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_all_orders", "args": {}, "id": "call_123",
                 "type": "tool_call"}
            ],
        )
        # Second response: final answer
        final_response = AIMessage(
            content="Vous avez 2 commandes : n°102 (expédiée) et n°101 (livrée)."
        )
        mock_llm_with_tools.invoke.side_effect = [first_response, final_response]

        bot = CustomerServiceBot(
            llm=mock_llm,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        response = bot.chat("Quelles sont mes commandes ?")
        assert "2 commandes" in response
        assert mock_llm_with_tools.invoke.call_count == 2

    def test_reset_conversation(self, test_db):
        mock_llm, _ = _make_mock_llm()
        bot = CustomerServiceBot(
            llm=mock_llm,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        # Simulate a previous turn in the history
        bot.chat_history.append(HumanMessage(content="test"))
        assert len(bot.chat_history) == 1

        bot.reset_conversation()
        assert len(bot.chat_history) == 0

    def test_chat_history_updated_after_response(self, test_db):
        """Each chat turn should append one HumanMessage and one AIMessage."""
        mock_llm, mock_llm_with_tools = _make_mock_llm()
        mock_llm_with_tools.invoke.return_value = AIMessage(
            content="Votre commande est en cours de livraison."
        )
        bot = CustomerServiceBot(
            llm=mock_llm,
            user_email="alice@test.com",
            routing_strategy="keywords",
            db_path=test_db,
        )
        assert len(bot.chat_history) == 0
        bot.chat("Statut de ma commande ?")
        assert len(bot.chat_history) == 2
        assert isinstance(bot.chat_history[0], HumanMessage)
        assert isinstance(bot.chat_history[1], AIMessage)


class TestToolDefinitions:
    def test_tools_have_required_attributes(self, alice_bot):
        """Each LangChain tool must expose name and description."""
        for t in alice_bot.tools:
            assert hasattr(t, "name")
            assert hasattr(t, "description")
            assert t.name
            assert t.description

    def test_all_expected_tools_present(self, alice_bot):
        tool_names = {t.name for t in alice_bot.tools}
        expected = {
            "get_all_orders",
            "get_order_details",
            "get_orders_by_status",
            "transfer_to_human",
        }
        assert tool_names == expected


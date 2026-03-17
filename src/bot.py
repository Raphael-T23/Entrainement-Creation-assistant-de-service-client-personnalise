"""Main bot module for the customer service assistant.

Orchestrates the LLM, database access, semantic routing, and conversation
management to provide a complete customer service experience.

This module uses the **LangChain** framework:

* ``ChatOpenAI`` (``langchain-openai``) replaces the raw ``openai.OpenAI``
  client.
* Tool functions are declared with LangChain's ``@tool`` decorator and bound
  to the model via ``ChatOpenAI.bind_tools()``.
* The conversation history is maintained as a list of typed LangChain
  ``BaseMessage`` objects (``SystemMessage``, ``HumanMessage``, ``AIMessage``,
  ``ToolMessage``), replacing the previous plain-dict format.
* The tool-calling loop is driven by the ``AIMessage.tool_calls`` attribute
  returned by LangChain, which avoids manual JSON parsing.
"""

from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .database import (
    format_order_summary,
    get_order_by_id,
    get_orders_by_status,
    get_orders_for_user,
    get_user_by_email,
)
from .prompts import SYSTEM_PROMPT_TEMPLATE
from .routing import SemanticRouter

OFF_TOPIC_RESPONSE = (
    "Je suis désolé, mais je suis un assistant dédié au service client. "
    "Je ne peux répondre qu'aux questions concernant vos commandes "
    "(statut, livraison, paiement, etc.). "
    "Comment puis-je vous aider avec vos commandes ?"
)

TRANSFER_RESPONSE = (
    "Je comprends votre demande. Un agent humain va prendre le relais "
    "dans la conversation pour vous aider. Veuillez patienter un instant, "
    "quelqu'un sera avec vous sous peu. Merci de votre patience !"
)


class CustomerServiceBot:
    """Customer service chatbot powered by LangChain and OpenAI.

    The bot uses LangChain's ``ChatOpenAI`` as the language model backend and
    exposes four ``@tool``-decorated functions to the model via
    ``bind_tools()``.  Multi-turn tool calling is handled by an explicit loop
    that inspects ``AIMessage.tool_calls``, executes each requested tool, and
    feeds the results back as ``ToolMessage`` objects.

    Attributes:
        llm: ``ChatOpenAI`` instance.
        llm_with_tools: LLM with the user-scoped tools bound via
            ``bind_tools()``.
        model: The OpenAI model name.
        user: Dict with authenticated user information.
        system_prompt: Formatted system-prompt string for the current user.
        tools: List of LangChain ``@tool`` functions scoped to the user.
        router: ``SemanticRouter`` instance for query classification.
        chat_history: Running list of ``BaseMessage`` objects (excludes the
            system message, which is prepended on every call).
        db_path: Path to the SQLite database.
    """

    def __init__(
        self,
        user_email: str,
        model: str = "gpt-4o-mini",
        routing_strategy: str = "embeddings",
        db_path: Optional[str] = None,
        llm: Optional[ChatOpenAI] = None,
        temperature: float = 0.3,
        # openai_client is kept for backward compatibility and is ignored;
        # use the `llm` parameter to inject a custom / mock LLM instead.
        openai_client=None,
    ):
        self.model = model
        self.db_path = db_path

        # Retrieve and validate the authenticated user
        self.user = get_user_by_email(user_email, db_path)
        if not self.user:
            raise ValueError(
                f"Utilisateur avec l'email '{user_email}' non trouvé dans la base."
            )

        # Build the system prompt with user info
        self.system_prompt: str = SYSTEM_PROMPT_TEMPLATE.format(
            first_name=self.user["first_name"],
            last_name=self.user["last_name"],
            email=self.user["email"],
        )

        # Initialise the LangChain LLM (inject a mock in tests via `llm=`)
        self.llm: ChatOpenAI = llm if llm is not None else ChatOpenAI(
            model=model, temperature=temperature
        )

        # Create user-scoped @tool functions and bind them to the LLM
        self.tools = self._create_user_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Conversation history (system message prepended on each call)
        self.chat_history: list[BaseMessage] = []

        # Semantic router
        self.router = SemanticRouter(strategy=routing_strategy, model=model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_tool_call(self, name: str, arguments: dict) -> str:
        """Execute a named tool and return its result as a string.

        Keeping this as a separate method makes unit-testing the tool logic
        straightforward without going through the LLM.
        """
        user_id = self.user["user_id"]

        if name == "get_all_orders":
            orders = get_orders_for_user(user_id, self.db_path)
            if not orders:
                return "Aucune commande trouvée pour cet utilisateur."
            return "\n\n".join(format_order_summary(o) for o in orders)

        elif name == "get_order_details":
            order_id = arguments.get("order_id")
            order = get_order_by_id(order_id, user_id, self.db_path)
            if not order:
                return (
                    f"Commande n°{order_id} non trouvée pour cet utilisateur. "
                    "Veuillez vérifier le numéro de commande."
                )
            return format_order_summary(order)

        elif name == "get_orders_by_status":
            status = arguments.get("status")
            orders = get_orders_by_status(user_id, status, self.db_path)
            if not orders:
                return f"Aucune commande avec le statut '{status}' trouvée."
            return "\n\n".join(format_order_summary(o) for o in orders)

        elif name == "transfer_to_human":
            reason = arguments.get("reason", "Demande de l'utilisateur")
            return f"[TRANSFERT] Raison : {reason}. {TRANSFER_RESPONSE}"

        return "Outil inconnu."

    def _create_user_tools(self) -> list:
        """Return a list of LangChain ``@tool`` functions scoped to this user.

        Each function delegates to :meth:`_execute_tool_call` so that the
        tool logic can be tested directly on the bot instance.
        """
        bot = self

        @tool
        def get_all_orders() -> str:
            """Récupère toutes les commandes de l'utilisateur authentifié.

            Utilise cette fonction quand l'utilisateur demande la liste
            de ses commandes ou des informations générales sur ses commandes.
            """
            return bot._execute_tool_call("get_all_orders", {})

        @tool
        def get_order_details(order_id: int) -> str:
            """Récupère les détails d'une commande spécifique par son numéro.

            Utilise cette fonction quand l'utilisateur demande des informations
            sur une commande précise en mentionnant un numéro de commande.
            """
            return bot._execute_tool_call("get_order_details", {"order_id": order_id})

        @tool
        def get_orders_by_status(status: str) -> str:
            """Récupère les commandes de l'utilisateur filtrées par statut.

            Les statuts possibles sont : 'invoiced' (facturée),
            'shipped' (expédiée), 'delivered' (livrée).
            """
            return bot._execute_tool_call("get_orders_by_status", {"status": status})

        @tool
        def transfer_to_human(reason: str) -> str:
            """Transfère la conversation à un agent humain.

            Utilise cette fonction quand l'utilisateur souhaite modifier
            ou annuler une commande, ou quand il a besoin d'une aide
            qui dépasse les capacités du bot.
            """
            return bot._execute_tool_call("transfer_to_human", {"reason": reason})

        return [get_all_orders, get_order_details, get_orders_by_status, transfer_to_human]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process a user message and return the bot's response.

        Steps:

        1. **Semantic routing** – off-topic queries are rejected before
           reaching the LLM.
        2. The user message and the running ``chat_history`` are assembled
           into a message list (prefixed by the ``SystemMessage``).
        3. The LLM (with tools bound) is invoked via
           ``self.llm_with_tools.invoke()``.
        4. If the response contains tool calls, each tool is executed and its
           result is fed back as a ``ToolMessage``; the LLM is called again
           until it returns a plain text answer.
        5. The human / assistant pair is appended to ``chat_history`` and the
           final text is returned.
        """
        # Step 1: Semantic routing
        if not self.router.is_customer_service(user_message):
            return OFF_TOPIC_RESPONSE

        # Step 2: Build the message list for this turn
        messages: list[BaseMessage] = (
            [SystemMessage(content=self.system_prompt)]
            + self.chat_history
            + [HumanMessage(content=user_message)]
        )

        # Step 3: First LLM call
        response: AIMessage = self.llm_with_tools.invoke(messages)

        # Step 4: Tool-calling loop
        while response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                result = self._execute_tool_call(tc["name"], tc["args"])
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )
            response = self.llm_with_tools.invoke(messages)

        # Step 5: Persist the turn in chat_history and return
        bot_response: str = response.content or ""
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=bot_response))
        return bot_response

    def reset_conversation(self) -> None:
        """Clear the conversation history (the system prompt is unaffected)."""
        self.chat_history = []


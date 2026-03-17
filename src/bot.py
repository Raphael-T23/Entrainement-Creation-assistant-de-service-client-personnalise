"""Main bot module for the customer service assistant.

Orchestrates the LLM, database access, semantic routing, and
conversation management to provide a complete customer service experience.
"""

import json
from typing import Optional

from openai import OpenAI

from .database import (
    format_order_summary,
    get_order_by_id,
    get_orders_by_status,
    get_orders_for_user,
    get_user_by_email,
)
from .prompts import SYSTEM_PROMPT_TEMPLATE
from .routing import SemanticRouter

# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_all_orders",
            "description": (
                "Récupère toutes les commandes de l'utilisateur authentifié. "
                "Utilise cette fonction quand l'utilisateur demande la liste "
                "de ses commandes ou des informations générales sur ses commandes."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": (
                "Récupère les détails d'une commande spécifique par son numéro. "
                "Utilise cette fonction quand l'utilisateur demande des informations "
                "sur une commande précise en mentionnant un numéro de commande."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "Le numéro de la commande.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders_by_status",
            "description": (
                "Récupère les commandes de l'utilisateur filtrées par statut. "
                "Les statuts possibles sont : 'invoiced' (facturée), "
                "'shipped' (expédiée), 'delivered' (livrée)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["invoiced", "shipped", "delivered"],
                        "description": "Le statut de la commande à filtrer.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": (
                "Transfère la conversation à un agent humain. "
                "Utilise cette fonction quand l'utilisateur souhaite modifier "
                "ou annuler une commande, ou quand il a besoin d'une aide "
                "qui dépasse les capacités du bot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "La raison du transfert.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]

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
    """Customer service chatbot powered by OpenAI with tool calling.

    Attributes:
        client: OpenAI client instance.
        model: The OpenAI model to use.
        user: Dict with authenticated user information.
        router: SemanticRouter instance for query classification.
        conversation: List of conversation messages.
        db_path: Path to the SQLite database.
    """

    def __init__(
        self,
        openai_client: OpenAI,
        user_email: str,
        model: str = "gpt-4o-mini",
        routing_strategy: str = "embeddings",
        db_path: Optional[str] = None,
    ):
        self.client = openai_client
        self.model = model
        self.db_path = db_path

        # Retrieve and validate the authenticated user
        self.user = get_user_by_email(user_email, db_path)
        if not self.user:
            raise ValueError(
                f"Utilisateur avec l'email '{user_email}' non trouvé dans la base."
            )

        # Build the system prompt with user info
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            first_name=self.user["first_name"],
            last_name=self.user["last_name"],
            email=self.user["email"],
        )

        self.conversation: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

        # Initialize the semantic router
        self.router = SemanticRouter(
            client=openai_client,
            strategy=routing_strategy,
            model=model,
        )

    def _execute_tool_call(self, name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as a string."""
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

    def chat(self, user_message: str) -> str:
        """Process a user message and return the bot's response.

        This method:
        1. Applies semantic routing to filter off-topic queries
        2. Sends the message to the LLM with tool definitions
        3. Handles any tool calls (database queries)
        4. Returns the final formatted response
        """
        # Step 1: Semantic routing
        if not self.router.is_customer_service(user_message):
            return OFF_TOPIC_RESPONSE

        # Step 2: Add user message to conversation
        self.conversation.append({"role": "user", "content": user_message})

        # Step 3: Call the LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            tools=TOOLS,
            temperature=0.3,
        )

        assistant_message = response.choices[0].message

        # Step 4: Handle tool calls if any
        while assistant_message.tool_calls:
            # Add the assistant's message with tool calls
            self.conversation.append(assistant_message.model_dump())

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                result = self._execute_tool_call(fn_name, fn_args)

                self.conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            # Get the next response from the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                tools=TOOLS,
                temperature=0.3,
            )
            assistant_message = response.choices[0].message

        # Step 5: Add and return the final response
        bot_response = assistant_message.content or ""
        self.conversation.append({"role": "assistant", "content": bot_response})
        return bot_response

    def reset_conversation(self) -> None:
        """Reset the conversation history, keeping only the system prompt."""
        self.conversation = [self.conversation[0]]

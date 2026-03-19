"""Module principal du bot pour l'assistant de service client.

Orchestre le LLM, l'accès à la base de données, le routage sémantique et la
gestion des conversations pour offrir une expérience complète de service client.

Ce module utilise le framework **LangChain** :

* ``ChatOpenAI`` (``langchain-openai``) remplace le client brut ``openai.OpenAI``.
* Les fonctions d'outil sont déclarées avec le décorateur ``@tool`` de LangChain
  et liées au modèle via ``ChatOpenAI.bind_tools()``.
* L'historique de conversation est maintenu sous forme de liste typée d'objets
  LangChain ``BaseMessage`` (``SystemMessage``, ``HumanMessage``, ``AIMessage``,
  ``ToolMessage``), remplaçant l'ancien format de dictionnaires.
* La boucle d'appel aux outils est pilotée par l'attribut ``AIMessage.tool_calls``
  retourné par LangChain, ce qui évite l'analyse JSON manuelle.
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

# --- Import du faux LLM pour le mode test (activé via MOCK_LLM=true dans .env) ---
from .fake_llm import FakeChatOpenAI

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
    """Chatbot de service client propulsé par LangChain et OpenAI.

    Le bot utilise ``ChatOpenAI`` de LangChain comme backend de modèle de langage
    et expose quatre fonctions décorées avec ``@tool`` au modèle via
    ``bind_tools()``. Les appels multi-tours aux outils sont gérés par une boucle
    explicite qui inspecte ``AIMessage.tool_calls``, exécute chaque outil demandé,
    et renvoie les résultats sous forme d'objets ``ToolMessage``.

    Attributes:
        llm: Instance de ``ChatOpenAI``.
        llm_with_tools: LLM avec les outils limités à l'utilisateur liés via
            ``bind_tools()``.
        model: Nom du modèle OpenAI.
        user: Dict contenant les informations de l'utilisateur authentifié.
        system_prompt: Chaîne du prompt système formatée pour l'utilisateur courant.
        tools: Liste des fonctions ``@tool`` de LangChain limitées à l'utilisateur.
        router: Instance de ``SemanticRouter`` pour la classification des requêtes.
        chat_history: Liste courante d'objets ``BaseMessage`` (exclut le message
            système, qui est préfixé à chaque appel).
        db_path: Chemin vers la base de données SQLite.
    """

    def __init__(
        self,
        user_email: str,
        model: str = "gpt-4o-mini",
        routing_strategy: str = "embeddings",
        db_path: Optional[str] = None,
        llm: Optional[ChatOpenAI] = None,
        temperature: float = 0.3,
        # --- Option mode mock : si True, utilise FakeChatOpenAI au lieu d'OpenAI ---
        # Activé par la variable d'environnement MOCK_LLM=true dans .env.
        # Permet de tester le bot sans clé API et sans frais.
        mock_llm: bool = False,
        # openai_client est conservé pour la compatibilité ascendante et est ignoré ;
        # utiliser le paramètre 'llm' pour injecter un LLM personnalisé ou mock.
        openai_client=None,
    ):
        self.model = model
        self.db_path = db_path
        self._mock_mode = mock_llm

        # Récupérer et valider l'utilisateur authentifié
        self.user = get_user_by_email(user_email, db_path)
        if not self.user:
            raise ValueError(f"Utilisateur avec l'email '{user_email}' non trouvé dans la base.")

        # Construire le prompt système avec les informations de l'utilisateur
        self.system_prompt: str = SYSTEM_PROMPT_TEMPLATE.format(
            first_name=self.user["first_name"],
            last_name=self.user["last_name"],
            email=self.user["email"],
        )

        # --- Initialisation du LLM ---
        # Priorité : 1) llm injecté (tests unitaires)  2) mock_llm  3) vrai ChatOpenAI
        if llm is not None:
            # LLM injecté directement (ex: tests unitaires avec MagicMock)
            self.llm = llm
        elif mock_llm:
            # Mode mock explicite (MOCK_LLM=true dans .env) : on utilise le faux
            # LLM et on force le routage par mots-clés car les stratégies
            # "embeddings" et "llm" nécessitent elles aussi une clé API valide.
            self.llm = FakeChatOpenAI(model=model, temperature=temperature)
            routing_strategy = "keywords"
        else:
            # Mode normal : vrai LLM OpenAI (nécessite OPENAI_API_KEY)
            self.llm = ChatOpenAI(model=model, temperature=temperature)

        # Créer les fonctions @tool limitées à l'utilisateur et les lier au LLM
        self.tools = self._create_user_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Historique de conversation (le message système est préfixé à chaque appel)
        self.chat_history: list[BaseMessage] = []

        # Routeur sémantique
        self.router = SemanticRouter(strategy=routing_strategy, model=model)

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _execute_tool_call(self, name: str, arguments: dict) -> str:
        """Exécute un outil nommé et retourne son résultat sous forme de chaîne.

        Garder cette logique dans une méthode séparée facilite les tests unitaires
        de la logique des outils sans passer par le LLM.
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
        """Retourne une liste de fonctions ``@tool`` LangChain limitées à cet utilisateur.

        Chaque fonction délègue à :meth:`_execute_tool_call` afin que la logique
        des outils puisse être testée directement sur l'instance du bot.
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
    # API publique
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Traite un message utilisateur et retourne la réponse du bot.

        Étapes :

        1. **Routage sémantique** – les requêtes hors sujet sont rejetées avant
           d'atteindre le LLM.
        2. Le message utilisateur et l'historique ``chat_history`` courant sont
           assemblés en une liste de messages (préfixée par le ``SystemMessage``).
        3. Le LLM (avec les outils liés) est invoqué via
           ``self.llm_with_tools.invoke()``.
        4. Si la réponse contient des appels d'outils, chaque outil est exécuté et
           son résultat est renvoyé sous forme de ``ToolMessage`` ; le LLM est
           rappelé jusqu'à ce qu'il retourne une réponse textuelle finale.
        5. La paire humain/assistant est ajoutée à ``chat_history`` et le texte
           final est retourné.
        """
        # Étape 1 : Routage sémantique
        if not self.router.is_customer_service(user_message):
            return OFF_TOPIC_RESPONSE

        # Étape 2 : Construire la liste de messages pour ce tour
        messages: list[BaseMessage] = (
            [SystemMessage(content=self.system_prompt)]
            + self.chat_history
            + [HumanMessage(content=user_message)]
        )

        # Étape 3 : Premier appel au LLM
        response: AIMessage = self.llm_with_tools.invoke(messages)

        # Étape 4 : Boucle d'appel aux outils
        while response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                result = self._execute_tool_call(tc["name"], tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            response = self.llm_with_tools.invoke(messages)

        # Étape 5 : Enregistrer le tour dans chat_history et retourner
        bot_response: str = response.content or ""
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=bot_response))
        return bot_response

    def reset_conversation(self) -> None:
        """Efface l'historique de conversation (le prompt système n'est pas affecté)."""
        self.chat_history = []

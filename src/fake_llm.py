"""Faux LLM (mock) pour tester le bot sans clé API OpenAI.

Pourquoi ce module existe :
    En activant la variable d'environnement MOCK_LLM=true dans le fichier .env,
    le bot utilise ce FakeChatOpenAI au lieu du vrai ChatOpenAI d'OpenAI.
    Cela permet de tester l'ensemble du flux conversationnel (routage, outils,
    historique) en local sans consommer de crédits API.

    Ce mode est strictement réservé au développement et aux tests manuels.
    En production, MOCK_LLM doit rester absent ou à false.

Comment ça marche :
    FakeChatOpenAI imite l'interface de ChatOpenAI de LangChain (bind_tools,
    invoke). Il analyse le dernier message utilisateur pour décider quel outil
    appeler, puis résume les résultats des outils dans une réponse textuelle.
    Le comportement n'est pas aussi riche qu'un vrai LLM, mais il est
    suffisant pour valider la mécanique du bot.
"""

import re
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

# ---------------------------------------------------------------------------
# Correspondance statut français → anglais pour l'appel d'outil
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[str, str] = {
    "livré": "delivered",
    "livrée": "delivered",
    "livrées": "delivered",
    "livrés": "delivered",
    "expédié": "shipped",
    "expédiée": "shipped",
    "expédiées": "shipped",
    "expédiés": "shipped",
    "facturé": "invoiced",
    "facturée": "invoiced",
    "facturées": "invoiced",
    "facturés": "invoiced",
}


# ---------------------------------------------------------------------------
# FakeChatOpenAI – le faux modèle de langage
# ---------------------------------------------------------------------------


class FakeChatOpenAI:
    """Simule ``ChatOpenAI`` de LangChain pour les tests sans clé API.

    Implémente uniquement les méthodes utilisées par le bot :
      - ``bind_tools(tools)`` → retourne une copie avec les outils mémorisés
      - ``invoke(messages)``  → retourne un ``AIMessage`` (avec ou sans
        ``tool_calls``) en analysant le dernier message humain.

    La logique de décision est basée sur des heuristiques simples (regex et
    mots-clés) et ne prétend pas reproduire la qualité d'un vrai LLM.
    """

    def __init__(self, **kwargs: Any) -> None:
        # On accepte les mêmes kwargs que ChatOpenAI pour la compatibilité
        self._tools: list = []
        self._kwargs = kwargs

    def bind_tools(self, tools: list) -> "FakeChatOpenAI":
        """Retourne une copie de ce faux LLM avec les outils enregistrés."""
        clone = FakeChatOpenAI(**self._kwargs)
        clone._tools = list(tools)
        return clone

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Analyse les messages et retourne une réponse simulée.

        Deux cas :
        1. Si des ``ToolMessage`` sont présents dans la conversation → on
           résume leurs contenus dans une réponse textuelle finale.
        2. Sinon → on analyse le dernier ``HumanMessage`` pour décider
           quel outil appeler (ou répondre directement).
        """
        # --- Cas 1 : des résultats d'outils sont disponibles → réponse finale ---
        tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
        if tool_results:
            summary = "\n\n".join(tool_results)
            return AIMessage(content="Voici les informations que j'ai trouvées :\n\n" + summary)

        # --- Cas 2 : pas encore d'appels d'outils → analyser le message ---
        last_human = self._get_last_human_text(messages)
        if not last_human:
            return AIMessage(content="Comment puis-je vous aider ?")

        text = last_human.lower()

        # Détection d'un numéro de commande précis → get_order_details
        match = re.search(r"commande\s*(?:n[°o]?\s*)?#?(\d+)", text)
        if match:
            order_id = int(match.group(1))
            return self._tool_call_message("get_order_details", {"order_id": order_id})

        # Demande de modification / annulation → transfert agent humain
        if any(mot in text for mot in ["annuler", "modifier", "annulation", "modification"]):
            return self._tool_call_message(
                "transfer_to_human",
                {"reason": "Demande de modification ou annulation"},
            )

        # Filtrage par statut → get_orders_by_status
        for fr_status, en_status in _STATUS_MAP.items():
            if fr_status in text:
                return self._tool_call_message("get_orders_by_status", {"status": en_status})

        # Requête générale sur les commandes → get_all_orders
        if any(mot in text for mot in ["commande", "commandes", "colis", "livraison", "suivi"]):
            return self._tool_call_message("get_all_orders", {})

        # Salutation
        if any(mot in text for mot in ["bonjour", "salut", "bonsoir", "hello", "coucou"]):
            return AIMessage(
                content=(
                    "Bonjour ! Je suis en mode test (Mock LLM). "
                    "Comment puis-je vous aider avec vos commandes "
                    "aujourd'hui ?"
                )
            )

        # Remerciement
        if any(mot in text for mot in ["merci", "thanks", "au revoir"]):
            return AIMessage(
                content=(
                    "Je vous en prie ! N'hésitez pas si vous avez "
                    "d'autres questions. Bonne journée !"
                )
            )

        # Réponse par défaut
        return AIMessage(
            content=(
                "Je suis en mode test (Mock LLM). "
                "Je peux répondre à des questions simples sur vos commandes. "
                "Essayez par exemple : « Où en est ma commande 101 ? » "
                "ou « Liste mes commandes »."
            )
        )

    # ------------------------------------------------------------------
    # Méthodes utilitaires internes
    # ------------------------------------------------------------------

    @staticmethod
    def _get_last_human_text(messages: list[BaseMessage]) -> str:
        """Extrait le texte du dernier HumanMessage de la conversation."""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return m.content
        return ""

    @staticmethod
    def _tool_call_message(name: str, args: dict) -> AIMessage:
        """Construit un AIMessage contenant un appel d'outil unique."""
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"fake_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call",
                }
            ],
        )

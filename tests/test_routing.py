"""Tests pour le module de routage sémantique."""

from unittest.mock import patch

import numpy as np

from src.routing import (
    SemanticRouter,
    _cosine_similarity,
    route_with_embeddings,
    route_with_keywords,
    route_with_llm,
    CUSTOMER_SERVICE_REFERENCES,
    OFF_TOPIC_REFERENCES,
)


class TestKeywordRouting:
    """Teste le routeur de repli basé sur les mots-clés."""

    def test_order_status_query(self):
        assert route_with_keywords("Où en est ma commande ?") == "service_client"

    def test_delivery_query(self):
        assert route_with_keywords("Quand sera livrée ma commande ?") == "service_client"

    def test_payment_query(self):
        assert route_with_keywords("Quel est le statut du paiement ?") == "service_client"

    def test_cancel_query(self):
        assert route_with_keywords("Je veux annuler ma commande") == "service_client"

    def test_modify_query(self):
        assert route_with_keywords("Puis-je modifier ma commande ?") == "service_client"

    def test_greeting(self):
        assert route_with_keywords("Bonjour") == "service_client"

    def test_thanks(self):
        assert route_with_keywords("Merci beaucoup") == "service_client"

    def test_help_request(self):
        assert route_with_keywords("J'ai besoin d'aide") == "service_client"

    def test_off_topic_weather(self):
        assert route_with_keywords("Quel temps fait-il ?") == "hors_sujet"

    def test_off_topic_general(self):
        assert route_with_keywords("Raconte-moi une blague") == "hors_sujet"

    def test_off_topic_recipe(self):
        assert route_with_keywords("Donne-moi une recette de cuisine") == "hors_sujet"

    def test_off_topic_politics(self):
        assert route_with_keywords("Qui est le président ?") == "hors_sujet"

    def test_tracking_keyword(self):
        assert route_with_keywords("Avez-vous un numéro de tracking ?") == "service_client"

    def test_problem_keyword(self):
        assert route_with_keywords("J'ai un problème") == "service_client"


class TestSemanticRouterKeywords:
    """Teste la classe SemanticRouter avec la stratégie mots-clés."""

    def test_customer_service_query(self):
        router = SemanticRouter(client=None, strategy="keywords")
        assert router.is_customer_service("Où en est ma commande ?") is True

    def test_off_topic_query(self):
        router = SemanticRouter(client=None, strategy="keywords")
        assert router.is_customer_service("Quel temps fait-il ?") is False

    def test_classify_returns_correct_label(self):
        router = SemanticRouter(client=None, strategy="keywords")
        assert router.classify("Ma livraison est en retard") == "service_client"
        assert router.classify("Quelle heure est-il ?") == "hors_sujet"

    def test_fallback_to_keywords_without_client(self):
        """Quand la stratégie est 'embeddings' mais sans client, bascule vers les mots-clés."""
        router = SemanticRouter(client=None, strategy="embeddings")
        assert router.classify("Statut de ma commande") == "service_client"


# -----------------------------------------------------------------------
# Tests de _cosine_similarity (calcul pur numpy, sans appel API)
# -----------------------------------------------------------------------


class TestCosineSimilarity:
    """Teste la fonction utilitaire de similarité cosinus."""

    def test_identical_vectors(self):
        """Deux vecteurs identiques doivent avoir une similarité de 1."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[1.0, 0.0, 0.0]])
        result = _cosine_similarity(a, b)
        assert result.shape == (1,)
        assert np.isclose(result[0], 1.0)

    def test_orthogonal_vectors(self):
        """Deux vecteurs orthogonaux doivent avoir une similarité de 0."""
        a = np.array([1.0, 0.0])
        b = np.array([[0.0, 1.0]])
        result = _cosine_similarity(a, b)
        assert np.isclose(result[0], 0.0)

    def test_multiple_references(self):
        """Vérifier la forme du résultat avec plusieurs vecteurs de référence."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        result = _cosine_similarity(a, b)
        assert result.shape == (3,)
        # Le premier est identique → max similarité
        assert np.argmax(result) == 0


# -----------------------------------------------------------------------
# Tests de route_with_embeddings (mock de _compute_embeddings)
# -----------------------------------------------------------------------


def _make_fake_embeddings(cs_score: float, ot_score: float):
    """Construit des embeddings synthétiques qui donneront les scores voulus.

    Retourne un array où :
    - index 0 = query (vecteur unitaire sur l'axe 0)
    - index 1..N = refs service client (vecteur orienté vers query selon cs_score)
    - index N+1.. = refs hors sujet (vecteur orienté vers query selon ot_score)
    """
    query = np.array([1.0, 0.0, 0.0])

    def _vec_with_similarity(target_sim: float) -> np.ndarray:
        """Construit un vecteur unitaire ayant ~target_sim de cosinus avec query."""
        v = np.array([target_sim, np.sqrt(1 - target_sim**2), 0.0])
        return v / np.linalg.norm(v)

    cs_vec = _vec_with_similarity(cs_score)
    ot_vec = _vec_with_similarity(ot_score)

    n_cs = len(CUSTOMER_SERVICE_REFERENCES)
    n_ot = len(OFF_TOPIC_REFERENCES)

    rows = [query]
    rows.extend([cs_vec] * n_cs)
    rows.extend([ot_vec] * n_ot)
    return np.array(rows)


class TestRouteWithEmbeddings:
    """Teste route_with_embeddings en mockant _compute_embeddings."""

    @patch("src.routing._compute_embeddings")
    def test_clear_customer_service(self, mock_embed):
        """Score CS nettement supérieur → 'service_client' sans appel LLM."""
        mock_embed.return_value = _make_fake_embeddings(cs_score=0.95, ot_score=0.3)
        assert route_with_embeddings("ma commande") == "service_client"

    @patch("src.routing._compute_embeddings")
    def test_clear_off_topic(self, mock_embed):
        """Score hors sujet nettement supérieur → 'hors_sujet' sans appel LLM."""
        mock_embed.return_value = _make_fake_embeddings(cs_score=0.3, ot_score=0.95)
        assert route_with_embeddings("la météo") == "hors_sujet"

    @patch("src.routing.route_with_llm", return_value="service_client")
    @patch("src.routing._compute_embeddings")
    def test_ambiguous_falls_back_to_llm(self, mock_embed, mock_llm):
        """Scores proches → cas ambigu, bascule vers route_with_llm."""
        mock_embed.return_value = _make_fake_embeddings(cs_score=0.7, ot_score=0.65)
        result = route_with_embeddings("question ambiguë")
        assert result == "service_client"
        mock_llm.assert_called_once_with("question ambiguë")


# -----------------------------------------------------------------------
# Tests de route_with_llm (mock de la chaîne LCEL)
# -----------------------------------------------------------------------


class TestRouteWithLlm:
    """Teste route_with_llm en mockant la chaîne LCEL complète."""

    @patch("src.routing.StrOutputParser")
    @patch("src.routing.ChatOpenAI")
    @patch("src.routing.ROUTING_CHAT_PROMPT")
    def test_returns_service_client(self, mock_prompt, mock_chat_cls, mock_parser):
        """Le LLM retourne 'service_client' → la fonction aussi."""
        # On mocke l'opérateur | pour que prompt | llm | parser retourne
        # un objet dont invoke() renvoie la réponse voulue.
        mock_chain = mock_prompt.__or__.return_value.__or__.return_value
        mock_chain.invoke.return_value = "service_client"
        assert route_with_llm("Où en est ma commande ?") == "service_client"

    @patch("src.routing.StrOutputParser")
    @patch("src.routing.ChatOpenAI")
    @patch("src.routing.ROUTING_CHAT_PROMPT")
    def test_returns_hors_sujet(self, mock_prompt, mock_chat_cls, mock_parser):
        """Le LLM retourne 'hors_sujet' → la fonction aussi."""
        mock_chain = mock_prompt.__or__.return_value.__or__.return_value
        mock_chain.invoke.return_value = "hors_sujet"
        assert route_with_llm("Quel temps fait-il ?") == "hors_sujet"


# -----------------------------------------------------------------------
# Tests de SemanticRouter avec stratégie "llm" (mock)
# -----------------------------------------------------------------------


class TestSemanticRouterLlm:
    """Teste la classe SemanticRouter avec la stratégie 'llm'."""

    @patch("src.routing.route_with_llm", return_value="service_client")
    def test_llm_strategy_service_client(self, mock_llm):
        """Stratégie 'llm' → délègue à route_with_llm."""
        router = SemanticRouter(strategy="llm")
        assert router.classify("Ma commande est en retard") == "service_client"
        mock_llm.assert_called_once()

    @patch("src.routing.route_with_llm", return_value="hors_sujet")
    def test_llm_strategy_off_topic(self, mock_llm):
        router = SemanticRouter(strategy="llm")
        assert router.classify("Raconte une blague") == "hors_sujet"

    @patch("src.routing.route_with_llm", side_effect=Exception("API down"))
    def test_llm_strategy_fallback_to_keywords(self, mock_llm):
        """Si l'API échoue en stratégie 'llm', bascule vers les mots-clés."""
        router = SemanticRouter(strategy="llm")
        assert router.classify("Statut de ma commande") == "service_client"
        assert router.classify("Quel temps fait-il ?") == "hors_sujet"

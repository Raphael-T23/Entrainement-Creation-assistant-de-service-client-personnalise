"""Tests pour le module de routage sémantique."""

from src.routing import (
    SemanticRouter,
    route_with_keywords,
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

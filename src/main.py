"""Point d'entrée interactif pour l'assistant de service client.

Fournit une interface de chat en ligne de commande pour tester le bot.
"""

import os
import sys

from dotenv import load_dotenv

from .bot import CustomerServiceBot


def main() -> None:
    """Lance le chatbot de service client interactif."""
    load_dotenv()

    # --- Lecture du mode mock depuis le fichier .env ---
    # Si MOCK_LLM=true, le bot utilise un faux LLM (FakeChatOpenAI) au lieu
    # d'OpenAI. Cela permet de tester le bot gratuitement, sans clé API.
    # En mode normal (MOCK_LLM absent ou false), la clé API reste obligatoire.
    mock_llm = os.getenv("MOCK_LLM", "false").strip().lower() == "true"

    if mock_llm:
        print("=" * 60)
        print("  \u26a0  MODE TEST (Mock LLM) \u2014 MOCK_LLM=true détecté")
        print("=" * 60)
        print()
        print("Le bot fonctionne avec un LLM simulé (FakeChatOpenAI).")
        print("Les réponses sont générées par des heuristiques simples,")
        print("pas par un vrai modèle de langage.")
        print("Pour utiliser OpenAI, retirez MOCK_LLM=true du fichier .env")
        print()
    else:
        # Comportement normal : la clé API OpenAI est obligatoire
        if not os.getenv("OPENAI_API_KEY"):
            print("Erreur : La variable d'environnement OPENAI_API_KEY n'est pas définie.")
            print("Créez un fichier .env basé sur .env.example et ajoutez votre clé API.")
            print("Astuce : ajoutez MOCK_LLM=true dans .env pour tester sans clé API.")
            sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    db_path = os.getenv("DATABASE_PATH", "orders.db")
    routing_strategy = os.getenv("ROUTING_STRATEGY", "embeddings")

    # Demander l'authentification de l'utilisateur
    print("=" * 60)
    print("  Assistant de Service Client - E-Commerce")
    print("=" * 60)
    print()

    email = input("Veuillez entrer votre adresse email : ").strip()
    if not email:
        print("Erreur : Email requis.")
        sys.exit(1)

    try:
        # --- Création du bot avec ou sans mock selon la config .env ---
        bot = CustomerServiceBot(
            user_email=email,
            model=model,
            routing_strategy=routing_strategy,
            db_path=db_path,
            mock_llm=mock_llm,
        )
    except ValueError as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    print(
        f"\nBonjour {bot.user['first_name']} {bot.user['last_name']} ! "
        "Comment puis-je vous aider aujourd'hui ?"
    )
    # --- Rappel visuel du mode mock dans le prompt de bienvenue ---
    if bot._mock_mode:
        print("(\U0001f9ea Mode test actif \u2014 réponses simulées, pas d'appel API)")
    print("(Tapez 'quit' ou 'exit' pour quitter)\n")

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir ! Bonne journée !")
            break

        response = bot.chat(user_input)
        print(f"\nAssistant : {response}\n")


if __name__ == "__main__":
    main()

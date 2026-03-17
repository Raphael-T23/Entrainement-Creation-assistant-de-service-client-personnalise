"""Interactive entry point for the customer service assistant.

Provides a command-line chat interface for testing the bot.
"""

import os
import sys

from dotenv import load_dotenv

from .bot import CustomerServiceBot


def main() -> None:
    """Run the interactive customer service chatbot."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Erreur : La variable d'environnement OPENAI_API_KEY n'est pas définie.")
        print("Créez un fichier .env basé sur .env.example et ajoutez votre clé API.")
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    db_path = os.getenv("DATABASE_PATH", "orders.db")
    routing_strategy = os.getenv("ROUTING_STRATEGY", "embeddings")

    # Prompt for user authentication
    print("=" * 60)
    print("  Assistant de Service Client - E-Commerce")
    print("=" * 60)
    print()

    email = input("Veuillez entrer votre adresse email : ").strip()
    if not email:
        print("Erreur : Email requis.")
        sys.exit(1)

    try:
        # ChatOpenAI reads OPENAI_API_KEY from the environment automatically
        bot = CustomerServiceBot(
            user_email=email,
            model=model,
            routing_strategy=routing_strategy,
            db_path=db_path,
        )
    except ValueError as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    print(
        f"\nBonjour {bot.user['first_name']} {bot.user['last_name']} ! "
        "Comment puis-je vous aider aujourd'hui ?"
    )
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


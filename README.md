# Assistant de Service Client - E-Commerce

Un assistant de chat automatisé pour le service client d'une entreprise de
e-commerce, capable de répondre aux questions des utilisateurs concernant le
statut de leurs commandes.

## Fonctionnalités

- **Consultation de commandes** : Vérifier le statut, les dates de livraison
  et les détails de toute commande passée.
- **Réponses en langage naturel** : Les statuts sont traduits en termes
  compréhensibles (ex : « expédiée » au lieu de « shipped »).
- **Transfert vers un agent humain** : Pour les demandes de modification ou
  d'annulation de commande.
- **Routage sémantique** : Les questions hors sujet sont filtrées avant
  d'atteindre le LLM.
- **Protection des données** : Un utilisateur ne peut accéder qu'à ses propres
  commandes. Les injections de prompt sont bloquées.
- **Mode test (Mock LLM)** : Un faux LLM intégré permet de tester le bot
  sans clé API OpenAI et sans frais, via la variable `MOCK_LLM=true`.

## Architecture

```
src/
├── __init__.py
├── database.py    # Couche d'accès à la base de données SQLite
├── prompts.py     # Prompts système et templates (ChatPromptTemplate)
├── routing.py     # Routage sémantique (embeddings, LCEL, mots-clés)
├── bot.py         # Logique principale du bot (ChatOpenAI, @tool, bind_tools)
├── fake_llm.py    # Faux LLM pour le mode test (MOCK_LLM=true)
└── main.py        # Point d'entrée interactif (CLI)

tests/
├── __init__.py
├── test_database.py
├── test_routing.py
└── test_bot.py
```

## Stack technique

Le projet s'appuie sur **LangChain** (`langchain` + `langchain-openai`) comme
couche d'orchestration LLM. Voici les composants utilisés :

| Composant            | Implémentation                                                         |
|----------------------|------------------------------------------------------------------------|
| Modèle LLM           | `langchain_openai.ChatOpenAI`                                          |
| Embeddings           | `langchain_openai.OpenAIEmbeddings`                                    |
| Routage LLM          | Chaîne LCEL : `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser`    |
| Définition des outils| Décorateur `@tool` de `langchain_core`                                 |
| Liaison outils/LLM   | `ChatOpenAI.bind_tools()`                                              |
| Historique           | Liste typée `BaseMessage` (HumanMessage, AIMessage, ToolMessage…)      |
| Templates de prompts | `ChatPromptTemplate` (via `get_system_prompt_template()`)              |

### Pourquoi LangChain ?

- **Composabilité LCEL** : le routage LLM s'écrit en une ligne de chaîne
  `prompt | llm | parser`, claire et maintenable.
- **Types de messages stricts** : `AIMessage.tool_calls` rend le traitement
  des appels d'outils robuste et explicite.
- **`@tool` décorateur** : les outils sont des fonctions Python ordinaires ;
  leurs schémas JSON sont générés automatiquement à partir des signatures et
  docstrings.
- **`bind_tools()`** : lie les outils au modèle de façon déclarative, sans
  duplication de la liste de définitions.
- **Écosystème** : compatibilité directe avec LangSmith (traçabilité),
  LangGraph (orchestration avancée) et des dizaines d'intégrations tierces.

## Prérequis

- Python 3.10+
- Une clé API OpenAI (facultative en mode test avec `MOCK_LLM=true`)

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Editez .env et ajoutez votre clé API OpenAI
```

## Configuration

Créez un fichier `.env` à la racine du projet :

```
OPENAI_API_KEY=sk-votre-cle-api
OPENAI_MODEL=gpt-4o-mini
DATABASE_PATH=orders.db
ROUTING_STRATEGY=embeddings
MOCK_LLM=false
```

| Variable           | Description                                           | Requis            |
|--------------------|-------------------------------------------------------|-------------------|
| `OPENAI_API_KEY`   | Clé API OpenAI                                        | Oui (sauf mock)   |
| `OPENAI_MODEL`     | Modèle à utiliser (défaut : `gpt-4o-mini`)            | Non               |
| `DATABASE_PATH`    | Chemin vers la base SQLite (défaut : `orders.db`)     | Non               |
| `ROUTING_STRATEGY` | Stratégie de routage (voir ci-dessous)                | Non               |
| `MOCK_LLM`         | `true` pour activer le faux LLM de test               | Non               |

### Stratégies de routage disponibles

| Stratégie    | Description                                                       | API requise |
|--------------|-------------------------------------------------------------------|-------------|
| `embeddings` | `OpenAIEmbeddings` + similarité cosinus                           | Oui         |
| `llm`        | Chaîne LCEL `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` | Oui         |
| `keywords`   | Correspondance par mots-clés (fallback, aucune API)               | Non         |

## Utilisation

```bash
python -m src.main
```

Le bot vous demandera votre adresse email pour vous authentifier, puis vous
pourrez poser vos questions en langage naturel.

### Mode test (Mock LLM)

Pour tester le bot sans clé API OpenAI, activez le mode mock dans `.env` :

```
MOCK_LLM=true
```

Dans ce mode :
- Le bot utilise `FakeChatOpenAI` (défini dans `src/fake_llm.py`), un faux
  LLM qui analyse les messages par mots-clés et regex.
- Le routage sémantique bascule automatiquement sur la stratégie `keywords`
  (aucun appel API).
- Les outils (consultation de commandes, transfert agent humain) fonctionnent
  normalement.
- Un bandeau d'avertissement s'affiche au démarrage pour rappeler que le mode
  test est actif.

Les réponses en mode mock sont moins naturelles qu'avec un vrai LLM, mais le
flux complet (routage → appel d'outils → réponse) est exercé.

Pour repasser en mode normal, retirez `MOCK_LLM=true` ou mettez
`MOCK_LLM=false`, et assurez-vous que `OPENAI_API_KEY` est bien définie.

### Exemples de questions supportées

- « Où en est ma commande ? »
- « Quand sera livrée ma commande n°42 ? »
- « Combien de commandes ai-je passées ? »
- « Je veux annuler ma commande. »
- « Quel est l'état du paiement ? »

## Tests

```bash
python -m pytest tests/ -v
```

## Base de données

La base SQLite `orders.db` contient deux tables :

- **users** (40 enregistrements) : informations des clients
- **orders** (100 enregistrements) : commandes avec statuts
  - `invoiced` : facturée (en attente d'expédition)
  - `shipped` : expédiée (en cours de livraison)
  - `delivered` : livrée

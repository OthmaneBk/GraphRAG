Développement d'un système GraphRAG (Graph Retrieval-Augmented Generation) permettant d'interroger des documents internes en s'appuyant sur un Knowledge Graph et des modèles de langage (LLM). Le projet vise à transformer des questions en langage naturel en requêtes Cypher, exécutées sur une base de données graphe afin de fournir des réponses précises et contextualisées.

1- Knowledge Graph : modélisation des relations entre personnes, rôles et départements dans une base de données graphe avec Memgraph.

2- Prompt Engineering : conception d'un prompt structuré pour guider le LLM dans la génération de requêtes Cypher précises.

3- Few-shot Prompting : utilisation de plusieurs exemples Question → Requête Cypher → Résultat afin d'améliorer les performances et la fiabilité du modèle.

4- LLM Orchestration : intégration de Llama 3.1 via Ollama avec LangChain pour automatiser la génération et l'exécution des requêtes.

5- GraphCypherQAChain : création d'une chaîne capable de transformer une question en langage naturel, d'exécuter la requête sur le graphe et de retourner la réponse.

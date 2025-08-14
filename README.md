# RAG Multimodal avec Graphe de Connaissances

## Installation

### Prérequis
- Python 3.8 ou supérieur
- Ollama (pour les modèles de langage)
- Neo4j Desktop (pour la base de données de graphe)

### Installation des dépendances
```bash
pip install ollama langchain langchain-experimental langchain-community \
    langchain-ollama json-repair neo4j langchain-neo4j \
    llama-index llama-index-graph-stores-neo4j \
    llama-index-storage-context llama-index-llms-langchain
```

## Configuration

### Configuration de Neo4j
1. Installer Neo4j Desktop
2. Créer un nouveau projet et une base de données
3. Créer un utilisateur avec les droits appropriés

### Configuration des clés API
1. Créer un compte sur https://atlas.nomic.ai/data
2. Définir les variables d'environnement nécessaires :
```bash
setx NOMIC_API_KEY "votre_api_key"
setx NEO4J_USER_LOGIN "votre_utilisateur_neo4j"
setx NEO4J_USER_PWD "votre_mot_de_passe_neo4j"
```
3. Redémarrer votre environnement de développement

# RAG Multimodal avec Graphe de Connaissances

## Installation

### Prérequis
- Python 3.8 ou supérieur
- Ollama (pour les modèles de langage)
- Neo4j Desktop (pour la base de données de graphe)

### Installation des dépendances
```bash
pip install ollama langchain langchain-experimental langchain-community \
    langchain-ollama json-repair neo4j langchain-neo4j \
    llama-index llama-index-graph-stores-neo4j \
    llama-index-storage-context llama-index-llms-langchain
```

## Configuration

### Configuration de Neo4j
1. Installer Neo4j Desktop
2. Créer un nouveau projet et une base de données
3. Créer un utilisateur avec les droits appropriés

### Configuration des clés API
1. Créer un compte sur https://atlas.nomic.ai/data
2. Définir les variables d'environnement nécessaires :
```bash
setx NOMIC_API_KEY "votre_api_key"
setx NEO4J_USER_LOGIN "votre_utilisateur_neo4j"
setx NEO4J_USER_PWD "votre_mot_de_passe_neo4j"
```
3. Redémarrer votre environnement de développement

## Structure du projet

```
multimodal-rag-Gemma-/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── data/                    # Données d'exemple
├── notebooks/               # Notebooks Jupyter
│   └── NLP_Project_Graph_Based_RAG.ipynb
└── src/                     # Code source
    ├── config.py            # Configuration
    ├── graph_utils.py       # Utilitaires Neo4j
    ├── models.py            # Modèles LLM
    └── rag.py               # Logique RAG
```

## Utilisation

### 1. Démarrer les services
```bash
# Démarrer Neo4j
neo4j start

# Démarrer Ollama (si nécessaire)
ollama serve
```

### 2. Extraire les connaissances
```python
from src.rag import KnowledgeGraphRAG

# Initialiser le RAG
rag = KnowledgeGraphRAG()

# Extraire les connaissances depuis un document
rag.process_document("chemin/vers/document.pdf")
```

### 3. Interroger le système
```python
# Poser une question
response = rag.query("Quelles sont les entités mentionnées dans le document ?")
print(response)
```

## Configuration des modèles

### Modèles par défaut
- **Embedding de texte** : DC1LEX/nomic-embed-text-v1.5-multimodal
- **Embedding d'images** : nomic-embed-vision-v1.5
- **Modèle d'inférence** : gemma3:4b

## Auteurs


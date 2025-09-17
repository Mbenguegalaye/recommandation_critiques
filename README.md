# recommandation_critiques
Système de recommandation de critiques de films basé sur TF-IDF et similarité textuelle

---

## 1️⃣ Contexte / Objectif
Ce projet permet à un utilisateur de voir, lorsqu’il lit une critique, les critiques les plus similaires pour le **même film**, avec la possibilité de pondérer par note et de trier par popularité (likes).

L’algorithme se base sur la similarité de contenu des critiques pour faire des recommandations pertinentes.

---

## 2️⃣ Préparation des données

Les critiques sont chargées depuis des fichiers CSV (fightclub_critiques.csv, interstellar_critiques.csv) et combinées dans un seul DataFrame.
Les lignes où review_content est manquant ont été supprimées, car cette colonne est clé pour la recommandation.
review_title contient beaucoup de valeurs manquantes et est optionnelle : elle est conservée pour affichage ou enrichissement ultérieur, mais pas intégrée au pipeline principal.
Les textes sont **nettoyés** via la fonction clean_text (suppression HTML et passage en minuscules) et stockés dans la colonne clean_review.
Un **TF-IDF** est calculé sur clean_review pour représenter chaque critique.

---

## 3️⃣ Fonction de recommandation

La fonction principale est :

```python
get_similar_reviews(critique_id, top_n=5, use_rating=False, rating_weight=0.5)
```

**Paramètres :**

**critique_id** : ID de la critique de référence
**top_n** : nombre de critiques à retourner
**use_rating** : si True, pondère la similarité selon la note
**rating_weight** : poids de la note dans le score combiné (0 à 1)

La fonction retourne un DataFrame contenant la critique d’entrée et les critiques similaires, triées par similarité puis par popularité.

---

## 4️⃣ Exemple d’utilisation

```python
# Exemple : récupérer les 5 critiques les plus similaires à la critique d'ID 10
result = get_similar_reviews(critique_id=10, top_n=5, use_rating=True, rating_weight=0.3)
print(result)
```
---

## 5️⃣ System Design

**Objectif** : permettre à un utilisateur de consulter rapidement des critiques similaires pour le même film.

**Architecture :**

- **Frontend :** page de lecture d’une critique + section “Critiques similaires”
- **Backend :** fonction Python get_similar_reviews
- Filtre par film
- Nettoyage texte (clean_text)
- Calcul TF-IDF + similarité cosinus
- Pondération par note si nécessaire
- Tri par similarité et popularité
- **Base de données / stockage :** critiques avec ID, film, texte, note et nombre de likes

**Extensions possibles :**

- Passage à des embeddings (SBERT/OpenAI) pour meilleure compréhension sémantique
- Stockage de vecteurs TF-IDF ou embeddings pour accélérer les requêtes
- Filtres par genre, année ou utilisateur

---

## 6️⃣ Installation et utilisation

```python
# 1️ Cloner le repository
git clone https://github.com/<votre-utilisateur>/recommandation_critiques.git
cd recommandation_critiques

# 2️ Installer les dépendances
pip install pandas numpy scikit-learn nltk

# 3️ Placer les fichiers CSV dans le dossier du projet
# Assurez-vous d’avoir téléchargé ou copié les fichiers suivants :
# fightclub_critiques.csv
# interstellar_critiques.csv

# 4️ Lancer le script principal
python recommandation_critiques.py
```
Cela exécutera le pipeline complet : chargement des données, nettoyage, TF-IDF et recommandations de critiques similaires.

---

## 7️⃣ Option Notebook

```python
# Utiliser la fonction dans un Jupyter Notebook

import pandas as pd
from recommandation_critiques import get_similar_reviews, load_and_clean_data

# Charger et nettoyer les données
df = load_and_clean_data(['fightclub_critiques.csv', 'interstellar_critiques.csv'])

# Récupérer les 5 critiques les plus similaires à la critique d'ID 10
result = get_similar_reviews(
    critique_id=10,      
    top_n=5,
    use_rating=True,     
    rating_weight=0.3    
)

# Afficher les résultats
print(result)
```
---

## 8️⃣ Extensions possibles

Utiliser des embeddings sémantiques (SBERT/OpenAI) pour une meilleure compréhension du texte.
Ajouter des filtres par genre, année ou utilisateur.
Pré-calculer et stocker la matrice TF-IDF pour accélérer les recommandations sur un grand nombre de critiques.
Intégrer la recommandation dans un frontend web pour afficher directement les critiques similaires lors de la lecture.

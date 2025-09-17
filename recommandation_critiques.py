# 1️⃣ Import et préparation
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk

# Télécharger les stopwords français
nltk.download("stopwords")
from nltk.corpus import stopwords
french_stopwords = stopwords.words("french")

# 2️⃣ Chargement des données
fightclub_df = pd.read_csv("fightclub_critiques.csv")
interstellar_df = pd.read_csv("interstellar_critiques.csv")

# Ajouter une colonne pour identifier le film
fightclub_df["movie_title"] = "Fight Club"
interstellar_df["movie_title"] = "Interstellar"

# Combiner les deux films dans un seul DataFrame
df = pd.concat([fightclub_df, interstellar_df], ignore_index=True)

# 3️⃣ Vérification et nettoyage des données
print("Valeurs manquantes par colonne :")
print(df.isnull().sum())

# Supprimer les lignes où review_content est NaN
df = df.dropna(subset=["review_content"]).reset_index(drop=True)

# Nettoyage des textes
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # retire les balises HTML
    text = text.lower()
    return text

# Appliquer le nettoyage sur toutes les critiques
df["clean_review"] = df["review_content"].apply(clean_text)

# 4️⃣ TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words=french_stopwords)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_review"])
print("TF-IDF calculé pour", tfidf_matrix.shape[0], "critiques et", tfidf_matrix.shape[1], "mots/features.")

# 5️⃣ Fonction de recommandation de critiques similaires
def get_similar_reviews(critique_id, top_n=5, use_rating=False, rating_weight=0.5):
    """
    Récupère les top N critiques les plus similaires à une critique donnée,
    avec possibilité de pondérer par note.

    Paramètres :
    - critique_id : ID de la critique de référence
    - top_n : nombre de critiques à retourner
    - use_rating : si True, pondère la similarité selon la note
    - rating_weight : poids de la note dans le score combiné (0 à 1)
    """
    # Vérifier que l'ID existe
    if critique_id not in df["id"].values:
        return f"Critique {critique_id} introuvable"
    
    # Récupérer l'indice de la critique
    idx = df.index[df["id"] == critique_id][0]
    movie = df.loc[idx, "movie_title"]
    rating_ref = df.loc[idx, "rating"]
    
    # Filtrer les critiques du même film
    movie_indices = df[df["movie_title"] == movie].index
    movie_tfidf = tfidf_matrix[movie_indices]
    
    # Calculer la similarité cosine
    sim_scores = cosine_similarity(tfidf_matrix[idx], movie_tfidf).flatten()
    
    # Exclure la critique elle-même
    sim_scores[df.index.get_indexer([idx])] = 0
    
    # Pondération par rating si demandé
    if use_rating:
        rating_diff = np.abs(df.loc[movie_indices, "rating"] - rating_ref)
        rating_score = 1 - (rating_diff / df["rating"].max())  # Normaliser entre 0 et 1
        sim_scores = (1 - rating_weight) * sim_scores + rating_weight * rating_score
    
    # Trier indices par score combiné
    top_indices = sim_scores.argsort()[::-1][:top_n]
    
    # Récupérer la critique d’entrée
    current_review = df.loc[[idx], ["id", "review_title", "review_content", "rating", "gen_review_like_count"]].copy()
    current_review["similarity"] = 1.0  # Similarité maximale avec elle-même
    
    # Récupérer les critiques similaires
    similar_reviews = df.iloc[movie_indices[top_indices]][
        ["id", "review_title", "review_content", "rating", "gen_review_like_count"]
    ].copy()
    similar_reviews["similarity"] = sim_scores[top_indices]
    
    # Trier secondairement par nombre de likes
    similar_reviews = similar_reviews.sort_values(by=["similarity", "gen_review_like_count"], ascending=[False, False])
    
    # Concaténer : critique d'entrée + résultats
    result = pd.concat([current_review, similar_reviews], ignore_index=True)
    
    return result.reset_index(drop=True)

# 6️⃣ Affichage complet du DataFrame
pd.set_option('display.max_colwidth', None)

# 7️⃣ Exemple d'utilisation de la fonction de recommandation
# Choisir un ID de critique existant dans ton DataFrame
critique_id_exemple = 10  # Remplacer par un ID réel de df

# Appeler la fonction pour récupérer les 5 critiques les plus similaires
top_similaires = get_similar_reviews(
    critique_id=critique_id_exemple,
    top_n=5,
    use_rating=True,     # Toujours pondérer avec la note
    rating_weight=0.3    # Poids de la note dans le score combiné
)

# Afficher le DataFrame résultat
print(top_similaires)

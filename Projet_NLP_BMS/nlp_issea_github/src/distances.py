"""
distances.py
══════════════════════════════════════════════════════════════════════════════
Calcul des Distances Sémantiques Enrichies — Pipeline NLP
══════════════════════════════════════════════════════════════════════════════

Implémente trois métriques de distance et le topic drift LDA
pour quantifier le glissement sémantique entre 2024 et 2025.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock

try:
    from src.config import STOPWORDS_BUDGET
except ImportError:
    from config import STOPWORDS_BUDGET


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCES SÉMANTIQUES
# ─────────────────────────────────────────────────────────────────────────────

def calculer_distances_completes(
    emb_2024: np.ndarray,
    emb_2025: np.ndarray,
) -> dict:
    """
    Calcule trois métriques de distance entre paires d'embeddings.
    
    Pour chaque article i, mesure la distance entre sa version 2024
    et sa version 2025 via :
      - Similarité cosinus (1 = identique, 0 = orthogonal)
      - Distance euclidienne L2
      - Distance de Manhattan L1
    
    Args:
        emb_2024: Matrice d'embeddings 2024 (n × d)
        emb_2025: Matrice d'embeddings 2025 (n × d)
    Returns:
        Dict avec clés 'cosinus', 'euclidienne', 'manhattan'
    """
    n = len(emb_2024)
    assert len(emb_2025) == n, "Les deux matrices doivent avoir le même nombre de lignes"
    
    sim_cosinus      = np.array([
        max(0.0, cosine_similarity(emb_2024[i:i+1], emb_2025[i:i+1])[0, 0])
        for i in range(n)
    ])
    dist_euclidienne = np.array([
        euclidean_distances(emb_2024[i:i+1], emb_2025[i:i+1])[0, 0]
        for i in range(n)
    ])
    dist_manhattan   = np.array([
        cityblock(emb_2024[i], emb_2025[i])
        for i in range(n)
    ])
    
    print(f"✓ Distances calculées pour {n} paires d'articles")
    print(f"  Cosinus   : mean={sim_cosinus.mean():.3f}  min={sim_cosinus.min():.3f}  max={sim_cosinus.max():.3f}")
    print(f"  Euclidienne: mean={dist_euclidienne.mean():.3f}")
    print(f"  Manhattan : mean={dist_manhattan.mean():.3f}")
    
    return {
        'cosinus':     sim_cosinus,
        'euclidienne': dist_euclidienne,
        'manhattan':   dist_manhattan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC DRIFT LDA
# ─────────────────────────────────────────────────────────────────────────────

def calculer_topic_drift(
    texts_2024: list,
    texts_2025: list,
    n_topics:   int = 5,
) -> np.ndarray:
    """
    Mesure le glissement thématique via LDA (Latent Dirichlet Allocation).
    
    Algorithme :
      1. Entraîne un LDA sur les textes 2024
      2. Project les textes 2025 sur le même espace thématique
      3. Calcule la norme L2 entre les distributions 2024 et 2025
      4. Normalise en [0, 1] (0 = stable, 1 = changement total)
    
    Args:
        texts_2024: Textes de la Loi de Finances 2024
        texts_2025: Textes de la Loi de Finances 2025
        n_topics:   Nombre de topics LDA
    Returns:
        Vecteur de topic drift normalisé (n_articles,)
    """
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=STOPWORDS_BUDGET,
        min_df=1,
    )
    X_2024 = vectorizer.fit_transform(texts_2024)
    X_2025 = vectorizer.transform(texts_2025)
    
    n_comp = min(n_topics, X_2024.shape[0] - 1, X_2024.shape[1] - 1)
    lda = LatentDirichletAllocation(
        n_components=n_comp,
        random_state=42,
        max_iter=50,
        learning_method='online',
    )
    
    topics_2024 = lda.fit_transform(X_2024)
    topics_2025 = lda.transform(X_2025)
    
    drift = np.array([
        np.linalg.norm(topics_2024[i] - topics_2025[i])
        for i in range(len(topics_2024))
    ])
    
    # Normalisation min-max
    drift_norm = (drift - drift.min()) / (drift.max() - drift.min() + 1e-10)
    
    print(f"✓ Topic drift LDA ({n_comp} topics) : mean={drift_norm.mean():.3f}  max={drift_norm.max():.3f}")
    
    return drift_norm


# ─────────────────────────────────────────────────────────────────────────────
# DELTA TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def calculer_delta_tfidf(
    texts_2024: list,
    texts_2025: list,
    top_n:      int = 20,
) -> pd.DataFrame:
    """
    Identifie les mots-clés dont la fréquence TF-IDF a le plus évolué
    entre les deux exercices budgétaires.
    
    Utile pour comprendre les changements de vocabulaire prioritaire
    (ex: apparition de "numérique", "résilience", "souveraineté").
    
    Args:
        texts_2024: Textes 2024
        texts_2025: Textes 2025
        top_n:      Nombre de mots-clés à retourner
    Returns:
        DataFrame : mot_cle | tfidf_2024 | tfidf_2025 | delta | sens
    """
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=STOPWORDS_BUDGET,
        ngram_range=(1, 2),
    )
    
    X_2024 = vectorizer.fit_transform(texts_2024)
    X_2025 = vectorizer.transform(texts_2025)
    
    mean_2024  = np.asarray(X_2024.mean(axis=0)).flatten()
    mean_2025  = np.asarray(X_2025.mean(axis=0)).flatten()
    delta      = mean_2025 - mean_2024
    delta_abs  = np.abs(delta)
    
    feature_names = vectorizer.get_feature_names_out()
    top_indices   = delta_abs.argsort()[-top_n:][::-1]
    
    df_delta = pd.DataFrame({
        'mot_cle':    feature_names[top_indices],
        'tfidf_2024': mean_2024[top_indices].round(5),
        'tfidf_2025': mean_2025[top_indices].round(5),
        'delta':      delta[top_indices].round(5),
    })
    df_delta['sens'] = df_delta['delta'].apply(
        lambda d: '↑ Émergent 2025' if d > 0 else '↓ Déclinant 2025'
    )
    
    print(f"✓ Delta TF-IDF : {top_n} mots-clés identifiés")
    
    return df_delta

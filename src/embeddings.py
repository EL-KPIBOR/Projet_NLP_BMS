"""
embeddings.py
══════════════════════════════════════════════════════════════════════════════
Génération d'Embeddings — Pipeline NLP
══════════════════════════════════════════════════════════════════════════════

Détection automatique du backend disponible :
  1. Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2) — recommandé
  2. TF-IDF + LSA (TruncatedSVD 200 dimensions) — fallback offline
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing import Tuple, Optional, Any

try:
    from src.config import STOPWORDS_BUDGET
except ImportError:
    from config import STOPWORDS_BUDGET


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION DU BACKEND
# ─────────────────────────────────────────────────────────────────────────────

def detecter_backend() -> Tuple[str, Optional[Any]]:
    """
    Détecte si Sentence-BERT est disponible, sinon active TF-IDF+LSA.
    
    Returns:
        Tuple (backend_name: str, model: Any|None)
        - ("sentence-bert", SentenceTransformer) si disponible
        - ("tfidf-lsa", None) sinon
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Backend : Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2)")
        print("   Modèle multilingue — français natif, dimension 384")
        return "sentence-bert", model
    except (ImportError, Exception) as e:
        print(f"⚠️  Sentence-BERT non disponible ({type(e).__name__})")
        print("   → Activation backend TF-IDF + LSA (offline)")
        return "tfidf-lsa", None


# ─────────────────────────────────────────────────────────────────────────────
# BACKENDS D'EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def embeddings_sentence_bert(
    texts: list,
    model: Any,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Génère des embeddings avec Sentence-BERT multilingue.
    
    Args:
        texts:      Liste de textes
        model:      Modèle SentenceTransformer chargé
        batch_size: Taille des batchs de traitement
    Returns:
        Matrice d'embeddings normalisés (n_texts × 384)
    """
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    embeddings = normalize(embeddings, norm='l2')
    print(f"✓ Embeddings Sentence-BERT : shape={embeddings.shape}")
    return embeddings


def embeddings_tfidf_lsa(
    texts: list,
    n_components: int = 200,
    ngram_range: tuple = (1, 3),
    max_features: int = 15000,
) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    """
    Génère des embeddings avec TF-IDF + Analyse Sémantique Latente (LSA).
    
    Paramètres adaptés au vocabulaire budgétaire français du MINFI :
    - Trigrammes pour capturer "budget d'investissement public"
    - 15 000 features pour couvrir la terminologie spécialisée
    - SVD 200 composantes pour réduire la dimensionnalité
    
    Args:
        texts:       Liste de textes
        n_components: Nombre de composantes SVD
        ngram_range:  Plage n-grammes (défaut: 1 à 3)
        max_features: Nombre maximum de features TF-IDF
    Returns:
        Tuple (embeddings_normalisés, vectorizer, svd_model)
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=max_features,
        min_df=1,
        max_df=0.95,
        stop_words=STOPWORDS_BUDGET,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    
    # Adapter n_components aux contraintes de la matrice
    n_comp = min(n_components, X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1)
    
    svd = TruncatedSVD(n_components=n_comp, random_state=42, n_iter=7)
    X_lsa = svd.fit_transform(X_tfidf)
    X_norm = normalize(X_lsa, norm='l2')
    
    variance_expliquee = svd.explained_variance_ratio_.sum() * 100
    print(f"✓ Embeddings TF-IDF+LSA : shape={X_norm.shape}")
    print(f"  Variance expliquée : {variance_expliquee:.1f}%")
    
    return X_norm, vectorizer, svd


def calculer_embeddings(
    texts: list,
) -> Tuple[np.ndarray, Optional[TfidfVectorizer], Optional[TruncatedSVD]]:
    """
    Calcule les embeddings avec détection automatique du backend.
    
    Args:
        texts: Liste de textes à encoder
    Returns:
        Tuple (embeddings, vectorizer|None, svd|None)
        Le vectorizer et svd sont None si backend Sentence-BERT
    """
    backend, model = detecter_backend()
    
    if backend == "sentence-bert":
        emb = embeddings_sentence_bert(texts, model)
        return emb, None, None
    else:
        return embeddings_tfidf_lsa(texts)

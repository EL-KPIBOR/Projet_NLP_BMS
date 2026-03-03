"""
ruptures.py
══════════════════════════════════════════════════════════════════════════════
Détection des Ruptures de Discours — Pipeline NLP ISSEA
══════════════════════════════════════════════════════════════════════════════

Implémente le pipeline complet d'audit sémantique :
  1. Seuils adaptatifs (percentiles Q1/Q3)
  2. Détection d'anomalies multi-critères (Isolation Forest)
  3. Classification : rupture_critique / rupture / attention / stable
  4. Score de gravité composite
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from src.config import POIDS_SECTIONS
except ImportError:
    from config import POIDS_SECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# SEUILS ADAPTATIFS
# ─────────────────────────────────────────────────────────────────────────────

def calculer_seuils_adaptatifs(similarities: np.ndarray) -> dict:
    """
    Calcule les seuils de rupture et d'attention basés sur les percentiles
    empiriques de la distribution de similarité cosinus.
    
    Approche percentile-based (plus robuste que des seuils fixes) :
      - Q25 → seuil de rupture (25% des articles les moins stables)
      - Q75 → seuil d'attention (75% des articles les plus stables)
    
    Args:
        similarities: Vecteur de similarités cosinus
    Returns:
        Dict avec clés 'rupture' et 'attention'
    """
    seuils = {
        'rupture':   float(np.percentile(similarities, 25)),
        'attention': float(np.percentile(similarities, 75)),
    }
    print(f"📊 Seuils adaptatifs (Q1/Q3) :")
    print(f"   🔴 Rupture  (Q1=25%) : cosinus < {seuils['rupture']:.3f}")
    print(f"   🟡 Attention(Q3=75%) : cosinus < {seuils['attention']:.3f}")
    return seuils


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION D'ANOMALIES
# ─────────────────────────────────────────────────────────────────────────────

def detecter_anomalies_isolation_forest(
    distances_df:  pd.DataFrame,
    contamination: float = 0.15,
) -> np.ndarray:
    """
    Détecte les articles budgétairement anormaux via Isolation Forest.
    
    Utilise une approche multi-critères en combinant :
      - Similarité cosinus
      - Distance euclidienne
      - Distance de Manhattan
      - Topic drift LDA (si disponible)
    
    Args:
        distances_df:  DataFrame contenant les colonnes de distance
        contamination: Proportion d'anomalies attendue (défaut 15%)
    Returns:
        Vecteur de labels : -1 (anomalie) ou 1 (normal)
    """
    features = ['cosinus', 'euclidienne', 'manhattan']
    if 'topic_drift' in distances_df.columns:
        features.append('topic_drift')
    
    # Ne garder que les features disponibles
    features = [f for f in features if f in distances_df.columns]
    
    X = distances_df[features].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    iso    = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X_norm)
    
    n_anomalies = (labels == -1).sum()
    print(f"✓ Isolation Forest : {n_anomalies} anomalies détectées sur {len(labels)} articles ({n_anomalies/len(labels)*100:.1f}%)")
    
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION DES RUPTURES
# ─────────────────────────────────────────────────────────────────────────────

def detecter_ruptures_avancees(
    df:                pd.DataFrame,
    seuils_adaptatifs: dict,
    anomalies:         np.ndarray,
) -> np.ndarray:
    """
    Classifie chaque article en : rupture_critique / rupture / attention / stable.
    
    Règles de classification multi-critères :
      - rupture_critique : anomalie ET cosinus < seuil_rupture
      - rupture          : cosinus < seuil_rupture OU anomalie isolée
      - attention        : seuil_rupture ≤ cosinus < seuil_attention
      - stable           : cosinus ≥ seuil_attention
    
    Args:
        df:                DataFrame avec colonne 'cosinus'
        seuils_adaptatifs: Dict {'rupture': float, 'attention': float}
        anomalies:         Vecteur labels Isolation Forest
    Returns:
        Vecteur de catégories (array de str)
    """
    n          = len(df)
    labels     = np.array(['stable'] * n, dtype=object)
    seuil_rupt = seuils_adaptatifs['rupture']
    seuil_att  = seuils_adaptatifs['attention']
    
    df_reset = df.reset_index(drop=True)
    
    for i in range(n):
        cos        = df_reset.loc[i, 'cosinus']
        is_anomaly = (anomalies[i] == -1)
        
        if is_anomaly and cos < seuil_rupt:
            labels[i] = 'rupture_critique'
        elif cos < seuil_rupt or is_anomaly:
            labels[i] = 'rupture'
        elif cos < seuil_att:
            labels[i] = 'attention'
        else:
            labels[i] = 'stable'
    
    counts = {cat: (labels == cat).sum() for cat in ['rupture_critique','rupture','attention','stable']}
    print(f"✓ Classification ruptures :")
    for cat, n_cat in counts.items():
        emoji = {'rupture_critique':'🔴🔴','rupture':'🔴','attention':'🟡','stable':'🟢'}[cat]
        print(f"  {emoji} {cat.upper():20s}: {n_cat:2d} ({n_cat/n*100:5.1f}%)")
    
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# SCORE DE GRAVITÉ COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────

def calculer_score_gravite(df: pd.DataFrame) -> np.ndarray:
    """
    Score de gravité composite sur [0, 100] intégrant :
      - Instabilité sémantique : (1 - cosinus)              poids 45%
      - Dérive thématique LDA : topic_drift normalisé        poids 30%
      - Variation budgétaire  : |ΔBudget| normalisé          poids 25%
    Pondéré par le poids de la section budgétaire.
    
    Formule :
      G = (0.45 × cos_inv + 0.30 × drift + 0.25 × var_abs) × poids × 100
    
    Args:
        df: DataFrame avec colonnes cosinus, topic_drift,
            variation_budget, type_section
    Returns:
        Vecteur de scores gravité (n_articles,)
    """
    scores = []
    
    cos_inv  = 1.0 - df['cosinus'].values
    
    if 'topic_drift' in df.columns:
        drift = df['topic_drift'].values
        drift = (drift - drift.min()) / (drift.max() - drift.min() + 1e-10)
    else:
        drift = cos_inv  # fallback
    
    if 'variation_budget' in df.columns:
        var_abs = np.abs(df['variation_budget'].fillna(0).values) / 100.0
        var_abs = np.clip(var_abs, 0, 1)
    else:
        var_abs = cos_inv
    
    poids = df.get('poids_section', pd.Series([1.0] * len(df))).fillna(1.0).values
    
    raw_scores = (0.45 * cos_inv + 0.30 * drift + 0.25 * var_abs) * poids * 100
    
    return np.clip(raw_scores, 0, 100).round(1)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLET AUDIT PARTIE I
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_audit_complet(
    df:              pd.DataFrame,
    embeddings_2024: np.ndarray,
    embeddings_2025: np.ndarray,
    distances:       dict,
    topic_drift:     np.ndarray,
) -> pd.DataFrame:
    """
    Exécute le pipeline complet de la Partie I :
    distances → détection anomalies → classification → score gravité.
    
    Args:
        df:              DataFrame principal
        embeddings_2024: Embeddings 2024 (n × d)
        embeddings_2025: Embeddings 2025 (n × d)
        distances:       Dict {cosinus, euclidienne, manhattan}
        topic_drift:     Vecteur topic drift LDA
    Returns:
        DataFrame enrichi avec colonnes d'audit
    """
    df = df.reset_index(drop=True).copy()
    
    # Ajout des distances
    df['cosinus']     = distances['cosinus']
    df['euclidienne'] = distances['euclidienne']
    df['manhattan']   = distances['manhattan']
    df['topic_drift'] = topic_drift
    
    # Seuils adaptatifs
    seuils = calculer_seuils_adaptatifs(df['cosinus'].values)
    
    # Anomalies
    print("\n🔍 Détection anomalies (Isolation Forest)...")
    anomalies = detecter_anomalies_isolation_forest(df)
    df['anomalie'] = (anomalies == -1)
    
    # Classification
    print("\n🚨 Classification ruptures...")
    df['categorie'] = detecter_ruptures_avancees(df, seuils, anomalies)
    
    # Score gravité
    print("\n⚖️  Calcul score gravité composite...")
    df['score_gravite'] = calculer_score_gravite(df)
    
    print(f"\n✅ Score gravité moyen : {df['score_gravite'].mean():.1f}/100")
    print(f"   Glissement sémantique global : {(1 - df['cosinus'].mean()) * 100:.1f}%")
    
    return df

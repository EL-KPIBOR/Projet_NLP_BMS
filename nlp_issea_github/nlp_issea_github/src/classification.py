"""
classification.py
══════════════════════════════════════════════════════════════════════════════
Classification Zero-Shot SND30 — Pipeline NLP ISSEA
══════════════════════════════════════════════════════════════════════════════

Aligne automatiquement les articles budgétaires sur les 4 piliers SND30
via deux backends complémentaires :

  Backend 1 (BART-MNLI) — Optimal, nécessite internet/GPU
    Modèle : facebook/bart-large-mnli (HuggingFace)
    Approche : Natural Language Inference multilingue

  Backend 2 (EH-NLI) — Fallback offline
    Expansion d'Hypothèses : 56 hypothèses × 4 piliers
    Algorithme : TF-IDF + Cosinus + MAX-pooling + Softmax(T=6)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from typing import Tuple, Optional, Any

try:
    from src.config import (
        CODES_PILIERS, LABELS_PILIERS, HYPOTHESES_SND30,
        SEUIL_MULTILABEL, SEUIL_INCERTITUDE,
    )
except ImportError:
    from config import (
        CODES_PILIERS, LABELS_PILIERS, HYPOTHESES_SND30,
        SEUIL_MULTILABEL, SEUIL_INCERTITUDE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION DU BACKEND
# ─────────────────────────────────────────────────────────────────────────────

def detecter_backend_classification() -> Tuple[str, Optional[Any]]:
    """
    Détecte si HuggingFace Transformers est disponible.
    
    Returns:
        ("bart-mnli", pipeline) ou ("eh-nli", None)
    """
    try:
        from transformers import pipeline
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
        print("✅ Backend : BART-MNLI (HuggingFace) — Niveau de précision optimal")
        return "bart-mnli", classifier
    except (ImportError, Exception) as e:
        print(f"⚠️  BART-MNLI non disponible ({type(e).__name__})")
        print("   → Activation backend EH-NLI (Expansion d'Hypothèses, offline)")
        return "eh-nli", None


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND BART-MNLI
# ─────────────────────────────────────────────────────────────────────────────

def classifier_bart_mnli(texte: str, classifier: Any) -> dict:
    """
    Classification avec BART-MNLI via Natural Language Inference.
    
    Args:
        texte:      Texte de l'article budgétaire
        classifier: Pipeline HuggingFace chargé
    Returns:
        Dict {'pilier', 'scores', 'confiance'}
    """
    candidate_labels = list(LABELS_PILIERS.values())
    result = classifier(texte[:512], candidate_labels, multi_label=True)
    
    # Mapper labels → codes piliers
    scores = {}
    for label, score in zip(result['labels'], result['scores']):
        code = [k for k, v in LABELS_PILIERS.items() if v == label][0]
        scores[code] = float(score)
    
    pilier_pred = max(scores, key=scores.get)
    
    return {
        'pilier':    pilier_pred,
        'scores':    scores,
        'confiance': scores[pilier_pred],
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND EH-NLI (Expansion d'Hypothèses)
# ─────────────────────────────────────────────────────────────────────────────

def classifier_eh_nli(texte: str, temperature: float = 6.0) -> dict:
    """
    Classification EH-NLI (Expansion d'Hypothèses) — mode entièrement offline.
    
    Algorithme en 4 étapes :
      1. Construit un corpus [texte + 56 hypothèses SND30]
      2. Vectorise par TF-IDF (trigrammes, vocabulaire budgétaire)
      3. Calcule la similarité cosinus texte ↔ chaque hypothèse
      4. MAX-pooling par pilier + Softmax calibré (T=6 pour stabilité)
    
    Choix de la température T=6 :
      - T faible (<2) : distribution trop concentrée (surconfiance)
      - T élevée (>10): distribution trop plate (sous-confiance)
      - T=6 : calibration empiriquement validée sur corpus budgétaire
    
    Args:
        texte:       Texte de l'article
        temperature: Paramètre de calibration Softmax
    Returns:
        Dict {'pilier', 'scores', 'confiance'}
    """
    # Construire corpus [texte_cible] + [toutes les hypothèses]
    all_hypotheses = []
    for p in CODES_PILIERS:
        all_hypotheses.extend(HYPOTHESES_SND30[p])
    
    corpus = [str(texte)] + all_hypotheses
    
    # TF-IDF sur corpus local
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True)
    X = vectorizer.fit_transform(corpus)
    
    # Similarités texte ↔ hypothèses
    similarities = cosine_similarity(X[0:1], X[1:])[0]
    
    # MAX-pooling par pilier
    scores_raw = {}
    idx = 0
    for p in CODES_PILIERS:
        n_hyp = len(HYPOTHESES_SND30[p])
        scores_raw[p] = float(similarities[idx: idx + n_hyp].max())
        idx += n_hyp
    
    # Softmax calibré
    scores_arr    = np.array([scores_raw[p] for p in CODES_PILIERS])
    scores_softmax = softmax(scores_arr / temperature)
    scores         = {p: float(s) for p, s in zip(CODES_PILIERS, scores_softmax)}
    
    pilier_pred = max(scores, key=scores.get)
    
    return {
        'pilier':    pilier_pred,
        'scores':    scores,
        'confiance': scores[pilier_pred],
    }


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def detecter_multilabel(scores: dict, seuil: float = SEUIL_MULTILABEL) -> list:
    """
    Détecte les piliers secondaires (articles thématiquement transversaux).
    Un pilier secondaire est retenu si son score ≥ seuil × score_dominant.
    
    Args:
        scores: Dict {pilier: score}
        seuil:  Ratio minimum du score secondaire vs dominant
    Returns:
        Liste des codes piliers secondaires
    """
    sorted_p = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    score_1st = sorted_p[0][1]
    return [p for p, s in sorted_p[1:] if s >= seuil * score_1st]


def evaluer_certitude(confiance: float, seuil: float = SEUIL_INCERTITUDE) -> str:
    """
    Évalue le niveau de certitude de la classification.
    
    Args:
        confiance: Score de confiance [0, 1]
        seuil:     Seuil d'incertitude
    Returns:
        'incertain' | 'modéré' | 'certain'
    """
    if confiance < seuil:    return 'incertain'
    elif confiance < 0.60:   return 'modéré'
    else:                    return 'certain'


def classifier_texte(
    texte:            str,
    backend:          str,
    classifier_model: Optional[Any] = None,
) -> dict:
    """
    Wrapper de classification — sélection automatique du backend.
    
    Args:
        texte:            Texte à classifier
        backend:          'bart-mnli' ou 'eh-nli'
        classifier_model: Pipeline HuggingFace (si bart-mnli)
    Returns:
        Dict {'pilier', 'scores', 'confiance'}
    """
    if backend == "bart-mnli" and classifier_model is not None:
        return classifier_bart_mnli(texte, classifier_model)
    else:
        return classifier_eh_nli(texte)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLET CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_classification_complet(
    df:               pd.DataFrame,
    col_texte:        str = 'texte_2025',
) -> pd.DataFrame:
    """
    Exécute la classification zero-shot SND30 sur tout le DataFrame.
    
    Args:
        df:        DataFrame avec colonne texte
        col_texte: Colonne texte à classifier
    Returns:
        DataFrame enrichi avec pilier_predit, scores P1-P4,
        confiance, certitude, multi_pilier, label_pilier
    """
    backend, classifier_model = detecter_backend_classification()
    
    # Colonne texte disponible
    if col_texte not in df.columns:
        col_texte = next((c for c in ['texte_2025','texte_2024','texte'] if c in df.columns), None)
        if col_texte is None:
            raise ValueError("Aucune colonne texte trouvée dans le DataFrame")
    
    print(f"\n📊 Classification de {len(df)} articles (colonne : '{col_texte}')...")
    print(f"   Backend : {backend}\n")
    
    resultats = []
    for idx, row in df.iterrows():
        texte = str(row.get(col_texte, row.get('article', '')))
        if len(texte) < 10:
            texte = str(row.get('article', ''))
        
        res = classifier_texte(texte, backend, classifier_model)
        
        piliers_sec   = detecter_multilabel(res['scores'])
        certitude     = evaluer_certitude(res['confiance'])
        
        resultats.append({
            'pilier_predit':  res['pilier'],
            'score_P1':       res['scores'].get('P1', 0.0),
            'score_P2':       res['scores'].get('P2', 0.0),
            'score_P3':       res['scores'].get('P3', 0.0),
            'score_P4':       res['scores'].get('P4', 0.0),
            'confiance':      round(res['confiance'], 3),
            'certitude':      certitude,
            'multi_pilier':   len(piliers_sec) > 0,
            'piliers_sec':    ','.join(piliers_sec) if piliers_sec else '',
        })
        
        if (idx + 1) % 5 == 0 or idx == len(df) - 1:
            print(f"  [{idx+1:3d}/{len(df)}] {row.get('article','?'):15s} → {res['pilier']} (conf={res['confiance']:.3f})")
    
    df_res = df.copy()
    for col, vals in pd.DataFrame(resultats).items():
        df_res[col] = vals.values
    
    # Label pilier lisible
    from src.config import LABELS_PILIERS as LP
    df_res['label_pilier'] = df_res['pilier_predit'].map(LP)
    
    print(f"\n✅ Classification terminée")
    print(f"   Confiance moyenne : {df_res['confiance'].mean():.3f}")
    print(f"   Articles multi-piliers : {df_res['multi_pilier'].sum()} ({df_res['multi_pilier'].mean()*100:.1f}%)")
    print(f"   Articles incertains    : {(df_res['certitude']=='incertain').sum()}")
    
    return df_res

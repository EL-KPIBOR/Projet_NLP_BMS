# 🇨🇲 Intelligence Artificielle & Finances Publiques
## Audit Sémantique et Classification Budgétaire par NLP — Loi de Finances Cameroun

> **ISSEA · ISE3-DS · Année Académique 2025-2026**  
> Projet encadré par **MBIA NDI Marie Thérèse**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table des Matières

- [Contexte & Problématique](#-contexte--problématique)
- [Architecture du Pipeline](#-architecture-du-pipeline)
- [Structure du Projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Dashboard Streamlit](#-dashboard-streamlit)
- [Résultats & Métriques](#-résultats--métriques)
- [Bibliographie](#-bibliographie)

---

## 🎯 Contexte & Problématique

Dans le cadre de la **Stratégie Nationale de Développement 2020-2030 (SND30)** du Cameroun, ce projet répond à la question :

> *Comment l'IA peut-elle mesurer mathématiquement l'évolution des priorités de l'État entre la Loi de Finances 2024 et 2025-2026 ? Existe-t-il un alignement statistiquement significatif entre le discours budgétaire et les piliers de la SND30 ?*

Le pipeline s'articule autour de **trois axes complémentaires** :

| Axe | Méthode | Objectif |
|-----|---------|----------|
| **I — Audit Sémantique** | Embeddings + Similarité Cosinus | Détecter les ruptures de discours 2024→2025 |
| **II — Classification Zero-Shot** | BART-MNLI / EH-NLI | Aligner les lignes budgétaires ↔ Piliers SND30 |
| **III — Analyse Statistique** | OLS, ANOVA, Clustering OPTICS | Corréler fréquences NLP ↔ montants BIP |

---

## 🏗️ Architecture du Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE NLP COMPLET                         │
│                                                                 │
│  PDF MINFI 2024/2025                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ EXTRACTION  │───▶│  PRÉTRAITEMENT   │───▶│  EMBEDDINGS   │  │
│  │  pdfplumber │    │  TF-IDF / Clean  │    │ SBERT / LSA   │  │
│  └─────────────┘    └──────────────────┘    └───────┬───────┘  │
│                                                     │           │
│            ┌────────────────────────────────────────┘           │
│            ▼                                                    │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  PARTIE I        │    │  PARTIE II       │                  │
│  │  AUDIT SÉMANTIQUE│    │  CLASSIF. SND30  │                  │
│  │  ─────────────── │    │  ──────────────  │                  │
│  │ • Cosinus        │    │ • BART-MNLI      │                  │
│  │ • Euclidienne    │    │ • EH-NLI fallback│                  │
│  │ • Topic Drift    │    │ • 4 Piliers SND30│                  │
│  │ • Isol. Forest   │    │ • F1 / Log-Loss  │                  │
│  │ • Score Gravité  │    │                  │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       ▼                                         │
│           ┌───────────────────────┐                             │
│           │       PARTIE III      │                             │
│           │  ANALYSE STATISTIQUE  │                             │
│           │  ──────────────────── │                             │
│           │ • Pearson / Spearman  │                             │
│           │ • ANOVA + Kruskal     │                             │
│           │ • Régression OLS HC3  │                             │
│           │ • Panel Data (FE/RE)  │                             │
│           │ • Clustering OPTICS   │                             │
│           └───────────┬───────────┘                             │
│                       │                                         │
│                       ▼                                         │
│           ┌───────────────────────┐                             │
│           │  DASHBOARD STREAMLIT  │                             │
│           │  Baromètre Sémantique │                             │
│           └───────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### Backends NLP (détection automatique)

| Composant | Backend Prioritaire | Fallback Offline |
|-----------|--------------------|--------------------|
| Embeddings | Sentence-BERT `paraphrase-multilingual-MiniLM-L12-v2` | TF-IDF + LSA (SVD 200 dim.) |
| Classification | BART-MNLI (`facebook/bart-large-mnli`) | EH-NLI (TF-IDF + MAX-pooling + Softmax) |

---

## 📁 Structure du Projet

```
nlp-audit-loi-finances-cameroun/
│
├── README.md                          # Documentation principale
├── requirements.txt                   # Dépendances Python
├── .gitignore                         # Fichiers ignorés
│
├── notebooks/
│   └── Pipeline_NLP_Complet_ISSEA.ipynb   # Notebook principal (Parties I+II+III)
│
├── src/                               # Code source modulaire
│   ├── __init__.py
│   ├── config.py                      # Configuration globale SND30
│   ├── extraction.py                  # Extraction PDF MINFI
│   ├── preprocessing.py               # Nettoyage et normalisation
│   ├── embeddings.py                  # Génération embeddings (SBERT / TF-IDF+LSA)
│   ├── distances.py                   # Calcul distances sémantiques
│   ├── ruptures.py                    # Détection ruptures + Isolation Forest
│   ├── classification.py              # Zero-shot SND30 (BART-MNLI / EH-NLI)
│   └── statistiques.py               # Tests statistiques + OLS + clustering
│
├── dashboard/
│   └── app.py                         # Dashboard Streamlit interactif
│
├── data/
│   ├── README_data.md                 # Instructions pour les PDF
│   └── demo_articles.csv              # Données de démonstration
│
├── outputs/                           # Résultats exportés (git-ignored)
│   ├── audit_resultats_complets.csv
│   ├── audit_ruptures_critiques.csv
│   └── audit_delta_tfidf.csv
│
└── tests/
    └── test_pipeline.py               # Tests unitaires
```

---

## ⚙️ Installation

### Prérequis
- Python **3.10+**
- pip ou conda

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/nlp-audit-loi-finances-cameroun.git
cd nlp-audit-loi-finances-cameroun
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. (Optionnel) Backends avancés

Pour activer Sentence-BERT et BART-MNLI (recommandé si connexion internet disponible) :

```bash
pip install sentence-transformers transformers torch
```

---

## 🚀 Utilisation

### Option A — Notebook Jupyter (recommandé pour la reproduction)

```bash
jupyter notebook notebooks/Pipeline_NLP_Complet_ISSEA.ipynb
```

**Avec vos PDF réels :**
1. Placez `Loi_Finance_Cameroun_2024.pdf` et `Loi_Finance_Cameroun_2025.pdf` à la racine
2. Dans la cellule de configuration : changez `MODE = "DEMO"` → `MODE = "PDF_REELS"`
3. Exécutez toutes les cellules (`Kernel > Restart & Run All`)

**Mode démonstration (sans PDF) :**
- Le notebook fonctionne directement avec des données enrichies simulant la réalité

### Option B — Pipeline Python (ligne de commande)

```bash
# Audit sémantique complet
python src/ruptures.py --pdf2024 Loi_Finance_Cameroun_2024.pdf \
                        --pdf2025 Loi_Finance_Cameroun_2025.pdf \
                        --output outputs/

# Classification SND30 uniquement
python src/classification.py --input outputs/audit_resultats_complets.csv
```

### Option C — Dashboard Streamlit

```bash
streamlit run dashboard/app.py
```

Ouvrir : [http://localhost:8501](http://localhost:8501)

---

## 📊 Dashboard Streamlit

Le tableau de bord interactif comprend **7 vues** :

| Onglet | Contenu |
|--------|---------|
| 🌡️ **Baromètre** | Jauge de glissement sémantique global + top ruptures |
| 📈 **Distributions** | Histogrammes cosinus/drift, box-plots, heatmap corrélations |
| 🗂️ **Tableau de bord** | Scatter topomap, tableau coloré, timeline stabilité |
| 💰 **Piliers SND30** | Répartition budgétaire BIP, radar gravité/cosinus |
| 🔑 **Delta TF-IDF** | Mots émergents 2025 vs déclinants 2024 |
| 🕸️ **Espace 3D** | Nuage 3D Cosinus × Euclidienne × Gravité |
| 📋 **Données brutes** | Recherche, filtre, export CSV |

**Fonctionnalités interactives :**
- Ajustement en temps réel des seuils de rupture/attention
- Filtrage par pilier SND30 et type de section
- Import de votre propre CSV de résultats
- Export des données filtrées

---

## 📈 Résultats & Métriques

### Partie I — Audit Sémantique

| Métrique | Valeur |
|----------|--------|
| Similarité cosinus moyenne | ~0.68 |
| Glissement sémantique global | ~32% |
| Ruptures critiques détectées | 2 articles |
| Topic drift moyen (LDA) | ~0.42 |

### Partie II — Classification SND30

| Métrique | Valeur |
|----------|--------|
| Accuracy (gold standard) | ~0.78 |
| F1-score pondéré | ~0.76 |
| Log-Loss global | ~1.07 |

### Partie III — Statistiques

- **ANOVA** : F = 3.24 (p < 0.05) → Différences significatives inter-piliers
- **OLS** : R² = 0.61, coefficient confiance NLP significatif (p < 0.05)
- **Clustering OPTICS** : 3 clusters, Silhouette = 0.52

---

## 📚 Bibliographie

1. Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805
2. Martin, L. et al. (2020). *CamemBERT: a Tasty French Language Model*. arXiv:1911.03894
3. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019
4. Lewis, M. et al. (2019). *BART: Denoising Sequence-to-Sequence Pre-training*. arXiv:1910.13461
5. République du Cameroun. *Document de Stratégie Nationale de Développement 2020-2030 (SND30)*. MINEPAT, 2020
6. République du Cameroun. *Loi de Finances 2024*. MINFI
7. République du Cameroun. *Loi de Finances 2025*. MINFI

---

## 👥 Auteurs

Projet réalisé dans le cadre du cours **Intelligence Artificielle et Finances Publiques**  
**ISSEA — Institut Sous-régional de Statistique et d'Économie Appliquée**  
Yaoundé, Cameroun · 2025-2026

---

## 📄 Licence

Ce projet est sous licence [MIT](LICENSE).  
Les données budgétaires sont issues de documents officiels du MINFI (domaine public).

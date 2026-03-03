# 🤝 Guide de Contribution — Projet NLP BMS

> **ISSEA · ISE3-DS · 2025-2026**  
> Dépôt : `git@github.com:EL-KPIBOR/Projet_NLP_BMS.git`

---

## 👥 Équipe & Responsabilités

| Membre | Branche | Responsabilité principale |
|--------|---------|--------------------------|
| **Membre 1** (Chef de projet) | `feature/audit-semantique` | Partie I — Embeddings, distances, ruptures |
| **Membre 2** | `feature/classification-snd30` | Partie II — Zero-shot, métriques F1/LogLoss |
| **Membre 3** | `feature/analyse-statistique` | Partie III — OLS, ANOVA, clustering |
| **Membre 4** | `feature/dashboard` | Dashboard Streamlit + README |

---

## 🌿 Structure des Branches

```
main          ← Version finale stable (protégée — merge uniquement via PR)
│
develop       ← Branche d'intégration (tests avant merge vers main)
│
├── feature/audit-semantique        ← Membre 1
├── feature/classification-snd30    ← Membre 2
├── feature/analyse-statistique     ← Membre 3
└── feature/dashboard               ← Membre 4
```

### Règles absolues
- ❌ **Jamais de push direct sur `main`**
- ❌ **Jamais de push direct sur `develop`**
- ✅ Toujours passer par une **Pull Request**
- ✅ Toujours partir de `develop` pour créer sa branche

---

## 🔄 Workflow Git Quotidien

### 1. Début de session — Synchroniser sa branche

```bash
# Se mettre sur sa branche feature
git checkout feature/ma-branche

# Récupérer les derniers changements de develop
git fetch origin
git merge origin/develop

# Si conflits → les résoudre, puis :
git add .
git commit -m "merge: synchronisation avec develop"
```

### 2. Travailler et committer

```bash
# Voir l'état de ses fichiers
git status

# Ajouter ses modifications
git add src/mon_fichier.py
# ou tout ajouter :
git add .

# Committer avec un message conventionnel (voir section ci-dessous)
git commit -m "feat(audit): calcul similarité cosinus + topic drift LDA"

# Pousser sur GitHub
git push origin feature/ma-branche
```

### 3. Fusionner vers develop — Pull Request

1. Aller sur GitHub → **"Compare & pull request"**
2. Base : `develop` ← Compare : `feature/ma-branche`
3. Titre clair + description des changements
4. **Assigner un reviewer** (un autre membre du groupe)
5. Attendre la validation → **Merge**

---

## 📝 Convention des Messages de Commit

Format : `type(scope): description courte`

| Type | Usage | Exemple |
|------|-------|---------|
| `feat` | Nouvelle fonctionnalité | `feat(audit): ajout Isolation Forest` |
| `fix` | Correction de bug | `fix(classification): KeyError colonne pilier` |
| `docs` | Documentation | `docs(readme): ajout section installation` |
| `style` | Formatage, pas de logique | `style(app): nettoyage CSS sidebar` |
| `refactor` | Refactoring sans nouvelle feature | `refactor(distances): optimisation boucle cosinus` |
| `test` | Ajout/modif tests | `test(preprocessing): test nettoyer_texte` |
| `data` | Données ou notebooks | `data(notebook): ajout cellules visualisation partie III` |
| `merge` | Merge de branches | `merge: synchronisation develop → feature/dashboard` |

---

## 🏗️ Structure du Projet

```
Projet_NLP_BMS/
│
├── README.md                    ← Documentation principale
├── CONTRIBUTING.md              ← Ce fichier
├── requirements.txt             ← Dépendances Python
├── .gitignore                   ← Fichiers ignorés
├── app.py                       ← Dashboard Streamlit (alias racine)
│
├── notebooks/
│   └── Pipeline_NLP_Complet_ISSEA.ipynb
│
├── src/                         ← Code source modulaire
│   ├── __init__.py
│   ├── config.py                ← Configuration globale SND30
│   ├── preprocessing.py         ← Nettoyage textes
│   ├── embeddings.py            ← SBERT / TF-IDF+LSA
│   ├── distances.py             ← Cosinus, Euclidienne, LDA
│   ├── ruptures.py              ← Détection ruptures + score gravité
│   └── classification.py        ← Zero-shot SND30
│
├── dashboard/
│   └── app.py                   ← Dashboard Streamlit complet
│
├── data/
│   ├── README_data.md
│   └── demo_articles.csv
│
└── outputs/                     ← Résultats (git-ignorés)
```

---

## ⚙️ Installation locale (chaque membre)

```bash
# 1. Cloner le dépôt
git clone git@github.com:EL-KPIBOR/Projet_NLP_BMS.git
cd Projet_NLP_BMS

# 2. Créer environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Se placer sur sa branche
git checkout feature/ma-branche
```

---

## 🔒 Protection des Branches (à configurer sur GitHub)

Aller sur : **Settings → Branches → Add rule**

Pour `main` :
- ✅ Require a pull request before merging
- ✅ Require approvals : **2**
- ✅ Dismiss stale pull request approvals
- ✅ Require status checks to pass

Pour `develop` :
- ✅ Require a pull request before merging
- ✅ Require approvals : **1**

---

## 🚨 Résolution de Conflits

```bash
# Si git merge génère un conflit :
git status           # Voir les fichiers en conflit

# Ouvrir le fichier en conflit, chercher :
# <<<<<<< HEAD
# ... ta version ...
# =======
# ... version distante ...
# >>>>>>> origin/develop

# Choisir/fusionner manuellement, puis :
git add fichier_conflit.py
git commit -m "fix: résolution conflit merge develop"
git push origin feature/ma-branche
```

---

## ✅ Checklist avant chaque Pull Request

- [ ] Mon code est testé et fonctionne sans erreur
- [ ] J'ai synchronisé ma branche avec `develop` récemment
- [ ] Mes messages de commit suivent la convention
- [ ] J'ai mis à jour `requirements.txt` si j'ai ajouté une dépendance
- [ ] Le notebook s'exécute de bout en bout (`Kernel > Restart & Run All`)
- [ ] J'ai assigné un reviewer

---

*Guide maintenu par le groupe · ISSEA ISE3-DS 2025-2026*

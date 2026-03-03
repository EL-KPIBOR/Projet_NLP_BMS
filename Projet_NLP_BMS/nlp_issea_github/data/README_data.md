# 📂 Données — Pipeline NLP ISSEA

## PDF des Lois de Finances

Ce dossier est destiné à recevoir les documents officiels du MINFI.

### Fichiers attendus

| Fichier | Description | Source |
|---------|-------------|--------|
| `Loi_Finance_Cameroun_2024.pdf` | Loi de Finances 2024 | MINFI Cameroun |
| `Loi_Finance_Cameroun_2025.pdf` | Loi de Finances 2025 | MINFI Cameroun |

> ⚠️ **Ces fichiers ne sont pas versionnés** dans le dépôt GitHub
> conformément au `.gitignore`. Placez-les à la **racine du projet**.

### Téléchargement

Les Lois de Finances sont disponibles sur le site officiel du MINFI :
- [minfi.cm](https://www.minfi.cm)
- Rubrique : Publications → Lois de Finances

### Mode démonstration

Si les PDF ne sont pas disponibles, le pipeline fonctionne automatiquement
en **mode démonstration** avec des données représentatives intégrées.

Pour activer le mode PDF réels :
```python
# Dans src/config.py
MODE = "PDF_REELS"  # au lieu de "DEMO"
```

## Fichier de démonstration

`demo_articles.csv` — Contient 21 articles représentatifs avec :
- Textes 2024 et 2025
- Montants budgétaires (Mds FCFA)
- Piliers SND30 pré-annotés (gold standard partiel)
- Distances sémantiques pré-calculées

"""
config.py
══════════════════════════════════════════════════════════════════════════════
Configuration Globale — Pipeline NLP Audit Sémantique
Loi de Finances Cameroun 2024-2025 | ISSEA · ISE3-DSM
══════════════════════════════════════════════════════════════════════════════

Ce module centralise toutes les constantes partagées entre les trois parties
du pipeline (Audit Sémantique, Classification SND30, Analyse Statistique).
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

PDF_2024 = ROOT_DIR / "Loi_Finance_Cameroun_2024.pdf"
PDF_2025 = ROOT_DIR / "Loi_Finance_Cameroun_2025.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# MODE D'EXÉCUTION
# ─────────────────────────────────────────────────────────────────────────────
#   "DEMO"       → données de démonstration intégrées (aucun PDF requis)
#   "PDF_REELS"  → extraction depuis les vrais PDF du MINFI
MODE = "DEMO"

# ─────────────────────────────────────────────────────────────────────────────
# SEUILS AUDIT SÉMANTIQUE
# ─────────────────────────────────────────────────────────────────────────────
SEUIL_RUPTURE_FIXE   = 0.60   # Similarité cosinus < ce seuil → rupture
SEUIL_ATTENTION_FIXE = 0.75   # Similarité cosinus < ce seuil → attention

# ─────────────────────────────────────────────────────────────────────────────
# POIDS SECTIONS BUDGÉTAIRES (pour le score de gravité composite)
# ─────────────────────────────────────────────────────────────────────────────
POIDS_SECTIONS = {
    'investissement': 1.5,
    'recettes':       1.3,
    'depenses':       1.2,
    'fiscal':         1.4,
    'gouvernance':    1.0,
    'autre':          0.8,
}

# ─────────────────────────────────────────────────────────────────────────────
# COULEURS AUDIT
# ─────────────────────────────────────────────────────────────────────────────
COLORS_AUDIT = {
    'rupture_critique': '#C0392B',
    'rupture':          '#E74C3C',
    'attention':        '#F39C12',
    'stable':           '#2ECC71',
}

# ─────────────────────────────────────────────────────────────────────────────
# STOPWORDS BUDGÉTAIRES (domaine Loi de Finances Cameroun)
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS_BUDGET = [
    'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou',
    'pour', 'dans', 'article', 'chapitre', 'alinea', 'paragraphe',
    'annexe', 'loi', 'decret', 'milliards', 'millions', 'fcfa',
    'francs', 'cfa', 'exercice', 'annee', 'est', 'sont', 'aux',
    'par', 'sur', 'en', 'au', 'ce', 'se', 'il', 'ils', 'elle',
    'elles', 'qui', 'que', 'qu', 'dont', 'avec', 'sans', 'entre',
    'sous', 'tel', 'tels', 'tout', 'tous', 'toute', 'toutes',
]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION SND30
# ─────────────────────────────────────────────────────────────────────────────
CODES_PILIERS = ['P1', 'P2', 'P3', 'P4']

LABELS_PILIERS = {
    'P1': "Transformation structurelle de l'économie",
    'P2': "Développement du capital humain",
    'P3': "Promotion de l'emploi et insertion économique",
    'P4': "Gouvernance et décentralisation",
}

LABELS_COURTS = {
    'P1': 'Transformation structurelle',
    'P2': 'Capital humain',
    'P3': 'Emploi / Insertion',
    'P4': 'Gouvernance',
}

COLORS_PILIERS = {
    'P1': '#2ECC71',
    'P2': '#3498DB',
    'P3': '#E67E22',
    'P4': '#9B59B6',
}

# Seuils classification
SEUIL_MULTILABEL  = 0.60   # Ratio score 2e pilier / 1er pilier → multi-label
SEUIL_INCERTITUDE = 0.40   # Confiance < ce seuil → classification incertaine

# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHÈSES EH-NLI (14 par pilier · 56 au total)
# Utilisées pour la classification offline (sans GPU)
# ─────────────────────────────────────────────────────────────────────────────
HYPOTHESES_P1 = [
    "transformation structurelle économie",
    "industrialisation développement",
    "infrastructure énergie transport",
    "agriculture modernisation productivité",
    "mines exploitation ressources",
    "numérique technologie innovation",
    "routes autoroutes aménagement",
    "électricité barrage centrale",
    "zones économiques spéciales",
    "compétitivité entreprises secteur privé",
    "production manufacturière industrie",
    "investissement infrastructure productive",
    "chaînes valeur ajoutée",
    "diversification économique exportation",
]

HYPOTHESES_P2 = [
    "capital humain développement",
    "éducation formation enseignement",
    "santé hôpital soins médicaux",
    "protection sociale solidarité",
    "école primaire secondaire université",
    "personnel enseignant formation continue",
    "structures sanitaires plateaux techniques",
    "lutte contre maladies prévention",
    "accès eau potable assainissement",
    "nutrition sécurité alimentaire",
    "jeunesse sport culture",
    "genre équité femmes enfants",
    "couverture maladie universelle",
    "qualifications compétences professionnelles",
]

HYPOTHESES_P3 = [
    "emploi insertion économique",
    "jeunes diplômés stage formation",
    "création entreprises entrepreneuriat",
    "microfinance crédit PME",
    "artisanat métiers secteur informel",
    "incubateurs startups innovation",
    "travaux haute intensité main œuvre",
    "insertion professionnelle accompagnement",
    "apprentissage qualification métiers",
    "auto-emploi activités génératrices revenus",
    "promotion entrepreneuriat féminin jeunesse",
    "fonds national emploi jeunes",
    "programme emploi rural agricole",
    "transition école vie active",
]

HYPOTHESES_P4 = [
    "gouvernance décentralisation",
    "administration publique modernisation",
    "collectivités territoriales communes",
    "déconcentration services publics",
    "transparence redevabilité anticorruption",
    "justice état droit institutions",
    "sécurité défense paix",
    "dématérialisation numérique administration",
    "cadastre foncier gestion domaine",
    "finances publiques contrôle audit",
    "réformes institutionnelles efficacité",
    "participation citoyenne démocratie locale",
    "services déconcentrés préfectures sous-préfectures",
    "renforcement capacités agents publics",
]

HYPOTHESES_SND30 = {
    'P1': HYPOTHESES_P1,
    'P2': HYPOTHESES_P2,
    'P3': HYPOTHESES_P3,
    'P4': HYPOTHESES_P4,
}

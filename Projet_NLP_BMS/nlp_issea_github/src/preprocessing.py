"""
preprocessing.py
══════════════════════════════════════════════════════════════════════════════
Prétraitement des textes budgétaires — Loi de Finances Cameroun
══════════════════════════════════════════════════════════════════════════════

Fonctions de nettoyage, normalisation et segmentation adaptées au
vocabulaire spécialisé du MINFI (Ministère des Finances).
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from src.config import STOPWORDS_BUDGET, POIDS_SECTIONS
except ImportError:
    from config import STOPWORDS_BUDGET, POIDS_SECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# NETTOYAGE DE TEXTE
# ─────────────────────────────────────────────────────────────────────────────

def nettoyer_texte(texte: str) -> str:
    """
    Normalisation Unicode + suppression caractères spéciaux.
    Conserve les décimales (points/virgules) pour les montants.
    
    Args:
        texte: Texte brut extrait du PDF
    Returns:
        Texte nettoyé en minuscules
    """
    if pd.isna(texte) or texte == "":
        return ""
    
    # Normalisation Unicode
    texte = unicodedata.normalize('NFC', str(texte))
    texte = texte.lower()
    
    # Suppression caractères spéciaux (sauf ponctuation utile)
    texte = re.sub(r'[^\w\s\.,;:!?\-éèêëàâùûüîïôœæç]', ' ', texte)
    
    # Normalisation espaces
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte


def extraire_montant_simple(texte: str) -> float:
    """
    Extrait le premier montant numérique (milliards FCFA) d'un texte.
    Gère les montants négatifs entre parenthèses ex: (985 milliards).
    
    Args:
        texte: Texte contenant un montant
    Returns:
        Montant en milliards FCFA (float), 0.0 si non trouvé
    """
    if not texte:
        return 0.0
    
    # Montant négatif entre parenthèses : (1 250 milliards)
    pattern_negatif = r'\((\d[\d\s]*(?:[.,]\d+)?)\s*(?:milliards?|millions?|mds?)?\s*(?:fcfa|f\.?cfa|francs?)?\)'
    match_neg = re.search(pattern_negatif, texte, re.IGNORECASE)
    if match_neg:
        val_str = re.sub(r'\s', '', match_neg.group(1)).replace(',', '.')
        try:
            return -float(val_str)
        except ValueError:
            pass
    
    # Montant positif standard : 1 380 milliards FCFA
    pattern_pos = r'(\d[\d\s]*(?:[.,]\d+)?)\s*(?:milliards?|mds?)\s*(?:fcfa|f\.?cfa|francs?)?'
    match_pos = re.search(pattern_pos, texte, re.IGNORECASE)
    if match_pos:
        val_str = re.sub(r'\s', '', match_pos.group(1)).replace(',', '.')
        try:
            return float(val_str)
        except ValueError:
            pass
    
    # Millions → convertir en milliards
    pattern_mil = r'(\d[\d\s]*(?:[.,]\d+)?)\s*millions?\s*(?:fcfa|f\.?cfa|francs?)?'
    match_mil = re.search(pattern_mil, texte, re.IGNORECASE)
    if match_mil:
        val_str = re.sub(r'\s', '', match_mil.group(1)).replace(',', '.')
        try:
            return float(val_str) / 1000.0
        except ValueError:
            pass
    
    return 0.0


def classifier_section(texte: str) -> str:
    """
    Classifie un texte budgétaire dans une section (investissement, recettes,
    depenses, fiscal, gouvernance, autre) via règles lexicales.
    
    Args:
        texte: Texte de l'article
    Returns:
        Catégorie de section (str)
    """
    texte_lower = str(texte).lower()
    
    keywords = {
        'investissement': [
            'investissement', 'bip', 'budget d\'investissement', 'infrastructure',
            'construction', 'réhabilitation', 'routes', 'énergie', 'barrage',
            'école', 'hôpital', 'adduction', 'électrification'
        ],
        'recettes': [
            'recettes', 'impôt', 'taxe', 'recouvrement', 'fiscalité',
            'douane', 'tva', 'ipp', 'is ', 'contribution'
        ],
        'fiscal': [
            'fiscal', 'exonération', 'déduction', 'abattement', 'régime fiscal',
            'assiette', 'taux d\'imposition'
        ],
        'gouvernance': [
            'gouvernance', 'décentralisation', 'commune', 'collectivité',
            'justice', 'sécurité', 'défense', 'administration', 'déconcentration'
        ],
        'depenses': [
            'salaire', 'rémunération', 'fonctionnement', 'dette',
            'transfert', 'subvention', 'masse salariale', 'pension'
        ],
    }
    
    for section, mots in keywords.items():
        if any(mot in texte_lower for mot in mots):
            return section
    
    return 'autre'


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DU DATAFRAME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def construire_dataframe_demo() -> pd.DataFrame:
    """
    Construit le DataFrame de démonstration enrichi simulant
    les articles des Lois de Finances 2024 et 2025.
    
    Returns:
        DataFrame avec colonnes : article, chapitre, texte_2024,
        texte_2025, montant_2024, montant_2025, type_section
    """
    data = {
        'article': [
            'Article 1',   'Article 3',   'Article 10',  'Article 21',
            'Article 25',  'Article 26',  'Article 30',  'Article 35',
            'Article 42',  'Article 56',  'Article 62',  'Article 70',
            'Article 82',  'BIP-010101',  'BIP-020101',  'BIP-020102',
            'BIP-020201',  'BIP-020301',  'BIP-020401',  'BIP-030101',
            'BIP-040101',
        ],
        'chapitre': [
            'Recettes fiscales',        'Dette publique',        'BIP Global',
            'Santé publique',           'Éducation nationale',   'Fonctionnement',
            'Grands Travaux',           'Transformation Numérique', 'Agriculture & Élevage',
            'Emploi & Insertion',       'Énergie électrique',    'Décentralisation',
            'Sécurité & Défense',       'BIP - Agriculture',     'BIP - Routes',
            'BIP - Énergie',            'BIP - Santé',           'BIP - Éducation',
            'BIP - Eau potable',        'BIP - Emploi',          'BIP - Décentralisation',
        ],
        'texte_2024': [
            "Les recettes fiscales de l'État pour l'exercice 2024 sont évaluées à 3 150 milliards FCFA, comprenant les impôts directs, indirects et les droits de douane.",
            "Le service de la dette publique pour 2024 est estimé à 780 milliards FCFA, dont 420 milliards pour la dette extérieure et 360 milliards pour la dette intérieure.",
            "Le Budget d'Investissement Public pour 2024 est fixé à 1 100 milliards FCFA, prioritairement orienté vers les infrastructures routières, l'énergie et le développement rural.",
            "Les crédits alloués au secteur de la santé pour 2024 s'élèvent à 312 milliards FCFA, visant à renforcer le plateau technique hospitalier et les soins de santé primaires.",
            "L'enveloppe budgétaire de l'éducation pour 2024 comprend 428 milliards FCFA dédiés aux salaires des enseignants, aux infrastructures scolaires et aux bourses d'excellence.",
            "Les dépenses de fonctionnement des services de l'État pour 2024 sont limitées à 650 milliards FCFA conformément aux directives de rationalisation budgétaire.",
            "Les grands projets d'infrastructure pour 2024 mobilisent 245 milliards FCFA notamment pour l'autoroute Yaoundé-Douala, le barrage de Nachtigal et les aménagements portuaires.",
            "Le programme de modernisation de l'administration publique pour 2024 prévoit 18 milliards FCFA pour la dématérialisation des services et le déploiement de la e-gouvernance.",
            "Le soutien au secteur agricole pour 2024 comprend 156 milliards FCFA orientés vers la mécanisation, l'irrigation et la sécurité alimentaire.",
            "Le Fonds national de l'emploi et les programmes d'insertion des jeunes disposent de 45 milliards FCFA pour 2024 ciblant 50 000 bénéficiaires.",
            "Les investissements dans le secteur énergétique pour 2024 atteignent 220 milliards FCFA pour l'extension du réseau de transport et la mise en service de centrales thermiques.",
            "Le Fonds de la décentralisation alloue 80 milliards FCFA aux collectivités territoriales décentralisées pour 2024.",
            "Le budget de défense et sécurité pour 2024 est fixé à 380 milliards FCFA pour le maintien de l'ordre, la lutte antiterroriste et la sécurisation des frontières.",
            "Le programme BIP agriculture 2024 finance l'aménagement de 8 000 hectares de terres agricoles et la distribution de 12 000 tonnes d'engrais subventionnés.",
            "Le programme BIP routes 2024 prévoit la construction et la réhabilitation de 1 200 km de routes bitumées, 3 400 km de routes en terre et 45 ouvrages d'art.",
            "Le BIP énergie 2024 couvre l'électrification de 850 villages, l'installation de 45 centrales solaires hybrides et le raccordement de 120 000 nouveaux abonnés.",
            "Le BIP santé 2024 finance la construction de 8 hôpitaux de district, la réhabilitation de 95 centres de santé et l'équipement en imagerie médicale.",
            "Le BIP éducation 2024 construit 1 800 salles de classe, réhabilite 350 établissements, équipe 120 lycées techniques.",
            "Le BIP eau potable 2024 approvisionne 1,2 million de personnes via 180 adductions d'eau et 850 forages équipés de pompes solaires.",
            "Le BIP emploi 2024 finance le Programme d'Appui à la Jeunesse Rurale avec 35 000 bénéficiaires et 450 groupements de producteurs.",
            "Le BIP décentralisation 2024 renforce les capacités de 120 communes et finance 850 micro-projets communaux.",
        ],
        'texte_2025': [
            "Les recettes fiscales globales pour 2025 sont projetées à 3 450 milliards FCFA, en hausse de 9,5% grâce aux réformes fiscales numériques et à l'élargissement de l'assiette fiscale.",
            "Le service de la dette 2025 atteint 920 milliards FCFA, reflétant l'augmentation des emprunts liés au Programme Triennal d'Investissements Publics et aux financements climatiques.",
            "Le Budget d'Investissement Public 2025 s'établit à 1 250 milliards FCFA avec un accent particulier sur la transformation numérique, la résilience climatique et les corridors économiques régionaux.",
            "Le budget santé 2025 est porté à 385 milliards FCFA intégrant un programme national de couverture santé universelle et la construction de 45 nouveaux centres de santé intégrés.",
            "L'éducation bénéficie de 502 milliards FCFA en 2025 avec introduction d'un curriculum numérique, recrutement de 8 000 enseignants et construction de 120 nouvelles salles de classe numériques.",
            "Les dépenses de fonctionnement 2025 sont maintenues à 680 milliards FCFA malgré la pression inflationniste, grâce aux économies réalisées par la dématérialisation administrative.",
            "En 2025, les grands travaux bénéficient de 310 milliards FCFA incluant la deuxième phase de l'autoroute, le pont sur le Wouri et les infrastructures de la CAN 2027.",
            "La stratégie numérique 2025 mobilise 87 milliards FCFA pour le déploiement de la fibre optique nationale, l'interopérabilité des systèmes d'information et la cybersécurité gouvernementale.",
            "L'agriculture reçoit 198 milliards FCFA en 2025 avec un programme d'intensification climatiquement intelligente, des semences améliorées et la réhabilitation de 15 000 hectares irrigués.",
            "Les dispositifs emploi 2025 bénéficient de 78 milliards FCFA avec extension du programme HYSACAM, création de 2 000 entreprises agropastorales et formation professionnelle accélérée.",
            "L'énergie mobilise 275 milliards FCFA en 2025 pour l'interconnexion régionale, les énergies renouvelables solaires et la réhabilitation du réseau de distribution urbain et rural.",
            "La décentralisation bénéficie de 115 milliards FCFA en 2025 marquant une étape décisive vers le transfert effectif de compétences dans les domaines de la santé et de l'éducation.",
            "Les dépenses sécuritaires 2025 atteignent 415 milliards FCFA intégrant la modernisation de l'équipement militaire et le renseignement électronique.",
            "Le BIP agriculture 2025 étend l'irrigation à 12 500 hectares, distribue 18 000 tonnes d'intrants améliorés et développe 6 agropoles régionaux.",
            "Le BIP routes 2025 accélère avec 1 650 km de bitumage, priorité aux axes de désenclavement du Grand Nord et à la connexion des zones de production agropastorale.",
            "Le BIP énergie 2025 vise l'électrification de 1 100 villages additionnels via le solaire, l'extension haute tension et la construction du barrage de Menchum.",
            "Le BIP santé 2025 prévoit 15 nouveaux hôpitaux, 200 postes de santé de base, un CHU à Bamenda et l'introduction du dossier médical électronique national.",
            "Le BIP éducation 2025 programme 2 400 nouvelles salles de classe numériques, 3 campus universitaires à vocation scientifique et 45 centres de formation aux métiers du numérique.",
            "Le BIP eau 2025 étend l'accès à l'eau potable à 1,8 million de personnes additionnelles et déploie un système de surveillance qualité en temps réel.",
            "Le BIP emploi 2025 double l'enveloppe pour atteindre 60 000 jeunes, intègre le programme GROW numérique et crée 15 incubateurs régionaux.",
            "Le BIP décentralisation 2025 accélère le transfert de compétences à 360 communes et introduit le budget participatif numérique.",
        ],
        'montant_2024': [
            3150, 780, 1100, 312, 428, 650, 245, 18,
            156,  45,  220,  80,  380, 120, 245, 95,
            102,  50,  42,   35,  72,
        ],
        'montant_2025': [
            3450, 920, 1250, 385, 502, 680, 310, 87,
            198,  78,  275,  115, 415, 145, 320, 210,
            168,  142, 89,   92,  98,
        ],
        'montant_BIP': [
            0, 0, 1250, 385, 502, 0, 310, 87,
            198, 92, 275, 115, 0, 145, 320, 210,
            168, 142, 89, 92, 98,
        ],
    }
    
    df = pd.DataFrame(data)
    
    # Prétraitement
    df['texte_2024_clean'] = df['texte_2024'].apply(nettoyer_texte)
    df['texte_2025_clean'] = df['texte_2025'].apply(nettoyer_texte)
    df['type_section']     = df['texte_2024_clean'].apply(classifier_section)
    df['poids_section']    = df['type_section'].map(POIDS_SECTIONS).fillna(0.8)
    
    # Variation budgétaire
    df['variation_budget'] = (
        (df['montant_2025'] - df['montant_2024']) /
        df['montant_2024'].replace(0, np.nan) * 100
    ).round(1)
    
    return df


def preparer_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le prétraitement standard à un DataFrame issu des PDF réels.
    
    Args:
        df_raw: DataFrame brut issu de l'extraction PDF
    Returns:
        DataFrame avec colonnes nettoyées et enrichies
    """
    df = df_raw.copy()
    
    # Nettoyage textes
    for col in ['texte_2024', 'texte_2025', 'texte']:
        if col in df.columns:
            clean_col = col + '_clean' if col != 'texte' else 'texte_clean'
            df[clean_col] = df[col].fillna('').apply(nettoyer_texte)
    
    # Colonnes texte nettoyé unifiées
    if 'texte_2024_clean' not in df.columns and 'texte_clean' in df.columns:
        df['texte_2024_clean'] = df['texte_clean']
    if 'texte_2025_clean' not in df.columns and 'texte_clean' in df.columns:
        df['texte_2025_clean'] = df['texte_clean']
    
    # Extraction montants si absents
    if 'montant_2024' not in df.columns and 'texte_2024' in df.columns:
        df['montant_2024'] = df['texte_2024'].apply(extraire_montant_simple)
    if 'montant_2025' not in df.columns and 'texte_2025' in df.columns:
        df['montant_2025'] = df['texte_2025'].apply(extraire_montant_simple)
    
    # Variation budgétaire
    if 'montant_2024' in df.columns and 'montant_2025' in df.columns:
        df['variation_budget'] = (
            (df['montant_2025'] - df['montant_2024']) /
            df['montant_2024'].replace(0, np.nan) * 100
        ).round(1)
    
    # Classification sections
    col_txt = next((c for c in ['texte_2024_clean', 'texte_clean', 'texte_2024'] if c in df.columns), None)
    if col_txt:
        df['type_section']  = df[col_txt].apply(classifier_section)
        df['poids_section'] = df['type_section'].map(POIDS_SECTIONS).fillna(0.8)
    
    return df

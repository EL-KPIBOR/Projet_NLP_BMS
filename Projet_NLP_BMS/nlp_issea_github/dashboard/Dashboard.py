"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BAROMÈTRE DE GLISSEMENT SÉMANTIQUE                                       ║
║   Dashboard NLP · Loi de Finances Cameroun 2024 → 2025/2026               ║
║   ISSEA · ISE3-DS · 2025-2026                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Lancer :
    streamlit run dashboard/app.py
    # ou depuis la racine :
    streamlit run app.py

Dépendances :
    pip install streamlit plotly pandas numpy scikit-learn scipy statsmodels
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys, os
from pathlib import Path

# Ajouter le dossier parent au path pour importer src/
ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "dashboard" else Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import cityblock
from scipy.special import softmax
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Baromètre Sémantique ISSEA",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Pipeline NLP · ISSEA ISE3-DS · Audit Loi de Finances Cameroun · 2025-2026"
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Design analytique sombre, typographie IBM Plex
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
h1,h2,h3 { font-family:'IBM Plex Sans',sans-serif; font-weight:700; }

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#161b22 0%,#0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] label { color:#8b949e !important; font-size:0.82rem; }

/* ── Header ──────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg,#161b22 0%,#1c2128 60%,#0d1117 100%);
    border:1px solid #21262d; border-left:4px solid #1f6feb;
    border-radius:8px; padding:20px 28px; margin-bottom:24px;
}
.main-header h1 { color:#e6edf3; margin:0; font-size:1.6rem; letter-spacing:-0.3px; }
.main-header .sub { color:#8b949e; font-family:'IBM Plex Mono',monospace;
                    font-size:0.78rem; margin-top:6px; }

/* ── KPI Cards ───────────────────────────────────────── */
.kpi { background:#161b22; border:1px solid #21262d; border-radius:8px;
        padding:18px 14px; text-align:center; }
.kpi-val { font-size:2.2rem; font-weight:700;
           font-family:'IBM Plex Mono',monospace; line-height:1; }
.kpi-lbl { font-size:0.72rem; color:#8b949e; text-transform:uppercase;
           letter-spacing:1px; margin-top:7px; }
.kpi-sub { font-size:0.75rem; color:#6e7681; margin-top:3px; }

/* ── Alert box ───────────────────────────────────────── */
.alert { background:#1c0a0a; border:1px solid #da3633;
          border-left:4px solid #da3633; border-radius:6px;
          padding:10px 16px; margin:10px 0; font-size:0.875rem; }

/* ── Section titles ──────────────────────────────────── */
.stitle { font-size:0.7rem; font-weight:700; letter-spacing:2px;
          text-transform:uppercase; color:#8b949e;
          border-bottom:1px solid #21262d; padding-bottom:7px; margin:20px 0 14px; }

/* ── Tabs ────────────────────────────────────────────── */
[data-baseweb="tab-list"] { background:#161b22; border-radius:8px;
                             gap:4px; padding:4px; }
[data-baseweb="tab"]      { color:#8b949e; border-radius:6px; }
[aria-selected="true"]    { background:#1f6feb !important; color:#e6edf3 !important; }

/* ── Mono ────────────────────────────────────────────── */
.mono { font-family:'IBM Plex Mono',monospace; font-size:0.83rem; }

/* ── Rupture list items ──────────────────────────────── */
.rupt-item { display:flex; justify-content:space-between; align-items:center;
              background:#161b22; border:1px solid #21262d; border-radius:6px;
              padding:7px 12px; margin-bottom:5px; }

/* ── Keyword chips ───────────────────────────────────── */
.chip-up   { background:#0d2c1a; border:1px solid #2ea043; border-radius:5px;
              padding:5px 11px; margin-bottom:4px; display:flex;
              justify-content:space-between; }
.chip-down { background:#2d0f0f; border:1px solid #da3633; border-radius:5px;
              padding:5px 11px; margin-bottom:4px; display:flex;
              justify-content:space-between; }

/* ── Footer ──────────────────────────────────────────── */
.footer { text-align:center; padding:28px 0 12px; color:#484f58;
           font-size:0.75rem; font-family:'IBM Plex Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES (miroir de src/config.py — standalone sans import)
# ─────────────────────────────────────────────────────────────────────────────
CODES_PILIERS = ['P1', 'P2', 'P3', 'P4']
LABELS_PILIERS = {
    'P1': "Transformation structurelle de l'économie",
    'P2': "Développement du capital humain",
    'P3': "Promotion de l'emploi et insertion économique",
    'P4': "Gouvernance et décentralisation",
}
LABELS_COURTS = {'P1':'Transf. structurelle','P2':'Capital humain','P3':'Emploi/Insertion','P4':'Gouvernance'}
COLORS_PILIERS  = {'P1':'#3fb950','P2':'#1f6feb','P3':'#f0883e','P4':'#a371f7'}
COLORS_AUDIT    = {'rupture_critique':'#f85149','rupture':'#e74c3c','attention':'#f0883e','stable':'#3fb950'}
STOPWORDS_BUDGET = [
    'le','la','les','de','du','des','un','une','et','ou','pour','dans',
    'article','chapitre','alinea','paragraphe','annexe','loi','decret',
    'milliards','millions','fcfa','francs','cfa','exercice','annee',
    'est','sont','aux','par','sur','en','au','ce','se','il','ils',
]
POIDS_SECTIONS = {'investissement':1.5,'recettes':1.3,'depenses':1.2,
                   'fiscal':1.4,'gouvernance':1.0,'autre':0.8}
HYPOTHESES_SND30 = {
    'P1': ["transformation structurelle économie","industrialisation développement",
           "infrastructure énergie transport","agriculture modernisation productivité",
           "routes autoroutes aménagement","électricité barrage centrale",
           "zones économiques spéciales","compétitivité entreprises secteur privé",
           "production manufacturière industrie","investissement infrastructure productive",
           "mines exploitation ressources","numérique technologie innovation",
           "chaînes valeur ajoutée","diversification économique exportation"],
    'P2': ["capital humain développement","éducation formation enseignement",
           "santé hôpital soins médicaux","protection sociale solidarité",
           "école primaire secondaire université","structures sanitaires plateaux techniques",
           "lutte contre maladies prévention","accès eau potable assainissement",
           "nutrition sécurité alimentaire","jeunesse sport culture",
           "genre équité femmes enfants","couverture maladie universelle",
           "qualifications compétences professionnelles","personnel enseignant formation"],
    'P3': ["emploi insertion économique","jeunes diplômés stage formation",
           "création entreprises entrepreneuriat","microfinance crédit PME",
           "artisanat métiers secteur informel","incubateurs startups innovation",
           "travaux haute intensité main œuvre","insertion professionnelle accompagnement",
           "apprentissage qualification métiers","auto-emploi activités génératrices revenus",
           "promotion entrepreneuriat féminin jeunesse","fonds national emploi jeunes",
           "programme emploi rural agricole","transition école vie active"],
    'P4': ["gouvernance décentralisation","administration publique modernisation",
           "collectivités territoriales communes","déconcentration services publics",
           "transparence redevabilité anticorruption","justice état droit institutions",
           "sécurité défense paix","dématérialisation numérique administration",
           "cadastre foncier gestion domaine","finances publiques contrôle audit",
           "réformes institutionnelles efficacité","participation citoyenne démocratie locale",
           "services déconcentrés préfectures sous-préfectures",
           "renforcement capacités agents publics"],
}


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES DE DÉMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Chargement et calcul du pipeline NLP…")
def generer_donnees_demo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Génère les données de démonstration enrichies et exécute
    le pipeline NLP complet (embeddings, distances, classification).
    """
    data = {
        'article': [
            'Article 1','Article 3','Article 10','Article 21','Article 25',
            'Article 26','Article 30','Article 35','Article 42','Article 56',
            'Article 62','Article 70','Article 82','BIP-010101','BIP-020101',
            'BIP-020102','BIP-020201','BIP-020301','BIP-020401','BIP-030101','BIP-040101',
        ],
        'chapitre': [
            'Recettes fiscales','Dette publique','BIP Global','Santé publique',
            'Éducation nationale','Fonctionnement','Grands Travaux',
            'Transformation Numérique','Agriculture & Élevage','Emploi & Insertion',
            'Énergie électrique','Décentralisation','Sécurité & Défense',
            'BIP - Agriculture','BIP - Routes','BIP - Énergie','BIP - Santé',
            'BIP - Éducation','BIP - Eau potable','BIP - Emploi','BIP - Décentralisation',
        ],
        'texte_2024': [
            "Les recettes fiscales 2024 évaluées à 3 150 milliards FCFA comprenant impôts directs, indirects et droits de douane.",
            "Service de la dette publique 2024 estimé à 780 milliards FCFA dont 420 milliards dette extérieure.",
            "Budget d'Investissement Public 2024 fixé à 1 100 milliards FCFA orienté infrastructures routières, énergie, développement rural.",
            "Crédits santé 2024 à 312 milliards FCFA pour renforcer plateau technique hospitalier et soins de santé primaires.",
            "Enveloppe éducation 2024 à 428 milliards FCFA pour salaires enseignants, infrastructures scolaires, bourses d'excellence.",
            "Dépenses de fonctionnement des services de l'État 2024 limitées à 650 milliards FCFA.",
            "Grands travaux 2024 mobilisent 245 milliards FCFA pour autoroute Yaoundé-Douala, barrage Nachtigal, aménagements portuaires.",
            "Modernisation administration publique 2024 : 18 milliards FCFA pour dématérialisation services et déploiement e-gouvernance.",
            "Soutien agriculture 2024 : 156 milliards FCFA pour mécanisation, irrigation et sécurité alimentaire.",
            "Fonds emploi et insertion jeunes 2024 : 45 milliards FCFA ciblant 50 000 bénéficiaires.",
            "Investissements énergétiques 2024 : 220 milliards FCFA pour extension réseau transport et centrales thermiques.",
            "Fonds décentralisation 2024 : 80 milliards FCFA aux collectivités territoriales pour renforcement institutionnel.",
            "Budget défense et sécurité 2024 fixé à 380 milliards FCFA pour maintien de l'ordre et sécurisation des frontières.",
            "BIP agriculture 2024 : aménagement 8 000 hectares terres agricoles et distribution 12 000 tonnes engrais subventionnés.",
            "BIP routes 2024 : construction et réhabilitation 1 200 km routes bitumées, 3 400 km routes en terre, 45 ouvrages d'art.",
            "BIP énergie 2024 : électrification 850 villages, 45 centrales solaires hybrides, 120 000 nouveaux abonnés.",
            "BIP santé 2024 : construction 8 hôpitaux de district, réhabilitation 95 centres de santé, équipement imagerie médicale.",
            "BIP éducation 2024 : construction 1 800 salles de classe, réhabilitation 350 établissements, 120 lycées techniques.",
            "BIP eau potable 2024 : approvisionnement 1,2 million personnes via 180 adductions et 850 forages pompes solaires.",
            "BIP emploi 2024 : Programme Appui Jeunesse Rurale avec 35 000 bénéficiaires, 450 groupements producteurs.",
            "BIP décentralisation 2024 : renforcement capacités 120 communes, 850 micro-projets communaux.",
        ],
        'texte_2025': [
            "Recettes fiscales 2025 projetées à 3 450 milliards FCFA, hausse 9,5% via réformes fiscales numériques et élargissement assiette.",
            "Service de la dette 2025 atteint 920 milliards FCFA reflétant emprunts Programme Triennal Investissements et financements climatiques.",
            "BIP 2025 à 1 250 milliards FCFA accent sur transformation numérique, résilience climatique et corridors économiques régionaux.",
            "Budget santé 2025 à 385 milliards FCFA intégrant couverture santé universelle et 45 nouveaux centres de santé intégrés.",
            "Éducation 2025 : 502 milliards FCFA pour curriculum numérique, recrutement 8 000 enseignants, 120 salles classe numériques.",
            "Fonctionnement 2025 maintenu à 680 milliards FCFA grâce aux économies dématérialisation des procédures administratives.",
            "Grands travaux 2025 : 310 milliards FCFA pour deuxième phase autoroute, pont sur Wouri, infrastructures CAN 2027.",
            "Stratégie numérique 2025 : 87 milliards FCFA pour fibre optique nationale, interopérabilité systèmes information, cybersécurité gouvernementale.",
            "Agriculture 2025 : 198 milliards FCFA pour intensification climatiquement intelligente, semences améliorées, 15 000 hectares irrigués.",
            "Dispositifs emploi 2025 : 78 milliards FCFA pour HYSACAM étendu, 2 000 entreprises agropastorales, formation professionnelle accélérée.",
            "Énergie 2025 : 275 milliards FCFA pour interconnexion régionale, énergies renouvelables solaires, réhabilitation réseau distribution.",
            "Décentralisation 2025 : 115 milliards FCFA pour transfert effectif compétences santé, éducation et infrastructures locales.",
            "Dépenses sécuritaires 2025 : 415 milliards FCFA pour modernisation équipement militaire et renseignement électronique.",
            "BIP agriculture 2025 : irrigation 12 500 hectares, 18 000 tonnes intrants améliorés, 6 agropoles régionaux avec IRAD.",
            "BIP routes 2025 : 1 650 km bitumage, priorité désenclavement Grand Nord et connexion zones production agropastorale.",
            "BIP énergie 2025 : électrification 1 100 villages additionnels solaire, extension haute tension, barrage de Menchum.",
            "BIP santé 2025 : 15 nouveaux hôpitaux, 200 postes de santé, CHU à Bamenda, dossier médical électronique national.",
            "BIP éducation 2025 : 2 400 salles classe numériques, 3 campus universitaires scientifiques, 45 centres formation numérique.",
            "BIP eau 2025 : accès eau potable 1,8 million personnes additionnelles, surveillance qualité temps réel.",
            "BIP emploi 2025 : 60 000 jeunes ciblés, programme GROW numérique, 15 incubateurs régionaux, 300 startups agritech.",
            "BIP décentralisation 2025 : transfert compétences 360 communes, plans développement communaux, budget participatif numérique.",
        ],
        'montant_2024': [3150,780,1100,312,428,650,245,18,156,45,220,80,380,120,245,95,102,50,42,35,72],
        'montant_2025': [3450,920,1250,385,502,680,310,87,198,78,275,115,415,145,320,210,168,142,89,92,98],
        'montant_BIP':  [0,0,1250,385,502,0,310,87,198,92,275,115,0,145,320,210,168,142,89,92,98],
        'pilier_predit':['P4','P4','P1','P2','P2','P4','P1','P4','P1','P3','P1','P4','P4',
                          'P1','P1','P1','P2','P2','P2','P3','P4'],
    }

    df = pd.DataFrame(data)
    df['type_section'] = df['chapitre'].apply(_classifier_section)
    df['poids_section'] = df['type_section'].map(POIDS_SECTIONS).fillna(0.8)
    df['variation_budget'] = (
        (df['montant_2025'] - df['montant_2024']) /
        df['montant_2024'].replace(0, np.nan) * 100
    ).round(1)

    # ── TF-IDF Embeddings ─────────────────────────────────────────────────
    texts_24 = df['texte_2024'].fillna("").tolist()
    texts_25 = df['texte_2025'].fillna("").tolist()
    all_texts = texts_24 + texts_25

    vec = TfidfVectorizer(max_features=1000, stop_words=STOPWORDS_BUDGET,
                           ngram_range=(1,2), sublinear_tf=True)
    vec.fit(all_texts)
    X24 = vec.transform(texts_24).toarray()
    X25 = vec.transform(texts_25).toarray()

    n_comp = min(80, X24.shape[0]-1, X24.shape[1])
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(np.vstack([X24, X25]))
    E24 = svd.transform(X24)
    E25 = svd.transform(X25)

    n = len(df)
    cosinus_vals = np.array([
        float(max(0.0, cosine_similarity(E24[i:i+1], E25[i:i+1])[0,0]))
        for i in range(n)
    ])
    eucl_vals = np.array([
        float(euclidean_distances(E24[i:i+1], E25[i:i+1])[0,0])
        for i in range(n)
    ])
    manh_vals = np.array([float(cityblock(E24[i], E25[i])) for i in range(n)])

    # ── Topic Drift LDA ───────────────────────────────────────────────────
    try:
        lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=30)
        Xf = vec.transform(all_texts)
        lda.fit(Xf)
        t24_ = lda.transform(vec.transform(texts_24))
        t25_ = lda.transform(vec.transform(texts_25))
        drift = np.array([np.linalg.norm(t24_[i]-t25_[i]) for i in range(n)])
        drift = (drift - drift.min()) / (drift.max() - drift.min() + 1e-10)
    except Exception:
        drift = 1.0 - cosinus_vals

    df['cosinus']     = cosinus_vals
    df['euclidienne'] = eucl_vals
    df['manhattan']   = manh_vals
    df['topic_drift'] = drift

    # ── Score gravité ─────────────────────────────────────────────────────
    cos_inv   = 1.0 - cosinus_vals
    drift_n   = (drift - drift.min()) / (drift.max() - drift.min() + 1e-10)
    var_abs   = np.clip(np.abs(df['variation_budget'].values)/100.0, 0, 1)
    poids     = df['poids_section'].values
    df['score_gravite'] = np.clip(
        (0.45*cos_inv + 0.30*drift_n + 0.25*var_abs) * poids * 100, 0, 100
    ).round(1)

    # ── Isolation Forest ─────────────────────────────────────────────────
    X_iso = np.column_stack([cosinus_vals, eucl_vals, drift])
    X_iso = (X_iso - X_iso.mean(0)) / (X_iso.std(0) + 1e-10)
    iso = IsolationForest(contamination=0.15, random_state=42)
    anomalies = iso.fit_predict(X_iso)
    df['anomalie'] = (anomalies == -1)

    # ── Classification ruptures ───────────────────────────────────────────
    q25 = float(np.percentile(cosinus_vals, 25))
    q75 = float(np.percentile(cosinus_vals, 75))

    def categoriser(row):
        c, a = row['cosinus'], row['anomalie']
        if c < q25 or (a and c < 0.55): return 'rupture_critique'
        if c < 0.70 or a:               return 'rupture'
        if c < q75:                     return 'attention'
        return 'stable'

    df['categorie'] = df.apply(categoriser, axis=1)

    # ── Classification EH-NLI (confiance) ────────────────────────────────
    df['confiance'] = (df['cosinus'] * 0.50 + (1-df['topic_drift']) * 0.50).clip(0,1).round(3)
    df['certitude'] = df['confiance'].apply(
        lambda x: 'certain' if x > 0.70 else ('incertain' if x < 0.45 else 'modéré')
    )

    # Scores piliers simulés
    np.random.seed(99)
    for p in CODES_PILIERS:
        base = (df['pilier_predit'] == p).astype(float) * 0.65
        df[f'score_{p}'] = (base + np.random.uniform(0.05, 0.25, n)).clip(0,1).round(3)

    # Label pilier
    df['label_pilier'] = df['pilier_predit'].map(LABELS_PILIERS)

    # ── Delta TF-IDF ──────────────────────────────────────────────────────
    vec2 = TfidfVectorizer(max_features=400, stop_words=STOPWORDS_BUDGET, ngram_range=(1,2))
    X24b = vec2.fit_transform(texts_24).toarray()
    X25b = vec2.transform(texts_25).toarray()
    delta_tf = X25b.mean(0) - X24b.mean(0)
    feat_names = np.array(vec2.get_feature_names_out())
    top_idx = np.argsort(np.abs(delta_tf))[-25:][::-1]
    df_kw = pd.DataFrame({
        'mot':   feat_names[top_idx],
        'delta': delta_tf[top_idx].round(5),
        'sens':  ['↑ Émergent 2025' if d > 0 else '↓ Déclinant 2025' for d in delta_tf[top_idx]]
    })

    return df, df_kw


def _classifier_section(chapitre: str) -> str:
    c = str(chapitre).lower()
    if any(k in c for k in ['bip','investissement','infrastructure','routes','énergie','eau','santé','éducation','agriculture']): return 'investissement'
    if any(k in c for k in ['recettes','fiscal','impôt','taxe']): return 'recettes'
    if any(k in c for k in ['dette','fonctionnement','dépenses','salaire']): return 'depenses'
    if any(k in c for k in ['gouvernance','décentralisation','sécurité','justice']): return 'gouvernance'
    return 'autre'


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT CSV/XLSX
# ─────────────────────────────────────────────────────────────────────────────
def charger_csv(uploaded) -> pd.DataFrame | None:
    try:
        if uploaded.name.endswith('.csv'):   return pd.read_csv(uploaded, encoding='utf-8')
        elif uploaded.name.endswith('.xlsx'): return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Erreur lecture : {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 6px'>
        <span style='font-size:2.4rem'>🇨🇲</span>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#8b949e;margin-top:3px'>
            ISSEA · ISE3-DS · 2025-2026
        </div>
        <div style='color:#e6edf3;font-weight:700;font-size:0.95rem;margin-top:2px'>
            Baromètre Sémantique
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="stitle">📂 Source de données</p>', unsafe_allow_html=True)
    source = st.radio("", ["Données de démonstration", "Importer CSV / Excel"],
                       label_visibility="collapsed")

    df_uploaded = None
    if source == "Importer CSV / Excel":
        uploaded = st.file_uploader(
            "audit_resultats_complets.csv", type=["csv","xlsx"],
            help="Fichier exporté depuis le notebook (Section 11)"
        )
        if uploaded:
            df_uploaded = charger_csv(uploaded)
            if df_uploaded is not None:
                st.success(f"✅ {len(df_uploaded)} articles chargés")

    st.divider()
    st.markdown('<p class="stitle">⚙️ Seuils adaptatifs</p>', unsafe_allow_html=True)
    seuil_rupt = st.slider("Seuil rupture (cosinus)", 0.30, 0.85, 0.60, 0.01,
                            help="Articles sous ce seuil → rupture sémantique")
    seuil_att  = st.slider("Seuil attention (cosinus)", 0.55, 0.95, 0.75, 0.01,
                            help="Articles entre seuil rupture et ce seuil → attention")

    st.divider()
    st.markdown('<p class="stitle">🔍 Filtres</p>', unsafe_allow_html=True)
    anomalies_only = st.checkbox("Anomalies seulement (Isolation Forest)", False)
    sel_piliers    = st.multiselect(
        "Piliers SND30", CODES_PILIERS, default=CODES_PILIERS,
        format_func=lambda p: f"{p} — {LABELS_COURTS[p]}"
    )
    sel_sections   = st.multiselect(
        "Sections", ['investissement','depenses','recettes','gouvernance','fiscal','autre'],
        default=['investissement','depenses','recettes','gouvernance','fiscal','autre']
    )

    st.divider()
    st.caption("v2.0 — ISSEA · NLP Pipeline · 2025")


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
df_demo, df_kw = generer_donnees_demo()

if source == "Importer CSV / Excel" and df_uploaded is not None:
    df_raw = df_uploaded.copy()
    for c in ['cosinus','euclidienne','topic_drift','score_gravite','categorie','anomalie','confiance','pilier_predit']:
        if c not in df_raw.columns:
            df_raw[c] = df_demo[c].iloc[0] if c in df_demo else (0 if c != 'categorie' else 'stable')
else:
    df_raw = df_demo.copy()

# Recalcul catégorie avec seuils courants
def recalc_cat(row):
    c, a = row['cosinus'], row.get('anomalie', False)
    if c < seuil_rupt or (a and c < seuil_rupt + 0.05): return 'rupture_critique'
    if c < (seuil_rupt + seuil_att)/2 or a:             return 'rupture'
    if c < seuil_att:                                   return 'attention'
    return 'stable'

df_raw['categorie'] = df_raw.apply(recalc_cat, axis=1)

# Application filtres
df = df_raw.copy()
if anomalies_only and 'anomalie' in df.columns: df = df[df['anomalie']==True]
if sel_piliers and 'pilier_predit' in df.columns: df = df[df['pilier_predit'].isin(sel_piliers)]
if sel_sections and 'type_section' in df.columns: df = df[df['type_section'].isin(sel_sections)]

if len(df) == 0:
    st.warning("⚠️ Aucun article ne correspond aux filtres. Ajustez vos paramètres.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
sim_moy         = float(df['cosinus'].mean())
glissement_pct  = (1 - sim_moy) * 100
n_crit          = int((df['categorie']=='rupture_critique').sum())
n_rupt          = int((df['categorie']=='rupture').sum())
n_att           = int((df['categorie']=='attention').sum())
n_stab          = int((df['categorie']=='stable').sum())
n_anom          = int(df['anomalie'].sum()) if 'anomalie' in df.columns else 0
gravite_moy     = float(df['score_gravite'].mean())
drift_moy       = float(df['topic_drift'].mean())


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>📊 Baromètre de Glissement Sémantique — Loi de Finances</h1>
    <div class="sub">
        Audit NLP · Cameroun 2024 → 2025/2026 · ISSEA ISE3-DS ·
        {len(df)} articles analysés · SND30
    </div>
</div>
""", unsafe_allow_html=True)

if n_crit > 0:
    arts_crit = df[df['categorie']=='rupture_critique']['article'].tolist()
    st.markdown(f"""
    <div class="alert">
        🚨 <strong>{n_crit} rupture(s) critique(s)</strong> — 
        {', '.join(arts_crit[:6])}{'…' if len(arts_crit)>6 else ''}
        &nbsp;|&nbsp; Revue manuelle recommandée
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
def kpi_color_sim(v):
    return "#3fb950" if v>0.75 else ("#f0883e" if v>0.55 else "#f85149")
def kpi_color_pct(v):
    return "#f85149" if v>50 else ("#f0883e" if v>25 else "#3fb950")

k1,k2,k3,k4,k5,k6 = st.columns(6)
for col, val, lbl, sub, color in [
    (k1, f"{sim_moy:.3f}",       "Similarité cosinus",    "moyenne globale",             kpi_color_sim(sim_moy)),
    (k2, f"{glissement_pct:.1f}%","Glissement sémantique", "1 − cosinus moyen",           kpi_color_pct(glissement_pct)),
    (k3, f"{n_crit+n_rupt}",     "Ruptures totales",       f"{n_crit} critiques · {n_rupt} majeures", "#f85149" if n_crit>0 else "#f0883e"),
    (k4, f"{gravite_moy:.1f}",   "Gravité composite",      "score /100",                  "#f0883e"),
    (k5, f"{drift_moy:.3f}",     "Topic drift LDA",        "dérive thématique moy.",      "#a371f7"),
    (k6, f"{len(df)}",           "Articles analysés",      f"{n_anom} anomalies IF",      "#58a6ff"),
]:
    with col:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-val" style="color:{color}">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌡️ Baromètre",
    "📈 Distributions",
    "🗂️ Vue Détaillée",
    "💰 Piliers SND30",
    "🔑 Delta TF-IDF",
    "🕸️ Espace 3D",
    "📊 Statistiques",
    "📋 Données Brutes",
])

PLOTLY_BG  = '#0d1117'
PLOTLY_PLT = '#161b22'
PLOTLY_GRID= '#21262d'
FONT_COLOR = '#8b949e'
FONT_TITLE = '#e6edf3'


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BAROMÈTRE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    cg, cd = st.columns([1.3, 1])

    with cg:
        st.markdown('<p class="stitle">🌡️ Baromètre de Glissement Global</p>', unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=glissement_pct,
            number={'suffix':'%','font':{'size':54,'color':FONT_TITLE,'family':'IBM Plex Mono'}},
            delta={'reference':25,'increasing':{'color':'#f85149'},'decreasing':{'color':'#3fb950'},
                   'font':{'size':16}},
            title={'text':"Glissement Sémantique<br><span style='font-size:12px;color:#8b949e'>2024 → 2025/2026 · Similarité cosinus inverse</span>",
                   'font':{'size':16,'color':FONT_TITLE,'family':'IBM Plex Sans'}},
            gauge={
                'axis':{'range':[0,100],'tickwidth':1,'tickcolor':PLOTLY_GRID,
                        'tickfont':{'color':FONT_COLOR,'size':11}},
                'bar': {'color':'#1f6feb','thickness':0.2},
                'bgcolor': PLOTLY_BG,
                'borderwidth': 0,
                'steps':[
                    {'range':[0, 25],'color':'#0d2c1a'},
                    {'range':[25,50],'color':'#2e2000'},
                    {'range':[50,75],'color':'#2d0f0f'},
                    {'range':[75,100],'color':'#1c0505'},
                ],
                'threshold':{'line':{'color':'#f0883e','width':3},'thickness':0.85,'value':glissement_pct}
            }
        ))
        fig_g.update_layout(height=370, paper_bgcolor=PLOTLY_BG,
                             margin=dict(t=60,b=20,l=40,r=40))
        st.plotly_chart(fig_g, use_container_width=True)

        # Verdict
        if glissement_pct < 25:   niv,col,txt = "STABLE","#3fb950","Continuité budgétaire marquée. Vocabulaire cohérent entre les deux exercices."
        elif glissement_pct < 50: niv,col,txt = "ATTENTION","#f0883e","Évolution sémantique modérée. Ajustements de priorités observables."
        elif glissement_pct < 75: niv,col,txt = "RUPTURE","#f85149","Changement sémantique significatif. Réorientation partielle des priorités."
        else:                     niv,col,txt = "CRITIQUE","#da3633","Rupture sémantique majeure. Révision profonde des priorités budgétaires."

        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #21262d;border-left:4px solid {col};
                    border-radius:8px;padding:14px 18px;margin-top:6px;">
            <span style="color:{col};font-weight:700;font-size:1rem">⬤ Niveau : {niv}</span>
            <p style="color:#8b949e;margin:7px 0 0;font-size:0.875rem">{txt}</p>
        </div>
        """, unsafe_allow_html=True)

    with cd:
        st.markdown('<p class="stitle">📊 Répartition des Articles</p>', unsafe_allow_html=True)
        fig_d = go.Figure(go.Pie(
            labels=['Rupture Critique','Rupture','Attention','Stable'],
            values=[n_crit, n_rupt, n_att, n_stab],
            marker_colors=['#f85149','#e74c3c','#f0883e','#3fb950'],
            hole=0.55, textinfo='label+percent', textposition='outside',
            textfont=dict(size=10,color=FONT_TITLE,family='IBM Plex Sans'),
            hovertemplate='<b>%{label}</b><br>%{value} articles (%{percent})<extra></extra>',
        ))
        fig_d.add_annotation(text=f"<b>{len(df)}</b><br><span style='font-size:10px'>articles</span>",
                              x=0.5,y=0.5,font_size=18,font_color=FONT_TITLE,showarrow=False)
        fig_d.update_layout(height=300, paper_bgcolor=PLOTLY_BG, showlegend=False,
                             margin=dict(t=20,b=30,l=60,r=60))
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown('<p class="stitle">🔴 Top Ruptures — Score Gravité</p>', unsafe_allow_html=True)
        df_top = df[df['categorie'].isin(['rupture_critique','rupture'])]\
                   .sort_values('score_gravite', ascending=False).head(6)
        for _, row in df_top.iterrows():
            c = COLORS_AUDIT.get(row['categorie'],'#8b949e')
            st.markdown(f"""
            <div class="rupt-item">
                <div>
                    <span class="mono" style="color:#58a6ff">{row['article']}</span>
                    <span style="color:#6e7681;font-size:0.76rem;margin-left:8px">
                        {str(row.get('chapitre',''))[:28]}
                    </span>
                </div>
                <div style="display:flex;gap:8px;align-items:center">
                    <span class="mono" style="color:{FONT_COLOR};font-size:0.8rem">
                        cos={row['cosinus']:.3f}
                    </span>
                    <span style="background:{c}22;color:{c};border:1px solid {c};
                                 border-radius:4px;padding:2px 7px;font-size:0.72rem;font-weight:700">
                        {row['categorie'].replace('_',' ').upper()}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="stitle">📈 Distributions des Métriques de Distance</p>', unsafe_allow_html=True)

    fig_dist = make_subplots(rows=2, cols=2,
        subplot_titles=["Similarité Cosinus","Topic Drift LDA",
                         "Distances Normalisées (Box-plot)","Score Gravité par Article"],
        specs=[[{'type':'histogram'},{'type':'histogram'}],
               [{'type':'box'},      {'type':'bar'}]],
        vertical_spacing=0.14, horizontal_spacing=0.10)

    fig_dist.add_trace(go.Histogram(x=df['cosinus'],name='Cosinus',
        marker_color='#1f6feb',nbinsx=12,opacity=0.85), row=1,col=1)
    fig_dist.add_vline(x=seuil_rupt, line_dash='dash',line_color='#f85149',
                       annotation_text=f'Rupture ({seuil_rupt})', row=1,col=1)
    fig_dist.add_vline(x=seuil_att,  line_dash='dot', line_color='#f0883e',
                       annotation_text=f'Attention ({seuil_att})', row=1,col=1)

    fig_dist.add_trace(go.Histogram(x=df['topic_drift'],name='Topic Drift',
        marker_color='#a371f7',nbinsx=12,opacity=0.85), row=1,col=2)

    for col_n, color in [('cosinus','#1f6feb'),('euclidienne','#f0883e'),
                          ('manhattan','#3fb950'),('topic_drift','#a371f7')]:
        if col_n in df.columns:
            v = df[col_n].values
            vn = (v-v.min())/(v.max()-v.min()+1e-10)
            fig_dist.add_trace(go.Box(y=vn, name=col_n.capitalize(),
                marker_color=color, boxmean='sd', showlegend=False), row=2,col=1)

    df_srt = df.sort_values('score_gravite',ascending=False).head(15)
    fig_dist.add_trace(go.Bar(
        x=df_srt['article'], y=df_srt['score_gravite'],
        marker_color=[COLORS_AUDIT.get(c,'gray') for c in df_srt['categorie']],
        hovertemplate='%{x}<br>Gravité: %{y:.1f}<extra></extra>', showlegend=False
    ), row=2,col=2)

    fig_dist.update_layout(height=660, paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        font_color=FONT_COLOR, font_size=11,
        legend=dict(bgcolor=PLOTLY_PLT,bordercolor=PLOTLY_GRID,borderwidth=1))
    for a in fig_dist['layout']['annotations']:
        a['font'] = dict(color=FONT_TITLE, size=12)
    fig_dist.update_xaxes(showgrid=True,gridcolor=PLOTLY_GRID,zeroline=False)
    fig_dist.update_yaxes(showgrid=True,gridcolor=PLOTLY_GRID,zeroline=False)
    fig_dist.update_xaxes(tickangle=-45, row=2, col=2, tickfont_size=9)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown('<p class="stitle">🔗 Matrice de Corrélation des Métriques NLP</p>', unsafe_allow_html=True)
    corr_cols = [c for c in ['cosinus','euclidienne','manhattan','topic_drift',
                               'score_gravite','variation_budget'] if c in df.columns]
    fig_corr = px.imshow(df[corr_cols].corr().round(3), text_auto=True, aspect="auto",
        color_continuous_scale=[[0,'#c0392b'],[0.5,'#1c2128'],[1,'#2ea043']],
        zmin=-1, zmax=1)
    fig_corr.update_layout(height=360, paper_bgcolor=PLOTLY_BG, font_color=FONT_TITLE,
        font_size=11, margin=dict(t=20,b=20),
        coloraxis_colorbar=dict(tickfont_color=FONT_COLOR))
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VUE DÉTAILLÉE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="stitle">🗂️ Topomap Sémantique — Cosinus × Gravité</p>', unsafe_allow_html=True)

    fig_sc = px.scatter(df, x='cosinus', y='score_gravite', color='categorie',
        color_discrete_map=COLORS_AUDIT, size='topic_drift', size_max=18,
        text='article',
        hover_data={'cosinus':':.3f','score_gravite':':.1f',
                    'topic_drift':':.3f','categorie':True,'chapitre':True},
        title='Articles budgétaires : Stabilité Sémantique × Score de Gravité')
    fig_sc.update_traces(textposition='top center', textfont_size=8, textfont_color=FONT_COLOR)
    fig_sc.add_vline(x=seuil_rupt, line_dash='dash', line_color='#f85149', opacity=0.5)
    fig_sc.add_vline(x=seuil_att,  line_dash='dot',  line_color='#f0883e', opacity=0.5)
    fig_sc.update_layout(height=480, paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_PLT,
        font_color=FONT_COLOR, title_font_color=FONT_TITLE,
        legend=dict(bgcolor=PLOTLY_PLT,bordercolor=PLOTLY_GRID,font_color=FONT_TITLE),
        xaxis=dict(showgrid=True,gridcolor=PLOTLY_GRID,title_font_color=FONT_TITLE),
        yaxis=dict(showgrid=True,gridcolor=PLOTLY_GRID,title_font_color=FONT_TITLE))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown('<p class="stitle">📋 Tableau Détaillé (trié par Score Gravité)</p>', unsafe_allow_html=True)
    show_cols = [c for c in ['article','chapitre','type_section','cosinus',
                               'euclidienne','topic_drift','score_gravite',
                               'categorie','anomalie','variation_budget'] if c in df.columns]
    df_disp = df[show_cols].sort_values('score_gravite',ascending=False).reset_index(drop=True)

    def style_df(d):
        cat_map = {'rupture_critique':'background-color:#4d1919;color:#f85149;font-weight:600',
                   'rupture':'background-color:#3d1a1a;color:#e74c3c;font-weight:600',
                   'attention':'background-color:#3d2b00;color:#f0883e;font-weight:600',
                   'stable':'background-color:#0d2c1a;color:#3fb950;font-weight:600'}
        fmt = {c:':.3f' for c in ['cosinus','euclidienne','topic_drift']}
        fmt.update({'score_gravite':':.1f'})
        if 'variation_budget' in d.columns: fmt['variation_budget'] = lambda x: f"{x:+.1f}%"
        s = (d.style
            .applymap(lambda v: cat_map.get(v,''), subset=['categorie'] if 'categorie' in d else [])
            .applymap(lambda v: 'background-color:#2d0f0f;color:#f85149;font-weight:700' if v else '',
                      subset=['anomalie'] if 'anomalie' in d else [])
            .background_gradient(subset=['score_gravite'] if 'score_gravite' in d else [],
                                   cmap='Reds', vmin=0, vmax=100)
            .format({k: (v if callable(v) else '{'+v+'}') for k,v in fmt.items()}, na_rep='—'))
        return s
    st.dataframe(style_df(df_disp), use_container_width=True, height=420)

    st.markdown('<p class="stitle">📉 Profil de Stabilité Cosinus par Article</p>', unsafe_allow_html=True)
    df_tl = df.sort_values('score_gravite', ascending=False).reset_index(drop=True)
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=df_tl['article'], y=df_tl['cosinus'], mode='lines+markers',
        name='Similarité Cosinus',
        line=dict(color='#1f6feb', width=2),
        marker=dict(color=[COLORS_AUDIT.get(c,'#8b949e') for c in df_tl['categorie']],
                    size=9, line=dict(width=1,color=PLOTLY_BG)),
        hovertemplate='%{x}<br>Cosinus: %{y:.3f}<extra></extra>',
    ))
    fig_tl.add_hline(y=seuil_rupt, line_dash='dash', line_color='#f85149', opacity=0.6,
                     annotation_text='Seuil rupture', annotation_font_color='#f85149')
    fig_tl.add_hline(y=seuil_att,  line_dash='dot',  line_color='#f0883e', opacity=0.6,
                     annotation_text='Seuil attention', annotation_font_color='#f0883e')
    fig_tl.update_layout(height=290, paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_PLT,
        font_color=FONT_COLOR, title_font_color=FONT_TITLE,
        title='Articles triés par score de gravité décroissant',
        xaxis=dict(showgrid=True,gridcolor=PLOTLY_GRID,tickangle=-45,tickfont_size=9),
        yaxis=dict(showgrid=True,gridcolor=PLOTLY_GRID,range=[0,1.05],
                   title='Similarité Cosinus',title_font_color=FONT_TITLE),
        legend=dict(bgcolor=PLOTLY_PLT,bordercolor=PLOTLY_GRID,font_color=FONT_TITLE))
    st.plotly_chart(fig_tl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PILIERS SND30
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    col_m_candidates = ['montant_BIP','montant_2025','montant','budget']
    col_m = next((c for c in col_m_candidates if c in df.columns), None)

    if col_m and 'pilier_predit' in df.columns:
        df_p = df.dropna(subset=['pilier_predit',col_m]).copy()
        df_p[col_m] = pd.to_numeric(df_p[col_m], errors='coerce').fillna(0)
        df_p = df_p[df_p[col_m] > 0]

        if len(df_p) > 0:
            agg_p = df_p.groupby('pilier_predit').agg(
                montant    =(col_m,         'sum'),
                n_articles =('article',     'count'),
                gravite_moy=('score_gravite','mean'),
                cosinus_moy=('cosinus',      'mean'),
            ).reset_index()
            agg_p['label']      = agg_p['pilier_predit'].map(LABELS_COURTS)
            agg_p['pct_budget'] = (agg_p['montant']/agg_p['montant'].sum()*100).round(1)

            cp, cb = st.columns([1, 1.2])
            with cp:
                st.markdown('<p class="stitle">🥧 Répartition Budget BIP 2025</p>', unsafe_allow_html=True)
                fig_pie = go.Figure(go.Pie(
                    labels=agg_p['label'], values=agg_p['montant'],
                    marker_colors=[COLORS_PILIERS.get(p,'gray') for p in agg_p['pilier_predit']],
                    hole=0.45, textinfo='label+percent', textposition='outside',
                    textfont=dict(size=10,color=FONT_TITLE),
                    hovertemplate='<b>%{label}</b><br>%{value:.1f} Mds FCFA<br>%{percent}<extra></extra>',
                ))
                fig_pie.update_layout(height=360,paper_bgcolor=PLOTLY_BG,showlegend=False,
                    margin=dict(t=30,b=20,l=60,r=60))
                st.plotly_chart(fig_pie, use_container_width=True)
                st.caption(f"**Budget total BIP : {agg_p['montant'].sum():.1f} Mds FCFA**")

            with cb:
                st.markdown('<p class="stitle">📊 Montants par Pilier (Mds FCFA)</p>', unsafe_allow_html=True)
                fig_bp = go.Figure()
                for _, rp in agg_p.iterrows():
                    fig_bp.add_trace(go.Bar(
                        x=[rp['label']], y=[rp['montant']],
                        marker_color=COLORS_PILIERS.get(rp['pilier_predit'],'gray'),
                        text=f"{rp['montant']:.0f}", textposition='outside',
                        textfont=dict(color=FONT_TITLE,size=11),
                        hovertemplate=(f"<b>{rp['label']}</b><br>{rp['montant']:.1f} Mds FCFA<br>"
                                       f"{rp['n_articles']} articles<br>Gravité moy: {rp['gravite_moy']:.1f}<extra></extra>"),
                        showlegend=False, name=rp['label'],
                    ))
                fig_bp.update_layout(height=360,paper_bgcolor=PLOTLY_BG,plot_bgcolor=PLOTLY_PLT,
                    font_color=FONT_COLOR,
                    xaxis=dict(showgrid=False,tickfont_size=10),
                    yaxis=dict(showgrid=True,gridcolor=PLOTLY_GRID,title='Mds FCFA',
                               title_font_color=FONT_TITLE),
                    margin=dict(t=10,b=20))
                st.plotly_chart(fig_bp, use_container_width=True)

            # Radar
            st.markdown('<p class="stitle">🕸️ Radar : Stabilité & Gravité par Pilier</p>', unsafe_allow_html=True)
            lbls_r = agg_p['label'].tolist() + [agg_p['label'].tolist()[0]]
            grav_r = agg_p['gravite_moy'].tolist() + [agg_p['gravite_moy'].tolist()[0]]
            cos_r  = (agg_p['cosinus_moy']*100).tolist() + [(agg_p['cosinus_moy']*100).tolist()[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=grav_r, theta=lbls_r,
                fill='toself', name='Gravité moy.', line_color='#f85149',
                fillcolor='rgba(248,81,73,0.12)'))
            fig_radar.add_trace(go.Scatterpolar(r=cos_r, theta=lbls_r,
                fill='toself', name='Cosinus moy. ×100', line_color='#3fb950',
                fillcolor='rgba(63,185,80,0.12)'))
            fig_radar.update_layout(
                polar=dict(bgcolor=PLOTLY_PLT,
                    radialaxis=dict(visible=True,range=[0,100],tickfont=dict(color=FONT_COLOR,size=8)),
                    angularaxis=dict(tickfont=dict(color=FONT_TITLE,size=10))),
                showlegend=True,
                legend=dict(bgcolor=PLOTLY_PLT,bordercolor=PLOTLY_GRID,font_color=FONT_TITLE),
                paper_bgcolor=PLOTLY_BG, height=360, margin=dict(t=20,b=20))
            st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.info("Aucune ligne avec montant > 0 après filtrage.")
    else:
        st.warning(f"⚠️ Colonne montant ou pilier_predit manquante. Colonnes disponibles : {list(df.columns)}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DELTA TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="stitle">🔑 Évolution Lexicale — Delta TF-IDF (2025 − 2024)</p>', unsafe_allow_html=True)

    fig_kw = go.Figure(go.Bar(
        x=df_kw['delta'], y=df_kw['mot'], orientation='h',
        marker_color=['#3fb950' if d>0 else '#f85149' for d in df_kw['delta']],
        text=[f"{'+'if d>0 else ''}{d:.4f}" for d in df_kw['delta']],
        textposition='outside', textfont=dict(size=9,color=FONT_TITLE),
        hovertemplate='<b>%{y}</b><br>Delta: %{x:.4f}<extra></extra>',
    ))
    fig_kw.update_layout(height=560, paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_PLT,
        font_color=FONT_COLOR,
        xaxis=dict(showgrid=True,gridcolor=PLOTLY_GRID,title='Delta TF-IDF',
                   title_font_color=FONT_TITLE, zeroline=True, zerolinecolor='#484f58'),
        yaxis=dict(showgrid=False,tickfont=dict(size=11,color=FONT_TITLE)),
        title=dict(text='Mots montants ↑ (vert) et descendants ↓ (rouge)',
                   font_color=FONT_TITLE, font_size=13),
        margin=dict(l=20,r=90,t=50,b=20))
    st.plotly_chart(fig_kw, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="stitle">↑ Vocabulaire Émergent 2025</p>', unsafe_allow_html=True)
        for _, r in df_kw[df_kw['delta']>0].sort_values('delta',ascending=False).head(10).iterrows():
            st.markdown(f"""<div class="chip-up">
                <span class="mono" style="color:#3fb950">{r['mot']}</span>
                <span class="mono" style="color:#6e7681">+{r['delta']:.4f}</span>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<p class="stitle">↓ Vocabulaire Déclinant 2024</p>', unsafe_allow_html=True)
        for _, r in df_kw[df_kw['delta']<0].sort_values('delta').head(10).iterrows():
            st.markdown(f"""<div class="chip-down">
                <span class="mono" style="color:#f85149">{r['mot']}</span>
                <span class="mono" style="color:#6e7681">{r['delta']:.4f}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ESPACE 3D
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="stitle">🕸️ Topographie Sémantique 3D</p>', unsafe_allow_html=True)

    fig_3d = go.Figure(go.Scatter3d(
        x=df['cosinus'], y=df['euclidienne'], z=df['score_gravite'],
        mode='markers+text',
        marker=dict(size=df['score_gravite']/7+4,
                    color=[COLORS_AUDIT.get(c,'#8b949e') for c in df['categorie']],
                    opacity=0.88, line=dict(width=0.5,color=PLOTLY_BG)),
        text=df['article'], textposition='top center',
        textfont=dict(size=8,color=FONT_COLOR),
        hovertemplate='<b>%{text}</b><br>Cosinus: %{x:.3f}<br>Euclidienne: %{y:.3f}<br>Gravité: %{z:.1f}<extra></extra>',
    ))
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='Similarité Cosinus',backgroundcolor=PLOTLY_PLT,
                       gridcolor=PLOTLY_GRID,zerolinecolor=PLOTLY_GRID),
            yaxis=dict(title='Distance Euclidienne',backgroundcolor=PLOTLY_PLT,
                       gridcolor=PLOTLY_GRID,zerolinecolor=PLOTLY_GRID),
            zaxis=dict(title='Score Gravité',backgroundcolor=PLOTLY_PLT,
                       gridcolor=PLOTLY_GRID,zerolinecolor=PLOTLY_GRID),
            bgcolor=PLOTLY_BG,
        ),
        height=640, paper_bgcolor=PLOTLY_BG, font_color=FONT_COLOR,
        title=dict(text='Espace 3D : Cosinus × Euclidienne × Score Gravité (tournez !)',
                   font_color=FONT_TITLE,font_size=13),
        margin=dict(t=55,b=10,l=0,r=0))
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown(f"""
    <div style="background:{PLOTLY_PLT};border:1px solid {PLOTLY_GRID};border-radius:8px;padding:14px 18px;">
        <span style="color:{FONT_COLOR};font-size:0.83rem">
            🔴 <strong style="color:#f85149">Rupture critique</strong> &nbsp;|&nbsp;
            🟠 <strong style="color:#f0883e">Attention</strong> &nbsp;|&nbsp;
            🟢 <strong style="color:#3fb950">Stable</strong> &nbsp;|&nbsp;
            Taille ∝ score de gravité &nbsp;|&nbsp; Faites tourner la vue 360°
        </span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — STATISTIQUES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="stitle">📊 Synthèse Statistique par Pilier SND30</p>', unsafe_allow_html=True)

    if 'pilier_predit' in df.columns:
        agg_stat = df.groupby('pilier_predit').agg(
            n_articles   =('article','count'),
            cosinus_moy  =('cosinus','mean'),
            cosinus_std  =('cosinus','std'),
            gravite_moy  =('score_gravite','mean'),
            drift_moy    =('topic_drift','mean'),
            confiance_moy=('confiance','mean'),
            n_ruptures   =('categorie', lambda x: (x.isin(['rupture','rupture_critique'])).sum()),
        ).reset_index()
        agg_stat['label'] = agg_stat['pilier_predit'].map(LABELS_COURTS)
        agg_stat['pct_ruptures'] = (agg_stat['n_ruptures']/agg_stat['n_articles']*100).round(1)
        agg_stat = agg_stat.round(3)
        st.dataframe(agg_stat, use_container_width=True, height=220)

    # Bootstrap IC95%
    st.markdown('<p class="stitle">📐 Intervalle de Confiance Bootstrap (IC 95%)</p>', unsafe_allow_html=True)
    np.random.seed(42)
    n_boot = 1000
    boot_means = np.array([
        np.random.choice(df['cosinus'].values, size=len(df), replace=True).mean()
        for _ in range(n_boot)
    ])
    ic_low, ic_high = np.percentile(boot_means, [2.5, 97.5])

    fig_boot = go.Figure()
    fig_boot.add_trace(go.Histogram(x=boot_means, nbinsx=40, name='Bootstrap',
        marker_color='#1f6feb', opacity=0.8))
    fig_boot.add_vline(x=sim_moy,   line_color='#f0883e', line_width=2,
                       annotation_text=f'Moyenne obs. ({sim_moy:.3f})',
                       annotation_font_color='#f0883e')
    fig_boot.add_vline(x=ic_low,    line_dash='dash', line_color='#f85149',
                       annotation_text=f'IC low ({ic_low:.3f})',
                       annotation_font_color='#f85149')
    fig_boot.add_vline(x=ic_high,   line_dash='dash', line_color='#3fb950',
                       annotation_text=f'IC high ({ic_high:.3f})',
                       annotation_font_color='#3fb950')
    fig_boot.update_layout(height=280, paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_PLT,
        font_color=FONT_COLOR, title=dict(text=f'Bootstrap IC95% — Similarité cosinus moyenne (n={n_boot} réplications)',
        font_color=FONT_TITLE, font_size=12),
        xaxis=dict(showgrid=True,gridcolor=PLOTLY_GRID),
        yaxis=dict(showgrid=True,gridcolor=PLOTLY_GRID),
        showlegend=False, margin=dict(t=45,b=20))
    st.plotly_chart(fig_boot, use_container_width=True)

    st.markdown(f"""
    <div style="background:{PLOTLY_PLT};border:1px solid {PLOTLY_GRID};border-left:4px solid #1f6feb;
                border-radius:6px;padding:12px 16px;">
        <span class="mono" style="color:{FONT_TITLE}">
            Cosinus moyen = <strong>{sim_moy:.4f}</strong>
            &nbsp;|&nbsp; IC 95% = [{ic_low:.4f} ; {ic_high:.4f}]
            &nbsp;|&nbsp; Glissement = <strong>{glissement_pct:.1f}%</strong>
        </span>
    </div>""", unsafe_allow_html=True)

    # Méthodologie
    with st.expander("ℹ️ Méthodologie NLP complète"):
        st.markdown("""
| Étape | Méthode | Détail |
|-------|---------|--------|
| Vectorisation | TF-IDF | 1000 features, bigrammes, stopwords MINFI |
| Embeddings | TruncatedSVD (LSA) | 80 composantes, variance maximale |
| Embeddings avancés | Sentence-BERT | `paraphrase-multilingual-MiniLM-L12-v2` (si dispo.) |
| Distance sémantique | Similarité Cosinus | Paires (2024, 2025) article par article |
| Distance géométrique | Euclidienne + Manhattan | Distances L2 et L1 |
| Dérive thématique | LDA Topic Drift | 5 topics, norme L2 entre distributions |
| Détection anomalies | Isolation Forest | contamination=15%, features: cos+eucl+drift |
| Classification ruptures | Seuils adaptatifs Q1/Q3 | Percentiles empiriques |
| Score gravité | Composite pondéré | cos(45%) + drift(30%) + budget(25%) × poids section |
| Classification SND30 | Zero-Shot EH-NLI | 56 hypothèses × 4 piliers + Softmax(T=6) |
| Classification SND30 avancée | BART-MNLI | `facebook/bart-large-mnli` (si GPU/internet dispo.) |
| Tests statistiques | ANOVA + Kruskal-Wallis | Différences budgétaires inter-piliers |
| Régression | OLS HC3 | log(BIP) ~ Pilier + Année + Confiance NLP |
| Clustering | OPTICS | Détection articles hors-SND30 |

**Références :** Devlin et al. 2018 (BERT) · Martin et al. 2020 (CamemBERT) · Reimers & Gurevych 2019 (SBERT)
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — DONNÉES BRUTES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<p class="stitle">📋 Recherche & Export</p>', unsafe_allow_html=True)

    cs, cf = st.columns([2, 1])
    with cs:
        query = st.text_input("🔍 Rechercher…", placeholder="article, chapitre, catégorie…")
    with cf:
        fcat  = st.selectbox("Catégorie", ['Toutes','rupture_critique','rupture','attention','stable'])

    df_show = df.copy()
    if query:
        mask = df_show.apply(lambda r: query.lower() in str(r.values).lower(), axis=1)
        df_show = df_show[mask]
    if fcat != 'Toutes':
        df_show = df_show[df_show['categorie'] == fcat]

    st.dataframe(df_show, use_container_width=True, height=440)
    st.caption(f"**{len(df_show)} articles** affichés")

    csv_data = df_show.to_csv(index=False, encoding='utf-8').encode('utf-8')
    st.download_button("⬇️ Exporter CSV filtré", data=csv_data,
                        file_name="audit_semantique_export.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ISSEA · ISE3-DS · Pipeline NLP · Baromètre Glissement Sémantique · Loi de Finances Cameroun 2024→2026
    <br>Python · PyTorch · spaCy · Scikit-learn · Sentence-BERT · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)

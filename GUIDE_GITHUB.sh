# 🚀 GUIDE COMPLET — Mise en place du dépôt GitHub
# Projet NLP BMS · git@github.com:EL-KPIBOR/Projet_NLP_BMS.git
# ISSEA · ISE3-DS · 4 membres · 2025-2026
# ═══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# ÉTAPE 0 — PRÉREQUIS (une seule fois, sur chaque machine)
# ══════════════════════════════════════════════════════════════

# Vérifier que Git est installé
git --version
# → git version 2.x.x  (si erreur : installer git sur git-scm.com)

# Configurer ton identité Git (remplace avec tes vraies infos)
git config --global user.name  "Prénom Nom"
git config --global user.email "ton.email@issea.cm"

# Vérifier la clé SSH (nécessaire pour git@github.com)
ssh -T git@github.com
# → "Hi EL-KPIBOR! You've successfully authenticated"
# Si erreur → générer une clé SSH :
#   ssh-keygen -t ed25519 -C "ton.email@github.com"
#   cat ~/.ssh/id_ed25519.pub   ← copier dans GitHub > Settings > SSH Keys


# ══════════════════════════════════════════════════════════════
# ÉTAPE 1 — CHEF DE PROJET UNIQUEMENT (Membre 1)
# Initialiser le dépôt depuis le ZIP téléchargé
# ══════════════════════════════════════════════════════════════

# 1.1 Décompresser le ZIP dans un dossier temporaire
# (décompresser nlp_issea_github.zip)

# 1.2 Renommer le dossier
mv nlp_issea_github Projet_NLP_BMS
cd Projet_NLP_BMS

# 1.3 Initialiser Git et connecter au dépôt distant
git init
git remote add origin git@github.com:EL-KPIBOR/Projet_NLP_BMS.git

# 1.4 Premier commit sur main
git add .
git commit -m "feat: initialisation pipeline NLP ISSEA - structure complète"

# 1.5 Pousser sur main
git branch -M main
git push -u origin main

# ✅ Vérifier sur GitHub que les fichiers sont bien là

# 1.6 Créer la branche develop depuis main
git checkout -b develop
git push -u origin develop

# ✅ Tu dois maintenant avoir 2 branches sur GitHub : main + develop

# 1.7 Créer les 4 branches feature depuis develop
git checkout develop

git checkout -b feature/audit-semantique
git push -u origin feature/audit-semantique

git checkout develop
git checkout -b feature/classification-snd30
git push -u origin feature/classification-snd30

git checkout develop
git checkout -b feature/analyse-statistique
git push -u origin feature/analyse-statistique

git checkout develop
git checkout -b feature/dashboard
git push -u origin feature/dashboard

# ✅ Tu dois avoir 6 branches sur GitHub :
#    main · develop · feature/audit-semantique
#    feature/classification-snd30 · feature/analyse-statistique · feature/dashboard

git branch -a
# Liste toutes les branches locales et distantes


# ══════════════════════════════════════════════════════════════
# ÉTAPE 2 — TOUS LES MEMBRES (après que le Chef a fait l'étape 1)
# Cloner et se placer sur sa branche
# ══════════════════════════════════════════════════════════════

# 2.1 Cloner le dépôt
git clone git@github.com:EL-KPIBOR/Projet_NLP_BMS.git
cd Projet_NLP_BMS

# 2.2 Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux / Mac
# venv\Scripts\activate       # Windows (PowerShell)

# 2.3 Installer les dépendances
pip install -r requirements.txt

# 2.4 Chaque membre se place sur SA branche
# --- Membre 1 ---
git checkout feature/audit-semantique

# --- Membre 2 ---
git checkout feature/classification-snd30

# --- Membre 3 ---
git checkout feature/analyse-statistique

# --- Membre 4 ---
git checkout feature/dashboard

# Vérifier sur quelle branche on est
git branch
# La branche active est précédée d'une étoile *


# ══════════════════════════════════════════════════════════════
# ÉTAPE 3 — WORKFLOW QUOTIDIEN (chaque membre, chaque session)
# ══════════════════════════════════════════════════════════════

# 3.1 Toujours commencer par synchroniser
git fetch origin
git merge origin/develop
# → Intègre les avancées des autres dans ta branche

# 3.2 Travailler sur ses fichiers...
#     (modifier src/audit_semantique.py, notebooks/, etc.)

# 3.3 Voir ce qui a changé
git status
git diff                      # Détail des modifications

# 3.4 Committer son travail
git add src/mon_fichier.py    # Ajouter un fichier précis
# ou
git add .                     # Ajouter tout

git commit -m "feat(audit): calcul distances cosinus + euclidienne"
#              ↑      ↑       ↑
#           type   scope    description

# 3.5 Pousser sur GitHub
git push origin feature/ma-branche

# 3.6 Quand une fonctionnalité est terminée → Pull Request sur GitHub
#     Base: develop  ←  Compare: feature/ma-branche
#     Assigner 1 reviewer parmi les autres membres


# ══════════════════════════════════════════════════════════════
# ÉTAPE 4 — PROTECTIONS DE BRANCHES (Chef de projet sur GitHub)
# ══════════════════════════════════════════════════════════════

# Sur GitHub.com → ton repo → Settings → Branches → Add branch rule

# Règle pour "main" :
#   Pattern : main
#   ✅ Require a pull request before merging
#   ✅ Required number of approvals : 2
#   ✅ Dismiss stale pull request approvals when new commits are pushed
#   ✅ Require linear history

# Règle pour "develop" :
#   Pattern : develop
#   ✅ Require a pull request before merging
#   ✅ Required number of approvals : 1


# ══════════════════════════════════════════════════════════════
# ÉTAPE 5 — AJOUTER LES COLLABORATEURS (Chef de projet)
# ══════════════════════════════════════════════════════════════

# Sur GitHub.com → Settings → Collaborators → Add people
# Entrer les usernames GitHub des 3 autres membres
# Chaque membre reçoit un email d'invitation → doit accepter

# Rôle recommandé : "Write" (peuvent push sur leurs branches)
# Le chef garde le rôle "Admin"


# ══════════════════════════════════════════════════════════════
# ÉTAPE 6 — COMMANDES UTILES DU QUOTIDIEN
# ══════════════════════════════════════════════════════════════

# Voir l'historique des commits
git log --oneline --graph --all
# Affichage complet arborescence des branches

# Annuler le dernier commit (sans perdre les fichiers)
git reset --soft HEAD~1

# Voir les différences entre sa branche et develop
git diff feature/ma-branche origin/develop

# Récupérer une branche distante qui n'existe pas en local
git checkout -b feature/dashboard origin/feature/dashboard

# Créer un tag pour la version finale
git tag -a v1.0 -m "Version finale rendue - 17/02/2026"
git push origin v1.0

# Voir tous les tags
git tag

# Lancer le dashboard depuis la racine
streamlit run app.py


# ══════════════════════════════════════════════════════════════
# ÉTAPE 7 — MERGE FINAL POUR LA REMISE (Chef de projet)
# ══════════════════════════════════════════════════════════════

# 7.1 Merger toutes les branches feature → develop
#     (via Pull Requests sur GitHub, avec reviews)

# 7.2 Tester que tout fonctionne sur develop
git checkout develop
git pull origin develop
pip install -r requirements.txt
streamlit run app.py
# → Vérifier que le dashboard tourne sans erreur

# 7.3 Merger develop → main (Pull Request finale)
#     Nécessite 2 approbations (protection de branche)

# 7.4 Créer le tag de version finale
git checkout main
git pull origin main
git tag -a v1.0 -m "Rendu final Projet NLP ISE3 - ISSEA 2025-2026"
git push origin v1.0

# ✅ Le dépôt est prêt pour évaluation !


# ══════════════════════════════════════════════════════════════
# RÉFÉRENCE RAPIDE — RÉSUMÉ DES COMMANDES
# ══════════════════════════════════════════════════════════════

# git status              → Voir l'état du repo
# git branch -a           → Lister toutes les branches
# git checkout <branche>  → Changer de branche
# git fetch origin        → Télécharger sans merger
# git pull origin <b>     → Télécharger + merger
# git add .               → Stager tous les changements
# git commit -m "msg"     → Committer
# git push origin <b>     → Pousser vers GitHub
# git log --oneline       → Historique court
# git diff                → Voir les modifications
# git merge origin/develop→ Intégrer develop dans sa branche

# ══════════════════════════════════════════════════════════
#  Archipel — Instructions de déploiement cloud (Colab)
# ══════════════════════════════════════════════════════════
#
#  Objectif : Exécuter test_validation_baseline.py sur GPU T4
#  Durée estimée : ~15 min
#  Coût : 0 € (Colab gratuit)
#
# ══════════════════════════════════════════════════════════
# ÉTAPE 0 — Pré-requis
# ══════════════════════════════════════════════════════════
#
# ✅ Compte Google (pour Google Colab)
# ✅ Dépôt GitHub contenant le projet Archipel
#    Si pas encore poussé :
#      cd Archipel_Project_Complete
#      git init && git add -A && git commit -m "archipel-train"
#      git remote add origin https://github.com/TON_USER/archipel.git
#      git push -u origin master
#
# ══════════════════════════════════════════════════════════
# ÉTAPE 1 — Ouvrir Colab
# ══════════════════════════════════════════════════════════
#
#   https://colab.research.google.com/
#
#   → Nouveau notebook
#
# ══════════════════════════════════════════════════════════
# ÉTAPE 2 — Activer le GPU
# ══════════════════════════════════════════════════════════
#
#   Menu : Exécution > Modifier le type d'exécution
#   Accélérateur matériel > GPU (T4)
#   > Enregistrer
#
# ══════════════════════════════════════════════════════════
# ÉTAPE 3 — Coller le code cellule par cellule
# ══════════════════════════════════════════════════════════
#
# Voir colab_archipel.ipynb pour les cellules pré-formatées,
# ou ci-dessous :

# CELLULE 1 — Vérif GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU : {torch.cuda.get_device_name(0)}")
else:
    print("❌ Pas de GPU — vérifie l'étape 2")

# CELLULE 2 — Clone
import os
REPO_URL = "https://github.com/TON_USER/archipel.git"   # ← REMPLACE
WORKDIR  = "/content/archipel"
if not os.path.exists(WORKDIR):
    os.system(f"git clone {REPO_URL} {WORKDIR}")
    print(f"✅ Cloné : {WORKDIR}")
else:
    print(f"📂 Present : {WORKDIR}")
os.chdir(WORKDIR)

# CELLULE 3 — Install
!pip install -q torch torchvision numpy pyyaml pytest
print("✅ Dépendances installées.")

# CELLULE 4 — Vérif structure
!ls archipel/src/archipel/

# CELLULE 5 — Lancer le test
!python test_validation_baseline.py --epochs 50 --batch-size 256

# ══════════════════════════════════════════════════════════
# ÉTAPE 4 — Récupérer les résultats
# ══════════════════════════════════════════════════════════
#
# Colab → panneau gauche (📁 Files) → /content/archipel/
# Télécharge :
#   📊 validation_results.json      ← résultats archipel vs MLP
#   📊 training_log.csv             ← métriques détaillées
#   📄 test_output.csv              ← évaluation
#
# ══════════════════════════════════════════════════════════

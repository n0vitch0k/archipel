#!/usr/bin/env bash
# ============================================================
#  Archipel Project — Colab Cloud Setup Script
#  Colle ce bloc dans une cellule Colab et exécute.
# ============================================================

set -e

echo "═══════════════════════════════════════════════════════"
echo "  Archipel Project — Configuration Cloud Colab"
echo "═══════════════════════════════════════════════════════"

# ---- 1. Vérification GPU ----
echo ""
echo "[1/5] Vérification GPU..."
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'AUCUN — vérifie Runtime > Change runtime type > GPU')"

# ---- 2. Clone du dépôt ----
echo ""
echo "[2/5] Clone du dépôt..."
REPO_URL="https://github.com/<TON_USER>/archipel.git"  # ⚠️ REMPLACE par l'URL de ton dépôt
WORKDIR="/content/archipel"

if [ ! -d "$WORKDIR" ]; then
    git clone "$REPO_URL" "$WORKDIR"
    echo "    Dépôt cloné : $WORKDIR"
else
    echo "    Dépôt déjà présent : $WORKDIR"
fi
cd "$WORKDIR"

# ---- 3. Installation des dépendances ----
echo ""
echo "[3/5] Installation des dépendances..."
pip install -q torch numpy torchvision pyyaml pytest typer
echo "    Dépendances installées."

# ---- 4. Vérification structure ----
echo ""
echo "[4/5] Vérification structure..."
ls -la archipel/src/archipel/
echo "    Structure OK"

# ---- 5. Lancement du test de validation ----
echo ""
echo "[5/5] Lancement du test de validation 50 époques..."
echo "    (~15 min sur T4, surveille la barre de progression)"

python test_validation_baseline.py --epochs 50 --batch-size 256

# ---- Récupération des résultats ----
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Test terminé !"
echo "  Fichiers générés :"
echo "    - training_log.csv"
echo "    - test_output.csv"
echo "    - validation_results.json (si pas de crash)"
echo ""
echo "  Télécharge-les depuis le panneau gauche (📁 Files)"
echo "═══════════════════════════════════════════════════════"

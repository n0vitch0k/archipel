# Archipel — Plan de travail

**Dernière mise à jour** : 2026-05-22  
**État** : Phase 2 STABLE — Niveau 1.2 ✅ — Niveau 1.4/1.5 ✅ (spécialisation à corriger)  
**Prochaine session** : Re-test Niveau 1.4/1.5 avec corrections Courant puis Niveau 1.6

---

## 📊 État actuel du projet

### Ce qui est stable et fonctionnel
- Îlots (BaseIsland) avec perte locale et régularisation homéostatique
- Ocean partagé (coherence_center, proximity_matrix, deposit/deposit_batch)
- Router par corrélation cosinus (top-k + ε-greedy + specialization boost)
- Courant (régulateur adaptatif λ_coh, λ_div, λ_ent)
- Cycle de vie : spawn (HyperNetwork) + kill (apoptose avec distillation)
- 45+ tests passent (16/16 archipel + 10/10 lifecycle + 7/7 specialization + 5/5 CSVLogger + 2/2 save_load + 3/3 train_script)
- Entraînement viable sur données structurées (loss −73.6%, diversité +0.31, 3 births/3 deaths)
- Entraînement viable sur MNIST réel (accuracy 96.3%, loss −54.9%, 6 births/6 deaths sur 2 époques)
- Save/load complet via `ArchipelPhase2.save_checkpoint()` / `load_checkpoint()`
- Tests assert-driven : plus aucun `return True/False` dans les tests pytest
- Script d'entraînement configurable via YAML (`archipel/train.py` + `archipel/configs/default.yaml`)
- Logger CSV (`archipel/src/archipel/training/logger.py`, 5 tests)

### Ce qui est en cours / à valider
- [x] Save/load complet (round-trip architecture dynamique + buffers non persistants)
- [x] Tests avec `assert` (au lieu de `return True/False`)
- [x] Script `train.py` + config YAML
- [x] 1.1 Logger CSV complet
- [x] 1.2 Test MNIST rapide — **VALIDÉ** : 2 époques → accuracy 96.3%, loss -54.9%, 6 births/6 deaths
- [x] 1.3 Run long MNIST 50 époques (CPU, ~2h)
- [x] 1.4/1.5 Test validation + baseline MLP — **TERMINÉ** (premier run)
  - Archipel accuracy test : 0.9887 | MLP accuracy test : 0.9903
  - Spécialisation : **échec** — toutes les îles prédisent classe 1 (spé = 0.0009)
  - Cause identifiée : `λ_coh` et `λ_diversity` bloqués par des plafonds trop bas et des cibles mal calibrées
  - Corrections appliquées sur `courant.py` (voir ci-dessous)
- [ ] 1.6 Re-test Niveau 1.4/1.5 avec corrections Courant
- [ ] 1.6 Ablation sans distillation (impact sur performance finale)

---

## 🔭 Vision long terme (Phase 3+)

### Étape A — Résonance Kuramoto
**Remplacer le routing statique par des oscillateurs couplés**  
→ Synchronisation de phase comme signal de routing, ODE différentiable, communication locale entre îles

### Étape B — Apprentissage local
**Remplacer la backprop globale par Hebb + consolidation**  
→ Chaque île apprend localement, le Courant envoie une récompense sparse, phase de sommeil consolide

### Étape C — Émergence Shapley
**Remplacer le seuil de vie/mort par une valeur d'utilité marginale**  
→ Naissance/mort basées sur la contribution réelle à la performance, pas sur une heuristique de variance

### Étape D — Benchmark compositionnel
**Valider la modularité sur une tâche compositionnelle**  
→ Tâche où les îles doivent apprendre des sous-fonctions distinctes et les recomposer

---

## 🗓️ Plan détaillé par niveau

### Niveau 0 — Fondation (2-3h)
**Objectif** : Rendre le projet utilisable par quelqu'un d'autre

#### 0.1 Save/load (1h) ✅ TERMINÉ
- `ArchipelPhase2.save_checkpoint()` sauvegarde config, `state_dict`, buffers `OceanSpace`, buffers `IslandSpecialization` et compteurs runtime lifecycle
- `ArchipelPhase2.load_checkpoint()` reconstruit l'architecture dynamique avant chargement
- Tests round-trip : architecture après spawn, sorties déterministes après reload, buffers non persistants restaurés

#### 0.2 Assert-driven tests (30min) ✅ TERMINÉ
- `test_archipel.py`, `test_lifecycle.py` et `test_specialization.py` ne retournent plus de booléens depuis les fonctions pytest
- Ajout de `test_assert_driven.py`, garde-fou AST qui échoue si un test pytest contient `return True` / `return False`
- Les warnings `PytestReturnNotNoneWarning` ont disparu

#### 0.3 Config + script train (1h) ✅ TERMINÉ
- Créé `archipel/configs/default.yaml` avec sections `model`, `training`, `data`
- Créé `archipel/train.py` avec CLI `--config`, `--seed`, `--quiet`
- Le script construit `ArchipelPhase2`, génère un dataset synthétique structuré, lance `train_loop_lifecycle()` et sauvegarde `final.pt`
- Ajout de `test_train_script.py` : config par défaut, builders importables, smoke run CLI avec checkpoint final
- Dépendances projet mises à jour : `numpy` et `PyYAML`

---

### Niveau 1 — Qualité (3-4h)
**Objectif** : Avoir des outils d'observation pour les runs longs + valider sur données réelles

#### 1.1 Logger CSV (2h) ✅ TERMINÉ
- `archipel/src/archipel/training/logger.py` (CSVLogger + `save_logs_to_csv`)
- 5 tests (`archipel/test_csv_logger.py`)

#### 1.2 Test MNIST rapide (1-2h) ✅ TERMINÉ
- `test_mnist_quick.py` — Class `MNISTArchipel(ArchipelPhase2)` + encodeur CNN 28×28→128
- Validation 2 époques : accuracy 96.3%, loss -54.9%, 6 births/6 deaths ✅
- Corrections appliquées : ocean_dim=128, encodeur optionnel dans `distill_island_to_neighbors()`

#### 1.3 Run long MNIST 20 époques 🔄 EN COURS
- `test_mnist_long.py` — logging loss/accuracy/diversité/nb_îlots par époque
- Objectifs :
  - Vérifier la convergence à long terme (10-20 époques)
  - Mesurer la stabilité du cycle de vie (births/deaths par époque)
  - Évaluer si la diversification des îles augmente ou stagne

---

### Niveau 2 — Recherche (selon disponibilité)
**Objectif** : Passer de MoE dynamique à Archipel théorie

Chaque étape est ~1-2 semaines de R&D. Voir la section "Prochaines étapes" du README pour les détails.

---

## 🔧 Corrections appliquées après Niveau 1.4/1.5

### Problème identifié — Spécialisation nulle
- Toutes les îles prédisent la classe 1 à ~11% → spé = 0.0009
- `λ_coh` et `λ_diversity` bloqués par des plafonds trop restrictifs
- Cibles `diversity_target` et `coherence_target` mal calibrées vs valeurs observées sur MNIST

### Corrections sur `archipel/src/archipel/current/courant.py`
| Paramètre | Avant | Après |
|-----------|-------|-------|
| `lambda_diversity_init` | 0.1 | **0.2** |
| `diversity_target` | 0.15 | **0.25** |
| `lambda_diversity` max | 1.0 | **3.0** |
| `lambda_diversity` min | 0.01 | **0.1** |
| `coherence_target` | 0.1 | **0.5** |
| `lambda_coherence` max | 1.0 | **2.0** |

### À valider
- Re-run Niveau 1.4/1.5 → vérifier que la spécialisation atteint spec_mean > 0.5
- Si toujours faible : tester `λ_diversity` fixe élevé (3.0 constant) pour isoler l'effet

---

## 📁 Structure du projet (fichiers stables)

```
Archipel_Project_Complete/
├── archipel/
│   ├── src/archipel/
│   │   ├── islands/
│   │   │   ├── base_island.py      ✅ STABLE
│   │   │   ├── lifecycle.py         ✅ STABLE
│   │   │   └── specialization.py   ✅ STABLE
│   │   ├── ocean/
│   │   │   └── ocean.py             ✅ STABLE
│   │   ├── current/
│   │   │   ├── router.py            ✅ STABLE
│   │   │   └── courant.py           ✅ STABLE
│   │   └── training/
│   │       ├── loop.py              ✅ STABLE
│   │       └── loop_lifecycle.py    ✅ STABLE
│   ├── configs/                     [À créer] Niveau 0
│   ├── train.py                     [À créer] Niveau 0
│   ├── test_archipel.py             ✅ 16/16 passent
│   ├── test_lifecycle.py            ✅ 10/10 passent
│   ├── test_save_load.py            ✅ 2/2 passent
│   ├── test_mnist_quick.py          [À créer] Niveau 1
│   └── README.md                    ✅ À jour
├── PROJET_ARCHIPEL_PLAN.md          📄 Ce fichier
└── (temporaires supprimés)
```

# Changelog — Archipel Project

## [2026-05-22] — Niveau 1.5 terminé + corrections Courant

### Terminé
- **Niveau 1.3** : Run long MNIST 50 époques CPU (training_log.csv complet)
- **Niveau 1.4** : Module `utils/specialization_matrix.py` — `compute_specialization_matrix()` et `specialization_score()` opérationnels
- **Niveau 1.5** : Baseline MLP (`baselines/mlp.py`, MLPBaseline 128→256→128→10) — comparaison Archipel vs MLP sur test set MNIST
- **Architectures modulaires** : packages `models/`, `baselines/`, `utils/` créés dans `archipel/src/archipel/`
- **`loop_lifecycle.py`** : ajout de `island_outputs` (per-island logits) dans le retour de `forward()`, requis par `compute_specialization_matrix()`
- **`test_validation_baseline.py`** : importsExternalisés vers les nouveaux modules (suppression duplication de code)

### Résultats Niveau 1.5 (premier run CPU)
| Métrique | Archipel | MLP Baseline |
|----------|:--------:|:------------:|
| Accuracy train | 0.9987 | 0.9985 |
| Accuracy test  | 0.9887 | 0.9903 |
| Loss finale     | 0.8384 | 0.0017 |

→ Équivalence sur test set, loss différente (attendue car Archipel = perte composite).

### Problème identifié — Niveau 1.4 (spécialisation)
- Score spé = 0.0009 — toutes les îles prédisent classe 1 à ~11%
- Causes : `λ_coh` et `λ_diversity` bloqués par des plafonds trop restrictifs dans `Courant`
- `diversity_target=0.15` vs diversité observée ~0.25
- `coherence_target=0.1` vs coherence_loss observée ~0.7–1.2

### Corrections appliquées
**Fichier** : `archipel/src/archipel/current/courant.py`

| Paramètre | Avant | Après |
|-----------|-------|-------|
| `lambda_diversity_init` | 0.1 | 0.2 |
| `diversity_target` | 0.15 | 0.25 |
| `lambda_diversity` max | 1.0 | 3.0 |
| `lambda_diversity` min | 0.01 | 0.1 |
| `coherence_target` | 0.1 | 0.5 |
| `lambda_coherence` max | 1.0 | 2.0 |

### Documentation mise à jour
- `PROJET_ARCHIPEL_PLAN.md` — état Niveau 1.3/1.4/1.5, section corrections
- `archipel/README.md` — section Courant détaillée (paramètres + régime d'adaptation), liste des fichiers stables étendue

### Tests
- 9/9 tests pytest passent (non-régression confirmée)
- `archipel/test_csv_logger.py` + `test_train_script.py` + `test_assert_driven.py`

---
## [2026-05-22] — Niveau 1.2 validé + corrections MNIST

### Corrections appliquées
- `MNISTArchipel` hérite maintenant `ArchipelPhase2` (bug wrapper lifecycle)
- `ocean_dim=128` aligné sur sortie encodeur CNN pré-existant
- `distill_island_to_neighbors()` et `kill_island()` acceptent un encodeur optionnel pour pré-encoder les images brutes pendant蒸馏

### Résultats Niveau 1.2 (rapide 2 époques)
- Accuracy : 96.3%
- Loss : 2.40 → 1.08 (−54.9%)
- Births/deaths : 6/6

---
## [2026-05-22] — Architecture stable Phase 2

### Stabilisation
- 26/26 tests passent
- 10/10 lifecycle + 16/16 archipel
- Save/load round-trip vérifié
- Scripts `train.py` + `default.yaml` opérationnels
- Logger CSV intégré

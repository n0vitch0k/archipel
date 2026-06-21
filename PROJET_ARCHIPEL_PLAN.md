# Archipel — Plan de travail

**Dernière mise à jour** : 2026-06-21  
**État** : Phase 2 STABLE (N1.0–N1.6) — Spécialisation fonctionnelle ✅  
**Cloud** : Modal CI opérationnel  
**Prochaine décision** : Phase 3 (Kuramoto) ou renforcement de la spécialisation

---

## 📊 État actuel du projet

### Ce qui est stable et validé (Phase 2)

| Composant | Statut | Preuve |
|-----------|--------|--------|
| **Curriculum top-k** 3→2→1 | ✅ 9 tests | `topk_curriculum.py`, testé sur Modal |
| **Routing cosinus** + ε-greedy + specialization boost | ✅ | `router.py`, testé unitairement |
| **Anti-churn** (protection des îles jeunes) | ✅  | `loop_lifecycle.py`, 0 morts prématurées observées |
| **Traitement d'exposition** (EMA) | ✅ | `specialization.py` section `update()` |
| **Cycle de vie** spawn/kill avec distillation | ✅ 8 tests | `lifecycle.py`, vérifié sur runs réels |
| **Courant** (régulateur adaptatif λ) | ✅  | `courant.py`, diversité/corhérence stables |
| **Logger CSV** | ✅ 5 tests | `logger.py`, toutes les métriques logging |
| **Save/load checkpoint** | ✅ 2 tests | Round-trip complet |
| **Train script** (CLI + YAML) | ✅ 3 tests | `train.py`, `configs/default.yaml` |
| **Tests unitaires** (19) | ✅ | 9 topk + 7 spec + 3 train — passent sur Modal |
| **Modal CI cloud** | ✅ | `modal_run.py`, `modal_validate.py` — temps réel |

### Validation MNIST — Résultats longs (50 epochs)

| Seed | Accuracy | spec_coverage | Births/Deaths | Îles finales |
|------|----------|---------------|---------------|--------------|
| 42 | 96.4% | — | 8/8 | 4 |
| **123** | **98.4%** | **3** ✅ | 54/54 | 4 |
| **256** | **98.9%** | **2** ✅ | 23/23 | 4 |

**Conclusion : Spécialisation reproductible (2/3 seeds ≥ 2).** L'architecture converge entre 96 et 99% d'accuracy sur MNIST selon l'initialisation.

---

### ⚠️ État — Spécialisation

---

## 🔧 Métriques de spécialisation

### Interne (EMA tracker) — `get_specialization_summary()`
Dans `specialization.py`, méthode `update()` :
```python
score = confidence_rate - 0.5 * class_usage
```
- `coverage` = nombre de classes distinctes avec au moins une île spécialisée
- Une île est spécialisée si : score ≥ 0.02 ET score > moyenne ET marge top-2 ≥ 0.02
- Mise à jour EMA (`alpha=0.1`) pour lisser

### Externe (pureté d'île) — `compute_specialization_coverage()`
Dans `specialization_matrix.py` :
- Calcule la pureté par île = `max(row) / row.sum()` sur la matrice îles×classes
- `coverage_raw` = sur la matrice brute (toutes prédictions)
- `coverage_fn` = sur la matrice filtrée (prédictions correctes seulement)
- Seuil de pureté par défaut : 0.30

### Limitation connue
Les métriques entropiques (`specialization_score_precision_weighted`) donnent ~0.001 même quand coverage=2 car les îles généralistes diluent la moyenne. **Utiliser coverage en priorité.**

---

## 🗺️ Prochaines étapes suggérées

### Court terme (1 session)
1. [x] ~~Lancer 3 runs 50 epochs~~ **FAIT** — coverage≥2 confirmé sur 2/3 seeds
2. [x] ~~Décision sur la spécialisation~~ **FAITE** — Oui, on peut avancer

### Prochaine étape — Phase 3 : Résonance Kuramoto
3. [ ] Implémenter un router par oscillateurs couplés (Kuramoto)
4. [ ] Valider sur MNIST : accuracy + spécialisation avec le nouveau routing
5. [ ] Comparer avec le routing cosinus actuel

**Plan détaillé** : `doc/phase3_kuramoto_plan.md`

**Principe** : Remplacer `cos(island_state, input_repr)` par `cos(θ_i - φ(x))` où θ_i est la phase oscillatoire de l'île et φ(x) la phase cible de l'entrée. Les phases évoluent selon la dynamique Kuramoto : synchronisation si co-traitement, désynchronisation sinon.

| Sprint | Contenu | Fichiers |
|--------|---------|----------|
| 1 | Routeur Kuramoto | `archipel/src/archipel/current/kuramoto.py` |
| 2 | Intégration Archipel + Phase 3 | `loop_lifecycle.py` (nouveau `ArchipelPhase3`) |
| 3 | Validation 3 seeds Modal | `modal_validate.py`, `validate_niveau1415.py` |

### Plus tard
6. [ ] Ablation sans curriculum top-k
7. [ ] Validation sur CIFAR-10 (tâche plus dure)
8. [ ] Apprentissage local (Hebb + consolidation)

---

## 📁 Structure du projet

```
Archipel_Project_Complete/
├── .github/workflows/tests.yml     ✅ CI GitHub Actions (bloqué billing)
├── .gitignore                       ✅ Propre
├── archipel/
│   ├── src/archipel/
│   │   ├── islands/
│   │   │   ├── base_island.py       ✅ STABLE
│   │   │   ├── lifecycle.py         ✅ STABLE
│   │   │   └── specialization.py    ✅ STABLE (EMA tracker)
│   │   ├── ocean/
│   │   │   └── ocean.py             ✅ STABLE
│   │   ├── current/
│   │   │   ├── router.py            ✅ STABLE (cosinus + boost)
│   │   │   ├── courant.py           ✅ STABLE (régulateur)
│   │   │   └── topk_curriculum.py   ✅ STABLE (3→2→1)
│   │   ├── training/
│   │   │   ├── loop.py              ✅ STABLE
│   │   │   └── loop_lifecycle.py    ✅ STABLE (anti-churn + diagnostics)
│   │   └── utils/
│   │       └── specialization_matrix.py  ✅ NOUVEAU (métriques coverage)
│   ├── train.py                     ✅ STABLE
│   ├── configs/default.yaml         ✅ STABLE
│   ├── test_topk_curriculum.py      ✅ 9 tests
│   ├── test_specialization.py       ✅ 7 tests
│   ├── test_train_script.py         ✅ 3 tests
│   └── README.md                    [À mettre à jour]
├── test_mnist_quick.py              ✅ Validation MNIST rapide
├── test_baseline_comparison.py      [Obsolète — fusionné dans validate_niveau1415.py]
├── validate_niveau1415.py           ✅ Validation complète + MLP + coverage
├── validate_niveau1415_quick.py     ✅ Validation rapide + coverage
├── modal_run.py                     ✅ Tests unitaires Modal
├── modal_validate.py                ✅ Suite complète Modal
├── PROJET_ARCHIPEL_PLAN.md          📄 Ce fichier
└── validation_results.json          [Ignoré par git]
```

---

## 🔧 Paramètres clés Courant (MNIST)

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `lambda_diversity_init` | 0.2 | Poids initial de la diversité |
| `diversity_target` | 0.25 | Cible de diversité entre îles |
| `lambda_diversity` max | 3.0 | Plafond du poids diversité |
| `coherence_target` | 0.5 | Cible de cohérence (proximité ocean) |
| `lambda_coherence` max | 2.0 | Plafond du poids cohérence |

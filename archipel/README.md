# Archipel — Phase 2 Lifecycle Implementation

**Status: Phase 2 STABLE** — All 26 tests passing (16/16 test_archipel + 10/10 test_lifecycle). Training loop verified with signal data: loss −73.6%, diversity +0.31, births/deaths working. No crashes on random or multi-run.

---

## What is Archipel?

Archipel is a biologically-inspired neural architecture where specialized sub-networks ("Îlots" / Islands) compete and cooperate through a shared latent space called the Ocean. Unlike dense or Mixture-of-Experts networks, Archipel uses:

- **Correlation-based routing**: Islands are selected by cosine-similarity between their latent states and the encoded input
- **Homeostatic regulation**: Each island maintains its own activation diversity via entropy-based regularizers
- **Adaptive loss (Courant)**: Lambda weights for coherence/diversity/entropy dynamically adjust each step
- **Island lifecycle**: Islands can be born (spawned via HyperNetwork) or die (apoptosis with distillation)

---

## Architecture Overview

```
Input x
    │
    ▼
┌─────────────┐      ┌──────────────────────────────────────────┐
│  Encoder    │      │              OCEAN (shared space)         │
│  (128→32)   │─────►│  coherence_center (running mean of        │
└─────────────┘      │  island embeddings)                       │
    │                 │  deposit_all() / get_island_embeddings()│
    ▼                 └──────────────────────────────────────────┘
┌─────────────┐                    ▲          ▲
│  Router     │◄──island_states────┘          │
│ (top_k sel) │                               │
└──────┬──────┘                               │
       │ routing_weights                      │ island_embeds
       ▼                                      │
  ┌─────────────┐  (batch,4)      ┌─────────────────────────┐
  │  Weighted   │◄───────────────│  ISLANDS (Îlots) ×4    │
  │  Aggregation│                │  local_projection (FC) │
  └──────┬──────┘                │  compute_local_loss()   │
         │ (batch, 32)           │  homeostatic_reg()     │
         ▼                        │  spawn/kill lifecycle  │
  ┌─────────────┐                └─────────────────────────┘
  │ Task Head   │  (batch, num_classes)
  └─────────────┘
```

---

## File Structure

```
archipel/src/archipel/
├── __init__.py
├── islands/
│   ├── __init__.py
│   ├── base_island.py     # BaseIsland (MoE expert, local loss, homeostatic reg)
│   └── lifecycle.py       # IslandLifecycle (birth/death), distill_island_to_neighbors()
├── ocean/
│   ├── __init__.py
│   └── ocean.py           # Ocean + OceanSpace (shared embedding, coherence center, EMA)
├── current/
│   ├── __init__.py
│   ├── router.py          # HyperNetworkRouter + HyperNetworkGenerator
│   └── courant.py          # Courant (adaptive lambda weights)
└── training/
    ├── __init__.py
    ├── loop.py             # train_loop() + compute_*_loss() functions
    └── loop_lifecycle.py   # ArchipelPhase2 + train_loop_lifecycle() (Phase 2)
```

---

## Core Classes

### BaseIsland (`islands/base_island.py`)
```
BaseIsland(island_id, input_dim, hidden_dim, output_dim, num_experts=4)
  ├── forward(x) → embeddings (batch, output_dim)
  ├── compute_local_loss(embeddings) → scalar (L2 dev from mean, no input needed)
  ├── compute_local_loss_from_input(x, noise_std) → scalar (L2 with noise proxy)
  ├── homeostatic_regularizers(x=None) → {activity, diversity}
  └── get_expert_usage(x) → gate probabilities (batch, num_experts)
```

### OceanSpace (`ocean/ocean.py`)
```
OceanSpace(num_islands=4, embedding_dim=32, memory_size=256)
  ├── deposit(island_id, embedding)   — EMA-update island embedding in Ocean (alpha=0.99)
  ├── deposit_batch(island_embeds)     — batch EMA-update all islands
  ├── get_context(island_id) → vector — retrieve nearby islands' embeddings
  ├── get_proximity_matrix() → matrix — pairwise cosine similarity
  ├── update_coherence_center()        — running mean of all island embeddings
  ├── compute_proximity() → proximity scores (uses detached buffers)
  └── interaction_counts: EMA counter per island for activity tracking

Ocean(OceanSpace)
  ├── deposit_all(island_embeds)  — all islands deposit simultaneously
  ├── get_island_embeddings() → tensor — current state of all islands (.clone().detach())
  ├── update_coherence_center(variance_weight=0.01) — running mean update
  ├── get_statistics() → dict — proximity_mean, variance, etc.
  └── coherence_center property (tensor, shape (ocean_dim,))
```

**Important**: `island_embeddings` and `interaction_counts` are buffers modified in-place via EMA (`copy_()`). They are always accessed with `.detach()` or `.clone().detach()` during forward to prevent autograd version conflicts. See Troubleshooting §1.

### HyperNetworkRouter (`current/router.py`)
```
HyperNetworkRouter(embedding_dim=32, num_islands=4, top_k=2)
  ├── compute_correlations(island_states, input_repr) → (batch, num_islands) cosine sims
  ├── forward(input_repr, island_states) → {routing_weights, entropy, correlations, sparsity}
  ├── island_thresholds (learnable, per-island activation threshold)
  └── epsilon_scale (learnable, adaptive exploration rate)
```

### Courant (`current/courant.py`)
```
Courant(
    num_islands=4,
    lambda_coherence_init=0.1,      # initial weight for coherence loss
    lambda_diversity_init=0.2,      # initial weight for diversity loss
    lambda_entropy_init=0.01,        # initial weight for entropy reg
    adaptation_rate=0.01,            # per-step weight update rate
    target_entropy=0.8,              # target routing entropy
    diversity_target=0.25,           # target diversity loss (calibrated MNIST)
    coherence_target=0.5,            # target coherence loss (calibrated MNIST)
)
  ├── step(entropy, diversity, coherence) → {
  │     lambda_coherence, lambda_diversity, lambda_entropy,
  │     epsilon_modulation, mean_entropy, mean_diversity }
  └── get_state_report() → dict with current step_count and lambda values
```

**Régime d'adaptation** (2026-05-22) :
- Si `diversity < 0.8×diversity_target` → `lambda_diversity *= 1.1` (jusqu'à 3.0)
- Si `diversity > 1.5×diversity_target` → `lambda_diversity *= 0.95` (min 0.1)
- Si `coherence < 0.7×coherence_target` → `lambda_coherence *= 0.95`
- Si `coherence > 2.0×coherence_target` → `lambda_coherence *= 1.1` (jusqu'à 2.0)
- Si `entropy < 0.8×target_entropy` → `lambda_entropy *= 1.1` (jusqu'à 0.1)
- Si `entropy > 1.2×target_entropy` → `lambda_entropy *= 0.95` (min 0.001)

### HyperNetworkGenerator (`current/router.py`)
```
HyperNetworkGenerator(seed_dim=64, context_dim=32, output_dim=8192, num_layers=3, hidden_dim=256)
  └── forward(seed, context) → (batch, output_dim) weight tensor for island initialization
```

### IslandLifecycle (`islands/lifecycle.py`)
```
IslandLifecycle(num_islands, input_dim, hidden_dim, ocean_dim, max_islands=8, min_islands=2)
  ├── compute_coherence_variance(active_embeds) → scalar
  ├── should_spawn(coherence_variance) → bool
  ├── should_kill() → List[int] of island IDs to kill
  ├── update_gradient_tracking(island_id, grad_norm)
  ├── step_gradient_history()
  ├── get_state_summary() → dict
  └── hypernet: HyperNetworkGenerator instance

distill_island_to_neighbors(dying_island, neighbor_islands, dataloader, steps=50, lr=1e-4, device="cpu")
  — Re-encodes samples through dying island (teacher) and matches neighbor embeddings (student)

get_context_for_spawn(active_island_embeds, active_routing_weights) → context vector
```

---

## Training Loop

### Phase 1 — `train_loop()` (`training/loop.py`)
```
train_loop(model, dataloader, optimizer, courant, epochs=1, log_every=10)
  → (logs: List[Dict], updated_courant: Courant)

Loss = task_loss
     + λ_coherence * coherence_loss
     + λ_diversity * diversity_loss
     + λ_entropy * entropy_reg
     + λ_structural * structural_reg (prev routing stability)
     + λ_homeostatic * sum_over_islands(homeostatic_loss)
```

### Phase 2 — `train_loop_lifecycle()` (`training/loop_lifecycle.py`)
Same as Phase 1 plus:
- Lifecycle evaluation each step: compute_coherence_variance → spawn decision
- Gradient norm tracking per island → death decision
- Birth event: spawn via HyperNetworkGenerator
- Death event: distill knowledge to neighbors before removal

---

## Tests

```bash
# Run all tests (from archipel/ directory)
cd archipel
python -m pytest test_archipel.py test_lifecycle.py -v

# Run specific test
python -m pytest test_lifecycle.py::test_lifecycle_training_loop -v
```

### Test Coverage

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_archipel.py` | 16 | ✅ ALL PASS |
|| `test_lifecycle.py` | 10 | ✅ ALL PASS |

**All tests passing as of latest fix** (`train_loop_lifecycle` lifecycle-ordering fix):
- `test_lifecycle_training_loop` — ✅ FIXED (was: autograd version conflict on `[64, 32]` at step ~17)
- `test_distillation_before_death` — ✅ FIXED (was: mock signature mismatch)

---

## Phase 1 Results Summary

```
16/16 test_archipel.py  — ALL PASSED
10/10 test_lifecycle.py — ALL PASSED ← Fixed: lifecycle eval moved post-backward step
Total: 26/26 tests passing (all green)

Key metrics from structured data convergence test (3 epochs, 200 samples):
  - Loss: 2.35 → 1.24  (47% improvement)
  - Coherence: 1.15 → 0.04
  - Diversity: 0.003 → 0.69
  - λ_coherence: 0.095 → 1.000 (full activation)
```

---

## Phase 2 Roadmap

### ✅ Complété (stable)
- [x] Dynamic island spawning (birth) via HyperNetwork → tested in LC6
- [x] Dynamic island death (apoptosis) with distillation → implemented in `distill_island_to_neighbors()`
- [x] Fix autograd version conflict in lifecycle training loop → lifecycle eval moved post-backward
- [x] Fix mock signature for distillation test
- [x] Save/load survive spawn/kill via `register_buffer()` resize (OceanSpace + IslandSpecialization)
- [x] Island specialization tracking per class label (IslandSpecialization with buffer resize)
- [x] Multi-agent coordination: spawn + kill in same step works (tested in `test_lifecycle.py`)

### En cours / À valider
- [x] Save/load fonctionnel pour ArchipelPhase2 (round-trip complet validé)
- [x] Assert-driven tests (plus aucun test pytest ne retourne `True/False`)
- [x] Script `train.py` + config YAML pour lancer un entraînement sans éditer le code
- [x] Logger CSV (`archipel/src/archipel/training/logger.py`, `CSVLogger`, `save_logs_to_csv`, 5 tests)
- [x] Validations sur données réelles (MNIST/CIFAR) — **MNIST V2 validé** : 2 époques → accuracy 96.3%, loss -54.9%, lifecycle 6b/6d

### À venir (recherche — Phase 3+)
- [ ] **Étape A** — Résonance oscillatoire Kuramoto dans l'Ocean (remplace routing cosinus)
- [ ] **Étape B** — Apprentissage local + consolidation périodique (remplace backprop globale)
- [ ] **Étape C** — Émergence de structure par valeur de Shapley (remplace seuil fixe)
- [ ] **Étape D** — Benchmark compositionnel pour valider la modularité

---

## Troubleshooting

### 1. RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation

**Symptom**: Training loop crashes at step ~17 with `RuntimeError` on tensor `[64, 32]` (AsStridedBackward0).

**Cause**: `island_embeddings` and `interaction_counts` are modified via `copy_()` in `deposit()`/`deposit_batch()` (EMA updates). If any autograd edge connects to the live version of a buffer, backward pass checks the version counter and fails when it has been incremented by in-place ops.

**✅ All buffer autograd issues resolved in `ocean.py`**:
- `deposit()`: detached existing embeddings before computing EMA blend; `interaction_counts` updated via cloned/detached tensor
- `deposit_batch()`: uses `[:n].data.copy_()` instead of `[:n].copy_()` to bypass version counter
- `compute_proximity()`: all buffer reads are detached
- `get_island_embeddings()`: returns `.clone().detach()` by default

**✅ Fixed: lifecycle evaluation ordering in `train_loop_lifecycle()`**:
- Root cause: `kill_island()` / `spawn_island()` was called **before** `loss.backward()`, mutating the `ModuleList` and router parameters while the autograd graph for the backward pass was still being built → `RuntimeError: modified by inplace operation: [batch, ocean_dim]`
- Fix: moved birth/death evaluation, `step_gradient_history()`, `courant_state` recompute, and `get_statistics()` calls to **after** `optimizer.step()`

**✅ Fixed: `OceanSpace.resize()` — buffer size mismatch on spawn**:
- Root cause: `proximity_matrix`, `interaction_counts`, `island_embeddings` were initialised with a fixed size in `__init__`. When a new island was spawned, only `island_embeddings` was extended — `proximity_matrix` stayed at its original size, causing `RuntimeError: tensor sizes must match` in `compute_proximity()` / `get_statistics()`.
- Fix: added `OceanSpace.resize()` that properly expands all island-dependent buffers via `register_buffer()` (creates new tensors, copies existing data, re-registers).

**✅ Fixed: `IslandSpecialization.resize()` — buffer overwrite + `_num_active_islands` desync**:
- Root cause: `self.scores = torch.zeros(...)` and `self.counts = torch.zeros(...)` assigned new Python objects without `register_buffer()`, breaking the buffer registration and causing silent zeroing on every resize.
- Root cause: `_num_active_islands` was computed from `num_islands` (which was already pre-synced to the new count by the caller), making the spawn/vs-kill detection always resolve to kill → `_num_active_islands = 1` after a spawn → `RuntimeError: tensor a (5) vs b (4)` in the routing boost addition.
- Fix: `resize()` now uses `self.register_buffer("scores", new_scores)` and `self.register_buffer("counts", new_counts)`. New parameter `is_spawn: bool` makes the intent explicit: `True` → all islands active, `False` → preserve activity ratio.

---

## Key Design Decisions

1. **No gradient coupling in structural reg**: `prev_routing_weights.detach()` prevents the structural loss from backpropagating into the router's previous step — avoids broken graphs when batch sizes change.
2. **Re-encode distillation**: Dying island re-encodes real samples through itself to get teacher embeddings; neighbors encode same samples to get student embeddings. Avoids the proxy-distillation instability of the original stored-embedding approach.
3. **Fixed small islands for Phase 1**: 4–8 islands, no dynamic spawning yet. Lifecycle infrastructure is in place for Phase 2.
4. **Homeostatic regularizers are negative entropy**: `diversity = -entropy` so the Courant optimizer naturally increases diversity (more negative = penalized → pushes entropy up).
5. **Buffer isolation via detach/clone**: All buffer reads during forward pass are detached to prevent autograd from tracking in-place mutations that would conflict with the version counter on backward.
6. **Buffer resize via `register_buffer()`**: When buffers must grow/shrink (spawn/kill), reassigning `self.buffer = new_tensor` silently drops the registration. Always use `self.register_buffer("name", new_tensor, persistent=False)` inside `resize()` methods.
7. **Lifecycle ops only after optimizer.step()**: Mutating `ModuleList` (adding/removing islands) or reassigning router parameters mid-backward corrupts the `grad_fn` version chain. All birth/death evaluation is sequestered to the post-backward, pre-log block of `train_loop_lifecycle()`.
8. **Explicit intent flag in `resize()`**: Passing `is_spawn=True` / `is_spawn=False` removes all ambiguity about whether `_num_active_islands` should grow to full capacity or preserve a ratio. Never infer from old/new sizes when callers pre-sync `num_islands`.

---

## Prochaines étapes

### Niveau 0 — Fondation (2-3h, à faire en premier)

#### Tâche 0.1 — Save/load fonctionnel ✅ terminé
**Fichier modifié** : `archipel/src/archipel/training/loop_lifecycle.py` (ArchipelPhase2)  
**Fichier créé** : `test_save_load.py`

API ajoutée :

```python
model.save_checkpoint("checkpoint.pt")
model2 = ArchipelPhase2.load_checkpoint("checkpoint.pt")
```

Ce checkpoint sauvegarde :
- la config constructeur (`num_islands`, dims, `top_k`, seuils lifecycle, etc.) ;
- le `state_dict()` des paramètres entraînables ;
- les buffers runtime non persistants d'`OceanSpace` (`island_embeddings`, `interaction_counts`, `proximity_matrix`) ;
- les buffers runtime d'`IslandSpecialization` (`scores`, `counts`) ;
- les compteurs runtime du lifecycle.

Validation : `test_save_load.py` couvre le round-trip après spawn et la restauration explicite des buffers non persistants.

---

#### Tâche 0.2 — Assert-driven tests ✅ terminé
**Fichiers modifiés** : `test_archipel.py`, `test_lifecycle.py`, `test_specialization.py`  
**Fichier créé** : `test_assert_driven.py`

Tous les `return True` / `return False` ont été retirés des fonctions pytest.
Les tests utilisent maintenant uniquement les assertions existantes (`check(...)`
qui lève `AssertionError`) et ne déclenchent plus de `PytestReturnNotNoneWarning`.

Le garde-fou `test_assert_driven.py` scanne les tests par AST et échoue si une
fonction `test_*` retourne de nouveau un booléen.

---

#### Tâche 0.3 — Script d'entraînement + config ✅ terminé
**Fichier créé** : `archipel/train.py`  
**Fichier créé** : `archipel/configs/default.yaml`  
**Fichier créé** : `test_train_script.py`

Usage :

```bash
python archipel/train.py --config archipel/configs/default.yaml --seed 42
```

Le script :
- charge une config YAML (`model`, `training`, `data`) ;
- construit `ArchipelPhase2` et le `Courant` ;
- génère un dataset synthétique structuré déterministe ;
- lance `train_loop_lifecycle()` ;
- sauvegarde un checkpoint final `final.pt` dans `training.checkpoint_dir`.

Le smoke test CLI vérifie aussi la création du checkpoint final.

```yaml
# configs/default.yaml
model:
  num_islands: 4
  input_dim: 128
  hidden_dim: 64
  ocean_dim: 32
  top_k: 2
  max_islands: 8
  min_islands: 2
  coherence_variance_threshold: 0.3

training:
  lr: 0.001
  epochs: 50
  batch_size: 16
  log_every: 10
  save_every: 10
  checkpoint_dir: checkpoints
  device: cpu

data:
  num_samples: 2000
  input_dim: 128
  num_classes: 10
  signal_strength: 0.7  # pour données structurées
```

Validation : `test_train_script.py` couvre la config par défaut, les fonctions
importables (`load_config`, `build_model`, `build_dataloader`) et un lancement
CLI court avec sauvegarde de checkpoint.

---

### Niveau 1 — Qualité (3-4h, cette semaine)

#### Tâche 1.1 — Logger CSV
**Fichier à créer** : `archipel/src/archipel/training/logger.py`

Callback simple qui écrit un CSV à chaque `log_every` batch :

```python
class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "w")
        self.keys_written = False

    def log(self, entry: dict):
        if not self.keys_written:
            self.file.write(",".join(entry.keys()) + "\n")
            self.keys_written = True
        self.file.write(",".join(str(v) for v in entry.values()) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
```

Intégré dans `train_loop_lifecycle()` via un paramètre `logger=None`.

#### Tâche 1.2 — Test MNIST rapide
**Fichier à créer** : `test_mnist_quick.py`

Vérifie que le modèle converge sur un vrai dataset (10 classes, images 28×28) :
```python
from torchvision import datasets, transforms
from archipel.src.archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.13,), (0.31,))])
mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)

# Encode 28×28 → 128 via un petit encoder
encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU())

model = ArchipelPhase2(num_islands=4, input_dim=128, ...)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
courant = Courant(num_islands=4)

# Train 5 epochs, vérifie que loss diminue et accuracy augmente
```

---

### Niveau 2 — Recherche (selon disponibilité)

#### Étape A — Résonance Kuramoto dans l'Ocean

**Objectif** : Remplacer le routing par corrélation cosinus par une synchronisation de phase d'oscillateurs.

**Architecture cible** :
```
Ocean Kuramoto:
  - Chaque île i a une phase θ_i et une fréquence naturelle ω_i
  - Mise à jour: dθ_i/dt = ω_i + K * Σ_j A_ij * sin(θ_j - θ_i)
  - Couplage A_ij = proximité cosinus entre îles (top-k connexions)
  - Entrée x encode en phase cible φ_target via un petit projecteur

Routage:
  - Îles sélectionnées = top-k des plus proches de φ_target en différence de phase
  - Si différence < π/2 : forte synchronisation → forte contribution
```

**Fichiers à modifier/créer** :
- `ocean/kuramoto.py` — solveur Euler différentiable pour dθ/dt
- `current/router_kuramoto.py` — RoutingKuramoto (remplace HyperNetworkRouter)
- `tests/test_kuramoto.py`

---

#### Étape B — Apprentissage local + consolidation

**Objectif** : Remplacer la backprop globale par une mise à jour locale des îles + signal de récompense du Courant.

**Architecture cible** :
```
Training step:
  1. Forward pass local (chaque île traite son batch alloué)
  2. Loss locale par île (L2 distance à un "but" fourni par l'encoder)
  3. Hebb update locale: Δw_ij = η * a_i * a_j (règle de Hebb)
  4. Toutes les N étapes: consolidation (sleep) — mini-batch global, backprop légère sur le Courant + générateur de buts
```

**Fichiers à modifier/créer** :
- `islands/local_learning.py` — règle Hebb, mise à jour locale
- `training/loop_local.py` — boucle d'entraînement local
- `current/consolidation.py` — phase de sommeil / consolidation

---

#### Étape C — Valeur de Shapley pour la naissance/mort

**Objectif** : Remplacer le seuil de variance par une estimation de l'utilité marginale de chaque île.

**Algorithme** :
```python
# approximation de Shapley via échantillonnage Monte Carlo
def estimate_shapley_value(island_i, model, dataloader, n_samples=10):
    base_perf = evaluate(model, dataloader)
    values = []
    for _ in range(n_samples):
        # Masquer aléatoirement d'autres îles
        model.set_active([i for i in range(model.num_islands) if i != island_i])
        perf_without_i = evaluate(model, dataloader)
        values.append(base_perf - perf_without_i)
    return mean(values)
```

**Critères** :
- Naissance : when max cluster utility > threshold (nouveau îlot créé sur un cluster latent non couvert)
- Mort : when Shapley value < epsilon pendant N steps consécutifs

---

## État actuel (rappel rapide)
- **26/26 tests** passent
- **Entraînement viable** : loss −73.6% sur données structurées, diversité +0.31, 3 births/3 deaths
- **Fichiers à considérer comme stables** :
  - `base_island.py`, `ocean.py`, `router.py`, `courant.py`, `lifecycle.py`, `loop_lifecycle.py`
  - `models/mnist.py` — MNISTEncoder, MNISTArchipel (Niveau 1.2+)
  - `baselines/mlp.py` — MLPBaseline (Niveau 1.5)
  - `utils/specialization_matrix.py` — compute_specialization_matrix, specialization_score (Niveau 1.4)
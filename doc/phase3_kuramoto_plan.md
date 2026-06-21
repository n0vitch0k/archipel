# Archipel — Phase 3 : Résonance Kuramoto

**Objectif** : Remplacer le routing par similarité cosinus par un routing basé sur des oscillateurs couplés (modèle Kuramoto), où les îles synchronisent leurs phases quand elles traitent les mêmes entrées et se désynchronisent pour des entrées différentes.

---

## 1. Concept physique

### État actuel (Phase 2)
```
input → encoder → cos(θ_island, θ_input) → top-k → aggregation
```
Les îles ont des états latents (vecteurs dans l'espace de l'Océan). Le routing se fait par similarité cosinus entre l'état de l'île et la représentation de l'entrée.

### État cible (Phase 3)
```
input → encoder → φ(input) → cos(θ_i - φ(x)) → top-k → aggregation
                 ↓
         θ_i(t+1) = θ_i(t) + Δt · [ω_i + Σ_j K_ij sin(θ_j - θ_i)]
```

Chaque île est un **oscillateur de phase** θ_i ∈ [0, 2π). Les entrées sont projetées à une **phase cible** φ(x). Le routing se fait par alignement de phase. Les îles couplées synchronisent leurs phases quand elles co-traitent les mêmes entrées.

---

## 2. Architecture du nouveau routeur

### 2.1 Phase d'île

Chaque île a un état oscillatoire scalaire **θ_i ∈ [0, 2π)** et une fréquence naturelle **ω_i**.

```python
self.theta = nn.Parameter(torch.rand(num_islands) * 2 * math.pi)
self.omega = nn.Parameter(torch.randn(num_islands) * 0.1)
```

### 2.2 Phase d'entrée

L'entrée x (encodée en `ocean_dim`) est projetée en une phase cible :

```python
self.input_phase_net = nn.Sequential(
    nn.Linear(embedding_dim, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
# φ(x) = 2π * sigmoid(net(x))  → borné dans [0, 2π)
```

Alternative : projection en 2D puis `atan2` pour le mapping circulaire :
```python
self.input_phase_net = nn.Linear(embedding_dim, 2)  # (sin, cos)
# φ(x) = atan2(sin, cos)  →  angle sur le cercle
```

### 2.3 Score de routing

Alignement de phase entre île i et entrée x :
```python
alignment = cos(θ_i - φ(x))  # ∈ [-1, 1], 1 = parfait alignement
```

Même mécanisme que le routeur actuel :
- Top-k sélection parmi les meilleurs alignements
- ε-greedy exploration (remplacement aléatoire)
- Seuils apprenables par île
- Boost de spécialisation

### 2.4 Dynamique Kuramoto (mise à jour des phases)

Après chaque batch, les phases évoluent selon l'équation des oscillateurs couplés :

```python
def update_phases(self, coupling_matrix=None, dt=0.1):
    N = self.num_islands
    # Couplage par défaut : all-to-all uniforme
    K = coupling_matrix if coupling_matrix is not None else self.K
    
    # Δθ_i = dt * (ω_i + Σ_j K_ij * sin(θ_j - θ_i))
    theta = self.theta.detach()  # shape (num_islands,)
    dtheta = self.omega.clone()  # fréquence naturelle
    
    for i in range(N):
        for j in range(N):
            if i != j:
                dtheta[i] += K[i,j] * torch.sin(theta[j] - theta[i])
    
    # Mise à jour Euler
    self.theta.data = (theta + dt * dtheta) % (2 * math.pi)
```

Pour l'efficacité, version vectorisée :
```python
def update_phases_vectorized(self, dt=0.1):
    theta = self.theta.detach()  # (N,)
    # Matrice des différences de phase : θ_j - θ_i
    diff = theta.unsqueeze(0) - theta.unsqueeze(1)  # (N, N)
    # Couplage : sin(θ_j - θ_i) * K_ij
    coupling = self.K * torch.sin(diff)  # (N, N)
    # Σ_j sur chaque ligne, excluant diag (sin(0) = 0 anyway)
    dtheta = self.omega + coupling.sum(dim=1)  # (N,)
    # Euler
    self.theta.data = (theta + dt * dtheta) % (2 * math.pi)
```

### 2.5 Matrice de couplage

Deux options :

**Option A — Couplage uniforme** K_ij = K (un scalaire apprenable)
- Simple, peu de paramètres
- Toutes les îles s'influencent également

**Option B — Couplage adaptatif par co-activation** 
```python
K_ij = K_base + K_boost * co_activation(i, j)
```
où `co_activation(i, j)` est l'EMA de la fréquence à laquelle i et j sont sélectionnés ensemble.
- Les îles qui co-traitent se synchronisent plus fortement
- Émerge naturellement en clusters

**Option C — Couplage appris** K_ij est un paramètre apprenable complet
- Plus flexible mais plus de paramètres
- Peut capturer des relations asymétriques

**Recommandation** : Commencer par Option A (uniforme), puis Option B (adaptatif) si nécessaire.

---

## 3. Fichiers à modifier/créer

### NOUVEAU : `archipel/src/archipel/current/kuramoto.py`
- `KuramotoIslandRouter(nn.Module)` — le routeur Kuramoto
  - `__init__(embedding_dim, num_islands, top_k, ...)` 
  - `forward(input_repr, island_states=None, specialization_boost=None, ...)`
  - `update_phases()` — pas Kuramoto post-batch
  - `compute_phase_alignment(input_phases)` — cos(θ_i - φ(x))
  - `input_to_phase(input_repr)` — projection entrée → phase
  - `get_phase_statistics()` — métriques de synchronisation
  - `resize(num_islands)` — compatibilité lifecycle
  - `set_top_k(top_k)` — curriculum

### MODIFIÉ : `archipel/src/archipel/training/loop_lifecycle.py`
- Ajout de `ArchipelPhase3(ArchipelPhase2)`
- Paramètre `routing_mode: str = "cosine" | "kuramoto"`
- Si `kuramoto` : utiliser `KuramotoIslandRouter` au lieu de `HyperNetworkRouter`
- Appeler `update_phases()` après chaque batch
- Logger les métriques Kuramoto (sync, locking, θ_i)

### MODIFIÉ : `validate_niveau1415.py` / `_quick.py`
- Paramètre `--routing cosine|kuramoto`
- Validation dans les deux modes

### NOUVEAU : `test_kuramoto.py`
- Tests unitaires du routeur Kuramoto
- Tests de synchronisation
- Tests de résilience (lifecycle, resize)

### MODIFIÉ : `modal_validate.py`
- Mode `--routing kuramoto` pour validation cloud

---

## 4. Plan de travail détaillé

### Sprint 1 — Routeur Kuramoto (fichier unique)

1. [ ] Créer `archipel/src/archipel/current/kuramoto.py` avec `KuramotoIslandRouter`
2. [ ] Implémenter la projection entrée → phase (MLP → atan2)
3. [ ] Implémenter le score d'alignement cos(θ_i - φ(x)) + top-k + ε-greedy
4. [ ] Implémenter la mise à jour Kuramoto vectorisée (Euler)
5. [ ] Implémenter `resize()` pour lifecycle
6. [ ] Tests unitaires : mise à jour phase, alignement, convergence vers sync

### Sprint 2 — Intégration dans Archipel

7. [ ] Créer `ArchipelPhase3(ArchipelPhase2)` avec routage paramétrable
8. [ ] Ajouter `update_phases()` dans la boucle d'entraînement
9. [ ] Logger phase θ_i, synchronisation, couplage K_ij
10. [ ] Tests d'intégration (MNIST 5 epochs)

### Sprint 3 — Validation et comparaison

11. [ ] Lancer MNIST 50 epochs en mode Kuramoto (3 seeds)
12. [ ] Comparer accuracy + coverage avec le mode cosinus
13. [ ] Analyser la synchronisation : clusters de phase par classe
14. [ ] Documenter les résultats dans PROJET_ARCHIPEL_PLAN.md

---

## 5. Métriques de synchronisation

Pour mesurer l'état de synchronisation du système :

```python
def compute_sync_metrics(theta):
    # Ordre de Kuramoto : R = |(1/N) Σ e^(iθ_j)|
    # R ∈ [0, 1], 1 = synchronisation parfaite, 0 = désordre total
    complex_sum = torch.exp(1j * theta).sum()
    order_param = abs(complex_sum) / N
    
    # Cluster de phase : variance circulaire
    circular_var = 1 - order_param
    
    # Verrouillage de phase : std des différences de phase
    phase_diffs = [(theta[i] - theta[j]) % (2π) for i, j in pairs]
    locking = 1 - circular_std(phase_diffs)
    
    return {"order_parameter": R, "circular_variance": var, "phase_locking": locking}
```

Quand R → 1 : toutes les îles sont synchronisées (peu de spécialisation)
Quand R → 0 : îles désynchronisées (spécialisation potentielle)
Idéalement : sous-groupes synchronisés en clusters locaux (R_local élevé, R_global modéré)

---

## 6. Risques et questions ouvertes

### Risques
1. **Synchronisation totale** : K trop fort → toutes les îles en phase → pas de spécialisation → solution : couplage adaptatif inhibiteur
2. **Désynchronisation totale** : K trop faible → les îles ignorent les voisines → pas de cluster → solution : K minimum garantissant sync locale
3. **Instabilité numérique** : θ non borné → solution : wrap dans [0, 2π) après chaque update
4. **Compatibilité Courant** : les λ Courant doivent rester efficaces avec des phases

### Questions ouvertes
1. **Δt apprenable ?** Un pas de temps différent par île ou global ?
2. **Phase scalaire vs vectorielle ?** 1D (cercle) vs nD (tore) → plus de dimensions = plus de clusters possibles
3. **Couplage asymétrique ?** K_ij ≠ K_ji possible (influence directionnelle)
4. **Bruit de phase ?** Ajouter du bruit pour l'exploration (comme ε-greedy)

### Réponse recommandée
Commencer simple : phase 1D, couplage uniforme K, Δt=0.1 global, sans bruit. Ajouter de la complexité si les résultats ne sont pas concluants.

---

## 7. Critères de succès

| Critère | Seuil | Priorité |
|---------|-------|----------|
| Accuracy MNIST ≥ 95% | ≥ routing cosinus | Haute |
| spec_coverage ≥ 2 | ≥ 2/3 seeds | Haute |
| Synchronisation par classe | Clusters visibles | Moyenne |
| Temps d'entraînement | ×1.5 max | Moyenne |
| Tests unitaires | 100% passent | Haute |

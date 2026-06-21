# Diagnostic Spécialisation — Problème de Bootstrap avec top_k=1

**Date** : 2026-05-27  
**Contexte** : Investigation sur l'échec du Niveau 1.4 (spec_score=0.0033 < 0.3 requis)

---

## Symptômes Observés

### Résultats de validation (Niveau 1.4)
```
Test Acc: 0.9798 | Loss: 1.9463
Births: 22 | Deaths: 22
Specialization Score: 0.0033 ❌ (objectif > 0.3)
```

### Analyse du training_log.csv (6459 entrées)
- `spec_max` monte rapidement à 0.99+ (presque 1.0) dès les premiers batchs
- `lambda_coherence` reste à 1.0 (plafond) tout au long de l'entraînement
- `coherence` oscille autour de 1.0 (au-dessus du `coherence_target=0.5`)
- `diversity` monte progressivement mais ne dépasse pas 0.35
- `spec_mean` monte à 0.99+ indiquant que les îles sont "confiantes" mais dans la MÊME direction

---

## Root Cause : Bootstrap de Spécialisation Impossible avec top_k=1

### Mécanisme détaillé

**Le problème fondamental** : Avec `top_k=1`, seule UNE île est active par batch. Le système de spécialisation dépend du routage pour distinguer quelles îles apprennent quelles classes.

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│   Input     │────▶│  Encoder    │────▶│  Représentation  │
│   x         │     │  (CNN)      │     │  r_x (128,)      │
└─────────────┘     └─────────────┘     └──────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
         ┌──────────────────┐      ┌──────────────────┐     ┌──────────────────┐
         │  Île 0           │      │  Île 1           │     │  Île 2, 3        │
         │  embedding h_0   │      │  embedding h_1   │     │  (potentiellement inactives)
         └──────────────────┘      └──────────────────┘     └──────────────────┘
                    ▲                        ▲
                    └────────────────────────┼────────────────────────┘
                                             │
                                    Cosine Similarity
                                             │
                                     top_k=1 sélection
                                             ▼
                              ┌──────────────────────────┐
                              │  Une seule île active  │
                              │  (celle avec meilleure  │
                              │  corrélation)          │
                              └──────────────────────────┘
```

### Causalité du problème

1. **Phase initiale** (batches 0-10) :
   - Les embeddings d'îles sont aléatoires (initialisation orthogonale dans `init_island_states`)
   - Une île (ex: île 0) a un léger avantage de corrélation
   - `top_k=1` sélectionne systématiquement cette île
   - Elle commence à accumuler les scores de spécialisation

2. **Boucle de feedback positive** :
   - L'île active améliore ses embeddings (apprentissage)
   - Son embedding devient plus "représentatif" des inputs
   - Le routage la sélectionne encore plus fréquemment
   - Les autres îles restent inactives, n'apprennent RIEN

3. **Effondrement de la spécialisation** :
   - `specialization_score` calcule l'entropie des prédictions par île
   - Si 4 îles sur la classe 1, entropie = log(10)/10 × 4 = log(10) (maximale)
   - Score = 1 - entropie/max = 0.0 (pas spécialisé)
   - Même si `spec_max=0.99`, si TOUTES les îles ont la même classe, le score est nul

---

## Observations Techniques Détailées

### 1. Parms Courant actuels (courant.py lignes 31-41)

```python
class Courant(nn.Module):
    def __init__(
        self,
        lambda_coherence_init: float = 0.1,   # ✅ Correct
        lambda_diversity_init: float = 0.2,    # ✅ Correct (ajusté 0.1→0.2)
        diversity_target: float = 0.25,        # ✅ Correct (ajusté 0.15→0.25)
        coherence_target: float = 0.5,         # ✅ Correct (ajusté 0.1→0.5)
    )
```

**Problème** : Les adaptations de λ ne sont pas efficaces car :
- `coherence ≈ 1.0` > `coherence_target * 2.0 = 1.0` → λ_coh monte mais est déjà au max
- La cohérence élevée signifie que les îles AGREE sur leur embedding, ce qui empêche la divergence

### 2. Router - Epsilon-Greedy (router.py lignes 111-122)

```python
def top_k_selection_with_noise(...):
    # Epsilon-greedy: 10% de chance de sélection aléatoire
    epsilon_k = self.epsilon * self.epsilon_scale.item()  # 0.1 * 1.0 = 0.1
    replace_mask = torch.rand(batch_size) < epsilon_k
```

**Problème** : 10% d'exploration est insuffisant pour un bootstrap de spécialisation.  
Avec 128 batch_size et 4 îles, il faut ~320 batches pour que chaque île soit sélectionnée.

### 3. Specialization Matrix (specialization_matrix.py)

```python
def specialization_score(matrix: torch.Tensor) -> float:
    total = matrix.sum(dim=1, keepdim=True).clamp(min=1)
    probs = matrix / total  # Distribution [num_islands, num_classes]
    entropy = -(probs * probs.log()).sum(dim=1)  # Entropie par île
    max_entropy = log(num_classes)  # log(10) ≈ 2.302
    avg_entropy = entropy.mean()
    return 1.0 - (avg_entropy / max_entropy)  # Score dans [0, 1]
```

**Interprétation** :
- Si chaque île prédit UNE seule classe différente → entropie = 0 → score = 1.0
- Si chaque île prédit uniformément sur 10 classes → entropie = log(10) → score = 0.0
- Si 4 îles prédisent toutes la classe 1 → entropie = log(10) → score = 0.0

---

## Solutions Envisagées

### Solution A : Augmenter l'exploration (epsilon)
Modifier `router.py` :
```python
epsilon_init: float = 0.1 → 0.3  # 30% d'exploration initiale
# Plus tard, modulation adaptative pourrait réduire epsilon
```

### Solution B : Warm-up top_k
Approche temporelle :
```python
# Phase 1 (batches 0-100) : top_k=2 pour exploration
# Phase 2 (batches 100+) : top_k=1 pour spécialisation
if batch_idx < 100:
    self.top_k = 2
else:
    self.top_k = 1
```

### Solution C : Modulation epsilon adaptative forte
Dans `courant.py`, intensifier la modulation :
```python
def get_epsilon_modulation(self) -> float:
    if mean_diversity < diversity_target * 0.2:  # Plus agressif
        return 2.0  # Au lieu de 1.5
```

### Solution D : Perturbation des embeddings d'îles
Initialement, les îles devraient avoir des embeddings plus distincts :
```python
# init_island_states dans router.py - utiliser plus de directions distinctes
for j in range(min(embedding_dim, 16)):  # Plus de directions
    states[i, j] += torch.sin(torch.tensor(angle + j * 0.25)) * 0.5  # Plus forte amplitude
```

---

## Statistiques Clés

| Métrique | Valeur actuelle | Cible | Statut |
|----------|----------------|-------|--------|
| spec_score | 0.0033 | > 0.3 | ❌ ÉCHOUÉ |
| spec_max | 0.99+ | - | Délir recommandation |
| λ_coherence | 1.0 | 0.01-2.0 | Bloqué haut |
| λ_diversity | 1.0 | 0.1-3.0 | Correct |
| top_k | 1 | 1 | Correct mais bootstrap impossible |
| coherence | ~1.0 | < 0.5 | Élevé (mauvais signe) |
| diversity | ~0.3 | > 0.25 | Correct mais tardif |

---

## Prochaines Actions

1. [ ] **Tester Solution A** : epsilon=0.3, validation Niveau 1.4
2. [ ] **Mesurer le temps d'exploration** nécessaire pour bootstrap spécialisation
3. [ ] **Corriger le mécanisme de modulation ε** pour réagir plus vite à la faible diversité
4. [ ] **Documenter le warm-up top_k** dans le plan projet
5. [ ] **Créer test de bootstrap spécialisation** pour validation continue

---

## Fichiers Impactés

- `archipel/src/archipel/current/router.py` - epsilon et top_k
- `archipel/src/archipel/current/courant.py` - get_epsilon_modulation
- `archipel/src/archipel/training/loop_lifecycle.py` - warm-up top_k
- `test_validation_niveau_1_4_1_5.py` - scénario de test
- `archipel/configs/default.yaml` - paramètres par défaut

---

## Historique des Modifications (2026-05-22)

Paramètres Courant ajustés précédemment (mais insuffisants) :
```python
# archipel/src/archipel/current/courant.py
lambda_diversity_init: 0.1 → 0.2
diversity_target: 0.15 → 0.25
# Max values : lambda_diversity max 3.0, lambda_coherence max 2.0
```

Ces ajustements ont amélioré la diversité mais n'ont pas résolu le bootstrap spécialisation.
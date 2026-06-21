# Dynamic Top-k Curriculum pour le Bootstrap de Spécialisation Archipel

**Date** : 2026-06-17  
**Statut** : V1 implémentée, tests unitaires + intégration passants, smoke training OK, diagnostic MNIST dynamique documenté  
**Niveau associé** : Niveau 1.8 — Exploration contrôlée du bootstrap de spécialisation  
**Problème ciblé** : `top_k=1` est nécessaire en régime final, mais il provoque un winner-take-all trop tôt si aucune exploration initiale n’est forcée.

---

## 1. Résumé exécutif

Archipel est arrivé à une conclusion importante :

```text
Le problème n’est pas de choisir un top_k fixe.
Le problème est de contrôler la trajectoire de compétition entre les îles.
```

Autrement dit, `top_k` ne doit pas être pensé comme un hyperparamètre statique, mais comme un **curriculum de compétition** :

```text
début de training :
    k élevé → exploration → plusieurs îles reçoivent des gradients

milieu de training :
    k diminue → compétition progressive → les îles commencent à se distinguer

fin de training :
    k=1 → spécialisation stricte → un seul îlot par échantillon
```

Cette trajectoire répond au dilemme suivant :

```text
k trop faible trop tôt
→ une île monopolise le routage
→ effet Matthieu
→ les autres îles restent inactives
→ spécialisation impossible

k trop élevé trop longtemps
→ plusieurs îles apprennent les mêmes échantillons
→ redondance
→ convergence des îles
→ spécialisation impossible
```

La solution retenue est donc :

```text
V1 : décroissance déterministe k=3 → k=2 → k=1 sur 300 steps
V2 : contrôle adaptatif lent basé sur routing_usage_entropy + min_usage_ratio + hystérésis
V3 : éventuellement reshuffling / pénalité fonctionnelle si spec_coverage reste faible
```

---

## 2. État d’implémentation V1

### 2.1 Fichiers modifiés

La V1 est maintenant intégrée dans la boucle de training :

- `archipel/src/archipel/current/topk_curriculum.py` : nouveau module `TopKCurriculum` + `RoutingUsageTracker`.
- `archipel/src/archipel/current/router.py` : ajout de `router.set_top_k(top_k)`.
- `archipel/src/archipel/training/loop_lifecycle.py` : application du curriculum avant chaque forward + logs routing/spécialisation.
- `archipel/src/archipel/islands/specialization.py` : ajout de `get_specialization_summary()` et `spec_coverage`.
- `archipel/src/archipel/training/csv_logger.py` : nouveaux champs CSV pour routing et spécialisation.
- `archipel/configs/default.yaml` : activation du curriculum par défaut.
- `archipel/train.py` : construction du curriculum depuis la config.
- `test_topk_curriculum.py` : tests unitaires et test d’intégration pour schedule, EMA routing, dead islands, effective top-k, spec coverage et logs qualitatifs.
- `scripts/diagnose_dynamic_mnist.py` : script de diagnostic MNIST dynamique dédié.

### 2.2 Comportement validé

Tests courts :

```text
uv run python -m pytest test_topk_curriculum.py -q
7 passed

uv run python -m pytest test_topk_curriculum.py test_train_script.py test_specialization.py -q
17 passed
```

Smoke training synthétique :

```text
uv run python archipel/train.py --config archipel/configs/quick_test.yaml --quiet --csv logs/dynamic_topk_quick.csv
Training complete
epochs: 2
num_logs: 26
num_islands: 2
final_loss: 4.048163890838623
```

Smoke training avec la config par défaut en mode court avant restauration de `training.epochs=50` :

```text
uv run python archipel/train.py --config archipel/configs/default.yaml --quiet --csv logs/dynamic_topk_default.csv
epochs: 1
num_logs: 125
num_islands: 4
final_loss: 1.5540145635604858
```

Diagnostic MNIST dynamique dédié :

```text
uv run python scripts/diagnose_dynamic_mnist.py \
  --epochs 3 \
  --batch-size 64 \
  --limit 4000 \
  --k-init 3 \
  --k-final 1 \
  --warmup-steps 60 \
  --freeze-step 120
```

Résultat observé :

```text
entries=189
unique_top_k=[1, 2, 3]
first_top_k=3
last_top_k=1
first_spec_coverage=1
max_spec_coverage=1
last_spec_coverage=0
max_dead_island_count=0
last_routing_usage_entropy≈0.914
last_min_usage_ratio≈0.112
last_effective_top_k=1.0
```

Interprétation : le curriculum dynamique fonctionne sur MNIST ; la trajectoire `3 → 2 → 1` est bien produite. Le diagnostic reste court pour conclure sur `spec_coverage`, mais les logs qualitatifs montrent le début du comportement attendu et le routing reste sain (`dead_island_count=0`).

Diagnostic MNIST rapide standard :

```text
uv run python test_mnist_quick.py --epochs 2 --batch-size 64
accuracy_final=0.9557
loss_improvement_pct=15.3
PASSED=True
```

Exemple de métriques observées au premier batch du run dynamique :

```text
current_top_k=3
scheduled_top_k=3
routing_usage_entropy≈0.999
min_usage_ratio≈0.229
dead_island_count=0
effective_top_k≈2.875
spec_coverage=0
qualitative_log=top-k curriculum active: k=3 | routing_usage_entropy=0.999: exploration encore large | routing sain: toutes les îles reçoivent de l'usage
```

Exemple de métriques observées en fin de run dynamique court :

```text
current_top_k=2
scheduled_top_k=2
routing_usage_entropy=0.9754979014396667
min_usage_ratio=0.19531920552253723
dead_island_count=0
effective_top_k=2.0
spec_coverage=0
```

Diagnostic MNIST rapide :

```text
uv run python test_mnist_quick.py --epochs 2 --batch-size 64
epochs: 2
loss_initial: 2.4120
loss_final: 1.9453
loss_improvement_pct: 19.3
accuracy_before: 0.0864
accuracy_final: 0.9462
num_islands_final: 4
num_births: 5
num_deaths: 5
num_logs: 324
PASSED: True
```

Diagnostic MNIST dynamique court avec curriculum injecté manuellement :

```text
MNIST dynamic top-k diagnostic
first_log:
  current_top_k=3
  routing_usage_entropy=0.9988693594932556
  min_usage_ratio=0.22604165971279144
  dead_island_count=0
  effective_top_k=2.96875
  spec_coverage=1

last_log:
  current_top_k=2
  routing_usage_entropy=0.9128360748291016
  min_usage_ratio=0.13351888954639435
  dead_island_count=0
  effective_top_k=1.9375
  spec_coverage=0
  spec_std=0.14089100062847137
```

Interprétation : le curriculum dynamique est bien branché et les métriques routing sont produites. Le run MNIST dynamique court est encore trop court pour conclure sur `spec_coverage` : il montre la trajectoire `k=3 → k=2`, mais la spécialisation fonctionnelle demande plus de steps ou un run avec `freeze_step` plus tardif.

---

## 3. Contexte Archipel

### 2.1 Architecture concernée

Le mécanisme touche principalement :

```text
archipel/src/archipel/current/router.py
archipel/src/archipel/current/courant.py
archipel/src/archipel/training/loop_lifecycle.py
archipel/src/archipel/islands/specialization.py
archipel/src/archipel/ocean/ocean.py
archipel/configs/default.yaml
```

### 2.2 Mécanisme actuel

Dans la boucle Phase 2 :

```text
Input x
  │
  ▼
Encoder
  │
  ▼
Représentation r_x
  │
  ▼
Islands embeddings h_i
  │
  ▼
Router par corrélation cosinus
  │
  ▼
routing_weights w ∈ R^(batch × num_islands)
  │
  ▼
Agrégation pondérée
  │
  ▼
Task head → prédiction globale
  │
  ▼
Loss + Courant + Lifecycle + Spécialisation tracking
```

Le router sélectionne les îles avec un mécanisme top-k :

```python
top_k = 1
```

C’est correct en régime final, car la spécialisation exige que chaque échantillon soit assigné à une seule île. Mais au début, avec des embeddings encore bruités, `top_k=1` amplifie le moindre avantage aléatoire.

---

## 4. Diagnostic du problème

### 3.1 Observation expérimentale

Validation Niveau 1.4/1.5 :

```text
spec_score ≈ 0.0033
spec_max ≈ 0.99
```

Interprétation :

```text
spec_max élevé :
    chaque île semble confiante / concentrée

spec_score proche de 0 :
    toutes les îles sont concentrées sur la même classe

Conclusion :
    les îles ne sont pas spécialisées sur des classes différentes.
    Elles sont redondantes.
```

### 3.2 Cause racine

Avec `top_k=1` immédiat :

```text
1. Une île a un léger avantage de corrélation au début.
2. Elle est sélectionnée plus souvent.
3. Elle reçoit plus de gradients.
4. Son embedding devient encore plus attractif.
5. Le router la sélectionne encore plus souvent.
6. Les autres îles restent inactives.
7. Le système converge vers une seule direction fonctionnelle.
```

C’est un effet Matthieu :

```text
celle qui reçoit apprend ;
celle qui apprend reçoit encore plus.
```

### 3.3 Pourquoi `top_k=2` permanent ne suffit pas

Avec `top_k=2` permanent :

```text
plusieurs îles apprennent les mêmes échantillons
```

Cela stabilise l’exploration mais empêche la spécialisation stricte.

Donc :

```text
top_k=1 fixe → monopoly initial
top_k=2 fixe → redondance permanente
```

La bonne solution est une trajectoire :

```text
top_k=3 au début
top_k=2 ensuite
top_k=1 à la fin
```

---

## 5. Théorie : `k` comme curriculum de compétition

### 4.1 Définition

Dans Archipel, `k` contrôle la pression de compétition entre îles.

```text
k élevé :
    plusieurs îles reçoivent un gradient
    exploration forte
    risque de redondance

k bas :
    peu d’îles reçoivent un gradient
    compétition forte
    risque de winner-take-all si trop tôt
```

Donc `k` est une variable de contrôle du compromis :

```text
exploration ↔ spécialisation
```

### 4.2 Formulation du dilemme

```text
k trop faible trop tôt :
    effet Matthieu
    monopolisation
    îles mortes

k trop élevé trop longtemps :
    redondance
    gradients dilués
    spécialisation faible
```

### 4.3 Objectif de la trajectoire

La trajectoire idéale doit satisfaire :

```text
1. Activer toutes les îles au début.
2. Éviter qu’une île monopolise le bootstrap.
3. Réduire progressivement la compétition.
4. Garantir k=1 en fin d’entraînement.
5. Produire une spécialisation fonctionnelle, pas seulement géométrique.
```

---

## 6. Signaux de diagnostic

Le contrôleur ne doit pas se baser uniquement sur `top_k`. Il doit observer l’état réel du système.

### 5.1 `routing_usage_ema`

Définition :

```python
u_t = beta * u_{t-1} + (1 - beta) * mean_batch(routing_weights)
```

Avec :

```text
u_t ∈ R^(num_islands)
beta = 0.90 au début, puis 0.95
```

Initialisation recommandée :

```python
u_0 = ones(num_islands) / num_islands
```

Pourquoi une EMA ?

```text
- plus légère qu’une fenêtre glissante ;
- plus stable qu’une moyenne par batch ;
- sensible aux changements lents ;
- adaptée à un contrôleur en ligne.
```

### 5.2 `routing_usage_entropy`

Définition :

```python
H = -sum(u_i * log(u_i + eps)) / log(num_islands)
```

Propriétés :

```text
H = 0.0 → une seule île utilisée
H = 1.0 → usage parfaitement équilibré
```

Attention :

```text
L’entropie par batch n’est pas le bon signal.
```

Avec `top_k=1`, l’entropie par batch est toujours basse.

Le bon signal est :

```text
l’entropie de l’usage cumulé des îles sur une fenêtre / EMA.
```

### 5.3 `min_usage_ratio`

Définition :

```python
min_usage_ratio = min(u_i)
```

Pourquoi c’est indispensable :

```text
L’entropie peut masquer une île morte.
```

Exemple avec 4 îles :

```text
u = [0.00, 0.33, 0.33, 0.33]
```

L’entropie peut rester correcte, mais une île est morte.

Donc :

```text
routing_usage_entropy = santé globale
min_usage_ratio = santé des minoritaires
```

### 5.4 `dead_island_count`

Définition :

```python
dead_threshold = 0.05
dead_island_count = count(u_i < dead_threshold)
```

Interprétation :

```text
dead_island_count = 0 :
    toutes les îles reçoivent au moins 5% de l’usage

dead_island_count > 0 :
    une ou plusieurs îles sont exclues
```

Ce signal est crucial pour détecter un effet Matthieu partiel avant l’effondrement total.

### 5.5 `effective_top_k`

Définition :

```python
effective_top_k = mean_batch(count(routing_weights > threshold, dim=1))
```

À ne pas confondre avec `current_top_k`.

```text
current_top_k :
    valeur demandée par le contrôleur

effective_top_k :
    nombre réel d’îles activées par batch
```

Si `current_top_k=3` mais `effective_top_k=1`, le contrôleur ne fonctionne pas ou le routage est déjà monopolisé.

### 5.6 `ocean_pairwise_similarity`

Définition :

```python
S = moyenne des cosinus hors diagonale entre island embeddings
```

Utilisation :

```text
Signal secondaire.
Ne pas l’utiliser seule.
```

Raison :

```text
Deux îles peuvent être différentes dans l’espace mais inutiles.
```

Donc la similarité doit être couplée à des signaux fonctionnels.

### 5.7 `spec_score`

Définition actuelle :

```python
score = 1 - average_entropy_per_island / log(num_classes)
```

Interprétation :

```text
1.0 :
    chaque île prédit une seule classe

0.0 :
    chaque île prédit uniformément ou toutes les îles prédisent la même classe
```

Limite :

```text
spec_score ne distingue pas toujours spécialisation utile et collapse sur une seule classe.
```

Donc il faut le compléter par `spec_std` et `spec_coverage`.

### 5.8 `spec_max`

Définition :

```python
spec_max = max(score de spécialisation par île)
```

Interprétation :

```text
spec_max élevé :
    chaque île semble concentrée

Mais si toutes les îles sont concentrées sur la même classe :
    spec_score reste faible
```

Donc :

```text
spec_max seul est trompeur.
```

### 5.9 `spec_std`

Définition :

```python
spec_std = std(per_island_specialization_strength)
```

Interprétation :

```text
spec_std faible + spec_max élevé :
    toutes les îles sont fortes de la même façon
    risque de collapse

spec_std modérée/élevée :
    certaines îles sont plus spécialisées que d’autres
```

### 5.10 `spec_coverage`

Définition :

```text
Pour chaque île :
    dominant_class = argmax_c scores[i, c]
    purity = max_c scores[i, c] / sum_c scores[i, c]

Une île compte comme spécialisée si :
    purity >= purity_threshold
    et max_score >= score_threshold

spec_coverage =
    nombre de classes différentes couvertes par les îles spécialisées
```

Exemple bon :

```text
île 0 → classe 3
île 1 → classe 7
île 2 → classe 8
île 3 → classe 1

spec_coverage = 4
```

Exemple mauvais :

```text
île 0 → classe 3
île 1 → classe 3
île 2 → classe 3
île 3 → classe 3

spec_coverage = 1
```

Pourquoi c’est important :

```text
La vraie spécialisation utile est une couverture fonctionnelle des classes.
```

---

## 7. Contrôleur V1 : décroissance déterministe

### 6.1 Objectif

Valider l’hypothèse principale :

```text
Un curriculum simple 3 → 2 → 1 suffit-il à résoudre le bootstrap ?
```

Avant d’ajouter un contrôleur adaptatif, il faut tester la version la plus simple.

### 6.2 Paramètres recommandés

```python
k_init = 3
k_final = 1
warmup_steps = 300
freeze_step = 1000
```

Pour un entraînement court ou un smoke test :

```python
warmup_steps = min(300, max(50, int(0.2 * total_steps)))
freeze_step = max(warmup_steps, int(0.8 * total_steps))
```

### 6.3 Formule

```python
progress = min(step / warmup_steps, 1.0)
k = ceil(k_final + (k_init - k_final) * (1 - progress))
k = clamp(k, k_final, k_init)
```

Pour `k_init=3`, `k_final=1`, `warmup_steps=300` :

```text
steps 0-99    → k=3
steps 100-199 → k=2
steps 200+    → k=1
```

Pourquoi `ceil` plutôt que `round` ?

```text
ceil garde l’exploration un peu plus longtemps.
C’est plus prudent pour éviter le winner-take-all.
```

### 6.4 Garantie finale

Après `freeze_step` :

```python
k = 1
```

Et :

```text
k ne peut plus remonter
```

C’est essentiel pour garantir la spécialisation finale.

---

## 8. Contrôleur V2 : adaptatif lent avec hystérésis

### 7.1 Principe

La V2 ne doit pas remplacer la V1. Elle doit seulement corriger les cas où la trajectoire déterministe ne suffit pas.

```text
V1 :
    curriculum déterministe

V2 :
    curriculum + surveillance des îles mortes + hystérésis
```

### 7.2 Conditions d’augmentation de `k`

Augmenter `k` si :

```python
usage_entropy < low_threshold
```

ou :

```python
min_usage_ratio < dead_threshold
```

Exemple :

```python
low_threshold = 0.35
dead_threshold = 0.05
```

Interprétation :

```text
Si l’usage devient trop concentré :
    augmenter k
    permettre à plus d’îles de recevoir des gradients
```

### 7.3 Conditions de diminution de `k`

Diminuer `k` si :

```python
usage_entropy > high_threshold
and ocean_pairwise_similarity > sim_high
```

Exemple :

```python
high_threshold = 0.75
sim_high = 0.85
```

Interprétation :

```text
Si l’usage est bien réparti
et que les îles sont encore trop similaires :
    on peut augmenter la compétition
    donc diminuer k
```

### 7.4 Hystérésis

Il faut éviter les oscillations :

```text
k=1 → monopolisation → k=2
k=2 → usage équilibré → k=1
k=1 → monopolisation → k=2
...
```

Donc :

```text
low_threshold < high_threshold
```

Exemple :

```python
low_threshold = 0.35
high_threshold = 0.75
```

Optionnel :

```python
k_cooldown_steps = 25
```

C’est-à-dire :

```text
ne pas changer k plus d’une fois toutes les 25 steps
```

### 7.5 Freeze

Après `freeze_step` :

```python
k = 1
```

Même si `dead_island_count > 0`.

Raison :

```text
Le but final est la spécialisation stricte.
L’adaptatif sert au bootstrap, pas à maintenir l’exploration indéfiniment.
```

---

## 9. Pseudocode de référence

### 8.1 Classe `TopKCurriculum`

```python
class TopKCurriculum:
    def __init__(
        self,
        k_init=3,
        k_final=1,
        warmup_steps=300,
        freeze_step=None,
        usage_entropy_low=0.35,
        usage_entropy_high=0.75,
        dead_threshold=0.05,
        similarity_high=0.85,
        adaptive=False,
        cooldown_steps=25,
    ):
        self.k_init = k_init
        self.k_final = k_final
        self.warmup_steps = warmup_steps
        self.freeze_step = freeze_step
        self.usage_entropy_low = usage_entropy_low
        self.usage_entropy_high = usage_entropy_high
        self.dead_threshold = dead_threshold
        self.similarity_high = similarity_high
        self.adaptive = adaptive
        self.cooldown_steps = cooldown_steps
        self.last_change_step = -cooldown_steps

    def scheduled_k(self, step):
        if self.warmup_steps <= 0:
            return self.k_final

        progress = min(step / self.warmup_steps, 1.0)
        k = math.ceil(
            self.k_final + (self.k_init - self.k_final) * (1 - progress)
        )
        return int(max(self.k_final, min(self.k_init, k)))

    def step(
        self,
        step,
        num_islands,
        usage_entropy=None,
        min_usage=None,
        similarity=None,
    ):
        k = self.scheduled_k(step)

        if self.freeze_step is not None and step >= self.freeze_step:
            return self.k_final

        if not self.adaptive:
            return k

        if step < self.warmup_steps:
            return k

        if step - self.last_change_step < self.cooldown_steps:
            return k

        should_increase = (
            usage_entropy is not None and usage_entropy < self.usage_entropy_low
        ) or (
            min_usage is not None and min_usage < self.dead_threshold
        )

        should_decrease = (
            usage_entropy is not None and usage_entropy > self.usage_entropy_high
            and similarity is not None and similarity > self.similarity_high
        )

        if should_increase:
            k = min(self.k_init, k + 1)
            self.last_change_step = step

        elif should_decrease:
            k = max(self.k_final, k - 1)
            self.last_change_step = step

        return int(k)
```

### 8.2 EMA d’usage

```python
def update_routing_usage_ema(prev, batch_usage, beta):
    return beta * prev + (1 - beta) * batch_usage
```

Initialisation :

```python
routing_usage_ema = torch.ones(num_islands) / num_islands
```

### 8.3 Entropie normalisée

```python
def normalized_entropy(p, eps=1e-8):
    p = p.clamp(min=eps)
    p = p / p.sum()
    entropy = -(p * torch.log(p)).sum()
    return entropy / torch.log(torch.tensor(p.numel(), dtype=torch.float32))
```

### 8.4 Similarité moyenne des îles

```python
def mean_pairwise_cosine(states):
    states = states / states.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = states @ states.T
    n = sim.shape[0]
    if n <= 1:
        return torch.tensor(0.0)
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim[mask].mean()
```

### 8.5 Couverture de spécialisation

```python
def specialization_coverage(scores, purity_threshold=0.65, score_threshold=0.05):
    total = scores.sum(dim=1, keepdim=True).clamp(min=1e-8)
    purity = scores.max(dim=1).values / total.squeeze(-1)
    max_score = scores.max(dim=1).values
    dominant = scores.argmax(dim=1)

    specialized = (purity >= purity_threshold) & (max_score >= score_threshold)

    if not specialized.any():
        return 0, [], [], []

    covered_classes = torch.unique(dominant[specialized])
    return covered_classes.numel(), dominant, purity, max_score
```

---

## 10. Intégration proposée dans le code

### 9.1 Router

Le router doit rester simple.

À éviter :

```text
Mettre toute la logique adaptative dans HyperNetworkRouter.
```

Le router doit seulement accepter un `top_k` courant :

```python
router.set_top_k(current_top_k)
```

ou :

```python
router.top_k = current_top_k
```

Responsabilité du router :

```text
calculer correlations
appliquer top-k
appliquer epsilon-greedy
retourner routing_weights
```

Responsabilité du contrôleur :

```text
décider current_top_k
```

### 9.2 Training loop

Dans `train_loop_lifecycle()` :

```text
avant forward :
    current_k = controller.step(...)
    router.set_top_k(current_k)

après forward :
    routing_weights = out["routing_weights"]
    batch_usage = routing_weights.mean(dim=0)
    routing_usage_ema = update_ema(...)
    metrics = compute_routing_metrics(...)

après optimizer.step() :
    lifecycle eval
    specialization update
    courant step
    log
```

Attention à l’ordre actuel :

```text
lifecycle eval doit rester après optimizer.step()
```

pour éviter les erreurs autograd liées aux mutations in-place.

### 9.3 Config YAML

Ajouter une section optionnelle :

```yaml
model:
  top_k: 1
  top_k_curriculum:
    enabled: true
    k_init: 3
    k_final: 1
    warmup_steps: 300
    freeze_step: 1000
    adaptive: false
    usage_entropy_low: 0.35
    usage_entropy_high: 0.75
    dead_threshold: 0.05
    similarity_high: 0.85
    cooldown_steps: 25
```

Pour V1 :

```yaml
adaptive: false
```

Pour V2 :

```yaml
adaptive: true
```

### 9.4 Logger CSV

Colonnes minimales V1 :

```text
epoch
batch
step
loss
task_loss
accuracy
current_top_k
scheduled_top_k
effective_top_k
routing_usage_entropy
min_usage_ratio
dead_island_count
spec_score
spec_max
spec_std
spec_coverage
births
deaths
```

Colonnes V2 :

```text
adaptive_top_k
top_k_action
top_k_reason
ocean_pairwise_similarity
```

### 9.5 Logs qualitatifs

Ajouter des messages lisibles :

```text
Île 2 inactive depuis 100 steps
k remonté de 1 à 2 pour cause de min_usage
k réduit de 2 à 1 : usage équilibré + similarité élevée
freeze atteint : k verrouillé à 1
collapse de spécialisation détecté : spec_coverage=1
```

Ces messages sont très utiles pendant les runs longs.

---

## 11. Expériences recommandées

### 10.1 Expérience A — Baseline actuelle

```text
top_k=1 fixe
```

Objectif :

```text
reproduire l’échec
```

Métriques attendues :

```text
routing_usage_entropy s’effondre
min_usage_ratio = 0
dead_island_count > 0
spec_score reste proche de 0
spec_max peut être élevé
spec_coverage = 1
```

### 10.2 Expérience B — V1 courte

```text
k=3 → 2 → 1 sur 300 steps
```

Objectif :

```text
vérifier que les îles restent actives au début
```

Critères de succès :

```text
dead_island_count = 0 pendant les 100 premiers steps
min_usage_ratio > 0.05 au début
spec_score progresse
spec_coverage augmente
accuracy ne chute pas
```

### 10.3 Expérience C — V1 longue

```text
k=3 → 2 → 1 sur 500 steps
```

Objectif :

```text
tester si plus d’exploration aide ou nuit
```

Signe de trop d’exploration :

```text
accuracy correcte
mais spec_score faible
spec_coverage faible
effective_top_k reste élevé trop longtemps
```

### 10.4 Expérience D — V2 adaptative

```text
k=3 → 2 → 1
+ remontée de k si dead_island_count > 0
+ diminution si usage_entropy haute + similarité haute
+ freeze_step
```

Objectif :

```text
corriger les cas où la V1 ne suffit pas
```

### 10.5 Matrice d’ablation

| Run | top_k | warmup | adaptatif | But |
|-----|-------|--------|-----------|-----|
| A | 1 fixe | non | non | baseline échec |
| B | 3→1 | 300 | non | V1 principale |
| C | 3→1 | 500 | non | exploration plus longue |
| D | 4→1 | 300 | non | explorer plus d’îles |
| E | 3→1 | 300 | oui | rescue adaptatif |
| F | 3→1 | 300 | oui + freeze | robustesse finale |

---

## 12. Critères de succès

### 11.1 Bootstrap sain

Pendant les premiers 100-200 steps :

```text
dead_island_count = 0
min_usage_ratio > 0.05
routing_usage_entropy stable
aucune île ne monopolise immédiatement
```

### 11.2 Spécialisation finale

Après convergence :

```text
current_top_k = 1
spec_score > 0.30 minimum
spec_score > 0.50 idéal
spec_coverage >= 3 pour 4 îles
spec_std non nulle
accuracy au moins équivalente à la baseline
```

### 11.3 Absence de collapse

Mauvais signe :

```text
spec_max élevé
spec_score faible
spec_coverage = 1
```

Cela signifie :

```text
toutes les îles sont fortes sur la même classe
```

### 11.4 Absence de redondance

Mauvais signe :

```text
current_top_k élevé trop longtemps
accuracy correcte
spec_score faible
ocean_pairwise_similarity élevée
```

Cela signifie :

```text
trop d’exploration / pas assez de compétition
```

---

## 13. Interprétation des métriques

### 12.1 Tableau de diagnostic

| Signal | Bas | Haut | Interprétation |
|--------|-----|----|----------------|
| `routing_usage_entropy` | 0.0 | 1.0 | Équilibre global d’usage |
| `min_usage_ratio` | 0.0 | 1/N | Plus petite part d’une île |
| `dead_island_count` | 0 | >0 | Nombre d’îles quasi-mortes |
| `effective_top_k` | 1 | N | Nombre réel d’îles activées |
| `ocean_pairwise_similarity` | 0.0 | 1.0 | Redondance géométrique |
| `spec_score` | 0.0 | 1.0 | Spécialisation globale |
| `spec_max` | 0.0 | 1.0 | Confiance maximale par île |
| `spec_std` | 0.0 | variable | Écart entre îles |
| `spec_coverage` | 1 | num_classes | Classes couvertes par les îles spécialisées |

### 12.2 Patterns

### Pattern 1 — Monopolisation

```text
current_top_k = 1
routing_usage_entropy basse
min_usage_ratio = 0
dead_island_count > 0
spec_score faible
```

Action :

```text
augmenter k pendant le bootstrap
```

### Pattern 2 — Collapse sur une classe

```text
spec_max élevé
spec_score faible
spec_coverage = 1
```

Action V1 :

```text
logger + diagnostic
```

Action future :

```text
pénalité fonctionnelle / reshuffling contrôlé
```

### Pattern 3 — Trop d’exploration

```text
current_top_k élevé
routing_usage_entropy haute
accuracy correcte
spec_score faible
ocean_pairwise_similarity élevée
```

Action :

```text
réduire warmup_steps
ou accélérer la descente vers k=1
```

### Pattern 4 — Bon bootstrap

```text
current_top_k décroît
dead_island_count = 0 au début
min_usage_ratio > 0
spec_coverage augmente
spec_score augmente
accuracy reste stable
```

---

## 14. Risques et garde-fous

### 13.1 Risque : oscillation de `k`

Solution :

```text
hystérésis
cooldown_steps
freeze_step
```

### 13.2 Risque : `k` reste élevé trop longtemps

Solution :

```text
descente garantie vers k=1
freeze_step fixe
```

### 13.3 Risque : similarité trompeuse

Solution :

```text
ne jamais utiliser similarity seule
coupler avec usage_entropy et spec_coverage
```

### 13.4 Risque : confiance softmax trompeuse

Solution :

```text
préférer purity / accuracy conditionnelle par île
```

### 13.5 Risque : gradients dilués

Avec `k=3`, chaque île reçoit une fraction du gradient.

À surveiller :

```text
task_loss
accuracy
spec_score
```

Si la loss stagne :

```text
réduire k_init
ou réduire warmup_steps
```

### 13.6 Risque : `spec_coverage` calculée trop tôt

Au début, les scores de spécialisation sont bruités.

Donc :

```text
spec_coverage est un diagnostic,
pas un signal de contrôle en V1.
```

---

## 15. Décisions de design validées

### 14.1 `top_k=1` reste le régime final

Oui.

```text
top_k=1 est requis pour spécialisation stricte.
```

### 14.2 `top_k=2` ou `3` peut être utile au bootstrap

Oui.

```text
top_k > 1 au début permet d’activer plusieurs îles.
```

### 14.3 Le contrôleur ne doit pas vivre dans le router

Oui.

```text
Le router doit rester un module de routage.
Le contrôleur vit dans la boucle d’entraînement ou dans un module dédié.
```

### 14.4 L’entropie d’usage doit être globale

Oui.

```text
routing_usage_entropy sur EMA ou fenêtre glissante.
Pas l’entropie par batch.
```

### 14.5 `min_usage_ratio` est indispensable

Oui.

```text
C’est le signal qui détecte les îles mortes.
```

### 14.6 `spec_coverage` est indispensable comme diagnostic

Oui.

```text
Il détecte le collapse où toutes les îles pointent vers la même classe.
```

### 14.7 Le freeze final est indispensable

Oui.

```text
Après freeze_step, k=1 et ne remonte plus.
```

---

## 16. Checklist d’implémentation V1

### Code

- [x] Créer `TopKCurriculum` ou équivalent.
- [x] Ajouter config YAML `top_k_curriculum`.
- [x] Permettre au router de recevoir `current_top_k` à chaque step.
- [x] Appeler le contrôleur avant `forward()`.
- [x] Logger `current_top_k`, `scheduled_top_k`, `effective_top_k`.
- [x] Calculer `routing_usage_ema`.
- [x] Calculer `routing_usage_entropy`.
- [x] Calculer `min_usage_ratio`.
- [x] Calculer `dead_island_count`.
- [x] Calculer `spec_coverage`.
- [ ] Ajouter logs qualitatifs.
- [x] Ajouter tests unitaires.

### Tests unitaires recommandés

```text
test_topk_curriculum_linear_decay
test_topk_curriculum_freeze
test_routing_usage_entropy
test_min_usage_dead_island
test_specialization_coverage_detects_collapse
```

### Validation MNIST

```bash
python test_validation_niveau_1_4_1_5.py --epochs 1 --batch-size 64
```

Puis, si concluant :

```bash
python test_validation_niveau_1_4_1_5.py --epochs 50 --batch-size 128
```

---

## 17. Checklist d’implémentation V2

À faire uniquement si V1 ne suffit pas.

- [ ] Activer mode `adaptive=true`.
- [ ] Calculer `ocean_pairwise_similarity`.
- [ ] Ajouter seuils `usage_entropy_low`, `usage_entropy_high`, `similarity_high`.
- [ ] Ajouter hystérésis.
- [ ] Ajouter `cooldown_steps`.
- [ ] Ajouter messages de raison :
  - `min_usage_low`
  - `entropy_low`
  - `entropy_high_similarity_high`
  - `scheduled_decay`
  - `freeze`
- [ ] Tester absence d’oscillation.
- [ ] Tester que `freeze_step` verrouille `k=1`.

---

## 18. Plan de recherche recommandé

### Étape 1 — Reproduire l’échec

Objectif :

```text
confirmer la baseline actuelle
```

À mesurer :

```text
routing_usage_entropy
min_usage_ratio
dead_island_count
spec_coverage
```

### Étape 2 — Implémenter V1

Objectif :

```text
valider que le curriculum simple suffit
```

À mesurer :

```text
activation initiale de toutes les îles
spécialisation finale
accuracy
```

### Étape 3 — Ablation des hyperparamètres

Tester :

```text
k_init ∈ {2, 3, 4}
warmup_steps ∈ {100, 300, 500}
beta ∈ {0.90, 0.95}
freeze_step ∈ {800, 1000, 80% total_steps}
```

### Étape 4 — Ajouter V2 si nécessaire

Objectif :

```text
corriger les cas de dead island sans casser la spécialisation finale
```

### Étape 5 — Étendre à plus d’îles

Une fois validé sur 4 îles :

```text
num_islands = 6
num_islands = 8
```

Adapter :

```text
k_init = min(3, num_islands)
dead_threshold = 0.5 / num_islands ou 0.05
```

---

## 19. Formules de référence

### 18.1 Usage EMA

```text
u_0 = [1/N, ..., 1/N]
u_t = beta * u_{t-1} + (1 - beta) * mean_batch(w_t)
```

### 18.2 Entropie normalisée

```text
H(u) = -Σ u_i log(u_i + ε) / log(N)
```

### 18.3 Min usage

```text
min_usage = min_i u_i
```

### 18.4 Dead islands

```text
dead_i = 1[u_i < dead_threshold]
dead_count = Σ dead_i
```

### 18.5 Similarité moyenne

```text
S = mean_{i≠j} cosine(h_i, h_j)
```

### 18.6 Purity par île

```text
purity_i = max_c score_{i,c} / Σ_c score_{i,c}
```

### 18.7 Coverage

```text
coverage = |{argmax_c score_{i,c} pour les îles spécialisées}|
```

---

## 20. Fichiers modifiés par la V1

### 20.1 `archipel/src/archipel/current/router.py`

Modifié : ajout de `set_top_k(k)` borné entre 1 et `num_islands`.

À garder :

```text
Le router ne décide pas du curriculum.
Il applique seulement le top_k courant fourni par la boucle.
```

### 20.2 `archipel/src/archipel/training/loop_lifecycle.py`

Modifié : curriculum appliqué avant chaque forward, métriques routing après forward, logs CSV enrichis, logs qualitatifs de diagnostic.

Complété en V1 :

```text
qualitative_log dans la boucle de training
messages sur routing sain, monopolisation, dead islands, collapse de spec_coverage
```

Contrôleur V2 adaptatif avec hystérésis : à faire plus tard.

### 20.3 `archipel/src/archipel/training/csv_logger.py`

Modifié : ajout du champ `qualitative_log` et conservation des champs routing/spécialisation.

### 20.4 `archipel/src/archipel/islands/specialization.py`

Modifié : ajout de `get_specialization_summary()` et `spec_coverage`.

### 20.4 `archipel/src/archipel/ocean/ocean.py`

Non modifié en V1 : la similarité ocean reste un signal futur pour V2.

### 20.5 `archipel/configs/default.yaml`

Modifié : ajout de la section `top_k_curriculum` avec `enabled: true`.

```yaml
top_k_curriculum:
  enabled: true
  k_init: 3
  k_final: 1
  warmup_steps: 300
  freeze_step: null
```

### 20.6 Tests

Ajouté : `test_topk_curriculum.py`.

Tests existants réutilisés : `test_train_script.py`, `test_specialization.py`.

---

## 21. Décision finale

La direction validée est :

```text
1. Garder top_k=1 comme objectif final.
2. Ajouter un curriculum k=3 → k=2 → k=1.
3. Logger routing_usage_entropy, min_usage_ratio, dead_island_count.
4. Logger spec_coverage pour détecter le collapse.
5. Geler k=1 après freeze_step.
6. Ajouter l’adaptatif seulement si la V1 ne suffit pas.
```

Cette approche est la plus saine car elle isole d’abord la contribution principale :

```text
le curriculum de compétition
```

avant d’ajouter un contrôleur adaptatif plus complexe.

---

## 22. Prochaine action

Lancer les runs d’ablation et décider si V2 adaptative est nécessaire :

```text
baseline top_k=1 fixe
V1 3→2→1 courte
V1 longue
V2 adaptatif seulement si V1 ne suffit pas
```

Diagnostic déjà disponible :

```bash
uv run python scripts/diagnose_dynamic_mnist.py --epochs 3 --batch-size 64 --limit 4000 --k-init 3 --k-final 1 --warmup-steps 60 --freeze-step 120
uv run python test_mnist_quick.py --epochs 2 --batch-size 64
```

Critère de passage rapide :

```text
dead_island_count = 0 au début
spec_coverage augmente
spec_score > baseline actuelle
accuracy non dégradée
```

# Découverte critique : top_k=1 pour la spécialisation

**Date initiale** : 2026-05-24  
**Contexte** : Niveau 1.4 validation spécialisation sur MNIST  
**Statut** : ⚠️ Problème de bootstrap identifié (voir SPECIALIZATION_BOOTSTRAP_DIAGNOSTIC.md)

## Historique

### 2026-05-24 : Découverte initiale
- Score spécialisation = 1.0000 avec top_k=1 (parfaite sur test court 200 images)

### 2026-05-27 : Dégradation sur validation longue
- Score spécialisation = 0.0033 sur 50 époques MNIST (échec Niveau 1.4)
- `spec_max=0.99` indique que TOUTES les îles se spécialisent sur la MÊME classe
- Problème de **bootstrap de spécialisation** identifié

## Diagnostic approfondi (2026-05-27)

### Symptômes observés
| Métrique | Valeur | Interprétation |
|----------|--------|---------------|
| spec_score | 0.0033 | Échec (toutes îles même classe) |
| spec_max | 0.99+ | Confade - une classe dominante |
| λ_coherence | 1.0 | Bloqué haut |
| coherence | ~1.0 | Trop élevé (target 0.5) |

### Cause racine : Bootstrap impossible
Avec `top_k=1`, seule UNE île est active par batch. Le mécanisme :

1. **Phase initiale** : Une île a un léger avantage de corrélation cosinus
2. **Sélection systématique** : `top_k=1` la sélectionne systématiquement
3. **Apprentissage concentrate** : Elle améliore ses embeddings
4. **Répulsion des autres îles** : Leurs embeddings restent inactifs, n'apprennent RIEN
5. **Effondrement** : Toutes les îles convergent vers la même direction

### Parallèle avec top_k=2
Avec `top_k=2`, plusieurs îles apprennent les mêmes patterns → spécialisation impossible.  
Avec `top_k=1`, le bootstrap échoue si l'exploration epsilon=0.1 est insuffisante.

## Solutions proposées

### Solution A : Augmenter epsilon
```python
# router.py
epsilon_init: float = 0.1 → 0.3  # 30% exploration initiale
```

### Solution B : Warm-up top_k
```python
# Phases 0-100 : top_k=2 (exploration)
# Phases 100+ : top_k=1 (spécialisation)
```

### Solution C : Modulation epsilon adaptative
```python
# courant.py - intensifier la modulation
if mean_diversity < diversity_target * 0.2:
    return 2.0  # Au lieu de 1.5
```

## Fichiers de référence
- `archipel/src/archipel/current/router.py` - epsilon et sélection top_k
- `archipel/src/archipel/current/courant.py` - adaptation λ et modulation epsilon
- `archipel/src/archipel/training/loop_lifecycle.py` - warm-up top_k
- `references/SPECIALIZATION_BOOTSTRAP_DIAGNOSTIC.md` - diagnostic détaillé (2026-05-27)
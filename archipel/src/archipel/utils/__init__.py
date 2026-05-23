"""Utilitaires d'analyse de spécialisation des îles Archipel.

Fournit :
- compute_specialization_matrix : construit la matrice îles × classes
- specialization_score : score de spécialisation global (0 = uniforme, 1 = spécialisé)
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader


def compute_specialization_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> torch.Tensor:
    """Retourne une matrice [num_islands, num_classes] : fréquence de prédiction
    de chaque classe par île.

    Args:
        model: ArchipelPhase2 (ou tout modèle retournant ``island_outputs``).
        loader: DataLoader MNIST ou équivalent (batch de (x, y)).
        device: Device de calcul.

    Returns:
        Matrice (num_islands, num_classes) de comptes entiers.
    """
    model.eval()
    num_islands = model.num_islands
    num_classes = model.task_head.out_features  # 10 pour MNIST par défaut
    counts = torch.zeros(num_islands, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            # out["island_outputs"] : [batch, num_islands, num_classes]
            island_preds = out["island_outputs"].argmax(dim=2)  # [batch, num_islands]
            for b in range(island_preds.size(0)):
                for i in range(num_islands):
                    cls = island_preds[b, i].item()
                    counts[i, cls] += 1

    model.train()
    return counts.float()


def specialization_score(matrix: torch.Tensor) -> float:
    """Score de spécialisation global des îles.

    1.0 = chaque île ne prédit qu'UNE seule classe (spécialisation parfaite)
    0.0 = toutes les îles répartissent uniformément sur toutes les classes

    Args:
        matrix: Matrice (num_islands, num_classes) de fréquences.

    Returns:
        Score float dans [0, 1].
    """
    total = matrix.sum(dim=1, keepdim=True).clamp(min=1)
    probs = matrix / total  # [num_islands, num_classes]
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)  # [num_islands]
    max_entropy = torch.log(torch.tensor(matrix.size(1), dtype=torch.float32))
    avg_entropy = entropy.mean()
    return 1.0 - (avg_entropy / max_entropy).item()

"""Utilitaires d'analyse de spécialisation des îles Archipel.

Fournit :
- compute_specialization_matrix : construit la matrice îles × classes
- compute_specialization_matrix_with_predictions : construit la matrice
  fonctionnelle en tenant compte des prédictions et cibles par batch
- specialization_score : score de spécialisation global (0 = uniforme, 1 = spécialisé)
- specialization_score_precision_weighted : score fonctionnel pondéré par précision
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


def compute_specialization_matrix_with_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construit une matrice fonctionnelle de spécialisation.

    Contrairement à ``compute_specialization_matrix`` qui ne compte que les
    classes prédites par chaque île, cette variante croise les prédictions
    d'îles avec les cibles réelles du batch. Elle permet de mesurer une
    spécialisation fonctionnelle : une île est d'autant plus spécialisée qu'elle
    prédit correctement une classe donnée.

    Returns:
        Tuple:
        - matrix fonctionnelle [num_islands, num_classes]
        - predictions agrégées [num_samples]
        - targets [num_samples]
    """
    model.eval()
    num_islands = model.num_islands
    num_classes = model.task_head.out_features
    counts = torch.zeros(num_islands, num_classes, dtype=torch.int64)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            out = model(x_batch)
            island_preds = out["island_outputs"].argmax(dim=2)  # [batch, num_islands]
            batch_preds = out["output"].argmax(dim=1)

            all_preds.append(batch_preds.detach().cpu())
            all_targets.append(y_batch.detach().cpu())

            correct = (island_preds == y_batch.unsqueeze(1)).float()
            # Compter uniquement les prédictions correctes par classe cible.
            # Cela évite de récompenser une île qui prédit beaucoup une classe
            # mais se trompe souvent.
            for b in range(island_preds.size(0)):
                target = int(y_batch[b].item())
                for i in range(num_islands):
                    if correct[b, i].item() > 0.0:
                        counts[i, target] += 1

    model.train()
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return counts.float(), preds, targets


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


def specialization_score_precision_weighted(
    matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Score fonctionnel de spécialisation pondéré par la précision globale.

    La matrice est déjà filtrée sur les prédictions correctes par classe cible.
    On applique donc un facteur de précision globale pour éviter de récompenser
    une spécialisation artificielle sur un modèle qui prédit mal.
    """
    if predictions.numel() == 0:
        return 0.0

    accuracy = (predictions == targets).float().mean().item()
    raw_score = specialization_score(matrix)
    return float(raw_score * accuracy)

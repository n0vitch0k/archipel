"""Modèles spécifiques pour benchmarks MNIST.

MNISTEncoder — encodeur CNN 28×28 → dim_latent
MNISTArchipel — ArchipelPhase2 adapté pour MNIST (héritage + override kill_island)
"""

import torch
import torch.nn as nn

from archipel.training.loop_lifecycle import ArchipelPhase2
from archipel.current.courant import Courant


class MNISTEncoder(nn.Module):
    """Encodeur CNN 28×28 → out_dim pour MNIST."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class MNISTArchipel(ArchipelPhase2):
    """ArchipelPhase2 adapté pour MNIST — encodeur CNN 28×28 → 128.

    Règle critique :
        self.encoder = nn.Identity()  → empêche double encodage
        self.mnist_encoder             → encodeur externe CNN
    """

    def __init__(self) -> None:
        super().__init__(
            num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=128,
            top_k=2, max_islands=8, min_islands=2,
            coherence_variance_threshold=0.3,
        )
        self.encoder = nn.Identity()          # empêche double encodage
        self.mnist_encoder = MNISTEncoder()   # encodeur externe

    def forward(self, x: torch.Tensor, targets=None) -> dict:
        input_repr = self.mnist_encoder(x)
        return super().forward(input_repr, targets=targets)

    def kill_island(self, island_id: int, distill: bool = True, dataloader=None) -> bool:
        dl = dataloader if dataloader is not None else self._dataloader_for_distillation
        enc = self.mnist_encoder if hasattr(self, "mnist_encoder") else self.encoder
        return super().kill_island(
            island_id, distill=distill, dataloader=dl, encoder=enc,
        )

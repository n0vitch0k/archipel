"""Baselines non modulaires pour comparer Archipel.

MLPBaseline — capacité paramétrique visuellement similaire à ArchipelPhase2
(encodeur partagé + classifieur dense).
"""

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """MLP encodeur CNN 28×28 → 128, puis 128→256→128→10.

    Capacité paramétrique comparable à 4 îlots Archipel pour une
    évaluation équitable sur MNIST.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flat = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flat(self.encoder(x)))

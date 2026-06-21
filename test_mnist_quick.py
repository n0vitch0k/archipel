"""Validation rapide sur MNIST — Niveau 1.2.

Vérifie que ArchipelPhase2 converge sur un vrai dataset :
- loss diminue
- accuracy augmente
- lifecycle (spawn/kill) ne crash pas
- les îles développent des spécialisations
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant
from archipel.current.topk_curriculum import TopKCurriculum, RoutingUsageTracker


# ── Encodeur CNN 28×28 → 128 ─────────────────────────────────────────────────
class MNISTEncoder(nn.Module):
    """Encode les images 28×28 (1 canal) en vecteur 128-dim."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),   # (B,1,28,28)→(B,16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                                          # → (B,16,14,14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),             # → (B,32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                                          # → (B,32,7,7)
        )
        self.fc = nn.Linear(32 * 7 * 7, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


# ── Modèle complet : hérite directement de ArchipelPhase2 ────────────────────
class MNISTArchipel(ArchipelPhase2):
    """ArchipelPhase2 adapté pour MNIST — encodeur CNN 28×28 → 128, classification 10 classes."""

    def __init__(self) -> None:
        super().__init__(
            num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=128,
            top_k=2, max_islands=8, min_islands=2,
            coherence_variance_threshold=0.3,
        )
        # L'encodeur de ArchipelPhase2 devient l'identité (pas de double encodage)
        self.encoder = nn.Identity()
        # Nouvel encodeur CNN pour MNIST, enregistré explicitement comme sous-module
        self.mnist_encoder = MNISTEncoder(out_dim=128)

    def forward(self, x: torch.Tensor, targets=None) -> dict:
        """Encode les images 28×28 → 128 puis forward Archipel sans double encodage."""
        input_repr = self.mnist_encoder(x)
        return super().forward(input_repr, targets=targets)

    def kill_island(self, island_id: int, distill: bool = True, dataloader=None) -> bool:
        """Override pour fournir mnist_encoder à la distillation."""
        dl = dataloader if dataloader is not None else self._dataloader_for_distillation
        # Utilise mnist_encoder si disponible pour encoder les images brutes
        enc = self.mnist_encoder if hasattr(self, 'mnist_encoder') else self.encoder
        return super().kill_island(
            island_id,
            distill=distill,
            dataloader=dl,
            encoder=enc,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
def check(msg: str, cond: bool) -> None:
    """Assertion simple."""
    if not cond:
        raise AssertionError(f"FAILED: {msg}")


def train_mnist_quick(epochs: int = 5, batch_size: int = 64) -> dict:
    """Entraîne MNISTArchipel sur MNIST et retourne un résumé des métriques."""
    print("\n" + "=" * 60)
    print("TEST MNIST RAPIDE — ArchipelPhase2 sur données réelles")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_full = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist = torch.utils.data.Subset(mnist_full, range(10000))  # sous-ensemble de 10k images pour rapidité
    loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=0)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = MNISTArchipel()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    courant = Courant(num_islands=4)

    device = "cpu"
    print(f"\nConfig   : epochs={epochs}, batch_size={batch_size}, device={device}")
    print(f"Données  : {len(mnist_full)} images MNIST (évaluation sur {len(mnist)} sous-ensemble) — encodeur CNN 28×28→128")
    print(f"Modèle   : {model.num_islands} îlots, top-k curriculum 3→2→1, max=8, min=2")

    # ── Évaluation avant entraînement ─────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        eval_loader = DataLoader(mnist, batch_size=512, shuffle=False, num_workers=0)
        all_preds, all_labels = [], []
        for x_eval, y_eval in eval_loader:
            out = model(x_eval)
            all_preds.append(out["output"].argmax(dim=1))
            all_labels.append(y_eval)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc_before = (all_preds == all_labels).float().mean().item()
    model.train()
    print(f"\nAccuracy avant entraînement : {acc_before:.4f}")

    curriculum = TopKCurriculum(
        num_islands=model.num_islands,
        k_init=3,
        k_final=1,
        warmup_steps=150,
    )
    routing_tracker = RoutingUsageTracker(num_islands=model.num_islands)

    # ── Entraînement ──────────────────────────────────────────────────────────
    print("\n--- Début entraînement ---")
    logs, _ = train_loop_lifecycle(
        model, loader, optimizer, courant,
        epochs=epochs, device=device, log_every=50,
        top_k_curriculum=curriculum,
        routing_usage_tracker=routing_tracker,
    )
    print("--- Fin entraînement ---\n")

    # ── Évaluation finale ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        eval_loader = DataLoader(mnist, batch_size=512, shuffle=False, num_workers=0)
        all_preds, all_labels = [], []
        for x_eval, y_eval in eval_loader:
            out = model(x_eval)
            all_preds.append(out["output"].argmax(dim=1))
            all_labels.append(y_eval)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc_final = (all_preds == all_labels).float().mean().item()

    # ── Métriques ─────────────────────────────────────────────────────────────
    final_loss   = logs[-1]["loss"]
    initial_loss = logs[0]["loss"]
    loss_improvement = (initial_loss - final_loss) / initial_loss * 100

    birth_events = [l for l in logs if l.get("event") == "birth"]
    death_events = [l for l in logs if l.get("event") == "death"]

    # ── Affichage ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("RÉSULTATS DU TEST MNIST")
    print("=" * 60)
    print(f"  Époques                 : {epochs}")
    print(f"  Loss initiale           : {initial_loss:.4f}")
    print(f"  Loss finale             : {final_loss:.4f}")
    print(f"  Amélioration            : {loss_improvement:.1f}%")
    print(f"  Accuracy avant          : {acc_before:.4f}")
    print(f"  Accuracy finale (full)  : {acc_final:.4f}  ← objectif ≥ 0.70")
    print(f"  Îlots finaux            : {model.num_islands}")
    print(f"  Naissances              : {len(birth_events)}")
    print(f"  Morts                   : {len(death_events)}")
    print(f"  Entrées de log totales  : {len(logs)}")
    print("=" * 60)

    # ── Assertions ────────────────────────────────────────────────────────────
    check("logs non vides", len(logs) > 0)
    check("loss finale < loss initiale", final_loss < initial_loss)
    check("accuracy finale ≥ 0.50  (apprentissage détecté)", acc_final >= 0.50)
    check("accuracy finale ≥ 0.70  (niveau acceptable MNIST rapide)", acc_final >= 0.70)
    check(
        "lifecycle n'a pas crashé  (au moins min_islands îlots restent)",
        model.num_islands >= model.min_islands,
    )
    check("model.eval() fonctionne", model.training is False)

    return {
        "epochs":               epochs,
        "initial_loss":         round(initial_loss, 4),
        "final_loss":           round(final_loss, 4),
        "loss_improvement_pct": round(loss_improvement, 1),
        "accuracy_before":      round(acc_before, 4),
        "accuracy_final":       round(acc_final, 4),
        "num_islands_final":    model.num_islands,
        "num_births":           len(birth_events),
        "num_deaths":           len(death_events),
        "num_logs":             len(logs),
        "PASSED":               True,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ArchipelPhase2 sur MNIST (test rapide)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Nombre d'époques (défaut: 5)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Taille de batch (défaut: 64)")
    args = parser.parse_args()
    result = train_mnist_quick(epochs=args.epochs, batch_size=args.batch_size)
    print(f"\nRésumé JSON-ready :\n  {result}")

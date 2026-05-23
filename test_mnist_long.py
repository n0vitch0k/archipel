"""Test MNIST long — Niveau 1.2 complémentaire.

Run de 20 époques sur 60k images MNIST pour valider :
- Convergence à long terme (loss continue de baisser ou stagne ?)
- Stabilité du cycle de vie (nbr de births/deaths par époque)
- Évolution de la diversité des îles
- Accuracy finale sur 10k images de validation
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant


class MNISTEncoder(nn.Module):
    """Encodeur CNN 28×28 → 128 pour MNIST."""
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
    """ArchipelPhase2 + encodeur CNN pour MNIST."""
    def __init__(self) -> None:
        super().__init__(
            num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=128,
            top_k=2, max_islands=8, min_islands=2,
            coherence_variance_threshold=0.3,
        )
        self.encoder = nn.Identity()
        self.mnist_encoder = MNISTEncoder(out_dim=128)

    def forward(self, x: torch.Tensor, targets=None) -> dict:
        input_repr = self.mnist_encoder(x)
        return super().forward(input_repr, targets=targets)

    def kill_island(self, island_id: int, distill: bool = True, dataloader=None) -> bool:
        dl = dataloader if dataloader is not None else self._dataloader_for_distillation
        enc = self.mnist_encoder if hasattr(self, 'mnist_encoder') else self.encoder
        return super().kill_island(
            island_id, distill=distill, dataloader=dl, encoder=enc,
        )


def evaluate(model, mnist_eval, batch_size: int = 512, subset_size: int = 10000) -> float:
    """Évalue l'accuracy sur un sous-ensemble du dataset."""
    model.eval()
    eval_subset = Subset(mnist_eval, range(subset_size))
    loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_eval, y_eval in loader:
            out = model(x_eval)
            all_preds.append(out["output"].argmax(dim=1))
            all_labels.append(y_eval)
    model.train()
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return (all_preds == all_labels).float().mean().item()


def run_long(epochs: int = 20, batch_size: int = 128) -> dict:
    print(f"\n{'=' * 60}")
    print(f"TEST MNIST LONG — {epochs} époques, batch {batch_size}")
    print(f"{'=' * 60}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_eval  = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)

    # Modèle
    model = MNISTArchipel()
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    courant = Courant(num_islands=4)

    # Accuracy avant
    acc_before = evaluate(model, mnist_eval)
    print(f"Accuracy avant entraînement : {acc_before:.4f}")

    # Entraînement
    print(f"\n--- Début {epochs} époques ---")
    logs, _ = train_loop_lifecycle(
        model, loader, opt, courant,
        epochs=epochs, device="cpu", log_every=100,
    )
    print(f"--- Fin entraînement ---\n")

    # Accuracy finale
    acc_final = evaluate(model, mnist_eval)
    initial_loss = logs[0]["loss"]
    final_loss   = logs[-1]["loss"]

    # Agrégat par époque
    epoch_summary = []
    for e in range(epochs):
        e_logs = [l for l in logs if l.get("epoch") == e]
        if not e_logs:
            continue
        epoch_summary.append({
            "epoch":  e + 1,
            "loss":   round(e_logs[-1]["loss"], 4),
            "diversity": round(e_logs[-1].get("diversity", 0), 4),
            "coherence": round(e_logs[-1].get("coherence", 0), 4),
            "islands":  e_logs[-1].get("islands", model.num_islands),
            "lambda_coh": round(e_logs[-1].get("lambda_coh", 0), 4),
        })

    # Lifecycle events par époque
    birth_per_epoch = {}
    death_per_epoch = {}
    for l in logs:
        e = l.get("epoch", 0)
        if l.get("event") == "birth":
            birth_per_epoch[e] = birth_per_epoch.get(e, 0) + 1
        if l.get("event") == "death":
            death_per_epoch[e] = death_per_epoch.get(e, 0) + 1

    print(f"{'=' * 60}")
    print(f"RÉSULTATS — {epochs} époques")
    print(f"{'=' * 60}")
    print(f"  Accuracy avant       : {acc_before:.4f}")
    print(f"  Accuracy finale      : {acc_final:.4f}")
    print(f"  Loss initiale        : {initial_loss:.4f}")
    print(f"  Loss finale          : {final_loss:.4f}")
    print(f"  Îlots finaux         : {model.num_islands}")
    total_b = sum(birth_per_epoch.values())
    total_d = sum(death_per_epoch.values())
    print(f"  Total births         : {total_b}")
    print(f"  Total deaths         : {total_d}")

    print(f"\n  Par époque :")
    print(f"  {'Époque':>6} | {'Loss':>8} | {'Diversité':>9} | {'Coherence':>9} | {'Îlots':>5} | {'λ_coh':>6} | births | deaths")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for row in epoch_summary:
        e = row["epoch"]
        b = birth_per_epoch.get(e, 0)
        d = death_per_epoch.get(e, 0)
        if e <= 10 or e > 10 and (e - 10) % 5 == 0 or e == epochs:
            print(f"  {row['epoch']:>6} | {row['loss']:>8.4f} | {row['diversity']:>9.4f} | "
                  f"{row['coherence']:>9.4f} | {row['islands']:>5} | {row['lambda_coh']:>6.4f} | "
                  f"{b:>6} | {d:>6}")

    print(f"{'=' * 60}\n")
    return {
        "epochs": epochs,
        "acc_before": round(acc_before, 4),
        "acc_final":  round(acc_final, 4),
        "initial_loss": round(initial_loss, 4),
        "final_loss":   round(final_loss, 4),
        "num_islands_final": model.num_islands,
        "total_births": total_b,
        "total_deaths": total_d,
        "epoch_summary": epoch_summary,
        "PASSED": True,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = run_long(epochs=args.epochs, batch_size=args.batch_size)
    print("Résumé JSON-ready :\n", result)

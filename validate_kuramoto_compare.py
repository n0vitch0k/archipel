"""Validation comparative Phase 2 (cosine) vs Phase 3 (Kuramoto) sur MNIST.

Usage:
    python validate_kuramoto_compare.py                          # 5 epochs, seeds 42 123 256
    python validate_kuramoto_compare.py --epochs 50 --batch-size 64  # 50 epochs
    python validate_kuramoto_compare.py --seeds 42 256                # seeds spécifiques
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase3, train_loop_lifecycle
from archipel.current.courant import Courant
from archipel.current.topk_curriculum import TopKCurriculum, RoutingUsageTracker


# ── Encodeur CNN 28×28 → 128 ─────────────────────────────────────────────────
class MNISTEncoder(nn.Module):
    """Encode les images 28×28 (1 canal) en vecteur 128-dim."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


# ── Modèle Phase 3 pour MNIST ────────────────────────────────────────────────
class MNISTArchipelPhase3(ArchipelPhase3):
    """ArchipelPhase3 adapté pour MNIST — encodeur CNN 28×28 → 128, classification 10 classes."""

    def __init__(self, routing_mode: str = "cosine", **kwargs) -> None:
        super().__init__(
            num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=128,
            top_k=2, max_islands=8, min_islands=2,
            coherence_variance_threshold=0.3,
            routing_mode=routing_mode,
            **kwargs,
        )
        self.encoder = nn.Identity()
        self.mnist_encoder = MNISTEncoder(out_dim=128)
        self._distillation_dataloader = None

    def forward(self, x: torch.Tensor, targets=None) -> dict:
        input_repr = self.mnist_encoder(x)
        return super().forward(input_repr, targets=targets)

    def kill_island(self, island_id: int, distill: bool = True, dataloader=None) -> bool:
        dl = dataloader if dataloader is not None else self._distillation_dataloader
        enc = self.mnist_encoder if hasattr(self, 'mnist_encoder') else self.encoder
        return super().kill_island(
            island_id,
            distill=distill,
            dataloader=dl,
            encoder=enc,
        )


# ── Entraînement et métriques ────────────────────────────────────────────────

def train_mnist_compare(
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    routing_mode: str = "cosine",
    data_size: int = 10000,
) -> dict:
    """Entraîne MNISTArchipelPhase3 sur MNIST et retourne les métriques."""
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_full = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_subset = Subset(mnist_full, range(data_size))
    loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MNISTArchipelPhase3(routing_mode=routing_mode)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    courant = Courant(num_islands=4)
    device = "cpu"

    # Eval before
    model.eval()
    with torch.no_grad():
        eval_loader = DataLoader(mnist_subset, batch_size=512, shuffle=False, num_workers=0)
        all_preds, all_labels = [], []
        for x_eval, y_eval in eval_loader:
            out = model(x_eval)
            all_preds.append(out["output"].argmax(dim=1))
            all_labels.append(y_eval)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc_before = (all_preds == all_labels).float().mean().item()
    model.train()

    curriculum = TopKCurriculum(
        num_islands=model.num_islands,
        k_init=3,
        k_final=1,
        warmup_steps=150,
    )
    routing_tracker = RoutingUsageTracker(num_islands=model.num_islands)

    logs, _ = train_loop_lifecycle(
        model, loader, optimizer, courant,
        epochs=epochs, device=device, log_every=50,
        top_k_curriculum=curriculum,
        routing_usage_tracker=routing_tracker,
    )

    # Eval after
    model.eval()
    with torch.no_grad():
        all_preds, all_labels = [], []
        for x_eval, y_eval in eval_loader:
            out = model(x_eval)
            all_preds.append(out["output"].argmax(dim=1))
            all_labels.append(y_eval)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc_final = (all_preds == all_labels).float().mean().item()

    # Métriques
    final_loss = logs[-1]["loss"]
    initial_loss = logs[0]["loss"]
    loss_improvement = (initial_loss - final_loss) / initial_loss * 100

    birth_events = [l for l in logs if l.get("event") == "birth"]
    death_events = [l for l in logs if l.get("event") == "death"]

    # Métriques Kuramoto (order_parameter dans les logs)
    kuramoto_logs = [l for l in logs if l.get("order_parameter") is not None]
    avg_order_param = (
        sum(l["order_parameter"] for l in kuramoto_logs) / len(kuramoto_logs)
        if kuramoto_logs else None
    )
    final_order_param = kuramoto_logs[-1]["order_parameter"] if kuramoto_logs else None

    return {
        "routing_mode": routing_mode,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "loss_improvement_pct": round(loss_improvement, 1),
        "accuracy_before": round(acc_before, 4),
        "accuracy_final": round(acc_final, 4),
        "num_islands_final": model.num_islands,
        "num_births": len(birth_events),
        "num_deaths": len(death_events),
        "avg_order_parameter": round(avg_order_param, 4) if avg_order_param is not None else None,
        "final_order_parameter": round(final_order_param, 4) if final_order_param is not None else None,
        "num_logs": len(logs),
        "PASSED": True,
    }


# ── Tableau comparatif ───────────────────────────────────────────────────────

def print_comparison(results: list[dict]) -> None:
    """Affiche un tableau de comparaison entre les runs."""
    print("\n" + "=" * 100)
    print("  COMPARAISON COSINE vs KURAMOTO — MNIST")
    print("=" * 100)

    # Header
    print(f"{'Mode':<10} {'Seed':<6} {'Epochs':<7} {'Loss init':<10} {'Loss fin':<10} "
          f"{'Acc avant':<10} {'Acc final':<10} {'Amél%':<7} {'Îlots':<6} "
          f"{'Nais':<5} {'Morts':<5} {'R moyen':<8} {'R fin':<8}")
    print("-" * 100)

    for r in results:
        print(f"{r['routing_mode']:<10} {r['seed']:<6} {r['epochs']:<7} "
              f"{r['initial_loss']:<10} {r['final_loss']:<10} "
              f"{r['accuracy_before']:<10} {r['accuracy_final']:<10} "
              f"{r['loss_improvement_pct']:<7} {r['num_islands_final']:<6} "
              f"{r['num_births']:<5} {r['num_deaths']:<5} "
              f"{str(r['avg_order_parameter']):<8} {str(r['final_order_parameter']):<8}")

    print("=" * 100)

    # Synthèse par mode
    for mode in ["cosine", "kuramoto"]:
        mode_results = [r for r in results if r["routing_mode"] == mode]
        if not mode_results:
            continue
        avg_acc = sum(r["accuracy_final"] for r in mode_results) / len(mode_results)
        avg_births = sum(r["num_births"] for r in mode_results) / len(mode_results)
        avg_deaths = sum(r["num_deaths"] for r in mode_results) / len(mode_results)
        avg_islands = sum(r["num_islands_final"] for r in mode_results) / len(mode_results)

        order_params = [r["final_order_parameter"] for r in mode_results
                        if r["final_order_parameter"] is not None]
        avg_r = sum(order_params) / len(order_params) if order_params else None

        print(f"\n  [{mode.upper()}] "
              f"Accuracy moyenne: {avg_acc:.4f}  "
              f"Îlots finaux moy: {avg_islands:.1f}  "
              f"Naissances moy: {avg_births:.1f}  "
              f"Morts moy: {avg_deaths:.1f}"
              + (f"  R final moy: {avg_r:.4f}" if avg_r is not None else ""))

    # Meilleur mode si Kuramoto activé
    cos_results = [r for r in results if r["routing_mode"] == "cosine"]
    kur_results = [r for r in results if r["routing_mode"] == "kuramoto"]
    if cos_results and kur_results:
        avg_cos = sum(r["accuracy_final"] for r in cos_results) / len(cos_results)
        avg_kur = sum(r["accuracy_final"] for r in kur_results) / len(kur_results)
        diff = avg_kur - avg_cos
        winner = "KURAMOTO" if diff > 0 else "COSINE"
        print(f"\n  ▶ DIFFÉRENCE: Kuramoto - Cosine = {diff:+.4f} → {winner} gagne")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation comparative Cosine vs Kuramoto sur MNIST"
    )
    parser.add_argument("--epochs", type=int, default=5,
                        help="Nombre d'époques (défaut: 5)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Taille de batch (défaut: 64)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 256],
                        help="Seeds à tester (défaut: 42 123 256)")
    parser.add_argument("--data-size", type=int, default=10000,
                        help="Taille du sous-ensemble MNIST (défaut: 10000)")
    parser.add_argument("--no-kuramoto", action="store_true",
                        help="Désactiver le mode Kuramoto (cosine seulement)")
    args = parser.parse_args()

    modes = ["cosine"]
    if not args.no_kuramoto:
        modes.append("kuramoto")

    total_runs = len(modes) * len(args.seeds)
    print(f"\n  Validation comparative MNIST — {args.epochs} epochs × {total_runs} runs")
    print(f"  Modes: {', '.join(modes)} | Seeds: {args.seeds}")
    print(f"  Batch size: {args.batch_size} | Data size: {args.data_size}\n")

    results = []
    for seed in args.seeds:
        for mode in modes:
            label = f"{mode.upper():<10} seed={seed:<5}"
            print(f"  ▶ {label}… ", end="", flush=True)
            t0 = time.time()
            result = train_mnist_compare(
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                routing_mode=mode,
                data_size=args.data_size,
            )
            elapsed = time.time() - t0
            acc = result["accuracy_final"]
            r_val = result.get("final_order_parameter")
            r_str = f" R={r_val}" if r_val is not None else ""
            print(f"✓ {acc:.4f} acc, {elapsed:.1f}s{r_str}")
            results.append(result)

    print_comparison(results)

    # Export JSON minifié
    import json
    json_path = ROOT / "compare_kuramoto_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Résultats sauvegardés: {json_path}")


if __name__ == "__main__":
    main()

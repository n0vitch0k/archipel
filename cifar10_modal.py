"""Validation CIFAR-10 sur Modal — Kuramoto vs Cosine.

Utilise un Volume Modal pour telecharger CIFAR-10 une seule fois.

Usage:
    modal run cifar10_modal.py                    # 3 seeds x 2 modes (5 epochs)
    modal run cifar10_modal.py --epochs 2 --seeds 42  # 1 seed smoke test
"""

import sys
from pathlib import Path

import modal

# ── Volume persistant pour CIFAR-10 (download unique) ─────────────────────
cifar10_volume = modal.Volume.from_name("cifar10-data", create_if_missing=True)

# ── App ──────────────────────────────────────────────────────────────────────
app = modal.App("cifar10-archipel")

# ── Image : torch CUDA + archipel depuis git HEAD ────────────────────────────
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("torch>=2.0.0", "torchvision>=0.15.0")
    .pip_install("numpy")
    .run_commands(
        "git clone --depth 1 https://github.com/n0vitch0k/archipel.git /archipel",
        "pip install -e /archipel/archipel",
    )
)


@app.local_entrypoint()
def main(epochs: int = 5, seeds: str = "42 123 256"):
    import json

    seeds = [int(s) for s in seeds.split()]
    print(f"\n{'='*60}")
    print(f"  CIFAR-10 Validation — {epochs} epochs, seeds={seeds}")
    print(f"  First run downloads dataset (~40 min), subsequent runs use cache")
    print(f"{'='*60}\n")

    results = {}
    for seed in seeds:
        for mode in ("cosine", "kuramoto"):
            print(f"  Running seed={seed} mode={mode}...")
            r = run_cifar10.remote(epochs=epochs, seed=seed, mode=mode)
            results[f"{mode}_{seed}"] = r
            print(f"  >> acc={r['accuracy_final']:.4f}, "
                  f"R_final={r.get('final_order_parameter', 'N/A')}")

    with open(str(Path(__file__).resolve().parent / "cifar10_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print("  DONE — results in cifar10_results.json")
    print(f"{'='*60}")


@app.function(image=image, gpu="T4", timeout=7200,
              volumes={"/data": cifar10_volume})
def run_cifar10(epochs: int, seed: int, mode: str) -> dict:
    import torch
    import sys as _sys
    _sys.path.insert(0, "/archipel/archipel/src")

    from archipel.training.loop_lifecycle import ArchipelPhase3, train_loop_lifecycle
    from archipel.current.courant import Courant
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── CNN Encoder 32x32x3 -> 128 ──
    class CIFAR10Encoder(nn.Module):
        def __init__(self, out_dim=128):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, out_dim),
            )
        def forward(self, x):
            return self.conv(x)

    # ── Modele CIFAR-10 ──
    class CIFAR10Router(ArchipelPhase3):
        def __init__(self, **kw):
            super().__init__(
                num_islands=4, input_dim=128, hidden_dim=64,
                ocean_dim=128, top_k=2, max_islands=8, min_islands=2,
                coherence_variance_threshold=0.3, **kw,
            )
            self.encoder = nn.Identity()
            self.cifar10_encoder = CIFAR10Encoder(out_dim=128)
            self._distillation_dataloader = None

        def forward(self, x, targets=None):
            input_repr = self.cifar10_encoder(x)
            return super().forward(input_repr, targets=targets)

    # ── Dataset CIFAR-10 (download unique via Volume) ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    cifar10 = datasets.CIFAR10("/data", train=True, download=True, transform=transform)
    loader = DataLoader(cifar10, batch_size=64, shuffle=True, num_workers=0)

    # ── Training ──
    torch.manual_seed(seed)
    model = CIFAR10Router(routing_mode=mode)
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    courant = Courant(num_islands=4).to(device)

    train_loop_lifecycle(model, loader, optimizer, courant, epochs=epochs,
                          device=device)

    # ── Eval ──
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)["output"]
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total

    # Sync metrics
    sm = {}
    if hasattr(model.router, "get_sync_metrics"):
        sm = model.router.get_sync_metrics()

    return {
        "accuracy_final": acc,
        "final_order_parameter": sm.get("order_parameter", 0.0),
        "num_births": getattr(model, "num_births", 0),
        "num_deaths": getattr(model, "num_deaths", 0),
    }


if __name__ == "__main__":
    main()

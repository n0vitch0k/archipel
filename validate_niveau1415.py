"""
Validation Niveau 1.4/1.5 — Test spécialisation + baseline MLP (2 époques rapides).

Objectif : Valider que les corrections du Courant améliorent la spécialisation.
"""
import sys, time, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))
# PYTHONPATH must be set for archipel package
import os
os.environ.setdefault("PYTHONPATH", str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant
from archipel.current.topk_curriculum import TopKCurriculum
from archipel.utils.specialization_matrix import (
    compute_specialization_matrix,
    compute_specialization_matrix_with_predictions,
    specialization_score,
    specialization_score_precision_weighted,
)


# ─── Encodeur CNN MNIST ───
class MNISTEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, out_dim)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


# ─── Archipel pour MNIST ───
class MNISTArchipel(ArchipelPhase2):
    def __init__(self):
        super().__init__(
            num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=128,
            top_k=1, max_islands=8, min_islands=2, coherence_variance_threshold=0.3,
        )
        self.encoder = nn.Identity()
        self.mnist_enc = MNISTEncoder(out_dim=128)
    def forward(self, x, t=None, targets=None):
        return super().forward(self.mnist_enc(x), targets=targets if targets is not None else t)
    def kill_island(self, i, distill=True, dl=None):
        dl2 = dl if dl else self._dataloader_for_distillation
        return super().kill_island(i, distill=distill, dataloader=dl2, encoder=self.mnist_enc)


# ─── Baseline MLP ───
class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MNISTEncoder(out_dim=128)
        self.clf = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
                                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        return self.clf(self.encoder(x))


# ─── Évaluation ───
@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    corr = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, ArchipelPhase2):
            preds = model(x)["output"].argmax(1)
        else:
            preds = model(x).argmax(1)
        corr += (preds == y).sum().item()
        tot += y.size(0)
    model.train()
    return corr / tot


# ─── Spécialisation ───
@torch.no_grad()
def compute_specialization_matrix(model, loader, device="cpu"):
    model.eval()
    num_islands = model.num_islands
    num_classes = 10
    counts = torch.zeros(num_islands, num_classes, dtype=torch.int64)
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        island_preds = out["island_outputs"].argmax(dim=2)
        for b in range(island_preds.size(0)):
            for i in range(num_islands):
                counts[i, island_preds[b, i].item()] += 1
    model.train()
    return counts.float()


def specialization_score(matrix):
    total = matrix.sum(dim=1, keepdim=True).clamp(min=1)
    probs = matrix / total
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
    max_entropy = torch.log(torch.tensor(matrix.size(1), dtype=torch.float32))
    return 1.0 - (entropy.mean() / max_entropy).item()


# ─── Main ───
def main(epochs=2, batch_size=64):
    device = "cpu"
    torch.manual_seed(123)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_ld = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_ld = DataLoader(mnist_test, batch_size=512, shuffle=False, num_workers=0)

    # ── Archipel ──
    print("=" * 55)
    print("ARCHIPEL — MNIST avec corrections Courant")
    print("=" * 55)
    model = MNISTArchipel()
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    courant = Courant(num_islands=4)
    model._dataloader_for_distillation = train_ld
    curriculum = TopKCurriculum(num_islands=4, k_init=3, k_final=1, warmup_steps=max(1, len(train_ld)))

    t0 = time.time()
    logs, _ = train_loop_lifecycle(model, train_ld, opt, courant, epochs=epochs, device=device, log_every=50, top_k_curriculum=curriculum)
    t1 = time.time()

    acc_arch = evaluate(model, test_ld, device)
    spec_matrix = compute_specialization_matrix(model, test_ld, device)
    spec_matrix_func, spec_preds, spec_targets = compute_specialization_matrix_with_predictions(model, test_ld, device)
    spec_score = specialization_score(spec_matrix)
    spec_score_func = specialization_score_precision_weighted(spec_matrix_func, spec_preds, spec_targets)
    b_arch = sum(1 for l in logs if l.get("event") == "birth")
    d_arch = sum(1 for l in logs if l.get("event") == "death")
    loss_final = logs[-1]["loss"]

    print(f"Accuracy test: {acc_arch:.4f}")
    print(f"Loss finale: {loss_final:.4f}")
    print(f"Births/Deaths: {b_arch}/{d_arch}")
    print(f"Score spécialisation entropique: {spec_score:.4f}")
    print(f"Score spécialisation fonctionnel: {spec_score_func:.4f}")
    print(f"\nMatrice spécialisation (îles × classes):")
    print("      ", end="")
    for c in range(10):
        print(f"{c:>8}", end="")
    print()
    for i in range(spec_matrix.size(0)):
        print(f"Île {i}: ", end="")
        for c in range(10):
            print(f"{int(spec_matrix[i, c].item()):>8}", end="")
        print()

    # Classe dominante par île
    print("\nClasse dominante par île:")
    for i in range(spec_matrix.size(0)):
        dom = spec_matrix[i].argmax().item()
        pct = spec_matrix[i, dom].item() / spec_matrix[i].sum().item() * 100
        print(f"  Île {i} → classe {dom} ({pct:.1f}%)")

    # ── MLP Baseline ──
    print("\n" + "=" * 55)
    print("MLP BASELINE")
    print("=" * 55)
    mlp = MLPBaseline()
    mlp.train()
    opt2 = optim.Adam(mlp.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt2.zero_grad()
            loss = crit(mlp(xb), yb)
            loss.backward()
            opt2.step()
        print(f"  Epoch {ep+1}/{epochs}  loss={loss.item():.4f}")

    acc_mlp = evaluate(mlp, test_ld, device)
    print(f"\nAccuracy test MLP: {acc_mlp:.4f}")

    # ── Comparaison ──
    print("\n" + "=" * 55)
    print("RÉSULTATS")
    print("=" * 55)
    gap = acc_arch - acc_mlp
    print(f"  Accuracy: Archipel={acc_arch:.4f} | MLP={acc_mlp:.4f} | gap={gap:+.4f}")
    print(f"  Spécialisation entropique: {spec_score:.4f}")
    print(f"  Spécialisation fonctionnelle: {spec_score_func:.4f}")
    if spec_score_func > 0.3:
        print("  ✅ Spécialisation fonctionnelle correcte !")
    elif spec_score_func > 0.15:
        print("  ⚠️ Spécialisation fonctionnelle partielle")
    else:
        print("  ❌ Spécialisation fonctionnelle insuffisante")
    if gap > 0.01:
        print("  ✅ Archipel surpasse MLP")
    elif gap < -0.01:
        print("  ⚠️ MLP surpasse Archipel")
    else:
        print("  → Équivalents")

    # Sauvegarde
    results = {
        "archipel": {"test_acc": round(acc_arch, 4), "loss_final": round(loss_final, 4),
                     "births": b_arch, "deaths": d_arch,
                     "spec_score": round(spec_score, 4),
                     "spec_score_func": round(spec_score_func, 4)},
        "mlp": {"test_acc": round(acc_mlp, 4), "loss_final": round(loss.item(), 4)},
        "epochs": epochs
    }
    out_path = ROOT / "validation_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
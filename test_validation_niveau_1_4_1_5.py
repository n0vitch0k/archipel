"""Validation Niveau 1.4/1.5 — Test set MNIST + Baseline MLP + Spécialisation.

Ce script importe depuis les modules intégrés (pattern integration-before-validation).
Utilisation : python test_validation_niveau_1_4_1_5.py --epochs 50 --batch-size 128
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant
from archipel.current.topk_curriculum import TopKCurriculum
from archipel.utils.specialization_matrix import (
    compute_specialization_matrix,
    compute_specialization_matrix_with_predictions,
    specialization_score,
    specialization_score_precision_weighted,
)


# ─── Encodeur MNIST (CNN 28×28 → 128) ───
class MNISTEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32*7*7,out_dim)
    def forward(self,x): return self.fc(self.conv(x).flatten(1))


# ─── Archipel avec top_k=1 pour spécialisation ───
class MNISTArchipel(ArchipelPhase2):
    def __init__(self):
        super().__init__(
            num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=128,
            top_k=1,  # CRITIQUE : top_k=1 pour spécialisation
            max_islands=8, min_islands=2, coherence_variance_threshold=0.3
        )
        self.encoder = nn.Identity()  # Double encodage empêché
        self.mnist_encoder = MNISTEncoder(out_dim=128)
    
    def forward(self, x, t=None, targets=None):
        return super().forward(self.mnist_encoder(x), targets=targets if targets is not None else t)
    
    def kill_island(self, island_id, distill=True, dataloader=None):
        dl = dataloader if dataloader is not None else self._dataloader_for_distillation
        return super().kill_island(island_id, distill=distill, dataloader=dl, encoder=self.mnist_encoder)


# ─── Baseline MLP équivalente capacité ───
class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MNISTEncoder(out_dim=128)
        self.clf = nn.Sequential(
            nn.Linear(128,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10)
        )
    def forward(self,x): return self.clf(self.encoder(x))


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
        corr += (preds == y).sum().item(); tot += y.size(0)
    model.train()
    return corr / tot


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    
    device = "cpu"
    torch.manual_seed(42)
    
    # Données
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    train_ld = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_ld = DataLoader(mnist_test, batch_size=512, shuffle=False, num_workers=0)
    
    # ── Archipel ──
    print("="*55)
    print(f"ARCHIPEL — {args.epochs} époques MNIST, top_k=1")
    print("="*55)
    model = MNISTArchipel()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    courant = Courant(num_islands=4)
    model._dataloader_for_distillation = train_ld
    curriculum = TopKCurriculum(num_islands=4, k_init=3, k_final=1, warmup_steps=6)

    logs, _ = train_loop_lifecycle(model, train_ld, opt, courant, epochs=args.epochs, device=device, log_every=100, top_k_curriculum=curriculum)
    
    acc_arch = evaluate(model, test_ld, device)
    spec_matrix = compute_specialization_matrix(model, test_ld, device)
    spec_matrix_func, spec_preds, spec_targets = compute_specialization_matrix_with_predictions(model, test_ld, device)
    spec_score = specialization_score(spec_matrix)
    spec_score_func = specialization_score_precision_weighted(spec_matrix_func, spec_preds, spec_targets)
    
    b_arch = sum(1 for l in logs if l.get('event')=='birth')
    d_arch = sum(1 for l in logs if l.get('event')=='death')
    final_loss = logs[-1]['loss']
    
    print(f"Test Acc: {acc_arch:.4f} | Loss: {final_loss:.4f}")
    print(f"Births: {b_arch} | Deaths: {d_arch}")
    print(f"Specialization entropy-score: {spec_score:.4f}")
    print(f"Specialization functional-score: {spec_score_func:.4f}")
    print(f"Matrice:\n{spec_matrix.int()}")
    
    # ── MLP Baseline ──
    print("\n" + "="*55)
    print("MLP BASELINE")
    print("="*55)
    mlp = MLPBaseline()
    opt2 = optim.Adam(mlp.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    for ep in range(args.epochs):
        for xb, yb in train_ld:
            opt2.zero_grad(); loss = crit(mlp(xb), yb); loss.backward(); opt2.step()
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1}/{args.epochs} loss={loss.item():.4f}")
    
    acc_mlp = evaluate(mlp, test_ld, device)
    print(f"Test Acc: {acc_mlp:.4f} | Loss: {loss.item():.4f}")
    
    # ── Comparaison ──
    print("\n" + "="*55)
    print("RÉSULTATS")
    print("="*55)
    print(f"{'':30} {'Archipel':>10} {'MLP':>10}")
    print(f"{'Test Accuracy':30} {acc_arch:>10.4f} {acc_mlp:>10.4f}")
    print(f"{'Specialization entropy':30} {spec_score:>10.4f} {'N/A':>10}")
    print(f"{'Specialization functional':30} {spec_score_func:>10.4f} {'N/A':>10}")
    
    delta = acc_arch - acc_mlp
    if delta > 0.01: print(f"\n✅ Archipel surpasse MLP de +{delta:.4f}")
    elif delta < -0.01: print(f"\n⚠️ MLP surpasse Archipel de {abs(delta):.4f}")
    else: print(f"\n→ Équivalents (delta={delta:+.4f})")
    
    # Validation Niveau 1.8 : curriculum dynamique + routing usage + spécialisation fonctionnelle.
    if spec_score_func > 0.3:
        print(f"\n✅ Spécialisation stricte visible : functional {spec_score_func:.4f} > 0.3")
    elif spec_score_func > 0.15:
        print(f"\n⚠️ Spécialisation partielle : functional {spec_score_func:.4f}; stabilisation à poursuivre")
    else:
        print(f"\nℹ️ Spécialisation stricte non conclue sur ce run : functional {spec_score_func:.4f}; vérifier logs routing EMA")


if __name__ == "__main__":
    main()
"""
Niveau 1.4/1.5 — Validation sur test set + baseline MLP non modulaire.

1. Entraîne Archipel 50 époques, évalue sur test set officiel MNIST (10k)
2. Entraîne MLP équivalent (même capacité paramétrique), évalue sur test set
3. Compare les deux
"""
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant
from archipel.models.mnist import MNISTArchipel
from archipel.baselines.mlp import MLPBaseline
from archipel.utils.specialization_matrix import compute_specialization_matrix, specialization_score


# ─── Évaluation ───
def evaluate(model, loader, device="cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if isinstance(model, ArchipelPhase2):
                out = model(x)
                preds = out["output"].argmax(dim=1)
            else:
                logits = model(x)
                preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    model.train()
    return correct / total


# ─── Main ───
def main(epochs: int = 50, batch_size: int = 128):
    device = "cpu"
    seed = 123
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    mnist_test  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(mnist_test,  batch_size=512,        shuffle=False, num_workers=0)

    # ── Archipel ──
    print("=" * 60)
    print("ARCHIPEL — entraînement 50 époques")
    print("=" * 60)
    model = MNISTArchipel()
    model.train()
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    courant = Courant(num_islands=4)

    acc_train_before = evaluate(model, train_loader, device)
    acc_test_before  = evaluate(model, test_loader,  device)
    print(f"Accuracy train avant : {acc_train_before:.4f}")
    print(f"Accuracy test  avant : {acc_test_before:.4f}")

    logs, _ = train_loop_lifecycle(
        model, train_loader, opt, courant,
        epochs=epochs, device=device, log_every=100,
    )
    final_loss = logs[-1]["loss"]

    acc_train_after = evaluate(model, train_loader, device)
    acc_test_after  = evaluate(model, test_loader,  device)

    # Spécialisation à la fin
    spec_matrix = compute_specialization_matrix(model, test_loader, device)
    spec_score  = specialization_score(spec_matrix)

    total_b = sum(1 for l in logs if l.get("event") == "birth")
    total_d = sum(1 for l in logs if l.get("event") == "death")

    print(f"\nAccuracy train après : {acc_train_after:.4f}")
    print(f"Accuracy test  après : {acc_test_after:.4f}")
    print(f"Loss finale          : {final_loss:.4f}")
    print(f"Îlots finaux         : {model.num_islands}")
    print(f"Births               : {total_b}")
    print(f"Deaths               : {total_d}")
    print(f"Score spécialisation : {spec_score:.4f}")

    print("\nMatrice de spécialisation (îles × classes) :")
    print("      ", end="")
    for c in range(10):
        print(f"{c:>8}", end="")
    print()
    for i in range(spec_matrix.size(0)):
        print(f"Île {i}: ", end="")
        for c in range(10):
            print(f"{spec_matrix[i, c].item():>8.0f}", end="")
        print()

    # Classe dominante par île
    print("\nClasse dominante par île :")
    for i in range(spec_matrix.size(0)):
        dom = spec_matrix[i].argmax().item()
        pct = spec_matrix[i, dom].item() / spec_matrix[i].sum().item() * 100
        print(f"  Île {i} → classe {dom} ({pct:.1f}% des prédictions)")

    archipel_results = {
        "acc_train": round(acc_train_after, 4),
        "acc_test":  round(acc_test_after, 4),
        "loss_final": round(final_loss, 4),
        "num_islands": model.num_islands,
        "births": total_b,
        "deaths": total_d,
        "spec_score": round(spec_score, 4),
    }

    # ── Baseline MLP ──
    print("\n" + "=" * 60)
    print("BASELINE MLP — entraînement 50 époques")
    print("=" * 60)
    mlp = MLPBaseline()
    mlp.train()
    opt_mlp = optim.Adam(mlp.parameters(), lr=1e-3)
    crit    = nn.CrossEntropyLoss()

    acc_train_mlp_before = evaluate(mlp, train_loader, device)
    acc_test_mlp_before  = evaluate(mlp, test_loader,  device)
    print(f"Accuracy train avant : {acc_train_mlp_before:.4f}")
    print(f"Accuracy test  avant : {acc_test_mlp_before:.4f}")

    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt_mlp.zero_grad()
            logits = mlp(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt_mlp.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs}  loss={loss.item():.4f}")

    acc_train_mlp_after = evaluate(mlp, train_loader, device)
    acc_test_mlp_after  = evaluate(mlp, test_loader,  device)
    final_loss_mlp = loss.item()

    print(f"\nAccuracy train après : {acc_train_mlp_after:.4f}")
    print(f"Accuracy test  après : {acc_test_mlp_after:.4f}")
    print(f"Loss finale          : {final_loss_mlp:.4f}")

    mlp_results = {
        "acc_train": round(acc_train_mlp_after, 4),
        "acc_test":  round(acc_test_mlp_after, 4),
        "loss_final": round(final_loss_mlp, 4),
    }

    # ── Comparaison ──
    print("\n" + "=" * 60)
    print("COMPARAISON ARCHIPEL vs BASELINE MLP")
    print("=" * 60)
    print(f"{'Métrique':>25} | {'Archipel':>10} | {'MLP':>10} | {'Diff':>8}")
    print("-" * 62)
    diff_train = archipel_results['acc_train'] - mlp_results['acc_train']
    diff_test  = archipel_results['acc_test']  - mlp_results['acc_test']
    diff_loss  = archipel_results['loss_final'] - mlp_results['loss_final']
    print(f"{'Accuracy train':>25} | {archipel_results['acc_train']:>10.4f} | {mlp_results['acc_train']:>10.4f} | {diff_train:>+8.4f}")
    print(f"{'Accuracy test':>25} | {archipel_results['acc_test']:>10.4f} | {mlp_results['acc_test']:>10.4f} | {diff_test:>+8.4f}")
    print(f"{'Loss finale':>25} | {archipel_results['loss_final']:>10.4f} | {mlp_results['loss_final']:>10.4f} | {diff_loss:>+8.4f}")

    gap_test = archipel_results['acc_test'] - mlp_results['acc_test']
    if gap_test > 0.005:
        print(f"\n✅ Archipel surpasse la baseline de +{gap_test:.4f} sur le test set")
    elif gap_test < -0.005:
        print(f"\n⚠️  Archipel est en dessous de la baseline de {gap_test:.4f}")
    else:
        print(f"\n→ Archipel et la baseline sont équivalents sur le test set")

    # Sauvegarde
    results = {"archipel": archipel_results, "mlp": mlp_results}
    out_path = ROOT / "validation_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés : {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)

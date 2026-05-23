"""Test more steps to find when the version conflict occurs."""
import sys, os
base = r"C:\Users\BROU WILLIAMS\Downloads\Archipel_Project_Complete"
sys.path.insert(0, os.path.join(base, "archipel", "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset
from archipel.training.loop_lifecycle import ArchipelPhase2
from archipel.current.courant import Courant

print("=== 20-step test WITH deposit_all ===")
model = ArchipelPhase2(4, 128, 64, 32, top_k=2, max_islands=8, min_islands=2,
                        coherence_variance_threshold=0.3)
courant = Courant(num_islands=4)
loader = DataLoader(TensorDataset(torch.randn(64, 128), torch.randint(0, 10, (64,))), batch_size=8)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

try:
    for i, (x, y) in enumerate(loader):
        out = model(x, targets=y)
        loss = out["output"].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  Step {i} OK loss={loss.item():.4f}")
        if i >= 19:
            break
    print("PASSED 20 steps")
except RuntimeError as e:
    print(f"FAILED at step {i}: {e}")

print()
print("=== 20-step test WITHOUT deposit_all ===")
model2 = ArchipelPhase2(4, 128, 64, 32, top_k=2, max_islands=8, min_islands=2,
                         coherence_variance_threshold=0.3)
courant2 = Courant(num_islands=4)
loader2 = DataLoader(TensorDataset(torch.randn(64, 128), torch.randint(0, 10, (64,))), batch_size=8)
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
model2.ocean.deposit_all = lambda x: None

try:
    for i, (x, y) in enumerate(loader2):
        out = model2(x, targets=y)
        loss = out["output"].mean()
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        print(f"  Step {i} OK loss={loss.item():.4f}")
        if i >= 19:
            break
    print("PASSED 20 steps without deposit_all")
except RuntimeError as e:
    print(f"FAILED at step {i}: {e}")
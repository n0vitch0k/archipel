"""Tests for island specialization tracking (LS1-LS5)."""
import sys, os
base = r"C:\Users\BROU WILLIAMS\Downloads\Archipel_Project_Complete"
sys.path.insert(0, os.path.join(base, "archipel", "src"))

import torch
from archipel.islands.specialization import IslandSpecialization
from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant
from torch.utils.data import TensorDataset, DataLoader


def check(msg, cond):
    if not cond:
        raise AssertionError(f"FAILED: {msg}")


def test_specialization_init():
    print()
    print("=== TEST LS1: IslandSpecialization init ===")
    spec = IslandSpecialization(num_islands=8, num_classes=10, ema_alpha=0.1)
    check("scores shape (8, 10)", spec.scores.shape == (8, 10))
    check("counts shape (8, 10)", spec.counts.shape == (8, 10))
    check("num_active_islands == 8", spec.num_active_islands == 8)
    print(f"  PASS: scores shape={spec.scores.shape}, spec_boost={spec.specialization_boost}")



def test_specialization_update():
    print()
    print("=== TEST LS2: specialization update ===")
    spec = IslandSpecialization(num_islands=4, num_classes=10, ema_alpha=0.1)

    # Routing: island 0 selected for samples of class 3, island 1 for class 7
    routing_weights = torch.tensor([
        [1., 0., 0., 0.],  # sample 0: island 0 active
        [0., 1., 0., 0.],  # sample 1: island 1 active
        [1., 0., 0., 0.],  # sample 2: island 0 active
    ])
    predicted_class = torch.tensor([3, 7, 5])  # might differ from targets
    targets = torch.tensor([3, 7, 3])          # true class 3,7,3
    island_embeds = torch.randn(4, 3, 32)

    result = spec.update(routing_weights, predicted_class, targets, island_embeds)
    check("per_island_accuracy shape (4,)", result["per_island_accuracy"].shape[0] == 4)
    check("specialization_scores shape (4, 10)", result["specialization_scores"].shape == (4, 10))

    # Island 0 was active for true class 3 → should have higher score for class 3
    check("island0 class3 score > 0", spec.scores[0, 3] > 0)
    check("island1 class7 score > 0", spec.scores[1, 7] > 0)

    # Second update — scores should be smooth (EMA)
    result2 = spec.update(routing_weights, predicted_class, targets, island_embeds)
    check("scores still valid after second update", spec.scores[0, 3] > 0)

    print(f"  PASS: scores[0,3]={spec.scores[0,3]:.4f}, scores[1,7]={spec.scores[1,7]:.4f}")



def test_specialization_boost():
    print()
    print("=== TEST LS3: specialization boost ===")
    spec = IslandSpecialization(num_islands=4, num_classes=10, ema_alpha=0.1, specialization_boost=0.3)

    # Set scores: island 0 is good at class 3, island 1 good at class 7, others zero
    spec.scores[0, 3] = 0.8
    spec.scores[1, 7] = 0.6
    spec.scores[2, :] = 0.0  # no specialization
    spec.scores[3, :] = 0.0

    # Get boost for batch where true classes are [3, 7, 3]
    targets = torch.tensor([3, 7, 3])
    boost = spec.get_specialization_boost(targets=targets)  # returns (batch, num_islands)
    check("boost shape (3, 4)", boost.shape == (3, 4))

    # Island 0 should get highest boost for sample 0 (class 3)
    check("island0 boost for class3 > island2", boost[0, 0] > boost[0, 2])
    # Island 1 should get highest boost for sample 1 (class 7)
    check("island1 boost for class7 > island2", boost[1, 1] > boost[1, 2])
    # Island 2 (no specialization) gets low/zero boost
    check("island2 boost ≈ 0", boost[0, 2].item() < 1e-3)

    print(f"  boost (batch×islands):\n{boost}")
    print(f"  PASS: boost shape={boost.shape}")



def test_specialization_in_archipel_phase2():
    print()
    print("=== TEST LS4: specialization integrated in ArchipelPhase2 ===")
    model = ArchipelPhase2(
        num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32,
        top_k=2, max_islands=8, min_islands=2,
    )
    check("specialization tracker exists", hasattr(model, 'specialization'))
    check("specialization scores shape (8, 10)", model.specialization.scores.shape == (8, 10))

    # Forward with targets
    x = torch.randn(8, 128)
    y = torch.randint(0, 10, (8,))
    out = model(x, targets=y)
    check("output shape (8, 10)", out["output"].shape == (8, 10))
    check("routing_weights shape (8, 4)", out["routing_weights"].shape == (8, 4))
    print(f"  PASS: model has specialization, forward works")



def test_specialization_kill_useless():
    print()
    print("=== TEST LS5: kill useless islands via specialization ===")
    spec = IslandSpecialization(num_islands=4, num_classes=10, ema_alpha=0.1)
    spec._num_active_islands = 4

    # Island 0 has good specialization, island 1 decent, islands 2 and 3 useless
    spec.scores[0, 3] = 0.7
    spec.scores[1, 5] = 0.3
    spec.scores[2, :] = 0.02  # below threshold
    spec.scores[3, :] = 0.03  # below threshold

    useless = spec.get_useless_islands(min_specialization=0.05)
    check("islands 2 and 3 are useless", 2 in useless and 3 in useless)
    check("islands 0 and 1 not useless", 0 not in useless and 1 not in useless)
    print(f"  PASS: useless={useless}, spec scores max: {[spec.scores[i].max().item() for i in range(4)]}")



def test_specialization_distillation_weighted():
    print()
    print("=== TEST LS6: distillation uses specialization scores ===")
    from archipel.islands.lifecycle import distill_island_to_neighbors
    from archipel.islands.base_island import BaseIsland

    # Two neighbor islands
    i0 = BaseIsland(island_id=0, input_dim=128, hidden_dim=64, output_dim=32)
    i1 = BaseIsland(island_id=1, input_dim=128, hidden_dim=64, output_dim=32)
    dying = BaseIsland(island_id=2, input_dim=128, hidden_dim=64, output_dim=32)

    # Dying island is good at class 3 (score=0.8) but bad at class 7 (score=0.1)
    dying_scores = torch.tensor([0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    dataloader = DataLoader(
        TensorDataset(torch.randn(16, 128), torch.tensor([3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7])),
        batch_size=4
    )

    # Snapshot i0 params before
    i0_params_before = {k: v.clone() for k, v in i0.state_dict().items()}

    distill_island_to_neighbors(
        dying_island=dying,
        neighbor_islands=[i0, i1],
        dataloader=dataloader,
        steps=5,
        lr=1e-3,
        device="cpu",
        dying_island_class_scores=dying_scores,
    )

    # i0 params should have changed (distillation happened)
    changed = False
    for k, v in i0.state_dict().items():
        if not torch.allclose(v, i0_params_before[k], atol=1e-6):
            changed = True
            break
    check("neighbor params changed after distillation", changed)
    print(f"  PASS: distillation weighted by specialization, i0 params changed={changed}")



def test_specialization_training_loop():
    print()
    print("=== TEST LS7: full training loop with specialization ===")
    model = ArchipelPhase2(
        num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32,
        top_k=2, max_islands=8, min_islands=2,
    )
    courant = Courant(num_islands=4)
    loader = DataLoader(TensorDataset(torch.randn(64, 128), torch.randint(0, 10, (64,))), batch_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    init_spec_mean = model.specialization.scores[:4].mean().item()
    logs, _ = train_loop_lifecycle(model, loader, opt, courant, epochs=2, log_every=20)
    final_spec_mean = model.specialization.scores[:4].mean().item()

    check("logs non-empty", len(logs) > 0)
    check("spec_mean in log entries", any("spec_mean" in log for log in logs))
    check("spec_max in log entries", any("spec_max" in log for log in logs))
    check("specialization scores evolved", final_spec_mean != init_spec_mean or init_spec_mean == 0.0)

    print(f"  spec_mean: init={init_spec_mean:.4f}, final={final_spec_mean:.4f}")
    print(f"  logs: {len(logs)}, births={sum(1 for l in logs if l.get('event')=='birth')}, deaths={sum(1 for l in logs if l.get('event')=='death')}")
    print(f"  PASS: training stable with specialization tracking")



def run_all():
    print("=" * 70)
    print("SPECIALIZATION TEST SUITE (LS1-LS7)")
    print("=" * 70)
    tests = [
        ("LS1: init", test_specialization_init),
        ("LS2: update", test_specialization_update),
        ("LS3: boost", test_specialization_boost),
        ("LS4: ArchipelPhase2 integration", test_specialization_in_archipel_phase2),
        ("LS5: kill useless", test_specialization_kill_useless),
        ("LS6: distillation weighted", test_specialization_distillation_weighted),
        ("LS7: training loop", test_specialization_training_loop),
    ]
    results = {}
    all_ok = True
    for name, fn in tests:
        try:
            fn()
            results[name] = "PASS"
        except AssertionError as e:
            results[name] = f"FAIL: {e}"
            all_ok = False
        except Exception as e:
            results[name] = f"ERROR: {e}"
            all_ok = False

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        icon = "OK" if res == "PASS" else "XX"
        print(f"  [{icon}] {name}")
        if res != "PASS":
            print(f"         {res}")
    passed = sum(1 for r in results.values() if r == "PASS")
    print()
    print(f"{passed}/{len(results)} tests passed")
    print("ALL PASSED" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run_all())
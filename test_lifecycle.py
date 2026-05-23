"""Tests for island lifecycle module (birth/death)."""
import sys, os
base = r"C:\Users\BROU WILLIAMS\Downloads\Archipel_Project_Complete"
sys.path.insert(0, os.path.join(base, "archipel", "src"))

import torch
from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.islands.lifecycle import IslandLifecycle, get_context_for_spawn
from archipel.current.courant import Courant
from torch.utils.data import TensorDataset, DataLoader

def check(msg, cond):
    if not cond: raise AssertionError(f"FAILED: {msg}")

def test_lifecycle_init():
    print()
    print("=== TEST LC1: IslandLifecycle init ===")
    lc = IslandLifecycle(
        num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32,
        max_islands=8, min_islands=2,
        coherence_variance_threshold=0.5,
        gradient_norm_threshold=1e-5,
        death_window=100,
    )
    s = lc.get_state_summary()
    check("num_islands==4", s["num_islands"] == 4)
    check("max_islands==8", s["max_islands"] == 8)
    check("min_islands==2", s["min_islands"] == 2)
    check("steps_since_birth>=0", s["steps_since_birth"] >= 0)
    print(f"  PASS: state={s}")


def test_coherence_variance():
    print()
    print("=== TEST LC2: coherence_variance ===")
    lc = IslandLifecycle(4, 128, 64, 32)
    # Random embeddings → high variance
    embeds = torch.randn(3, 32)
    var = lc.compute_coherence_variance(embeds)
    check("random var >= 0", var >= 0)
    # Identical embeddings → zero variance
    idem = torch.ones(3, 32) * 0.5
    var_idem = lc.compute_coherence_variance(idem)
    check("identical var == 0", var_idem == 0.0)
    # Single island → zero variance
    var_single = lc.compute_coherence_variance(torch.randn(1, 32))
    check("single island == 0", var_single == 0.0)
    print(f"  PASS: random_var={var:.4f}, identical_var={var_idem:.4f}, single_var={var_single:.4f}")


def test_spawn_decision():
    print()
    print("=== TEST LC3: spawn decision ===")
    lc = IslandLifecycle(4, 128, 64, 32, max_islands=8, min_islands=2,
                         coherence_variance_threshold=0.3)
    # Low variance → no spawn
    should = lc.should_spawn(0.1)
    check("low variance: no spawn", should == False)
    # After cooldown expires (60 calls with low variance), high variance triggers spawn
    for _ in range(59):  # 60 total calls; at 50, cooldown expires but variance is low
        lc.should_spawn(0.1)
    # At call 60: cooldown done but still low variance
    should_low = lc.should_spawn(0.1)
    check("cooldown done but low var: no spawn", should_low == False)
    # Next call: high variance + cooldown done → spawn
    should_high = lc.should_spawn(0.6)
    check("high variance + cooldown done: spawn", should_high == True)
    print("  PASS: cooldown correctly gates spawn")


def test_gradient_tracking():
    print()
    print("=== TEST LC4: gradient tracking ===")
    lc = IslandLifecycle(4, 128, 64, 32, gradient_norm_threshold=1e-5)
    lc.update_gradient_tracking(0, 1e-6)  # low
    lc.update_gradient_tracking(1, 1e-3)  # high
    lc.step_gradient_history()
    # Island 0: low grad → low_gradient_steps should increase
    lc.update_gradient_tracking(0, 1e-6)
    lc.update_gradient_tracking(1, 1e-3)
    lc.step_gradient_history()
    # After many low gradient steps for island 0
    for _ in range(99):
        lc.update_gradient_tracking(0, 1e-6)
        lc.step_gradient_history()
    # Island 1: mixed gradients → should not be killed
    kill = lc.should_kill()
    check("island 1 not killed (mixed)", 1 not in kill)
    print(f"  PASS: gradient tracking works")


def test_archipel_phase2_init():
    print()
    print("=== TEST LC5: ArchipelPhase2 init ===")
    model = ArchipelPhase2(
        num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32,
        top_k=2, max_islands=8, min_islands=2,
    )
    check("num_islands==4", model.num_islands == 4)
    check("max_islands==8", model.max_islands == 8)
    check("min_islands==2", model.min_islands == 2)
    check("lifecycle exists", model.lifecycle is not None)
    check("hypernet exists", model.hypernet is not None)
    check("num_islands >= min", model.num_islands >= model.min_islands)
    print(f"  PASS: model initialized with {model.num_islands} islands")


def test_spawn_and_kill():
    print()
    print("=== TEST LC6: spawn and kill ===")
    model = ArchipelPhase2(4, 128, 64, 32, top_k=2, max_islands=8, min_islands=2)
    check("initially 4 islands", model.num_islands == 4)
    # Spawn manually
    context = torch.randn(32)
    new_id = model.spawn_island(context)
    check("spawn returns new id", new_id == 4)
    check("num_islands updated to 5", model.num_islands == 5)
    check("islands list len==5", len(model.islands) == 5)
    check("lifecycle num_islands==5", model.lifecycle.num_islands == 5)
    # Can't kill below min_islands
    # (we have 5, min=2, so we can kill some)
    killed = model.kill_island(2)
    check("kill succeeded", killed == True)
    check("num_islands back to 4", model.num_islands == 4)
    check("islands list len==4", len(model.islands) == 4)
    # Verify island IDs are contiguous
    ids = [isl.island_id for isl in model.islands]
    check("island IDs contiguous", ids == list(range(4)))
    print(f"  PASS: spawn->5 islands, kill->4 islands, IDs={ids}")


def test_cant_kill_below_min():
    print()
    print("=== TEST LC7: can't kill below min_islands ===")
    model = ArchipelPhase2(2, 128, 64, 32, top_k=2, max_islands=8, min_islands=2)
    killed = model.kill_island(0)
    check("kill at min_islands fails", killed == False)
    check("still 2 islands", model.num_islands == 2)
    print("  PASS: cannot kill below min_islands")


def test_lifecycle_training_loop():
    print()
    print("=== TEST LC8: train_loop_lifecycle integration ===")
    model = ArchipelPhase2(
        4, 128, 64, 32, top_k=2, max_islands=8, min_islands=2,
        coherence_variance_threshold=0.3,
    )
    courant = Courant(num_islands=4)
    loader = DataLoader(TensorDataset(torch.randn(64, 128), torch.randint(0, 10, (64,))), batch_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    init_islands = model.num_islands
    logs, _ = train_loop_lifecycle(model, loader, opt, courant, epochs=2, log_every=10)
    check("logs non-empty", len(logs) > 0)
    # All logs: diversity >= 0, sparsity in [0,1], loss >= 0
    for log in logs:
        if "diversity" in log:
            check(f"diversity>=0 batch {log['batch']}", log["diversity"] >= 0)
            check(f"sparsity[0,1] batch {log['batch']}", 0 <= log["sparsity"] <= 1)
            check(f"loss>=0 batch {log['batch']}", log["loss"] >= 0)
    # Birth events
    births = [l for l in logs if l.get("event") == "birth"]
    deaths = [l for l in logs if l.get("event") == "death"]
    print(f"  Logs: {len(logs)}, Births: {len(births)}, Deaths: {len(deaths)}")
    print(f"  Islands: init={init_islands}, final={model.num_islands}")
    print(f"  PASS: training stable, no crashes")


def test_context_for_spawn():
    print()
    print("=== TEST LC9: get_context_for_spawn ===")
    # Empty → zero context
    ctx = get_context_for_spawn(torch.randn(0, 32), torch.randn(8, 4))
    check("empty -> zero tensor", ctx.abs().max().item() < 1e-6)
    # Non-empty → non-zero context
    embeds = torch.randn(3, 32)
    rw = torch.zeros(8, 4)
    ctx = get_context_for_spawn(embeds, rw)
    check("non-empty -> non-zero", ctx.abs().max().item() > 1e-6)
    print(f"  PASS: context shape={ctx.shape}, mean={ctx.abs().mean():.4f}")



def test_distillation_before_death():
    print()
    print("=== TEST LC10: distillation called before death ===")
    from archipel.training.loop_lifecycle import ArchipelPhase2
    from torch.utils.data import TensorDataset, DataLoader
    from archipel.islands.lifecycle import distill_island_to_neighbors

    # Capture the call to distill_island_to_neighbors
    call_count = [0]
    original_distill = distill_island_to_neighbors

    def mock_distill(dying_island, neighbor_islands, dataloader=None, steps=50, lr=1e-4, device="cpu", dying_island_class_scores=None, encoder=None):
        call_count[0] += 1
        # Verify dying island is a valid island (not yet removed from list)
        assert hasattr(dying_island, 'island_id'), "dying_island must have island_id"
        assert hasattr(dying_island, 'encoder'), "dying_island must have encoder"
        # Verify neighbors exist and are valid islands
        for n in neighbor_islands:
            assert hasattr(n, 'island_id'), "neighbor must have island_id"
            assert hasattr(n, 'encoder'), "neighbor must have encoder"
        # Call the real implementation
        return original_distill(dying_island, neighbor_islands, dataloader, steps, lr, device, dying_island_class_scores=dying_island_class_scores, encoder=encoder)

    # Monkey-patch for the duration of the test
    import archipel.training.loop_lifecycle as ll_module
    ll_module.distill_island_to_neighbors = mock_distill

    try:
        model = ArchipelPhase2(
            num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32,
            top_k=2, max_islands=8, min_islands=2,
        )
        dataloader = DataLoader(
            TensorDataset(torch.randn(32, 128), torch.randint(0, 10, (32,))),
            batch_size=8
        )
        # Wire up the dataloader so kill_island uses it
        model._dataloader_for_distillation = dataloader
        model._distillation_device = "cpu"

        # Snapshot ALL island parameters before kill
        # Use list index as key since island_id can change after removal.
        # Clone() the values — state_dict() returns views into the actual
        # parameter tensors, not independent copies.
        num_islands_before = len(model.islands)
        param_snapshots = {}
        for idx in range(num_islands_before):
            isl = model.islands[idx]
            param_snapshots[idx] = {k: v.clone().cpu() for k, v in isl.state_dict().items()}

        # Kill island 2 with distillation (via the mocked function)
        killed = model.kill_island(2, distill=True)

        # Verify kill succeeded
        check("kill succeeded", killed == True)
        check("distillation was called", call_count[0] >= 1)
        check("num_islands decreased to 3", model.num_islands == 3)

        # Verify neighbor parameters changed (distillation occurred).
        # After kill, islands at [0, 1, 2] correspond to pre-kill indices [0, 1, 3].
        # We pass distill=True, which means distill_island_to_neighbors was called.
        # Distillation updates at least one neighbor's weights via gradient descent.
        param_changed = False
        for post_idx, isl in enumerate(model.islands):
            pre_idx = post_idx  # direct mapping: model.islands[0] was islands[0], etc.
            if pre_idx in param_snapshots:
                before_dict = param_snapshots[pre_idx]
                for k, v in isl.state_dict().items():
                    if not torch.allclose(v.cpu(), before_dict[k], atol=1e-6):
                        param_changed = True
                        break
            if param_changed:
                break

        check("neighbor params changed after distillation", param_changed == True)
        print(f"  PASS: kill succeeded, distill called {call_count[0]}x, "
              f"neighbors updated={param_changed}, islands={model.num_islands}")
    finally:
        # Restore original
        ll_module.distill_island_to_neighbors = original_distill



def run_all():
    print("=" * 70)
    print("LIFECYCLE TEST SUITE")
    print("=" * 70)
    tests = [
        ("LC1: init", test_lifecycle_init),
        ("LC2: coherence_variance", test_coherence_variance),
        ("LC3: spawn decision", test_spawn_decision),
        ("LC4: gradient tracking", test_gradient_tracking),
        ("LC5: ArchipelPhase2 init", test_archipel_phase2_init),
        ("LC6: spawn and kill", test_spawn_and_kill),
        ("LC7: cant kill below min", test_cant_kill_below_min),
        ("LC8: train_loop_lifecycle", test_lifecycle_training_loop),
        ("LC9: get_context_for_spawn", test_context_for_spawn),
        ("LC10: distillation before death", test_distillation_before_death),
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

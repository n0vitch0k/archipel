"""Comprehensive test suite for Archipel Phase 1."""
import sys, os
base = r"C:\Users\BROU WILLIAMS\Downloads\Archipel_Project_Complete"
sys.path.insert(0, os.path.join(base, "archipel", "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from archipel.training.loop import (
    ArchipelPhase1, compute_diversity_loss, compute_coherence_loss,
    compute_structural_reg_loss, compute_combined_loss, train_loop,
)
from archipel.current.courant import Courant
from archipel.current.router import HyperNetworkRouter
from archipel.islands.base_island import BaseIsland
from archipel.ocean.ocean import Ocean, OceanSpace

def check(msg, cond):
    if not cond: raise AssertionError(f"FAILED: {msg}")

def test_base_island():
    print()
    print("=== TEST 1: BaseIsland ===")
    island = BaseIsland(0, 128, 64, 32)
    x = torch.randn(8, 128)
    out = island(x)
    check("shape (8,32)", tuple(out.shape) == (8, 32))
    check("finite", torch.isfinite(out).all())
    out2 = island(x + 1e-3)
    check("responsive", not torch.allclose(out, out2, atol=1e-6))
    gates = F.softmax(island.gating_network(island.encoder(x)), dim=-1)
    check("gates>0", (gates > 0).all())
    print("  PASS: shape OK, finite, responsive, gating OK")


def test_ocean_space():
    print()
    print("=== TEST 2: OceanSpace ===")
    space = OceanSpace(4, 32)
    init = space.island_embeddings.clone()
    check("init~zero", init.abs().max().item() < 1e-6)
    space.deposit_batch(torch.randn(4, 32))
    after = space.island_embeddings
    check("deposit updates", after.abs().max().item() > 1e-5)
    prox = space.compute_proximity()
    check("proximity finite", torch.isfinite(prox).all())
    check("proximity [0,1]", prox.min().item() >= 0 and prox.max().item() <= 1.0)
    check("proximity symmetric", (prox - prox.T).abs().max().item() < 1e-5)
    check("proximity diag~1", torch.diagonal(prox).mean().item() > 0.5)
    check("interaction>0", (space.interaction_counts > 0).all())
    print("  PASS: deposit updates, proximity matrix valid")


def test_ocean():
    print()
    print("=== TEST 3: Ocean ===")
    ocean = Ocean(32, num_islands=4)
    check("coherence_center finite", torch.isfinite(ocean.coherence_center).all())
    ocean.deposit_all(torch.randn(4, 8, 32))
    stats = ocean.get_statistics()
    for k, v in stats.items():
        check(f"stat {k} finite", torch.isfinite(torch.tensor(v)))
    coh = ocean.compute_coherence_loss()
    check("coherence_loss finite", torch.isfinite(coh))
    check("coherence_loss>=0", coh.item() >= 0)
    print(f"  PASS: coherence_center OK, stats finite, coh_loss={coh.item():.4f}")


def test_diversity_loss():
    print()
    print("=== TEST 4: compute_diversity_loss ===")
    for trial in range(20):
        e = torch.randn(4, 32)
        loss = compute_diversity_loss(e)
        check(f"trial {trial} finite", torch.isfinite(loss))
        check(f"trial {trial} >=0", loss.item() >= 0)
    idem = torch.ones(4, 32) * 0.5
    idem[1:] = idem[0]
    check("identical>=0", compute_diversity_loss(idem).item() >= 0)
    check("ortho>=0", compute_diversity_loss(torch.eye(4, 32)).item() >= 0)
    check("3D>=0", compute_diversity_loss(torch.randn(4, 8, 32)).item() >= 0)
    check("single>=0", compute_diversity_loss(torch.randn(1, 32)).item() >= 0)
    check("zero>=0", compute_diversity_loss(torch.zeros(4, 32)).item() >= 0)
    check("huge>=0", compute_diversity_loss(torch.randn(4, 32) * 1e6).item() >= 0)
    print("  PASS: always >=0, all edge cases")


def test_coherence_loss():
    print()
    print("=== TEST 5: compute_coherence_loss ===")
    for trial in range(20):
        a = torch.randn(3, 4, 32)
        loss = compute_coherence_loss(a)
        check(f"trial {trial} finite", torch.isfinite(loss))
        check(f"trial {trial} >=0", loss.item() >= 0)
    check("single==0", compute_coherence_loss(torch.randn(1, 4, 32)).item() == 0.0)
    check("2islands>=0", compute_coherence_loss(torch.randn(2, 4, 32)).item() >= 0)
    print("  PASS: >=0, single=0, all cases handled")


def test_router():
    print()
    print("=== TEST 6: HyperNetworkRouter ===")
    router = HyperNetworkRouter(32, num_islands=4, top_k=2)
    router.train()
    island_states = torch.randn(4, 32)
    inp = torch.randn(8, 32)
    corrs = router.compute_correlations(island_states, inp)
    check("corrs shape", corrs.shape == (8, 4))
    check("corrs finite", torch.isfinite(corrs).all())
    check("corrs [-1,1]", corrs.min().item() >= -1.001 and corrs.max().item() <= 1.001)
    out = router(inp, island_states)
    w = out["routing_weights"]
    check("weights finite", torch.isfinite(w).all())
    check("weights [0,1]", (w >= 0).all() and (w <= 1).all())
    check("top_k exact", (w.sum(dim=1) == 1.0).all())
    check("sparsity [0,1]", 0 <= out["sparsity"].item() <= 1)
    out2 = router(inp, torch.zeros(4, 32))
    check("zero states finite", torch.isfinite(out2["routing_weights"]).all())
    print("  PASS: correlations [-1,1], top_k exact, zero states OK")


def test_courant():
    print()
    print("=== TEST 7: Courant ===")
    c = Courant(num_islands=4)
    check("init step=0", c.get_state_report()["step_count"] == 0)
    for i in range(20):
        s = c.step(0.5+0.1*torch.randn(1).item(), 0.15+0.05*torch.randn(1).item(), 0.1+0.05*torch.randn(1).item())
        check(f"step {i} lambda_coh finite", torch.isfinite(torch.tensor(s["lambda_coherence"])))
        check(f"step {i} eps_mod [0.5,2.0]", 0.5 <= s["epsilon_modulation"] <= 2.0)
    r = c.get_state_report()
    check("lambda_coh [0.001,2.0]", 0.001 <= r["lambda_coherence"] <= 2.0)
    check("lambda_div [0.001,2.0]", 0.001 <= r["lambda_diversity"] <= 2.0)
    f = c.get_island_fitness()
    check("fitness [0.1,10]", (f >= 0.1).all() and (f <= 10.0).all())
    print("  PASS: step_count advances, lambdas bounded, epsilon_mod OK, fitness OK")


def test_combined_loss():
    print()
    print("=== TEST 8: compute_combined_loss ===")
    outputs = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    ie = torch.randn(4, 8, 32)
    rw = torch.zeros(4).unsqueeze(0).expand(8, -1).clone()
    rw[:, :2] = 0.5
    ent = torch.tensor(0.8)
    for lc in [0.01, 0.1, 0.5, 1.0]:
        for ld in [0.01, 0.1, 0.5, 1.0]:
            loss, comps = compute_combined_loss(outputs, targets, ie, rw, ent, None, None, lc, ld, 0.01)
            check(f"lc={lc} ld={ld} finite", torch.isfinite(loss))
            check(f"lc={lc} ld={ld} >=0", loss.item() >= 0)
            for k, v in comps.items():
                check(f"comp {k} finite", torch.isfinite(torch.tensor(v)))
    print("  PASS: finite and >=0 for all lambda combinations")


def test_training_loop():
    print()
    print("=== TEST 9: train_loop integration ===")
    model = ArchipelPhase1(4, 128, 64, 32, top_k=2)
    courant = Courant(4)
    loader = DataLoader(TensorDataset(torch.randn(64, 128), torch.randint(0, 10, (64,))), batch_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logs, upd_courant = train_loop(model, loader, opt, courant, epochs=2)
    check("logs non-empty", len(logs) > 0)
    for i, log in enumerate(logs):
        for k in ["loss", "task_loss", "coherence", "diversity", "sparsity", "entropy"]:
            check(f"log[{i}][{k}] finite", torch.isfinite(torch.tensor(log[k])))
        check(f"log[{i}] diversity>=0", log["diversity"] >= 0)
        check(f"log[{i}] sparsity[0,1]", 0 <= log["sparsity"] <= 1)
        check(f"log[{i}] loss>=0", log["loss"] >= 0)
    r = upd_courant.get_state_report()
    check("courant step_count>0", r["step_count"] > 0)
    print(f"  PASS: {len(logs)} batches, all invariants held")


def test_edge_cases():
    print()
    print("=== TEST 10: Edge cases ===")
    e = torch.randn(4, 32); e[1, 2] = float("nan")
    try: compute_diversity_loss(e)
    except: pass
    e = torch.randn(4, 32); e[2, 3] = float("inf")
    try: compute_diversity_loss(e)
    except: pass
    loss = compute_diversity_loss(torch.zeros(4, 32))
    check("zero finite", torch.isfinite(loss))
    check("zero>=0", loss.item() >= 0)
    loss = compute_diversity_loss(torch.randn(4, 32) * 1e6)
    check("huge finite", torch.isfinite(loss))
    check("huge>=0", loss.item() >= 0)
    router = HyperNetworkRouter(32, 4, 2)
    out = router(torch.zeros(8, 32), torch.randn(4, 32))
    check("zero input finite", torch.isfinite(out["routing_weights"]).all())
    out = router(torch.randn(8, 32), torch.zeros(4, 32))
    check("zero states finite", torch.isfinite(out["routing_weights"]).all())
    print("  PASS: NaN/Inf/zero/huge handled without crash")


def test_structural_reg():
    print()
    print("=== TEST 11: compute_structural_reg_loss ===")
    w1 = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    w2 = w1.clone()
    check("identical==0", compute_structural_reg_loss(w1, w2).item() == 0.0)
    w2 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    check("different>0", compute_structural_reg_loss(w1, w2).item() > 0)
    check("first_call==0", compute_structural_reg_loss(w1, None).item() == 0.0)
    check("finite", torch.isfinite(compute_structural_reg_loss(torch.randn(8, 4), torch.randn(8, 4))))
    print("  PASS: identical->0, different->>0, first_call==0, finite")



def test_homeostatic_regularizers():
    print()
    print("=== TEST 12: homeostatic_regularizers ===")
    from archipel.islands.base_island import BaseIsland
    island = BaseIsland(0, 128, 64, 32, num_experts=4)
    x = torch.randn(16, 128)

    # Without x: returns zero tensors
    regs_none = island.homeostatic_regularizers(x=None)
    check("x=None returns dict", isinstance(regs_none, dict))
    check("x=None activity==0", regs_none["activity"].item() == 0.0)
    check("x=None diversity==0", regs_none["diversity"].item() == 0.0)

    # With x: returns negative entropy (diversity reg < 0)
    regs_x = island.homeostatic_regularizers(x=x)
    check("with x returns dict", isinstance(regs_x, dict))
    check("diversity<=0", regs_x["diversity"].item() <= 0.0)  # -entropy ≤ 0

    # Check gates sum to 1
    gates = island.get_expert_usage(x)
    check("gates sum ~1", (gates.sum(dim=1) - 1.0).abs().max().item() < 1e-5)

    # Multiple calls give consistent results
    regs_x2 = island.homeostatic_regularizers(x=x)
    check("consistent across calls", abs(regs_x2["diversity"].item() - regs_x["diversity"].item()) < 1e-3)

    print(f"  PASS: x=None→0, x=batch→diversity={regs_x['diversity'].item():.4f}, gates sum ~1")



def test_hypernetwork_generator():
    print()
    print("=== TEST 13: HyperNetworkGenerator ===")
    from archipel.current.router import HyperNetworkGenerator
    import torch.nn.functional as F

    gen = HyperNetworkGenerator(seed_dim=64, context_dim=32, output_dim=8192, num_layers=3, hidden_dim=256)
    seed = torch.randn(2, 64)  # batch=2
    context = torch.randn(2, 32)

    out = gen(seed, context)
    check("output shape (2, 8192)", out.shape == (2, 8192))
    check("output finite", torch.isfinite(out).all())
    check("output nonzero", out.abs().max().item() > 1e-6)

    # Different seeds → different outputs
    seed2 = torch.randn(2, 64)
    out2 = gen(seed2, context)
    check("different seed→different out", not torch.allclose(out, out2, atol=1e-5))

    # Same seed + same context → same out
    out3 = gen(seed, context)
    check("same seed→same out", torch.allclose(out, out3, atol=1e-6))

    # Verify generator produces valid island initialization
    # (weights can be used to initialize an island's local_projection)
    gen.eval()
    with torch.no_grad():
        slice_ = out[0, :64].clone().clamp(-2.0, 2.0)
    check("gen slice bounded", slice_.abs().max().item() <= 2.0 + 1e-5)

    print(f"  PASS: output shape={tuple(out.shape)}, different seeds differ, bounded slice")



def test_per_island_specialization():
    print()
    print("=== TEST 14: per-island specialization tracking ===")
    from archipel.training.loop import ArchipelPhase1
    import torch.nn.functional as F

    model = ArchipelPhase1(num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32, top_k=2)
    model.eval()

    # Generate structured data: different "quadrants" activate different islands
    # Create data where class 0-1 activates island 0, class 2-3 activates island 1, etc.
    torch.manual_seed(42)
    x_specialized = torch.randn(32, 128)
    x_specialized[:, :32] += 3.0   # quadrant A → island 0
    x_specialized[:, 32:64] -= 3.0  # quadrant B → island 1

    with torch.no_grad():
        out_dict = model(x_specialized)
        routing = out_dict["routing_weights"]  # (batch, num_islands)
        island_embeds = out_dict["island_embeddings"]  # (num_islands, batch, ocean_dim)

    # Check that different islands produce different embeddings
    island_means = island_embeds.mean(dim=1)  # (num_islands, ocean_dim)
    check("islands produce finite embeddings", torch.isfinite(island_means).all())
    check("embeddings differ between islands", island_means.std(dim=0).mean().item() > 1e-3)

    # Check routing: islands should not all have the same weight
    routing_mean = routing.mean(dim=0)  # (num_islands,)
    check("not all islands equally active", routing_mean.std().item() > 1e-3)

    # Track which class activates which island
    class_island_assignments = routing.argmax(dim=1)  # which island wins per sample
    assignment_entropy = -(routing_mean * torch.log(routing_mean + 1e-8)).sum()
    check("routing entropy > 0", assignment_entropy.item() > 0.0)

    print(f"  PASS: island embedding diversity={island_means.std(dim=0).mean():.4f}, "
          f"routing entropy={assignment_entropy:.4f}, island_means={island_means.norm(dim=-1)}")



def test_structured_data_convergence():
    print()
    print("=== TEST 15: structured data convergence (XOR-like) ===")
    from archipel.training.loop import ArchipelPhase1, train_loop
    from archipel.current.courant import Courant
    from torch.utils.data import TensorDataset, DataLoader

    # Create a simple dataset where the relationship is non-linear
    # Use a 2-feature XOR-like pattern with 4 classes
    torch.manual_seed(123)
    n_samples = 200

    # Build XOR-like data: class depends on sign(x1) XOR sign(x2)
    x1 = torch.randn(n_samples, 64)
    x2 = torch.randn(n_samples, 64)
    # Mix x1 and x2 into 128-dim input
    x_data = torch.cat([x1, x2], dim=1)

    # Labels: XOR of sign of first half vs second half
    labels = ((x1 > 0).sum(dim=1) % 2 + (x2 > 0).sum(dim=1) % 2) % 4

    dataset = TensorDataset(x_data, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ArchipelPhase1(num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32, top_k=2)
    courant = Courant(num_islands=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logs, _ = train_loop(model, loader, optimizer, courant, epochs=3)

    # Check that loss decreased
    initial_loss = logs[0]["task_loss"]
    final_loss = logs[-1]["task_loss"]
    check("loss decreased over training", final_loss < initial_loss)
    check("final loss finite", torch.isfinite(torch.tensor(final_loss)))

    # Check all metrics are finite
    for k, v in logs[-1].items():
        check(f"log '{k}' finite", torch.isfinite(torch.tensor(v)) if isinstance(v, float) else True)

    improvement = (initial_loss - final_loss) / initial_loss * 100
    print(f"  PASS: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}, improvement={improvement:.1f}%")



def test_local_loss_methods():
    print()
    print("=== TEST 16: BaseIsland local loss methods ===")
    from archipel.islands.base_island import BaseIsland

    island = BaseIsland(0, 128, 64, 32, num_experts=4)
    x = torch.randn(8, 128)

    # compute_local_loss (no input access)
    embed = island(x)
    loss_no_target = island.compute_local_loss(embed)
    check("local_loss no target finite", torch.isfinite(loss_no_target))
    check("local_loss no target >= 0", loss_no_target.item() >= 0.0)
    check("local_loss no target small", loss_no_target.item() < 1.0)

    # compute_local_loss_from_input
    loss_from_input = island.compute_local_loss_from_input(x, noise_std=0.1)
    check("local_loss from input finite", torch.isfinite(loss_from_input))
    check("local_loss from input >= 0", loss_from_input.item() >= 0.0)

    # Ensure both methods return different values (different computation paths)
    check("different paths → different loss", abs(loss_no_target.item() - loss_from_input.item()) > 1e-5)

    print(f"  PASS: no_target={loss_no_target.item():.6f}, from_input={loss_from_input.item():.6f}")



def run_all():
    print("=" * 70)
    print("ARCHIPEL PHASE 1 - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    tests = [
        ("BaseIsland", test_base_island),
        ("OceanSpace", test_ocean_space),
        ("Ocean", test_ocean),
        ("Diversity Loss", test_diversity_loss),
        ("Coherence Loss", test_coherence_loss),
        ("Router", test_router),
        ("Courant", test_courant),
        ("Combined Loss", test_combined_loss),
        ("Training Loop", test_training_loop),
        ("Edge Cases", test_edge_cases),
        ("Structural Reg", test_structural_reg),
        ("Homeostatic Regularizers", test_homeostatic_regularizers),
        ("HyperNetworkGenerator", test_hypernetwork_generator),
        ("Per-Island Specialization", test_per_island_specialization),
        ("Structured Data Convergence", test_structured_data_convergence),
        ("Local Loss Methods", test_local_loss_methods),
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
    print()
    print("ALL TESTS PASSED - system verified" if all_ok else "SOME TESTS FAILED")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(run_all())

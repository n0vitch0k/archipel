"""Tests for dynamic top-k curriculum and routing usage diagnostics."""
import sys, os
base = r"C:\Users\BROU WILLIAMS\Downloads\Archipel_Project_Complete"
sys.path.insert(0, os.path.join(base, "archipel", "src"))

import math
import torch

from archipel.current.topk_curriculum import (
    TopKCurriculum,
    RoutingUsageTracker,
    compute_effective_top_k,
    compute_normalized_entropy,
)
from archipel.current.router import HyperNetworkRouter
from archipel.islands.specialization import IslandSpecialization
from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.current.courant import Courant
from torch.utils.data import DataLoader, TensorDataset


def check(msg, cond):
    if not cond:
        raise AssertionError(f"FAILED: {msg}")


def test_topk_curriculum_linear_descent_freeze_and_clamp():
    print()
    print("=== TEST DTK1: top-k curriculum schedule ===")
    curriculum = TopKCurriculum(num_islands=4, k_init=3, k_final=1, warmup_steps=300)
    checks = [
        (0, 3),
        (100, 2),
        (150, 2),
        (200, 2),
        (299, 2),
        (300, 1),
        (1000, 1),
    ]
    for step, expected in checks:
        actual = curriculum.get_top_k(step)
        check(f"step {step} -> {expected}", actual == expected)

    frozen = TopKCurriculum(num_islands=4, k_init=3, k_final=1, warmup_steps=300, freeze_step=250)
    check("freeze_step locks k=1", frozen.get_top_k(250) == 1)
    check("freeze_step stays k=1", frozen.get_top_k(1000) == 1)

    clamped = TopKCurriculum(num_islands=2, k_init=5, k_final=1, warmup_steps=100)
    check("k_init clamped to num_islands", clamped.get_top_k(0) == 2)
    print("  PASS: linear descent, freeze, clamp")


def test_topk_curriculum_freeze_step_none_is_not_inferred():
    print()
    print("=== TEST DTK1b: freeze_step=None stays schedule-driven ===")
    curriculum = TopKCurriculum(num_islands=4, k_init=3, k_final=1, warmup_steps=300, freeze_step=None)
    check("freeze_step remains None", curriculum.freeze_step is None)
    checks = [
        (0, 3),
        (100, 2),
        (299, 2),
        (300, 1),
        (1000, 1),
    ]
    for step, expected in checks:
        actual = curriculum.get_top_k(step)
        check(f"step {step} -> {expected}", actual == expected)
    print("  PASS: None does not create implicit freeze")


def test_routing_usage_metrics_detect_dead_island_hidden_by_entropy():
    print()
    print("=== TEST DTK2: routing usage metrics ===")
    usage = torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3])
    entropy = compute_normalized_entropy(usage)
    check("dead-island usage entropy is not zero", entropy > 0.7)
    check("dead-island usage entropy below perfect", entropy < 1.0)
    check("min_usage_ratio detects dead island", torch.isclose(torch.tensor(entropy), torch.tensor(0.792481), atol=1e-4))

    tracker = RoutingUsageTracker(num_islands=4, beta=0.0, initial_beta=0.0, dead_threshold=0.05)
    routing_weights = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.5, 0.5, 0.0],
    ])
    metrics = tracker.update(routing_weights)
    check("routing_usage_entropy finite", math.isfinite(metrics["routing_usage_entropy"]))
    check("min_usage_ratio detects island 0 dead", metrics["min_usage_ratio"] < 0.05)
    check("dead_island_count == 1", metrics["dead_island_count"] == 1)
    check("effective_top_k finite", math.isfinite(metrics["effective_top_k"]))
    print("  PASS: entropy + min_usage + dead island")


def test_routing_usage_tracker_ema_initializes_uniformly_and_resizes():
    print()
    print("=== TEST DTK3: routing usage tracker EMA ===")
    tracker = RoutingUsageTracker(num_islands=4, beta=0.0, initial_beta=0.0)
    routing_weights = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    metrics = tracker.update(routing_weights)
    expected_usage = torch.tensor([1.0, 0.0, 0.0, 0.0])
    check("EMA follows first batch with beta=0", torch.allclose(tracker.usage_ema, expected_usage))
    check("min_usage_ratio zero", metrics["min_usage_ratio"] == 0.0)
    check("dead_island_count three", metrics["dead_island_count"] == 3)

    tracker.resize(3)
    check("EMA resized to 3 islands", tracker.usage_ema.shape == (3,))
    check("EMA preserves first entries", torch.allclose(tracker.usage_ema[:3], expected_usage[:3]))
    print("  PASS: EMA initialization and resize")


def test_effective_top_k_counts_active_islands_per_sample():
    print()
    print("=== TEST DTK4: effective top-k ===")
    routing_weights = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    effective = compute_effective_top_k(routing_weights, threshold=0.0)
    check("effective_top_k with zero threshold", math.isclose(effective, 2.3333333, rel_tol=0, abs_tol=1e-6))
    effective_sparse = compute_effective_top_k(routing_weights, threshold=0.3)
    check("effective_top_k with threshold", math.isclose(effective_sparse, 1.0, rel_tol=0, abs_tol=1e-6))
    print("  PASS: effective top-k")


def test_router_set_top_k_clamps_and_updates_forward():
    print()
    print("=== TEST DTK5: router dynamic top-k setter ===")
    router = HyperNetworkRouter(32, num_islands=4, top_k=1)
    router.set_top_k(3)
    check("router top_k set to 3", router.top_k == 3)
    router.set_top_k(9)
    check("router top_k clamped to num_islands", router.top_k == 4)
    out = router(torch.randn(8, 32), torch.randn(4, 32))
    check("forward sums to 1", torch.allclose(out["routing_weights"].sum(dim=1), torch.ones(8)))
    check("forward selects up to 4 islands", out["routing_weights"].shape == (8, 4))
    print("  PASS: router top-k setter")


def test_specialization_coverage_detects_class_collapse():
    print()
    print("=== TEST DTK6: specialization coverage ===")
    spec = IslandSpecialization(num_islands=4, num_classes=10)
    spec.scores.copy_(torch.zeros_like(spec.scores))
    spec.scores[0, 3] = 0.8
    spec.scores[1, 7] = 0.7
    spec.scores[2, 3] = 0.6
    spec.scores[3, 7] = 0.6
    summary = spec.get_state_summary()
    check("spec_coverage two classes", summary["spec_coverage"] == 2)
    check("specialized islands four", summary["specialized_island_count"] == 4)
    check("dominant class summary string present", "dominant_classes" in summary)
    check("best class score mean present", "best_class_score_mean" in summary)

    spec.scores.copy_(torch.zeros_like(spec.scores))
    spec.scores[:, 3] = 0.8
    summary = spec.get_state_summary()
    check("collapse coverage one class", summary["spec_coverage"] == 1)
    print("  PASS: coverage detects collapse")


def test_specialization_update_uses_exposure_normalized_score():
    print()
    print("=== TEST DTK6b: functional specialization score ===")
    spec = IslandSpecialization(num_islands=2, num_classes=3, ema_alpha=1.0)
    spec.scores.copy_(torch.zeros_like(spec.scores))

    # Island 0 active on 3 class-1 samples and 1 class-0 sample,
    # so it is correct and over-performing for class 0 after exposure normalization.
    routing_weights = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    predicted_class = torch.tensor([1, 1, 1, 0])
    targets = torch.tensor([1, 1, 1, 0])
    island_embeds = torch.zeros(2, 4, 4)

    result = spec.update(routing_weights, predicted_class, targets, island_embeds)
    check("scores updated", result["specialization_scores"].shape == (2, 3))
    check("island 0 class 0 positive", spec.scores[0, 0].item() > 0)
    check("island 1 remains non-positive", (spec.scores[1] <= 0).all())

    summary = spec.get_state_summary()
    check("coverage one class", summary["spec_coverage"] == 1)
    check("specialized island one", summary["specialized_island_count"] == 1)
    print("  PASS: exposure-normalized specialization")


def test_training_loop_logs_topk_routing_and_specialization_metrics():
    print()
    print("=== TEST DTK7: training loop integration metrics ===")
    torch.manual_seed(7)
    model = ArchipelPhase2(
        num_islands=3,
        input_dim=16,
        hidden_dim=8,
        ocean_dim=8,
        top_k=1,
        max_islands=4,
        min_islands=2,
    )
    curriculum = TopKCurriculum(
        num_islands=3,
        k_init=2,
        k_final=1,
        warmup_steps=3,
        freeze_step=3,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(16, 16), torch.randint(0, 10, (16,))),
        batch_size=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logs, _ = train_loop_lifecycle(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        courant=Courant(num_islands=3),
        epochs=1,
        device="cpu",
        log_every=999,
        top_k_curriculum=curriculum,
    )

    required = {
        "current_top_k",
        "scheduled_top_k",
        "routing_usage_entropy",
        "routing_usage_min",
        "min_usage_ratio",
        "dead_island_count",
        "effective_top_k",
        "spec_coverage",
        "specialized_island_count",
        "spec_std",
        "best_class_score_mean",
        "negative_score_count",
        "dominant_score_max",
        "qualitative_log",
    }
    check("at least one log entry", len(logs) >= 1)
    first = next(log for log in logs if "loss" in log)
    check("required metrics logged", required.issubset(first.keys()))
    check("first scheduled top-k is curriculum k=2", first["scheduled_top_k"] == 2)
    check("routing entropy finite", math.isfinite(first["routing_usage_entropy"]))
    check("min usage finite", math.isfinite(first["min_usage_ratio"]))
    check("effective top-k finite", math.isfinite(first["effective_top_k"]))
    check("qualitative log is string", isinstance(first["qualitative_log"], str))
    check("qualitative log mentions curriculum", "top-k curriculum active" in first["qualitative_log"])
    print("  PASS: training loop exposes curriculum and routing diagnostics")

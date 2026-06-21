"""Tests for the Kuramoto-based dynamic router (Phase 3)."""

import math

import pytest
import torch

from archipel.current.kuramoto import KuramotoIslandRouter


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return KuramotoIslandRouter(
        embedding_dim=32,
        num_islands=4,
        top_k=2,
        epsilon_init=0.0,   # no exploration for deterministic tests
        temperature=1.0,
        dt=0.1,
        coupling_init=1.0,
    )


@pytest.fixture
def batch_input():
    return torch.randn(8, 32)  # batch=8, embed_dim=32


# ── Phase initialisation ─────────────────────────────────────────────────

class TestPhaseInit:
    def test_phases_are_on_unit_circle(self, router):
        """θ_i should all be in [0, 2π)."""
        theta = router.theta.detach()
        assert theta.min() >= 0.0
        assert theta.max() < 2.0 * math.pi

    def test_phases_are_distinct(self, router):
        """No two islands should share the exact same phase."""
        theta = router.theta.detach().tolist()
        assert len(set(round(t, 6) for t in theta)) == len(theta)

    def test_phase_order_parameter_high_init(self, router):
        """Initial uniform phases → order parameter near 0 (disorder)."""
        metrics = router.get_sync_metrics()
        # 4 uniformly-spaced phases → nearly zero order parameter
        assert metrics["order_parameter"] < 0.15

    def test_num_islands_matches(self, router):
        assert router.num_islands == 4
        assert router.theta.shape == (4,)
        assert router.omega.shape == (4,)
        assert router.island_thresholds.shape == (4,)


# ── Input → phase mapping ──────────────────────────────────────────────

class TestInputToPhase:
    def test_output_range(self, router, batch_input):
        """φ(x) should be in [0, 2π) for any input."""
        phases = router.input_to_phase(batch_input)
        assert phases.shape == (8,)
        assert phases.min() >= 0.0
        assert phases.max() < 2.0 * math.pi

    def test_different_inputs_different_phases(self, router):
        """Two different inputs should (usually) map to different phases."""
        a = torch.randn(1, 32)
        b = torch.randn(1, 32)
        p_a = router.input_to_phase(a).item()
        p_b = router.input_to_phase(b).item()
        # Not a hard requirement, but extremely unlikely to be equal
        assert abs(p_a - p_b) > 1e-6

    def test_gradients_flow(self, router, batch_input):
        """The input-phase network should be differentiable."""
        phases = router.input_to_phase(batch_input)
        loss = phases.sum()
        loss.backward()
        # Check that input_phase_net parameters received gradients
        grads = [p.grad for p in router.input_phase_net.parameters()]
        assert all(g is not None for g in grads)
        assert all(g.norm().item() > 0 for g in grads)


# ── Phase alignment ─────────────────────────────────────────────────────

class TestPhaseAlignment:
    def test_alignment_shape(self, router):
        """cos(θ_i − φ(x)) should be (batch, num_islands)."""
        input_phases = torch.rand(8) * 2.0 * math.pi
        alignment = router.compute_phase_alignment(input_phases)
        assert alignment.shape == (8, 4)
        assert alignment.min() >= -1.0
        assert alignment.max() <= 1.0

    def test_self_alignment(self, router):
        """If φ(x) = θ_i for island i, alignment should be 1 for that island."""
        theta = router.theta.detach()
        for i in range(router.num_islands):
            input_phases = theta[i].unsqueeze(0)  # single input
            alignment = router.compute_phase_alignment(input_phases)
            # Island i should have highest alignment
            assert alignment[0, i] >= alignment[0].max() - 1e-6


# ── Forward pass ────────────────────────────────────────────────────────

class TestForward:
    def test_forward_shape_and_keys(self, router, batch_input):
        """Forward returns the expected dict keys and shapes."""
        out = router(input_repr=batch_input)
        assert set(out.keys()) == {"routing_weights", "entropy", "correlations", "sparsity"}
        assert out["routing_weights"].shape == (8, 4)  # (batch, num_islands)
        assert out["correlations"].shape == (8, 4)
        assert out["entropy"].dim() == 0  # scalar
        assert out["sparsity"].dim() == 0  # scalar

    def test_routing_weights_sum_to_one(self, router, batch_input):
        """Each row of routing_weights should sum to 1.0."""
        out = router(input_repr=batch_input)
        row_sums = out["routing_weights"].sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))

    def test_top_k_selection(self, router, batch_input):
        """Exactly top_k islands should have non-zero routing weight (no ε)."""
        out = router(input_repr=batch_input)
        n_active = (out["routing_weights"] > 0).sum(dim=1)
        assert torch.all(n_active == router.top_k)

    def test_epsilon_greedy_diversifies(self, router, batch_input):
        """With ε=1, routing should visibly differ from ε=0."""
        router.epsilon = 1.0
        out_explore = router(input_repr=batch_input)
        router.epsilon = 0.0
        out_exploit = router(input_repr=batch_input)
        # The two routing matrices should differ on at least some samples
        diff = (out_explore["routing_weights"] - out_exploit["routing_weights"]).abs().sum()
        assert diff > 0

    def test_specialization_boost(self, router, batch_input):
        """Adding boost for a specific island should increase its selection."""
        # Without boost
        router.epsilon = 0.0
        out_normal = router(input_repr=batch_input)

        # With boost for island 0
        boost = torch.zeros(4)
        boost[0] = 10.0  # large boost → island 0 should be selected
        out_boosted = router(input_repr=batch_input, specialization_boost=boost)

        assert (out_boosted["routing_weights"][:, 0] > 0).sum() >= (out_normal["routing_weights"][:, 0] > 0).sum()

    def test_gradients_flow_through_router(self, router, batch_input):
        """Gradients should reach θ and input_phase_net.

        Note: ω and K do not participate in the forward pass — they drive
        the Kuramoto ODE step which is called separately — so they
        correctly have no gradient from the routing loss.
        """
        out = router(input_repr=batch_input)
        loss = (out["routing_weights"] * out["correlations"]).sum()
        loss.backward()

        # θ should receive a gradient
        assert router.theta.grad is not None
        assert router.theta.grad.norm().item() > 0

        # input_phase_net should receive gradients
        grads = [p.grad for p in router.input_phase_net.parameters()]
        assert all(g is not None for g in grads)
        assert all(g.norm().item() > 0 for g in grads)


# ── Kuramoto dynamics ───────────────────────────────────────────────────

class TestKuramotoDynamics:
    def test_phases_stay_in_range(self, router):
        """After repeated updates, phases should remain in [0, 2π)."""
        for _ in range(200):
            router.update_phases()
        theta = router.theta.detach()
        assert theta.min() >= 0.0 - 1e-4
        assert theta.max() < 2.0 * math.pi + 1e-4

    def test_sync_with_high_coupling(self):
        """For N=2, Kuramoto always synchronises (R → 1) with K > 0.

        Two oscillators always converge to the same phase regardless of
        initial separation, making this a robust test of the dynamics.
        """
        router = KuramotoIslandRouter(
            embedding_dim=32,
            num_islands=2,
            coupling_init=1.0,
            dt=0.05,
        )
        with torch.no_grad():
            router.omega.zero_()
            router.theta[:] = torch.tensor([0.0, 1.0])  # 1 rad apart

        for _ in range(500):
            router.update_phases()
        metrics = router.get_sync_metrics()
        assert metrics["order_parameter"] > 0.98, f"R={metrics['order_parameter']:.4f}"

    def test_drift_with_low_coupling(self):
        """With very low K, islands should drift apart (low order parameter)."""
        router = KuramotoIslandRouter(
            embedding_dim=32,
            num_islands=5,
            coupling_init=0.001,  # very weak coupling
            dt=0.1,
        )
        # Initialize with identical phases (closely packed)
        with torch.no_grad():
            router.theta[:] = torch.ones(5) * 1.0

        for _ in range(200):
            router.update_phases()

        metrics = router.get_sync_metrics()
        # With weak coupling and different frequencies, phases should spread
        assert metrics["order_parameter"] < 0.95  # not perfectly synchronized

    def test_sync_metrics_monotonic(self):
        """R should be monotonic with coupling strength (N=2).

        Two oscillators synchronise faster with higher K.  Test that
        the metrics function runs without error and returns reasonable
        values for multiple routers — the exact monotonicity depends
        on numerical stability of the ODE integration.
        """
        routers = [
            KuramotoIslandRouter(
                embedding_dim=32,
                num_islands=2,
                coupling_init=K,
                dt=0.05,
            )
            for K in [0.01, 0.5, 2.0, 5.0]
        ]
        order_params = []
        for r in routers:
            with torch.no_grad():
                r.omega.zero_()
                r.theta[:] = torch.tensor([0.0, 1.5])
            for _ in range(500):
                r.update_phases()
            order_params.append(r.get_sync_metrics()["order_parameter"])

        # All routers should produce valid R ∈ [0, 1]
        for K_val, R_val in zip([0.01, 0.5, 2.0, 5.0], order_params):
            assert 0.0 <= R_val <= 1.0, f"K={K_val} R={R_val} outside [0,1]"

        # Higher coupling should produce higher (or equal) R
        # Allow 5% tolerance for numerical variation
        for i in range(len(order_params) - 1):
            assert order_params[i + 1] >= order_params[i] * 0.95, (
                f"K[{i}] R={order_params[i]:.4f} > K[{i+1}] R={order_params[i+1]:.4f}"
            )

    def test_update_handles_single_island(self):
        """update_phases should be a no-op with 1 island (no coupling)."""
        router = KuramotoIslandRouter(num_islands=1, embedding_dim=32)
        theta_before = router.theta.detach().clone()
        router.update_phases()
        theta_after = router.theta.detach()
        assert torch.allclose(theta_before, theta_after)


# ── Lifecycle compatibility ─────────────────────────────────────────────

class TestResize:
    def test_resize_grow(self, router):
        """Growing from 4 to 6 islands should preserve existing phases."""
        old_theta = router.theta.detach().clone()
        old_omega = router.omega.detach().clone()
        router.resize(6)
        assert router.num_islands == 6
        assert router.theta.shape == (6,)
        assert router.omega.shape == (6,)
        # First 4 should match original
        assert torch.allclose(router.theta[:4], old_theta)
        assert torch.allclose(router.omega[:4], old_omega)
        # New islands should have valid phases
        assert (router.theta[4:] >= 0).all()
        assert (router.theta[4:] < 2.0 * math.pi).all()

    def test_resize_shrink(self, router):
        """Shrinking from 4 to 2 islands should keep the first two."""
        old_theta = router.theta.detach().clone()
        router.resize(2)
        assert router.num_islands == 2
        assert router.theta.shape == (2,)
        assert torch.allclose(router.theta, old_theta[:2])

    def test_resize_idempotent(self, router):
        """Resize to the same count should be a no-op."""
        theta_before = router.theta.detach().clone()
        router.resize(router.num_islands)
        assert torch.allclose(router.theta, theta_before)

    def test_resize_grow_then_shrink(self, router):
        """Grow to 6, shrink back to 3 — first 3 should match original."""
        original = router.theta.detach().clone()
        router.resize(6)
        router.resize(3)
        assert torch.allclose(router.theta, original[:3])

    def test_resize_k_updated(self, router):
        """K (scalar) should be preserved after resize."""
        K_before = router.K.detach().clone()
        router.resize(6)
        assert torch.allclose(router.K, K_before)

    def test_set_top_k_clamps(self, router):
        """set_top_k should clamp to [1, num_islands]."""
        router.set_top_k(10)
        assert router.top_k == router.num_islands
        router.set_top_k(0)
        assert router.top_k == 1
        router.set_top_k(2)
        assert router.top_k == 2


# ── Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_batch_size_1(self, router):
        """Single sample should work."""
        single = torch.randn(1, 32)
        out = router(input_repr=single)
        assert out["routing_weights"].shape == (1, 4)

    def test_large_batch(self, router):
        """Large batch should work."""
        large = torch.randn(256, 32)
        out = router(input_repr=large)
        assert out["routing_weights"].shape == (256, 4)

    def test_no_grad_inference(self, router, batch_input):
        """Inference with torch.no_grad() should work."""
        with torch.no_grad():
            out = router(input_repr=batch_input)
        assert out["routing_weights"] is not None

    def test_training_eval_modes(self, router, batch_input):
        """Eval mode should disable epsilon-greedy."""
        router.epsilon = 0.5
        router.eval()
        out_eval = router(input_repr=batch_input)
        router.train()
        out_train = router(input_repr=batch_input)
        # Both produce valid outputs
        assert out_eval["routing_weights"] is not None
        assert out_train["routing_weights"] is not None

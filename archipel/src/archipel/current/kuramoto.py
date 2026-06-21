"""Kuramoto-based dynamic router for Archipel Phase 3.

Phase 3 replaces cosine-similarity routing with coupled Kuramoto
oscillators.  Each island has a scalar phase θ_i ∈ [0, 2π) and a natural
frequency ω_i.  Inputs are projected to a target phase φ(x) via a learnable
mapping.  Routing scores are based on phase alignment cos(θ_i − φ(x)).

After each batch, island phases evolve via the Kuramoto ODE:
    dθ_i/dt = ω_i + K · Σ_j sin(θ_j − θ_i)

Islands that co-process the same inputs synchronise their phases; islands
that process different inputs drift apart — specialisation emerges from
the physics of the system.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KuramotoIslandRouter(nn.Module):
    """Phase-based router using Kuramoto coupled oscillators.

    Each island has a scalar phase θ_i ∈ [0, 2π) and a natural frequency
    ω_i.  Inputs are projected to a target phase φ(x) ∈ [0, 2π) by a small
    MLP.  Routing scores are computed as cos(θ_i − φ(x)).

    After each batch :meth:`update_phases` advances all island phases
    according to the Kuramoto ODE (vectorised Euler step).

    The forward interface mirrors :class:`HyperNetworkRouter` so the two
    can be swapped in ``ArchipelPhase3`` with no glue code.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        num_islands: int = 4,
        top_k: int = 2,
        epsilon_init: float = 0.1,
        temperature: float = 1.0,
        dt: float = 0.1,
        coupling_init: float = 1.0,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_islands = num_islands
        self.top_k = top_k
        self.epsilon = epsilon_init
        self.temperature = temperature
        self.dt = dt

        # ── Learnable oscillator parameters ──────────────────────────────

        # Initialise phases uniformly around the unit circle for maximum
        # initial diversity, then add small jitter so no two islands start
        # at exactly the same phase.
        theta_init = torch.linspace(0.0, 2.0 * math.pi, steps=num_islands + 1)[:-1]
        theta_init = theta_init + torch.randn_like(theta_init) * 0.05  # jitter
        theta_init = theta_init % (2.0 * math.pi)
        self.register_parameter(
            "theta", nn.Parameter(theta_init, requires_grad=True)
        )

        # Natural frequencies — small random values so islands drift apart
        # even without coupling.  Learning ω lets the system adapt.
        self.register_parameter(
            "omega", nn.Parameter(torch.randn(num_islands) * 0.1, requires_grad=True)
        )

        # Coupling matrix K_ij (num_islands × num_islands) — learnable
        # pairwise coupling strengths.  Stored in log-space: K_ij = softplus(K_raw).
        # This guarantees non-negative coupling (no repulsive dynamics).
        # The matrix is symmetric with zero diagonal, allowing cluster
        # synchronisation to emerge (some pairs couple strongly, others weakly).
        target_init = coupling_init / max(1, num_islands)
        # Inverse of softplus: raw = log(exp(target) - 1)
        K_raw_init_val = math.log(max(math.exp(target_init) - 1.0, 1e-8))
        K_raw = torch.full((num_islands, num_islands), K_raw_init_val)
        # Small noise breaks initial symmetry
        noise = torch.randn(num_islands, num_islands) * 0.01
        K_raw = K_raw + noise
        K_raw = (K_raw + K_raw.T) / 2.0  # symmetrise
        K_raw.fill_diagonal_(0)  # no self-coupling
        # Clamp to avoid extreme values (nan from log(0))
        K_raw = K_raw.clamp(min=-10.0, max=10.0)
        self.register_parameter(
            "K", nn.Parameter(K_raw, requires_grad=True)
        )
        self._coupling_init = coupling_init  # stored for resize

        # ── Cascaded modulator (same role as in HyperNetworkRouter) ──────
        self.register_parameter(
            "island_thresholds",
            nn.Parameter(torch.zeros(num_islands), requires_grad=True),
        )
        self.register_parameter(
            "epsilon_scale",
            nn.Parameter(torch.tensor(1.0), requires_grad=False),
        )

        # ── Input → phase projection ─────────────────────────────────────
        # Maps the encoder's representation (embedding_dim) to a (sin, cos)
        # pair whose atan2 gives the target phase φ(x) ∈ [0, 2π).
        # Two layers + ReLU let the mapping learn non-linear phase contours.
        self.input_phase_net = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # (sin component, cos component)
        )

    # ── Public API  ─────────────────────────────────────────────────────

    def set_top_k(self, top_k: int) -> None:
        """Update the active top-k while keeping it valid."""
        self.top_k = max(1, min(int(top_k), max(1, self.num_islands)))

    def resize(self, num_islands: int) -> None:
        """Resize all oscillator buffers after a birth/death event.

        Args:
            num_islands: New island count (≥ 1).
        """
        old_n = self.num_islands
        if num_islands == old_n:
            return
        if num_islands < 1:
            return

        device = self.theta.device
        with torch.no_grad():
            # ── theta ──
            if num_islands > old_n:
                # New islands: insert phases midway between existing pairs
                # to keep the circle well-covered.
                gap = 2.0 * math.pi / num_islands
                extra = torch.arange(0, num_islands - old_n, device=device, dtype=torch.float32)
                # Offset from a random existing phase
                base_theta = self.theta[0].item() if old_n > 0 else 0.0
                extra_theta = (base_theta + (old_n + extra) * gap) % (2.0 * math.pi)
                new_theta = torch.cat([self.theta.detach(), extra_theta])
            else:
                new_theta = self.theta[:num_islands].detach().clone()
            self.theta = nn.Parameter(new_theta)

            # ── omega ──
            if num_islands > old_n:
                new_omega = torch.cat([self.omega.detach(), torch.zeros(num_islands - old_n, device=device)])
            else:
                new_omega = self.omega[:num_islands].detach().clone()
            self.omega = nn.Parameter(new_omega)

            # ── K (matrix) — reshape for new island count ────────────────
            if num_islands > old_n:
                # New rows/cols: initialise in log-space so that
                # softplus(K_raw) ≈ avg effective coupling of existing pairs.
                with torch.no_grad():
                    K_eff = F.softplus(self.K.detach())
                    avg_eff = K_eff.mean().item() if old_n > 1 else self._coupling_init / max(1, num_islands)
                    # Inverse softplus: raw = log(exp(avg_eff) - 1)
                    avg_raw = max(math.log(max(math.exp(avg_eff) - 1.0, 1e-8)), -10.0)
                new_K = torch.full((num_islands, num_islands), avg_raw, device=device)
                new_K.fill_diagonal_(0)
                new_K[:old_n, :old_n] = self.K.detach()
                self.K = nn.Parameter(new_K.clamp(min=-10.0, max=10.0))
            elif num_islands < old_n:
                self.K = nn.Parameter(self.K[:num_islands, :num_islands].detach().clone())

            # ── island_thresholds ──
            if num_islands > old_n:
                new_t = torch.cat([self.island_thresholds.detach(), torch.zeros(num_islands - old_n, device=device)])
            else:
                new_t = self.island_thresholds[:num_islands].detach().clone()
            self.island_thresholds = nn.Parameter(new_t)

        self.num_islands = num_islands

    # ── Kuramoto dynamics  ──────────────────────────────────────────────

    def get_sync_metrics(self) -> Dict[str, float]:
        """Return scalar metrics describing the current oscillator state.

        * **order_parameter**  — Kuramoto order parameter R ∈ [0, 1],
          1 = perfect global synchronisation, 0 = full disorder.
        * **circular_variance**  — 1 − R.
        * **phase_std** — Circular standard deviation.
        """
        with torch.no_grad():
            theta = self.theta.detach().float()
            N = theta.numel()
            if N == 0:
                return {"order_parameter": 0.0, "circular_variance": 1.0, "phase_std": math.pi}

            # Complex order parameter: R = |(1/N) Σ e^{iθ_j}|
            cos_val = float(torch.cos(theta).sum().detach().item())
            sin_val = float(torch.sin(theta).sum().detach().item())
            R_raw = cos_val * cos_val + sin_val * sin_val
            R = math.sqrt(R_raw) / N if R_raw > 0 else 0.0

            # Circular variance
            circ_var = 1.0 - R

            # Circular std approximation (safe: R ∈ [0, 1] Python float)
            R = min(R, 1.0)  # cap at 1 (floating-point can overshoot)
            if R < 1e-7:
                phase_std = math.pi  # fully disordered
            else:
                phase_std = math.sqrt(-2.0 * math.log(R))

            return {
                "order_parameter": round(R, 4),
                "circular_variance": round(circ_var, 4),
                "phase_std": round(phase_std, 4),
            }

    def update_phases(self, dt: Optional[float] = None) -> None:
        """Advance all island phases by one Kuramoto step (vectorised Euler).

        Uses the circular (shortest-path) phase difference to compute
        the correct coupling on the ring, even when phases wrap around.
        The Kuramoto model on a circle requires sin(θ_j − θ_i) where the
        difference is taken in (−π, π].

        Args:
            dt: Step size.  Falls back to ``self.dt`` if ``None``.
        """
        dt = dt if dt is not None else self.dt
        if self.num_islands < 2:
            return  # nothing to couple

        theta = self.theta.detach()  # (N,)

        # Circular phase difference: θ_j − θ_i  (Kuramoto convention)
        diff = theta.unsqueeze(0) - theta.unsqueeze(1)  # (N, N)
        diff = (diff + math.pi) % (2.0 * math.pi) - math.pi

        # Pairwise coupling: Σ_j K_ij · sin(θ_j − θ_i)  with matrix K_ij
        K_raw = self.K.detach()  # (N, N)
        # Numerically enforce symmetry (gradients can drift it)
        K_raw = (K_raw + K_raw.T) / 2.0
        # Ensure non-negative coupling: negative K_ij would be repulsive
        # and destabilise the dynamics.  Softplus gives a smooth positive
        # lower bound without discontinuities.
        K_mat = F.softplus(K_raw)
        sin_diff = torch.sin(diff)  # (N, N)
        coupling = (K_mat * sin_diff).sum(dim=1)  # (N,)

        # Euler step
        dtheta = self.omega.detach() + coupling  # (N,)
        self.theta.data = (theta + dt * dtheta) % (2.0 * math.pi)

    # ── Input encoding  ─────────────────────────────────────────────────

    def input_to_phase(self, input_repr: torch.Tensor) -> torch.Tensor:
        """Project an encoded input to a target phase φ(x) ∈ [0, 2π).

        Args:
            input_repr: Encoded input (batch_size, embedding_dim).

        Returns:
            Target phases (batch_size,) in radians.
        """
        raw = self.input_phase_net(input_repr)  # (batch, 2)
        sin_val, cos_val = raw[:, 0], raw[:, 1]
        # atan2 returns angles in [-π, π], wrap to [0, 2π)
        phase = torch.atan2(sin_val, cos_val)
        return phase % (2.0 * math.pi)

    # ── Routing  ────────────────────────────────────────────────────────

    def compute_phase_alignment(
        self, input_phases: torch.Tensor
    ) -> torch.Tensor:
        """Compute routing alignment cos(θ_i − φ(x)) for every (input, island) pair.

        Args:
            input_phases: Target phases (batch_size,).

        Returns:
            Alignment scores (batch_size, num_islands) ∈ [-1, 1].
        """
        # input_phases: (batch,) → (batch, 1)
        # self.theta:   (num_islands,) → (1, num_islands)
        diff = input_phases.unsqueeze(1) - self.theta.unsqueeze(0)  # (batch, N)
        return torch.cos(diff)

    def top_k_selection_with_noise(
        self, scores: torch.Tensor, island_fitness: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k islands with adaptive epsilon-greedy exploration.

        Args:
            scores: Phase alignment scores (batch_size, num_islands).
            island_fitness: Optional per-island fitness (ignored in V1).

        Returns:
            routing_weights: Sparse one-hot-ish weights (batch_size, num_islands).
            entropy: Routing entropy for regularisation.
        """
        batch_size, num_islands = scores.shape

        # Softmax → probabilities for entropy computation
        probs = F.softmax(scores / self.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # Top-k deterministic selection
        _, top_indices = torch.topk(scores, k=self.top_k, dim=-1)  # (batch, top_k)

        # Build uniform routing weights for the top-k
        routing_weights = torch.zeros_like(probs)
        routing_weights.scatter_(1, top_indices, 1.0 / self.top_k)

        # Adaptive epsilon-greedy exploration
        if self.training:
            epsilon_k = self.epsilon * self.epsilon_scale.item()
            replace_mask = torch.rand(batch_size, device=scores.device) < epsilon_k
            random_indices = torch.randint(
                0, num_islands, (batch_size, 1), device=scores.device
            )
            for b in range(batch_size):
                if replace_mask[b]:
                    slot = torch.randint(0, self.top_k, (1,), device=scores.device)
                    top_indices[b, slot] = random_indices[b]

            # Rebuild with dedup
            routing_weights = torch.zeros_like(probs)
            for b in range(batch_size):
                unique_idx = torch.unique(top_indices[b], sorted=False)
                routing_weights[b, unique_idx] = 1.0 / unique_idx.numel()

            entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1).mean()

        return routing_weights, entropy

    def forward(
        self,
        input_repr: torch.Tensor,
        island_states: Optional[torch.Tensor] = None,
        island_fitness: Optional[torch.Tensor] = None,
        specialization_boost: Optional[torch.Tensor] = None,
        predicted_class: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: compute routing weights via phase alignment.

        Args:
            input_repr: Encoded input (batch_size, embedding_dim).
            island_states: Ignored (included for API compatibility with
                ``HyperNetworkRouter``).
            island_fitness: Optional per-island fitness (forwarded to top-k).
            specialization_boost: Optional (batch, num_islands) boost to add
                to alignment scores before selection.
            predicted_class: Ignored in V1 (reserved for future use).
            targets: Ignored in V1.

        Returns:
            Dictionary with ``routing_weights``, ``entropy``, ``correlations``
            (the raw alignment scores), and ``sparsity``.
        """
        # Map input to target phase
        input_phases = self.input_to_phase(input_repr)  # (batch,)

        # Phase-alignment scores
        alignment = self.compute_phase_alignment(input_phases)  # (batch, N)

        # Per-island thresholds (learnable bias)
        thresholds = torch.sigmoid(self.island_thresholds) * 0.5
        adjusted = alignment - thresholds.unsqueeze(0)

        # Specialisation boost
        if specialization_boost is not None:
            if specialization_boost.dim() == 1:
                adjusted = adjusted + specialization_boost.unsqueeze(0)
            elif specialization_boost.shape[0] == input_repr.shape[0]:
                adjusted = adjusted + specialization_boost
            elif specialization_boost.shape[-1] == input_repr.shape[0]:
                adjusted = adjusted + specialization_boost.T

        # Top-k selection
        routing_weights, entropy = self.top_k_selection_with_noise(
            adjusted, island_fitness
        )

        return {
            "routing_weights": routing_weights,
            "entropy": entropy,
            "correlations": alignment,  # raw alignment before thresholds/boost
            "sparsity": (routing_weights > 0).float().mean(),
        }

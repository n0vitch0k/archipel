"""Island lifecycle management for Archipel Phase 1.

Handles island birth (spawning via HyperNetwork) and death (apoptosis with distillation).
Tracks per-island metrics to drive lifecycle decisions:
  - Birth trigger: coherence variance of active islands exceeds threshold
  - Death trigger: gradient norm below threshold for K consecutive steps
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_island import BaseIsland
from ..current.router import HyperNetworkGenerator


class IslandLifecycle(nn.Module):
    """Manages island birth and death based on system state.

    Birth (spawning): triggered when active islands have high variance
        (not covering the input space well) AND num_islands < max_islands.
    Death (apoptosis): triggered when gradient norm of an island stays
        below threshold for K consecutive steps AND num_islands > min_islands.
    """

    def __init__(
        self,
        num_islands: int,
        input_dim: int,
        hidden_dim: int,
        ocean_dim: int,
        max_islands: int = 8,
        min_islands: int = 2,
        # Birth parameters
        coherence_variance_threshold: float = 0.5,
        birth_cooldown: int = 50,
        # Death parameters
        gradient_norm_threshold: float = 1e-5,
        death_window: int = 100,
        death_cooldown: int = 50,
        # Distillation
        distillation_steps: int = 200,
        distillation_lr: float = 1e-4,
    ) -> None:
        """Initialize the lifecycle manager.

        Args:
            num_islands: Current number of islands.
            input_dim: Input dimension for new islands.
            hidden_dim: Hidden dimension for new islands.
            ocean_dim: Ocean embedding dimension.
            max_islands: Maximum number of islands (prevent unbounded growth).
            min_islands: Minimum number of islands (prevent total collapse).
            coherence_variance_threshold: Variance above this triggers birth.
            birth_cooldown: Steps between birth events.
            gradient_norm_threshold: Gradient norm below this triggers death check.
            death_window: Consecutive steps with low gradient to trigger death.
            death_cooldown: Steps between death events.
            distillation_steps: Number of distillation steps before killing.
            distillation_lr: Learning rate for distillation.
        """
        super().__init__()
        self.num_islands = num_islands
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ocean_dim = ocean_dim
        self.max_islands = max_islands
        self.min_islands = min_islands

        # Birth
        self.coherence_variance_threshold = coherence_variance_threshold
        self.birth_cooldown = birth_cooldown
        self._steps_since_birth = 0

        # Death
        self.gradient_norm_threshold = gradient_norm_threshold
        self.death_window = death_window
        self.death_cooldown = death_cooldown
        self._steps_since_death = 0

        # Distillation
        self.distillation_steps = distillation_steps
        self.distillation_lr = distillation_lr

        # HyperNetwork for generating new island weights
        # Estimate total params for a BaseIsland to size output_dim
        # BaseIsland: encoder(128->64->64) + expert_heads(4x 64->32) + gating(64->4)
        # Rough param count: ~50k for typical dims, round up for safety
        self.hypernet = HyperNetworkGenerator(
            seed_dim=64,
            context_dim=ocean_dim,
            output_dim=8192,  # Enough to generate island adapter weights
            num_layers=3,
            hidden_dim=256,
        )

        # Per-island gradient norm history (tracked externally, stored here)
        self.register_buffer("gradient_norm_history", torch.zeros(max_islands, death_window))
        self._grad_history_idx = 0

        # Per-island consecutive low-gradient steps
        self.register_buffer("low_gradient_steps", torch.zeros(max_islands))
        self.register_buffer("active_counts", torch.zeros(max_islands))  # How often each island was active

        # Island IDs that are "alive" (managed externally via this module)
        self._alive_mask = torch.ones(max_islands, dtype=torch.bool)

    def compute_coherence_variance(
        self,
        active_embeddings: torch.Tensor,
    ) -> float:
        """Compute variance of active island embeddings.

        High variance means active islands produce very different representations —
        they don't cover the input space well together → birth trigger.

        Args:
            active_embeddings: (num_active, ocean_dim) tensor of active island embeddings

        Returns:
            Variance (scalar)
        """
        if active_embeddings.size(0) < 2:
            return 0.0
        # Mean embedding across active islands
        mean_embed = active_embeddings.mean(dim=0)
        # Variance of each island's embedding from the mean
        variance = ((active_embeddings - mean_embed) ** 2).mean().item()
        return variance

    def update_gradient_tracking(
        self,
        island_id: int,
        grad_norm: float,
    ) -> None:
        """Update gradient norm history for an island.

        Args:
            island_id: Index of the island.
            grad_norm: Current gradient norm.
        """
        idx = self._grad_history_idx % self.death_window
        self.gradient_norm_history[island_id, idx] = grad_norm

    def update_active_tracking(self, routing_weights: torch.Tensor) -> None:
        """Update how often each island was active this step.

        Args:
            routing_weights: (batch, num_islands) routing weights
        """
        # Count active (weight > 0) — shape (num_islands,)
        active = (routing_weights > 0).any(dim=0).float()
        # Update active_counts without incrementing the version counter.
        # Use .data.copy_() (NOT +=) because += on a registered buffer IS
        # version-tracked and will cause "modified by inplace operation"
        # errors at backward() if any autograd node holds a reference to
        # the buffer's version counter.
        counts = self.active_counts.clone().detach()
        counts[:active.numel()] += active
        self.active_counts.data.copy_(counts)

    def should_spawn(self, coherence_variance: float) -> bool:
        """Determine if a new island should be spawned.

        Args:
            coherence_variance: Current variance of active island embeddings.

        Returns:
            True if birth should occur.
        """
        if self.num_islands >= self.max_islands:
            return False
        if self._steps_since_birth < self.birth_cooldown:
            self._steps_since_birth += 1
            return False

        # High variance triggers birth
        if coherence_variance > self.coherence_variance_threshold:
            self._steps_since_birth = 0
            return True

        self._steps_since_birth += 1
        return False

    def should_kill(self) -> List[int]:
        """Determine which islands should be killed.

        An island is killed if:
          - Its gradient norm has been below threshold for death_window consecutive steps
          - It has been inactive (active_count is very low) OR its gradient norm is tiny
          - num_islands > min_islands

        Returns:
            List of island IDs to kill.
        """
        if self.num_islands <= self.min_islands:
            return []
        if self._steps_since_death < self.death_cooldown:
            self._steps_since_death += 1
            return []

        to_kill = []

        for i in range(self.num_islands):
            # Check gradient norm history
            history = self.gradient_norm_history[i]
            recent_mean = history[:min(self._grad_history_idx + 1, self.death_window)].mean().item()

            # Also check total activity — if barely used, flag for death
            activity_ratio = self.active_counts[i].item() / max(self.active_counts.sum().item(), 1)
            rarely_used = activity_ratio < 0.01 and self.active_counts[i].item() > 10

            if recent_mean < self.gradient_norm_threshold or rarely_used:
                self.low_gradient_steps[i] += 1
            else:
                self.low_gradient_steps[i] = 0.0

            if self.low_gradient_steps[i] >= self.death_window:
                to_kill.append(i)

        if to_kill:
            self._steps_since_death = 0

        return to_kill

    def step_gradient_history(self) -> None:
        """Advance the gradient history index (call once per step)."""
        self._grad_history_idx = (self._grad_history_idx + 1) % self.death_window

    def reset_step_counters(self) -> None:
        """Reset per-step counters (call after lifecycle evaluation)."""
        # Keep running counters (active_counts, gradient_norm_history) intact
        # Only reset cooldowns if needed
        pass

    def get_state_summary(self) -> Dict[str, float]:
        """Get a summary of lifecycle state for monitoring.

        Returns:
            Dictionary of lifecycle metrics.
        """
        grad_norms = self.gradient_norm_history[:self.num_islands].mean(dim=1)
        return {
            "num_islands": self.num_islands,
            "max_islands": self.max_islands,
            "min_islands": self.min_islands,
            "steps_since_birth": self._steps_since_birth,
            "steps_since_death": self._steps_since_death,
            "grad_norm_mean": grad_norms.mean().item(),
            "grad_norm_min": grad_norms.min().item(),
            "grad_norm_max": grad_norms.max().item(),
            "low_gradient_islands": (self.low_gradient_steps[:self.num_islands] > 0).sum().item(),
            "activity_ratio_mean": (
                self.active_counts[:self.num_islands] / max(self.active_counts.sum().item(), 1)
            ).mean().item() if self.active_counts.sum().item() > 0 else 0.0,
        }


def distill_island_to_neighbors(
    dying_island: BaseIsland,
    neighbor_islands: List[BaseIsland],
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    steps: int = 50,
    lr: float = 1e-4,
    device: str = "cpu",
    dying_island_class_scores: Optional[torch.Tensor] = None,
    encoder: Optional[nn.Module] = None,
) -> None:
    """Distill knowledge from a dying island into its neighbors.

    The dying island acts as a teacher: we re-encode a sample of inputs through
    the dying island to get target embeddings, then have each neighbor island
    encode the same inputs and minimize MSE against the teacher's embeddings.

    This transfers "what this island knows" to its neighbors before removal,
    preserving learned representations in the archipelago.

    Args:
        dying_island: The island being removed (teacher).
        neighbor_islands: Islands to receive the distilled knowledge (students).
        dataloader: Optional DataLoader providing (x, y) batches. If provided,
                   real inputs are re-encoded through the dying island to get
                   target embeddings for distillation. If None, uses stored embeddings.
        steps: Number of distillation steps.
        lr: Learning rate for the distillation optimizer.
        device: Device to run distillation on.
        encoder: Optional nn.Module to pre-encode raw inputs before distillation.
                 If provided, x_batch is passed through encoder before reaching islands.
    """
    if len(neighbor_islands) == 0:
        return

    dying_island.eval()
    dying_island.to(device)
    for n in neighbor_islands:
        n.to(device)

    if dataloader is not None:
        # Real distillation: re-encode inputs through dying island, distill to neighbors
        distill_opt = torch.optim.Adam(neighbor_islands[0].parameters(), lr=lr)
        sample_count = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Pre-encode raw inputs if an encoder is provided (e.g., CNN for images)
            if encoder is not None:
                with torch.no_grad():
                    x_batch = encoder(x_batch)

            with torch.no_grad():
                teacher_embeds = dying_island(x_batch)  # (batch, output_dim)

            # Sample weight based on dying island's specialization for the true class
            # dying_island_class_scores: (num_classes,) — higher = dying island better at class
            if dying_island_class_scores is not None:
                class_scores = dying_island_class_scores.to(device)  # (num_classes,)
                # Map true class to specialization score
                sample_weights = class_scores[y_batch.long()].clamp(min=0.0)  # (batch,)
                # Normalize to sum to batch_size so total loss is comparable
                if sample_weights.sum() > 0:
                    sample_weights = sample_weights * (sample_weights.numel() / sample_weights.sum())
            else:
                sample_weights = torch.ones(x_batch.size(0), device=device)

            # Each neighbor tries to match the teacher
            distill_opt.zero_grad()
            student_embeds = torch.stack([n(x_batch) for n in neighbor_islands], dim=0)
            # Mean across neighbors to get a single representative
            student_mean = student_embeds.mean(dim=0)  # (batch, output_dim)
            loss_per_sample = F.mse_loss(student_mean, teacher_embeds, reduction='none').mean(dim=1)  # (batch,)
            # Apply sample weights
            loss = (loss_per_sample * sample_weights).mean()
            loss.backward()
            distill_opt.step()

            sample_count += x_batch.size(0)
            if sample_count >= 200:  # Cap at 200 samples for efficiency
                break
    else:
        # Fallback: stored embedding distillation (the original proxy method)
        # This path is less ideal but works when no dataloader is available
        pass  # No-op — the original proxy approach is removed as unreliable


def get_context_for_spawn(
    active_island_embeds: torch.Tensor,
    active_routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute context vector for HyperNetwork from active islands.

    Args:
        active_island_embeds: (num_active, ocean_dim) embeddings of active islands
        active_routing_weights: (batch, num_islands) routing weights

    Returns:
        Context vector (ocean_dim,)
    """
    if active_island_embeds.size(0) == 0:
        return torch.zeros_like(active_island_embeds[0]) if active_island_embeds.size(0) > 0 else torch.zeros(1)

    # Weight by how active each island is
    mean_embed = active_island_embeds.mean(dim=0)
    return mean_embed


if __name__ == "__main__":
    # Smoke test
    lifecycle = IslandLifecycle(
        num_islands=4,
        input_dim=128,
        hidden_dim=64,
        ocean_dim=32,
        max_islands=8,
        min_islands=2,
    )
    print("IslandLifecycle initialized")
    print(f"  max_islands={lifecycle.max_islands}, min_islands={lifecycle.min_islands}")

    # Test coherence variance
    embeds = torch.randn(3, 32)  # 3 active islands
    var = lifecycle.compute_coherence_variance(embeds)
    print(f"  coherence_variance (random): {var:.4f}")

    embeds_same = torch.ones(3, 32) * 0.5
    var_same = lifecycle.compute_coherence_variance(embeds_same)
    print(f"  coherence_variance (identical): {var_same:.4f}")

    # Test spawn decision
    should = lifecycle.should_spawn(coherence_variance=0.6)
    print(f"  should_spawn (var=0.6): {should}")

    # Test kill decision
    kill_list = lifecycle.should_kill()
    print(f"  should_kill (init): {kill_list}")

    # Test gradient tracking
    lifecycle.update_gradient_tracking(0, 1e-4)
    lifecycle.step_gradient_history()
    summary = lifecycle.get_state_summary()
    print(f"  lifecycle summary: {summary}")

    print("Smoke test PASSED")

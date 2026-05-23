"""Courant module for Archipel Phase 1.

Le Courant is the homeostatic regulator of the Archipel system — a lightweight
governance mechanism that optimizes three simultaneous objectives:
  1. Coherence: active islands should agree on the representation
  2. Diversity: different islands should maintain distinct specializations
  3. Useful surprise: exploration of new configurations over local optima

The Courant does NOT dictate behavior to individual islands. Instead, it acts
as an environmental force — adapting loss term weights and router parameters
to guide the system toward the multi-objective optimum.

Phase 1 implements:
  - Dynamic loss weight adaptation based on system state
  - Entropy-based exploration modulation
  - Fitness tracking for islands (for adaptive epsilon in router)
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Courant(nn.Module):
    """Homeostatic regulator for Archipel — optimizes coherence, diversity, and surprise.

    Acts as a "light governance" mechanism: influences fitness landscape without
    imposing specific decisions on islands.
    """

    def __init__(
        self,
        num_islands: int = 4,
        lambda_coherence_init: float = 0.1,
        lambda_diversity_init: float = 0.2,
        lambda_entropy_init: float = 0.01,
        adaptation_rate: float = 0.01,
        target_entropy: float = 0.8,
        diversity_target: float = 0.25,
        coherence_target: float = 0.5,
    ) -> None:
        """Initialize the Courant regulator.

        Args:
            num_islands: Number of islands in the archipelago.
            lambda_coherence_init: Initial weight for coherence loss.
            lambda_diversity_init: Initial weight for diversity loss.
            lambda_entropy_init: Initial weight for entropy regularization.
            adaptation_rate: Rate at which weights adapt to system state.
            target_entropy: Target routing entropy (encourage exploration).
            diversity_target: Target diversity loss value (maintain specialization).
            coherence_target: Target coherence loss value.
        """
        super().__init__()
        self.num_islands = num_islands
        self.adaptation_rate = adaptation_rate
        self.target_entropy = target_entropy
        self.diversity_target = diversity_target
        self.coherence_target = coherence_target

        # Adaptive loss weights (learnable but also adapt online)
        self.lambda_coherence = lambda_coherence_init
        self.lambda_diversity = lambda_diversity_init
        self.lambda_entropy = lambda_entropy_init

        # Island fitness tracking (for adaptive epsilon-greedy in router)
        self.register_buffer("island_fitness", torch.ones(num_islands), persistent=False)
        self.fitness_decay = 0.99
        self.fitness_boost = 1.05

        # System state tracking
        self.register_buffer("routing_entropy_history", torch.zeros(100), persistent=False)
        self.register_buffer("diversity_history", torch.zeros(100), persistent=False)
        self.history_idx = 0

        # Coherence target tracking
        self.register_buffer("coherence_history", torch.zeros(100), persistent=False)

        # Counters for adaptation
        self.step_count = 0

    def update_fitness(
        self,
        island_losses: Dict[int, torch.Tensor],
        active_islands: List[int],
    ) -> None:
        """Update island fitness scores based on recent loss contributions.

        Args:
            island_losses: Dict mapping island_id -> loss contribution.
            active_islands: List of island IDs that were active.
        """
        with torch.no_grad():
            # Decay all fitness scores
            self.island_fitness *= self.fitness_decay

            # Boost active islands based on their loss (lower loss = higher fitness)
            for island_id, loss in island_losses.items():
                if loss.item() < 1.0:  # Only boost if reasonable performance
                    self.island_fitness[island_id] *= self.fitness_boost

            # Clip fitness to reasonable range
            self.island_fitness.clamp_(min=0.1, max=10.0)

    def update_routing_entropy(self, entropy: float) -> None:
        """Track routing entropy history for adaptive modulation.

        Args:
            entropy: Current routing entropy value.
        """
        idx = self.history_idx % self.routing_entropy_history.size(0)
        self.routing_entropy_history[idx] = entropy
        self.history_idx += 1

    def update_diversity(self, diversity: float) -> None:
        """Track diversity loss history.

        Args:
            diversity: Current diversity loss value.
        """
        idx = self.history_idx % self.diversity_history.size(0)
        self.diversity_history[idx] = diversity

    def update_coherence(self, coherence: float) -> None:
        """Track coherence loss history.

        Args:
            coherence: Current coherence loss value.
        """
        idx = self.history_idx % self.coherence_history.size(0)
        self.coherence_history[idx] = coherence

    def get_adaptive_weights(self) -> Dict[str, float]:
        """Compute adaptive loss weights based on system state.

        The Courant adapts loss term weights to balance three objectives:
        - When entropy is low (routing collapsed), increase entropy penalty
        - When diversity is low (islands similar), increase diversity weight
        - When coherence is high (islands agree), reduce coherence weight

        Returns:
            Dict with adapted lambda values for each loss term.
        """
        # Compute recent averages (ignoring zero padding)
        recent_len = min(self.history_idx, 100)
        if recent_len == 0:
            return {
                "lambda_coherence": self.lambda_coherence,
                "lambda_diversity": self.lambda_diversity,
                "lambda_entropy": self.lambda_entropy,
            }

        # Average over history (only valid entries)
        start_idx = max(0, self.history_idx - recent_len)
        entropy_slice = self.routing_entropy_history[start_idx:self.history_idx]
        diversity_slice = self.diversity_history[start_idx:self.history_idx]
        coherence_slice = self.coherence_history[start_idx:self.history_idx]

        mean_entropy = entropy_slice.mean().item()
        mean_diversity = diversity_slice.mean().item()
        mean_coherence = coherence_slice.mean().item()

        # Adapt weights based on targets
        # Low entropy -> increase entropy penalty
        if mean_entropy < self.target_entropy * 0.8:
            self.lambda_entropy = min(0.1, self.lambda_entropy * 1.1)
        elif mean_entropy > self.target_entropy * 1.2:
            self.lambda_entropy = max(0.001, self.lambda_entropy * 0.95)

        # Low diversity -> increase diversity weight
        if mean_diversity < self.diversity_target * 0.8:
            self.lambda_diversity = min(3.0, self.lambda_diversity * 1.1)
        elif mean_diversity > self.diversity_target * 1.5:
            self.lambda_diversity = max(0.1, self.lambda_diversity * 0.95)

        # High coherence (good = low loss) -> reduce coherence weight
        # coherence_target is a loss to minimize, so low is good
        if mean_coherence < self.coherence_target * 0.7:
            self.lambda_coherence = max(0.01, self.lambda_coherence * 0.95)
        elif mean_coherence > self.coherence_target * 2.0:
            self.lambda_coherence = min(2.0, self.lambda_coherence * 1.1)

        return {
            "lambda_coherence": self.lambda_coherence,
            "lambda_diversity": self.lambda_diversity,
            "lambda_entropy": self.lambda_entropy,
        }

    def get_island_fitness(self) -> torch.Tensor:
        """Get current island fitness scores for router epsilon-greedy.

        Returns:
            Fitness scores (num_islands,)
        """
        return self.island_fitness

    def get_epsilon_modulation(self) -> float:
        """Compute epsilon modulation factor for router exploration.

        When diversity is low, epsilon should increase (more exploration).
        When diversity is high, epsilon can decrease (exploitation).

        Returns:
            Epsilon modulation factor in [0.5, 2.0]
        """
        recent_len = min(self.history_idx, 100)
        if recent_len < 10:
            return 1.0

        start_idx = max(0, self.history_idx - recent_len)
        diversity_slice = self.diversity_history[start_idx:self.history_idx]
        mean_diversity = diversity_slice.mean().item()

        # If diversity is low, boost epsilon for exploration
        if mean_diversity < self.diversity_target * 0.5:
            return 1.5
        elif mean_diversity > self.diversity_target * 2.0:
            return 0.7
        return 1.0

    def step(
        self,
        entropy: float,
        diversity: float,
        coherence: float,
        island_losses: Optional[Dict[int, torch.Tensor]] = None,
        active_islands: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Single step of the Courant regulation.

        Call this at the end of each training step to adapt weights and fitness.

        Args:
            entropy: Current routing entropy.
            diversity: Current diversity loss.
            coherence: Current coherence loss.
            island_losses: Optional dict of per-island loss contributions.
            active_islands: Optional list of active island indices.

        Returns:
            Dict with adapted weights and modulation factors.
        """
        self.step_count += 1

        # Update history tracking
        self.update_routing_entropy(entropy)
        self.update_diversity(diversity)
        self.update_coherence(coherence)

        # Update island fitness
        if island_losses is not None and active_islands is not None:
            self.update_fitness(island_losses, active_islands)

        # Compute adaptive weights
        adapted_weights = self.get_adaptive_weights()

        # Compute epsilon modulation
        epsilon_mod = self.get_epsilon_modulation()

        return {
            "lambda_coherence": adapted_weights["lambda_coherence"],
            "lambda_diversity": adapted_weights["lambda_diversity"],
            "lambda_entropy": adapted_weights["lambda_entropy"],
            "epsilon_modulation": epsilon_mod,
            "mean_entropy": self.routing_entropy_history[:min(self.history_idx, 100)].mean().item(),
            "mean_diversity": self.diversity_history[:min(self.history_idx, 100)].mean().item(),
        }

    def get_state_report(self) -> Dict[str, float]:
        """Get a comprehensive report of the Courant's current state.

        Returns:
            Dict with all tracked metrics and adapted weights.
        """
        recent_len = min(self.history_idx, 100)
        valid_start = max(0, self.history_idx - recent_len)

        return {
            "step_count": self.step_count,
            "lambda_coherence": self.lambda_coherence,
            "lambda_diversity": self.lambda_diversity,
            "lambda_entropy": self.lambda_entropy,
            "mean_entropy": self.routing_entropy_history[valid_start:self.history_idx].mean().item() if recent_len > 0 else 0.0,
            "mean_diversity": self.diversity_history[valid_start:self.history_idx].mean().item() if recent_len > 0 else 0.0,
            "mean_coherence": self.coherence_history[valid_start:self.history_idx].mean().item() if recent_len > 0 else 0.0,
            "epsilon_modulation": self.get_epsilon_modulation(),
            "island_fitness_min": self.island_fitness.min().item(),
            "island_fitness_max": self.island_fitness.max().item(),
            "island_fitness_mean": self.island_fitness.mean().item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: identity (Courant is a regulator, not a layer).

        Args:
            x: Input tensor (passed through unchanged)

        Returns:
            Same input tensor (Courant doesn't transform data)
        """
        return x


def update_courant_from_logs(courant: Courant, logs: List[Dict[str, float]]) -> None:
    """Update Courant state from training logs (convenience function).

    Args:
        courant: Courant instance to update.
        logs: List of log dicts from training.
    """
    for log in logs:
        courant.update_routing_entropy(log.get("entropy", 1.0))
        courant.update_diversity(log.get("diversity", 0.0))
        courant.update_coherence(log.get("coherence", 0.0))


if __name__ == "__main__":
    # Test the Courant
    courant = Courant(num_islands=4)

    print("=== Initial Courant State ===")
    print(f"Lambda coherence: {courant.lambda_coherence:.4f}")
    print(f"Lambda diversity: {courant.lambda_diversity:.4f}")
    print(f"Lambda entropy: {courant.lambda_entropy:.4f}")

    # Simulate training steps
    print("\n=== Simulating Steps ===")
    for i in range(20):
        entropy = 0.5 + 0.1 * torch.randn(1).item()
        diversity = 0.15 + 0.05 * torch.randn(1).item()
        coherence = 0.1 + 0.05 * torch.randn(1).item()

        state = courant.step(entropy, diversity, coherence)

        if i % 5 == 0:
            print(f"Step {i}: entropy={state['mean_entropy']:.3f}, "
                  f"diversity={state['mean_diversity']:.3f}, "
                  f"λ_coh={state['lambda_coherence']:.3f}, "
                  f"λ_div={state['lambda_diversity']:.3f}, "
                  f"λ_ent={state['lambda_entropy']:.3f}")

    print("\n=== Final State Report ===")
    report = courant.get_state_report()
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
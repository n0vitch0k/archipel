"""Island specialization tracking per class label for Archipel Phase 2.

Tracks which islands are best at predicting which classes. Each island
maintains a specialization score per class, updated via EMA after each step
based on whether the island contributed to correct predictions.

The specialization matrix (num_islands, num_classes) is used to:
  1. Bias routing: boost correlations for islands that specialize in the
     predicted class (when available at test time)
  2. Guide distillation: dying islands preferentially distill to neighbors
     that specialize in the same classes
  3. Inform lifecycle: islands that specialize in nothing useful are
     candidates for death
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class IslandSpecialization(nn.Module):
    """Tracks per-island per-class performance scores.

    The specialization score s[i, c] is an EMA of per-step accuracy contributions:
    s[i, c] += alpha * (correct_contribution[i, c] - s[i, c])

    Where correct_contribution is 1.0 when island i contributed to a correct
    prediction for class c, 0.0 otherwise.
    """

    def __init__(
        self,
        num_islands: int,
        num_classes: int,
        ema_alpha: float = 0.1,
        specialization_boost: float = 0.3,
    ) -> None:
        """Initialize specialization tracker.

        Args:
            num_islands: Maximum number of islands (tracks all, even dead ones).
            num_classes: Number of output classes.
            ema_alpha: EMA smoothing factor for score updates (higher = faster adapt).
            specialization_boost: How much to boost routing correlation for
                                  specialized islands (added to correlation before top-k).
        """
        super().__init__()
        self.num_islands = num_islands
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha
        self.specialization_boost = specialization_boost

        # Specialization scores: (num_islands, num_classes)
        # Softmax-normalised accuracy contribution per island per class
        self.register_buffer(
            "scores",
            torch.zeros(num_islands, num_classes),
            persistent=False,
        )

        # Per-island per-class sample counts (for normalization)
        self.register_buffer(
            "counts",
            torch.zeros(num_islands, num_classes),
            persistent=False,
        )

        # Number of active islands (can be < num_islands during lifecycle)
        self._num_active_islands = num_islands

    @property
    def num_active_islands(self) -> int:
        return self._num_active_islands

    @num_active_islands.setter
    def num_active_islands(self, value: int) -> None:
        self._num_active_islands = value

    def update(
        self,
        routing_weights: torch.Tensor,
        predicted_class: torch.Tensor,
        targets: torch.Tensor,
        island_embeddings: torch.Tensor,
        island_outputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Update specialization scores from a training step.

        An island "contributes" to a prediction when it has routing weight > 0
        for that sample. We track how often island i → class c → correct,
        and update s[i, c] via EMA.

        Args:
            routing_weights: (batch, num_islands) routing weights per island.
            predicted_class: (batch,) predicted class indices.
            targets: (batch,) ground-truth class indices.
            island_embeddings: (num_islands, batch, ocean_dim) per-island embeddings.

        Returns:
            Dictionary with per-island accuracy stats for this step.
        """
        batch_size = routing_weights.shape[0]
        num_active = self._num_active_islands

        # Which islands were active per sample: (batch, num_islands)
        active = (routing_weights > 0).float()[:, :num_active]  # (batch, num_active)

        # Correct predictions per sample: (batch,)
        correct = (predicted_class == targets).float()

        # True class one-hot: (batch, num_classes)
        true_class_one_hot = torch.zeros(batch_size, self.num_classes, device=targets.device)
        true_class_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Predicted class one-hot: (batch, num_classes)
        predicted_one_hot = torch.zeros(batch_size, self.num_classes, device=predicted_class.device)
        predicted_one_hot.scatter_(1, predicted_class.unsqueeze(1), 1.0)

        # Exposure: which islands were active AND the true class was c
        # active: (batch, num_islands) → reshape to (batch, num_islands, 1)
        # true_class_one_hot: (batch, num_classes) → reshape to (batch, 1, num_classes)
        # result: (batch, num_islands, num_classes)
        exposure = active.unsqueeze(2) * true_class_one_hot.unsqueeze(1)  # (batch, num_islands, num_classes)

        # Accurate contribution: island was active AND final prediction was class c AND correct
        # predicted_one_hot: (batch, num_classes) → (batch, 1, num_classes)
        accurate_contrib = (
            active.unsqueeze(2) * predicted_one_hot.unsqueeze(1) * correct.unsqueeze(1).unsqueeze(2)
        )  # (batch, num_islands, num_classes)

        # Confidence contribution for the true class. Early in training the
        # global prediction can be wrong even when an island already has useful
        # evidence for the target class. Using true-class confidence gives the
        # specialization EMA a smoother bootstrap signal, while still penalizing
        # class collapse through the class_usage term.
        if island_outputs is not None:
            island_probs = torch.softmax(island_outputs[:, :num_active, :], dim=-1)
            true_class_prob = (island_probs * true_class_one_hot.unsqueeze(1)).sum(dim=-1)
            confidence_contrib = (
                active.unsqueeze(2)
                * true_class_prob.unsqueeze(2)
                * true_class_one_hot.unsqueeze(1)
            )
        else:
            confidence_contrib = accurate_contrib

        # Per-island per-class specialization score.
        # The score is normalized by exposure to avoid rewarding islands that
        # merely see a class often. The class-usage penalty prevents all islands
        # from collapsing onto the same class during strict top-k=1.
        with torch.no_grad():
            # Sum over batch: (num_islands, num_classes)
            exposure_sum = exposure.sum(dim=0)[:num_active]  # (num_active, num_classes)
            confidence_sum = confidence_contrib.sum(dim=0)[:num_active]  # (num_active, num_classes)
            accurate_sum = accurate_contrib.sum(dim=0)[:num_active]  # (num_active, num_classes)

            # Functional specialization score:
            # confidence_rate = true-class confidence for (island, class) / exposure
            # class_usage = total exposure for class c / total exposure across all classes
            # score = confidence_rate - 0.5 * class_usage.
            eps = 0.5
            confidence_rate = confidence_sum / (exposure_sum + eps)
            total_exposure = exposure_sum.sum().clamp(min=1.0)
            class_usage = exposure_sum.sum(dim=0, keepdim=True) / total_exposure
            smooth_score = confidence_rate - (0.5 * class_usage)

            # Keep scores in a stable range for routing boost and diagnostics.
            smooth_score = smooth_score.clamp(-1.0, 1.0)

            # EMA update: scores = alpha * smooth_score + (1-alpha) * scores
            # Use .copy_() on the buffer slice — this does NOT increment the
            # buffer's version counter (unlike [:n] = ...), avoiding autograd
            # version conflicts during backward().
            new_scores = (
                self.ema_alpha * smooth_score
                + (1 - self.ema_alpha) * self.scores[:num_active]
            )
            self.scores[:num_active].copy_(new_scores)

            # Update counts — use .data.copy_() to avoid incrementing the
            # version counter of a registered buffer that may be in the
            # autograd graph (same issue as scores update above).
            new_counts = self.counts[:num_active] + exposure_sum
            self.counts[:num_active].data.copy_(new_counts)

        # Per-island overall accuracy for logging
        per_island_acc = (accurate_contrib.sum(dim=2).sum(dim=0) / (active.sum(dim=0) + 1e-8))[:num_active]

        return {
            "per_island_accuracy": per_island_acc,
            "specialization_scores": self.scores[:num_active].clone(),
        }

    def get_specialization_boost(
        self,
        predicted_class: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-island routing boost based on specialization.

        When predicted_class is available (during training when labels are known),
        we boost the routing correlation for islands that specialize in that class.

        When only targets are available, we use the true class to select which
        class to specialize in — this is the preferred training signal.

        Args:
            predicted_class: (batch,) predicted class indices, or None.
            targets: (batch,) true class indices (used as fallback or primary).

        Returns:
            Boost tensor (batch, num_active_islands) added to correlations.
        """
        if predicted_class is None and targets is None:
            return torch.zeros(1, self._num_active_islands, device=self.scores.device)

        # Use targets as the class reference (more reliable signal than predictions)
        use_class = targets if targets is not None else predicted_class

        # Use detached scores for the boost computation to avoid autograd
        # version conflicts: scores is a registered buffer that is also
        # modified via EMA in `update()` after each backward pass. Using
        # detached values ensures no graph edge from forward to update.
        scores = self.scores[:self._num_active_islands].detach()
        class_indices = use_class.long().clamp(0, self.num_classes - 1)  # safety clamp
        island_boosts = scores[:, class_indices]  # (batch, num_islands)

        # Scale by specialization_boost — scores are in [0,1], boost in [0, 0.3]
        boost = island_boosts * self.specialization_boost  # (batch, num_islands)

        return boost.T  # (num_islands, batch) — will be added to correlations (num_islands, batch)

    def get_class_assignment(self) -> Dict[int, int]:
        """Get the primary specialization of each island.

        Returns:
            Dictionary mapping island_id → assigned class label.
        """
        assignments = {}
        for i in range(self._num_active_islands):
            best_class = self.scores[i].argmax().item()
            assignments[i] = best_class
        return assignments

    def get_specialization_summary(
        self,
        purity_threshold: float = 0.65,
        score_threshold: float = 0.02,
    ) -> Dict[str, float | str]:
        """Return functional specialization diagnostics.

        `spec_coverage` counts how many distinct classes have at least one
        specialized island. This detects the collapse case where all islands are
        specialized on the same class.

        With the functional score, an island is specialized when its best class
        score is positive and clearly above the island's average score.
        """
        active = self.scores[: self._num_active_islands].detach().float()
        if active.numel() == 0:
            return {
                "spec_coverage": 0,
                "dominant_classes": "",
                "specialized_island_count": 0,
                "specialization_purity_mean": 0.0,
                "best_class_score_mean": 0.0,
                "negative_score_count": 0.0,
                "dominant_score_max": 0.0,
            }

        # Purity should measure dominance among positive evidence, not be diluted
        # by negative anti-specialization scores. This makes coverage reflect
        # "which classes have positive specialized evidence?" instead of
        # "which class has the least negative score?"
        positive_scores = active.clamp(min=0.0)
        row_positive_sum = positive_scores.sum(dim=1).clamp(min=1e-8)
        max_scores, dominant_classes = positive_scores.max(dim=1)
        purity = max_scores / row_positive_sum
        row_mean = positive_scores.mean(dim=1)

        if positive_scores.size(1) >= 2:
            top2_purity, _ = positive_scores.topk(2, dim=1)
            dominance_margin = top2_purity[:, 0] - top2_purity[:, 1]
        else:
            dominance_margin = top2_purity = positive_scores

        # A specialized island must have a dominant class, enough positive
        # evidence, and a visible margin over the runner-up. The margin guard
        # prevents counting generic islands as specialized when they are weakly
        # positive on many classes.
        specialized = (
            (max_scores >= score_threshold)
            & (max_scores > row_mean)
            & (dominance_margin >= 0.02)
        )

        if specialized.any():
            coverage = int(dominant_classes[specialized].unique().numel())
            specialized_classes = dominant_classes[specialized].tolist()
        else:
            coverage = 0
            specialized_classes = []

        return {
            "spec_coverage": coverage,
            "dominant_classes": ",".join(str(int(c)) for c in dominant_classes.tolist()),
            "specialized_island_count": int(specialized.sum().item()),
            "specialization_purity_mean": float(purity.mean().item()),
            "specialized_classes": ",".join(str(int(c)) for c in specialized_classes),
            "best_class_score_mean": float(max_scores.mean().item()),
            "negative_score_count": float((active < 0).sum().item()),
            "dominant_score_max": float(max_scores.max().item()),
        }

    def get_useless_islands(self, min_specialization: float = 0.05) -> list:
        """Find islands with no strong specialization.

        An island is "useless" if its best specialization score is below
        the threshold — it doesn't meaningfully contribute to any class.

        Args:
            min_specialization: Minimum score to be considered specialized.

        Returns:
            List of island IDs that are not specialized.
        """
        useless = []
        for i in range(self._num_active_islands):
            if self.scores[i].max().item() < min_specialization:
                useless.append(i)
        return useless

    def get_state_summary(self) -> Dict[str, float | str]:
        """Summary of specialization state for logging."""
        active = self.scores[:self._num_active_islands]
        spec_summary = self.get_specialization_summary()
        return {
            "spec_mean": active.mean().item(),
            "spec_max": active.max().item(),
            "spec_min": active.min().item(),
            "spec_std": active.std().item(),
            "num_specialized": (active.max(dim=1)[0] > 0.2).sum().item(),
            "num_useless": len(self.get_useless_islands()),
            **spec_summary,
        }

    def resize(self, new_num_islands: int, is_spawn: bool = False) -> None:
        """Resize specialization tracker when islands are added or removed.

        New slots are initialized to zero. Removed slots are discarded.

        Args:
            new_num_islands: New total island count.
            is_spawn: True when this is a spawn (growth). All new islands become
                      active by default. False for kill/shrink — preserve activity
                      ratio scaled to the new capacity.
        """
        old_scores = self.scores.clone()
        old_counts = self.counts.clone()
        old_self_num = self.num_islands  # capture current capacity BEFORE overwriting

        new_scores = torch.zeros(new_num_islands, self.num_classes, device=old_scores.device)
        new_counts = torch.zeros(new_num_islands, self.num_classes, device=old_counts.device)

        copy_count = min(old_scores.shape[0], new_num_islands)
        new_scores[:copy_count] = old_scores[:copy_count]
        new_counts[:copy_count] = old_counts[:copy_count]

        # Use register_buffer() to properly resize the registered buffers
        self.register_buffer("scores", new_scores, persistent=False)
        self.register_buffer("counts", new_counts, persistent=False)

        self.num_islands = new_num_islands
        if is_spawn:
            # Spawn: all islands are active
            self._num_active_islands = new_num_islands
        else:
            # Kill/shrink: preserve activity ratio, clamp to [1, new_num_islands]
            if old_self_num > 0:
                ratio = self._num_active_islands / old_self_num
                new_active = int(ratio * new_num_islands + 0.5)
            else:
                new_active = 1
            self._num_active_islands = max(min(new_active, new_num_islands), 1)

if __name__ == "__main__":
    # Smoke test
    spec = IslandSpecialization(num_islands=4, num_classes=10, ema_alpha=0.1)
    print(f"IslandSpecialization: scores shape={spec.scores.shape}")

    # Simulate update
    routing_weights = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])  # batch=2, 4 islands
    predicted_class = torch.tensor([3, 5])
    targets = torch.tensor([3, 7])  # first correct, second wrong
    island_embeds = torch.randn(4, 2, 32)

    result = spec.update(routing_weights, predicted_class, targets, island_embeds)
    print(f"  update result keys: {list(result.keys())}")
    print(f"  per_island_accuracy: {result['per_island_accuracy']}")
    print(f"  spec scores[0]: {spec.scores[0]}")
    print(f"  class_assignment: {spec.get_class_assignment()}")

    # Test boost
    boost = spec.get_specialization_boost(targets=targets)
    print(f"  boost shape: {boost.shape}, mean: {boost.mean():.4f}")

    # Test resize
    spec.resize(6)
    print(f"  after resize to 6: scores shape={spec.scores.shape}")

    # Test useless
    useless = spec.get_useless_islands()
    print(f"  useless islands: {useless}")

    print("Smoke test PASSED")

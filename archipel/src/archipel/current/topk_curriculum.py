"""Dynamic top-k curriculum and routing usage diagnostics for Archipel.

The router itself stays stateless with respect to training time: it only
receives the current `top_k`. This module owns the curriculum policy and the
usage metrics used to debug specialization bootstrap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


def clamp_top_k(top_k: int, num_islands: int) -> int:
    """Clamp a requested top-k to the valid range [1, num_islands]."""
    if num_islands < 1:
        raise ValueError("num_islands must be >= 1")
    return max(1, min(int(top_k), int(num_islands)))


def compute_effective_top_k(routing_weights: torch.Tensor, threshold: float = 0.0) -> float:
    """Return the average number of islands receiving non-zero routing mass."""
    active_count = (routing_weights.detach().float() > threshold).sum(dim=1).float()
    return active_count.mean().item() if active_count.numel() else 0.0


def compute_normalized_entropy(usage: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute normalized entropy of a routing usage distribution.

    Args:
        usage: Routing usage vector of shape (num_islands,).
        eps: Small value used to avoid log(0).

    Returns:
        Normalized entropy in [0, 1], where 1 means uniform usage.
    """
    usage = usage.detach().float().flatten()
    n = usage.numel()
    if n <= 1:
        return 0.0

    probs = usage.clamp(min=eps)
    probs = probs / probs.sum().clamp(min=eps)
    entropy = -(probs * torch.log(probs)).sum()
    return float((entropy / math.log(n)).clamp(0.0, 1.0).item())


def compute_routing_usage_metrics(
    routing_weights: torch.Tensor,
    usage_ema: torch.Tensor,
    dead_threshold: float = 0.05,
    effective_threshold: float = 0.0,
) -> Dict[str, float]:
    """Compute routing usage diagnostics from current weights and EMA usage."""
    usage = usage_ema.detach().float().flatten()
    min_usage = float(usage.min().item()) if usage.numel() else 0.0
    dead_count = int((usage < dead_threshold).sum().item()) if usage.numel() else 0

    return {
        "routing_usage_entropy": compute_normalized_entropy(usage),
        "routing_usage_min": min_usage,
        "min_usage_ratio": min_usage,
        "dead_island_count": dead_count,
        "effective_top_k": compute_effective_top_k(routing_weights, threshold=effective_threshold),
    }


@dataclass
class TopKCurriculum:
    """Deterministic top-k curriculum for specialization bootstrap.

    V1 policy:
        k_init -> k_final over warmup_steps, then k_final forever.

    The default trajectory for four islands is 3 -> 2 -> 1 over 300 steps.
    """

    num_islands: int
    k_init: int = 3
    k_final: int = 1
    warmup_steps: int = 300
    freeze_step: Optional[int] = None
    gamma: float = 1.0

    def __post_init__(self) -> None:
        self.num_islands = max(1, int(self.num_islands))
        self.k_final = clamp_top_k(self.k_final, self.num_islands)
        self.k_init = clamp_top_k(self.k_init, self.num_islands)
        if self.k_init < self.k_final:
            self.k_init, self.k_final = self.k_final, self.k_init
        self.warmup_steps = max(0, int(self.warmup_steps))
        if self.freeze_step is not None:
            self.freeze_step = max(0, int(self.freeze_step))
        self.gamma = float(self.gamma)

    @property
    def scheduled_k(self) -> int:
        """Alias for `get_top_k` when callers want the current scheduled value."""
        return self.get_top_k(self._last_step)

    def resize(self, num_islands: int) -> None:
        """Resize curriculum after island birth/death."""
        self.num_islands = max(1, int(num_islands))
        self.k_final = clamp_top_k(self.k_final, self.num_islands)
        self.k_init = clamp_top_k(self.k_init, self.num_islands)
        if self.k_init < self.k_final:
            self.k_init, self.k_final = self.k_final, self.k_init

    def scheduled_top_k(self, step: int) -> int:
        """Return scheduled top-k before freeze is applied.

        The schedule is a staircase: for k_init=3, k_final=1 and warmup=300,
        steps 0-99 use k=3, steps 100-199 use k=2, steps 200+ use k=1.
        """
        if self.warmup_steps <= 0 or self.k_init == self.k_final:
            return self.k_final
        if step >= self.warmup_steps:
            return self.k_final

        levels = self.k_init - self.k_final
        level = 0
        base_interval = max(1, self.warmup_steps // (levels + 1))
        for transition in range(1, levels + 1):
            threshold = self.warmup_steps if transition == levels else transition * base_interval
            if step >= threshold:
                level = transition
        return clamp_top_k(self.k_init - level, self.num_islands)

    def get_top_k(self, step: int) -> int:
        """Return current top-k for a global training step."""
        self._last_step = int(step)
        if self.freeze_step is not None and step >= self.freeze_step:
            return self.k_final
        return self.scheduled_top_k(step)


class RoutingUsageTracker:
    """Exponential moving average of routing usage per island."""

    def __init__(
        self,
        num_islands: int,
        beta: float = 0.95,
        dead_threshold: float = 0.05,
        initial_beta: float = 0.9,
        initial_beta_steps: int = 50,
    ) -> None:
        self.num_islands = max(1, int(num_islands))
        self.beta = float(beta)
        self.dead_threshold = float(dead_threshold)
        self.initial_beta = float(initial_beta)
        self.initial_beta_steps = max(0, int(initial_beta_steps))
        self.usage_ema = torch.ones(self.num_islands, dtype=torch.float32) / self.num_islands
        self.step = 0

    def resize(self, num_islands: int) -> None:
        """Resize usage EMA while preserving existing island order."""
        num_islands = max(1, int(num_islands))
        if num_islands == self.num_islands:
            return

        old_usage = self.usage_ema.detach().float().cpu()
        new_usage = torch.ones(num_islands, dtype=torch.float32) / num_islands
        copy_n = min(old_usage.numel(), num_islands)
        new_usage[:copy_n] = old_usage[:copy_n]
        self.usage_ema = new_usage
        self.num_islands = num_islands

    def update(self, routing_weights: torch.Tensor) -> Dict[str, float]:
        """Update EMA with a batch of routing weights and return metrics."""
        batch_usage = routing_weights.detach().float().mean(dim=0).cpu()
        if batch_usage.numel() != self.num_islands:
            self.resize(batch_usage.numel())

        beta = self.initial_beta if self.step < self.initial_beta_steps else self.beta
        with torch.no_grad():
            self.usage_ema = (beta * self.usage_ema + (1.0 - beta) * batch_usage).clamp(min=0.0)
            total = self.usage_ema.sum().clamp(min=1e-8)
            self.usage_ema = self.usage_ema / total

        metrics = compute_routing_usage_metrics(
            routing_weights,
            self.usage_ema,
            dead_threshold=self.dead_threshold,
        )
        metrics["routing_usage_beta"] = beta
        self.step += 1
        return metrics

"""Training loop skeleton for Archipel Phase 1.

Minimal loop demonstrating fixed islands, HyperNetwork routing,
homeostatic regularizers, and global optimization via Courant.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..islands.base_island import BaseIsland
from ..ocean.ocean import Ocean, OceanSpace
from ..current.router import HyperNetworkRouter, HyperNetworkGenerator, init_island_states
from ..current.courant import Courant


class ArchipelPhase1(nn.Module):
    """Phase 1 Archipel model: fixed islands + ocean + router + courant."""

    def __init__(
        self,
        num_islands: int = 4,
        input_dim: int = 128,
        hidden_dim: int = 64,
        ocean_dim: int = 32,
        top_k: int = 2,
        max_islands: int = 8,
    ) -> None:
        super().__init__()
        self.num_islands = num_islands
        self.ocean_dim = ocean_dim
        self.max_islands = max_islands

        # Fixed islands (Phase 1: no dynamic birth/death yet)
        self.islands: nn.ModuleList[BaseIsland] = nn.ModuleList([
            BaseIsland(i, input_dim, hidden_dim, ocean_dim) for i in range(num_islands)
        ])

        # Ocean: shared latent space with resonance
        self.ocean = Ocean(
            embedding_dim=ocean_dim,
            num_islands=num_islands,  # actual num islands, not max
        )

        # HyperNetwork router
        self.router = HyperNetworkRouter(
            embedding_dim=ocean_dim,
            num_islands=num_islands,
            top_k=top_k,
        )

        # Shared encoder (projects input to ocean_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ocean_dim),
        )

        # Task head (shared decoder)
        self.task_head = nn.Linear(ocean_dim, 10)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with correlation-based routing and ocean integration.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Dictionary with output, embeddings, routing weights, metrics
        """
        # Encode input to ocean space
        input_repr = self.encoder(x)  # (batch, ocean_dim)

        # Get embeddings from all islands
        island_embeds = []
        for island in self.islands:
            embed = island(x)  # (batch, ocean_dim)
            island_embeds.append(embed)
        island_embeds = torch.stack(island_embeds, dim=0)  # (num_islands, batch, ocean_dim)

        # Get island states from ocean
        island_states = self.ocean.get_island_embeddings()[:self.num_islands]

        # Router: correlation-based island selection
        router_output = self.router(
            input_repr=input_repr,
            island_states=island_states,
        )
        routing_weights = router_output["routing_weights"]  # (batch, num_islands)

        # Aggregate island embeddings using routing weights
        island_embeds_T = island_embeds.transpose(0, 1)  # (batch, num_islands, ocean_dim)
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # (batch, num_islands, 1)
        ocean_embed = torch.sum(routing_weights_expanded * island_embeds_T, dim=1)  # (batch, ocean_dim)

        # Deposit island embeddings into ocean
        self.ocean.deposit_all(island_embeds)

        # Task output
        output = self.task_head(ocean_embed)

        return {
            "output": output,
            "embeddings": ocean_embed,
            "island_embeddings": island_embeds,
            "routing_weights": routing_weights,
            "correlations": router_output["correlations"],
            "entropy": router_output["entropy"],
            "sparsity": router_output["sparsity"],
        }


def compute_coherence_loss(
    active_embeddings: torch.Tensor,
    ocean_center: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Coherence loss: active islands should produce SIMILAR embeddings.

    Active islands should agree (high mutual cosine similarity).
    We measure mean off-diagonal similarity among active islands.
    Low off-diagonal = islands disagree = high loss.
    High off-diagonal = islands agree = low loss.

    Args:
        active_embeddings: Active island embeddings (batch, num_active, ocean_dim)
        ocean_center: Optional coherence center (1, ocean_dim)

    Returns:
        Coherence loss (scalar, lower = more coherent)
    """
    if active_embeddings.size(0) < 2:
        return torch.tensor(0.0, device=active_embeddings.device)

    per_island_mean = active_embeddings.mean(dim=0)  # (num_active, ocean_dim)
    normed = F.normalize(per_island_mean, p=2, dim=-1)
    similarity = torch.matmul(normed, normed.T)  # (num_active, num_active)

    # Use OFF-diagonal to measure agreement between DIFFERENT islands
    mask = 1.0 - torch.eye(normed.size(0), device=normed.device)
    off_diagonal_mean = (similarity * mask).sum() / mask.sum()

    # Low off-diagonal similarity = islands disagree = high coherence loss
    # High off-diagonal similarity = islands agree = low coherence loss
    coherence = 1.0 - off_diagonal_mean

    if ocean_center is not None:
        center_dist = (normed - F.normalize(ocean_center, p=2, dim=-1)).norm(dim=-1).mean()
        coherence = coherence + 0.1 * center_dist

    return coherence


def compute_diversity_loss(island_embeddings: torch.Tensor) -> torch.Tensor:
    """Diversity loss: penalize similarity between island latent states.

    We want DIFFERENT islands to have DIFFERENT embeddings.
    Off-diagonal cosine similarity should be LOW → penalize HIGH similarity.
    Loss = mean(max(0, similarity - margin)) for off-diagonal elements.

    Args:
        island_embeddings: All island embeddings (num_islands, ocean_dim) or (num_islands, batch, ocean_dim)

    Returns:
        Diversity loss (scalar, always >= 0, higher = more similar = bad)
    """
    if island_embeddings.dim() == 3:
        island_embeddings = island_embeddings.mean(dim=1)

    normed = F.normalize(island_embeddings, p=2, dim=-1)
    similarity = torch.matmul(normed, normed.T)  # (num_islands, num_islands)

    # Penalize off-diagonal similarity above margin
    margin = 0.3
    mask = 1.0 - torch.eye(normed.size(0), device=normed.device)
    # Guard against single-island case (mask is all zeros → sum=0 → NaN)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=normed.device)
    excess_similarity = (similarity - margin).clamp(min=0)
    diversity = (excess_similarity * mask).sum() / mask.sum()
    return diversity


def compute_structural_reg_loss(
    routing_weights: torch.Tensor,
    prev_routing_weights: Optional[torch.Tensor] = None,
    lambda_smooth: float = 0.01,
) -> torch.Tensor:
    """Structural regularization: penalize abrupt routing changes for stability."""
    if prev_routing_weights is None:
        return torch.tensor(0.0, device=routing_weights.device)
    # Detach prev_routing_weights to avoid graph coupling; guard against batch-size mismatches
    if prev_routing_weights.shape != routing_weights.shape:
        return torch.tensor(0.0, device=routing_weights.device)
    diff = routing_weights - prev_routing_weights.detach()
    return lambda_smooth * (diff ** 2).mean()


def compute_combined_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    island_embeddings: torch.Tensor,
    routing_weights: torch.Tensor,
    entropy: torch.Tensor,
    ocean_center: Optional[torch.Tensor] = None,
    prev_routing_weights: Optional[torch.Tensor] = None,
    lambda_coherence: float = 0.1,
    lambda_diversity: float = 0.1,
    lambda_entropy: float = 0.01,
    per_island_homeostatic: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    lambda_homeostatic: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined loss with all regularization terms.

    Total loss = task_loss
               + lambda_coherence * coherence_loss
               + lambda_diversity * diversity_loss
               + lambda_entropy * entropy_reg
               + lambda_homeostatic * homeostatic_loss
               + struct_reg

    Args:
        outputs: Task predictions (batch, num_classes)
        targets: Ground truth labels (batch,)
        island_embeddings: All island embeddings (num_islands, batch, ocean_dim)
        routing_weights: Routing weights from router (batch, num_islands)
        entropy: Routing entropy
        ocean_center: Optional coherence center from Ocean
        prev_routing_weights: Previous step routing weights for stability
        lambda_coherence: Weight for coherence loss
        lambda_diversity: Weight for diversity loss
        lambda_entropy: Weight for entropy regularization
        per_island_homeostatic: Optional dict mapping island_id -> {activity, diversity}
                               tensors from each island's homeostatic_regularizers()
        lambda_homeostatic: Weight for homeostatic regularization

    Returns:
        Combined loss and dict of individual loss components
    """
    # Task loss
    task_loss = F.cross_entropy(outputs, targets)

    # Identify active islands
    active_mask = routing_weights.sum(dim=0) > 0  # (num_islands,)
    num_active = active_mask.sum().item()

    # Coherence loss
    if num_active >= 2:
        active_embeds = island_embeddings[active_mask].transpose(0, 1)  # (batch, num_active, ocean_dim)
        coherence_loss = compute_coherence_loss(active_embeds, ocean_center)
    else:
        coherence_loss = torch.tensor(0.0, device=outputs.device)

    # Diversity loss
    diversity_loss = compute_diversity_loss(island_embeddings)

    # Structural regularization
    struct_reg = compute_structural_reg_loss(routing_weights, prev_routing_weights)

    # Entropy regularization
    entropy_reg = -entropy  # maximize entropy

    # Homeostatic loss: aggregate per-island regularizers
    homeostatic_loss = torch.tensor(0.0, device=outputs.device)
    homeostatic_activity = 0.0
    homeostatic_diversity = 0.0
    if per_island_homeostatic is not None:
        for island_id, regs in per_island_homeostatic.items():
            activity = regs.get("activity", torch.tensor(0.0))
            diversity = regs.get("diversity", torch.tensor(0.0))
            if isinstance(activity, torch.Tensor) and activity.device == outputs.device:
                homeostatic_loss = homeostatic_loss + activity + diversity
                homeostatic_activity += activity.item()
                homeostatic_diversity += diversity.item()
            elif isinstance(activity, torch.Tensor):
                # CPU tensor — move safely
                homeostatic_loss = homeostatic_loss + activity.to(outputs.device)
                homeostatic_diversity += diversity.item() if isinstance(diversity, torch.Tensor) else diversity

    # Total
    total_loss = (
        task_loss
        + lambda_coherence * coherence_loss
        + lambda_diversity * diversity_loss
        + lambda_entropy * entropy_reg
        + lambda_homeostatic * homeostatic_loss
        + struct_reg
    )

    return total_loss, {
        "task": task_loss.item(),
        "coherence": coherence_loss.item(),
        "diversity": diversity_loss.item(),
        "structural_reg": struct_reg.item(),
        "entropy_reg": entropy_reg.item(),
        "homeostatic": homeostatic_loss.item(),
        "homeostatic_activity": homeostatic_activity,
        "homeostatic_diversity": homeostatic_diversity,
        "total": total_loss.item(),
    }


def train_loop(
    model: ArchipelPhase1,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    courant: Courant,
    epochs: int = 1,
    device: str = "cpu",
) -> Tuple[List[Dict[str, float]], Courant]:
    """Run training loop with Courant-regulated adaptive weights.

    Args:
        model: ArchipelPhase1 model
        dataloader: Training data loader
        optimizer: Optimizer for model parameters
        courant: Courant regulator (manages adaptive weights)
        epochs: Number of epochs
        device: Device to train on

    Returns:
        Tuple of (logs, updated_courant)
    """
    model.train()
    model.to(device)
    courant.to(device)

    logs: List[Dict[str, float]] = []
    prev_routing_weights: Optional[torch.Tensor] = None

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Forward pass
            out_dict = model(x)
            outputs = out_dict["output"]
            routing_weights = out_dict["routing_weights"]
            entropy = out_dict["entropy"]
            island_embeds = out_dict["island_embeddings"]
            ocean_center = model.ocean.coherence_center

            # Courant step: get adaptive weights BEFORE loss computation
            courant_state = courant.step(
                entropy=entropy.item(),
                diversity=compute_diversity_loss(island_embeds).item(),
                coherence=0.0,  # will be computed properly below
            )

            # Collect homeostatic regularizers from each island
            per_island_homeostatic = {}
            for island in model.islands:
                regs = island.homeostatic_regularizers(x)
                per_island_homeostatic[island.island_id] = regs

            # Compute combined loss with Courant's adapted weights
            loss, loss_components = compute_combined_loss(
                outputs=outputs,
                targets=y,
                island_embeddings=island_embeds,
                routing_weights=routing_weights,
                entropy=entropy,
                ocean_center=ocean_center,
                prev_routing_weights=prev_routing_weights,
                lambda_coherence=courant_state["lambda_coherence"],
                lambda_diversity=courant_state["lambda_diversity"],
                lambda_entropy=courant_state["lambda_entropy"],
                per_island_homeostatic=per_island_homeostatic,
                lambda_homeostatic=0.01,
            )

            # Recompute coherence with proper value for tracking
            active_mask = routing_weights.sum(dim=0) > 0
            if active_mask.sum().item() >= 2:
                active_embeds = island_embeds[active_mask].transpose(0, 1)
                coherence_loss_val = compute_coherence_loss(active_embeds, ocean_center).item()
            else:
                coherence_loss_val = 0.0

            # Update Courant with actual coherence
            courant.update_coherence(coherence_loss_val)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Collect per-island gradient norms AFTER backward (grads now exist)
            per_island_grad_norms = {}
            for island in model.islands:
                grad_norm_sq = 0.0
                for p in island.parameters():
                    if p.grad is not None:
                        grad_norm_sq += p.grad.data.norm(2).item() ** 2
                per_island_grad_norms[island.island_id] = grad_norm_sq ** 0.5

            optimizer.step()

            # Get ocean statistics
            ocean_stats = model.ocean.get_statistics()

            # Log metrics
            log_entry = {
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss_components["total"],
                "task_loss": loss_components["task"],
                "coherence": coherence_loss_val,
                "diversity": loss_components["diversity"],
                "entropy_reg": loss_components["entropy_reg"],
                "homeostatic": loss_components["homeostatic"],
                "homeostatic_activity": loss_components["homeostatic_activity"],
                "homeostatic_diversity": loss_components["homeostatic_diversity"],
                "sparsity": out_dict["sparsity"].item(),
                "entropy": entropy.item(),
                "lambda_coherence": courant_state["lambda_coherence"],
                "lambda_diversity": courant_state["lambda_diversity"],
                "lambda_entropy": courant_state["lambda_entropy"],
                "epsilon_mod": courant_state["epsilon_modulation"],
                "ocean_proximity_mean": ocean_stats.get("proximity_mean", 0.0),
                "ocean_total_interactions": ocean_stats.get("total_interactions", 0.0),
                "grad_norm_mean": sum(per_island_grad_norms.values()) / max(len(per_island_grad_norms), 1),
                "grad_norm_max": max(per_island_grad_norms.values()) if per_island_grad_norms else 0.0,
            }
            logs.append(log_entry)

            # Print every 10 batches
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx:3d} | "
                    f"Loss: {loss_components['total']:.4f} | "
                    f"Task: {loss_components['task']:.4f} | "
                    f"Coherence: {coherence_loss_val:.4f} | "
                    f"Diversity: {loss_components['diversity']:.4f} | "
                    f"Sparsity: {out_dict['sparsity'].item():.3f} | "
                    f"λ_coh={courant_state['lambda_coherence']:.3f} | "
                    f"λ_div={courant_state['lambda_diversity']:.3f} | "
                    f"grad={log_entry['grad_norm_mean']:.4f}"
                )

            prev_routing_weights = routing_weights.detach()

    return logs, courant


if __name__ == "__main__":
    # Test with dummy data
    model = ArchipelPhase1(num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32, top_k=2)
    courant = Courant(num_islands=4)
    dummy_data = TensorDataset(torch.randn(64, 128), torch.randint(0, 10, (64,)))
    loader = DataLoader(dummy_data, batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("=== Training Archipel Phase 1 with Courant ===")
    logs, courant = train_loop(model, loader, optimizer, courant, epochs=2)

    print(f"\n=== Final Metrics ===")
    last_log = logs[-1]
    for k, v in last_log.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (float, int)) else f"  {k}: {v}")

    print(f"\n=== Courant State Report ===")
    report = courant.get_state_report()
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

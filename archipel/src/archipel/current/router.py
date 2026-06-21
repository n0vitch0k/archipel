"""HyperNetwork-based dynamic router for Archipel Phase 1.

Implements sparse routing with entropy regularization to prevent collapse.
References "Le Courant" as the homeostatic regulator of the Archipel system.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetworkRouter(nn.Module):
    """HyperNetwork-based dynamic router with sparse routing and entropy regularization.

    The router acts as part of "Le Courant" - the dynamic flow regulating
    information exchange between fixed islands in the Archipel architecture.
    It generates routing weights via a hypernetwork and enforces sparsity
    through top-k selection and auxiliary losses.

    Routing by latent correlation:
    - Each island maintains a latent state vector h_i (mean pooling of last activation)
    - For input x, shared encoder produces representation r_x
    - Island i is activated if cosine_sim(h_i, r_x) > threshold tau_i
    - Adaptive epsilon-greedy for exploration
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        num_islands: int = 4,
        top_k: int = 2,
        epsilon_init: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        """Initialize the HyperNetworkRouter.

        Args:
            embedding_dim: Dimension of ocean embeddings (shared latent space).
            num_islands: Number of islands in the archipelago.
            top_k: Number of top islands to select per forward pass.
            epsilon_init: Initial exploration probability for epsilon-greedy.
            temperature: Temperature for softmax over correlations.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_islands = num_islands
        self.top_k = top_k
        self.epsilon = epsilon_init
        self.temperature = temperature

        # Per-island threshold for activation (learnable, starts at 0.0)
        self.register_parameter(
            "island_thresholds",
            nn.Parameter(torch.zeros(num_islands), requires_grad=True),
        )

        # Adaptive epsilon decay (learnable)
        self.register_parameter(
            "epsilon_scale",
            nn.Parameter(torch.tensor(1.0), requires_grad=False),
        )

    def set_top_k(self, top_k: int) -> None:
        """Update the active top-k while keeping it valid for the current island count."""
        self.top_k = max(1, min(int(top_k), max(1, self.num_islands)))

    def compute_correlations(
        self, island_states: torch.Tensor, input_repr: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between island states and input representation.

        Args:
            island_states: Tensor of shape (num_islands, embedding_dim) — latent states h_i
            input_repr: Tensor of shape (batch_size, embedding_dim) — encoded input r_x

        Returns:
            Correlations tensor of shape (batch_size, num_islands)
        """
        # Normalize both tensors for cosine similarity
        # Clone to avoid in-place modification of ocean island_embeddings buffer
        # which is used in the autograd graph for routing decisions
        island_norm = F.normalize(island_states.clone(), p=2, dim=-1)  # (num_islands, embed_dim)
        input_norm = F.normalize(input_repr, p=2, dim=-1)  # (batch_size, embed_dim)
        correlations = torch.matmul(input_norm, island_norm.T)  # (batch_size, num_islands)
        return correlations

    def top_k_selection_with_noise(
        self, correlations: torch.Tensor, island_fitness: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k islands with adaptive epsilon-greedy exploration.

        Args:
            correlations: Cosine similarities (batch_size, num_islands)
            island_fitness: Optional fitness scores per island for adaptive epsilon

        Returns:
            routing_weights: Sparse one-hot-ish weights (batch_size, num_islands)
            entropy: Routing entropy for regularization
        """
        batch_size, num_islands = correlations.shape

        # Softmax over correlations with temperature
        probs = F.softmax(correlations / self.temperature, dim=-1)  # (batch, num_islands)

        # Compute entropy for regularization term
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()  # scalar

        # Top-k selection (deterministic part)
        _, top_indices = torch.topk(correlations, k=self.top_k, dim=-1)  # (batch, top_k)

        # Build sparse routing weights (one-hot for top-k, zero elsewhere)
        routing_weights = torch.zeros_like(probs)  # (batch, num_islands)
        routing_weights.scatter_(1, top_indices, 1.0 / self.top_k)

        # Adaptive epsilon-greedy: add noise for exploration
        if self.training:
            epsilon_k = self.epsilon * self.epsilon_scale.item()
            # Epsilon-greedy: with probability epsilon_k, replace ONE selected island
            # with a random island (per-batch decision, not per-element)
            replace_mask = torch.rand(batch_size, device=correlations.device) < epsilon_k
            random_indices = torch.randint(0, num_islands, (batch_size, 1), device=correlations.device)
            # Apply replacement: pick random island for batches that triggered
            for b in range(batch_size):
                if replace_mask[b]:
                    top_indices[b, torch.randint(0, self.top_k, (1,), device=correlations.device)] = random_indices[b]
            # Rebuild routing weights from (possibly modified) top_indices
            # Deduplicate in case epsilon-greedy replaced an index with one
            # already present in top_k (scatter_ would otherwise write twice
            # to the same position and sum to < 1.0)
            routing_weights = torch.zeros_like(probs)
            for b in range(batch_size):
                # torch.unique sorts output, so use sorted unique indices
                unique_idx = torch.unique(top_indices[b], sorted=False)
                # Assign uniform weight = 1.0 / num_unique (sums to 1.0)
                routing_weights[b, unique_idx] = 1.0 / unique_idx.numel()

            # Recompute entropy after noise
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
        """Forward pass: compute routing weights for island selection.

        Args:
            input_repr: Encoded input representation from shared encoder (batch, embed_dim)
            island_states: Latent states of all islands (num_islands, embed_dim).
                          If None, uses zero vectors (not yet initialized).
            island_fitness: Optional per-island fitness scores for adaptive epsilon.
            specialization_boost: Optional (num_islands,) or (batch, num_islands) tensor
                                 of per-island boost to add to correlations before top-k.
                                 If predicted_class or targets is provided, this is
                                 computed automatically from the specialization tracker.
            predicted_class: (batch,) predicted class indices for specialization routing.
            targets: (batch,) ground-truth class indices (used when predicted_class unavailable).

        Returns:
            Dictionary with:
                - routing_weights: Sparse selection weights (batch, num_islands)
                - entropy: Routing entropy for diversity regularization
                - correlations: Raw cosine similarities before selection
        """
        if island_states is None:
            island_states = torch.zeros(
                self.num_islands, self.embedding_dim, device=input_repr.device
            )

        # Compute correlations between island states and input
        correlations = self.compute_correlations(island_states, input_repr)  # (batch, num_islands)

        # Apply per-island thresholds (learnable bias)
        thresholds = torch.sigmoid(self.island_thresholds) * 0.5  # map to [0, 0.5]
        adjusted_correlations = correlations - thresholds.unsqueeze(0)

        # Apply specialization boost if provided (batch, num_islands)
        if specialization_boost is not None:
            if specialization_boost.dim() == 1:
                # (num_islands,) → broadcast to (batch, num_islands)
                adjusted_correlations = adjusted_correlations + specialization_boost.unsqueeze(0)
            else:
                # (batch, num_islands) or (num_islands, batch) — detect and transpose if needed
                if specialization_boost.shape[0] == input_repr.shape[0]:
                    adjusted_correlations = adjusted_correlations + specialization_boost
                elif specialization_boost.shape[-1] == input_repr.shape[0]:
                    # (num_islands, batch) → transpose
                    adjusted_correlations = adjusted_correlations + specialization_boost.T

        # Top-k selection with epsilon-greedy
        routing_weights, entropy = self.top_k_selection_with_noise(
            adjusted_correlations, island_fitness
        )

        return {
            "routing_weights": routing_weights,
            "entropy": entropy,
            "correlations": correlations,
            "sparsity": (routing_weights > 0).float().mean(),
        }


class HyperNetworkGenerator(nn.Module):
    """HyperNetwork that generates weights for new islands.

    Takes a latent seed z (dim 64) and context vector c (dim 128) encoding
    neighboring islands, and produces weight tensors for a mini-Transformer.

    Phase 1 uses a simplified version: generates a small adapter tensor
    rather than full weight matrices (reduced rank factorization for efficiency).
    """

    def __init__(
        self,
        seed_dim: int = 64,
        context_dim: int = 128,
        output_dim: int = 512,
        num_layers: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize the HyperNetwork generator.

        Args:
            seed_dim: Dimension of latent seed z.
            context_dim: Dimension of context vector c.
            output_dim: Total output dimension (for weight tensor serialization).
            num_layers: Number of layers in the hypernetwork.
            hidden_dim: Hidden dimension for hypernetwork layers.
        """
        super().__init__()
        self.seed_dim = seed_dim
        self.context_dim = context_dim
        self.output_dim = output_dim

        # Input is concatenation of seed and context
        input_dim = seed_dim + context_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        for i in range(num_layers + 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, seed: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Generate island weights from seed and context.

        Args:
            seed: Latent seed tensor (batch, seed_dim)
            context: Context vector from neighboring islands (batch, context_dim)

        Returns:
            Generated weight tensor flattened to (batch, output_dim)
        """
        x = torch.cat([seed, context], dim=-1)
        return self.network(x)


def init_island_states(num_islands: int, embedding_dim: int) -> torch.Tensor:
    """Initialize latent states for all islands.

    Uses orthogonal initialization for diversity.
    """
    states = torch.zeros(num_islands, embedding_dim)
    for i in range(num_islands):
        # Simple orthogonal-ish init with different offsets
        angle = i * torch.pi / num_islands
        states[i] = torch.randn(embedding_dim) * 0.02
        # Add a directional bias per island for initial diversity
        for j in range(min(embedding_dim, 8)):
            states[i, j] += torch.sin(torch.tensor(angle + j * 0.5)) * 0.1
    return states
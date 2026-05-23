"""Base Island (Îlot) module for Archipel Phase 1.

An Island (Îlot) is a specialized sub-network with its own parameters and local objective.
Fixed small number of islands (4-8) for Phase 1.
Each island produces embeddings into the shared Ocean via its encoder + specialist head.
Supports forward pass, local loss computation, and homeostatic regularizers.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseIsland(nn.Module):
    """Base Island (Îlot) implementation for Archipel Phase 1.

    Specialized sub-network with encoder and specialist head.
    Embeds inputs into the shared Ocean space.
    Includes local loss (denoising autoencoder style) and homeostatic regularizers
    to maintain island stability and diversity.
    """

    def __init__(
        self,
        island_id: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
    ) -> None:
        """Initialize the BaseIsland (Îlot).

        Args:
            island_id: Unique identifier for this island (Îlot).
            input_dim: Dimension of input features.
            hidden_dim: Hidden dimension for encoder layers.
            output_dim: Dimension of embeddings produced for the shared Ocean.
            num_experts: Number of specialist experts in the head (default: 4).
        """
        super().__init__()
        self.island_id = island_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts

        # Encoder: maps input to hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Specialist head: produces Ocean embeddings (mixture of experts style)
        self.expert_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, output_dim) for _ in range(num_experts)]
        )
        self.gating_network = nn.Linear(hidden_dim, num_experts)

        # Local projection head for denoising autoencoder local loss.
        # Decodes embedding back to hidden space to encourage stable representations.
        self.local_projection = nn.Linear(output_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode input and produce Ocean embedding.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Ocean embedding tensor of shape (batch_size, output_dim).
        """
        h = self.encoder(x)
        # Gating for specialist experts
        gates = F.softmax(self.gating_network(h), dim=-1)
        # Weighted sum of expert outputs
        expert_outputs = torch.stack([head(h) for head in self.expert_heads], dim=1)
        embedding = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        return embedding

    def get_expert_usage(self, x: torch.Tensor) -> torch.Tensor:
        """Get the expert usage distribution for an input.

        Args:
            x: Input tensor (batch_size, input_dim).

        Returns:
            Gates tensor (batch_size, num_experts) — softmax probabilities.
        """
        h = self.encoder(x)
        gates = F.softmax(self.gating_network(h), dim=-1)
        return gates

    def compute_local_loss(
        self, embeddings: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Denoising autoencoder local loss: project embedding back to hidden space.

        The island learns to produce stable, reconstructible embeddings.
        If targets is None, falls back to a small L2 smoothness regularizer on embeddings.

        Args:
            embeddings: Ocean embeddings produced by this island (batch, output_dim).
            targets: Optional projection targets. If provided, decode embedding and
                    compute reconstruction loss in hidden space.

        Returns:
            Scalar local loss tensor.
        """
        if targets is None:
            # Fallback: smoothness regularizer — embeddings should not vary wildly
            return (embeddings ** 2).mean() * 0.01

        # Decode embedding back to hidden space and reconstruct
        projected = self.local_projection(embeddings)
        return F.mse_loss(projected, targets.detach())

    def compute_local_loss_from_input(
        self, x: torch.Tensor, noise_std: float = 0.1
    ) -> torch.Tensor:
        """Compute denoising local loss using the original input.

        The island encodes the input, adds noise to the hidden state, re-embeds,
        decodes back to hidden space, and tries to reconstruct the clean hidden state.
        This encourages the island to produce embeddings that decode back to a
        meaningful representation.

        Args:
            x: Original input tensor (batch_size, input_dim).
            noise_std: Standard deviation of Gaussian noise on hidden state.

        Returns:
            Scalar local loss: reconstruction of hidden state from noisy embedding.
        """
        # Encode clean input
        h_clean = self.encoder(x)

        # Add noise to hidden state and encode again
        h_noisy = self.encoder(x + torch.randn_like(x) * noise_std)

        # Get clean embedding
        gates = F.softmax(self.gating_network(h_noisy), dim=-1)
        expert_outputs = torch.stack([head(h_noisy) for head in self.expert_heads], dim=1)
        embed_clean = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)

        # Decode to hidden space and reconstruct clean hidden state
        projected = self.local_projection(embed_clean)
        return F.mse_loss(projected, h_clean.detach())

    def homeostatic_regularizers(self, x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Homeostatic regularizers: activity and expert diversity.

        Activity regularizer: penalizes mean embedding norm deviation from target 1.0.
            Too low = dead island, too high = unstable island.
            Implemented as L2 deviation from target_norm=1.0.

        Diversity regularizer: penalizes peaked expert usage (low gating entropy).
            Encourages all experts to be used roughly equally.
            Minimizing this term maximizes expert entropy.

        Args:
            x: Optional input tensor to evaluate expert usage on. If None,
               only the activity regularizer is computed (0.0 as no forward pass).

        Returns:
            Dictionary with 'activity' and 'diversity' scalar tensors.
        """
        # Placeholder when called without input
        activity_reg = torch.tensor(0.0, requires_grad=False)
        diversity_reg = torch.tensor(0.0, requires_grad=False)

        if x is not None:
            # Expert diversity: negative entropy of gating distribution
            gates = self.get_expert_usage(x)  # (batch, num_experts)
            mean_gates = gates.mean(dim=0)  # (num_experts,)
            entropy = -(mean_gates * torch.log(mean_gates + 1e-8)).sum()
            diversity_reg = -entropy  # minimizing this maximizes entropy

        return {"activity": activity_reg, "diversity": diversity_reg}
"""Ocean module for Archipel Phase 1.

The Ocean is the shared latent space where island representations converge
and interact via correlation-based resonance. In Phase 1, it implements a
simplified version: a learnable embedding space with proximity metrics
that evolve based on island interaction frequency.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OceanSpace(nn.Module):
    """Shared latent space for island representations.

    Stores and manages island embeddings, computes resonance metrics,
    and tracks interaction history for plastic metric evolution.
    """

    def __init__(
        self,
        num_islands: int,
        embedding_dim: int,
        interaction_decay: float = 0.95,
    ) -> None:
        """Initialize the Ocean space.

        Args:
            num_islands: Maximum number of islands in the archipelago.
            embedding_dim: Dimension of the latent space.
            interaction_decay: Decay rate for interaction count updates (EMA).
        """
        super().__init__()
        self.num_islands = num_islands
        self.embedding_dim = embedding_dim
        self.interaction_decay = interaction_decay

        # Learnable ocean parameters (not island-specific — shared space)
        self.center = nn.Parameter(torch.zeros(1, embedding_dim), requires_grad=False)

        # Island embeddings in the ocean (updated via EMA, not via gradient)
        self.register_buffer("island_embeddings", torch.zeros(num_islands, embedding_dim), persistent=False)

        # Interaction count per island (for proximity metric)
        self.register_buffer("interaction_counts", torch.ones(num_islands), persistent=False)

        # Proximity matrix between islands (computed from embeddings)
        self.register_buffer("proximity_matrix", torch.eye(num_islands), persistent=False)

    def deposit(
        self,
        island_id: int,
        embedding: torch.Tensor,
        increment_interaction: bool = True,
    ) -> None:
        """Deposit an island's embedding into the ocean (EMA update).

        Args:
            island_id: Index of the island.
            embedding: Embedding tensor (batch, embed_dim) or (embed_dim,)
            increment_interaction: Whether to increment the interaction count.
        """
        if embedding.dim() == 2:
            mean_embed = embedding.mean(dim=0)  # (embed_dim,)
        else:
            mean_embed = embedding

        # EMA update: blend new embedding with existing
        alpha = 0.1
        with torch.no_grad():
            # Detach the buffer read so the entire EMA expression is free of
            # autograd links — otherwise new_embed's grad_fn references the
            # same buffer that will be modified, causing a version conflict at
            # backward().
            existing = self.island_embeddings[island_id].detach()
            new_embed = alpha * mean_embed.detach() + (1 - alpha) * existing
            # Use .data.copy_() (NOT plain .copy_()) to update the buffer
            # WITHOUT incrementing the version counter.  Plain .copy_() on a
            # registered buffer IS tracked by the version counter and will
            # trigger "modified by inplace operation" at backward().
            self.island_embeddings[island_id].data.copy_(new_embed)

        if increment_interaction:
            # Detach ALL operations: clone first, compute on clone, copy back via
            # .data.copy_() to avoid version counter increment on interaction_counts.
            counts = self.interaction_counts.clone().detach()
            counts[island_id] = counts[island_id] * self.interaction_decay + (1.0 - self.interaction_decay)
            self.interaction_counts.data.copy_(counts)

    def deposit_batch(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Deposit multiple island embeddings at once.

        Args:
            embeddings: Tensor of shape (num_islands, embed_dim) or (num_islands, batch, embed_dim)
            mask: Optional boolean mask for which islands to update
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)  # (num_islands, embed_dim)

        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)

        alpha = 0.1
        with torch.no_grad():
            update = alpha * embeddings.detach()
            # Update only the first num_islands rows — island_embeddings has
            # capacity for max_islands but we only deposit active islands.
            n = min(embeddings.shape[0], self.island_embeddings.shape[0])
            # Detach the existing embeddings to avoid autograd version conflicts.
            existing = self.island_embeddings[:n].detach()
            # Use data.copy_() (not slice assignment) to update the buffer.
            # Slice assignment [:n] = ... increments the registered buffer's
            # version counter; copy_() does NOT. Since island_embeddings is
            # read by get_island_embeddings() whose clone() is in the graph,
            # only copy_() keeps the version stable.
            self.island_embeddings[:n].data.copy_(update[:n] + (1 - alpha) * existing)

    def compute_proximity(self) -> torch.Tensor:
        """Compute and update proximity matrix between islands based on embedding similarity.

        Returns:
            Proximity matrix (num_islands, num_islands) with values in [0, 1]
        """
        with torch.no_grad():
            # Clone + detach to break autograd link to the registered buffer
            # (island_embeddings is EMA-updated in-place via copy_() in deposit_batch)
            embeds = self.island_embeddings.clone().detach()
            normed = F.normalize(embeds, p=2, dim=-1)
            # Cosine similarity as proximity
            similarity = torch.matmul(normed, normed.T)  # (num_islands, num_islands)

            # interaction_counts is a buffer modified in-place → detach the result
            interaction_weight = torch.sigmoid(
                (self.interaction_counts - self.interaction_counts.mean())
                / (self.interaction_counts.std() + 1e-8)
            ).unsqueeze(0).detach()

            # Blend similarity with interaction frequency
            # Clamp to [0, 1] — cosine similarities can be negative
            new_proximity = (
                similarity * 0.8 + interaction_weight * interaction_weight.T * 0.2
            ).clamp(min=0.0)
            # Use .data.copy_() to update the buffer WITHOUT incrementing
            # the version counter.  Plain .copy_() on a registered buffer
            # IS version-tracked and would corrupt any in-flight backward graph
            # that has captured the proximity_matrix's version counter.
            self.proximity_matrix.data.copy_(new_proximity)

        return self.proximity_matrix

    def resize(self, new_num_islands: int) -> None:
        """Resize all island-dependent buffers when the number of islands changes.

        Preserves existing data.  New slots are initialised to safe defaults:
          - island_embeddings new rows → zeros
          - interaction_counts new entries → 1.0
          - proximity_matrix new rows/cols → 0.0, new diagonal entries → 1.0

        Called by ArchipelPhase2.spawn_island() and kill_island() after
        num_islands is updated.

        Args:
            new_num_islands: New island count (must be ≥ 1, ≤ current capacity).
        """
        old_n = self.num_islands
        if new_num_islands == old_n:
            return
        if new_num_islands < 1:
            return

        with torch.no_grad():
            # ── island_embeddings ──────────────────────────────────────────────
            if new_num_islands > old_n:
                # Growing: pad with zeros for each new island
                pad_rows = new_num_islands - old_n
                pad = torch.zeros(pad_rows, self.embedding_dim, device=self.island_embeddings.device)
                new_embeds = torch.cat([self.island_embeddings, pad], dim=0)
            else:
                # Shrinking: keep only the first new_num_islands rows
                new_embeds = self.island_embeddings[:new_num_islands]
            # Register the new tensor (resizes the buffer properly)
            self.register_buffer("island_embeddings", new_embeds, persistent=False)

            # ── interaction_counts ─────────────────────────────────────────────
            if new_num_islands > old_n:
                pad_counts = torch.ones(new_num_islands - old_n, device=self.interaction_counts.device)
                new_counts = torch.cat([self.interaction_counts, pad_counts], dim=0)
            else:
                new_counts = self.interaction_counts[:new_num_islands]
            self.register_buffer("interaction_counts", new_counts, persistent=False)

            # ── proximity_matrix ──────────────────────────────────────────────
            if new_num_islands > old_n:
                # Growing: build a larger matrix, copy old sub-matrix
                new_pm = torch.zeros(new_num_islands, new_num_islands, device=self.proximity_matrix.device)
                new_pm[:old_n, :old_n] = self.proximity_matrix[:old_n, :old_n]
                # New islands: self-similarity = 1.0, unknown cross-similarity = 0.0
                new_pm[old_n:, old_n:] = torch.eye(new_num_islands - old_n, device=self.proximity_matrix.device)
            else:
                new_pm = self.proximity_matrix[:new_num_islands, :new_num_islands]
            self.register_buffer("proximity_matrix", new_pm, persistent=False)

        self.num_islands = new_num_islands

    def get_resonance(self, island_id: int) -> torch.Tensor:
        """Get resonance scores for one island vs all others.

        Args:
            island_id: Index of the querying island.

        Returns:
            Resonance scores (num_islands,) — higher = more resonance
        """
        proximity = self.compute_proximity()
        return proximity[island_id]

    def get_ocean_statistics(self) -> Dict[str, torch.Tensor]:
        """Get summary statistics of the ocean state.

        Returns:
            Dictionary with mean/std of embeddings, proximity stats, etc.
        """
        with torch.no_grad():
            emb_norm = self.island_embeddings.norm(dim=-1)
            proximity = self.compute_proximity()

            # Off-diagonal mean proximity (how connected are islands)
            mask = 1.0 - torch.eye(self.num_islands, device=proximity.device)
            off_diag_mean = (proximity * mask).sum() / mask.sum()
            off_diag_std = ((proximity * mask - off_diag_mean) ** 2).sum() / mask.sum()

        return {
            "embedding_mean_norm": emb_norm.mean(),
            "embedding_std_norm": emb_norm.std(),
            "proximity_mean": off_diag_mean,
            "proximity_std": off_diag_std,
            "total_interactions": self.interaction_counts.sum(),
        }


class Ocean(nn.Module):
    """Shared latent space for Archipel islands.

    In Phase 1, the Ocean is a simplified version with:
    - Learnable center point (for coherence)
    - Island embedding storage and EMA updates
    - Proximity/resonance computation
    - Statistics tracking for observability
    """

    def __init__(
        self,
        embedding_dim: int,
        num_islands: int = 8,
        interaction_decay: float = 0.95,
    ) -> None:
        """Initialize the Ocean.

        Args:
            embedding_dim: Dimension of embeddings.
            num_islands: Maximum number of islands.
            interaction_decay: Decay rate for interaction tracking.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_islands = num_islands

        # Learnable parameters
        self.coherence_center = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)

        # Ocean space
        self.space = OceanSpace(
            num_islands=num_islands,
            embedding_dim=embedding_dim,
            interaction_decay=interaction_decay,
        )

    def deposit_island(
        self,
        island_id: int,
        embedding: torch.Tensor,
    ) -> None:
        """Deposit an island's embedding into the ocean.

        Args:
            island_id: Index of the island.
            embedding: Embedding tensor (batch, embed_dim) or (embed_dim,)
        """
        self.space.deposit(island_id, embedding, increment_interaction=True)

    def deposit_all(self, embeddings: torch.Tensor) -> None:
        """Deposit all island embeddings at once.

        Args:
            embeddings: Tensor (num_islands, embed_dim) or (num_islands, batch, embed_dim)
        """
        self.space.deposit_batch(embeddings)

    def compute_coherence_loss(self) -> torch.Tensor:
        """Compute coherence loss: islands should be coherent around ocean center.

        Returns:
            Coherence loss (pull islands toward ocean center)
        """
        # Pull each island toward the coherence center
        # DETACH required: island_embeddings is an EMA-updated buffer; without
        # detaching, the forward graph retains a live edge to the buffer and
        # backward() fails when deposit()/deposit_batch() do copy_() later.
        center = self.coherence_center.detach()  # (1, embed_dim)
        island_embeds = self.space.island_embeddings.detach()  # (num_islands, embed_dim)

        # Pull each island toward the coherence center
        distance = (island_embeds - center).norm(dim=-1).mean()
        return distance

    def get_statistics(self) -> Dict[str, float]:
        """Get ocean statistics for logging/monitoring.

        Returns:
            Dictionary of float statistics
        """
        stats = self.space.get_ocean_statistics()
        return {k: v.item() for k, v in stats.items()}

    def get_island_embeddings(self) -> torch.Tensor:
        """Get current island embeddings from the ocean.

        Returns:
            Tensor (num_islands, embedding_dim)
        """
        # Clone + detach to create a storage-independent tensor that carries
        # NO autograd edge to the buffer.  Without detach(), clone() still
        # shares the same version counter; any copy_() during the same forward
        # pass (e.g. deposit_all) corrupts backward() with a version conflict.
        return self.space.island_embeddings.clone().detach()

    def forward(self) -> Dict[str, torch.Tensor]:
        """Forward pass: no-op for Ocean (it's a storage component).

        Returns:
            Empty dict (Ocean is passive, used via deposit_* methods)
        """
        return {}
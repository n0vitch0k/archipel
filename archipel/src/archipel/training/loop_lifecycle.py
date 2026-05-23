"""Training loop with island lifecycle management for Archipel Phase 1.

Adds dynamic island birth (spawning) and death (apoptosis) on top of the
existing Phase 1 training infrastructure.
"""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..islands.base_island import BaseIsland
from ..islands.lifecycle import IslandLifecycle, get_context_for_spawn, distill_island_to_neighbors
from ..islands.specialization import IslandSpecialization
from ..ocean.ocean import Ocean, OceanSpace
from ..current.router import HyperNetworkRouter, HyperNetworkGenerator, init_island_states
from ..current.courant import Courant
from .loop import (
    ArchipelPhase1 as ArchipelPhase1Base,
    compute_coherence_loss,
    compute_diversity_loss,
    compute_structural_reg_loss,
    compute_combined_loss,
)


class ArchipelPhase2(ArchipelPhase1Base):
    """Phase 2 Archipel model: dynamic islands + lifecycle + Courant.

    Extends Phase 1 with:
    - Dynamic island birth (spawning via HyperNetwork)
    - Dynamic island death (apoptosis with distillation)
    - Lifecycle-aware training loop
    """

    def __init__(
        self,
        num_islands: int = 4,
        input_dim: int = 128,
        hidden_dim: int = 64,
        ocean_dim: int = 32,
        top_k: int = 2,
        max_islands: int = 8,
        min_islands: int = 2,
        # Lifecycle params
        coherence_variance_threshold: float = 0.5,
        gradient_norm_threshold: float = 1e-5,
        death_window: int = 100,
        birth_cooldown: int = 50,
        death_cooldown: int = 50,
    ) -> None:
        """Initialize Phase 2 Archipel with dynamic islands.

        Args:
            num_islands: Initial number of islands.
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension for islands.
            ocean_dim: Ocean embedding dimension.
            top_k: Number of islands to activate per forward pass.
            max_islands: Maximum number of islands (birth stops here).
            min_islands: Minimum number of islands (death stops here).
            coherence_variance_threshold: Variance above this triggers birth.
            gradient_norm_threshold: Gradient norm below this triggers death check.
            death_window: Consecutive low-gradient steps before death.
            birth_cooldown: Minimum steps between births.
            death_cooldown: Minimum steps between deaths.
        """
        super().__init__(
            num_islands=num_islands,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            ocean_dim=ocean_dim,
            top_k=top_k,
            max_islands=max_islands,
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.min_islands = min_islands
        self.coherence_variance_threshold = coherence_variance_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        self.death_window = death_window
        self.birth_cooldown = birth_cooldown
        self.death_cooldown = death_cooldown

        # Lifecycle manager
        self.lifecycle = IslandLifecycle(
            num_islands=num_islands,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            ocean_dim=ocean_dim,
            max_islands=max_islands,
            min_islands=min_islands,
            coherence_variance_threshold=coherence_variance_threshold,
            gradient_norm_threshold=gradient_norm_threshold,
            death_window=death_window,
            birth_cooldown=birth_cooldown,
            death_cooldown=death_cooldown,
        )

        # HyperNetwork for spawning new islands (stored in lifecycle, accessible here)
        self.hypernet = self.lifecycle.hypernet

        # Dataloader reference for distillation (set by train_loop_lifecycle)
        self._dataloader_for_distillation: Optional[DataLoader] = None
        self._distillation_device: str = "cpu"

        # Specialization tracker — tracks which island is good at which class
        # num_classes=10 by default (task_head outputs 10 classes)
        self.specialization = IslandSpecialization(
            num_islands=max_islands,
            num_classes=10,  # matches task_head output
            ema_alpha=0.1,
            specialization_boost=0.3,
        )
        # Sync _num_active_islands with actual island count
        self.specialization._num_active_islands = num_islands

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration needed to recreate the model.

        The current `num_islands` is stored, not only the initial count, so a
        checkpoint can restore a dynamically grown/shrunk architecture before
        loading the parameter tensors.
        """
        return {
            "num_islands": self.num_islands,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "ocean_dim": self.ocean_dim,
            "top_k": self.top_k,
            "max_islands": self.max_islands,
            "min_islands": self.min_islands,
            "coherence_variance_threshold": self.coherence_variance_threshold,
            "gradient_norm_threshold": self.gradient_norm_threshold,
            "death_window": self.death_window,
            "birth_cooldown": self.birth_cooldown,
            "death_cooldown": self.death_cooldown,
        }

    def save_checkpoint(self, path: Any) -> None:
        """Save a complete Phase 2 checkpoint.

        `state_dict()` is not sufficient for ArchipelPhase2 because several
        runtime tensors are non-persistent buffers (`OceanSpace` EMA state and
        specialization matrices). This method stores them explicitly alongside
        the constructor config and normal trainable state.
        """
        checkpoint = {
            "format_version": 1,
            "model_class": self.__class__.__name__,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "ocean_space": {
                "num_islands": self.ocean.space.num_islands,
                "island_embeddings": self.ocean.space.island_embeddings.detach().cpu().clone(),
                "interaction_counts": self.ocean.space.interaction_counts.detach().cpu().clone(),
                "proximity_matrix": self.ocean.space.proximity_matrix.detach().cpu().clone(),
            },
            "specialization": {
                "num_islands": self.specialization.num_islands,
                "num_active_islands": self.specialization.num_active_islands,
                "scores": self.specialization.scores.detach().cpu().clone(),
                "counts": self.specialization.counts.detach().cpu().clone(),
            },
            "lifecycle_runtime": {
                "steps_since_birth": self.lifecycle._steps_since_birth,
                "steps_since_death": self.lifecycle._steps_since_death,
                "grad_history_idx": self.lifecycle._grad_history_idx,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: Any, map_location: Any = "cpu") -> "ArchipelPhase2":
        """Load a complete Phase 2 checkpoint saved by `save_checkpoint`."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict) or "config" not in checkpoint:
            raise ValueError("Checkpoint ArchipelPhase2 invalide: champ 'config' absent")

        config = dict(checkpoint["config"])
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])

        ocean_state = checkpoint.get("ocean_space", {})
        if ocean_state:
            model.ocean.space.resize(int(ocean_state["num_islands"]))
            device = model.ocean.space.island_embeddings.device
            with torch.no_grad():
                model.ocean.space.island_embeddings.data.copy_(ocean_state["island_embeddings"].to(device))
                model.ocean.space.interaction_counts.data.copy_(ocean_state["interaction_counts"].to(device))
                model.ocean.space.proximity_matrix.data.copy_(ocean_state["proximity_matrix"].to(device))

        spec_state = checkpoint.get("specialization", {})
        if spec_state:
            device = model.specialization.scores.device
            scores = spec_state["scores"].to(device)
            counts = spec_state["counts"].to(device)
            model.specialization.register_buffer("scores", scores.clone())
            model.specialization.register_buffer("counts", counts.clone())
            model.specialization.num_islands = int(spec_state["num_islands"])
            model.specialization.num_active_islands = int(spec_state["num_active_islands"])

        lifecycle_runtime = checkpoint.get("lifecycle_runtime", {})
        model.lifecycle._steps_since_birth = int(lifecycle_runtime.get("steps_since_birth", 0))
        model.lifecycle._steps_since_death = int(lifecycle_runtime.get("steps_since_death", 0))
        model.lifecycle._grad_history_idx = int(lifecycle_runtime.get("grad_history_idx", 0))

        return model

    def spawn_island(
        self,
        context: torch.Tensor,
        seed: Optional[torch.Tensor] = None,
    ) -> int:
        """Spawn a new island using the HyperNetwork.

        The HyperNetwork generates a weight vector which is used to initialize
        the adapter layer of the new island. This allows spawning with
        a "genetic memory" from the current archipelago state.

        Args:
            context: Context vector from active islands (ocean_dim,) or (1, ocean_dim).
            seed: Optional latent seed (seed_dim,) or (1, seed_dim). If None,
                  sample from N(0,I).

        Returns:
            island_id of the newly created island.
        """
        if seed is None:
            seed = torch.randn(1, self.lifecycle.hypernet.seed_dim, device=context.device)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        # Generate adapter weights via HyperNetwork
        generated_weights = self.hypernet(seed, context)  # (1, output_dim)

        new_id = len(self.islands)
        new_island = BaseIsland(
            island_id=new_id,
            input_dim=self.islands[0].input_dim,
            hidden_dim=self.islands[0].hidden_dim,
            output_dim=self.islands[0].output_dim,
        )

        # Apply generated weights as a learned bias to the new island.
        # We use the generated tensor to initialize the local_projection bias,
        # shifting the new island's output away from zero (avoiding dead start).
        hidden_dim_actual = self.islands[0].hidden_dim  # e.g., 64
        gen_slice = generated_weights[0, :hidden_dim_actual].clone()
        gen_slice = gen_slice.clamp(-2.0, 2.0)
        with torch.no_grad():
            new_island.local_projection.bias.data[:gen_slice.numel()] = gen_slice

        self.islands.append(new_island)
        self.num_islands += 1

        # Expand ocean space (resize all island-dependent buffers)
        self.ocean.space.resize(self.num_islands)

        # Expand router thresholds (no_grad to avoid leaf Variable in-place issue)
        old_num = self.router.num_islands
        self.router.num_islands = self.num_islands
        old_thresholds = self.router.island_thresholds
        with torch.no_grad():
            new_thresholds = torch.zeros(self.num_islands, device=old_thresholds.device)
            new_thresholds[:old_num] = old_thresholds.data.clone()
        self.router.island_thresholds = nn.Parameter(new_thresholds)

        self.lifecycle.num_islands = self.num_islands

        # Sync specialization capacity before resize
        self.specialization.num_islands = self.num_islands

        # Resize specialization tracker for new island (all islands active on spawn)
        self.specialization.resize(self.num_islands, is_spawn=True)

        return new_id

    def kill_island(
        self,
        island_id: int,
        distill: bool = True,
        dataloader: Optional[DataLoader] = None,
        encoder: Optional[nn.Module] = None,
    ) -> bool:
        """Kill (apoptose) an island.

        Args:
            island_id: ID of the island to kill.
            distill: If True, distill dying island's knowledge into neighbors
                     before removal. Requires self._dataloader_for_distillation
                     or the dataloader arg to be set.
            dataloader: Optional DataLoader for distillation. Takes precedence
                        over self._dataloader_for_distillation.
            encoder: Optional nn.Module to pre-encode raw inputs during distillation.

        Returns:
            True if killed, False if not possible (at min_islands).
        """
        if self.num_islands <= self.min_islands:
            return False
        if island_id >= len(self.islands):
            return False

        # Extract the dying island BEFORE removing from list
        island_to_kill = self.islands[island_id]

        # Collect neighbors (all islands except the one being killed)
        neighbor_islands = [isl for i, isl in enumerate(self.islands) if i != island_id]

        # ── Distillation: transfer knowledge from dying → neighbors ──
        if distill and len(neighbor_islands) > 0:
            dl = dataloader if dataloader is not None else self._dataloader_for_distillation
            if dl is not None:
                # Get dying island's specialization scores for targeted distillation
                dying_scores = self.specialization.scores[island_id].clone() if hasattr(self, 'specialization') else None
                distill_island_to_neighbors(
                    dying_island=island_to_kill,
                    neighbor_islands=neighbor_islands,
                    dataloader=dl,
                    steps=self.lifecycle.distillation_steps,
                    lr=self.lifecycle.distillation_lr,
                    device=self._distillation_device,
                    dying_island_class_scores=dying_scores,
                    encoder=encoder,
                )

        # Remove from islands list
        self.islands.pop(island_id)

        # Update island IDs for all islands after the removed one
        for i in range(island_id, len(self.islands)):
            self.islands[i].island_id = i

        self.num_islands -= 1

        # Resize ocean space buffers for the new island count
        self.ocean.space.resize(self.num_islands)

        # Update router (no_grad to avoid leaf Variable in-place issue)
        self.router.num_islands = self.num_islands
        with torch.no_grad():
            new_thresholds = torch.zeros(self.num_islands, device=self.router.island_thresholds.device)
            # Copy all except the removed island
            kept = [i for i in range(len(self.islands) + 1) if i != island_id]
            for new_i, old_i in enumerate(kept):
                if old_i < len(self.router.island_thresholds):
                    new_thresholds[new_i] = self.router.island_thresholds[old_i].detach().clone()
        self.router.island_thresholds = nn.Parameter(new_thresholds)

        # Update lifecycle
        self.lifecycle.num_islands = self.num_islands

        # Sync specialization capacity before resize
        self.specialization.num_islands = self.num_islands

        # Resize specialization tracker after removal
        self.specialization.resize(self.num_islands)

        return True

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with lifecycle tracking and specialization routing.

        Args:
            x: Input tensor (batch_size, input_dim)
            targets: Optional (batch_size,) ground-truth class indices.
                     If provided, used to compute specialization routing boost.

        Returns:
            Dictionary with output, embeddings, routing weights, metrics.
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
        # Detach to break the autograd connection to the ocean's registered
        # buffer (island_embeddings is EMA-updated after each forward via
        # deposit_all). Without detach, the buffer version conflict causes
        # "modified by inplace operation" errors at backward() after ~17 steps.
        island_states = self.ocean.get_island_embeddings()[:self.num_islands].detach()

        # ── Specialization routing boost ─────────────────────────────────────
        # Compute boost before router so the boost is based on current targets
        # (while the specialization update itself happens post-loss in train_loop)
        spec_boost = None
        if targets is not None:
            # get_specialization_boost returns (num_islands, batch)
            spec_boost = self.specialization.get_specialization_boost(
                predicted_class=None, targets=targets
            )  # (num_islands, batch)

        # Router: correlation-based island selection (with specialization boost)
        router_output = self.router(
            input_repr=input_repr,
            island_states=island_states,
            specialization_boost=spec_boost,
        )
        routing_weights = router_output["routing_weights"]  # (batch, num_islands)

        # Aggregate island embeddings using routing weights
        island_embeds_T = island_embeds.transpose(0, 1)  # (batch, num_islands, ocean_dim)
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # (batch, num_islands, 1)
        ocean_embed = torch.sum(routing_weights_expanded * island_embeds_T, dim=1)  # (batch, ocean_dim)

# Deposit island embeddings into ocean
        # DETACH before deposit_all: island_embeds has grad_fn=StackBackward0
        # (computed from island forward passes). deposit_all() calls deposit_batch()
        # which does in-place copy_() on the island_embeddings buffer. Without detach,
        # the autograd version counter is corrupted when backward() runs, causing
        # "modified by inplace operation" errors at step ~17.
        # DETACH TWICE: once for deposit_all() and a second time to sever the
        # remaining grad_fn link from the island_states clone in compute_correlations
        # (island_states.clone() preserves the computational graph to island weights,
        # which connects backward to this step's deposit_all in-place modification).
        self.ocean.deposit_all(island_embeds.detach().contiguous())

        # Track which islands were active for lifecycle
        self.lifecycle.update_active_tracking(routing_weights)

        # Task output
        output = self.task_head(ocean_embed)

        # Per-island logits for specialization analysis
        # island_embeds: (num_islands, batch, ocean_dim)
        num_islands, batch_sz, _ = island_embeds.shape
        flat = island_embeds.reshape(num_islands * batch_sz, self.ocean_dim)
        island_logits = self.task_head(flat)                      # (num_islands * batch, num_classes)
        island_outputs = island_logits.reshape(num_islands, batch_sz, -1).transpose(0, 1)  # (batch, num_islands, num_classes)

        return {
            "output": output,
            "embeddings": ocean_embed,
            "island_embeddings": island_embeds,
            "island_outputs": island_outputs,
            "routing_weights": routing_weights,
            "correlations": router_output["correlations"],
            "entropy": router_output["entropy"],
            "sparsity": router_output["sparsity"],
        }


def train_loop_lifecycle(
    model: ArchipelPhase2,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    courant: Courant,
    epochs: int = 1,
    device: str = "cpu",
    log_every: int = 10,
) -> Tuple[List[Dict[str, float]], Courant]:
    """Run training loop with island lifecycle management.

    Args:
        model: ArchipelPhase2 model with dynamic islands.
        dataloader: Training data loader.
        optimizer: Optimizer for model parameters.
        courant: Courant regulator.
        epochs: Number of epochs.
        device: Device to train on.
        log_every: Log every N batches.

    Returns:
Tuple of (logs, updated_courant).
    """
    model.train()
    model.to(device)
    courant.to(device)
    model.lifecycle.to(device)

    # Wire up dataloader for distillation (lifecycle uses it when killing islands)
    model._dataloader_for_distillation = dataloader
    model._distillation_device = device

    logs: List[Dict[str, float]] = []
    prev_routing_weights: Optional[torch.Tensor] = None

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

# Forward pass (targets=y enables specialization routing boost)
            out_dict = model(x, targets=y)
            outputs = out_dict["output"]
            routing_weights = out_dict["routing_weights"]
            entropy = out_dict["entropy"]
            # DETACH island_embeds immediately: as long as it carries the StackBackward0
            # graph from model.forward(), any view created from it (indexing, transpose,
            # mean…) stays in the autograd graph until loss.backward().  deposit_all()
            # then does in-place copy_() on the same buffer, incrementing the version
            # counter and corrupting the backward pass (step ~17).  Detaching at source
            # breaks every downstream view cleanly.
            island_embeds = out_dict["island_embeddings"].clone().detach()
            # Detach coherence_center (nn.Parameter) since deposit_all() modifies
            # island_embeddings in-place before backward(). Without detach, the
            # autograd graph connects coherence_center → loss.backward(), but the
            # in-place copy_() on island_embeddings (via deposit_all/deposit_batch)
            # increments the version counter and causes "modified by inplace
            # operation" errors at backward(). Detaching severs this link.
            ocean_center = model.ocean.coherence_center.detach()

            # Predicted class for specialization update (stored for post-backward use)
            predicted_class = outputs.argmax(dim=1)

            # ─── Compute loss components BEFORE backward ─────────────────────
            active_mask = routing_weights.sum(dim=0) > 0
            if active_mask.sum().item() >= 2:
                active_embeds = island_embeds[active_mask].transpose(0, 1)
                coherence_loss_val = compute_coherence_loss(active_embeds, ocean_center).item()
            else:
                coherence_loss_val = 0.0
            diversity_loss_val = compute_diversity_loss(island_embeds).item()

            courant_state = courant.step(
                entropy=entropy.item(),
                diversity=diversity_loss_val,
                coherence=coherence_loss_val,
            )

# Compute combined loss
            # island_embeds is already detached at source (line ~376) so every
            # downstream view (coherence_variance, context_for_spawn, deposit_all,
            # compute_combined_loss) carries NO grad_fn edge to the buffer.
            loss, loss_components = compute_combined_loss(
                outputs=outputs,
                targets=y,
                island_embeddings=island_embeds,  # already detached
                routing_weights=routing_weights,
                entropy=entropy,
                ocean_center=ocean_center,
                prev_routing_weights=prev_routing_weights,
                lambda_coherence=courant_state["lambda_coherence"],
                lambda_diversity=courant_state["lambda_diversity"],
                lambda_entropy=courant_state["lambda_entropy"],
            )

            # ─── Backward pass ───────────────────────────────────────────────
            torch.autograd.set_detect_anomaly(True, check_nan=False)
            optimizer.zero_grad()
            loss.backward()
            torch.autograd.set_detect_anomaly(False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Track per-island gradient norms for lifecycle
            with torch.no_grad():
                for i, island in enumerate(model.islands):
                    total_norm = sum(p.grad.norm().item() for p in island.parameters() if p.grad is not None)
                    model.lifecycle.update_gradient_tracking(i, total_norm)

            optimizer.step()

            # ─── Update specialization tracker (post-loss, after optimizer step) ─
            with torch.no_grad():
                # Update specialization scores (reads routing_weights, writes scores buffer)
                spec_result = model.specialization.update(
                    routing_weights=routing_weights,
                    predicted_class=predicted_class,
                    targets=y,
                    island_embeddings=island_embeds,
                )

            # ─── Lifecycle evaluation (POST-backward: no graph corruption) ───
            # 1. Compute coherence variance of active islands
            active_mask_v = routing_weights.sum(dim=0) > 0
            num_active_v = active_mask_v.sum().item()
            if num_active_v >= 2:
                active_embeds_v = island_embeds[active_mask_v].mean(dim=1)
                coherence_variance = model.lifecycle.compute_coherence_variance(active_embeds_v)
            else:
                coherence_variance = 0.0

            # 2. Check for birth
            should_birth = model.lifecycle.should_spawn(coherence_variance)
            if should_birth and model.num_islands < model.max_islands:
                if num_active_v >= 1:
                    context = get_context_for_spawn(
                        island_embeds[active_mask_v].mean(dim=1),
                        routing_weights,
                    )
                else:
                    context = torch.zeros(model.ocean_dim, device=device)
                new_id = model.spawn_island(context)
                logs.append({
                    "epoch": epoch, "batch": batch_idx,
                    "event": "birth", "new_island_id": new_id,
                    "coherence_variance": coherence_variance,
                    "num_islands": model.num_islands,
                })

            # 3. Check for death
            kill_list = model.lifecycle.should_kill()
            useless = model.specialization.get_useless_islands(min_specialization=0.05)
            for uid in useless:
                if uid not in kill_list and model.num_islands - len(kill_list) > model.min_islands:
                    kill_list.append(uid)
            if kill_list:
                for island_id in kill_list:
                    if model.kill_island(island_id):
                        logs.append({
                            "epoch": epoch, "batch": batch_idx,
                            "event": "death", "killed_island_id": island_id,
                            "num_islands": model.num_islands,
                        })

            # 4. Advance gradient history index (post-backward, for death window)
            model.lifecycle.step_gradient_history()

            # Re-compute courant state for logging (post-backward state)
            courant_state = courant.step(
                entropy=entropy.item(),
                diversity=diversity_loss_val,
                coherence=coherence_loss_val,
            )

            # ─── Build log entry ─────────────────────────────────────────────
            ocean_stats = model.ocean.get_statistics()
            lifecycle_state = model.lifecycle.get_state_summary()
            spec_state = model.specialization.get_state_summary()

            # ─── Build log entry ─────────────────────────────────────────────
            log_entry = {
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss_components["total"],
                "task_loss": loss_components["task"],
                "coherence": coherence_loss_val,
                "diversity": loss_components["diversity"],
                "entropy_reg": loss_components["entropy_reg"],
                "sparsity": out_dict["sparsity"].item(),
                "entropy": entropy.item(),
                "lambda_coherence": courant_state["lambda_coherence"],
                "lambda_diversity": courant_state["lambda_diversity"],
                "lambda_entropy": courant_state["lambda_entropy"],
                "epsilon_mod": courant_state["epsilon_modulation"],
                "coherence_variance": coherence_variance,
                "num_islands": model.num_islands,
                "ocean_proximity_mean": ocean_stats.get("proximity_mean", 0.0),
                "spec_mean": spec_state["spec_mean"],
                "spec_max": spec_state["spec_max"],
                **lifecycle_state,
            }
            logs.append(log_entry)

            # Print every log_every batches
            if batch_idx % log_every == 0:
                event_str = ""
                if "event" in log_entry:
                    event_str = f" [{log_entry['event'].upper()}]"
                print(
                    f"Epoch {epoch} | Batch {batch_idx:3d} | "
                    f"Loss: {loss_components['total']:.4f} | "
                    f"Task: {loss_components['task']:.4f} | "
                    f"Coherence: {coherence_loss_val:.4f} | "
                    f"Diversity: {loss_components['diversity']:.4f} | "
                    f"Islands: {model.num_islands} | "
                    f"λ_coh={courant_state['lambda_coherence']:.3f}"
                    f"{event_str}"
                )

            prev_routing_weights = routing_weights.detach()

    return logs, courant


if __name__ == "__main__":
    # Smoke test with dummy data
    print("=== Testing ArchipelPhase2 with lifecycle ===")
    model = ArchipelPhase2(
        num_islands=4,
        input_dim=128,
        hidden_dim=64,
        ocean_dim=32,
        top_k=2,
        max_islands=8,
        min_islands=2,
    )
    courant = Courant(num_islands=4)

    dummy_data = torch.randn(64, 128)
    targets = torch.randint(0, 10, (64,))
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(dummy_data, targets), batch_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Initial islands: {model.num_islands}")
    logs, updated_courant = train_loop_lifecycle(model, loader, optimizer, courant, epochs=2, log_every=10)
    print(f"\nFinal islands: {model.num_islands}")
    print(f"Total log entries: {len(logs)}")

    # Count lifecycle events
    birth_events = [l for l in logs if l.get("event") == "birth"]
    death_events = [l for l in logs if l.get("event") == "death"]
    print(f"  Birth events: {len(birth_events)}")
    print(f"  Death events: {len(death_events)}")

    # Final metrics
    last = logs[-1]
    print(f"\n=== Final Metrics ===")
    print(f"  loss: {last['loss']:.4f}")
    print(f"  num_islands: {last['num_islands']}")
    print(f"  diversity: {last['diversity']:.4f} (>= 0)")
    print(f"  coherence: {last['coherence']:.4f}")
    print(f"  sparsity: {last['sparsity']:.3f}")

    print("\n=== Smoke test PASSED ===")

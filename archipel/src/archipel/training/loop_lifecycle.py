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
from ..current.topk_curriculum import TopKCurriculum, RoutingUsageTracker
from ..current.kuramoto import KuramotoIslandRouter
from .loop import (
    ArchipelPhase1,
    compute_coherence_loss,
    compute_diversity_loss,
    compute_structural_reg_loss,
    compute_combined_loss,
)


def _format_messages(messages: List[str]) -> str:
    return " | ".join(msg for msg in messages if msg)


def _routing_diagnostic_messages(
    metrics: Dict[str, Any],
    num_islands: int,
    scheduled_top_k: int,
    global_step: int,
) -> List[str]:
    messages: List[str] = []
    if global_step == 0:
        messages.append(f"top-k curriculum active: k={scheduled_top_k}")

    dead_count = int(metrics.get("dead_island_count", 0))
    min_usage = float(metrics.get("min_usage_ratio", 1.0))
    entropy = float(metrics.get("routing_usage_entropy", 1.0))
    effective_top_k = float(metrics.get("effective_top_k", 0.0))

    if dead_count > 0:
        messages.append(f"dead_island_count={dead_count}: au moins une île est sous-utilisée")
    elif min_usage < 0.05:
        messages.append(f"min_usage_ratio={min_usage:.4f}: risque de monopolisation")
    elif scheduled_top_k == 1 and entropy < 0.35:
        messages.append(f"routing_usage_entropy={entropy:.3f}: usage très concentré")
    elif scheduled_top_k > 1 and entropy > 0.75:
        messages.append(f"routing_usage_entropy={entropy:.3f}: exploration encore large")

    if effective_top_k < max(1.0, scheduled_top_k - 0.75):
        messages.append(
            f"effective_top_k={effective_top_k:.2f} < scheduled_top_k={scheduled_top_k}"
        )

    if num_islands > 1 and dead_count == 0 and min_usage >= 0.05:
        messages.append("routing sain: toutes les îles reçoivent de l'usage")

    return messages


def _specialization_diagnostic_messages(spec_state: Dict[str, Any], global_step: int) -> List[str]:
    messages: List[str] = []
    coverage = int(spec_state.get("spec_coverage", 0))
    specialized = int(spec_state.get("specialized_island_count", 0))
    spec_std = float(spec_state.get("spec_std", 0.0))
    spec_max = float(spec_state.get("spec_max", 0.0))
    purity_mean = float(spec_state.get("specialization_purity_mean", 0.0))

    if specialized > 1 and coverage <= 1:
        messages.append(f"spec_coverage={coverage}: risque de collapse sur une seule classe")
    elif coverage == 0 and global_step >= 20 and specialized == 0:
        messages.append("spec_coverage=0: spécialisation fonctionnelle pas encore visible")
    elif coverage >= 2:
        messages.append(f"spec_coverage={coverage}: couverture fonctionnelle en cours")

    if spec_std < 1e-4 and spec_max > 0.5:
        messages.append("spec_std≈0: les îles spécialisées ont une force très similaire")

    if purity_mean >= 0.75:
        messages.append(f"purity_mean={purity_mean:.3f}: scores de spécialisation nets")

    return messages


def _lifecycle_diagnostic_messages(lifecycle_state: Dict[str, Any]) -> List[str]:
    messages: List[str] = []
    phase = str(lifecycle_state.get("lifecycle_phase", ""))
    if phase == "birth":
        messages.append("lifecycle=birth: nouvelle île créée")
    elif phase == "death":
        messages.append("lifecycle=death: île retirée")
    elif phase == "stable":
        messages.append("lifecycle=stable")
    return messages


class ArchipelPhase2(ArchipelPhase1):
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
        **kwargs: Any,
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
            specialization_boost=1.0,
        )

        # The shared task_head keeps the main Archipel output compact, while this
        # bias lets island_outputs diverge by island during strict specialization.
        self.island_output_bias = nn.Parameter(torch.randn(max_islands, 10) * 0.5)

        self.specialization._num_active_islands = num_islands
        # Absorb extra kwargs for backward compat (e.g. routing_mode from Phase3)
        _ = kwargs

    def post_training_step(self) -> None:
        """Hook called after each optimizer step in the training loop.

        ArchipelPhase2: no-op. ArchipelPhase3 overrides this to call
        Kuramoto phase updates.
        """
        pass

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
                "output_bias": self.island_output_bias.detach().cpu().clone(),
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

        # Resize island_output_bias to match checkpoint state_dict size.
        # After spawn/kill the bias is resized in-memory, so the checkpoint
        # may have fewer rows than max_islands.  We must match before
        # load_state_dict.
        with torch.no_grad():
            ckpt_bias_shape = checkpoint["state_dict"]["island_output_bias"].size(0)
            if ckpt_bias_shape < model.island_output_bias.size(0):
                model.island_output_bias = nn.Parameter(
                    model.island_output_bias[:ckpt_bias_shape].clone()
                )

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
            if "output_bias" in spec_state:
                bias = spec_state["output_bias"].to(model.island_output_bias.device)
                copy_rows = min(bias.size(0), model.island_output_bias.size(0))
                with torch.no_grad():
                    model.island_output_bias[:copy_rows, :bias.size(1)].copy_(bias[:copy_rows, :bias.size(1)])

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
        self.num_islands = len(self.islands)

        # Expand ocean space (resize all island-dependent buffers)
        self.ocean.space.resize(self.num_islands)

        # Resize router for new island count
        old_num = self.router.num_islands
        # KuramotoIslandRouter: full resize via router.resize()
        # HyperNetworkRouter: manual island_thresholds resize (backward compat)
        if hasattr(self.router, "resize"):
            self.router.resize(self.num_islands)
        else:
            self.router.num_islands = self.num_islands
            old_thresholds = self.router.island_thresholds
            with torch.no_grad():
                new_thresholds = torch.zeros(self.num_islands, device=old_thresholds.device)
                new_thresholds[:old_num] = old_thresholds.data.clone()
            self.router.island_thresholds = nn.Parameter(new_thresholds)

        old_bias = self.island_output_bias
        with torch.no_grad():
            new_bias = torch.zeros(self.num_islands, old_bias.size(1), device=old_bias.device, dtype=old_bias.dtype)
            new_bias[:old_num, :old_bias.size(1)] = old_bias[:old_num, :old_bias.size(1)]
        self.island_output_bias = nn.Parameter(new_bias)

        # Update lifecycle
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

        self.num_islands = len(self.islands)

        # Resize ocean space buffers for the new island count
        self.ocean.space.resize(self.num_islands)

        # Resize router for new island count
        # KuramotoIslandRouter: full resize via router.resize()
        # HyperNetworkRouter: manual island_thresholds resize (backward compat)
        if hasattr(self.router, "resize"):
            self.router.resize(self.num_islands)
        else:
            self.router.num_islands = self.num_islands
            with torch.no_grad():
                new_thresholds = torch.zeros(self.num_islands, device=self.router.island_thresholds.device)
                # Copy all except the removed island
                kept_thresholds = [i for i in range(len(self.router.island_thresholds)) if i != island_id]
                kept_thresholds = kept_thresholds[: self.num_islands]
                for new_i, old_i in enumerate(kept_thresholds):
                    if old_i < len(self.router.island_thresholds):
                        new_thresholds[new_i] = self.router.island_thresholds[old_i].detach().clone()
            self.router.island_thresholds = nn.Parameter(new_thresholds)

        old_bias = self.island_output_bias
        with torch.no_grad():
            new_bias = torch.zeros(self.num_islands, old_bias.size(1), device=old_bias.device, dtype=old_bias.dtype)
            kept_bias = [i for i in range(old_bias.size(0)) if i != island_id]
            kept_bias = kept_bias[: self.num_islands]
            for new_i, old_i in enumerate(kept_bias):
                new_bias[new_i, :old_bias.size(1)] = old_bias[old_i, :old_bias.size(1)]
        self.island_output_bias = nn.Parameter(new_bias)

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
        island_logits = island_logits.reshape(num_islands, batch_sz, -1).transpose(0, 1)  # (batch, num_islands, num_classes)
        island_logits = island_logits + self.island_output_bias[:num_islands].unsqueeze(0)
        island_outputs = island_logits

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


class ArchipelPhase3(ArchipelPhase2):
    """Phase 3 Archipel model: Kuramoto routing + Phase 2 lifecycle.

    Extends Phase 2 with optional Kuramoto oscillator routing.  When
    ``routing_mode='kuramoto'`` the cosine-similarity router
    (:class:`HyperNetworkRouter`) is replaced by a coupled-oscillator
    router (:class:`KuramotoIslandRouter`).  Ridge phases evolve after
    each training step via ``post_training_step()``.

    When ``routing_mode='cosine'`` (the default) the model behaves
    identically to :class:`ArchipelPhase2`.
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
        routing_mode: str = "cosine",
        # Lifecycle params
        coherence_variance_threshold: float = 0.5,
        gradient_norm_threshold: float = 1e-5,
        death_window: int = 100,
        birth_cooldown: int = 50,
        death_cooldown: int = 50,
        # Kuramoto params
        kuramoto_dt: float = 0.1,
        kuramoto_coupling_init: float = 1.0,
    ) -> None:
        self.routing_mode = routing_mode
        self._kuramoto_dt = kuramoto_dt
        self._kuramoto_coupling_init = kuramoto_coupling_init

        super().__init__(
            num_islands=num_islands,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            ocean_dim=ocean_dim,
            top_k=top_k,
            max_islands=max_islands,
            min_islands=min_islands,
            coherence_variance_threshold=coherence_variance_threshold,
            gradient_norm_threshold=gradient_norm_threshold,
            death_window=death_window,
            birth_cooldown=birth_cooldown,
            death_cooldown=death_cooldown,
        )

        # Replace router if Kuramoto mode
        if routing_mode == "kuramoto":
            self.router = KuramotoIslandRouter(
                embedding_dim=ocean_dim,
                num_islands=num_islands,
                top_k=top_k,
                dt=kuramoto_dt,
                coupling_init=kuramoto_coupling_init,
            )

    def post_training_step(self) -> None:
        """Advance Kuramoto phases after each optimizer step."""
        if self.routing_mode == "kuramoto":
            self.router.update_phases()

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["routing_mode"] = self.routing_mode
        config["kuramoto_dt"] = self._kuramoto_dt
        config["kuramoto_coupling_init"] = self._kuramoto_coupling_init
        return config

    @classmethod
    def load_checkpoint(cls, path: Any, map_location: Any = "cpu") -> "ArchipelPhase3":
        """Load a Phase 3 checkpoint.

        Handles both Phase 2 and Phase 3 checkpoints transparently.
        Phase 2 checkpoints are loaded into an ArchipelPhase3 instance
        with ``routing_mode='cosine'`` (identical behaviour).
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict) or "config" not in checkpoint:
            raise ValueError("Checkpoint invalide: champ 'config' absent")

        config = dict(checkpoint["config"])
        # Normalise for backward compat — older Phase 2 checkpoints
        # won't have routing_mode; default to cosine.
        if "routing_mode" not in config:
            config["routing_mode"] = "cosine"
        if "kuramoto_dt" not in config:
            config["kuramoto_dt"] = 0.1
        if "kuramoto_coupling_init" not in config:
            config["kuramoto_coupling_init"] = 1.0

        model = cls(**config)

        # Resize island_output_bias to match checkpoint state_dict size
        with torch.no_grad():
            ckpt_bias_shape = checkpoint["state_dict"]["island_output_bias"].size(0)
            if ckpt_bias_shape < model.island_output_bias.size(0):
                model.island_output_bias = nn.Parameter(
                    model.island_output_bias[:ckpt_bias_shape].clone()
                )

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        # Restore ocean space
        ocean_state = checkpoint.get("ocean_space", {})
        if ocean_state:
            model.ocean.space.resize(int(ocean_state["num_islands"]))
            device = model.ocean.space.island_embeddings.device
            with torch.no_grad():
                model.ocean.space.island_embeddings.data.copy_(
                    ocean_state["island_embeddings"].to(device)
                )
                model.ocean.space.interaction_counts.data.copy_(
                    ocean_state["interaction_counts"].to(device)
                )
                model.ocean.space.proximity_matrix.data.copy_(
                    ocean_state["proximity_matrix"].to(device)
                )

        # Restore specialization
        spec_state = checkpoint.get("specialization", {})
        if spec_state:
            device = model.specialization.scores.device
            scores = spec_state["scores"].to(device)
            counts = spec_state["counts"].to(device)
            model.specialization.register_buffer("scores", scores.clone())
            model.specialization.register_buffer("counts", counts.clone())
            model.specialization.num_islands = int(spec_state["num_islands"])
            model.specialization.num_active_islands = int(
                spec_state["num_active_islands"]
            )
            if "output_bias" in spec_state:
                bias = spec_state["output_bias"].to(model.island_output_bias.device)
                copy_rows = min(bias.size(0), model.island_output_bias.size(0))
                with torch.no_grad():
                    model.island_output_bias[:copy_rows, :bias.size(1)].copy_(
                        bias[:copy_rows, :bias.size(1)]
                    )

        # Restore lifecycle runtime
        lifecycle_runtime = checkpoint.get("lifecycle_runtime", {})
        model.lifecycle._steps_since_birth = int(
            lifecycle_runtime.get("steps_since_birth", 0)
        )
        model.lifecycle._steps_since_death = int(
            lifecycle_runtime.get("steps_since_death", 0)
        )
        model.lifecycle._grad_history_idx = int(
            lifecycle_runtime.get("grad_history_idx", 0)
        )

        return model



def train_loop_lifecycle(
    model: ArchipelPhase2,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    courant: Courant,
    epochs: int = 1,
    device: str = "cpu",
    log_every: int = 10,
    top_k_curriculum: Optional[TopKCurriculum] = None,
    routing_usage_tracker: Optional[RoutingUsageTracker] = None,
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
        top_k_curriculum: Optional dynamic top-k curriculum controller.
        routing_usage_tracker: Optional routing usage EMA tracker.

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
    stable_specialization_batches = 0
    if top_k_curriculum is None:
        top_k_curriculum = TopKCurriculum(
            num_islands=model.num_islands,
            k_init=model.top_k,
            k_final=model.top_k,
            warmup_steps=0,
        )
    else:
        top_k_curriculum.resize(model.num_islands)
        # Keep freeze_step explicit: None means the schedule itself decides
        # (e.g. k=3 -> 2 -> 1 over warmup_steps, then k_final forever).
        # Only infer a freeze when a legacy curriculum already supplied one
        # through model config or caller state.
        if top_k_curriculum.freeze_step is not None:
            top_k_curriculum.freeze_step = max(top_k_curriculum.warmup_steps, int(top_k_curriculum.freeze_step))
    if routing_usage_tracker is None:
        routing_usage_tracker = RoutingUsageTracker(num_islands=model.num_islands)

    global_step = 0
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            scheduled_top_k = top_k_curriculum.get_top_k(global_step)
            model.top_k = scheduled_top_k
            model.router.set_top_k(scheduled_top_k)
            routing_usage_tracker.resize(model.num_islands)

# Forward pass (targets=y enables specialization routing boost)
            out_dict = model(x, targets=y)
            outputs = out_dict["output"]
            routing_weights = out_dict["routing_weights"]
            entropy = out_dict["entropy"]
            routing_metrics = routing_usage_tracker.update(routing_weights)
            scheduled_top_k = top_k_curriculum.get_top_k(global_step)

            # DETACH island_embeds immediately: as long as it carries the StackBackward0

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

            # ─── Specialization pressure in strict top-k phase ───────────────
            # Top-k=1 makes exactly one island responsible per sample. Penalizing
            # the entropy of the active island logits pushes islands toward
            # confident class decisions instead of staying uniformly uncertain.
            island_outputs = out_dict["island_outputs"]  # [batch, islands, classes]
            active_mask = routing_weights > 0
            active_logits = island_outputs[active_mask]
            active_targets = y.repeat_interleave(active_mask.sum(dim=1).clamp(min=1).long())
            active_probs = active_logits.softmax(dim=-1)
            active_entropy = -(active_probs * active_probs.clamp_min(1e-8).log()).sum(dim=-1)
            specialization_loss_val = active_entropy.mean().item()
            specialization_lambda = 0.20 if scheduled_top_k == 1 else 0.0
            specialization_loss = active_entropy.mean() * specialization_lambda
            active_ce = F.cross_entropy(active_logits, active_targets) if active_logits.numel() > 0 else torch.tensor(0.0, device=device)
            specialization_loss = specialization_loss + active_ce * 0.30

            # Output diversity pressure: if all islands learn identical class
            # histograms, functional specialization cannot be measured. Penalize
            # class-wise agreement between islands so each island is pushed to
            # develop a different predictive bias.
            island_logits_raw = island_outputs.detach()
            classwise_diversity = island_logits_raw.var(dim=1, unbiased=False).mean(dim=0)
            output_diversity_loss_val = float(classwise_diversity.mean().item())
            output_diversity_lambda = 0.50 if scheduled_top_k == 1 else 0.0
            output_diversity_loss = -classwise_diversity.mean() * output_diversity_lambda

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
            loss = loss + specialization_loss + output_diversity_loss
            loss_components = {
                **loss_components,
                "specialization": specialization_loss_val,
                "specialization_lambda": specialization_lambda,
                "output_diversity": output_diversity_loss_val,
                "output_diversity_lambda": output_diversity_lambda,
            }

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
                    island_outputs=out_dict["island_outputs"].detach(),
                )

            # ─── Post-training step hook (Kuramoto phase update, etc.) ─
            model.post_training_step()

            # ─── Lifecycle evaluation (POST-backward: no graph corruption) ───
            # 1. Compute coherence variance of active islands
            active_mask_v = routing_weights.sum(dim=0) > 0
            num_active_v = active_mask_v.sum().item()
            if num_active_v >= 2:
                active_embeds_v = island_embeds[active_mask_v].mean(dim=1)
                coherence_variance = model.lifecycle.compute_coherence_variance(active_embeds_v)
            else:
                coherence_variance = 0.0

            # 2. Check for birth — disabled during exploration and early strict
            # phase to avoid churn while specialization scores bootstrap.
            should_birth = model.lifecycle.should_spawn(coherence_variance)
            allow_birth = (
                scheduled_top_k == 1
                and stable_specialization_batches >= 3
                and global_step >= top_k_curriculum.warmup_steps + len(dataloader)
            )
            if should_birth and allow_birth and model.num_islands < model.max_islands:
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
                stable_specialization_batches = 0

            # 3. Check for death
            kill_list = model.lifecycle.should_kill()
            allow_death = (
                scheduled_top_k == 1
                and global_step >= top_k_curriculum.warmup_steps + (2 * len(dataloader))
            )
            if not allow_death:
                kill_list = []
                useless = []
            else:
                # Do not kill islands for weak specialization during the exploration
            # phase. Specialization scores are still forming while k > 1, and
            # early kills collapse the island population before diversity can
            # emerge. In strict k=1 phase, preserve diversity until at least two
            # classes have functional coverage; otherwise pruning can leave only
            # generic islands and destroy the strict-specialization signal.
                if scheduled_top_k == 1:
                    spec_state_before_pruning = model.specialization.get_state_summary()
                    has_min_coverage = (
                        int(spec_state_before_pruning.get("spec_coverage", 0)) >= 2
                        and int(spec_state_before_pruning.get("specialized_island_count", 0)) >= 2
                    )
                    if has_min_coverage:
                        stable_specialization_batches = min(stable_specialization_batches + 1, 10)
                    elif stable_specialization_batches > 0:
                        stable_specialization_batches -= 1

                    if (
                        has_min_coverage
                        and stable_specialization_batches >= 3
                        and model.num_islands > model.min_islands + 1
                    ):
                        useless = model.specialization.get_useless_islands(min_specialization=0.05)
                        # Safety guard for long runs: never prune down to the minimum
                        # island count while specialization coverage is absent. The old
                        # long-run failure mode was exactly "2 islands + spec_coverage=0".
                        if model.num_islands <= model.min_islands + 1 and not has_min_coverage:
                            useless = []
                    else:
                        useless = []
                else:
                    useless = []
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
                        routing_usage_tracker.resize(model.num_islands)
                        stable_specialization_batches = 0

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

            diagnostic_messages = (
                _routing_diagnostic_messages(routing_metrics, model.num_islands, scheduled_top_k, global_step)
                + _specialization_diagnostic_messages(spec_state, global_step)
                + _lifecycle_diagnostic_messages(lifecycle_state)
            )
            qualitative_log = _format_messages(diagnostic_messages)

            # ─── Build log entry ─────────────────────────────────────────────
            log_entry = {
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss_components["total"],
                "task_loss": loss_components["task"],
                "coherence": coherence_loss_val,
                "diversity": loss_components["diversity"],
                "entropy_reg": loss_components["entropy_reg"],
                "specialization": specialization_loss_val,
                "specialization_lambda": specialization_lambda,
                "output_diversity": output_diversity_loss_val,
                "output_diversity_lambda": output_diversity_lambda,
                "sparsity": out_dict["sparsity"].item(),
                "entropy": entropy.item(),
                "current_top_k": scheduled_top_k,
                "scheduled_top_k": scheduled_top_k,
                **routing_metrics,
                "spec_coverage": spec_state["spec_coverage"],
                "specialized_island_count": spec_state["specialized_island_count"],
                "specialization_purity_mean": spec_state["specialization_purity_mean"],
                "best_class_score_mean": spec_state["best_class_score_mean"],
                "negative_score_count": spec_state["negative_score_count"],
                "dominant_score_max": spec_state["dominant_score_max"],
                "lambda_coherence": courant_state["lambda_coherence"],
                "lambda_entropy": courant_state["lambda_entropy"],
                "epsilon_mod": courant_state["epsilon_modulation"],
                "coherence_variance": coherence_variance,
                "num_islands": model.num_islands,
                "ocean_proximity_mean": ocean_stats.get("proximity_mean", 0.0),
                "spec_mean": spec_state["spec_mean"],
                "spec_max": spec_state["spec_max"],
                "spec_std": spec_state["spec_std"],
                # Kuramoto sync metrics (if router provides them)
                **(
                    model.router.get_sync_metrics()
                    if hasattr(model.router, "get_sync_metrics")
                    else {}
                ),
                "qualitative_log": qualitative_log,
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
                if qualitative_log:
                    print(f"  Diag: {qualitative_log}")

            prev_routing_weights = routing_weights.detach()
            global_step += 1

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

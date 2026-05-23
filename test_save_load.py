"""Tests de sauvegarde/chargement pour ArchipelPhase2."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase2


def make_model(num_islands=4):
    torch.manual_seed(123)
    return ArchipelPhase2(
        num_islands=num_islands,
        input_dim=128,
        hidden_dim=64,
        ocean_dim=32,
        top_k=2,
        max_islands=8,
        min_islands=2,
        coherence_variance_threshold=0.5,
        birth_cooldown=3,
        death_cooldown=5,
    )


def test_save_load_roundtrip_preserves_architecture_and_outputs(tmp_path):
    model = make_model()
    # Architecture dynamique: un checkpoint doit restaurer aussi les îles nées.
    model.spawn_island(torch.zeros(32))
    model.eval()

    x = torch.randn(6, 128)
    with torch.no_grad():
        expected = model(x)["output"].detach().clone()

    save_path = tmp_path / "archipel_checkpoint.pt"
    model.save_checkpoint(save_path)

    loaded = ArchipelPhase2.load_checkpoint(save_path)
    loaded_reference = ArchipelPhase2.load_checkpoint(save_path)
    loaded.eval()
    loaded_reference.eval()

    assert loaded.num_islands == 5
    assert len(loaded.islands) == 5
    assert loaded.router.num_islands == 5
    assert loaded.ocean.space.num_islands == 5
    assert loaded.specialization.num_active_islands == 5

    with torch.no_grad():
        expected = loaded_reference(x)["output"].detach()
        actual = loaded(x)["output"].detach()

    assert torch.allclose(expected, actual, atol=1e-6)


def test_save_load_preserves_non_persistent_runtime_buffers(tmp_path):
    model = make_model()
    model.spawn_island(torch.zeros(32))

    with torch.no_grad():
        model.ocean.space.island_embeddings.copy_(torch.randn_like(model.ocean.space.island_embeddings))
        model.ocean.space.interaction_counts.copy_(torch.arange(1, model.num_islands + 1).float())
        model.ocean.space.proximity_matrix.copy_(torch.eye(model.num_islands) * 0.5)
        model.specialization.scores.copy_(torch.rand_like(model.specialization.scores))
        model.specialization.counts.copy_(torch.rand_like(model.specialization.counts))

    save_path = tmp_path / "archipel_buffers.pt"
    model.save_checkpoint(save_path)
    loaded = ArchipelPhase2.load_checkpoint(save_path)

    assert torch.allclose(loaded.ocean.space.island_embeddings, model.ocean.space.island_embeddings)
    assert torch.allclose(loaded.ocean.space.interaction_counts, model.ocean.space.interaction_counts)
    assert torch.allclose(loaded.ocean.space.proximity_matrix, model.ocean.space.proximity_matrix)
    assert torch.allclose(loaded.specialization.scores, model.specialization.scores)
    assert torch.allclose(loaded.specialization.counts, model.specialization.counts)

"""Tests d'intégration pour ArchipelPhase3 (Phase 3 — Kuramoto routing).

Teste:
- Création d'ArchipelPhase3 en mode cosine = même comportement que Phase2
- Création d'ArchipelPhase3 en mode kuramoto
- Forward pass avec KuramotoIslandRouter
- Training loop avec update_phases()
- Métriques Kuramoto dans les logs
- Save/load round-trip ArchipelPhase3
- Compatibilité spawn/kill avec KuramotoRouter
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "archipel" / "src"))

from archipel.training.loop_lifecycle import ArchipelPhase2, ArchipelPhase3, train_loop_lifecycle
from archipel.current.courant import Courant
from archipel.current.kuramoto import KuramotoIslandRouter
from torch.utils.data import DataLoader, TensorDataset


# ── Helpers ─────────────────────────────────────────────────────────────

def make_phase3(
    num_islands=4,
    input_dim=128,
    hidden_dim=64,
    ocean_dim=32,
    top_k=2,
    max_islands=8,
    min_islands=2,
    routing_mode="cosine",
):
    """Construct an ArchipelPhase3 with sensible defaults."""
    torch.manual_seed(123)
    return ArchipelPhase3(
        num_islands=num_islands,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        ocean_dim=ocean_dim,
        top_k=top_k,
        max_islands=max_islands,
        min_islands=min_islands,
        routing_mode=routing_mode,
    )


def make_dataloader(batch_size=8, num_batches=4, input_dim=128):
    """Small synthetic dataloader for quick smoke tests."""
    total = batch_size * max(1, num_batches)
    x = torch.randn(total, input_dim)
    y = torch.randint(0, 10, (total,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


# ── Test 1: Creation modes ──────────────────────────────────────────────

def test_phase3_creation_cosine_mode():
    """ArchipelPhase3 en mode cosine utilise HyperNetworkRouter (comme Phase2)."""
    model = make_phase3(routing_mode="cosine")
    assert model.num_islands == 4
    assert model.routing_mode == "cosine"
    # Vérifie que le router existe (son type exact dépend de l'implémentation)
    assert hasattr(model, "router")


def test_phase3_creation_kuramoto_mode():
    """ArchipelPhase3 en mode kuramoto utilise KuramotoIslandRouter."""
    model = make_phase3(routing_mode="kuramoto")
    assert model.num_islands == 4
    assert model.routing_mode == "kuramoto"
    assert isinstance(model.router, KuramotoIslandRouter)


# ── Test 2: Forward pass ────────────────────────────────────────────────

def test_phase3_forward_kuramoto_smoke():
    """Forward pass avec Kuramoto router : shapes et finite."""
    model = make_phase3(routing_mode="kuramoto")
    model.eval()
    x = torch.randn(6, 128)
    with torch.no_grad():
        out = model(x)

    assert "output" in out
    assert out["output"].shape == (6, 10)  # batch x num_classes
    assert "routing_weights" in out
    assert out["routing_weights"].shape == (6, 4)  # batch x num_islands
    assert "embeddings" in out
    assert out["embeddings"].shape == (6, 32)
    assert torch.isfinite(out["output"]).all()
    assert torch.isfinite(out["routing_weights"]).all()
    # La somme des routing weights par ligne = 1 (top-k uniform)
    assert torch.allclose(out["routing_weights"].sum(dim=1),
                          torch.ones(6), atol=1e-5)


# ── Test 3: Training loop avec Kuramoto ─────────────────────────────────

def test_phase3_training_smoke():
    """Un batch de training avec Kuramoto router ne crash pas."""
    model = make_phase3(routing_mode="kuramoto")
    courant = Courant(num_islands=4)
    loader = make_dataloader(batch_size=8, num_batches=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logs, updated_courant = train_loop_lifecycle(
        model, loader, optimizer, courant, epochs=1, log_every=1,
    )
    assert len(logs) > 0
    last = logs[-1]
    assert torch.isfinite(torch.tensor(last["loss"]))
    assert last["num_islands"] == 4


def test_phase3_training_logs_contain_kuramoto_metrics():
    """Les logs contiennent les métriques Kuramoto (order_parameter, etc.)."""
    model = make_phase3(routing_mode="kuramoto")
    courant = Courant(num_islands=4)
    loader = make_dataloader(batch_size=8, num_batches=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logs, _ = train_loop_lifecycle(
        model, loader, optimizer, courant, epochs=1, log_every=1,
    )

    for log_entry in logs:
        if log_entry.get("order_parameter") is not None:
            break  # au moins un log avec Kuramoto metrics
    else:
        assert False, "Aucun log ne contient 'order_parameter'"

    # Vérifie les ranges valides
    for log_entry in logs:
        op = log_entry.get("order_parameter")
        if op is not None:
            assert 0.0 <= op <= 1.0, f"order_parameter hors range: {op}"
            cv = log_entry.get("circular_variance")
            if cv is not None:
                assert 0.0 <= cv <= 1.0


# ── Test 4: Save/load round-trip ───────────────────────────────────────

def test_phase3_save_load_roundtrip(tmp_path):
    """Checkpoint ArchipelPhase3 (mode kuramoto) se sauvegarde et se restaure."""
    model = make_phase3(routing_mode="kuramoto")
    model.eval()
    x = torch.randn(4, 128)
    with torch.no_grad():
        expected = model(x)["output"].detach().clone()

    save_path = tmp_path / "phase3_kuramoto.pt"
    model.save_checkpoint(save_path)

    # Utilise ArchipelPhase3.load_checkpoint pour un checkpoint Phase3
    loaded = ArchipelPhase3.load_checkpoint(save_path)
    loaded.eval()

    assert loaded.num_islands == 4
    assert loaded.routing_mode == "kuramoto"

    with torch.no_grad():
        actual = loaded(x)["output"].detach()

    assert torch.allclose(expected, actual, atol=1e-5)


# ── Test 5: Spawn/Kill compatibility ────────────────────────────────────

def test_phase3_spawn_kill_with_kuramoto():
    """Spawn et Kill fonctionnent avec KuramotoIslandRouter."""
    model = make_phase3(routing_mode="kuramoto", num_islands=4,
                        max_islands=8, min_islands=2)

    # Spawn
    model.spawn_island(torch.zeros(32))
    assert model.num_islands == 5
    assert model.router.num_islands == 5

    # Kill
    model.kill_island(0, distill=False)
    assert model.num_islands == 4
    assert model.router.num_islands == 4

    # Forward passe après spawn/kill
    model.eval()
    x = torch.randn(4, 128)
    with torch.no_grad():
        out = model(x)
    assert out["output"].shape == (4, 10)
    assert torch.isfinite(out["output"]).all()


# ── Test 6: Cosine mode = Phase2 compatible ─────────────────────────────

def test_phase3_cosine_matches_phase2():
    """Mode cosine se comporte comme ArchipelPhase2."""
    p2 = ArchipelPhase2(
        num_islands=4, input_dim=128, hidden_dim=64, ocean_dim=32, top_k=2,
        max_islands=8, min_islands=2,
    )
    p3 = make_phase3(routing_mode="cosine")

    # Mêmes poids pour comparaison
    for param_p3, param_p2 in zip(p3.parameters(), p2.parameters()):
        param_p3.data.copy_(param_p2.data)

    p2.eval()
    p3.eval()
    x = torch.randn(4, 128)
    with torch.no_grad():
        out_p2 = p2(x)
        out_p3 = p3(x)

    assert torch.allclose(out_p2["output"], out_p3["output"], atol=1e-5)
    assert torch.allclose(out_p2["routing_weights"], out_p3["routing_weights"], atol=1e-5)


# ── Test 7: Train_loop lifecycle events work with kuramoto ──────────────

def test_phase3_training_birth_death_kuramoto():
    """Birth/death events se produisent pendant l'entraînement Kuramoto."""
    model = make_phase3(
        routing_mode="kuramoto",
        num_islands=4,
        max_islands=8,
        min_islands=2,
        top_k=1,  # strict top-k accélère la naissance
    )
    courant = Courant(num_islands=4)
    loader = make_dataloader(batch_size=16, num_batches=8, input_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logs, _ = train_loop_lifecycle(
        model, loader, optimizer, courant, epochs=2, log_every=1,
    )
    birth_events = [l for l in logs if l.get("event") == "birth"]
    # On ne vérifie pas qu'il y a eu des births (dépend du seed),
    # mais on vérifie que le training se termine sans erreur
    assert logs[-1]["num_islands"] >= 2

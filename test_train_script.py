"""Tests for the configurable Archipel training script."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "archipel" / "train.py"
DEFAULT_CONFIG = ROOT / "archipel" / "configs" / "default.yaml"


def load_train_module():
    spec = importlib.util.spec_from_file_location("archipel_train_script", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_training_config_exists_and_has_required_sections():
    assert DEFAULT_CONFIG.exists(), f"Config manquante: {DEFAULT_CONFIG}"
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    assert set(config) >= {"model", "training", "data"}
    assert config["model"]["num_islands"] >= config["model"]["min_islands"]
    assert config["model"]["max_islands"] >= config["model"]["num_islands"]
    assert config["training"]["epochs"] > 0
    assert config["training"]["batch_size"] > 0
    assert config["data"]["num_samples"] >= config["training"]["batch_size"]


def test_train_script_exposes_configurable_building_blocks():
    train = load_train_module()
    config = train.load_config(DEFAULT_CONFIG)

    model = train.build_model(config)
    dataloader = train.build_dataloader(config, seed=123)

    assert model.num_islands == config["model"]["num_islands"]
    assert model.input_dim == config["model"]["input_dim"]
    batch_x, batch_y = next(iter(dataloader))
    assert batch_x.shape[1] == config["model"]["input_dim"]
    assert batch_y.ndim == 1


def test_train_script_cli_smoke_run_saves_checkpoint(tmp_path):
    config = {
        "model": {
            "num_islands": 3,
            "input_dim": 16,
            "hidden_dim": 12,
            "ocean_dim": 8,
            "top_k": 2,
            "max_islands": 5,
            "min_islands": 2,
            "coherence_variance_threshold": 10.0,
        },
        "training": {
            "lr": 0.001,
            "epochs": 1,
            "batch_size": 4,
            "log_every": 100,
            "save_every": 1,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "device": "cpu",
        },
        "data": {
            "num_samples": 12,
            "input_dim": 16,
            "num_classes": 10,
            "signal_strength": 0.3,
        },
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path), "--seed", "123", "--quiet"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    final_checkpoint = Path(config["training"]["checkpoint_dir"]) / "final.pt"
    assert final_checkpoint.exists(), f"Checkpoint final absent: {final_checkpoint}"
    assert "Training complete" in result.stdout

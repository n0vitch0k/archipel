"""Configurable training entrypoint for Archipel Phase 2.

Usage:
    python archipel/train.py --config archipel/configs/default.yaml --seed 42
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "archipel" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from archipel.current.courant import Courant
from archipel.training.loop_lifecycle import ArchipelPhase2, train_loop_lifecycle
from archipel.training.csv_logger import save_logs_to_csv


Config = Dict[str, Dict[str, Any]]


def load_config(path: str | Path) -> Config:
    """Load a YAML training config and validate required sections."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config invalide: {path}")

    for section in ("model", "training", "data"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Section config manquante ou invalide: {section}")

    return config  # type: ignore[return-value]


def build_model(config: Config) -> ArchipelPhase2:
    """Build ArchipelPhase2 from the `model` section."""
    model_cfg = config["model"]
    return ArchipelPhase2(
        num_islands=int(model_cfg["num_islands"]),
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        ocean_dim=int(model_cfg["ocean_dim"]),
        top_k=int(model_cfg.get("top_k", 2)),
        max_islands=int(model_cfg.get("max_islands", model_cfg["num_islands"])),
        min_islands=int(model_cfg.get("min_islands", 1)),
        coherence_variance_threshold=float(model_cfg.get("coherence_variance_threshold", 0.5)),
    )


def build_dataloader(config: Config, seed: int = 42) -> DataLoader:
    """Create a deterministic synthetic structured classification dataset."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    input_dim = int(data_cfg.get("input_dim", config["model"]["input_dim"]))
    num_samples = int(data_cfg.get("num_samples", 2000))
    num_classes = int(data_cfg.get("num_classes", 10))
    signal_strength = float(data_cfg.get("signal_strength", 0.7))
    batch_size = int(train_cfg.get("batch_size", 16))

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, input_dim, generator=generator)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    # Inject a simple class-dependent signal in feature bands so the task is not pure noise.
    band = max(1, input_dim // num_classes)
    for class_id in range(num_classes):
        mask = labels == class_id
        start = (class_id * band) % input_dim
        end = min(input_dim, start + band)
        x[mask, start:end] += signal_strength

    return DataLoader(TensorDataset(x, labels), batch_size=batch_size, shuffle=True, generator=generator)


def train_from_config(
    config: Config, 
    seed: int = 42, 
    quiet: bool = False,
    csv_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Run training from config, save checkpoints, and return a compact summary.
    
    Args:
        config: Training configuration dictionary.
        seed: Random seed for reproducibility.
        quiet: If True, suppress per-batch training logs.
        csv_path: Optional path to save training logs as CSV.
    """
    torch.manual_seed(seed)
    model = build_model(config)
    dataloader = build_dataloader(config, seed=seed)
    courant = Courant(num_islands=model.num_islands)

    train_cfg = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    epochs = int(train_cfg.get("epochs", 1))
    device = str(train_cfg.get("device", "cpu"))
    log_every = int(train_cfg.get("log_every", 10))

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            logs, _ = train_loop_lifecycle(model, dataloader, optimizer, courant, epochs=epochs, device=device, log_every=log_every)
    else:
        logs, _ = train_loop_lifecycle(model, dataloader, optimizer, courant, epochs=epochs, device=device, log_every=log_every)

    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "checkpoints")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = checkpoint_dir / "final.pt"
    model.save_checkpoint(final_checkpoint)
    
    # Save logs to CSV if path provided
    csv_file = None
    if csv_path:
        csv_file = save_logs_to_csv(logs, csv_path)

    summary = {
        "epochs": epochs,
        "num_logs": len(logs),
        "num_islands": model.num_islands,
        "final_checkpoint": str(final_checkpoint),
        "final_loss": logs[-1].get("loss") if logs else None,
        "csv_log": str(csv_file) if csv_file else None,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ArchipelPhase2 from a YAML config")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-batch training logs")
    parser.add_argument("--csv", type=str, default=None, help="Path to save training logs as CSV")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    summary = train_from_config(config, seed=args.seed, quiet=args.quiet, csv_path=args.csv)
    print("Training complete")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

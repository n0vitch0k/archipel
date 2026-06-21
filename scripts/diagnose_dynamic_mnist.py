"""Diagnostic MNIST dynamique pour le curriculum Top-K Archipel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "archipel" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archipel.current.courant import Courant
from archipel.current.topk_curriculum import TopKCurriculum
from archipel.training.loop_lifecycle import train_loop_lifecycle
from test_mnist_quick import MNISTArchipel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Répertoire MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=4000, help="Nombre maximal d'images")
    parser.add_argument("--k-init", type=int, default=3)
    parser.add_argument("--k-final", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--freeze-step", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_loader(data_dir: str, limit: int, batch_size: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    if limit and len(dataset) > limit:
        dataset = Subset(dataset, list(range(limit)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def summarize_logs(logs: List[Dict[str, object]]) -> Dict[str, object]:
    metric_logs = [
        entry for entry in logs
        if "scheduled_top_k" in entry and "routing_usage_entropy" in entry
    ]
    if not metric_logs:
        return {"entries": len(logs)}

    first = metric_logs[0]
    last = metric_logs[-1]
    k_values = [entry["scheduled_top_k"] for entry in metric_logs]
    spec_coverage_values = [entry["spec_coverage"] for entry in metric_logs]
    dead_island_counts = [entry["dead_island_count"] for entry in metric_logs]

    return {
        "entries": len(metric_logs),
        "total_log_entries": len(logs),
        "first_top_k": first["scheduled_top_k"],
        "last_top_k": last["scheduled_top_k"],
        "unique_top_k": sorted(set(k_values)),
        "first_spec_coverage": first["spec_coverage"],
        "max_spec_coverage": max(spec_coverage_values),
        "last_spec_coverage": last["spec_coverage"],
        "max_dead_island_count": max(dead_island_counts),
        "last_routing_usage_entropy": last["routing_usage_entropy"],
        "last_min_usage_ratio": last["min_usage_ratio"],
        "last_effective_top_k": last["effective_top_k"],
        "last_qualitative_log": last.get("qualitative_log", ""),
    }


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    loader = build_loader(args.data_dir, args.limit, args.batch_size)
    model = MNISTArchipel()
    model.top_k = args.k_init
    model.max_islands = max(6, args.k_init)
    model.min_islands = 2
    model.router.set_top_k(args.k_init)
    curriculum = TopKCurriculum(
        num_islands=model.num_islands,
        k_init=args.k_init,
        k_final=args.k_final,
        warmup_steps=args.warmup_steps,
        freeze_step=args.freeze_step,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logs, _ = train_loop_lifecycle(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        courant=Courant(num_islands=model.num_islands),
        epochs=args.epochs,
        device=args.device,
        log_every=25,
        top_k_curriculum=curriculum,
    )

    summary = summarize_logs(logs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

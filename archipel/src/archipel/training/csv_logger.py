"""CSV logging utilities for Archipel training runs."""
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _flatten_value(value: Any) -> str | float | int | None:
    """Convert arbitrary log values to CSV-safe scalars."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return str(value)
    return str(value)


def save_logs_to_csv(logs: Sequence[Dict[str, Any]], path: str | Path) -> Path:
    """Save training logs to a CSV file.

    The CSV writer infers the union of all keys across log entries. This keeps
    lifecycle event rows (birth/death) compatible with regular training rows
    without requiring every entry to carry every field.
    """
    if not logs:
        raise ValueError("Cannot save empty logs to CSV")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for log in logs:
        for key in log.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs:
            writer.writerow({key: _flatten_value(log.get(key)) for key in fieldnames})

    return csv_path


def load_logs_from_csv(path: str | Path) -> List[Dict[str, Any]]:
    """Load training logs from a CSV file."""
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]

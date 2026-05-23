"""CSV logger for Archipel training metrics.

Niveau 1.1: Persist training logs to CSV file for analysis and visualization.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Any


# Champs standards dans l'ordre attendu pour le CSV
CSV_FIELDS = [
    "epoch",
    "batch", 
    "loss",
    "task_loss",
    "coherence",
    "diversity",
    "entropy_reg",
    "sparsity",
    "entropy",
    "lambda_coherence",
    "lambda_diversity",
    "lambda_entropy",
    "epsilon_mod",
    "coherence_variance",
    "num_islands",
    "ocean_proximity_mean",
    "spec_mean",
    "spec_max",
    "grad_history_idx",
    "lifecycle_phase",
]


class CSVLogger:
    """Logger qui écrit les métriques d'entraînement dans un fichier CSV.
    
    Usage:
        logger = CSVLogger("training_log.csv")
        logger.write(logs)  # logs = List[Dict[str, float]]
    """
    
    def __init__(self, path: str | Path):
        """Initialise le logger CSV.
        
        Args:
            path: Chemin du fichier CSV de sortie.
        """
        self.path = Path(path)
        self._file = None
        self._writer = None
        
    def __enter__(self) -> "CSVLogger":
        """Ouvre le fichier et écrit l'en-tête."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ferme le fichier."""
        if self._file:
            self._file.close()
            
    def write(self, logs: List[Dict[str, Any]]) -> None:
        """Écrit une liste de logs dans le CSV.
        
        Args:
            logs: Liste de dictionnaires contenant les métriques.
        """
        if self._writer is None:
            raise RuntimeError("CSVLogger doit être utilisé en context manager (with)")
            
        for log in logs:
            # Filtrer aux seuls champs connus
            filtered = {k: v for k, v in log.items() if k in CSV_FIELDS}
            self._writer.writerow(filtered)
            
    def write_log(self, log: Dict[str, Any]) -> None:
        """Écrit un seul log dans le CSV (pour logging en temps réel).
        
        Args:
            log: Dictionnaire contenant les métriques.
        """
        if self._writer is None:
            raise RuntimeError("CSVLogger doit être utilisé en context manager (with)")
        filtered = {k: v for k, v in log.items() if k in CSV_FIELDS}
        self._writer.writerow(filtered)


def save_logs_to_csv(logs: List[Dict[str, Any]], path: str | Path) -> Path:
    """Fonction utilitaire pour sauvegarder des logs vers CSV.
    
    Args:
        logs: Liste de logs à sauvegarder.
        path: Chemin du fichier CSV.
        
    Returns:
        Le chemin absolu du fichier créé.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for log in logs:
            filtered = {k: v for k, v in log.items() if k in CSV_FIELDS}
            writer.writerow(filtered)
            
    return path.resolve()
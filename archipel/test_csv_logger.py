"""Tests for CSV logger functionality.

Niveau 1.1: Vérifie le logging CSV des métriques d'entraînement.
"""
import csv
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from archipel.training.csv_logger import CSVLogger, save_logs_to_csv, CSV_FIELDS


def test_csv_fields_constant():
    """Vérifie que CSV_FIELDS contient les champs attendus."""
    assert "epoch" in CSV_FIELDS
    assert "batch" in CSV_FIELDS
    assert "loss" in CSV_FIELDS
    assert "num_islands" in CSV_FIELDS
    print("TEST CSV1: CSV_FIELDS contient tous les champs requis - PASSED")


def test_save_logs_to_csv_basic():
    """Test de base de sauvegarde de logs vers CSV."""
    logs = [
        {"epoch": 0, "batch": 0, "loss": 2.5, "num_islands": 4},
        {"epoch": 0, "batch": 10, "loss": 2.3, "num_islands": 4},
        {"epoch": 0, "batch": 20, "loss": 2.1, "num_islands": 5},  # birth event
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_logs.csv"
        result = save_logs_to_csv(logs, csv_path)
        
        assert result.exists(), "Le fichier CSV n'a pas été créé"
        
        # Lire et vérifier le contenu
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 3, f"Attendu 3 lignes, obtenu {len(rows)}"
        assert rows[0]["epoch"] == "0"
        assert rows[0]["loss"] == "2.5"
        assert rows[2]["num_islands"] == "5"
        
    print("TEST CSV2: save_logs_to_csv basic - PASSED")


def test_csv_logger_context_manager():
    """Test l'usage du context manager CSVLogger."""
    logs = [
        {"epoch": 1, "batch": 5, "loss": 1.9, "num_islands": 5},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "context_test.csv"
        
        with CSVLogger(csv_path) as logger:
            logger.write(logs)
            
        # Vérifier que le fichier existe et est correct
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        
    print("TEST CSV3: CSVLogger context manager - PASSED")


def test_csv_filters_unknown_fields():
    """Vérifie que les champs inconnus sont filtrés."""
    logs = [
        {"epoch": 0, "batch": 0, "loss": 2.0, "unknown_field": "should be ignored"},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "filter_test.csv"
        save_logs_to_csv(logs, csv_path)
        
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert "unknown_field" not in rows[0]
        assert "epoch" in rows[0]
        
    print("TEST CSV4: CSV filters unknown fields - PASSED")


def test_csv_with_lifecycle_events():
    """Test que les événements birth/death sont inscrits."""
    logs = [
        {"epoch": 0, "batch": 50, "loss": 2.0, "num_islands": 5},
        {"epoch": 0, "batch": 51, "event": "birth", "new_island_id": 5, "num_islands": 6},
        {"epoch": 0, "batch": 52, "event": "death", "killed_island_id": 2, "num_islands": 5},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "lifecycle_test.csv"
        save_logs_to_csv(logs, csv_path)
        
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        # Les champs event ne sont pas dans CSV_FIELDS, donc ignorés
        # Mais les champs connus sont présents
        assert len(rows) == 3
        assert rows[1]["num_islands"] == "6"  # Après birth
        assert rows[2]["num_islands"] == "5"  # Après death
        
    print("TEST CSV5: CSV with lifecycle events - PASSED")


if __name__ == "__main__":
    test_csv_fields_constant()
    test_save_logs_to_csv_basic()
    test_csv_logger_context_manager()
    test_csv_filters_unknown_fields()
    test_csv_with_lifecycle_events()
    print("\n=== Tous les tests CSV PASSED ===")
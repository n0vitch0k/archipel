"""Garde-fou pytest: les tests doivent utiliser assert, pas return True/False."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_FILES = [
    ROOT / "test_archipel.py",
    ROOT / "test_lifecycle.py",
    ROOT / "test_specialization.py",
    ROOT / "test_save_load.py",
]


def test_pytest_tests_do_not_return_booleans():
    offenders = []
    for path in TEST_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Constant):
                    if isinstance(child.value.value, bool):
                        offenders.append(f"{path.name}:{child.lineno} {node.name} returns {child.value.value}")

    assert offenders == [], "Tests pytest avec return bool:\n" + "\n".join(offenders)

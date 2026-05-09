"""Validate local runtime, imports, and Kaggle credential placement."""

from __future__ import annotations

import importlib
import platform
from pathlib import Path

REQUIRED_IMPORTS = [
    "duckdb",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "seaborn",
    "sklearn",
    "statsmodels",
]


def main() -> int:
    print(f"Python: {platform.python_version()}")
    if not platform.python_version().startswith("3.11."):
        raise RuntimeError("Expected Python 3.11.x for this project.")

    for module in REQUIRED_IMPORTS:
        importlib.import_module(module)
        print(f"import ok: {module}")

    kaggle_token = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_token.exists():
        print(f"Kaggle token found: {kaggle_token}")
        importlib.import_module("kaggle")
        print("import ok: kaggle")
    else:
        print(f"Kaggle token missing: {kaggle_token}")
        print("Skipping kaggle import because the Kaggle package authenticates at import time.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

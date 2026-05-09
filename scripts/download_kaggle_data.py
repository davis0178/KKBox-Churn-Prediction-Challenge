"""Download project datasets from Kaggle."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from churn_retention.config import DATASETS


def kaggle_token_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def main() -> int:
    token = kaggle_token_path()
    if not token.exists():
        print(
            f"Kaggle API token not found at {token}. "
            "Download kaggle.json from Kaggle account settings before running this script.",
            file=sys.stderr,
        )
        return 2

    for dataset in DATASETS:
        dataset.raw_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset.kaggle_slug,
            "-p",
            str(dataset.raw_dir),
            "--unzip",
        ]
        print(f"Downloading {dataset.name}: {dataset.kaggle_slug}")
        subprocess.run(command, check=True)

    print("Kaggle data download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

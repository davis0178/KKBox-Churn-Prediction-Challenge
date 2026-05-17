"""Path helpers for project entrypoints and notebooks."""

from __future__ import annotations

from pathlib import Path


DEFAULT_PROJECT_ROOT_HINTS = (Path(r"F:/KKBox's Churn Prediction Challenge"),)


def find_project_root(start=None) -> Path:
    """Find the repository root containing this project's ``src`` package."""
    start_path = Path.cwd() if start is None else Path(start)
    resolved_start = start_path.resolve()

    for path in (resolved_start, *resolved_start.parents):
        if (path / "src" / "data" / "churn_labeller.py").exists():
            return path

    for path in DEFAULT_PROJECT_ROOT_HINTS:
        if (path / "src" / "data" / "churn_labeller.py").exists():
            return path

    raise FileNotFoundError(
        "Could not find project root containing src/data/churn_labeller.py."
    )


__all__ = ["find_project_root"]


"""Data discovery, loading, and profiling helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_retention.config import DatasetConfig


def normalize_column_name(column: str) -> str:
    """Return a snake_case-ish version of a tabular column name."""

    return (
        column.strip()
        .replace("%", "pct")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .lower()
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a DataFrame with normalized column names."""

    normalized = df.copy()
    normalized.columns = [normalize_column_name(str(column)) for column in normalized.columns]
    return normalized


def find_csv_files(directory: Path) -> list[Path]:
    """Find CSV files under a directory, sorted by size descending."""

    if not directory.exists():
        return []
    return sorted(directory.rglob("*.csv"), key=lambda path: path.stat().st_size, reverse=True)


def load_primary_csv(config: DatasetConfig) -> pd.DataFrame:
    """Load the largest CSV for a configured dataset."""

    csv_files = find_csv_files(config.raw_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found for {config.name} under {config.raw_dir}. "
            "Run scripts/download_kaggle_data.py first."
        )
    return normalize_columns(pd.read_csv(csv_files[0]))


def resolve_target_column(df: pd.DataFrame, preferred_target: str) -> str:
    """Find the target column for a dataset."""

    if preferred_target in df.columns:
        return preferred_target

    candidates = [column for column in df.columns if preferred_target in column]
    if candidates:
        return candidates[0]

    if preferred_target == "churn":
        churn_candidates = [column for column in df.columns if "churn" in column]
        if churn_candidates:
            return churn_candidates[0]

    raise ValueError(
        f"Target column '{preferred_target}' not found. Available columns: {list(df.columns)}"
    )


def build_schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact schema and missingness report."""

    row_count = len(df)
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": [int(df[column].isna().sum()) for column in df.columns],
            "missing_pct": [
                float(df[column].isna().sum() / row_count) if row_count else 0.0
                for column in df.columns
            ],
            "unique_count": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )


def write_profile(config: DatasetConfig, output_dir: Path) -> Path:
    """Load a configured dataset and write a schema profile CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_primary_csv(config)
    schema = build_schema_report(df)
    output_path = output_dir / f"{config.name}_schema.csv"
    schema.to_csv(output_path, index=False)
    return output_path

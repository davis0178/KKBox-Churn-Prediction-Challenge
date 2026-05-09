"""Business-facing analysis summaries."""

from __future__ import annotations

import pandas as pd


def target_rate_by_segment(
    df: pd.DataFrame,
    target: str,
    segment: str,
    min_count: int = 10,
) -> pd.DataFrame:
    """Summarize target rate by one categorical segment."""

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")
    if segment not in df.columns:
        raise KeyError(f"Segment column '{segment}' not found.")

    summary = (
        df.groupby(segment, dropna=False)[target]
        .agg(count="size", target_rate="mean")
        .reset_index()
        .sort_values(["target_rate", "count"], ascending=[False, False])
    )
    return summary[summary["count"] >= min_count].reset_index(drop=True)

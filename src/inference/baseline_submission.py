"""Submission helpers for the Kaggle baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_train_and_submission_users(
    data_dir,
    sample_submission_path,
    train_file: str = "train_v2.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train labels and submission users, then build the combined base frame."""
    data_path = Path(data_dir)
    train = pd.read_csv(
        data_path / train_file,
        dtype={"msno": "string", "is_churn": "float32"},
    )
    sample_submission = pd.read_csv(
        sample_submission_path,
        dtype={"msno": "string"},
        usecols=["msno"],
    )

    test_template = sample_submission.copy()
    test_template["is_churn"] = np.nan
    final_base_data = pd.concat([train, test_template], axis=0, ignore_index=True)

    return train, sample_submission, final_base_data


def make_submission(
    sample_submission: pd.DataFrame,
    test_ids: Sequence[object],
    predictions: Sequence[float],
    output_path=None,
) -> pd.DataFrame:
    """Merge predictions into sample-submission order and optionally write CSV."""
    preds = pd.DataFrame(
        {
            "msno": pd.Series(test_ids, dtype="string").to_numpy(),
            "is_churn": predictions,
        }
    )
    submission = sample_submission[["msno"]].merge(
        preds,
        on="msno",
        how="left",
        sort=False,
    )

    if len(submission) != len(sample_submission):
        raise AssertionError(
            f"Expected {len(sample_submission):,} submission rows, "
            f"got {len(submission):,}."
        )
    if submission["is_churn"].isna().any():
        missing_count = int(submission["is_churn"].isna().sum())
        raise ValueError(f"Submission contains {missing_count:,} missing predictions.")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output, index=False)

    return submission


__all__ = ["load_train_and_submission_users", "make_submission"]


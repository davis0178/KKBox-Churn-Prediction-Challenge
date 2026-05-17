"""Feature engineering helpers for the Kaggle baseline notebook."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_TRANSACTION_FILES = ("transactions.csv", "transactions_v2.csv")
DEFAULT_USER_LOG_FILES = ("user_logs.csv", "user_logs_v2.csv")

TRANSACTION_MEAN_COLUMNS = [
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "is_cancel",
    "payment_price_diff",
]
TRANSACTION_AGG_COLUMNS = [*TRANSACTION_MEAN_COLUMNS, "n_transactions"]
TRANSACTION_DTYPES = {
    "msno": "string",
    "payment_method_id": "float32",
    "payment_plan_days": "float32",
    "plan_list_price": "float32",
    "actual_amount_paid": "float32",
    "is_auto_renew": "float32",
    "transaction_date": "int32",
    "is_cancel": "float32",
}
TRANSACTION_USECOLS = [
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "is_cancel",
]

LOG_VALUE_COLUMNS = [
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
]
LOG_AGG_COLUMNS = [
    "ul_active_days",
    *[f"ul_{column}_sum" for column in LOG_VALUE_COLUMNS],
    *[f"ul_{column}_mean" for column in LOG_VALUE_COLUMNS],
]
LOG_USECOLS = ["msno", "date", *LOG_VALUE_COLUMNS]
LOG_DTYPES = {
    "msno": "string",
    "date": "int32",
    "num_25": "float32",
    "num_50": "float32",
    "num_75": "float32",
    "num_985": "float32",
    "num_100": "float32",
    "num_unq": "float32",
    "total_secs": "float32",
}


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _require_sparse():
    try:
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "scipy is required to build sparse baseline matrices. "
            "Install this project's requirements."
        ) from exc

    return sparse


def _normalize_target_msnos(target_msnos) -> set[str]:
    return set(pd.Series(list(target_msnos), dtype="string").dropna().astype(str))


def load_member_features(data_dir, members_file: str = "members_v3.csv") -> pd.DataFrame:
    """Load and transform static member profile features."""
    member_dtypes = {
        "msno": "string",
        "city": "float32",
        "bd": "float32",
        "gender": "string",
        "registered_via": "float32",
        "registration_init_time": "string",
    }
    members = pd.read_csv(Path(data_dir) / members_file, dtype=member_dtypes)

    members["gender"] = (
        members["gender"].map({"female": 1, "male": 2}).fillna(0).astype("int8")
    )
    registration_date = pd.to_datetime(
        members["registration_init_time"],
        format="%Y%m%d",
        errors="coerce",
    )
    members["reg_fulldate"] = pd.to_numeric(
        members["registration_init_time"],
        errors="coerce",
    ).astype("float32")
    members["reg_year"] = registration_date.dt.year.astype("float32")
    members["reg_month"] = registration_date.dt.month.astype("float32")
    members["reg_mday"] = registration_date.dt.day.astype("float32")
    members["reg_wday"] = (
        (registration_date.dt.dayofweek + 1) % 7 + 1
    ).astype("float32")

    return members.drop(columns=["registration_init_time"])


def build_member_features(
    base_df: pd.DataFrame,
    member_features: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge static member features onto a base frame by ``msno``."""
    base = base_df.copy()
    base["msno"] = base["msno"].astype("string")
    if member_features is None:
        return base

    members = member_features.copy()
    members["msno"] = members["msno"].astype("string")
    return base.merge(members, on="msno", how="left", sort=False)


def empty_transaction_agg() -> pd.DataFrame:
    """Return an empty transaction aggregate frame with baseline columns."""
    return pd.DataFrame(
        {
            "msno": pd.Series(dtype="string"),
            **{
                column: pd.Series(dtype="float32")
                for column in TRANSACTION_AGG_COLUMNS
            },
        }
    )


def finalize_transaction_parts(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-chunk transaction aggregate sums into final mean/count features."""
    if not parts:
        return empty_transaction_agg()

    transaction_sums = pd.concat(parts).groupby(level=0, observed=True).sum()
    transactions_agg = transaction_sums.copy()
    transactions_agg[TRANSACTION_MEAN_COLUMNS] = transactions_agg[
        TRANSACTION_MEAN_COLUMNS
    ].div(transactions_agg["n_transactions"], axis=0)
    transactions_agg = transactions_agg.reset_index()
    transactions_agg["msno"] = transactions_agg["msno"].astype("string")
    for column in TRANSACTION_AGG_COLUMNS:
        transactions_agg[column] = transactions_agg[column].astype("float32")
    return transactions_agg


def build_transaction_aggregates_by_window(
    data_dir,
    target_msnos_by_window: Mapping[str, object],
    cutoff_by_window: Mapping[str, int],
    transaction_files: Sequence[str] = DEFAULT_TRANSACTION_FILES,
    chunksize: int = 2_000_000,
    logger: Callable[[str], None] | None = print,
) -> dict[str, pd.DataFrame]:
    """Build transaction aggregate features for each target-user/cutoff window."""
    if set(target_msnos_by_window) != set(cutoff_by_window):
        raise ValueError(
            "target_msnos_by_window and cutoff_by_window must have the same keys."
        )

    data_path = Path(data_dir)
    window_keys = list(target_msnos_by_window)
    normalized_targets = {
        key: _normalize_target_msnos(target_msnos_by_window[key])
        for key in window_keys
    }
    cutoffs = {key: int(cutoff_by_window[key]) for key in window_keys}
    parts_by_window = {key: [] for key in window_keys}
    rows_kept_by_window = {key: 0 for key in window_keys}
    rows_seen = 0

    for transaction_file in transaction_files:
        file_seen = 0
        for chunk_no, chunk in enumerate(
            pd.read_csv(
                data_path / transaction_file,
                usecols=TRANSACTION_USECOLS,
                dtype=TRANSACTION_DTYPES,
                chunksize=chunksize,
            ),
            start=1,
        ):
            rows_seen += len(chunk)
            file_seen += len(chunk)
            chunk_counts = {}

            for window_key in window_keys:
                target_msnos = normalized_targets[window_key]
                if not target_msnos:
                    continue

                window_chunk = chunk[
                    chunk["msno"].isin(target_msnos)
                    & chunk["transaction_date"].le(cutoffs[window_key])
                ].copy()
                if window_chunk.empty:
                    continue

                window_chunk["payment_price_diff"] = (
                    window_chunk["plan_list_price"]
                    - window_chunk["actual_amount_paid"]
                )
                grouped = window_chunk.groupby("msno", observed=True).agg(
                    payment_method_id=("payment_method_id", "sum"),
                    payment_plan_days=("payment_plan_days", "sum"),
                    plan_list_price=("plan_list_price", "sum"),
                    actual_amount_paid=("actual_amount_paid", "sum"),
                    is_auto_renew=("is_auto_renew", "sum"),
                    is_cancel=("is_cancel", "sum"),
                    payment_price_diff=("payment_price_diff", "sum"),
                    n_transactions=("payment_method_id", "count"),
                )
                parts_by_window[window_key].append(grouped.astype("float32"))
                rows_kept_by_window[window_key] += len(window_chunk)
                chunk_counts[window_key] = len(window_chunk)

                del window_chunk, grouped

            if chunk_counts:
                kept_summary = ", ".join(
                    f"{key}={value:,}" for key, value in chunk_counts.items()
                )
                _log(
                    logger,
                    f"{transaction_file} chunk {chunk_no}: "
                    f"file_seen={file_seen:,}, kept {kept_summary}",
                )
            else:
                _log(
                    logger,
                    f"{transaction_file} chunk {chunk_no}: no matching window rows",
                )

            del chunk
            gc.collect()

    aggregates_by_window = {
        key: finalize_transaction_parts(parts_by_window[key]) for key in window_keys
    }
    _log(logger, f"transaction rows seen: {rows_seen:,}")
    for key in window_keys:
        _log(
            logger,
            f"{key}: transaction rows kept={rows_kept_by_window[key]:,}, "
            f"users with transaction features={len(aggregates_by_window[key]):,}",
        )

    return aggregates_by_window


def add_frames(
    left: pd.DataFrame | None,
    right: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Add two aggregate frames while preserving missing users."""
    if right is None or right.empty:
        return left
    right = right.astype("float64")
    if left is None:
        return right
    return left.add(right, fill_value=0)


def empty_user_logs_agg() -> pd.DataFrame:
    """Return an empty user-log aggregate frame with baseline columns."""
    return pd.DataFrame(
        {
            "msno": pd.Series(dtype="string"),
            **{column: pd.Series(dtype="float32") for column in LOG_AGG_COLUMNS},
        }
    )


def finalize_user_log_aggregates(
    log_sums: pd.DataFrame | None,
    log_active_days: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine per-chunk user-log sums and active-day counts."""
    if log_sums is None or log_active_days is None:
        return empty_user_logs_agg()

    user_logs_agg = log_active_days.join(log_sums, how="outer")
    for column in LOG_VALUE_COLUMNS:
        user_logs_agg[f"ul_{column}_sum"] = user_logs_agg[column]
        user_logs_agg[f"ul_{column}_mean"] = user_logs_agg[column].div(
            user_logs_agg["ul_active_days"]
        )
    user_logs_agg = user_logs_agg.drop(columns=LOG_VALUE_COLUMNS).reset_index()
    user_logs_agg["msno"] = user_logs_agg["msno"].astype("string")

    for column in LOG_AGG_COLUMNS:
        user_logs_agg[column] = user_logs_agg[column].astype("float32")
    return user_logs_agg


def build_user_log_aggregates_by_window(
    data_dir,
    target_msnos_by_window: Mapping[str, object],
    cutoff_by_window: Mapping[str, int],
    user_log_files: Sequence[str] = DEFAULT_USER_LOG_FILES,
    chunksize: int = 2_000_000,
    logger: Callable[[str], None] | None = print,
) -> dict[str, pd.DataFrame]:
    """Build user-log aggregate features for each target-user/cutoff window."""
    if set(target_msnos_by_window) != set(cutoff_by_window):
        raise ValueError(
            "target_msnos_by_window and cutoff_by_window must have the same keys."
        )

    data_path = Path(data_dir)
    window_keys = list(target_msnos_by_window)
    normalized_targets = {
        key: _normalize_target_msnos(target_msnos_by_window[key])
        for key in window_keys
    }
    cutoffs = {key: int(cutoff_by_window[key]) for key in window_keys}
    log_sums_by_window = {key: None for key in window_keys}
    log_active_days_by_window = {key: None for key in window_keys}
    rows_kept_by_window = {key: 0 for key in window_keys}
    rows_seen = 0

    for log_file in user_log_files:
        file_seen = 0
        for chunk_no, chunk in enumerate(
            pd.read_csv(
                data_path / log_file,
                usecols=LOG_USECOLS,
                dtype=LOG_DTYPES,
                chunksize=chunksize,
            ),
            start=1,
        ):
            rows_seen += len(chunk)
            file_seen += len(chunk)
            chunk_counts = {}

            for window_key in window_keys:
                target_msnos = normalized_targets[window_key]
                if not target_msnos:
                    continue

                window_chunk = chunk[
                    chunk["msno"].isin(target_msnos)
                    & chunk["date"].le(cutoffs[window_key])
                ].drop_duplicates().copy()
                if window_chunk.empty:
                    continue

                grouped = window_chunk.groupby("msno", observed=True)
                sums = grouped[LOG_VALUE_COLUMNS].sum()
                active_days = grouped.size().rename("ul_active_days").to_frame()

                log_sums_by_window[window_key] = add_frames(
                    log_sums_by_window[window_key],
                    sums,
                )
                log_active_days_by_window[window_key] = add_frames(
                    log_active_days_by_window[window_key],
                    active_days,
                )
                rows_kept_by_window[window_key] += len(window_chunk)
                chunk_counts[window_key] = len(window_chunk)

                del window_chunk, grouped, sums, active_days

            if chunk_counts:
                kept_summary = ", ".join(
                    f"{key}={value:,}" for key, value in chunk_counts.items()
                )
                _log(
                    logger,
                    f"{log_file} chunk {chunk_no}: "
                    f"file_seen={file_seen:,}, kept {kept_summary}",
                )
            else:
                _log(logger, f"{log_file} chunk {chunk_no}: no matching window rows")

            del chunk
            gc.collect()

    aggregates_by_window = {
        key: finalize_user_log_aggregates(
            log_sums_by_window[key],
            log_active_days_by_window[key],
        )
        for key in window_keys
    }
    _log(logger, f"user log rows seen: {rows_seen:,}")
    for key in window_keys:
        _log(
            logger,
            f"{key}: user log rows kept={rows_kept_by_window[key]:,}, "
            f"users with user log features={len(aggregates_by_window[key]):,}",
        )

    return aggregates_by_window


def build_model_frame(
    base_df: pd.DataFrame,
    transaction_agg: pd.DataFrame,
    user_log_agg: pd.DataFrame,
    use_duplicate: bool = False,
    member_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Merge baseline features and return the model frame plus numeric features."""
    model_df = base_df.copy()
    model_df["msno"] = model_df["msno"].astype("string")

    if "is_duplicate" in model_df.columns:
        model_df = model_df.drop(columns=["is_duplicate"])
    if use_duplicate:
        model_df["is_duplicate"] = model_df["msno"].duplicated(keep=False).astype(
            "int8"
        )

    model_df = build_member_features(model_df, member_features)

    transaction_agg = transaction_agg.copy()
    transaction_agg["msno"] = transaction_agg["msno"].astype("string")
    user_log_agg = user_log_agg.copy()
    user_log_agg["msno"] = user_log_agg["msno"].astype("string")

    model_df = model_df.merge(transaction_agg, on="msno", how="left", sort=False)
    model_df = model_df.merge(user_log_agg, on="msno", how="left", sort=False)

    feature_names = [
        column for column in model_df.columns if column not in {"msno", "is_churn"}
    ]
    if any(column in feature_names for column in ["msno", "is_churn"]):
        raise AssertionError("Feature matrix includes an identifier or target column.")

    features = (
        model_df[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float32")
    )
    return model_df, features, feature_names


def build_window_dataset(
    labels_df: pd.DataFrame,
    transaction_agg: pd.DataFrame,
    user_log_agg: pd.DataFrame,
    use_duplicate: bool = False,
    member_features: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build a sparse modelling dataset for one labelled rolling window."""
    sparse = _require_sparse()
    labels = labels_df.loc[:, ["msno", "is_churn"]].copy()
    labels["msno"] = labels["msno"].astype("string")
    labels["is_churn"] = labels["is_churn"].astype("float32")

    model_df, features, feature_names = build_model_frame(
        labels,
        transaction_agg=transaction_agg,
        user_log_agg=user_log_agg,
        use_duplicate=use_duplicate,
        member_features=member_features,
    )
    y = model_df["is_churn"].astype("float32").to_numpy()
    if len(y) != len(labels_df):
        raise AssertionError(f"Expected {len(labels_df):,} labels, got {len(y):,}.")

    return {
        "data": model_df,
        "features": features,
        "feature_names": feature_names,
        "matrix": sparse.csr_matrix(features.to_numpy(dtype=np.float32)),
        "y": y,
    }


__all__ = [
    "DEFAULT_TRANSACTION_FILES",
    "DEFAULT_USER_LOG_FILES",
    "LOG_AGG_COLUMNS",
    "LOG_DTYPES",
    "LOG_USECOLS",
    "LOG_VALUE_COLUMNS",
    "TRANSACTION_AGG_COLUMNS",
    "TRANSACTION_DTYPES",
    "TRANSACTION_MEAN_COLUMNS",
    "TRANSACTION_USECOLS",
    "add_frames",
    "build_member_features",
    "build_model_frame",
    "build_transaction_aggregates_by_window",
    "build_user_log_aggregates_by_window",
    "build_window_dataset",
    "empty_transaction_agg",
    "empty_user_logs_agg",
    "finalize_transaction_parts",
    "finalize_user_log_aggregates",
    "load_member_features",
]


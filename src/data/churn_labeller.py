"""Reusable WSDM KKBox churn label generation utilities.

The logic in this module mirrors the official WSDM churn labeller ordering and
renewal-gap rules while keeping the prediction window configurable.
"""

from __future__ import annotations

from datetime import datetime
from functools import cmp_to_key
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_TRANSACTION_FILES = ("transactions.csv", "transactions_v2.csv")
REQUIRED_TRANSACTION_COLUMNS = (
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to generate churn labels. "
            "Install it with `pip install pandas pyarrow` or install this project's requirements."
        ) from exc

    return pd


def _pandas_string_dtype(pd):
    try:
        pd.Series([""], dtype="string[pyarrow]")
    except (ImportError, TypeError, ValueError):
        return "string"
    return "string[pyarrow]"


def find_repo_root(start=None) -> Path:
    """Find the repository root containing data/raw."""
    start_path = Path.cwd() if start is None else Path(start)
    for path in [start_path, *start_path.parents]:
        if (path / "data" / "raw").exists():
            return path
    raise FileNotFoundError(
        "Could not find data/raw. Run from the repo root or a child directory."
    )


def read_transactions(
    data_dir,
    transaction_files: Sequence[str] = DEFAULT_TRANSACTION_FILES,
    drop_duplicates: bool = True,
    usecols: Sequence[str] | None = None,
):
    """Read and concatenate transaction CSV files as a pandas DataFrame.

    CSV fields are read as pandas strings. Pass
    ``usecols=REQUIRED_TRANSACTION_COLUMNS`` when only churn labels are needed.
    """
    pd = _require_pandas()
    data_path = Path(data_dir)
    file_names = tuple(transaction_files)
    if not file_names:
        raise ValueError("transaction_files must contain at least one file name.")

    missing = [name for name in file_names if not (data_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing transaction files under {data_path}: {missing}")

    string_dtype = _pandas_string_dtype(pd)
    read_kwargs = {"dtype": string_dtype}
    if usecols is not None:
        read_kwargs["usecols"] = list(usecols)

    transaction_dfs = [
        pd.read_csv(data_path / file_name, **read_kwargs) for file_name in file_names
    ]
    data = pd.concat(transaction_dfs, ignore_index=True)

    if drop_duplicates:
        data = data.drop_duplicates(ignore_index=True)

    return data


def _field(row, name: str) -> str:
    value = row[name]
    return "" if value is None else str(value)


def _transaction_signature(row) -> str:
    return (
        _field(row, "plan_list_price")
        + _field(row, "payment_plan_days")
        + _field(row, "payment_method_id")
    )


def _transaction_precedes(x, y) -> bool:
    x_transaction_date = _field(x, "transaction_date")
    y_transaction_date = _field(y, "transaction_date")

    if x_transaction_date != y_transaction_date:
        return x_transaction_date < y_transaction_date

    x_signature = _transaction_signature(x)
    y_signature = _transaction_signature(y)

    if x_signature != y_signature:
        return x_signature > y_signature

    x_is_cancel = _field(x, "is_cancel")
    y_is_cancel = _field(y, "is_cancel")
    x_expire_date = _field(x, "membership_expire_date")
    y_expire_date = _field(y, "membership_expire_date")

    if x_is_cancel == "1" and y_is_cancel == "1":
        return x_expire_date > y_expire_date
    if x_is_cancel == "0" and y_is_cancel == "0":
        return x_expire_date < y_expire_date

    return x_is_cancel < y_is_cancel


def _compare_transactions(x, y) -> int:
    if _transaction_precedes(x, y):
        return -1
    if _transaction_precedes(y, x):
        return 1
    return 0


def _ordered_transactions(rows: Iterable) -> list:
    return sorted(list(rows or []), key=cmp_to_key(_compare_transactions))


def _sort_transactions_pandas(transactions_df):
    pd = _require_pandas()
    string_dtype = _pandas_string_dtype(pd)
    data = transactions_df.copy()
    sort_data = data.copy()

    for column in (
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
    ):
        sort_data[column] = sort_data[column].astype(string_dtype).fillna("")

    sort_data["_transaction_signature"] = (
        sort_data["plan_list_price"]
        + sort_data["payment_plan_days"]
        + sort_data["payment_method_id"]
    )

    expire_numeric = pd.to_numeric(
        sort_data["membership_expire_date"], errors="coerce"
    )
    expire_order = pd.Series(0.0, index=sort_data.index)
    cancel_mask = sort_data["is_cancel"].eq("1")
    active_mask = sort_data["is_cancel"].eq("0")
    expire_order.loc[active_mask] = expire_numeric.loc[active_mask].fillna(
        float("-inf")
    )
    expire_order.loc[cancel_mask] = -expire_numeric.loc[cancel_mask].fillna(
        float("-inf")
    )
    sort_data["_expire_order"] = expire_order

    sorted_index = sort_data.sort_values(
        by=[
            "transaction_date",
            "_transaction_signature",
            "is_cancel",
            "_expire_order",
        ],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).index

    return data.loc[sorted_index]


def _parse_yyyymmdd(value):
    return datetime.strptime(str(value), "%Y%m%d").date()


def _validate_yyyymmdd(value: str, name: str) -> None:
    try:
        _parse_yyyymmdd(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a YYYYMMDD string, got {value!r}.") from exc


def calculate_last_day(rows):
    """Return the latest effective membership expiration date from transaction rows."""
    ordered_rows = _ordered_transactions(rows)
    if not ordered_rows:
        return None
    return _field(ordered_rows[-1], "membership_expire_date")


def calculate_renewal_gap(rows, last_expiration):
    """Return days between effective expiration and the next non-cancel transaction."""
    ordered_rows = _ordered_transactions(rows)
    last_expire_date = _parse_yyyymmdd(last_expiration)
    gap = 9999

    for row in ordered_rows:
        if gap != 9999:
            break

        transaction_date = _parse_yyyymmdd(_field(row, "transaction_date"))
        expire_date = _parse_yyyymmdd(_field(row, "membership_expire_date"))
        is_cancel = _field(row, "is_cancel")

        if is_cancel == "1":
            if expire_date < last_expire_date:
                last_expire_date = expire_date
        else:
            gap = (transaction_date - last_expire_date).days

    return int(gap)


def _validate_transaction_columns(transactions_df) -> None:
    missing = [
        column
        for column in REQUIRED_TRANSACTION_COLUMNS
        if column not in transactions_df.columns
    ]
    if missing:
        raise ValueError(f"transactions_df is missing required columns: {missing}")


def _validate_make_label_args(
    transactions_df,
    history_cutoff: str,
    target_expire_start: str,
    target_expire_end: str,
    churn_gap_days: int,
    history_start: str | None,
) -> None:
    for value, name in (
        (history_cutoff, "history_cutoff"),
        (target_expire_start, "target_expire_start"),
        (target_expire_end, "target_expire_end"),
    ):
        _validate_yyyymmdd(value, name)
    if history_start is not None:
        _validate_yyyymmdd(history_start, "history_start")
    if churn_gap_days <= 0:
        raise ValueError("churn_gap_days must be positive.")

    _validate_transaction_columns(transactions_df)


def _empty_pandas_labels(pd):
    return pd.DataFrame(
        {
            "msno": pd.Series(dtype=_pandas_string_dtype(pd)),
            "is_churn": pd.Series(dtype=bool),
        }
    )


def make_churn_labels(
    transactions_df,
    history_cutoff: str,
    target_expire_start: str,
    target_expire_end: str,
    churn_gap_days: int = 30,
    history_start: str | None = None,
):
    """Generate a pandas DataFrame with columns msno and is_churn.

    Parameters are YYYYMMDD strings. ``history_start`` is optional; when omitted,
    all transactions up to ``history_cutoff`` are used to identify the current
    expiration date.
    """
    pd = _require_pandas()
    _validate_make_label_args(
        transactions_df,
        history_cutoff,
        target_expire_start,
        target_expire_end,
        churn_gap_days,
        history_start,
    )

    data = transactions_df.loc[:, REQUIRED_TRANSACTION_COLUMNS].copy()
    string_dtype = _pandas_string_dtype(pd)
    for column in REQUIRED_TRANSACTION_COLUMNS:
        data[column] = data[column].astype(string_dtype)

    history_filter = data["transaction_date"].le(history_cutoff).fillna(False)
    if history_start is not None:
        history_filter = (
            data["transaction_date"].ge(history_start).fillna(False) & history_filter
        )
    future_filter = data["transaction_date"].gt(history_cutoff).fillna(False)

    history_data = data.loc[history_filter]
    if history_data.empty:
        return _empty_pandas_labels(pd)

    history_sorted = _sort_transactions_pandas(history_data)
    user_expire = (
        history_sorted.drop_duplicates("msno", keep="last")
        .loc[:, ["msno", "membership_expire_date"]]
        .rename(columns={"membership_expire_date": "last_expire"})
    )

    candidate_filter = (
        user_expire["last_expire"].ge(target_expire_start).fillna(False)
        & user_expire["last_expire"].le(target_expire_end).fillna(False)
    )
    prediction_candidates = user_expire.loc[candidate_filter, ["msno", "last_expire"]]
    if prediction_candidates.empty:
        return _empty_pandas_labels(pd)

    future_candidates = data.loc[future_filter].merge(
        prediction_candidates, on="msno", how="inner", sort=False
    )
    no_activity = prediction_candidates.loc[
        ~prediction_candidates["msno"].isin(future_candidates["msno"]), ["msno"]
    ].copy()
    no_activity["is_churn"] = True

    renewal_rows = []
    if not future_candidates.empty:
        future_candidates = _sort_transactions_pandas(future_candidates)
        for (msno, last_expire), group in future_candidates.groupby(
            ["msno", "last_expire"], sort=False, dropna=False
        ):
            rows = group.loc[:, REQUIRED_TRANSACTION_COLUMNS[1:]].to_dict("records")
            gap = calculate_renewal_gap(rows, last_expire)
            renewal_rows.append(
                {
                    "msno": msno,
                    "is_churn": bool(gap >= churn_gap_days),
                }
            )

    renewal_labels = pd.DataFrame(renewal_rows, columns=["msno", "is_churn"])
    labels = pd.concat(
        [renewal_labels, no_activity.loc[:, ["msno", "is_churn"]]],
        ignore_index=True,
    )
    if labels.empty:
        return _empty_pandas_labels(pd)

    labels = labels.loc[:, ["msno", "is_churn"]]
    labels["is_churn"] = labels["is_churn"].astype(bool)
    return labels


__all__ = [
    "DEFAULT_TRANSACTION_FILES",
    "REQUIRED_TRANSACTION_COLUMNS",
    "calculate_last_day",
    "calculate_renewal_gap",
    "find_repo_root",
    "make_churn_labels",
    "read_transactions",
]

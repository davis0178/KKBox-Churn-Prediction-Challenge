"""Rolling-label and rolling-validation helpers for the Kaggle baseline."""

from __future__ import annotations

import calendar
import gc
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.churn_labeller import (
    REQUIRED_TRANSACTION_COLUMNS,
    make_churn_labels,
    read_transactions,
)
from src.features.baseline_features import build_window_dataset


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required to run rolling validation. "
            "Install this project's requirements."
        ) from exc

    return xgb


def _require_sparse():
    try:
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "scipy is required to run rolling validation. "
            "Install this project's requirements."
        ) from exc

    return sparse


def _require_log_loss():
    try:
        from sklearn.metrics import log_loss
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required to compute rolling validation log loss. "
            "Install this project's requirements."
        ) from exc

    return log_loss


def _yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def make_month_label_window(year: int, month: int) -> dict[str, str]:
    """Return the label-generation config for one target expiration month."""
    target_start = date(year, month, 1)
    target_end = date(year, month, calendar.monthrange(year, month)[1])
    history_cutoff = target_start - timedelta(days=1)
    return {
        "window": f"{year:04d}_{month:02d}",
        "history_cutoff": _yyyymmdd(history_cutoff),
        "target_expire_start": _yyyymmdd(target_start),
        "target_expire_end": _yyyymmdd(target_end),
    }


def generate_rolling_labels(
    data_dir,
    label_windows: Sequence[Mapping[str, str]],
    transaction_files: Sequence[str],
    official_train_path=None,
    logger: Callable[[str], None] | None = print,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Generate rolling churn labels for configured monthly windows."""
    rolling_labels_by_window: dict[str, pd.DataFrame] = {}
    rolling_label_summary = []

    rolling_transactions_for_labels = read_transactions(
        data_dir,
        transaction_files=transaction_files,
        usecols=REQUIRED_TRANSACTION_COLUMNS,
    )
    _log(
        logger,
        "transaction rows available for labelling: "
        f"{len(rolling_transactions_for_labels):,}",
    )

    for window_config in label_windows:
        window_key = str(window_config["window"])
        labels = make_churn_labels(
            transactions_df=rolling_transactions_for_labels,
            history_cutoff=str(window_config["history_cutoff"]),
            target_expire_start=str(window_config["target_expire_start"]),
            target_expire_end=str(window_config["target_expire_end"]),
        )
        if labels.empty:
            raise AssertionError(f"Rolling label window {window_key} is empty.")

        labels = labels.loc[:, ["msno", "is_churn"]].copy()
        labels["msno"] = labels["msno"].astype("string")
        label_values = set(labels["is_churn"].dropna().astype("int8").unique().tolist())
        if not label_values <= {0, 1}:
            raise AssertionError(f"Unexpected label values for {window_key}: {label_values}")
        labels["is_churn"] = labels["is_churn"].astype("float32")

        rolling_labels_by_window[window_key] = labels
        rolling_label_summary.append(
            {
                "window": window_key,
                "history_cutoff": window_config["history_cutoff"],
                "target_expire_start": window_config["target_expire_start"],
                "target_expire_end": window_config["target_expire_end"],
                "rows": len(labels),
                "churn_rate": float(labels["is_churn"].mean()),
            }
        )
        _log(
            logger,
            f"{window_key}: labels={len(labels):,}, "
            f"churn_rate={labels['is_churn'].mean():.5f}",
        )

    if official_train_path is not None and "2017_02" in rolling_labels_by_window:
        official_path = Path(official_train_path)
        if official_path.exists():
            official_feb = pd.read_csv(
                official_path,
                dtype={"msno": "string", "is_churn": "int8"},
            )
            feb_labels = rolling_labels_by_window["2017_02"].rename(
                columns={"is_churn": "is_churn_labeller"}
            )
            sanity = official_feb.merge(feb_labels, on="msno", how="inner", sort=False)
            sanity_match_rate = sanity["is_churn"].astype("int8").eq(
                sanity["is_churn_labeller"].astype("int8")
            ).mean()
            _log(
                logger,
                "2017_02 official train.csv sanity: "
                f"official_rows={len(official_feb):,}, "
                f"labeller_rows={len(feb_labels):,}, "
                f"intersection={len(sanity):,}, "
                f"match_rate={sanity_match_rate:.6f}, "
                f"official_churn_rate={official_feb['is_churn'].mean():.5f}, "
                f"labeller_churn_rate={feb_labels['is_churn_labeller'].mean():.5f}",
            )
            del official_feb, feb_labels, sanity

    del rolling_transactions_for_labels
    gc.collect()

    return rolling_labels_by_window, pd.DataFrame(rolling_label_summary)


def run_rolling_validation(
    rolling_labels_by_window: Mapping[str, pd.DataFrame],
    transaction_aggs_by_window: Mapping[str, pd.DataFrame],
    user_log_aggs_by_window: Mapping[str, pd.DataFrame],
    member_features: pd.DataFrame | None,
    label_windows: Sequence[Mapping[str, str]],
    xgb_params: Mapping[str, object],
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    use_duplicate: bool = False,
    logger: Callable[[str], None] | None = print,
) -> tuple[pd.DataFrame, int]:
    """Run expanding-window validation and return results plus selected rounds."""
    xgb = _require_xgboost()
    sparse = _require_sparse()
    log_loss = _require_log_loss()

    rolling_datasets_by_window = {}
    rolling_feature_names = None
    ordered_window_keys = [
        str(window_config["window"])
        for window_config in label_windows
        if str(window_config["window"]) in rolling_labels_by_window
    ]
    if len(ordered_window_keys) < 2:
        raise AssertionError("Rolling validation needs at least two labelled windows.")

    for window_key in ordered_window_keys:
        dataset = build_window_dataset(
            labels_df=rolling_labels_by_window[window_key],
            transaction_agg=transaction_aggs_by_window[window_key],
            user_log_agg=user_log_aggs_by_window[window_key],
            use_duplicate=use_duplicate,
            member_features=member_features,
        )
        if len(dataset["y"]) != len(rolling_labels_by_window[window_key]):
            raise AssertionError(f"{window_key}: y length does not match label rows.")
        if rolling_feature_names is None:
            rolling_feature_names = dataset["feature_names"]
        elif dataset["feature_names"] != rolling_feature_names:
            raise AssertionError(
                f"{window_key}: rolling feature columns differ from previous windows."
            )

        rolling_datasets_by_window[window_key] = dataset
        _log(
            logger,
            f"{window_key}: matrix={dataset['matrix'].shape}, "
            f"churn_rate={dataset['y'].mean():.5f}",
        )

    rolling_result_rows = []
    for fold_idx in range(1, len(ordered_window_keys)):
        train_windows = ordered_window_keys[:fold_idx]
        valid_window = ordered_window_keys[fold_idx]

        x_train_roll = sparse.vstack(
            [rolling_datasets_by_window[key]["matrix"] for key in train_windows],
            format="csr",
        )
        y_train_roll = np.concatenate(
            [rolling_datasets_by_window[key]["y"] for key in train_windows]
        )
        x_valid_roll = rolling_datasets_by_window[valid_window]["matrix"]
        y_valid_roll = rolling_datasets_by_window[valid_window]["y"]

        if len(y_valid_roll) != len(rolling_labels_by_window[valid_window]):
            raise AssertionError(f"{valid_window}: validation label length mismatch.")

        dtrain_roll = xgb.DMatrix(
            data=x_train_roll,
            label=y_train_roll,
            missing=np.nan,
            feature_names=rolling_feature_names,
        )
        dvalid_roll = xgb.DMatrix(
            data=x_valid_roll,
            label=y_valid_roll,
            missing=np.nan,
            feature_names=rolling_feature_names,
        )

        _log(
            logger,
            f"rolling fold {fold_idx}: train_windows={train_windows}, "
            f"valid_window={valid_window}, train_rows={len(y_train_roll):,}, "
            f"valid_rows={len(y_valid_roll):,}",
        )
        rolling_model = xgb.train(
            params=dict(xgb_params),
            dtrain=dtrain_roll,
            num_boost_round=num_boost_round,
            evals=[(dtrain_roll, "train"), (dvalid_roll, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )

        best_iteration_attr = getattr(rolling_model, "best_iteration", None)
        if best_iteration_attr is None:
            best_iteration = num_boost_round
        else:
            best_iteration = int(best_iteration_attr) + 1

        best_score_attr = getattr(rolling_model, "best_score", None)
        if best_score_attr is None:
            valid_pred = rolling_model.predict(
                dvalid_roll,
                iteration_range=(0, best_iteration),
            )
            valid_logloss = float(log_loss(y_valid_roll, valid_pred, labels=[0, 1]))
        else:
            valid_logloss = float(best_score_attr)

        rolling_result_rows.append(
            {
                "valid_window": valid_window,
                "train_windows": ",".join(train_windows),
                "train_rows": int(len(y_train_roll)),
                "valid_rows": int(len(y_valid_roll)),
                "valid_churn_rate": float(np.mean(y_valid_roll)),
                "best_iteration": int(best_iteration),
                "valid_logloss": valid_logloss,
            }
        )

        del (
            x_train_roll,
            y_train_roll,
            x_valid_roll,
            y_valid_roll,
            dtrain_roll,
            dvalid_roll,
            rolling_model,
        )
        gc.collect()

    rolling_results = pd.DataFrame(rolling_result_rows)
    if rolling_results.empty:
        raise AssertionError("Rolling validation produced no folds.")

    best_iter = int(np.median(rolling_results["best_iteration"]))
    if best_iter < 1:
        raise AssertionError(f"best_iter must be positive, got {best_iter}.")

    return rolling_results, best_iter


__all__ = [
    "generate_rolling_labels",
    "make_month_label_window",
    "run_rolling_validation",
]


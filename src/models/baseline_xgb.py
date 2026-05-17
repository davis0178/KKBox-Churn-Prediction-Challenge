"""XGBoost helpers for the Kaggle baseline."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for baseline model training. "
            "Install this project's requirements."
        ) from exc

    return xgb


def default_xgb_params(random_state: int = 12345) -> dict[str, object]:
    """Return the tuned baseline XGBoost parameter set from the notebook."""
    return {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.02,
        "gamma": 1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": random_state,
    }


def make_dmatrix(
    data,
    label=None,
    feature_names: Sequence[str] | None = None,
    missing=np.nan,
):
    """Build an XGBoost DMatrix with the baseline missing-value convention."""
    xgb = _require_xgboost()
    kwargs = {
        "data": data,
        "missing": missing,
        "feature_names": feature_names,
    }
    if label is not None:
        kwargs["label"] = label
    return xgb.DMatrix(**kwargs)


def train_final_model(
    dtrain,
    xgb_params: Mapping[str, object],
    num_boost_round: int,
    verbose_eval: int | bool = 100,
):
    """Train the final baseline model."""
    xgb = _require_xgboost()
    return xgb.train(
        params=dict(xgb_params),
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train")],
        verbose_eval=verbose_eval,
    )


def feature_importance_frame(
    model,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Return feature importance as a sorted DataFrame."""
    importance = model.get_score(importance_type=importance_type)
    return (
        pd.DataFrame(
            [{"feature": feature, importance_type: value} for feature, value in importance.items()]
        )
        .sort_values(importance_type, ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    model,
    max_num_features: int,
    importance_type: str = "gain",
    height: float = 0.5,
):
    """Plot XGBoost feature importance and return the matplotlib axis."""
    xgb = _require_xgboost()
    return xgb.plot_importance(
        model,
        max_num_features=max_num_features,
        importance_type=importance_type,
        height=height,
    )


__all__ = [
    "default_xgb_params",
    "feature_importance_frame",
    "make_dmatrix",
    "plot_feature_importance",
    "train_final_model",
]


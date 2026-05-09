"""Reusable baseline modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_retention.config import DatasetConfig
from churn_retention.data import resolve_target_column


@dataclass(frozen=True)
class ModelResult:
    """Evaluation payload for one fitted model."""

    dataset: str
    model_name: str
    target_column: str
    row_count: int
    feature_count: int
    roc_auc: float | None
    pr_auc: float | None
    classification_report: dict[str, Any]


def _target_as_binary(series: pd.Series) -> pd.Series:
    """Convert common binary target encodings into 0/1 integers."""

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapped = series.astype(str).str.strip().str.lower().map(
        {
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "churn": 1,
            "churned": 1,
            "not churned": 0,
            "not_churned": 0,
            "active": 0,
            "inactive": 1,
        }
    )
    if mapped.isna().any():
        bad_values = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported target labels: {bad_values}")
    return mapped.astype(int)


def build_feature_target(
    df: pd.DataFrame,
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Split a dataset into feature matrix and binary target."""

    target_column = resolve_target_column(df, config.preferred_target)
    drop_columns = {target_column, *config.id_columns, *config.leakage_columns}
    available_drop_columns = [column for column in drop_columns if column in df.columns]
    x = df.drop(columns=available_drop_columns)
    y = _target_as_binary(df[target_column])
    return x, y, target_column


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical columns."""

    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in x.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=25)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Build baseline classification pipelines."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", "passthrough"),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", "passthrough"),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocess", "passthrough"),
                ("classifier", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def _score_probabilities(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    decision = model.decision_function(x_test)
    return 1 / (1 + np.exp(-decision))


def fit_and_evaluate(
    df: pd.DataFrame,
    config: DatasetConfig,
    model_name: str = "logistic_regression",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, ModelResult, pd.DataFrame, pd.Series]:
    """Fit one baseline model and return model, metrics, test data, and test target."""

    x, y, target_column = build_feature_target(df, config)
    if y.nunique() != 2:
        raise ValueError(f"Expected binary target for {config.name}; got {y.nunique()} classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train)
    models = make_models(random_state=random_state)
    if model_name not in models:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from {sorted(models)}.")

    model = models[model_name]
    model.set_params(preprocess=preprocessor)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_score = _score_probabilities(model, x_test)
    roc_auc = float(roc_auc_score(y_test, y_score))
    pr_auc = float(average_precision_score(y_test, y_score))

    result = ModelResult(
        dataset=config.name,
        model_name=model_name,
        target_column=target_column,
        row_count=len(df),
        feature_count=x.shape[1],
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        classification_report=classification_report(y_test, y_pred, output_dict=True),
    )
    return model, result, x_test, y_test


def write_evaluation_figures(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    """Write ROC, PR, and confusion matrix figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    roc_path = output_dir / f"{prefix}_roc_curve.png"
    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()
    written.append(roc_path)

    pr_path = output_dir / f"{prefix}_precision_recall_curve.png"
    PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=160)
    plt.close()
    written.append(pr_path)

    confusion_path = output_dir / f"{prefix}_confusion_matrix.png"
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=160)
    plt.close()
    written.append(confusion_path)

    return written

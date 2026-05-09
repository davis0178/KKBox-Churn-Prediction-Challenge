"""Project constants for datasets, paths, and modeling targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a Kaggle-backed dataset."""

    name: str
    kaggle_slug: str
    raw_dir: Path
    preferred_target: str
    id_columns: tuple[str, ...] = ()
    leakage_columns: tuple[str, ...] = ()


SUBSCRIPTION_CHURN = DatasetConfig(
    name="subscription_churn",
    kaggle_slug="jayjoshi37/customer-subscription-churn-and-usage-patterns",
    raw_dir=RAW_DATA_DIR / "subscription_churn",
    preferred_target="churn",
    id_columns=("customer_id", "customerid", "id"),
)

OTT_DROPOFF = DatasetConfig(
    name="ott_dropoff",
    kaggle_slug="eklavya16/ott-viewer-drop-off-and-retention-risk-dataset",
    raw_dir=RAW_DATA_DIR / "ott_dropoff",
    preferred_target="drop_off",
    id_columns=("show_id",),
    leakage_columns=("drop_off_probability", "retention_risk"),
)

DATASETS = (SUBSCRIPTION_CHURN, OTT_DROPOFF)

# Subscription Churn Retention Modeling

Portfolio project for subscription churn prediction, retention analysis, and OTT viewer drop-off risk modeling.

This repository uses two public Kaggle datasets as reproducible data sources and keeps all project code, analysis, and reporting local to this repo.

## Data Sources

| Dataset | Kaggle slug | Role |
| --- | --- | --- |
| Customer Subscription Churn and Usage Patterns | `jayjoshi37/customer-subscription-churn-and-usage-patterns` | Customer-level churn prediction and retention strategy analysis |
| OTT Viewer Drop-Off and Retention Risk Dataset | `eklavya16/ott-viewer-drop-off-and-retention-risk-dataset` | Episode-level drop-off prediction and content retention risk analysis |

The datasets are educational/synthetic sources. Results in this repository should be interpreted as portfolio and modeling demonstrations, not claims about real user populations.

## Project Structure

```text
.
├── data/
│   ├── raw/                  # Kaggle downloads, ignored by Git
│   └── processed/            # Local profiling outputs, ignored by Git
├── notebooks/                # EDA and modeling notebooks
├── reports/
│   └── figures/              # Generated plots, ignored by Git
├── scripts/                  # Download, validation, profiling, and training commands
├── src/churn_retention/      # Reusable project package
└── tests/                    # Unit tests for local project code
```

## Environment Setup

Use Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m ipykernel install --user --name churn-retention --display-name "Python (churn-retention)"
```

Validate the environment:

```powershell
.\.venv\Scripts\python scripts\validate_environment.py
```

## Kaggle Setup

Download `kaggle.json` from your Kaggle account settings and place it at:

```powershell
%USERPROFILE%\.kaggle\kaggle.json
```

Then download both datasets:

```powershell
.\.venv\Scripts\python scripts\download_kaggle_data.py
```

This writes files under `data/raw/`, which is intentionally ignored by Git.

## Run the Project

Profile downloaded data:

```powershell
.\.venv\Scripts\python scripts\profile_data.py
```

Train baseline models and write metrics/figures:

```powershell
.\.venv\Scripts\python scripts\train_baselines.py
```

Run tests and linting:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
```

## Modeling Scope

The project contains two supervised modeling tracks:

- Subscription churn: predict customer churn from subscription plan, fee, engagement, support, payment, tenure, and recent activity signals.
- OTT drop-off: predict episode-level drop-off from pacing, hook strength, content metadata, viewing behavior, and cognitive load features.

Baseline models include logistic regression and tree-based models. Evaluation reports include ROC-AUC, PR-AUC, classification reports, class balance figures, and feature importance summaries where supported.

## Notes

- Raw Kaggle data is never committed.
- Public reporting should cite the Kaggle dataset pages and note synthetic/educational usage.
- Kaggle notebooks may be useful references, but this repository's source code and analysis are written locally for reproducibility and portfolio review.

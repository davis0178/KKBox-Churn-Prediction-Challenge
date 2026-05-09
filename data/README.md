# Data Directory

This directory is intentionally split into local-only data zones:

- `raw/`: Kaggle downloads from `scripts/download_kaggle_data.py`.
- `processed/`: generated schema profiles and derived analysis tables.

Both zones are ignored by Git. Commit data dictionaries, source code, and reproducible scripts instead of raw Kaggle files.

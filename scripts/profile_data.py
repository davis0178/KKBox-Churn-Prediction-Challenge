"""Generate schema and missingness reports for downloaded Kaggle data."""

from __future__ import annotations

from churn_retention.config import DATASETS, PROCESSED_DATA_DIR
from churn_retention.data import write_profile


def main() -> int:
    written = []
    for dataset in DATASETS:
        output_path = write_profile(dataset, PROCESSED_DATA_DIR)
        written.append(output_path)
        print(f"Wrote {output_path}")

    print(f"Profiled {len(written)} datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

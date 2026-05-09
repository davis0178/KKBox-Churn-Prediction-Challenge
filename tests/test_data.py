from pathlib import Path

import pandas as pd

from churn_retention.data import build_schema_report, find_csv_files, normalize_columns


def test_normalize_columns():
    df = pd.DataFrame({"Customer ID": [1], "Drop-Off %": [0.2]})

    normalized = normalize_columns(df)

    assert normalized.columns.tolist() == ["customer_id", "drop_off_pct"]


def test_build_schema_report_counts_missing_values():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "x", "y"]})

    report = build_schema_report(df)

    assert report.loc[report["column"] == "a", "missing_count"].item() == 1
    assert report.loc[report["column"] == "b", "unique_count"].item() == 2


def test_find_csv_files_sorts_by_size(tmp_path: Path):
    small = tmp_path / "small.csv"
    large = tmp_path / "large.csv"
    small.write_text("a\n1\n", encoding="utf-8")
    large.write_text("a\n1\n2\n3\n", encoding="utf-8")

    files = find_csv_files(tmp_path)

    assert files == [large, small]

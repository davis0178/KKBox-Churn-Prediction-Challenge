import pandas as pd

from churn_retention.reporting import target_rate_by_segment


def test_target_rate_by_segment_filters_small_segments():
    df = pd.DataFrame(
        {
            "plan": ["basic", "basic", "pro"],
            "churn": [1, 0, 1],
        }
    )

    summary = target_rate_by_segment(df, target="churn", segment="plan", min_count=2)

    assert summary["plan"].tolist() == ["basic"]
    assert summary["target_rate"].item() == 0.5

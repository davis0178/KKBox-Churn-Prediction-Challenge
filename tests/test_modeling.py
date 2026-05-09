import pandas as pd

from churn_retention.config import SUBSCRIPTION_CHURN
from churn_retention.modeling import build_feature_target, fit_and_evaluate


def test_build_feature_target_drops_id_and_target():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "plan": ["basic", "pro", "basic", "pro"],
            "usage": [1.0, 5.0, 2.0, 6.0],
            "churn": [1, 0, 1, 0],
        }
    )

    x, y, target = build_feature_target(df, SUBSCRIPTION_CHURN)

    assert target == "churn"
    assert "customer_id" not in x.columns
    assert "churn" not in x.columns
    assert y.tolist() == [1, 0, 1, 0]


def test_fit_and_evaluate_returns_auc_metrics():
    df = pd.DataFrame(
        {
            "customer_id": list(range(40)),
            "plan": ["basic", "pro"] * 20,
            "usage": [float(i % 10) for i in range(40)],
            "support_tickets": [i % 3 for i in range(40)],
            "churn": [1 if i % 4 in (0, 1) else 0 for i in range(40)],
        }
    )

    _, result, _, _ = fit_and_evaluate(df, SUBSCRIPTION_CHURN, model_name="logistic_regression")

    assert result.dataset == "subscription_churn"
    assert result.roc_auc is not None
    assert result.pr_auc is not None

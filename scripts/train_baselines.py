"""Train baseline churn/drop-off classifiers and write evaluation artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict

from churn_retention.config import DATASETS, FIGURES_DIR, REPORTS_DIR
from churn_retention.data import load_primary_csv
from churn_retention.modeling import fit_and_evaluate, write_evaluation_figures


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = []
    figures = []
    for dataset in DATASETS:
        df = load_primary_csv(dataset)
        for model_name in ("logistic_regression", "random_forest", "gradient_boosting"):
            model, result, x_test, y_test = fit_and_evaluate(
                df=df,
                config=dataset,
                model_name=model_name,
            )
            metrics.append(asdict(result))
            if model_name == "random_forest":
                figures.extend(
                    write_evaluation_figures(
                        model=model,
                        x_test=x_test,
                        y_test=y_test,
                        output_dir=FIGURES_DIR,
                        prefix=f"{dataset.name}_{model_name}",
                    )
                )

    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics: {metrics_path}")
    for figure in figures:
        print(f"Wrote figure: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

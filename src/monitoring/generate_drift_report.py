from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models.train_baseline import BASE_DIR, load_params


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def bool_status(value: Any) -> str:
    return "обнаружен" if bool(value) else "не обнаружен"


def build_report(metrics: dict[str, Any], drift: dict[str, Any]) -> str:
    data_drift = drift.get("data_drift", {})
    target_drift = drift.get("target_drift", {})
    concept_drift = drift.get("concept_drift", {})
    top_features = data_drift.get("top_drifted_features", [])[:10]

    top_feature_lines = "\n".join(
        f"- `{feature}`: PSI {format_float(data_drift.get('per_feature', {}).get(feature, {}).get('psi'))}"
        for feature in top_features
    )
    if not top_feature_lines:
        top_feature_lines = "- Нет данных"

    return f"""# Drift Report

## Summary

| Drift type | Status |
| --- | --- |
| Data drift | {bool_status(data_drift.get("drift_detected", False))} |
| Target drift | {bool_status(target_drift.get("drift_detected", False))} |
| Concept drift | {bool_status(concept_drift.get("drift_detected", False))} |

## Dataset Windows

| Split | Windows |
| --- | ---: |
| Train | {drift.get("window_counts", {}).get("train", "n/a")} |
| Validation | {drift.get("window_counts", {}).get("val", "n/a")} |
| Test | {drift.get("window_counts", {}).get("test", "n/a")} |

## Model Quality

| Metric | Value |
| --- | ---: |
| PR-AUC | {format_float(metrics.get("pr_auc"))} |
| ROC-AUC | {format_float(metrics.get("roc_auc"))} |
| F1 | {format_float(metrics.get("f1"))} |
| Precision | {format_float(metrics.get("precision"))} |
| Recall | {format_float(metrics.get("recall"))} |
| Threshold | {format_float(metrics.get("threshold"))} |
| Test positive rate | {format_percent(metrics.get("test_positive_rate"))} |

## Data Drift

- Feature count: {data_drift.get("feature_count", "n/a")}
- Drifted feature count: {data_drift.get("drifted_feature_count", "n/a")}
- Mean PSI: {format_float(data_drift.get("mean_psi"))}
- Max PSI: {format_float(data_drift.get("max_psi"))}
- PSI threshold: {format_float(data_drift.get("drift_threshold"))}

Top features by PSI:

{top_feature_lines}

## Target Drift

- Reference positive rate: {format_percent(target_drift.get("reference_positive_rate"))}
- Current positive rate: {format_percent(target_drift.get("current_positive_rate"))}
- Absolute positive rate change: {format_percent(target_drift.get("absolute_positive_rate_change"))}
- PSI: {format_float(target_drift.get("psi"))}

## Concept Drift

- Baseline split: {concept_drift.get("baseline_split", "n/a")}
- Current split: {concept_drift.get("current_split", "n/a")}
- Primary metric: {concept_drift.get("primary_metric", "n/a")}
- Primary metric drop: {format_float(concept_drift.get("primary_metric_drop"))}
- Drift threshold: {format_float(concept_drift.get("drift_threshold"))}
"""


def save_report(report: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    params = load_params()
    metrics_params = params["metrics"]
    metrics = load_json(BASE_DIR / metrics_params["output_path"])
    drift = load_json(BASE_DIR / metrics_params["drift_output_path"])
    report_path = BASE_DIR / metrics_params.get(
        "drift_report_path",
        "reports/drift_report.md",
    )

    report = build_report(metrics, drift)
    save_report(report, report_path)
    print(f"Drift report saved: {report_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

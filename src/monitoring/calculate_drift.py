from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluate_baseline import (
    evaluate_model,
    load_model,
    load_preprocessing,
    prepare_split,
)
from src.models.train_baseline import BASE_DIR, load_params


DEFAULT_PSI_BINS = 10
DEFAULT_DATA_DRIFT_THRESHOLD = 0.2
DEFAULT_TARGET_DRIFT_THRESHOLD = 0.1
DEFAULT_CONCEPT_DRIFT_THRESHOLD = 0.05
DEFAULT_PRIMARY_CONCEPT_METRIC = "pr_auc"
CONCEPT_METRICS = ("pr_auc", "roc_auc", "f1", "precision", "recall")


def _as_clean_series(values: pd.Series | np.ndarray | list) -> pd.Series:
    return pd.Series(values).dropna()


def calculate_categorical_psi(
    reference: pd.Series | np.ndarray | list,
    current: pd.Series | np.ndarray | list,
    epsilon: float = 1e-6,
) -> float:
    reference_series = _as_clean_series(reference)
    current_series = _as_clean_series(current)

    categories = sorted(set(reference_series.unique()) | set(current_series.unique()))
    if not categories:
        return 0.0

    reference_counts = reference_series.value_counts(normalize=True)
    current_counts = current_series.value_counts(normalize=True)

    psi = 0.0
    for category in categories:
        reference_share = max(float(reference_counts.get(category, 0.0)), epsilon)
        current_share = max(float(current_counts.get(category, 0.0)), epsilon)
        psi += (current_share - reference_share) * np.log(current_share / reference_share)

    return float(psi)


def calculate_numeric_psi(
    reference: pd.Series | np.ndarray | list,
    current: pd.Series | np.ndarray | list,
    bins: int = DEFAULT_PSI_BINS,
    epsilon: float = 1e-6,
) -> float:
    reference_series = _as_clean_series(reference)
    current_series = _as_clean_series(current)

    if reference_series.empty or current_series.empty:
        return 0.0

    if reference_series.nunique() <= 2:
        return calculate_categorical_psi(reference_series, current_series, epsilon)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference_series.astype(float), quantiles))
    if len(edges) < 3:
        return calculate_categorical_psi(reference_series, current_series, epsilon)

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_bins = pd.cut(reference_series, bins=edges, include_lowest=True)
    current_bins = pd.cut(current_series, bins=edges, include_lowest=True)
    reference_share = reference_bins.value_counts(sort=False, normalize=True)
    current_share = current_bins.value_counts(sort=False, normalize=True)

    psi = 0.0
    for interval in reference_share.index:
        expected = max(float(reference_share.get(interval, 0.0)), epsilon)
        actual = max(float(current_share.get(interval, 0.0)), epsilon)
        psi += (actual - expected) * np.log(actual / expected)

    return float(psi)


def calculate_feature_drift(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    bins: int = DEFAULT_PSI_BINS,
    drift_threshold: float = DEFAULT_DATA_DRIFT_THRESHOLD,
) -> dict:
    per_feature = {}
    for feature in reference_features.columns:
        psi = calculate_numeric_psi(
            reference=reference_features[feature],
            current=current_features[feature],
            bins=bins,
        )
        per_feature[feature] = {
            "psi": psi,
            "drift_detected": bool(psi >= drift_threshold),
        }

    psi_values = [feature_metrics["psi"] for feature_metrics in per_feature.values()]
    drifted_features = [
        feature
        for feature, feature_metrics in per_feature.items()
        if feature_metrics["drift_detected"]
    ]

    return {
        "feature_count": len(per_feature),
        "drifted_feature_count": len(drifted_features),
        "drifted_feature_share": (
            float(len(drifted_features) / len(per_feature)) if per_feature else 0.0
        ),
        "mean_psi": float(np.mean(psi_values)) if psi_values else 0.0,
        "max_psi": float(np.max(psi_values)) if psi_values else 0.0,
        "drift_threshold": float(drift_threshold),
        "drift_detected": bool(drifted_features),
        "top_drifted_features": sorted(
            per_feature,
            key=lambda feature: per_feature[feature]["psi"],
            reverse=True,
        )[:10],
        "per_feature": per_feature,
    }


def calculate_target_drift(
    reference_target: pd.Series,
    current_target: pd.Series,
    drift_threshold: float = DEFAULT_TARGET_DRIFT_THRESHOLD,
) -> dict:
    reference_positive_rate = float(reference_target.mean())
    current_positive_rate = float(current_target.mean())
    absolute_positive_rate_change = abs(
        current_positive_rate - reference_positive_rate
    )
    psi = calculate_categorical_psi(reference_target, current_target)

    return {
        "reference_positive_rate": reference_positive_rate,
        "current_positive_rate": current_positive_rate,
        "absolute_positive_rate_change": float(absolute_positive_rate_change),
        "psi": psi,
        "drift_threshold": float(drift_threshold),
        "drift_detected": bool(absolute_positive_rate_change >= drift_threshold),
    }


def calculate_concept_drift(
    baseline_metrics: dict,
    current_metrics: dict,
    drift_threshold: float = DEFAULT_CONCEPT_DRIFT_THRESHOLD,
    primary_metric: str = DEFAULT_PRIMARY_CONCEPT_METRIC,
) -> dict:
    metric_deltas = {}
    metric_relative_drops = {}

    for metric in CONCEPT_METRICS:
        baseline_value = float(baseline_metrics[metric])
        current_value = float(current_metrics[metric])
        absolute_drop = baseline_value - current_value
        metric_deltas[metric] = {
            "baseline": baseline_value,
            "current": current_value,
            "absolute_drop": float(absolute_drop),
        }
        metric_relative_drops[metric] = (
            float(absolute_drop / abs(baseline_value)) if baseline_value != 0 else 0.0
        )

    primary_drop = metric_deltas[primary_metric]["absolute_drop"]
    return {
        "baseline_split": "val",
        "current_split": "test",
        "primary_metric": primary_metric,
        "drift_threshold": float(drift_threshold),
        "primary_metric_drop": float(primary_drop),
        "drift_detected": bool(primary_drop >= drift_threshold),
        "metric_deltas": metric_deltas,
        "metric_relative_drops": metric_relative_drops,
    }


def build_drift_report(params: dict) -> dict:
    data_params = params["data"]
    model_params = params["model"]
    metrics_params = params["metrics"]
    drift_params = params.get("drift", {})

    model = load_model(BASE_DIR / model_params["artifact_path"])
    preprocessing = load_preprocessing(BASE_DIR / model_params["preprocessing_path"])

    train_df = pd.read_csv(BASE_DIR / data_params["train_path"])
    val_df = pd.read_csv(BASE_DIR / data_params["val_path"])
    test_df = pd.read_csv(BASE_DIR / data_params["test_path"])

    X_train, y_train = prepare_split(train_df, preprocessing)
    X_val, y_val = prepare_split(val_df, preprocessing)
    X_test, y_test = prepare_split(test_df, preprocessing)

    threshold = metrics_params.get("prediction_threshold", 0.5)
    baseline_metrics = evaluate_model(model, X_val, y_val, threshold=threshold)
    current_metrics = evaluate_model(model, X_test, y_test, threshold=threshold)

    return {
        "reference_split": "train",
        "baseline_split": "val",
        "current_split": "test",
        "window_size": int(preprocessing["window_size"]),
        "window_counts": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "data_drift": calculate_feature_drift(
            reference_features=X_train,
            current_features=X_test,
            bins=drift_params.get("psi_bins", DEFAULT_PSI_BINS),
            drift_threshold=drift_params.get(
                "data_drift_threshold",
                DEFAULT_DATA_DRIFT_THRESHOLD,
            ),
        ),
        "target_drift": calculate_target_drift(
            reference_target=y_train,
            current_target=y_test,
            drift_threshold=drift_params.get(
                "target_drift_threshold",
                DEFAULT_TARGET_DRIFT_THRESHOLD,
            ),
        ),
        "concept_drift": calculate_concept_drift(
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            drift_threshold=drift_params.get(
                "concept_drift_threshold",
                DEFAULT_CONCEPT_DRIFT_THRESHOLD,
            ),
            primary_metric=drift_params.get(
                "primary_concept_metric",
                DEFAULT_PRIMARY_CONCEPT_METRIC,
            ),
        ),
    }


def save_drift_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    params = load_params()
    drift_output_path = params.get("metrics", {}).get(
        "drift_output_path",
        "metrics/drift_metrics.json",
    )
    report = build_drift_report(params)
    save_drift_report(report, BASE_DIR / drift_output_path)

    print("Drift metrics saved:")
    print(drift_output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

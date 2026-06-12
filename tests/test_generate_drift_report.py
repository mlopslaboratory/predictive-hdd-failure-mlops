from src.monitoring.generate_drift_report import build_report


def test_build_report_includes_drift_summary():
    metrics = {
        "pr_auc": 0.7,
        "roc_auc": 0.8,
        "f1": 0.5,
        "precision": 0.4,
        "recall": 0.6,
        "threshold": 0.5,
        "test_positive_rate": 0.1,
    }
    drift = {
        "window_counts": {"train": 10, "val": 4, "test": 3},
        "data_drift": {
            "drift_detected": True,
            "feature_count": 2,
            "drifted_feature_count": 1,
            "mean_psi": 0.2,
            "max_psi": 0.4,
            "drift_threshold": 0.2,
            "top_drifted_features": ["feature_a"],
            "per_feature": {"feature_a": {"psi": 0.4}},
        },
        "target_drift": {
            "drift_detected": False,
            "reference_positive_rate": 0.1,
            "current_positive_rate": 0.2,
            "absolute_positive_rate_change": 0.1,
            "psi": 0.05,
        },
        "concept_drift": {
            "drift_detected": False,
            "baseline_split": "val",
            "current_split": "test",
            "primary_metric": "pr_auc",
            "primary_metric_drop": 0.01,
            "drift_threshold": 0.05,
        },
    }

    report = build_report(metrics, drift)

    assert "# Drift Report" in report
    assert "| Data drift | обнаружен |" in report
    assert "`feature_a`: PSI 0.4000" in report
    assert "| PR-AUC | 0.7000 |" in report

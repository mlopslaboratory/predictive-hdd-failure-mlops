import pandas as pd
import pytest

from src.monitoring.calculate_drift import (
    calculate_concept_drift,
    calculate_feature_drift,
    calculate_numeric_psi,
    calculate_target_drift,
)


def test_calculate_numeric_psi_is_zero_for_same_distribution():
    reference = pd.Series([0.0, 1.0, 2.0, 3.0])
    current = pd.Series([0.0, 1.0, 2.0, 3.0])

    assert calculate_numeric_psi(reference, current, bins=2) == 0.0


def test_calculate_feature_drift_flags_shifted_feature():
    reference = pd.DataFrame(
        {
            "stable_feature": [0.0, 1.0, 2.0, 3.0],
            "shifted_feature": [0.0, 1.0, 2.0, 3.0],
        }
    )
    current = pd.DataFrame(
        {
            "stable_feature": [0.0, 1.0, 2.0, 3.0],
            "shifted_feature": [100.0, 101.0, 102.0, 103.0],
        }
    )

    result = calculate_feature_drift(
        reference_features=reference,
        current_features=current,
        bins=2,
        drift_threshold=0.2,
    )

    assert result["drift_detected"] is True
    assert result["per_feature"]["stable_feature"]["drift_detected"] is False
    assert result["per_feature"]["shifted_feature"]["drift_detected"] is True
    assert result["top_drifted_features"][0] == "shifted_feature"


def test_calculate_target_drift_uses_positive_rate_change():
    reference = pd.Series([0, 0, 1, 1])
    current = pd.Series([1, 1, 1, 1])

    result = calculate_target_drift(
        reference_target=reference,
        current_target=current,
        drift_threshold=0.1,
    )

    assert result["reference_positive_rate"] == 0.5
    assert result["current_positive_rate"] == 1.0
    assert result["absolute_positive_rate_change"] == 0.5
    assert result["drift_detected"] is True
    assert result["psi"] > 0.0


def test_calculate_concept_drift_detects_primary_metric_drop():
    baseline_metrics = {
        "pr_auc": 0.8,
        "roc_auc": 0.9,
        "f1": 0.7,
        "precision": 0.75,
        "recall": 0.65,
    }
    current_metrics = {
        "pr_auc": 0.6,
        "roc_auc": 0.85,
        "f1": 0.68,
        "precision": 0.7,
        "recall": 0.64,
    }

    result = calculate_concept_drift(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        drift_threshold=0.05,
        primary_metric="pr_auc",
    )

    assert result["drift_detected"] is True
    assert result["primary_metric_drop"] == pytest.approx(0.2)
    assert result["metric_deltas"]["pr_auc"]["baseline"] == 0.8
    assert result["metric_deltas"]["pr_auc"]["current"] == 0.6

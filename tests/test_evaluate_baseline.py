import json

import numpy as np
import pandas as pd

from src.models.evaluate_baseline import (
    evaluate_model,
    load_preprocessing,
    prepare_split,
)


class DummyModel:
    def predict_proba(self, X):
        return np.array([[1.0 - value, value] for value in X["feature_a_t"]])


def test_prepare_split_applies_saved_preprocessing():
    split_df = pd.DataFrame(
        {
            "serial_number": ["a", "a", "a"],
            "feature_a": [2.0, 4.0, 6.0],
            "target": [0, 1, 0],
        }
    )
    preprocessing = {
        "base_feature_cols": ["feature_a"],
        "window_feature_cols": ["feature_a_t-1", "feature_a_t"],
        "target_col": "target",
        "id_col": "serial_number",
        "window_size": 2,
        "normalization": {
            "mean": {"feature_a": 2.0},
            "std": {"feature_a": 2.0},
        },
    }

    X, y = prepare_split(split_df, preprocessing)

    assert X.values.tolist() == [[0.0, 1.0], [1.0, 2.0]]
    assert y.tolist() == [1, 0]


def test_evaluate_model_returns_threshold_metrics():
    X_test = pd.DataFrame({"feature_a_t": [0.1, 0.8, 0.7, 0.2]})
    y_test = pd.Series([0, 1, 1, 0])

    metrics = evaluate_model(DummyModel(), X_test, y_test, threshold=0.5)

    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["threshold"] == 0.5
    assert metrics["test_windows"] == 4
    assert metrics["test_positive_rate"] == 0.5
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_load_preprocessing_reads_json(tmp_path):
    preprocessing_path = tmp_path / "preprocessing.json"
    preprocessing_path.write_text(
        json.dumps({"window_size": 7}),
        encoding="utf-8",
    )

    assert load_preprocessing(preprocessing_path) == {"window_size": 7}

import json
import pickle

import numpy as np

from src.api import main as api_main


class DummyModel:
    def __init__(self):
        self.columns = []
        self.values = []

    def predict_proba(self, X):
        self.columns = X.columns.tolist()
        self.values = X.iloc[0].tolist()
        return np.array([[0.25, 0.75]])

    def predict(self, X):
        return np.array([1])


def test_load_artifacts_reads_model_features_and_preprocessing(tmp_path):
    model_path = tmp_path / "rf_model.pkl"
    features_path = tmp_path / "features.json"
    preprocessing_path = tmp_path / "preprocessing.json"

    with model_path.open("wb") as file:
        pickle.dump(DummyModel(), file)
    features_path.write_text(
        json.dumps(["feature_a_t-1", "feature_a_t"]),
        encoding="utf-8",
    )
    preprocessing_path.write_text(
        json.dumps({"window_size": 2, "target_col": "target"}),
        encoding="utf-8",
    )

    model, feature_cols, preprocessing = api_main.load_artifacts(
        model_path=model_path,
        features_path=features_path,
        preprocessing_path=preprocessing_path,
    )

    assert isinstance(model, DummyModel)
    assert feature_cols == ["feature_a_t-1", "feature_a_t"]
    assert preprocessing == {"window_size": 2, "target_col": "target"}


def test_predict_uses_saved_feature_order(monkeypatch):
    model = DummyModel()
    monkeypatch.setattr(api_main, "model", model)
    monkeypatch.setattr(api_main, "feature_cols", ["feature_b_t", "feature_a_t"])
    api_main.prediction_history.clear()

    response = api_main.predict(
        api_main.PredictionRequest(
            features={
                "feature_a_t": 1.0,
                "feature_b_t": 2.0,
            }
        )
    )

    assert response.failure_probability == 0.75
    assert response.prediction == 1
    assert response.anomaly is True
    assert model.columns == ["feature_b_t", "feature_a_t"]
    assert model.values == [2.0, 1.0]
    assert len(api_main.recent_predictions()) == 1
    assert api_main.recent_predictions()[0].feature_count == 2


def test_model_info_includes_preprocessing_metadata(monkeypatch):
    monkeypatch.setattr(api_main, "model", DummyModel())
    monkeypatch.setattr(api_main, "feature_cols", ["feature_a_t"])
    monkeypatch.setattr(
        api_main,
        "preprocessing_metadata",
        {
            "base_feature_cols": ["feature_a"],
            "window_size": 1,
            "target_col": "target",
        },
    )

    response = api_main.model_info()

    assert response["model_class"] == "DummyModel"
    assert response["features_count"] == 1
    assert response["preprocessing"] == {
        "base_features_count": 1,
        "window_size": 1,
        "target_col": "target",
    }


def test_openapi_schema_exposes_inference_endpoint():
    schema = api_main.app.openapi()

    assert "/predict" in schema["paths"]
    assert schema["paths"]["/predict"]["post"]["tags"] == ["inference"]
    assert "/predictions" in schema["paths"]
    assert "/drift-status" in schema["paths"]
    assert "/retrain" not in schema["paths"]


def test_build_drift_status_summarizes_flags(tmp_path):
    drift_path = tmp_path / "drift_metrics.json"
    drift_path.write_text(
        json.dumps(
            {
                "reference_split": "train",
                "current_split": "test",
                "window_counts": {"train": 10, "test": 4},
                "data_drift": {
                    "drift_detected": True,
                    "drifted_feature_count": 2,
                    "max_psi": 0.42,
                    "top_drifted_features": ["feature_a", "feature_b"],
                },
                "target_drift": {
                    "drift_detected": False,
                    "current_positive_rate": 0.25,
                },
                "concept_drift": {
                    "drift_detected": False,
                    "primary_metric": "pr_auc",
                    "primary_metric_drop": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )

    status = api_main.build_drift_status(drift_path)

    assert status.available is True
    assert status.data_drift is True
    assert status.target_drift is False
    assert status.concept_drift is False
    assert status.any_drift is True
    assert status.summary["drifted_feature_count"] == 2


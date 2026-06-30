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


def use_temp_prediction_db(monkeypatch, tmp_path):
    db_path = tmp_path / "predictions.db"
    monkeypatch.setattr(api_main, "PREDICTIONS_DB_PATH", db_path)
    monkeypatch.setattr(api_main, "MODEL_VERSION", "test-version")
    api_main.init_predictions_db()
    return db_path


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


def test_predict_uses_saved_feature_order(monkeypatch, tmp_path):
    use_temp_prediction_db(monkeypatch, tmp_path)
    model = DummyModel()
    monkeypatch.setattr(api_main, "model", model)
    monkeypatch.setattr(api_main, "feature_cols", ["feature_b_t", "feature_a_t"])

    response = api_main.predict(
        api_main.PredictionRequest(
            disk_id="disk-test-001",
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
    assert api_main.recent_predictions()[0].disk_id == "disk-test-001"
    assert api_main.recent_predictions()[0].feature_count == 2


def test_save_prediction_record_persists_to_db(monkeypatch, tmp_path):
    use_temp_prediction_db(monkeypatch, tmp_path)

    saved = api_main.save_prediction_record(
        request=api_main.PredictionRequest(
            serial_number="serial-123",
            features={"feature_a_t": 1.0},
        ),
        failure_probability=0.91,
        prediction=1,
        anomaly=True,
    )
    records = api_main.fetch_prediction_records()

    assert saved.id == 1
    assert saved.disk_id == "serial-123"
    assert len(records) == 1
    assert records[0].failure_probability == 0.91
    assert records[0].model_version == "test-version"


def test_save_prediction_record_generates_readable_disk_id(monkeypatch, tmp_path):
    use_temp_prediction_db(monkeypatch, tmp_path)

    first = api_main.save_prediction_record(
        request=api_main.PredictionRequest(features={"feature_a_t": 1.0}),
        failure_probability=0.11,
        prediction=0,
        anomaly=False,
    )
    second = api_main.save_prediction_record(
        request=api_main.PredictionRequest(features={"feature_a_t": 2.0}),
        failure_probability=0.12,
        prediction=0,
        anomaly=False,
    )

    assert first.disk_id == "Disk-000001"
    assert second.disk_id == "Disk-000002"
    assert [record.disk_id for record in api_main.recent_predictions(limit=10)] == [
        "Disk-000002",
        "Disk-000001",
    ]


def test_recent_predictions_returns_saved_records(monkeypatch, tmp_path):
    use_temp_prediction_db(monkeypatch, tmp_path)
    api_main.save_prediction_record(
        request=api_main.PredictionRequest(
            disk_id="disk-a",
            features={"feature_a_t": 1.0},
        ),
        failure_probability=0.2,
        prediction=0,
        anomaly=False,
    )
    api_main.save_prediction_record(
        request=api_main.PredictionRequest(
            disk_id="disk-b",
            features={"feature_a_t": 2.0},
        ),
        failure_probability=0.8,
        prediction=1,
        anomaly=True,
    )

    records = api_main.recent_predictions(limit=10)

    assert [record.disk_id for record in records] == ["disk-b", "disk-a"]
    assert len(api_main.recent_predictions(limit=10, only_anomalies=True)) == 1


def test_prediction_stats_counts_records_and_latest(monkeypatch, tmp_path):
    use_temp_prediction_db(monkeypatch, tmp_path)
    api_main.save_prediction_record(
        request=api_main.PredictionRequest(
            disk_id="disk-a",
            features={"feature_a_t": 1.0},
        ),
        failure_probability=0.2,
        prediction=0,
        anomaly=False,
    )
    api_main.save_prediction_record(
        request=api_main.PredictionRequest(
            disk_id="disk-b",
            features={"feature_a_t": 2.0},
        ),
        failure_probability=0.86,
        prediction=1,
        anomaly=True,
    )

    stats = api_main.prediction_stats()

    assert stats.total_predictions == 2
    assert stats.risky_predictions == 1
    assert stats.latest_disk_id == "disk-b"
    assert stats.latest_failure_probability == 0.86


def test_risky_disks_filters_only_risky_predictions(monkeypatch, tmp_path):
    use_temp_prediction_db(monkeypatch, tmp_path)
    api_main.save_prediction_record(
        request=api_main.PredictionRequest(
            disk_id="normal-disk",
            features={"feature_a_t": 1.0},
        ),
        failure_probability=0.1,
        prediction=0,
        anomaly=False,
    )
    api_main.save_prediction_record(
        request=api_main.PredictionRequest(
            disk_id="prediction-risk",
            features={"feature_a_t": 2.0},
        ),
        failure_probability=0.7,
        prediction=1,
        anomaly=True,
    )
    api_main.save_prediction_record(
        request=api_main.PredictionRequest(
            disk_id="anomaly-risk",
            features={"feature_a_t": 3.0},
        ),
        failure_probability=0.9,
        prediction=0,
        anomaly=True,
    )

    records = api_main.risky_disks(limit=10)

    assert [record.disk_id for record in records] == [
        "anomaly-risk",
        "prediction-risk",
    ]


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
    assert "/prediction-stats" in schema["paths"]
    assert "/risky-disks" in schema["paths"]
    assert "/drift-status" in schema["paths"]
    assert "/retrain" not in schema["paths"]


def test_openapi_prediction_request_has_executable_example():
    schema = api_main.app.openapi()
    example = schema["components"]["schemas"]["PredictionRequest"]["example"]

    assert "disk_id" not in example
    assert len(example["features"]) == len(api_main.build_prediction_request_example()["features"])
    assert "additionalProp1" not in example["features"]


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


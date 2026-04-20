from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def train_model(X_train, y_train, model_params: dict):
    print("Обучение Random Forest началось...")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
    }


def save_artifacts(model, feature_cols: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    features_path = output_dir / "features.json"

    joblib.dump(model, model_path)
    features_path.write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_training(X_train, y_train, X_test, y_test, config: dict, output_dir: Path):
    model_name = config["model"]["name"]
    model_params = config["model"]["params"]
    feature_cols = X_train.columns.tolist()

    with mlflow.start_run():
        model = train_model(X_train, y_train, model_params=model_params)
        metrics = evaluate_model(model, X_test, y_test)
        save_artifacts(model, feature_cols, output_dir)

        print("Логирую параметры, метрики и модель в MLflow...")
        mlflow.log_param("model", model_name)
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    return model, metrics

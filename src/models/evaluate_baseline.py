from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.train_baseline import (
    BASE_DIR,
    configure_mlflow,
    create_windows,
    load_params,
    normalize,
)


def load_model(model_path: Path):
    with model_path.open("rb") as file:
        return pickle.load(file)


def load_preprocessing(preprocessing_path: Path) -> dict:
    return json.loads(preprocessing_path.read_text(encoding="utf-8"))


def load_mlflow_run_info(run_info_path: Path) -> dict:
    return json.loads(run_info_path.read_text(encoding="utf-8"))


def prepare_split(
    split_df: pd.DataFrame,
    preprocessing: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = preprocessing["base_feature_cols"]
    target_col = preprocessing["target_col"]
    id_col = preprocessing["id_col"]
    window_size = preprocessing["window_size"]
    mean = pd.Series(preprocessing["normalization"]["mean"])
    std = pd.Series(preprocessing["normalization"]["std"])

    normalized_df = normalize(split_df, mean, std, feature_cols)
    X, y = create_windows(
        df=normalized_df,
        feature_cols=feature_cols,
        target_col=target_col,
        id_col=id_col,
        window_size=window_size,
    )
    return X[preprocessing["window_feature_cols"]], y


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred)),
        "threshold": float(threshold),
        "test_windows": int(len(y_test)),
        "test_positive_rate": float(y_test.mean()),
    }


def save_metrics(metrics: dict, metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def log_metrics_and_register_model(metrics: dict, params: dict) -> None:
    mlflow_params = params["mlflow"]
    run_info = load_mlflow_run_info(BASE_DIR / mlflow_params["run_info_path"])

    configure_mlflow(params)

    with mlflow.start_run(run_id=run_info["run_id"]):
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})

    mlflow.register_model(
        model_uri=run_info["model_uri"],
        name=mlflow_params["registered_model_name"],
    )


def main() -> None:
    params = load_params()
    model_params = params["model"]
    data_params = params["data"]
    metrics_params = params["metrics"]

    model = load_model(BASE_DIR / model_params["artifact_path"])
    preprocessing = load_preprocessing(BASE_DIR / model_params["preprocessing_path"])
    test_df = pd.read_csv(BASE_DIR / data_params["test_path"])
    X_test, y_test = prepare_split(test_df, preprocessing)

    threshold = metrics_params.get("prediction_threshold", 0.5)
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    save_metrics(metrics, BASE_DIR / metrics_params["output_path"])
    log_metrics_and_register_model(metrics, params)

    print("Evaluation metrics saved:")
    print(metrics_params["output_path"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

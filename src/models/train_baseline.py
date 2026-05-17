from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = BASE_DIR / "params.yaml"


def load_params(params_path: Path = PARAMS_PATH) -> dict:
    with params_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize(
    df: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    normalized = df.copy()
    safe_std = std.replace(0, 1)
    normalized[feature_cols] = (df[feature_cols] - mean) / safe_std
    return normalized


def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    id_col: str,
    window_size: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X, y = [], []

    for _, group in df.groupby(id_col):
        series = group[feature_cols].values
        labels = group[target_col].values
        if len(series) < window_size:
            continue

        for i in range(len(series) - window_size + 1):
            X.append(series[i : i + window_size].reshape(-1))
            y.append(labels[i + window_size - 1])

    window_feature_cols = build_window_feature_names(feature_cols, window_size)
    return pd.DataFrame(X, columns=window_feature_cols), pd.Series(y, name=target_col)


def build_window_feature_names(feature_cols: list[str], window_size: int) -> list[str]:
    return [
        f"{feature}_t-{window_size - step - 1}"
        if step < window_size - 1
        else f"{feature}_t"
        for step in range(window_size)
        for feature in feature_cols
    ]


def get_feature_columns(params: dict) -> list[str]:
    smart_columns = params["preprocessing"]["smart_columns"]

    if params["features"]["use_delta_features"]:
        return [f"{column}_delta" for column in smart_columns]

    return smart_columns


def train_baseline(params: dict) -> tuple[RandomForestClassifier, dict]:
    data_params = params["data"]
    preprocessing_params = params["preprocessing"]
    feature_params = params["features"]
    model_params = params["model"]

    train_path = BASE_DIR / data_params["train_path"]
    train_df = pd.read_csv(train_path)

    feature_cols = get_feature_columns(params)
    target_col = preprocessing_params["target_column"]
    id_col = preprocessing_params["id_column"]
    window_size = feature_params["window_size"]

    train_mean = train_df[feature_cols].mean()
    train_std = train_df[feature_cols].std().replace(0, 1)
    normalized_train_df = normalize(train_df, train_mean, train_std, feature_cols)
    X_train, y_train = create_windows(
        df=normalized_train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        id_col=id_col,
        window_size=window_size,
    )

    model = RandomForestClassifier(**model_params["params"])
    model.fit(X_train, y_train)

    preprocessing = {
        "base_feature_cols": feature_cols,
        "window_feature_cols": X_train.columns.tolist(),
        "target_col": target_col,
        "id_col": id_col,
        "window_size": int(window_size),
        "normalization": {
            "mean": train_mean.to_dict(),
            "std": train_std.to_dict(),
        },
    }

    return model, preprocessing


def configure_mlflow(params: dict) -> None:
    import mlflow

    mlflow_params = params["mlflow"]
    tracking_uri = mlflow_params["tracking_uri"]
    tracking_path = BASE_DIR / tracking_uri

    if tracking_path.suffix == ".db":
        mlflow.set_tracking_uri(f"sqlite:///{tracking_path.as_posix()}")
    else:
        mlflow.set_tracking_uri(f"file:///{tracking_path.as_posix()}")

    mlflow.set_experiment(mlflow_params["experiment_name"])


def save_artifacts(
    model: RandomForestClassifier,
    preprocessing: dict,
    params: dict,
) -> None:
    model_params = params["model"]
    model_path = BASE_DIR / model_params["artifact_path"]
    features_path = BASE_DIR / model_params["features_path"]
    preprocessing_path = BASE_DIR / model_params["preprocessing_path"]

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as file:
        pickle.dump(model, file)

    features_path.write_text(
        json.dumps(preprocessing["window_feature_cols"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    preprocessing_path.write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_mlflow_run_info(run_id: str, params: dict) -> None:
    mlflow_params = params["mlflow"]
    run_info_path = BASE_DIR / mlflow_params["run_info_path"]
    model_artifact_name = mlflow_params["model_artifact_name"]

    run_info_path.parent.mkdir(parents=True, exist_ok=True)
    run_info = {
        "run_id": run_id,
        "model_uri": f"runs:/{run_id}/{model_artifact_name}",
        "experiment_name": mlflow_params["experiment_name"],
        "registered_model_name": mlflow_params["registered_model_name"],
    }
    run_info_path.write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def log_training_run(
    model: RandomForestClassifier,
    preprocessing: dict,
    params: dict,
) -> str:
    import mlflow
    import mlflow.sklearn

    mlflow_params = params["mlflow"]
    model_params = params["model"]

    configure_mlflow(params)

    with mlflow.start_run(run_name="baseline_random_forest") as run:
        mlflow.log_param("model_name", model_params["name"])
        mlflow.log_params(model_params["params"])
        mlflow.log_param("window_size", preprocessing["window_size"])
        mlflow.log_param("feature_count", len(preprocessing["window_feature_cols"]))
        mlflow.log_param("base_feature_count", len(preprocessing["base_feature_cols"]))

        mlflow.sklearn.log_model(
            sk_model=model,
            name=mlflow_params["model_artifact_name"],
        )
        mlflow.log_artifact(BASE_DIR / model_params["features_path"])
        mlflow.log_artifact(BASE_DIR / model_params["preprocessing_path"])

        return run.info.run_id


def main() -> None:
    params = load_params()
    model, preprocessing = train_baseline(params)
    save_artifacts(model, preprocessing, params)
    run_id = log_training_run(model, preprocessing, params)
    save_mlflow_run_info(run_id, params)

    print("Baseline model artifacts saved:")
    print(params["model"]["artifact_path"])
    print(params["model"]["features_path"])
    print(params["model"]["preprocessing_path"])
    print(params["mlflow"]["run_info_path"])


if __name__ == "__main__":
    main()

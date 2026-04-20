from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.models.train_model import run_training


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_training_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = BASE_DIR / config["training"]["train_path"]
    test_path = BASE_DIR / config["training"]["test_path"]

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def prepare_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    target_column = config["training"]["target_column"]
    drop_columns = config["training"].get("drop_columns", [])

    feature_drop = [target_column, *drop_columns]

    X_train = train_df.drop(columns=feature_drop, errors="ignore")
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=feature_drop, errors="ignore")
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test


def configure_mlflow(config: dict) -> None:
    tracking_uri = config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(f"file:///{BASE_DIR / tracking_uri}")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def main() -> None:
    config = load_config()
    configure_mlflow(config)

    train_df, test_df = load_training_data(config)
    X_train, y_train, X_test, y_test = prepare_datasets(train_df, test_df, config)

    run_training(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config,
    )


if __name__ == "__main__":
    main()

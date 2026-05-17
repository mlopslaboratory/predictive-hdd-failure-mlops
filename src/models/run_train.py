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
    tracking_path = BASE_DIR / tracking_uri

    if tracking_path.suffix == ".db":
        mlflow.set_tracking_uri(f"sqlite:///{tracking_path.as_posix()}")
    else:
        mlflow.set_tracking_uri(f"file:///{tracking_path.as_posix()}")

    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def main() -> None:
    print("Запуск обучения Random Forest...")
    config = load_config()
    configure_mlflow(config)
    output_dir = BASE_DIR / "artifacts"

    print("Читаю train/test parquet из конфига...")
    train_df, test_df = load_training_data(config)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    X_train, y_train, X_test, y_test = prepare_datasets(train_df, test_df, config)
    print("Данные подготовлены. Начинаю обучение модели...")

    _, metrics = run_training(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config,
        output_dir=output_dir,
    )

    print("Обучение завершено.")
    print(f"Артефакты сохранены в: {output_dir}")
    print("Метрики на test:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"F1:      {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

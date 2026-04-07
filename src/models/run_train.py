from src.data.load_data import load_data
from src.features.build_features import build_features
from src.split.split import prepare_train_test, get_feature_cols
from src.models.train_model import run_training
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/raw/data.csv"
MODEL_NAME = "HGST HUH721212ALN604"


def main():
    # loader
    df = load_data(DATA_PATH, MODEL_NAME)

    # features builder
    df = build_features(df)

    # features
    feature_cols = get_feature_cols(df)

    # split
    X_train, X_test, y_train, y_test = prepare_train_test(df, feature_cols)

    # trait, MLflow
    run_training(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()


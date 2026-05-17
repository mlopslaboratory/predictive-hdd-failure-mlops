import pandas as pd

from src.models.train_baseline import (
    build_window_feature_names,
    create_windows,
    normalize,
)


def test_build_window_feature_names_uses_lag_suffixes():
    result = build_window_feature_names(
        feature_cols=["smart_1_raw_delta", "smart_5_raw_delta"],
        window_size=3,
    )

    assert result == [
        "smart_1_raw_delta_t-2",
        "smart_5_raw_delta_t-2",
        "smart_1_raw_delta_t-1",
        "smart_5_raw_delta_t-1",
        "smart_1_raw_delta_t",
        "smart_5_raw_delta_t",
    ]


def test_normalize_replaces_zero_std_with_one():
    df = pd.DataFrame(
        {
            "feature_a": [2.0, 4.0],
            "feature_b": [10.0, 10.0],
        }
    )
    mean = pd.Series({"feature_a": 2.0, "feature_b": 10.0})
    std = pd.Series({"feature_a": 2.0, "feature_b": 0.0})

    result = normalize(
        df=df,
        mean=mean,
        std=std,
        feature_cols=["feature_a", "feature_b"],
    )

    assert result["feature_a"].tolist() == [0.0, 1.0]
    assert result["feature_b"].tolist() == [0.0, 0.0]


def test_create_windows_flattens_each_serial_sequence():
    df = pd.DataFrame(
        {
            "serial_number": ["a", "a", "a", "b", "b", "b"],
            "feature_a": [1, 2, 3, 10, 20, 30],
            "feature_b": [4, 5, 6, 40, 50, 60],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    X, y = create_windows(
        df=df,
        feature_cols=["feature_a", "feature_b"],
        target_col="target",
        id_col="serial_number",
        window_size=2,
    )

    assert X.columns.tolist() == [
        "feature_a_t-1",
        "feature_b_t-1",
        "feature_a_t",
        "feature_b_t",
    ]
    assert X.values.tolist() == [
        [1, 4, 2, 5],
        [2, 5, 3, 6],
        [10, 40, 20, 50],
        [20, 50, 30, 60],
    ]
    assert y.tolist() == [1, 0, 0, 1]

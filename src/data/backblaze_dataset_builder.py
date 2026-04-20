from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_COLUMNS = ["date", "serial_number", "model", "failure"]
SMART_COLUMNS = [
    "smart_1_raw",
    "smart_2_raw",
    "smart_3_raw",
    "smart_5_raw",
    "smart_12_raw",
    "smart_22_raw",
    "smart_192_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_199_raw",
]
DROP_COLUMNS = [
    "serial_number",
    "model",
    "datacenter",
    "cluster_id",
    "vault_id",
    "pod_id",
]


def load_data(
    data_dir: str | Path,
    model_name: str,
    smart_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Loads only required columns from daily Backblaze CSV files."""

    smart_columns = smart_columns or SMART_COLUMNS
    usecols = BASE_COLUMNS + smart_columns
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        daily_df = pd.read_csv(csv_path, usecols=usecols)
        daily_df = daily_df[daily_df["model"] == model_name]
        if daily_df.empty:
            continue
        frames.append(daily_df)

    if not frames:
        raise ValueError(f"No rows found for model {model_name!r} in {data_dir!s}")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    return df


def build_target(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Builds a horizon target without using failure-day rows."""

    failure_dates = (
        df.loc[df["failure"] == 1, ["serial_number", "date"]]
        .groupby("serial_number", as_index=False)["date"]
        .min()
        .rename(columns={"date": "failure_date"})
    )

    prepared = df.merge(failure_dates, on="serial_number", how="left")
    prepared["days_to_failure"] = (
        prepared["failure_date"] - prepared["date"]
    ).dt.days
    prepared["label"] = (
        prepared["days_to_failure"].between(0, horizon_days, inclusive="both")
    ).astype(int)

    prepared = prepared[prepared["failure"] == 0].copy()

    max_date = prepared["date"].max()
    censoring_cutoff = max_date - pd.Timedelta(days=horizon_days)
    prepared = prepared[
        (prepared["failure_date"].notna()) | (prepared["date"] <= censoring_cutoff)
    ].copy()

    return prepared.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Builds time-series features from the past only."""

    featured = df.sort_values(["serial_number", "date"]).copy()
    feature_cols = list(SMART_COLUMNS)
    rolling_windows = [3, 7, 14]

    for window in rolling_windows:
        for column in SMART_COLUMNS:
            mean_name = f"{column}_roll_mean_{window}"
            std_name = f"{column}_roll_std_{window}"
            featured[mean_name] = featured.groupby("serial_number")[column].transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            featured[std_name] = featured.groupby("serial_number")[column].transform(
                lambda values: values.rolling(window, min_periods=1).std()
            )
            feature_cols.extend([mean_name, std_name])

    for column in SMART_COLUMNS:
        for shift_days in [1, 3]:
            delta_name = f"{column}_delta_{shift_days}d"
            featured[delta_name] = featured.groupby("serial_number")[column].transform(
                lambda values: values.diff(shift_days)
            )
            feature_cols.append(delta_name)

    for column in ["smart_5_raw", "smart_197_raw", "smart_198_raw"]:
        flag_name = f"{column}_nonzero"
        featured[flag_name] = (featured[column] > 0).astype(int)
        feature_cols.append(flag_name)

    featured["disk_age_days"] = featured.groupby("serial_number").cumcount() + 1
    feature_cols.append("disk_age_days")

    featured[feature_cols] = featured[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    featured[feature_cols] = featured[feature_cols].fillna(0.0)

    return featured, feature_cols


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = 0.2,
    negative_ratio: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by serial_number and downsamples only the training negatives."""

    serial_targets = (
        df.groupby("serial_number")["label"].max().rename("has_failure").reset_index()
    )

    train_serials, test_serials = train_test_split(
        serial_targets["serial_number"],
        test_size=test_size,
        random_state=random_state,
        stratify=serial_targets["has_failure"],
    )

    train_df = df[df["serial_number"].isin(train_serials)].copy()
    test_df = df[df["serial_number"].isin(test_serials)].copy()

    positive_train = train_df[train_df["label"] == 1]
    negative_train = train_df[train_df["label"] == 0]
    max_negative = len(positive_train) * negative_ratio

    if len(positive_train) > 0 and len(negative_train) > max_negative:
        negative_train = negative_train.sample(
            n=max_negative, random_state=random_state
        )

    train_df = (
        pd.concat([positive_train, negative_train], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    output_columns = ["date", "serial_number", "label"] + feature_cols
    train_df = train_df[output_columns].copy()
    test_df = test_df[output_columns].copy()

    removable_columns = [
        col for col in DROP_COLUMNS if col in train_df.columns and col != "serial_number"
    ]
    train_df = train_df.drop(columns=removable_columns)
    removable_columns = [
        col for col in DROP_COLUMNS if col in test_df.columns and col != "serial_number"
    ]
    test_df = test_df.drop(columns=removable_columns)

    numeric_train_cols = [col for col in train_df.columns if col not in {"date", "serial_number"}]
    numeric_test_cols = [col for col in test_df.columns if col not in {"date", "serial_number"}]
    train_df[numeric_train_cols] = train_df[numeric_train_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    test_df[numeric_test_cols] = test_df[numeric_test_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)

    return train_df, test_df


def build_dataset(
    data_dir: str | Path,
    model_name: str,
    horizon_days: int = 7,
    test_size: float = 0.2,
    negative_ratio: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs the full pipeline and returns full, train, and test datasets."""

    loaded_df = load_data(data_dir=data_dir, model_name=model_name)
    target_df = build_target(loaded_df, horizon_days=horizon_days)
    featured_df, feature_cols = build_features(target_df)
    train_df, test_df = split_data(
        featured_df,
        feature_cols=feature_cols,
        test_size=test_size,
        negative_ratio=negative_ratio,
        random_state=random_state,
    )

    full_df = featured_df[["date", "serial_number", "label"] + feature_cols].copy()
    return full_df, train_df, test_df


def summarize_split(full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    """Returns a compact summary for notebook output."""

    full_serials = set(full_df["serial_number"])
    train_serials = set(train_df["serial_number"])
    test_serials = set(test_df["serial_number"])

    return {
        "full_rows": int(len(full_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "full_positive_rate": float(full_df["label"].mean()),
        "train_positive_rate": float(train_df["label"].mean()),
        "test_positive_rate": float(test_df["label"].mean()),
        "full_serials": int(len(full_serials)),
        "train_serials": int(len(train_serials)),
        "test_serials": int(len(test_serials)),
        "serial_overlap": int(len(train_serials & test_serials)),
    }

"""Legacy feature pipeline from the first training workflow."""

import pandas as pd


SMART_COLS = [
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


def build_target(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """
    Create label: fault  in N days
    """

    failure_dates = df[df["failure"] == 1][["serial_number", "date"]].rename(
        columns={"date": "failure_date"}
    )

    df = df.merge(failure_dates, on="serial_number", how="left")

    df["days_to_failure"] = (df["failure_date"] - df["date"]).dt.days

    df["label"] = (
        (df["days_to_failure"] >= 0) & (df["days_to_failure"] <= horizon_days)
    ).astype(int)

    # Delete rows with faults
    df = df[df["failure"] == 0]

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    windows = [3, 7, 14]

    for window in windows:
        for col in SMART_COLS:
            df[f"{col}_roll_mean_{window}"] = df.groupby("serial_number")[
                col
            ].transform(lambda x: x.rolling(window, min_periods=1).mean())

            df[f"{col}_roll_std_{window}"] = (
                df.groupby("serial_number")[col]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
                .fillna(0)
            )

    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in SMART_COLS:
        for shift in [1, 3]:
            df[f"{col}_delta_{shift}d"] = (
                df.groupby("serial_number")[col].transform(lambda x: x.diff(shift)).fillna(0)
            )

    return df


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["smart_5_raw", "smart_197_raw", "smart_198_raw"]:
        df[f"{col}_nonzero"] = (df[col] > 0).astype(int)

    return df


def add_disk_age(df: pd.DataFrame) -> pd.DataFrame:
    df["disk_age_days"] = df.groupby("serial_number").cumcount() + 1
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall features pipeline
    """

    df = df.sort_values(["serial_number", "date"])

    df = build_target(df)
    df = add_rolling_features(df)
    df = add_delta_features(df)
    df = add_flags(df)
    df = add_disk_age(df)

    # fill na
    df = df.fillna(0)

    return df

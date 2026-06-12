from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = BASE_DIR / "params.yaml"


def load_params(params_path: Path = PARAMS_PATH) -> dict:
    with params_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def select_balanced_serials(
    df: pd.DataFrame,
    id_column: str,
    failure_column: str,
    keep_healthy_fraction: float,
    seed: int,
) -> pd.DataFrame:
    failed_serials = df[df[failure_column] == 1][id_column].unique()
    healthy_serials = df[~df[id_column].isin(failed_serials)][id_column].unique()

    rng = np.random.default_rng(seed)
    healthy_count = int(len(healthy_serials) * keep_healthy_fraction)
    selected_healthy = rng.choice(
        healthy_serials,
        size=healthy_count,
        replace=False,
    )

    return df[df[id_column].isin(failed_serials) | df[id_column].isin(selected_healthy)].copy()


def interpolate_smart_columns(
    df: pd.DataFrame,
    id_column: str,
    smart_columns: list[str],
) -> pd.DataFrame:
    prepared = df.copy()
    prepared[smart_columns] = prepared.groupby(id_column)[smart_columns].transform(
        lambda values: values.interpolate(limit_direction="both")
    )
    return prepared


def create_deltas(
    df: pd.DataFrame,
    id_column: str,
    smart_columns: list[str],
) -> pd.DataFrame:
    prepared = df.copy()
    for column in smart_columns:
        prepared[f"{column}_delta"] = (
            prepared.groupby(id_column)[column].diff().fillna(0)
        )
    return prepared


def mark_risk_zone(
    df: pd.DataFrame,
    id_column: str,
    failure_column: str,
    target_column: str,
    days_before_failure: int,
) -> pd.DataFrame:
    prepared = df.copy()
    prepared[target_column] = prepared[failure_column]

    for _, group in prepared.groupby(id_column):
        if group[failure_column].any():
            group_indexes = group.index.to_list()
            failure_positions = np.flatnonzero(group[failure_column].to_numpy() == 1)
            last_failure_position = int(failure_positions[-1])
            start_position = max(0, last_failure_position - days_before_failure)
            risk_indexes = group_indexes[start_position : last_failure_position + 1]
            prepared.loc[risk_indexes, target_column] = 1

    return prepared


def split_data_by_serial(
    df: pd.DataFrame,
    id_column: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    serials = rng.permutation(df[id_column].unique())

    train_end = int(len(serials) * train_ratio)
    val_end = train_end + int(len(serials) * val_ratio)

    train_serials = serials[:train_end]
    val_serials = serials[train_end:val_end]
    test_serials = serials[val_end:]

    train_df = df[df[id_column].isin(train_serials)].copy()
    val_df = df[df[id_column].isin(val_serials)].copy()
    test_df = df[df[id_column].isin(test_serials)].copy()
    return train_df, val_df, test_df


def prepare_data(params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = params["seed"]
    data_params = params["data"]
    preprocessing_params = params["preprocessing"]

    id_column = preprocessing_params["id_column"]
    failure_column = preprocessing_params["failure_column"]
    target_column = preprocessing_params["target_column"]
    date_column = preprocessing_params["date_column"]
    smart_columns = preprocessing_params["smart_columns"]

    raw_path = BASE_DIR / data_params["raw_path"]
    df = pd.read_csv(raw_path)
    df = df.sort_values([id_column, date_column]).reset_index(drop=True)

    prepared = select_balanced_serials(
        df=df,
        id_column=id_column,
        failure_column=failure_column,
        keep_healthy_fraction=preprocessing_params["keep_healthy_fraction"],
        seed=seed,
    )
    prepared = interpolate_smart_columns(
        df=prepared,
        id_column=id_column,
        smart_columns=smart_columns,
    )
    prepared = create_deltas(
        df=prepared,
        id_column=id_column,
        smart_columns=smart_columns,
    )
    prepared = mark_risk_zone(
        df=prepared,
        id_column=id_column,
        failure_column=failure_column,
        target_column=target_column,
        days_before_failure=preprocessing_params["days_before_failure"],
    )

    train_df, val_df, test_df = split_data_by_serial(
        df=prepared,
        id_column=id_column,
        train_ratio=preprocessing_params["train_ratio"],
        val_ratio=preprocessing_params["val_ratio"],
        seed=seed,
    )

    return prepared, train_df, val_df, test_df


def save_data(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
) -> None:
    data_params = params["data"]
    processed_dir = BASE_DIR / data_params["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(BASE_DIR / data_params["full_prepared_path"], index=False)
    train_df.to_csv(BASE_DIR / data_params["train_path"], index=False)
    val_df.to_csv(BASE_DIR / data_params["val_path"], index=False)
    test_df.to_csv(BASE_DIR / data_params["test_path"], index=False)


def main() -> None:
    params = load_params()
    full_df, train_df, val_df, test_df = prepare_data(params)
    save_data(full_df, train_df, val_df, test_df, params)

    print("Prepared datasets saved:")
    print(f"full:  {full_df.shape} -> {params['data']['full_prepared_path']}")
    print(f"train: {train_df.shape} -> {params['data']['train_path']}")
    print(f"val:   {val_df.shape} -> {params['data']['val_path']}")
    print(f"test:  {test_df.shape} -> {params['data']['test_path']}")


if __name__ == "__main__":
    main()

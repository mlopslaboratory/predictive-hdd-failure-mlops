"""Legacy loader from the first CSV-based training pipeline."""

import glob
from pathlib import Path

import pandas as pd


def load_data(data_path: str, model_name: str) -> pd.DataFrame:
    """
    Load CSV files, filter by Disk model
    """

    path = Path(data_path)

    if path.is_file():
        df = pd.read_csv(path)

    else:
        files = sorted(glob.glob(str(path / "*.csv")))
        dfs = []

        for f in files:
            tmp = pd.read_csv(f)
            tmp = tmp[tmp["model"] == model_name]
            dfs.append(tmp)

        df = pd.concat(dfs, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])

    return df

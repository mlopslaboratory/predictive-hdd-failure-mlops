import pandas as pd
import glob
from pathlib import Path


def load_data(data_path: str, model_name: str) -> pd.DataFrame:
    """
    Load CSV files, filter by Disk model
    """

    files = sorted(glob.glob(str(Path(data_path) / "*.csv")))

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        # filter by a model
        df = df[df["model"] == model_name]

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # make types
    df["date"] = pd.to_datetime(df["date"])

    return df
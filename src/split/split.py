import pandas as pd


def time_split(df: pd.DataFrame, test_size: float = 0.25):
    """
    Split by a times
    """

    split_date = df['date'].quantile(1 - test_size)

    train_mask = df['date'] <= split_date
    test_mask = df['date'] > split_date

    return train_mask, test_mask


def prepare_train_test(df: pd.DataFrame, feature_cols: list):
    """
    Return X_train, X_test, y_train, y_test
    """

    train_mask, test_mask = time_split(df)

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'label']

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, 'label']

    return X_train, X_test, y_train, y_test


def get_feature_cols(df: pd.DataFrame):
    exclude = ['date', 'serial_number', 'failure', 'label', 'failure_date', 'days_to_failure']
    return [col for col in df.columns if col not in exclude]


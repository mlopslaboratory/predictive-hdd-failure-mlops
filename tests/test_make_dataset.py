import pandas as pd

from src.data.make_dataset import (
    create_deltas,
    mark_risk_zone,
    split_data_by_serial,
)


def test_create_deltas_adds_grouped_diff_features():
    df = pd.DataFrame(
        {
            "serial_number": ["a", "a", "b", "b"],
            "smart_1_raw": [10, 13, 5, 8],
        }
    )

    result = create_deltas(
        df=df,
        id_column="serial_number",
        smart_columns=["smart_1_raw"],
    )

    assert result["smart_1_raw_delta"].tolist() == [0.0, 3.0, 0.0, 3.0]


def test_mark_risk_zone_marks_rows_before_failure():
    df = pd.DataFrame(
        {
            "serial_number": ["disk-1"] * 5,
            "failure": [0, 0, 0, 0, 1],
        },
        index=[10, 11, 12, 13, 14],
    )

    result = mark_risk_zone(
        df=df,
        id_column="serial_number",
        failure_column="failure",
        target_column="target",
        days_before_failure=2,
    )

    assert result["target"].tolist() == [0, 0, 1, 1, 1]


def test_mark_risk_zone_does_not_mark_other_serials_with_sparse_indexes():
    df = pd.DataFrame(
        {
            "serial_number": ["disk-1", "disk-2", "disk-1", "disk-2", "disk-1"],
            "failure": [0, 0, 0, 0, 1],
        },
        index=[10, 20, 30, 40, 50],
    )

    result = mark_risk_zone(
        df=df,
        id_column="serial_number",
        failure_column="failure",
        target_column="target",
        days_before_failure=1,
    )

    assert result.loc[[10, 20, 40], "target"].tolist() == [0, 0, 0]
    assert result.loc[[30, 50], "target"].tolist() == [1, 1]


def test_split_data_by_serial_has_no_serial_overlap():
    df = pd.DataFrame(
        {
            "serial_number": [f"disk-{idx}" for idx in range(20) for _ in range(2)],
            "value": list(range(40)),
        }
    )

    train_df, val_df, test_df = split_data_by_serial(
        df=df,
        id_column="serial_number",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )

    train_serials = set(train_df["serial_number"])
    val_serials = set(val_df["serial_number"])
    test_serials = set(test_df["serial_number"])

    assert train_serials.isdisjoint(val_serials)
    assert train_serials.isdisjoint(test_serials)
    assert val_serials.isdisjoint(test_serials)
    assert train_serials | val_serials | test_serials == set(df["serial_number"])

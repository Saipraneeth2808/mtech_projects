import pandas as pd
from health_disease_model_training_v2 import prepare_dataframe


def test_prepare_dataframe_creates_target_and_lowercase():
    X = pd.DataFrame({
        "Age": [50, 60],
        "Sex": [1, 0]
    })
    y = pd.Series([0, 2], name="Num")

    df = prepare_dataframe(X, y)

    assert "sex" in df.columns
    assert df["sex"].tolist() == [1, 0]
    assert all(col == col.lower() for col in df.columns)


def test_prepare_dataframe_handles_missing_values():
    X = pd.DataFrame({
        "age": [50, None],
        "sex": [1, 0]
    })
    y = pd.Series([0, 1], name="num")

    df = prepare_dataframe(X, y)

    assert df.isnull().sum().sum() == 0

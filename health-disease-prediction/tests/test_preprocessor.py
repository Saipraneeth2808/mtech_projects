import pandas as pd
from sklearn.compose import ColumnTransformer
from health_disease_model_training_v2 import build_preprocessor


def test_build_preprocessor_returns_column_transformer():
    numeric_cols = ["age", "chol"]
    categorical_cols = ["sex", "cp"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2

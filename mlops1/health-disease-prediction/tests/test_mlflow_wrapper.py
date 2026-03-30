import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from health_disease_model_training_v2 import train_and_log_with_mlflow


def test_train_and_log_with_mlflow(monkeypatch):
    # -----------------------------
    # Mock MLflow
    # -----------------------------
    monkeypatch.setattr("health_disease_model_training_v2.mlflow.set_experiment", MagicMock())
    monkeypatch.setattr("health_disease_model_training_v2.mlflow.start_run", MagicMock())
    monkeypatch.setattr("health_disease_model_training_v2.mlflow.log_metrics", MagicMock())
    monkeypatch.setattr("health_disease_model_training_v2.mlflow.log_params", MagicMock())
    monkeypatch.setattr("health_disease_model_training_v2.mlflow.log_artifact", MagicMock())
    monkeypatch.setattr("health_disease_model_training_v2.mlflow.sklearn.log_model", MagicMock())

    # -----------------------------
    # Mock GridSearch
    # -----------------------------
    mock_estimator = MagicMock()
    mock_gs = MagicMock()
    mock_gs.best_estimator_ = mock_estimator
    mock_gs.best_params_ = {"clf__C": 1.0}

    # -----------------------------
    # Mock cross_validate
    # -----------------------------
    fake_cv = {
        "test_accuracy": np.array([0.8, 0.85]),
        "test_precision": np.array([0.75, 0.8]),
        "test_recall": np.array([0.7, 0.78]),
        "test_f1": np.array([0.72, 0.79]),
        "test_roc_auc": np.array([0.82, 0.86]),
    }

    monkeypatch.setattr(
        "health_disease_model_training_v2.cross_validate",
        lambda *args, **kwargs: fake_cv
    )

    # -----------------------------
    # Dummy Data
    # -----------------------------
    X = pd.DataFrame({"age": [50, 60], "sex": [1, 0]})
    y = pd.Series([0, 1])

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc_ovr",
    }

    # -----------------------------
    # Execute
    # -----------------------------
    train_and_log_with_mlflow(
        experiment_name="test_exp",
        run_name="test_run",
        grid_search=mock_gs,
        X=X,
        y=y,
        scoring=scoring,
        artifact_path="testModel"
    )

    # -----------------------------
    # Assertions
    # -----------------------------
    assert mock_gs.fit.called

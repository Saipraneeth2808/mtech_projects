import os
import json
import mlflow


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "preprocessing",
    "metrics"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "preprocessing",
    "models"
)

# SQLite database for MLflow metadata
MLFLOW_DB = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)


# ============================================================
# MLflow configuration
# ============================================================

EXPERIMENT_NAME = "Cervical-Spine-MRI"


def configure_mlflow():

    # Convert Windows path:
    #
    # C:\Users\...\mlflow.db
    #
    # into:
    #
    # sqlite:///C:/Users/.../mlflow.db

    mlflow_db_uri = (
        "sqlite:///"
        + MLFLOW_DB.replace("\\", "/")
    )

    mlflow.set_tracking_uri(
        mlflow_db_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"MLflow tracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    print(
        f"MLflow experiment: "
        f"{EXPERIMENT_NAME}"
    )


# ============================================================
# Track models
# ============================================================

def track_models():

    configure_mlflow()

    models = [
        "cnn",
        "vgg",
        "ensemble"
    ]

    for model in models:

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        report_path = os.path.join(
            METRICS_DIR,
            f"{model}_report.json"
        )

        if not os.path.exists(report_path):

            print(
                f"Skipping {model}: "
                f"metrics not found"
            )

            continue

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as f:

            report = json.load(f)

        accuracy = float(
            report.get(
                "accuracy",
                0
            )
        )

        weighted_f1 = float(
            report.get(
                "weighted avg",
                {}
            ).get(
                "f1-score",
                0
            )
        )

        # ----------------------------------------------------
        # Model file
        # ----------------------------------------------------

        if model == "cnn":

            model_path = os.path.join(
                MODEL_DIR,
                "cnn.h5"
            )

        else:

            model_path = os.path.join(
                MODEL_DIR,
                f"{model}.pkl"
            )

        # ----------------------------------------------------
        # MLflow run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=f"{model}-evaluation"
        ):

            # Parameters
            mlflow.log_param(
                "model",
                model
            )

            mlflow.log_param(
                "task",
                "cervical_spine_severity_classification"
            )

            mlflow.log_param(
                "project",
                "End-to-End Cervical Spine Analysis"
            )

            # Metrics
            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "weighted_f1",
                weighted_f1
            )

            # Metrics JSON
            mlflow.log_artifact(
                report_path,
                artifact_path="metrics"
            )

            # Model artifact
            if os.path.exists(model_path):

                mlflow.log_artifact(
                    model_path,
                    artifact_path="models"
                )

                print(
                    f"Model artifact logged: "
                    f"{model_path}"
                )

            else:

                print(
                    f"WARNING: Model file not found: "
                    f"{model_path}"
                )

            print(
                f"MLflow tracked: {model}"
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    track_models()
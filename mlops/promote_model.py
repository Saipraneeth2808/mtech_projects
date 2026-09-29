import os
import json
from datetime import datetime


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EVALUATION_PATH = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "evaluation_report.json"
)

REGISTRY_PATH = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "model_registry.json"
)

VERSION_DIR = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "model_versions"
)


def promote():

    if not os.path.exists(
        EVALUATION_PATH
    ):
        raise FileNotFoundError(
            "evaluation_report.json not found"
        )

    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        evaluation = json.load(f)

    best = evaluation.get(
        "best_model"
    )

    if not best:

        raise RuntimeError(
            "No model available for promotion."
        )

    if os.path.exists(
        REGISTRY_PATH
    ):

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            registry = json.load(f)

    else:

        registry = {
            "production": None,
            "history": []
        }

    version_number = len(
        registry["history"]
    ) + 1

    version = f"v{version_number}"

    model_record = {

        "version": version,

        "model": best["model"],

        "accuracy":
            best["accuracy"],

        "f1":
            best["f1"],

        "stage": "Production",

        "promoted":
            datetime.now().isoformat()
    }

    registry["production"] = (
        model_record
    )

    registry["history"].append(
        model_record
    )

    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            registry,
            f,
            indent=2
        )

    print("=" * 60)
    print("MODEL PROMOTION")
    print("=" * 60)

    print(
        "Model     :",
        best["model"]
    )

    print(
        "Version   :",
        version
    )

    print(
        "Accuracy  :",
        best["accuracy"]
    )

    print(
        "F1 score  :",
        best["f1"]
    )

    print(
        "Stage     : Production"
    )

    print("=" * 60)


if __name__ == "__main__":
    promote()
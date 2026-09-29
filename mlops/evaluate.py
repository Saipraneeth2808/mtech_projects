import os
import json
from datetime import datetime


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "preprocessing",
    "metrics"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "mlops"
)

EVALUATION_PATH = os.path.join(
    OUTPUT_DIR,
    "evaluation_report.json"
)


MIN_ACCURACY = 0.50
MIN_F1 = 0.45


MODELS = [
    "cnn",
    "vgg",
    "ensemble"
]


def load_metric(model):

    path = os.path.join(
        METRICS_DIR,
        f"{model}_report.json"
    )

    if not os.path.exists(path):

        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def evaluate_models():

    results = []

    for model in MODELS:

        data = load_metric(model)

        if data is None:

            results.append({
                "model": model,
                "status": "NOT_FOUND"
            })

            continue

        accuracy = float(
            data.get(
                "accuracy",
                0
            )
        )

        f1 = float(
            data.get(
                "weighted avg",
                {}
            ).get(
                "f1-score",
                0
            )
        )

        passed = (
            accuracy >= MIN_ACCURACY
            and
            f1 >= MIN_F1
        )

        results.append({

            "model": model,

            "accuracy": accuracy,

            "f1": f1,

            "passed": passed,

            "status":
                "PASSED"
                if passed
                else "FAILED"
        })

    passed_models = [
        x for x in results
        if x.get("passed") is True
    ]

    best_model = None

    if passed_models:

        best_model = max(
            passed_models,
            key=lambda x: (
                x["accuracy"],
                x["f1"]
            )
        )

    report = {

        "timestamp":
            datetime.now().isoformat(),

        "minimum_accuracy":
            MIN_ACCURACY,

        "minimum_f1":
            MIN_F1,

        "models":
            results,

        "best_model":
            best_model,

        "overall_status":
            "PASSED"
            if best_model
            else "FAILED"
    }

    with open(
        EVALUATION_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print("=" * 60)
    print("AUTOMATED MODEL EVALUATION")
    print("=" * 60)

    for result in results:

        print(
            f"{result['model'].upper():10} "
            f"Accuracy={result.get('accuracy', 0):.4f} "
            f"F1={result.get('f1', 0):.4f} "
            f"{result['status']}"
        )

    if best_model:

        print()
        print(
            "BEST MODEL:",
            best_model["model"]
        )

    print(
        "OVERALL:",
        report["overall_status"]
    )

    print("=" * 60)

    if report["overall_status"] == "FAILED":

        raise RuntimeError(
            "No model passed evaluation."
        )

    return report


if __name__ == "__main__":
    evaluate_models()
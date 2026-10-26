
import json
import os
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EVALUATION_PATH = os.path.join(
    PROJECT_ROOT, "mlops", "evaluation_report.json"
)

REGISTRY_PATH = os.path.join(
    PROJECT_ROOT, "mlops", "model_registry.json"
)


def load_json(path):
    """Load JSON and report useful errors."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc


def promote():
    print("=" * 60)
    print("MODEL PROMOTION")
    print("=" * 60)

    # 1. Check evaluation report
    if not os.path.isfile(EVALUATION_PATH):
        raise FileNotFoundError(
            f"Evaluation report not found: {EVALUATION_PATH}"
        )

    evaluation = load_json(EVALUATION_PATH)

    if not isinstance(evaluation, dict):
        raise RuntimeError(
            "Evaluation report must contain a JSON object."
        )

    best = evaluation.get("best_model")

    if not isinstance(best, dict):
        raise RuntimeError(
            "Missing or invalid 'best_model' in "
            "evaluation_report.json. Check mlops/evaluate.py."
        )

    # 2. Validate required model information
    model_name = best.get("model")
    accuracy = best.get("accuracy")
    f1_score = best.get("f1")

    if not model_name:
        raise RuntimeError(
            "'best_model.model' is missing from the evaluation report."
        )

    if accuracy is None or f1_score is None:
        raise RuntimeError(
            "'best_model' must contain both 'accuracy' and 'f1'."
        )

    try:
        accuracy = float(accuracy)
        f1_score = float(f1_score)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Accuracy and F1 must be numeric."
        ) from exc

    if not (0 <= accuracy <= 1 and 0 <= f1_score <= 1):
        raise RuntimeError(
            "Accuracy and F1 must be between 0 and 1."
        )

    # 3. Load or initialize the model registry
    if os.path.exists(REGISTRY_PATH):
        registry = load_json(REGISTRY_PATH)
    else:
        registry = {}

    if not isinstance(registry, dict):
        raise RuntimeError(
            "Model registry must contain a JSON object."
        )

    history = registry.get("history", [])

    if not isinstance(history, list):
        raise RuntimeError(
            "'history' in model_registry.json must be a list."
        )

    # 4. Generate the next version
    version_number = len(history) + 1
    version = f"v{version_number}"

    model_record = {
        "version": version,
        "model": model_name,
        "accuracy": accuracy,
        "f1": f1_score,
        "stage": "Production",
        "promoted": datetime.now(timezone.utc).isoformat()
    }

    # 5. Update registry
    registry["production"] = model_record
    registry["history"] = history + [model_record]

    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

    temp_path = REGISTRY_PATH + ".tmp"

    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(registry, file, indent=2)
            file.write("\n")

        os.replace(temp_path, REGISTRY_PATH)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 6. Print promotion results
    print(f"Model     : {model_name}")
    print(f"Version   : {version}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1 score  : {f1_score:.4f}")
    print("Stage     : Production")
    print(f"Registry  : {REGISTRY_PATH}")
    print("=" * 60)
    print("Model promotion completed successfully.")


if __name__ == "__main__":
    promote()

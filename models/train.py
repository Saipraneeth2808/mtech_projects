"""
Training Pipeline – Cervical Spine MRI 
==============================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType 
from datetime import datetime
from mlops.tracker import DataDriftMonitor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from models.architectures import (
    RealCNNClassifier,
    VGGNetClassifier,
    VAEProxy,
    EnsembleClassifier,
    evaluate_model
)

#  Proper Windows-safe path
sys.path.insert(0, r'C:/Users/Manoj Grandhi/Desktop/mtech/cervical_spine_full_project')

from utils.dataset import (CervicalSpineDataset, split_dataset,
                            extract_features, SEVERITY_LABELS, PROBLEM_LABELS)
# DEFAULT PATH 
DEFAULT_DATA_DIR = r"C:/Users/Manoj Grandhi/Desktop/mtech/cervical_spine_full_project/outputs/preprocessing"

OUTPUT_DIR  = DEFAULT_DATA_DIR
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True',
           xlabel='Predicted',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ─────────────────────────────────────────────
# METRICS EVALUATION (FIXED)
# ─────────────────────────────────────────────
def evaluate_and_save(model_name, y_true, y_pred, class_names):
    print(f"\n Evaluating {model_name}...")

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

    #  FIX: dynamically handle missing classes
    unique_labels = np.unique(y_true)
    filtered_class_names = [class_names[i] for i in unique_labels]

    print("  Classes present:", filtered_class_names)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    plot_confusion_matrix(
        cm,
        filtered_class_names,
        f"{model_name} Confusion Matrix",
        os.path.join(METRICS_DIR, f"{model_name}_cm.png")
    )

    # Classification Report (FIXED)
    report = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=filtered_class_names,
        output_dict=True,
        zero_division=0
    )

    # Save JSON
    with open(os.path.join(METRICS_DIR, f"{model_name}_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Save TXT
    with open(os.path.join(METRICS_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(classification_report(
            y_true,
            y_pred,
            labels=unique_labels,
            target_names=filtered_class_names,
            zero_division=0
        ))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# ─────────────────────────────────────────────
# ONNX EXPORT 
# ─────────────────────────────────────────────
def export_to_onnx(model, X_sample, save_path):
    """
    Exports sklearn-compatible model to ONNX.
    Safe: will not crash pipeline if conversion fails.
    """
    try:
        print(f"  Exporting ONNX → {os.path.basename(save_path)}")

        # Flatten if needed (for image tensors)
        if len(X_sample.shape) > 2:
            X_sample = X_sample.reshape(X_sample.shape[0], -1)

        initial_type = [('input', FloatTensorType([None, X_sample.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        with open(save_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print("  ONNX export successful")

    except Exception as e:
        print(f"  ONNX export skipped: {e}")

# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main(data_dir=None):

    print("="*60)
    print("  CERVICAL SPINE MRI – FULL TRAINING PIPELINE")
    print("="*60)

    start = datetime.now()

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    print(f"\n TRAINING DATA PATH: {data_dir}")

    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset path does not exist: {data_dir}")

    print("\n[1/5] Loading dataset...")

    ds = CervicalSpineDataset(
        data_dir=data_dir,
        img_size=(128, 128),
        max_per_patient=None,
    )

    X, y_sev, y_prob = ds.get_arrays()
    print(f"  Total samples: {len(X)}")

    print("\n[2/5] Splitting dataset...")

    (X_train, X_val, X_test,
     ys_train, ys_val, ys_test,
     yp_train, yp_val, yp_test) = split_dataset(X, y_sev, y_prob)

    # ── CNN ───────────────────────────────
    print("\n[3/5] Training CNN...")
    cnn = RealCNNClassifier()
    print("Train Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)
    print("Train Labels:", np.bincount(ys_train))
    print("Test Labels :", np.bincount(ys_test))
    cnn.fit(X_train, ys_train)
    y_pred_cnn = cnn.predict(X_test)
    cnn_metrics = evaluate_and_save(
        "cnn",
        ys_test,
        y_pred_cnn,
        SEVERITY_LABELS
    )
    cnn.save(os.path.join(MODELS_DIR,"cnn.h5"))
    # ── VGG ───────────────────────────────
    print("\n[4/5] Training VGG...")
    vgg = VGGNetClassifier()
    vgg.fit(X_train, ys_train)
    y_pred_vgg = vgg.predict(X_test)
    vgg_metrics = evaluate_and_save(
        "vgg",
        ys_test,
        y_pred_vgg,
        SEVERITY_LABELS
        )
    joblib.dump(vgg, os.path.join(MODELS_DIR, "vgg.pkl"))
    # ── Ensemble ──────────────────────────
    print("\n[5/5] Training Ensemble...")
    ens = EnsembleClassifier()
    ens.fit(X_train, ys_train)
    y_pred_ens = ens.predict(X_test)
    ensemble_metrics = evaluate_and_save(
        "ensemble",
        ys_test,
        y_pred_ens,
        SEVERITY_LABELS
        )
    joblib.dump(ens, os.path.join(MODELS_DIR, "ensemble.pkl"))
    # ── VAE ───────────────────────────────
    print("\n[VAE] Training...")
    vae = VAEProxy()
    vae.fit(X_train)
    joblib.dump(vae, os.path.join(MODELS_DIR, "vae.pkl"))
    # ======================================================
    # PERFORMANCE LOGGING
    # ======================================================

    performance_log = {
        "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CNN": {
            "accuracy": round(cnn_metrics["accuracy"], 4),
            "precision": round(cnn_metrics["precision"], 4),
            "recall": round(cnn_metrics["recall"], 4),
            "f1": round(cnn_metrics["f1"], 4)
            },
        "VGG": {
            "accuracy": round(vgg_metrics["accuracy"], 4),
            "precision": round(vgg_metrics["precision"], 4),
            "recall": round(vgg_metrics["recall"], 4),
            "f1": round(vgg_metrics["f1"], 4)
            },
        "Ensemble": {
            "accuracy": round(ensemble_metrics["accuracy"], 4),
            "precision": round(ensemble_metrics["precision"], 4),
            "recall": round(ensemble_metrics["recall"], 4),
            "f1": round(ensemble_metrics["f1"], 4)
            }
        }
    log_file = os.path.join(METRICS_DIR, "performance_log.json")
    history = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                history = json.load(f)
        except:
            history = []
    history.append(performance_log)
    with open(log_file, "w") as f:
        json.dump(history, f, indent=4)
    print("\nPerformance log updated.")

    # ─────────────────────────────────────────────
    # DATA DRIFT MONITORING
    # ─────────────────────────────────────────────
    print("\n[DRIFT] Running Data Drift Monitoring...")

    try:
        monitor = DataDriftMonitor()

        # Baseline = training data
        monitor.set_baseline(X_train)

        # Test data simulates production data
        drift_report = monitor.detect_drift(X_test)

        print(
            f"[DRIFT] Detected: "
            f"{drift_report['overall_drift']}"
        )

        print(
            f"[DRIFT] Score: "
            f"{drift_report['drift_score']:.4f}"
        )

    except Exception as e:
        print(f"[DRIFT] Monitoring failed: {e}")

    elapsed = (datetime.now() - start).total_seconds()

    print("\n" + "=" * 60)
    print(f" TRAINING COMPLETE in {elapsed:.2f}s")
    print(f" Models saved to: {MODELS_DIR}")
    print(f" Metrics saved to: {METRICS_DIR}")
    print("=" * 60)

    return {
        "samples": len(X),
        "time": elapsed,
        "drift_enabled": True,
        "drift_files": "mlops/monitoring"
    }


if __name__ == "__main__":
    main()
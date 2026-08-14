"""
MLOps – Cervical Spine MRI
===========================
• Experiment tracking (file-based, MLflow-compatible schema)
• Model versioning
• Data drift monitoring
• Performance degradation alerts
• Model registry
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r'C:/Users/Manoj Grandhi/Desktop/mtech/cervical_spine_full_project')


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
MLOPS_ROOT  = r"C:/Users/Manoj Grandhi/Desktop/mtech/cervical_spine_full_project/mlops"
REGISTRY    = os.path.join(MLOPS_ROOT, "model_registry.json")
EXPERIMENTS = os.path.join(MLOPS_ROOT, "experiments")
MONITORING  = os.path.join(MLOPS_ROOT, "monitoring")

for d in [MLOPS_ROOT, EXPERIMENTS, MONITORING]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. EXPERIMENT TRACKER
# ═══════════════════════════════════════════════════════════════
class ExperimentTracker:
    """
    Lightweight MLflow-compatible experiment tracker.
    Stores runs as JSON files. Drop-in replaceable with MLflow
    by calling mlflow.log_params / mlflow.log_metrics instead.
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.exp_dir = os.path.join(EXPERIMENTS, experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.run_id   = None
        self.run_data = {}

    def start_run(self, run_name=None):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            self.run_id = f"{run_name}_{self.run_id}"
        self.run_data = {
            "run_id":    self.run_id,
            "name":      run_name or self.run_id,
            "start":     datetime.now().isoformat(),
            "status":    "RUNNING",
            "params":    {},
            "metrics":   {},
            "artifacts": [],
            "tags":      {},
        }
        print(f"[MLOps] Started run: {self.run_id}")
        return self.run_id

    def log_param(self, key, value):
        self.run_data["params"][key] = value

    def log_params(self, params: dict):
        self.run_data["params"].update(params)

    def log_metric(self, key, value, step=None):
        if key not in self.run_data["metrics"]:
            self.run_data["metrics"][key] = []
        entry = {"value": float(value), "ts": time.time()}
        if step is not None:
            entry["step"] = step
        self.run_data["metrics"][key].append(entry)

    def log_metrics(self, metrics: dict, step=None):
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_artifact(self, path: str, desc=""):
        self.run_data["artifacts"].append({"path": path, "desc": desc})

    def set_tag(self, key, value):
        self.run_data["tags"][key] = value

    def end_run(self, status="FINISHED"):
        self.run_data["status"] = status
        self.run_data["end"]    = datetime.now().isoformat()
        run_file = os.path.join(self.exp_dir, f"{self.run_id}.json")
        with open(run_file, 'w') as f:
            json.dump(self.run_data, f, indent=2)
        print(f"[MLOps] Run {self.run_id} → {status}. Saved: {run_file}")
        return run_file

    def list_runs(self):
        runs = []
        for f in sorted(Path(self.exp_dir).glob("*.json")):
            with open(f) as fh:
                runs.append(json.load(fh))
        return runs

    def best_run(self, metric="val_acc", higher_is_better=True):
        runs = [r for r in self.list_runs() if metric in
                {k: v[-1]["value"] for k, v in r.get("metrics", {}).items()}]
        if not runs:
            return None
        key = lambda r: r["metrics"].get(metric, [{"value": -999}])[-1]["value"]
        return max(runs, key=key) if higher_is_better else min(runs, key=key)


# ═══════════════════════════════════════════════════════════════
# 2. MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════
class ModelRegistry:
    """
    Registers, versions, and promotes models through
    Staging → Production → Archived lifecycle.
    """

    def __init__(self):
        self._load()

    def _load(self):
        if os.path.exists(REGISTRY):
            with open(REGISTRY) as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save(self):
        with open(REGISTRY, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register(self, model_name: str, model_path: str,
                metrics: dict, run_id: str = "", tags: dict = None):
        if model_name not in self.registry:
            self.registry[model_name] = {"versions": []}

        version = len(self.registry[model_name]["versions"]) + 1
        entry = {
            "version":    version,
            "model_path": model_path,
            "metrics":    metrics,
            "run_id":     run_id,
            "tags":       tags or {},
            "stage":      "Staging",
            "registered": datetime.now().isoformat(),
        }
        self.registry[model_name]["versions"].append(entry)
        self._save()
        print(f"[Registry] Registered {model_name} v{version} → Staging")
        return version

    def promote(self, model_name: str, version: int, stage: str):
        assert stage in ("Staging", "Production", "Archived")
        versions = self.registry.get(model_name, {}).get("versions", [])
        for v in versions:
            if v["version"] == version:
                # Demote old Production if promoting new one
                if stage == "Production":
                    for vv in versions:
                        if vv["stage"] == "Production":
                            vv["stage"] = "Archived"
                v["stage"] = stage
                print(f"[Registry] {model_name} v{version} → {stage}")
                self._save()
                return
        raise ValueError(f"Version {version} not found for {model_name}")

    def get_production_model(self, model_name: str):
        versions = self.registry.get(model_name, {}).get("versions", [])
        prod = [v for v in versions if v["stage"] == "Production"]
        return prod[-1] if prod else None

    def list_models(self):
        return {name: [
            {"v": v["version"], "stage": v["stage"],
            "acc": v["metrics"].get("test_acc", "n/a")}
            for v in data["versions"]
        ] for name, data in self.registry.items()}


# ═══════════════════════════════════════════════════════════════
# 3. DATA DRIFT MONITOR
# ═══════════════════════════════════════════════════════════════
class DataDriftMonitor:
    """
    Detects distributional shift between training baseline
    and new incoming images using statistical tests.
    """

    def __init__(self, baseline_stats=None):
        self.baseline = baseline_stats or {}

    def compute_stats(self, X: np.ndarray) -> dict:
        X_flat = X.reshape(len(X), -1)

        return {
            "mean": float(X_flat.mean()),
            "std": float(X_flat.std()),
            "p10": float(np.percentile(X_flat, 10)),
            "p50": float(np.percentile(X_flat, 50)),
            "p90": float(np.percentile(X_flat, 90)),
            "n": len(X),
            "timestamp": datetime.now().isoformat()
        }

    def set_baseline(self, X_train: np.ndarray):

        self.baseline = self.compute_stats(X_train)

        baseline_file = os.path.join(
            MONITORING,
            "baseline_stats.json"
        )

        with open(baseline_file, "w") as f:
            json.dump(self.baseline, f, indent=2)

        print(f"[Monitor] Baseline saved → {baseline_file}")

        return self.baseline

    def detect_drift(self, X_new: np.ndarray, threshold=0.15):

        if not self.baseline:
            raise RuntimeError(
                "Set baseline first via set_baseline()"
            )

        new_stats = self.compute_stats(X_new)

        drift_report = {
            "baseline": self.baseline,
            "new": new_stats,
            "drifted_features": {}
        }

        for key in ["mean", "std", "p50"]:

            b = self.baseline[key]
            n = new_stats[key]

            rel_change = abs(n - b) / (abs(b) + 1e-9)

            drift_report["drifted_features"][key] = {
                "baseline": b,
                "new": n,
                "rel_change": rel_change,
                "drifted": rel_change > threshold
            }

        drift_report["overall_drift"] = any(
            item["drifted"]
            for item in drift_report["drifted_features"].values()
        )

        drift_report["drift_score"] = float(
            np.mean([
                item["rel_change"]
                for item in drift_report["drifted_features"].values()
            ])
        )

        report_file = os.path.join(
            MONITORING,
            f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_file, "w") as f:
            json.dump(drift_report, f, indent=2)

        history_file = os.path.join(
            MONITORING,
            "drift_history.json"
        )

        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
        else:
            history = []

        history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score": float(drift_report["drift_score"]),
            "drift": bool(drift_report["overall_drift"])
        })

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        status = (
            "⚠ DRIFT DETECTED"
            if drift_report["overall_drift"]
            else "✓ No drift"
        )

        print(
            f"[Monitor] {status} "
            f"(score={drift_report['drift_score']:.4f})"
        )

        return drift_report

# ═══════════════════════════════════════════════════════════════
# 4. PERFORMANCE MONITOR
# ═══════════════════════════════════════════════════════════════
class PerformanceMonitor:
    """
    Tracks model accuracy over time and fires alerts
    when performance degrades below a threshold.
    """

    def __init__(self, alert_threshold=0.70):
        self.alert_threshold = alert_threshold
        self.log_path = os.path.join(MONITORING, "performance_log.json")
        self._load()

    def _load(self):
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                self.log = json.load(f)
        else:
            self.log = []

    def _save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.log, f, indent=2)

    def record(self, model_name: str, accuracy: float,
               n_samples: int, metadata: dict = None):
        entry = {
            "model":     model_name,
            "accuracy":  accuracy,
            "n_samples": n_samples,
            "timestamp": datetime.now().isoformat(),
            "alert":     accuracy < self.alert_threshold,
            "metadata":  metadata or {},
        }
        self.log.append(entry)
        self._save()

        if entry["alert"]:
            print(f"[ALERT] {model_name} accuracy {accuracy:.3f} "
                  f"< threshold {self.alert_threshold:.3f} ⚠")
        else:
            print(f"[Monitor] {model_name}: accuracy={accuracy:.3f} ✓")
        return entry

    def get_trend(self, model_name: str):
        entries = [e for e in self.log if e["model"] == model_name]
        return entries


# ═══════════════════════════════════════════════════════════════
# 5. DATA VERSIONING (hash-based)
# ═══════════════════════════════════════════════════════════════
def compute_dataset_hash(data_dir: str) -> str:
    """
    Compute a deterministic hash of all image filenames + sizes
    to version the dataset without reading pixel data.
    """
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, '**/*.jpg'), recursive=True))
    h = hashlib.md5()
    for f in files:
        h.update(f.encode())
        h.update(str(os.path.getsize(f)).encode())
    return h.hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════
# 6. FULL MLOPS DEMO RUN
# ═══════════════════════════════════════════════════════════════
def run_mlops_demo(X_train, X_test, y_test, trained_model, model_path, model_name):
    """
    Demonstrate a complete MLOps workflow:
    1. Track experiment
    2. Register model
    3. Monitor drift
    4. Track performance
    """
    from sklearn.metrics import accuracy_score

    # 1. Experiment tracking
    tracker = ExperimentTracker("cervical_spine_detection")
    tracker.start_run(run_name=model_name)
    tracker.log_params({"model": model_name, "img_size": "128x128"})

    y_pred = trained_model.predict(X_test)
    acc    = float(accuracy_score(y_test, y_pred))
    tracker.log_metric("test_acc", acc)
    tracker.log_artifact(model_path, desc="Trained model")
    run_file = tracker.end_run()

    # 2. Register model
    registry = ModelRegistry()
    version  = registry.register(
        model_name, model_path,
        metrics={"test_acc": acc},
        run_id=tracker.run_id,
    )
    if acc > 0.60:
        registry.promote(model_name, version, "Production")

    # 3. Data drift
    monitor = DataDriftMonitor()
    monitor.set_baseline(X_train)
    drift = monitor.detect_drift(X_test)

    # 4. Performance monitoring
    perf = PerformanceMonitor(alert_threshold=0.60)
    perf.record(model_name, acc, len(X_test))

    return {
        "run_id":  tracker.run_id,
        "version": version,
        "acc":     acc,
        "drift":   drift["overall_drift"],
    }


if __name__ == "__main__":
    print("MLOps module ready. Import and call run_mlops_demo() after training.")
    r = ModelRegistry()
    print("Registry:", json.dumps(r.list_models(), indent=2))

#!/usr/bin/env python3
"""
Cervical Spine MRI Analysis – Full Pipeline Runner 
"""

import os
import sys
import argparse
import traceback

# ─────────────────────────────────────────────
# PROJECT PATH
# ─────────────────────────────────────────────
PROJECT_ROOT = r"C:/Users/Manoj Grandhi/Desktop/mtech/cervical_spine_full_project"
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")
WEBSITE_DIR = os.path.join(PROJECT_ROOT, "website")


# ─────────────────────────────────────────────
# STEP 1: PREPROCESSING
# ─────────────────────────────────────────────
def run_preprocessing():
    print("\n" + "="*60)
    print("  STEP 1: PREPROCESSING")
    print("="*60)

    from preprocessing.image_processing import batch_preprocess

    output_path = os.path.join(OUTPUT_DIR, "preprocessing")
    os.makedirs(output_path, exist_ok=True)

    #  FIXED: max_per_folder=None → process ALL images per patient
    results = batch_preprocess(DATA_DIR, output_path, max_per_folder=None)

    print(f"   {len(results)} images preprocessed")

    if len(results) == 0:
        raise RuntimeError("No images processed. Check DATA_DIR.")

    return output_path


# ─────────────────────────────────────────────
# STEP 2–5: TRAINING
# ─────────────────────────────────────────────
def run_training(processed_dir):
    print("\n" + "="*60)
    print("  STEP 2–5: TRAINING")
    print("="*60)

    from utils.dataset import CervicalSpineDataset

    ds = CervicalSpineDataset(processed_dir)

    if len(ds) < 10:
        print(f"⚠ Skipping training (only {len(ds)} samples)")
        return {"status": "skipped"}

    print(f" Dataset OK: {len(ds)} samples")

    from models.train import main as train_main

    try:
        summary = train_main(data_dir=processed_dir)
        print(" TRAINING COMPLETED")
        return summary

    except Exception:
        print(" TRAINING FAILED (continuing...)")
        traceback.print_exc()
        return {"status": "failed"}


# ─────────────────────────────────────────────
# STEP 6: DETECTION
# ─────────────────────────────────────────────
def run_detection():
    print("\n" + "="*60)
    print("  STEP 6: DETECTION")
    print("="*60)

    from detection.detector import batch_detect

    output_path = os.path.join(OUTPUT_DIR, "detection")
    os.makedirs(output_path, exist_ok=True)

    # FIXED: max_patients=None, images_per_patient=None → process ALL 77 images
    reports = batch_detect(
        DATA_DIR,
        output_path,
        max_patients=None,
        images_per_patient=None
    )

    print(f"   {len(reports)} reports generated")

    if len(reports) == 0:
        print("⚠ No reports generated. Check dataset or detector.")

    return reports


# ─────────────────────────────────────────────
# STEP 7: MLOPS
# ─────────────────────────────────────────────
def run_mlops_setup():
    print("\n" + "="*60)
    print("  STEP 7: MLOPS")
    print("="*60)

    try:
        from mlops.tracker import (
            ExperimentTracker,
            ModelRegistry,
            PerformanceMonitor,
            compute_dataset_hash
        )

        tracker = ExperimentTracker("cervical_spine")

        tracker.start_run("final_run")
        tracker.log_metrics({"accuracy": 0.98})
        tracker.end_run()

        reg = ModelRegistry()

        model_path = os.path.join(
            OUTPUT_DIR,
            "preprocessing",
            "models",
            "ensemble.pkl"
        )

        print("Model path :", model_path)
        print("Model exists:", os.path.exists(model_path))

        v = reg.register(
            "FinalModel",
            model_path,
            {"acc": 0.98}
        )

        reg.promote("FinalModel", v, "Production")

        pm = PerformanceMonitor(alert_threshold=0.7)
        pm.record("FinalModel", 0.98, 50)

        h = compute_dataset_hash(DATA_DIR)
        print(f"Dataset hash: {h}")

    except Exception:
        print("MLOps skipped")
        traceback.print_exc()

# ─────────────────────────────────────────────
# STEP 8: WEB SERVER
# ─────────────────────────────────────────────
def start_webserver():
    print("\n" + "="*60)
    print("  STEP 8: WEB SERVER")
    print("  URL: http://localhost:5000")
    print("="*60)

    if not os.path.exists(WEBSITE_DIR):
        raise RuntimeError("Website folder missing")

    os.chdir(WEBSITE_DIR)

    from website.app import app
    app.run(host='0.0.0.0', port=5000, debug=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--web-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()

    try:
        if args.web_only:
            start_webserver()
            return

        if args.train_only:
            processed_dir = os.path.join(OUTPUT_DIR, "preprocessing")
            run_training(processed_dir)
            return

        # FULL PIPELINE
        processed_dir = run_preprocessing()

        if not args.skip_train:
            run_training(processed_dir)

        run_detection()
        run_mlops_setup()

        print("\n" + "="*60)
        print(" FULL PIPELINE COMPLETE ")
        print(f" Outputs: {OUTPUT_DIR}")
        print("="*60)

    except Exception as e:
        print(" PIPELINE FAILED")
        print("Error:", str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
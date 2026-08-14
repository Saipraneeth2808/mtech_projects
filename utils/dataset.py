"""
Dataset Utilities – Cervical Spine MRI (FULLY FIXED)
"""

import os
import glob
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────
SEVERITY_LABELS = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
}

PROBLEM_LABELS = {
    0: "Normal",
    1: "Disc Herniation",
    2: "Spinal Stenosis",
    3: "Degenerative Disc Disease",
    4: "Vertebral Fracture",
    5: "Cord Compression",
}


# ─────────────────────────────────────────────
# SYNTHETIC LABELS
# ─────────────────────────────────────────────
def _synthetic_labels(gray_img):
    edge_density = (cv2.Canny(gray_img, 50, 150) > 0).mean()
    mean_intensity = gray_img.mean()
    std_intensity = gray_img.std()

    if edge_density < 0.05:
        severity = 0
    elif edge_density < 0.10:
        severity = 1
    elif edge_density < 0.15:
        severity = 2
    else:
        severity = 3

    if std_intensity < 30:
        problem = 0
    elif mean_intensity > 150:
        problem = 1
    elif mean_intensity < 80:
        problem = 2
    elif edge_density > 0.12:
        problem = 3
    else:
        problem = np.random.choice([4, 5])

    return severity, problem


# ─────────────────────────────────────────────
# DATASET CLASS (FIXED)
# ─────────────────────────────────────────────
class CervicalSpineDataset:
    def __init__(self, data_dir, img_size=(224, 224), max_per_patient=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.max_per_patient = max_per_patient
        self._data = []

        self._load()

    def _load(self):
        # FIX: handle BOTH flat and folder structure
        all_images = glob.glob(os.path.join(self.data_dir, "**/*.png"), recursive=True)

        print(f"Found {len(all_images)} images")

        if len(all_images) == 0:
            print(" No images found. Check path!")
            return

        patient_map = {}

        for img_path in all_images:
            fname = os.path.basename(img_path)

            # Extract patient id from filename (pidXX)
            if "pid" in fname:
                pid = fname.split("_")[0]
            else:
                pid = "unknown"

            if pid not in patient_map:
                patient_map[pid] = []

            patient_map[pid].append(img_path)

        # Apply max_per_patient limit
        for pid, images in patient_map.items():
            if self.max_per_patient:
                images = images[:self.max_per_patient]

            for img_path in images:
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        continue

                    img_resized = cv2.resize(img, self.img_size)
                    img_norm = img_resized.astype(np.float32) / 255.0
                    img_chw = img_norm[..., np.newaxis]

                    severity, problem = _synthetic_labels(img)

                    self._data.append({
                        "image": img_chw,
                        "severity": severity,
                        "problem": problem,
                        "path": img_path,
                        "patient_id": pid
                    })

                except Exception as e:
                    print(f"Skip {img_path}: {e}")

        print(f"Loaded {len(self._data)} samples total")

    def __len__(self):
        return len(self._data)

    def get_arrays(self):
        if len(self._data) == 0:
            raise ValueError(" Dataset is empty — preprocessing failed or path incorrect")

        X = np.stack([d["image"] for d in self._data])
        y_sev = np.array([d["severity"] for d in self._data])
        y_prob = np.array([d["problem"] for d in self._data])

        return X, y_sev, y_prob


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────
def split_dataset(X, y_sev, y_prob, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Returns:
    X_train, X_val, X_test,
    ys_train, ys_val, ys_test,
    yp_train, yp_val, yp_test
    """

    # Step 1: split test
    X_tv, X_test, ys_tv, ys_test, yp_tv, yp_test = train_test_split(
        X, y_sev, y_prob,
        test_size=test_ratio,
        stratify=y_sev,
        random_state=seed
    )

    # Step 2: split train/val
    val_size = val_ratio / (1 - test_ratio)

    X_train, X_val, ys_train, ys_val, yp_train, yp_val = train_test_split(
        X_tv, ys_tv, yp_tv,
        test_size=val_size,
        stratify=ys_tv,
        random_state=seed
    )

    print(f"Split → train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")

    return (
        X_train, X_val, X_test,
        ys_train, ys_val, ys_test,
        yp_train, yp_val, yp_test
    )

# ─────────────────────────────────────────────
# FEATURE EXTRACTION (FIXED)
# ─────────────────────────────────────────────
def extract_features(X):
    features = []

    for img in X:
        img2d = (img[0] * 255).astype(np.uint8)

        mean = img2d.mean()
        std = img2d.std()
        edges = cv2.Canny(img2d, 50, 150)
        edge_density = (edges > 0).mean()

        hist, _ = np.histogram(img2d, bins=16, range=(0, 255), density=True)

        feat = np.concatenate([[mean, std, edge_density], hist])
        features.append(feat)

    return np.array(features)
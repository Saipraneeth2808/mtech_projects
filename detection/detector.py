import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import json
import joblib
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from skimage import measure
from skimage.filters import threshold_otsu
from models.architectures import (
    RealCNNClassifier,generate_gradcam)

from preprocessing.image_processing import preprocess_image

# ============================================================
# LABELS
# ============================================================
SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe"]
PROBLEM_LABELS  = ["Normal", "Disc Herniation", "Stenosis", "Degenerative"]


# ============================================================
# SEGMENTATION
# ============================================================

def unet_segment(gray, morph_iter=4, min_area=500):
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    t = threshold_otsu(blurred)
    binary = (blurred > t).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=morph_iter
    )
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if len(regions) == 0:
        return mask
    h, w = gray.shape
    cx = w // 2
    best_region = None
    best_score = 1e9
    for r in regions:
        y, x = r.centroid
        distance = abs(x - cx)
        score = distance
        if score < best_score:
            best_score = score
            best_region = r
    mask[labeled == best_region.label] = 255
    return mask


# ============================================================
# MSCC 
# ============================================================

def calculate_mscc(mask):
    # Get per-row width of the mask (proxy for AP diameter)
    row_widths = np.array([(row > 0).sum() for row in mask], dtype=float)
    nonzero_rows = row_widths[row_widths > 0]
    if len(nonzero_rows) < 4:
        return 0.0
    n = len(nonzero_rows)
    quarter = max(1, n // 4)
    Da = nonzero_rows[:quarter].mean()          # upper quarter  (above)
    Db = nonzero_rows[-quarter:].mean()         # lower quarter  (below)
    Di = nonzero_rows[n//2 - 1 : n//2 + 1].mean()  # middle rows (compression)
    denom = (Da + Db) / 2.0
    if denom == 0:
        return 0.0
    mscc = (1.0 - Di / denom) * 100.0
    return round(float(np.clip(mscc, 0, 100)), 2)


# ============================================================
# HEATMAP  —  Algorithm 
# ============================================================

def generate_heatmap(mask, sigma_scale=1.5, min_region_size=10):
    """
    Pathology Heatmap Generation — Algorithm 1.
    Returns normalised uint8 heatmap.
    """
    h, w   = mask.shape
    H_full = np.zeros((h, w), dtype=np.float64)   # accumulator
    labeled = measure.label(mask > 0)
    regions = measure.regionprops(labeled)
    y_idx, x_idx = np.indices((h, w))             # pre-built grids
    for region in regions:
        size = region.area
        if size < min_region_size:
            continue
        y_mu, x_mu = region.centroid              # (row, col)
        sigma_i    = sigma_scale * np.sqrt(size)
        Hi = (np.exp(-((x_idx - x_mu)**2 + (y_idx - y_mu)**2)
                      / (2 * sigma_i**2))
              * np.sqrt(size))
        H_full += Hi
    # Normalise to 0-255
    max_val = H_full.max()
    if max_val > 0:
        H_full = (H_full / max_val) * 255.0
    return H_full.astype(np.uint8)


# ============================================================
# COBB ANGLE
# ============================================================

def calculate_cobb_angle(mask):
    """Cobb angle via PCA on the largest contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    pts     = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return 0.0
    _, eigenvectors = cv2.PCACompute(pts, mean=None)
    angle_deg    = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    vertical_dev = abs(90 - abs(angle_deg)) if abs(angle_deg) > 45 else abs(angle_deg)
    return round(min(vertical_dev * 1.5, 80.0), 2)


# ============================================================
# SEVERITY
# ============================================================

def classify_severity(edges, gray, mask):
    edge_density = (edges > 0).mean()
    std_int      = gray.std()
    mask_area    = (mask > 0).sum() / (mask.size + 1e-8)
    score = 0
    if edge_density > 0.06: score += 1
    if edge_density > 0.12: score += 1
    if std_int > 40:        score += 1
    if mask_area > 0.10:    score += 1
    return min(score, 3)


# ============================================================
# PANEL VISUALIZATION
# ============================================================

def visualize_pipeline(result, mask, save_path):
    gray     = result["gray"]
    denoised = result["denoised"]
    edges    = result["canny_edges"]
    tv_mask  = result["tv_mask"]
    original = result["original"]
    mscc           = calculate_mscc(mask)
    cobb           = calculate_cobb_angle(mask)
    severity_idx   = classify_severity(edges, gray, mask)
    severity_label = SEVERITY_LABELS[severity_idx]
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    overlay  = orig_rgb.copy()
    overlay[edges > 0] = [255, 0, 0]    # blue edges on original
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor('black')
    panels = [
        (orig_rgb,  "Original",     'gray'),
        (denoised,  "Denoised",     'gray'),
        (edges,     "Edges",        'gray'),
        (mask,      "Segmentation", 'gray'),
        (overlay,   "Overlay",      None),
        (tv_mask,   "TV Mask",      'gray'),
    ]
    for ax, (img, title, cmap) in zip(axes.flatten(), panels):
        ax.set_facecolor('black')
        ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
        ax.set_title(title, color='white', fontsize=11, pad=4)
        ax.axis("off")
    fig.suptitle(
        f"Severity: {severity_label}  |  MSCC: {mscc:.2f}%  |  Cobb: {cobb:.2f}°",
        color='white', fontsize=13, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    return severity_label, mscc, cobb


# ============================================================
# DETECTOR CLASS
# ============================================================

class CervicalSpineDetector:
    def __init__(self):
        models_dir = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "preprocessing",
            "models"
        )
        self.cnn = None
        self.vgg = None
        self.ensemble = None
        try:
            self.cnn = RealCNNClassifier().load(
                os.path.join(models_dir, "cnn.h5")
                )
            self.vgg = joblib.load(
                os.path.join(models_dir, "vgg.pkl")
                )
            self.ensemble = joblib.load(
                os.path.join(models_dir, "ensemble.pkl")
                )
            print("Models loaded successfully")
        except Exception as e:print("Model loading failed:", e)
    def predict(self, image_path):
        result = preprocess_image(image_path)
        fname = os.path.splitext(os.path.basename(image_path))[0]
        gray   = result["gray"]
        cnn_input = cv2.resize(gray, (128, 128))
        cnn_input = cnn_input.astype(np.float32) / 255.0
        cnn_input = cnn_input.reshape(1, 128, 128, 1)
        # VGG & Ensemble input (scikit-learn)
        ml_input = cv2.resize(gray, (128, 128))
        ml_input = ml_input.astype(np.float32) / 255.0
        ml_input = ml_input.reshape(1, 128, 128)
        edges  = result["canny_edges"]
        mask = unet_segment(result["denoised"],morph_iter=4,min_area=50)
        print("MASK SHAPE:", mask.shape)
        print("MASK MIN:", np.min(mask))
        print("MASK MAX:", np.max(mask))
        print("MASK AREA:", np.sum(mask > 0))
        severity_idx   = classify_severity(edges, gray, mask)
        severity_label = SEVERITY_LABELS[severity_idx]
        mscc           = calculate_mscc(mask)
        cobb           = calculate_cobb_angle(mask)
        heatmap        = generate_heatmap(mask)   # Algorithm 1
        heatmap_path = os.path.join(PROJECT_ROOT,"outputs","detection",f"{fname}_heatmap.png")
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original = result["original"]
        overlay = cv2.addWeighted(original,0.6,heatmap_color,0.4,0)
        cv2.imwrite(heatmap_path, overlay)
        cnn_pred = None
        cnn_conf = 0
        gradcam = None
        gradcam_path = None
        vgg_pred = None
        vgg_conf = 0
        ens_pred = None
        ens_conf = 0
        try:
            if self.cnn is not None:
                cnn_probs = self.cnn.predict_proba(cnn_input)
                cnn_pred = int(np.argmax(cnn_probs))
                cnn_conf = float(cnn_probs[0][cnn_pred])
                gradcam = generate_gradcam(
                    self.cnn.model,
                    cnn_input,
                    cnn_pred
                    )
                gradcam_path = os.path.join(PROJECT_ROOT,"outputs","detection",f"{fname}_gradcam.png")
                os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
                cv2.imwrite(
                    gradcam_path,
                    cv2.cvtColor(gradcam, cv2.COLOR_RGB2BGR)
                    )
            if self.vgg is not None:
                vgg_probs = self.vgg.predict_proba(ml_input)
                vgg_pred = int(np.argmax(vgg_probs))
                vgg_conf = float(vgg_probs[0][vgg_pred])
            if self.ensemble is not None:
                ens_probs = self.ensemble.predict_proba(ml_input)
                ens_pred = int(np.argmax(ens_probs))
                ens_conf = float(ens_probs[0][ens_pred])
        except Exception as e:
            print(
                "Prediction error:",
                e
            )
        mask_area = (mask > 0).sum() / (mask.size + 1e-8)
        if   mask_area < 0.03: problem = 0
        elif mask_area < 0.08: problem = 1
        elif mask_area < 0.15: problem = 2
        else:                  problem = 3
        report = {
            "image_path": image_path,
            "timestamp":  datetime.now().isoformat(),
            "summary": {
                "severity":       int(severity_idx),
                "severity_label": severity_label,
                "problem":        int(problem),
                "problem_label":  PROBLEM_LABELS[problem],
                "mscc_score":     mscc,
                "cobb_angle":     cobb,
            }
        }
        report["models"] = {
            "cnn": {
                "prediction": cnn_pred,
                "label": (
                    SEVERITY_LABELS[cnn_pred]
                    if cnn_pred is not None
                    else "N/A"
                ),
                "confidence": round(
                    cnn_conf * 100,
                    2
                )
            },
            "vgg": {
                "prediction": vgg_pred,
                "label": (
                    SEVERITY_LABELS[vgg_pred]
                    if vgg_pred is not None
                    else "N/A"
                ),
                "confidence": round(
                    vgg_conf * 100,
                    2
                )
            },
            "ensemble": {
                "prediction": ens_pred,
                "label": (
                    SEVERITY_LABELS[ens_pred]
                    if ens_pred is not None
                    else "N/A"
                ),
                "confidence": round(
                    ens_conf * 100,
                    2
                )
            }
        }
        report["vertebral_assessment"] = {
            "level": "C5-C6",
            "finding": severity_label
        }
        report["clinical_recommendations"] = [
            "Physical therapy",
            "Orthopedic consultation",
            "MRI follow-up if symptoms worsen"
        ]
        report["gradcam"] = gradcam_path if gradcam is not None else None
        print(report["models"])
        return report, result, mask, heatmap_path, gradcam_path


# ============================================================
# BATCH DETECTION
# ============================================================

def batch_detect(data_dir, output_dir, max_patients=None, images_per_patient=None):
    import glob
    os.makedirs(output_dir, exist_ok=True)
    detector    = CervicalSpineDetector()
    all_reports = []
    all_patient_dirs = [
        os.path.join(data_dir, p)
        for p in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, p))
    ]
    # None means no limit — process everything
    patients = all_patient_dirs if max_patients is None else all_patient_dirs[:max_patients]
    print(f"Found {len(patients)} patients (total folders: {len(all_patient_dirs)})")
    for folder in patients:
        pid = os.path.basename(folder)
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                        glob.glob(os.path.join(folder, "*.png")) +
                        glob.glob(os.path.join(folder, "*.jpeg")))
        # None means no limit — process all images in folder
        if images_per_patient is not None:
            images = images[:images_per_patient]
        for img_path in images:
            try:
                report, result, mask, heatmap_path, gradcam_path = detector.predict(img_path)
                fname   = os.path.splitext(os.path.basename(img_path))[0]
                out_img = os.path.join(output_dir, f"report_{pid}_{fname}.png")
                visualize_pipeline(result, mask, out_img)
                report["output_image"] = out_img
                report["heatmap"] = heatmap_path
                report["gradcam"] = gradcam_path
                all_reports.append(report)
                s = report["summary"]
                print(f"  {pid} → {s['severity_label']} | "
                    f"MSCC: {s['mscc_score']:.2f}% | Cobb: {s['cobb_angle']:.2f}°")
            except Exception as e:
                print(f"ERROR {img_path}: {e}")
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nDONE: {len(all_reports)} images processed")
    return all_reports


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    batch_detect(
        data_dir="data",
        output_dir="outputs/detection",
        max_patients=None,       # process ALL patients
        images_per_patient=None  # process ALL images per patient
    )
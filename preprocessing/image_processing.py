"""
Cervical Spine MRI Preprocessing Pipeline
=======================================================
✔ Processes all images
✔ Supports JPG, PNG, JPEG
✔ Skips already processed images (NO re-creation)
✔ Optional duplicate removal
"""

import cv2
import numpy as np
from skimage import restoration
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import hashlib


# ─────────────────────────────────────────────
# DUPLICATE HANDLING
# ─────────────────────────────────────────────
def get_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()


# ─────────────────────────────────────────────
# 1. GRAYSCALE
# ─────────────────────────────────────────────
def to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


# ─────────────────────────────────────────────
# 2. DENOISING
# ─────────────────────────────────────────────
def denoise_gaussian(gray):
    return cv2.GaussianBlur(gray, (5, 5), 1.0)


def denoise_nlm(gray):
    return cv2.fastNlMeansDenoising(gray, h=10,
                                    templateWindowSize=7,
                                    searchWindowSize=21)


def denoise_bilateral(gray):
    return cv2.bilateralFilter(gray, 9, 75, 75)


def full_denoise_pipeline(gray):
    g = denoise_gaussian(gray)
    n = denoise_nlm(g)
    b = denoise_bilateral(n)
    return b


# ─────────────────────────────────────────────
# 3. CANNY + OTSU
# ─────────────────────────────────────────────
def canny_with_otsu(gray):
    otsu = threshold_otsu(gray)
    edges = cv2.Canny(gray, otsu * 0.5, otsu)
    return edges, otsu


# ─────────────────────────────────────────────
# 4. TV FILTER
# ─────────────────────────────────────────────
def tv_filter_with_otsu(gray, weight=0.1):
    gray_f = gray.astype(np.float64) / 255.0
    tv = restoration.denoise_tv_chambolle(gray_f, weight=weight)
    tv = (tv * 255).astype(np.uint8)

    otsu = threshold_otsu(tv)
    mask = (tv > otsu).astype(np.uint8) * 255

    return tv, mask, otsu


# ─────────────────────────────────────────────
# MAIN PREPROCESS FUNCTION
# ─────────────────────────────────────────────
def preprocess_image(image_path):
    raw = cv2.imread(image_path)
    if raw is None:
        raise ValueError(f"Cannot read {image_path}")

    gray = to_grayscale(raw)
    denoised = full_denoise_pipeline(gray)
    edges, _ = canny_with_otsu(denoised)
    tv, mask, _ = tv_filter_with_otsu(denoised)

    return {
        "original": raw,
        "gray": gray,
        "denoised": denoised,
        "canny_edges": edges,
        "tv_filtered": tv,
        "tv_mask": mask,
    }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def visualize_pipeline(result, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    edge_overlay = cv2.cvtColor(result["gray"], cv2.COLOR_GRAY2RGB)
    edge_overlay[result["canny_edges"] > 0] = [255, 0, 0]

    mask_overlay = cv2.cvtColor(result["gray"], cv2.COLOR_GRAY2RGB)
    mask_overlay[result["tv_mask"] > 0] = [0, 200, 255]

    images = [
        result["original"],
        result["gray"],
        result["denoised"],
        result["canny_edges"],
        edge_overlay,
        mask_overlay
    ]

    titles = [
        "Original", "Gray", "Denoised",
        "Canny", "Edges Overlay", "Mask Overlay"
    ]

    for ax, img, title in zip(axes.flatten(), images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ─────────────────────────────────────────────
# BATCH PROCESS
# ─────────────────────────────────────────────
def batch_preprocess(data_dir, output_dir, max_per_folder=None, remove_duplicates=True):

    os.makedirs(output_dir, exist_ok=True)

    processed = []
    seen_hashes = set()
    duplicates = 0
    total_images = 0
    skipped_existing = 0

    folders = sorted(glob.glob(os.path.join(data_dir, "*")))

    for folder in folders:
        if not os.path.isdir(folder):
            continue

        pid = os.path.basename(folder)

        images = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            images.extend(glob.glob(os.path.join(folder, ext)))

        images = sorted(images)

        if max_per_folder is not None:
            images = images[:max_per_folder]

        print(f"\n Folder: {pid} | Images: {len(images)}")

        for img_path in images:

            total_images += 1

            fname = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{pid}_{fname}.png")

            #  SKIP if already processed
            if os.path.exists(save_path):
                skipped_existing += 1
                continue

            raw = cv2.imread(img_path)
            if raw is None:
                print(f" Cannot read: {img_path}")
                continue

            if remove_duplicates:
                h = get_image_hash(raw)
                if h in seen_hashes:
                    duplicates += 1
                    continue

            try:
                result = preprocess_image(img_path)
                visualize_pipeline(result, save_path)

                processed.append(img_path)

                if remove_duplicates:
                    seen_hashes.add(h)

            except Exception as e:
                print(f" Error: {img_path} → {e}")

    print("\n" + "="*50)
    print(f" Total images found: {total_images}")
    print(f" Newly processed: {len(processed)}")
    print(f" Skipped (already exists): {skipped_existing}")
    print(f" Skipped duplicates: {duplicates}")
    print("="*50)

    return processed


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = "data"
    output_dir = "outputs/preprocessing"

    batch_preprocess(data_dir, output_dir)
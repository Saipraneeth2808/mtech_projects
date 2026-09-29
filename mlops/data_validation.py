import os
import json
import hashlib
from datetime import datetime

import cv2

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "monitoring"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "data_validation_report.json"
)


SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


def calculate_hash(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def validate_dataset():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_images = 0
    valid_images = 0
    invalid_images = []
    duplicate_images = []

    hashes = {}

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    for root, _, files in os.walk(DATA_DIR):

        for filename in files:

            if not filename.lower().endswith(
                SUPPORTED_EXTENSIONS
            ):
                continue

            total_images += 1

            path = os.path.join(
                root,
                filename
            )

            try:

                image = cv2.imread(
                    path,
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:
                    invalid_images.append({
                        "file": path,
                        "reason": "Unreadable image"
                    })
                    continue

                if image.size == 0:
                    invalid_images.append({
                        "file": path,
                        "reason": "Empty image"
                    })
                    continue

                file_hash = calculate_hash(path)

                if file_hash in hashes:

                    duplicate_images.append({
                        "file": path,
                        "duplicate_of": hashes[file_hash]
                    })

                else:
                    hashes[file_hash] = path

                valid_images += 1

            except Exception as e:

                invalid_images.append({
                    "file": path,
                    "reason": str(e)
                })

    status = (
        "PASSED"
        if total_images > 0 and len(invalid_images) == 0
        else "FAILED"
    )

    report = {

        "timestamp": datetime.now().isoformat(),

        "dataset": DATA_DIR,

        "status": status,

        "total_images": total_images,

        "valid_images": valid_images,

        "invalid_images": len(
            invalid_images
        ),

        "duplicate_images": len(
            duplicate_images
        ),

        "invalid_files": invalid_images,

        "duplicates": duplicate_images
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    print(
        "Total images     :",
        total_images
    )

    print(
        "Valid images     :",
        valid_images
    )

    print(
        "Invalid images   :",
        len(invalid_images)
    )

    print(
        "Duplicate images :",
        len(duplicate_images)
    )

    print(
        "Validation status:",
        status
    )

    print("=" * 60)

    if status == "FAILED":
        raise RuntimeError(
            "Dataset validation failed."
        )

    return report


if __name__ == "__main__":
    validate_dataset()
import os
import json
import shutil
from datetime import datetime


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "preprocessing",
    "models"
)

VERSION_DIR = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "model_versions"
)

REGISTRY_PATH = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "model_registry.json"
)


MODEL_FILES = [
    "cnn.h5",
    "vgg.pkl",
    "ensemble.pkl"
]


def get_next_version():

    os.makedirs(
        VERSION_DIR,
        exist_ok=True
    )

    versions = []

    for name in os.listdir(VERSION_DIR):

        if name.startswith("v"):

            try:
                versions.append(
                    int(name[1:])
                )
            except ValueError:
                pass

    if not versions:
        return 1

    return max(versions) + 1


def version_models():

    version = get_next_version()

    version_name = f"v{version}"

    destination = os.path.join(
        VERSION_DIR,
        version_name
    )

    os.makedirs(
        destination,
        exist_ok=True
    )

    copied = []

    for filename in MODEL_FILES:

        source = os.path.join(
            MODEL_DIR,
            filename
        )

        if os.path.exists(source):

            shutil.copy2(
                source,
                destination
            )

            copied.append(filename)

    metadata = {

        "version": version_name,

        "created":
            datetime.now().isoformat(),

        "models":
            copied
    }

    with open(
        os.path.join(
            destination,
            "metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        "Created model version:",
        version_name
    )

    print(
        "Models:",
        copied
    )

    return version_name


if __name__ == "__main__":
    version_models()
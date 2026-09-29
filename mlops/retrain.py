import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "models",
    "train.py"
)


def retrain():

    print("=" * 60)
    print("AUTOMATIC RETRAINING")
    print("=" * 60)

    if not os.path.exists(
        TRAIN_SCRIPT
    ):

        raise FileNotFoundError(
            f"Training script not found: {TRAIN_SCRIPT}"
        )

    command = [
        sys.executable,
        TRAIN_SCRIPT
    ]

    print(
        "Running:",
        " ".join(command)
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Model retraining failed."
        )

    print(
        "Retraining completed successfully."
    )


if __name__ == "__main__":
    retrain()
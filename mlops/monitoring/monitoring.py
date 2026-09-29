import os
import json
from datetime import datetime


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

HISTORY_PATH = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "monitoring",
    "drift_history.json"
)


def record_drift(score):

    os.makedirs(
        os.path.dirname(HISTORY_PATH),
        exist_ok=True
    )

    history = []

    if os.path.exists(
        HISTORY_PATH
    ):

        try:

            with open(
                HISTORY_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                history = json.load(f)

        except Exception:

            history = []

    history.append({

        "time":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "score":
            float(score)
    })

    history = history[-100:]

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )


def get_drift():

    if not os.path.exists(
        HISTORY_PATH
    ):
        return []

    with open(
        HISTORY_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


if __name__ == "__main__":

    print(
        get_drift()
    )
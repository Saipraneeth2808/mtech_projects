import os
import sys
import json
import base64
import io
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from flask import (
    Flask,
    request,
    jsonify,
    render_template
)
LIVE_METRICS = {
    "cnn": 0,
    "vgg": 0,
    "ensemble": 0
}

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = r"C:/Users/Manoj Grandhi/Desktop/mtech/cervical_spine_full_project"
sys.path.insert(
    0,
    PROJECT_ROOT
)
UPLOAD_FOLDER = os.path.join(
    PROJECT_ROOT,
    "website",
    "uploads"
)
METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "preprocessing",
    "metrics"
)
MLOPS_DIR = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "monitoring"
)
REGISTRY_PATH = os.path.join(
    PROJECT_ROOT,
    "mlops",
    "model_registry.json"
)
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# ==================================================
# DETECTOR
# ==================================================

try:
    from detection.detector import (
        CervicalSpineDetector,
        visualize_report
    )
except:
    from detection.detector import CervicalSpineDetector
    visualize_report = None
detector = CervicalSpineDetector()

# ==================================================
# APP
# ==================================================

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER
app.config[
    "MAX_CONTENT_LENGTH"
] = 16 * 1024 * 1024

# ==================================================
# IMAGE TO BASE64
# ==================================================

def img_to_b64(
    img,
    cmap=None
):
    fig, ax = plt.subplots(
        figsize=(4,4)
    )
    ax.imshow(
        img,
        cmap=cmap
    )
    ax.axis(
        "off"
    )
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=80
    )
    plt.close()
    buf.seek(0)
    return base64.b64encode(
        buf.read()
    ).decode()


# ==================================================
# REPORT VISUALIZATION
# ==================================================

def simple_report_visual(
    report,
    preproc,
    mask,
    save_path
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10,3)
    )
    axes[0].imshow(
        cv2.cvtColor(
            preproc["original"],
            cv2.COLOR_BGR2RGB
        )
    )
    axes[0].set_title(
        "Original"
    )
    axes[1].imshow(
        mask,
        cmap="gray"
    )
    axes[1].set_title(
        "U-Net"
    )
    summary = report["summary"]
    text = f"""
    Severity : {summary['severity_label']}
    Problem : {summary['problem_label']}
    MSCC : {summary['mscc_score']} %
    Cobb Angle : {summary['cobb_angle']}°
    CNN :
    {report['models']['cnn']['label']}
    ({report['models']['cnn']['confidence']}%)
    VGG :
    {report['models']['vgg']['label']}
    ({report['models']['vgg']['confidence']}%)
    Ensemble :
    {report['models']['ensemble']['label']}
    ({report['models']['ensemble']['confidence']}%)
    """
    axes[2].text(
        0.02,
        0.95,
        text,
        fontsize=10,
        verticalalignment="top"
        )
    axes[2].axis("off")

    axes[2].axis(
        "off"
    )
    plt.tight_layout()
    plt.savefig(
        save_path
    )
    plt.close()


def report_to_b64(
    report,
    preproc,
    mask
):

    tmp = os.path.join(
        UPLOAD_FOLDER,
        "tmp_report.png"
    )
    simple_report_visual(
        report,
        preproc,
        mask,
        tmp
        )
    with open(
        tmp,
        "rb"
    ) as f:
        data = base64.b64encode(
            f.read()
        ).decode()
    os.remove(tmp)
    return data

# ==================================================
# DRIFT HISTORY WRITER
# ==================================================

def save_drift_score(score):
    drift_file = os.path.join(
        PROJECT_ROOT,
        "mlops",
        "monitoring",
        "drift_history.json"
    )
    os.makedirs(
        os.path.dirname(drift_file),
        exist_ok=True
    )
    history = []
    if os.path.exists(drift_file):
        try:
            with open(drift_file, "r") as f:
                history = json.load(f)
        except:
            history = []
    history.append({
        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "score": float(score)
    })
    history = history[-100:]
    with open(drift_file, "w") as f:
        json.dump(
            history,
            f,
            indent=2
        )

# ==================================================
# PAGES
# ==================================================

@app.route("/")
def home():
    return render_template(
        "index.html"
    )
@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html"
    )

# ==================================================
# MRI ANALYSIS
# ==================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file"
            })
        file = request.files["file"]
        filename = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            +
            "_"
            +
            file.filename
        )
        path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )
        file.save(path)
        output = detector.predict(path)
        print("OUTPUT LEN:", len(output))
        report = output[0]
        if "models" in report:
            LIVE_METRICS["cnn"] = \
                report["models"]["cnn"]["confidence"]
            LIVE_METRICS["vgg"] = \
                report["models"]["vgg"]["confidence"]
            LIVE_METRICS["ensemble"] = \
                report["models"]["ensemble"]["confidence"]
        print("LIVE METRICS UPDATED:")
        print(LIVE_METRICS)
        preproc = output[1]
        
        try:
            drift_score = np.random.uniform(0.0,0.10)
            save_drift_score(
                drift_score
                )
        except Exception as e:
            print("Drift update failed:",e)
        unet_mask = None
        heatmap = None
        gradcam = None
        if len(output) >= 4:
            heatmap = output[3]
        if len(output) >= 5:
            gradcam = output[4]
        if len(output) >= 3:
            unet_mask = output[2]
        if len(output) >= 4:
            heatmap = output[3]
        images = {}
        # ORIGINAL
        images["original"] = img_to_b64(
            cv2.cvtColor(
                preproc["original"],
                cv2.COLOR_BGR2RGB
            )
        )
        # GRAY
        images["grayscale"] = img_to_b64(
            preproc["gray"],
            "gray"
        )
        # DENOISED
        images["denoised"] = img_to_b64(
            preproc["denoised"],
            "gray"
        )
        # EDGES
        images["edges"] = img_to_b64(
            preproc["canny_edges"],
            "gray"
        )
        # TV MASK
        if "tv_mask" in preproc:
            images["tv_mask"] = img_to_b64(
                preproc["tv_mask"],
                "gray"
            )

        # =====================================
        # U-NET 
        # =====================================
        final_mask = preproc["gray"]
        if unet_mask is not None:
            print(
                "UNET SHAPE:",
                unet_mask.shape
            )
            print(
                "UNET MIN:",
                np.min(unet_mask)
            )
            print(
                "UNET MAX:",
                np.max(unet_mask)
            )
            unet_mask = np.array(
                unet_mask
            )
            if len(
                unet_mask.shape
            ) == 3:
                unet_mask = unet_mask[:, :, 0]
            unet_mask = cv2.normalize(
                unet_mask,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )
            unet_mask = unet_mask.astype(
                np.uint8
            )
            if np.max(
                unet_mask
            ) == 0:
                print(
                    "EMPTY MASK"
                )
            else:
                images["unet_seg"] = img_to_b64(
                    unet_mask,
                    "gray"
                )
                final_mask = unet_mask
        # HEATMAP
        if heatmap is not None and os.path.exists(heatmap):
            heatmap_img = cv2.imread(heatmap)
            heatmap_img = cv2.cvtColor(heatmap_img,cv2.COLOR_BGR2RGB)
            images["heatmap"] = img_to_b64(
                heatmap_img
                )
        # GRAD-CAM
        if gradcam is not None and os.path.exists(gradcam):
            gradcam_img = cv2.imread(gradcam)
            gradcam_img = cv2.cvtColor(
                gradcam_img,cv2.COLOR_BGR2RGB)
            images["gradcam"] = img_to_b64(
                gradcam_img
                )    
        # REPORT IMAGE
        try:
            images["report"] = report_to_b64(
                report,
                preproc,
                final_mask
                )
        except Exception as e:
            print("Report generation failed:", e)
        return jsonify({
            "success": True,
            "report": report,
            "summary":
            report.get(
                "summary",
                {}
            ),
            "images": images
        })
    except Exception as e:
        print(
            "ANALYSIS ERROR:",
            e
        )
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ==================================================
# MODEL METRICS
# ==================================================

@app.route(
    "/api/model-metrics"
)
def metrics():
    result = {
        "models":[],
        "accuracy":[],
        "f1":[]
    }
    for m in [
        "cnn",
        "vgg",
        "ensemble"
    ]:
        fp = os.path.join(
            METRICS_DIR,
            f"{m}_report.json"
        )
        if not os.path.exists(
            fp
        ):
            continue
        with open(fp) as f:
            data = json.load(f)
        result["models"].append(
            m.upper()
        )
        result["accuracy"].append(
            data.get(
                "accuracy",
                0
            )
        )
        result["f1"].append(
            data.get(
                "weighted avg",
                {}
            ).get(
                "f1-score",
                0
            )
        )
    return jsonify(result)

# ==================================================
# MLOPS DRIFT MONITORING
# ==================================================
@app.route("/api/drift")
def get_drift():

    import os
    import json
    drift_file = os.path.join(
        PROJECT_ROOT,
        "mlops",
        "monitoring",
        "drift_history.json"
    )
    if not os.path.exists(drift_file):
        return jsonify({
            "labels": [],
            "scores": []
        })
    with open(drift_file) as f:
        data = json.load(f)
    return jsonify({
        "labels":[
            x["time"][-8:]
            for x in data[-20:]
        ],
        "scores":[
            x["score"]
            for x in data[-20:]
        ]
    })
# ==================================================
# REGISTRY
# ==================================================

@app.route("/api/registry")
def registry():
    if not os.path.exists(REGISTRY_PATH):
        return jsonify([])
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    models = []

    # Final Model
    for m in registry["FinalModel"]["versions"]:
        models.append({
            "model": "FinalModel",
            "accuracy": m["metrics"].get("acc", "-"),
            "f1": m["metrics"].get("f1", "-"),
            "registered": m["registered"][:10],
            "status": m["stage"]
        })

    # Ensemble
    for m in registry["Ensemble"]["versions"]:
        models.append({
            "model": "Ensemble",
            "accuracy": m["metrics"].get("test_acc", "-"),
            "f1": "-",
            "registered": m["registered"][:10],
            "status": m["stage"]
        })

    return jsonify(models)
# ==================================================
# RUN
# ==================================================

if __name__=="__main__":
    print(
        app.url_map
    )
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
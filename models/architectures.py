"""
Cervical Spine – Model Architectures (FINAL STABLE FIX)
"""
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import tensorflow as tf
import matplotlib.cm as cm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
# SAFE PCA HELPER
# ─────────────────────────────────────────────
def safe_pca_components(X, requested):
    """
    Automatically adjust PCA components to avoid crash.
    """
    n_samples, n_features = X.shape
    safe_n = min(requested, n_samples - 1, n_features)
    return max(2, safe_n)  # ensure >=2


# ═════════════════════════════════════════════
# 1. CNN PROXY
# ═════════════════════════════════════════════
class CNNClassifier:
    def __init__(self, C=1.0, gamma='scale'):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, kernel='rbf', gamma=gamma,
                        probability=True, random_state=42)),
        ])
        self.name = "CNN_Proxy_SVM"

    def fit(self, X, y):
        X_flat = X.reshape(len(X), -1)
        self.pipeline.fit(X_flat, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X.reshape(len(X), -1))

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X.reshape(len(X), -1))
    
class RealCNNClassifier:

    def __init__(self):

        self.num_classes = None
        self.model = None

        self.model = Sequential([
            Conv2D(
                32,
                (3,3),
                activation="relu",
                input_shape=(128,128,1)
            ),
            MaxPooling2D((2,2)),

            Conv2D(
                64,
                (3,3),
                activation="relu"
            ),
            MaxPooling2D((2,2)),

            Conv2D(
                128,
                (3,3),
                activation="relu"
            ),
            MaxPooling2D((2,2)),

            Flatten(),

            Dense(
                128,
                activation="relu"
            ),

            Dropout(0.3),
        ])

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.model = Sequential([
            Input(shape=(128,128,1)),
            Conv2D(32,(3,3),activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Conv2D(64,(3,3),activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Conv2D(128,(3,3),activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(256,activation="relu"),
            Dropout(0.5),
            Dense(128,activation="relu"),
            Dropout(0.3),
            Dense(self.num_classes,activation="softmax")
            ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )
        X = X.astype(np.float32)
        if X.max() > 1:
            X /= 255.0
        if X.ndim == 3:
            X = X[..., np.newaxis]
        print("CNN Input:", X.shape)
        print("CNN Labels:", np.bincount(y))
        y_cat = to_categorical(
            y,
            num_classes=self.num_classes
            )
        self.model.fit(
            X,
            y_cat,
            epochs=30,
            batch_size=16,
            validation_split=0.15,
            shuffle=True,
            verbose=1
            )
        return self

    def predict(self, X):
        X = X.astype(np.float32)
        if X.max() > 1:
            X = X / 255.0
        if X.ndim==3:
            X = X[...,np.newaxis]
        probs = self.model.predict(
            X,
            verbose=0
        )
        return np.argmax(
            probs,
            axis=1
        )

    def predict_proba(self, X):
        X = X.astype(np.float32)
        if X.max() > 1:
            X = X / 255.0
        if X.ndim==3:
            X = X[...,np.newaxis]    
        return self.model.predict(
            X,
            verbose=0
        )

    def save(self, folder):
        os.makedirs(folder,exist_ok=True)
        self.model.save(
            os.path.join(folder,"cnn.h5")
            )
    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        return self

# ═════════════════════════════════════════════
# 2. VGG PROXY 
# ═════════════════════════════════════════════
class VGGNetClassifier:
    def __init__(self, n_components=128, hidden_sizes=(256, 256, 128)):
        self.requested_components = n_components
        self.hidden_sizes = hidden_sizes
        self.pipeline = None
        self.name = "VGGNet_Proxy_MLP"

    def fit(self, X, y):
        X_flat = X.reshape(len(X), -1)

        #  AUTO FIX PCA
        n_comp = safe_pca_components(X_flat, self.requested_components)

        print(f"VGG PCA auto-adjusted → {n_comp}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=42)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=self.hidden_sizes,
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ])

        self.pipeline.fit(X_flat, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X.reshape(len(X), -1))

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X.reshape(len(X), -1))


# ═════════════════════════════════════════════
# 3. U-NET PROXY
# ═════════════════════════════════════════════
class UNetSegmenter:
    def __init__(self, morph_iter=2, min_area=200):
        self.morph_iter = morph_iter
        self.min_area = min_area

    def segment(self, gray):
        from skimage.filters import threshold_otsu
        from skimage import measure

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        t = threshold_otsu(blurred)
        binary = (blurred > t).astype(np.uint8)

        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        mask = np.zeros_like(gray)
        valid = []

        for r in regions:
            if r.area >= self.min_area:
                mask[labeled == r.label] = 255
                valid.append(r.area)

        return mask, len(valid)


# ═════════════════════════════════════════════
# 4. VAE PROXY 
# ═════════════════════════════════════════════
class VAEProxy:
    def __init__(self, latent_dim=32):
        self.latent_dim = latent_dim
        self.pca = None
        self.scaler = StandardScaler()

    def fit(self, X):
        X_flat = X.reshape(len(X), -1)

        #  SAFE PCA
        n_comp = safe_pca_components(X_flat, self.latent_dim)
        print(f"VAE PCA auto-adjusted → {n_comp}")

        self.pca = PCA(n_components=n_comp, random_state=42)

        X_sc = self.scaler.fit_transform(X_flat)
        self.pca.fit(X_sc)
        return self

    def reconstruction_error(self, X):
        X_flat = X.reshape(len(X), -1)
        X_sc = self.scaler.transform(X_flat)
        z = self.pca.transform(X_sc)
        recon = self.pca.inverse_transform(z)
        return np.mean((X_sc - recon) ** 2, axis=1)


# ═════════════════════════════════════════════
# ENSEMBLE 
# ═════════════════════════════════════════════
class EnsembleClassifier:
    def __init__(self):
        self.cnn = RealCNNClassifier()
        self.vgg = VGGNetClassifier()
        self.gbm = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_flat = X.reshape(len(X), -1)

        # 🔥 SAFE PCA for GBM
        n_comp = safe_pca_components(X_flat, 64)
        print(f"GBM PCA auto-adjusted → {n_comp}")

        self.gbm = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=42)),
            ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ])

        self.cnn.fit(X, y)
        self.vgg.fit(X, y)
        self.gbm.fit(X_flat, y)

        return self

    def predict_proba(self, X):
        X_flat = X.reshape(len(X), -1)

        p1 = self.cnn.predict_proba(X)
        p2 = self.vgg.predict_proba(X)
        p3 = self.gbm.predict_proba(X_flat)

        return (p1 + p2 + p3) / 3

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ═════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, label_map):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nModel:", model.__class__.__name__)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    return acc

# ==========================================================
# GRAD-CAM
# ==========================================================

def generate_gradcam(model, image, class_index):
    """
    Generate Grad-CAM visualization.

    Parameters
    ----------
    model : tf.keras.Model
        Trained CNN model

    image : numpy array
        Shape = (1,128,128,1)

    class_index : int
        Predicted class

    Returns
    -------
    RGB GradCAM image
    """

    # Last convolution layer
    last_conv_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image)

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-10

    heatmap = heatmap.numpy()

    heatmap = cv2.resize(
        heatmap,
        (128,128)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original = image[0,:,:,0]

    original = np.uint8(
        original * 255
    )

    original = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2RGB
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    gradcam = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    return gradcam
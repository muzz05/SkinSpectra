import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.applications import EfficientNetV2B2  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore
from sklearn.utils.class_weight import compute_class_weight

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = str(_ROOT / "models" / "facial_analysis")

CFG = {
    "img_size":        260,
    "batch_size":      16,
    "phase1_epochs":   30,
    "phase2_epochs":   30,
    "phase3_epochs":   20,
    "num_classes":     3,
    "model_dir":       _MODEL_DIR,
    "weights_path":    str(_ROOT / "models" / "facial_analysis" / "efficient_net_v2_b2.weights.h5"),
    "model_path":      str(_ROOT / "models" / "facial_analysis" / "skin_type_final_v2b2.h5"),
    "class_names":     {0: "Dry", 1: "Normal", 2: "Oily"},
    "dataset_name":    "shakyadissanayake/oily-dry-and-normal-skin-types-dataset",
    "dataset_subdir":  "Oily-Dry-Skin-Types",
    "label_smoothing_coarse": 0.1,
    "label_smoothing_fine":   0.05,
    "fine_tune_ratio": 0.7,
}

def detect_and_crop_face(image_path: str) -> np.ndarray | None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[facial_analysis] Error: could not load image from {image_path}")
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    if len(faces) == 0:
        print("[facial_analysis] No face detected. Use a clear, front-facing photo.")
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.1 * w)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)
    return image[y : y + h, x : x + w]


def preprocess_face(face_image: np.ndarray, img_size: int) -> np.ndarray:
    face_resized      = cv2.resize(face_image, (img_size, img_size))
    face_rgb          = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_preprocessed = preprocess_input(face_rgb)
    return np.expand_dims(face_preprocessed, axis=0)

def build_model(img_size: int, num_classes: int):
    base = EfficientNetV2B2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    for units, drop in [(1024, 0.5), (512, 0.4), (256, 0.3)]:
        x = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(drop)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model   = keras.Model(inputs=base.input, outputs=outputs)
    return model, base

def build_data_generators(dataset_path: str, img_size: int, batch_size: int):
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")
    use_split  = not os.path.exists(valid_path)

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2 if use_split else 0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_gen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training" if use_split else None,
        shuffle=True,
    )
    val_data = (train_gen if use_split else valid_gen).flow_from_directory(
        train_path if use_split else valid_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation" if use_split else None,
        shuffle=False,
    )

    labels        = train_data.classes
    weights       = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))

    print(f"[facial_analysis] Classes : {train_data.class_indices}")
    print(f"[facial_analysis] Train   : {train_data.samples} samples")
    print(f"[facial_analysis] Val     : {val_data.samples} samples")
    print(f"[facial_analysis] Weights : {class_weights}")
    return train_data, val_data, class_weights

def _make_callbacks(weights_path: str) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
    ]


def _report_best(history, phase_name: str) -> None:
    if not history or "val_accuracy" not in history.history:
        return
    best = int(np.argmax(history.history["val_accuracy"]))
    print(
        f"  {phase_name}: best epoch {best + 1}  "
        f"train={history.history['accuracy'][best]*100:.2f}%  "
        f"val={history.history['val_accuracy'][best]*100:.2f}%"
    )


def run_training(dataset_path: str, cfg: dict = CFG) -> None:
    np.random.seed(42)
    tf.random.set_seed(42)

    Path(cfg["model_dir"]).mkdir(parents=True, exist_ok=True)

    train_data, val_data, class_weights = build_data_generators(
        dataset_path, cfg["img_size"], cfg["batch_size"]
    )

    model, base_model = build_model(cfg["img_size"], cfg["num_classes"])
    print(f"[facial_analysis] Total parameters: {model.count_params():,}")

    callbacks     = _make_callbacks(cfg["weights_path"])
    ls_coarse     = cfg["label_smoothing_coarse"]
    ls_fine       = cfg["label_smoothing_fine"]
    fit_kwargs    = dict(
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\n" + "=" * 60)
    print("PHASE 1: Training top layers only")
    print("=" * 60)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=ls_coarse),
        metrics=["accuracy"],
    )
    h1 = model.fit(train_data, epochs=cfg["phase1_epochs"], **fit_kwargs)
    model.load_weights(cfg["weights_path"])

    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning top 30 % of base model")
    print("=" * 60)
    base_model.trainable = True
    freeze_until = int(len(base_model.layers) * cfg["fine_tune_ratio"])
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"[facial_analysis] Trainable layers: {trainable}/{len(base_model.layers)}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=ls_coarse),
        metrics=["accuracy"],
    )
    h2 = model.fit(train_data, epochs=cfg["phase2_epochs"], **fit_kwargs)
    model.load_weights(cfg["weights_path"])

    print("\n" + "=" * 60)
    print("PHASE 3: Fine-tuning entire model")
    print("=" * 60)
    for layer in base_model.layers:
        layer.trainable = True
    print(f"[facial_analysis] All {len(base_model.layers)} base layers trainable")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=ls_fine),
        metrics=["accuracy"],
    )
    h3 = model.fit(train_data, epochs=cfg["phase3_epochs"], **fit_kwargs)
    model.load_weights(cfg["weights_path"])

    model.save(cfg["model_path"])
    print(f"\n[facial_analysis] Model saved → {cfg['model_path']}")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    _, train_acc = model.evaluate(train_data, verbose=0)
    _, val_acc   = model.evaluate(val_data, verbose=0)
    print(f"  Training accuracy  : {train_acc * 100:.2f}%")
    print(f"  Validation accuracy: {val_acc * 100:.2f}%")

    print("\nTRAINING SUMMARY")
    _report_best(h1, "Phase 1 (top layers)")
    _report_best(h2, "Phase 2 (30 % fine-tune)")
    _report_best(h3, "Phase 3 (full fine-tune)")

class FacialAnalyzer:

    def __init__(self, model, cfg: dict = CFG):
        self.model = model
        self.cfg   = cfg

    def predict(self, image_path: str) -> dict | None:
        t0          = time.perf_counter()
        class_names = self.cfg["class_names"]
        img_size    = self.cfg["img_size"]

        cropped = detect_and_crop_face(image_path)
        if cropped is None:
            return None

        face_input  = preprocess_face(cropped, img_size)
        predictions = self.model.predict(face_input, verbose=0)

        pred_idx    = int(np.argmax(predictions[0]))
        confidence  = float(predictions[0][pred_idx])

        return {
            "skin_type":        class_names[pred_idx],
            "confidence":       round(confidence, 4),
            "all_probabilities": {
                class_names[i]: round(float(predictions[0][i]), 4)
                for i in range(len(class_names))
            },
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def batch_predict(self, image_paths: list[str]) -> list[dict | None]:
        return [self.predict(p) for p in image_paths]

    @staticmethod
    def display_result(result: dict | None) -> None:
        if result is None:
            return
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"\n  Skin type  : {result['skin_type']}")
        print(f"  Confidence : {result['confidence'] * 100:.2f}%")
        print(f"  Latency    : {result['latency_ms']} ms")
        print("\n  Detailed probabilities:")
        for skin_type, prob in result["all_probabilities"].items():
            bar = "█" * int(prob * 40) + "░" * (40 - int(prob * 40))
            print(f"    {skin_type:8s} {bar} {prob * 100:5.2f}%")
        print("=" * 60)

    @classmethod
    def load(cls, model_dir: str | None = None, cfg: dict = CFG) -> "FacialAnalyzer":
        import h5py
        model_dir    = model_dir or cfg["model_dir"]
        weights_path = os.path.join(model_dir, "efficient_net_v2_b2.weights.h5")
        print("[facial_analysis] Building architecture ...")
        model, _     = build_model(cfg["img_size"], cfg["num_classes"])
        print(f"[facial_analysis] Loading weights from {weights_path}")
        with h5py.File(weights_path, "r") as f:
            layer_names_raw = f.attrs.get("layer_names", [])
            layer_names = [
                n.decode("utf-8") if isinstance(n, bytes) else n
                for n in layer_names_raw
            ]
            layer_map = {l.name: l for l in model.layers}
            loaded = skipped = 0
            for lname in layer_names:
                if lname not in layer_map or lname not in f:
                    continue
                g = f[lname]
                weight_names = [
                    wn.decode("utf-8") if isinstance(wn, bytes) else wn
                    for wn in g.attrs.get("weight_names", [])
                ]
                if not weight_names:
                    continue
                weight_values = [np.array(g[wn]) for wn in weight_names]
                try:
                    layer_map[lname].set_weights(weight_values)
                    loaded += 1
                except Exception:
                    skipped += 1
        print(f"[facial_analysis] Weights loaded: {loaded} layers ({skipped} skipped)")
        print("[facial_analysis] Model ready")
        return cls(model, cfg)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkinSpectra Facial Analysis — train or predict"
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Download dataset and train the model")
    train_p.add_argument("--dataset-path", default=None)
    train_p.add_argument("--model-dir",      default=CFG["model_dir"])
    train_p.add_argument("--batch-size",     type=int, default=CFG["batch_size"])
    train_p.add_argument("--phase1-epochs",  type=int, default=CFG["phase1_epochs"])
    train_p.add_argument("--phase2-epochs",  type=int, default=CFG["phase2_epochs"])
    train_p.add_argument("--phase3-epochs",  type=int, default=CFG["phase3_epochs"])

    pred_p = sub.add_parser("predict", help="Predict skin type from an image")
    pred_p.add_argument("image")
    pred_p.add_argument("--model-dir", default=CFG["model_dir"])

    args = parser.parse_args()

    if args.command == "train":
        cfg = {**CFG,
               "model_dir":      args.model_dir,
               "weights_path":   os.path.join(args.model_dir, "efficient_net_v2_b2.weights.h5"),
               "model_path":     os.path.join(args.model_dir, "skin_type_final_v2b2.h5"),
               "batch_size":     args.batch_size,
               "phase1_epochs":  args.phase1_epochs,
               "phase2_epochs":  args.phase2_epochs,
               "phase3_epochs":  args.phase3_epochs}

        if args.dataset_path:
            dataset_path = args.dataset_path
        else:
            try:
                import kagglehub
            except ImportError:
                print(
                    "[facial_analysis] kagglehub not installed. "
                    "Run: pip install kagglehub  or pass --dataset-path."
                )
                sys.exit(1)
            print("[facial_analysis] Downloading dataset via kagglehub …")
            raw_path     = kagglehub.dataset_download(cfg["dataset_name"])
            dataset_path = os.path.join(raw_path, cfg["dataset_subdir"])
            if not os.path.exists(dataset_path):
                dataset_path = os.path.join(raw_path, os.listdir(raw_path)[0])

        run_training(dataset_path, cfg)

    elif args.command == "predict":
        cfg      = {**CFG, "model_dir": args.model_dir}
        analyzer = FacialAnalyzer.load(cfg=cfg)
        result   = analyzer.predict(args.image)
        FacialAnalyzer.display_result(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

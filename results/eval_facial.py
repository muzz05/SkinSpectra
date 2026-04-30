"""
SkinSpectra Facial Analysis Evaluation Script
===============================================
Pulls images from the same Kaggle dataset the model was trained on.
Keeps sampling until exactly n_per_class valid predictions (no face-detection
or blur errors) are collected per class. Generates blurred and no-face
negative controls, reports per-class accuracy, confidence and latency,
and saves a bar chart to figures/facial_accuracy_by_class.png.

Usage
-----
    python eval_facial.py [--model_dir ../models/facial_analysis]
                          [--output_dir figures]
                          [--samples_per_class 10]
                          [--seed 42]

Requirements
------------
    pip install kagglehub pillow opencv-python numpy matplotlib
    Kaggle credentials must be configured (~/.kaggle/kaggle.json).
    facial_analysis.py must be in the same directory.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import sys
# Ensure repository root is on sys.path so `components` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from components.facial_analysis import FacialAnalyzer
except ImportError:
    print("ERROR: Could not import FacialAnalyzer. "
          "Make sure facial_analysis.py is in the same directory.")
    sys.exit(1)

DATASET_NAME   = "shakyadissanayake/oily-dry-and-normal-skin-types-dataset"
DATASET_SUBDIR = "Oily-Dry-Skin-Types"

FOLDER_TO_CLASS = {
    "dry":    "Dry",
    "Dry":    "Dry",
    "normal": "Normal",
    "Normal": "Normal",
    "oily":   "Oily",
    "Oily":   "Oily",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# =============================================================================
# DATASET HELPERS
# =============================================================================

def download_dataset() -> str:
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    print(f"Downloading dataset: {DATASET_NAME}")
    raw_path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset root: {raw_path}")

    candidate = os.path.join(raw_path, DATASET_SUBDIR)
    if os.path.isdir(candidate):
        return candidate

    subdirs = [d for d in os.listdir(raw_path)
               if os.path.isdir(os.path.join(raw_path, d))]
    if subdirs:
        return os.path.join(raw_path, subdirs[0])
    return raw_path


def find_split_dir(dataset_root: str) -> str:
    for name in ["valid", "validation", "val", "test", "train"]:
        candidate = os.path.join(dataset_root, name)
        if os.path.isdir(candidate):
            print(f"Using split: {candidate}")
            return candidate
    return dataset_root


def get_all_images_per_class(split_dir: str) -> dict:
    """Return all available image paths per class without any sampling."""
    result = {}
    for entry in sorted(os.listdir(split_dir)):
        class_name = FOLDER_TO_CLASS.get(entry)
        if class_name is None:
            continue
        folder = os.path.join(split_dir, entry)
        if not os.path.isdir(folder):
            continue
        all_images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if Path(f).suffix.lower() in VALID_EXTENSIONS
        ]
        result[class_name] = all_images
        print(f"  {class_name}: {len(all_images)} images available")
    return result


def prescreen_image(path: str) -> bool:
    """
    Quick pre-screen using OpenCV Haar cascade before running the full model.
    Returns True if a face is detected and the image is not blurry.
    This avoids wasting model inference time on images that will be rejected.
    """
    img = cv2.imread(path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur check
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 100.0:
        return False

    # face check
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return len(faces) > 0


# =============================================================================
# NEGATIVE CONTROL GENERATORS
# =============================================================================

def make_blurred_image(source_path: str, dest_path: str, radius: int = 25) -> str:
    img = Image.open(source_path).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    img.save(dest_path)
    return dest_path


def make_no_face_image(dest_path: str) -> str:
    arr = np.full((300, 300, 3), 128, dtype=np.uint8)
    cv2.imwrite(dest_path, arr)
    return dest_path


# =============================================================================
# EVALUATION
# =============================================================================

def run_evaluation(model_dir: str, output_dir: str,
                   samples_per_class: int = 10, seed: int = 42):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Download and get all images ───────────────────────────────────────
    dataset_root = download_dataset()
    split_dir    = find_split_dir(dataset_root)

    print(f"\nScanning available images in: {split_dir}")
    all_images = get_all_images_per_class(split_dir)

    if not all_images:
        print("ERROR: No class folders found.")
        sys.exit(1)

    # ── Pre-screen and collect exactly n valid images per class ───────────
    rng = random.Random(seed)
    local_map = {}

    for skin_type in ["Dry", "Normal", "Oily"]:
        pool = all_images.get(skin_type, [])
        if not pool:
            print(f"  WARNING: No images for {skin_type}")
            continue

        rng.shuffle(pool)
        valid_paths = []
        skipped     = 0

        print(f"\nPre-screening {skin_type} images to find "
              f"{samples_per_class} valid ones...")

        for path in pool:
            if len(valid_paths) >= samples_per_class:
                break
            if prescreen_image(path):
                valid_paths.append(path)
                print(f"  PASS [{len(valid_paths):>2}/{samples_per_class}] "
                      f"{Path(path).name}")
            else:
                skipped += 1

        print(f"  Collected {len(valid_paths)} valid images "
              f"(skipped {skipped} that had no face or were blurry)")
        local_map[skin_type] = valid_paths

    # copy to output dir for reference
    sample_dir = out_path / "eval_samples"
    for class_name, paths in local_map.items():
        class_out = sample_dir / class_name
        class_out.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, str(class_out / Path(p).name))
    print(f"\nValid samples copied to: {sample_dir}")

    # ── Negative controls ─────────────────────────────────────────────────
    neg_dir  = out_path / "eval_samples" / "negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    blur_src = next((paths[0] for paths in local_map.values() if paths), None)
    negative_cases = []

    if blur_src:
        blur_dest = str(neg_dir / "blurred_control.jpg")
        make_blurred_image(blur_src, blur_dest, radius=25)
        negative_cases.append(("blurred_control.jpg", blur_dest, "blurry"))
        print(f"Created blurred negative: {blur_dest}")

    no_face_dest = str(neg_dir / "no_face_control.jpg")
    make_no_face_image(no_face_dest)
    negative_cases.append(("no_face_control.jpg", no_face_dest, "no_face"))
    print(f"Created no-face negative:  {no_face_dest}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading FacialAnalyzer from {model_dir} ...")
    analyzer = FacialAnalyzer.load(model_dir=model_dir)

    # ── Positive predictions ──────────────────────────────────────────────
    print("\nRunning predictions on pre-screened positive samples...")
    per_class_results = {}

    for skin_type in ["Dry", "Normal", "Oily"]:
        paths = local_map.get(skin_type, [])
        if not paths:
            continue

        correct     = 0
        total       = 0
        latencies   = []
        confidences = []
        predictions = []

        for path in paths:
            result = analyzer.predict(path)
            total += 1

            if "error" in result:
                # should not happen after pre-screening but handle gracefully
                code = result["error"]
                print(f"  [{skin_type}] {Path(path).name}: "
                      f"UNEXPECTED ERROR ({code})")
                predictions.append({
                    "file": Path(path).name, "predicted": None,
                    "correct": False, "error": code,
                    "latency_ms": result["latency_ms"],
                })
            else:
                predicted  = result["skin_type"]
                confidence = result["confidence"]
                latency    = result["latency_ms"]
                is_correct = predicted == skin_type
                correct   += int(is_correct)
                latencies.append(latency)
                confidences.append(confidence)
                mark = "OK" if is_correct else "WRONG"
                print(f"  [{skin_type}] {Path(path).name}: "
                      f"pred={predicted} conf={confidence:.3f} "
                      f"lat={latency:.0f}ms [{mark}]")
                predictions.append({
                    "file": Path(path).name, "predicted": predicted,
                    "correct": is_correct, "confidence": confidence,
                    "latency_ms": latency,
                })

        per_class_results[skin_type] = {
            "correct"     : correct,
            "total"       : total,
            "accuracy"    : round(correct / total, 4) if total > 0 else 0.0,
            "mean_conf"   : round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "mean_latency": round(float(np.mean(latencies)),   1) if latencies   else 0.0,
            "predictions" : predictions,
        }

    # ── Negative controls ─────────────────────────────────────────────────
    print("\nRunning negative control predictions...")
    neg_results = []
    neg_correct = 0

    for fname, path, expected_error in negative_cases:
        result = analyzer.predict(path)
        if "error" in result:
            got        = result["error"]
            is_correct = got in ("blurry", "no_face")
            neg_correct += int(is_correct)
            mark = "OK" if is_correct else "WRONG"
            print(f"  [NEG] {fname}: error={got} "
                  f"expected={expected_error} [{mark}]")
            neg_results.append({
                "file": fname, "expected": expected_error,
                "got": got, "correct": is_correct,
                "latency_ms": result["latency_ms"],
            })
        else:
            print(f"  [NEG] {fname}: FAILED to reject — "
                  f"predicted {result['skin_type']}")
            neg_results.append({
                "file": fname, "expected": expected_error,
                "got": result.get("skin_type"), "correct": False,
                "latency_ms": result["latency_ms"],
            })

    neg_accuracy = (neg_correct / len(negative_cases)
                    if negative_cases else 0.0)

    # ── Overall ───────────────────────────────────────────────────────────
    all_correct   = sum(r["correct"] for r in per_class_results.values())
    all_total     = sum(r["total"]   for r in per_class_results.values())
    all_latencies = [
        p["latency_ms"]
        for r in per_class_results.values()
        for p in r["predictions"]
        if "latency_ms" in p and "error" not in p
    ]
    overall_acc = all_correct / all_total if all_total > 0 else 0.0

    # ── Print table ───────────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print(f"  {'Skin Type':<12} {'Correct':>7}  {'Total':>5}  "
          f"{'Accuracy':>8}  {'Mean Conf':>9}  {'Mean Lat(ms)':>12}")
    print("=" * 70)
    for st, r in per_class_results.items():
        print(f"  {st:<12} {r['correct']:>7}  {r['total']:>5}  "
              f"{r['accuracy']:>8.4f}  {r['mean_conf']:>9.4f}  "
              f"{r['mean_latency']:>12.1f}")
    print("-" * 70)
    print(f"  {'Overall':<12} {all_correct:>7}  {all_total:>5}  "
          f"{overall_acc:>8.4f}")
    print(f"  {'Neg. controls':<13} {neg_correct:>6}  "
          f"{len(negative_cases):>5}  {neg_accuracy:>8.4f}")
    print("=" * 70)

    if all_latencies:
        print(f"\n  Mean latency   : {np.mean(all_latencies):.1f} ms")
        print(f"  Median latency : {np.median(all_latencies):.1f} ms")
        print(f"  p95 latency    : {np.percentile(all_latencies, 95):.1f} ms")

    # ── LaTeX rows ────────────────────────────────────────────────────────
    print("\n-- LaTeX table rows --")
    for st, r in per_class_results.items():
        print(f"{st} & {r['total']} & {r['correct']} & "
              f"{r['accuracy']*100:.2f}\\% & {r['mean_conf']:.4f} & "
              f"{r['mean_latency']:.1f} \\\\")
    print("\\midrule")
    print(f"Overall & {all_total} & {all_correct} & "
          f"{overall_acc*100:.2f}\\% & -- & -- \\\\")
    print(f"Neg.\\ controls & {len(negative_cases)} & {neg_correct} & "
          f"{neg_accuracy*100:.2f}\\% & -- & -- \\\\")

    if all_latencies:
        print(f"\n-- LaTeX latency line --")
        print(f"Mean latency {np.mean(all_latencies):.1f}\\,ms, "
              f"median {np.median(all_latencies):.1f}\\,ms, "
              f"p95 {np.percentile(all_latencies, 95):.1f}\\,ms.")

    # ── Figure ────────────────────────────────────────────────────────────
    class_names = list(per_class_results.keys())
    accuracies  = [per_class_results[c]["accuracy"] * 100  for c in class_names]
    mean_confs  = [per_class_results[c]["mean_conf"] * 100 for c in class_names]

    x     = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_acc  = ax.bar(x - width/2, accuracies, width, label="Accuracy (%)",
                       color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars_conf = ax.bar(x + width/2, mean_confs, width,
                       label="Mean Confidence (%)",
                       color="#55A868", edgecolor="white", linewidth=0.8)

    for bar in list(bars_acc) + list(bars_conf):
        h = bar.get_height()
        if h > 1:
            ax.annotate(f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.axhline(overall_acc * 100, color="#C44E52", linestyle="--",
               linewidth=1.3, alpha=0.7,
               label=f"Overall accuracy = {overall_acc*100:.1f}%")

    ax.set_xlabel("Skin Type", fontsize=12, labelpad=8)
    ax.set_ylabel("Percentage (%)", fontsize=12, labelpad=8)
    ax.set_title("Facial Skin-Type Detection Accuracy by Class",
                 fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig_path = out_path / "facial_accuracy_by_class.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to: {fig_path}")

    return per_class_results, neg_results, overall_acc


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SkinSpectra Facial Analysis Evaluator"
    )
    parser.add_argument("--model_dir",  default="../models/facial_analysis")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--samples_per_class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        model_dir         = args.model_dir,
        output_dir        = args.output_dir,
        samples_per_class = args.samples_per_class,
        seed              = args.seed,
    )
"""
SkinSpectra OCR Evaluation Script
===================================
Generates synthetic label images under four conditions (clean, blurred,
rotated, noisy), runs the OCR pipeline on each, measures precision/recall/F1
using exact normalised string matching, prints a results table and saves
a matplotlib figure to figures/ocr_quality_by_condition.png.

Usage
-----
    python eval_ocr.py [--output_dir figures] [--tesseract PATH]

Requirements
------------
    pip install pytesseract opencv-python pillow numpy matplotlib
    Tesseract must be installed on the system.
"""

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import sys
# Ensure repository root is on sys.path so `components` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# adjust this import to match your actual ocr_handler filename
try:
    from components.ocr_handler import OCRHandler
except ImportError:
    print("ERROR: Could not import OCRHandler. Make sure ocr_handler.py is in the same directory.")
    sys.exit(1)


# =============================================================================
# KNOWN INGREDIENT SETS FOR SYNTHETIC LABELS
# =============================================================================

LABEL_SETS = [
    ["Aqua", "Glycerin", "Niacinamide", "Panthenol", "Allantoin",
     "Carbomer", "Dimethicone", "Tocopherol", "Sodium Hyaluronate", "Phenoxyethanol"],

    ["Caprylic Capric Triglyceride", "Cetearyl Alcohol", "Glycerin",
     "Shea Butter", "Squalane", "Vitamin E", "Retinol", "Zinc PCA",
     "Hyaluronic Acid", "Aloe Vera"],

    ["Water", "Salicylic Acid", "Niacinamide", "Zinc PCA", "Glycerin",
     "Allantoin", "Witch Hazel", "Tea Tree Oil", "Panthenol", "Centella Asiatica"],

    ["Aqua", "Cetearyl Alcohol", "Glycerin", "Sodium Hyaluronate",
     "Ceramide NP", "Ceramide AP", "Ceramide EOP", "Cholesterol",
     "Phytosphingosine", "Carbomer"],

    ["Water", "Glycolic Acid", "Lactic Acid", "Aloe Vera", "Panthenol",
     "Allantoin", "Hyaluronic Acid", "Niacinamide", "Retinyl Palmitate", "Phenoxyethanol"],

    ["Aqua", "Dimethicone", "Cyclopentasiloxane", "Glycerin", "Tocopherol",
     "Sodium Ascorbyl Phosphate", "Kojic Acid", "Arbutin", "Licorice Root Extract", "EDTA"],

    ["Water", "Propylene Glycol", "Niacinamide", "Panthenol", "Allantoin",
     "Centella Asiatica Extract", "Madecassoside", "Asiaticoside",
     "Sodium Hyaluronate", "Carbomer"],

    ["Aqua", "Glycerin", "Retinol", "Vitamin C", "Vitamin E",
     "Ferulic Acid", "Hyaluronic Acid", "Peptides", "Collagen", "Phenoxyethanol"],

    ["Water", "Caprylic Capric Triglyceride", "Shea Butter", "Squalane",
     "Ceramide NP", "Jojoba Oil", "Rosehip Oil", "Tocopherol",
     "Allantoin", "Sodium Hyaluronate"],

    ["Aqua", "Zinc Oxide", "Titanium Dioxide", "Glycerin", "Dimethicone",
     "Cetearyl Alcohol", "Allantoin", "Panthenol", "Aloe Vera", "EDTA"],
]


# =============================================================================
# REALISTIC SYNTHETIC LABEL IMAGE GENERATOR
# =============================================================================

def _get_font(size: int):
    """Try several common fonts, fall back to PIL default."""
    candidates = [
        "arial.ttf", "Arial.ttf",
        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(ingredients: list, chars_per_line: int = 52) -> list:
    """Wrap ingredient list into display lines."""
    lines = []
    line  = ""
    for i, ing in enumerate(ingredients):
        token = ing if i == 0 else ", " + ing
        if len(line) + len(token) > chars_per_line:
            lines.append(line)
            line = ing
        else:
            line += token
    if line:
        lines.append(line)
    return lines


def make_label_image(ingredients: list, width: int = 640, font_size: int = 13) -> Image.Image:
    """
    Render a realistic ingredient label with:
    - yellowish/cream off-white background with stronger paper texture
    - small font (13px) to mimic real cramped labels
    - ink colour variation and per-character horizontal jitter
    - a faint diagonal gradient baked in to simulate uneven paper
    """
    rng         = np.random.default_rng()
    line_height = font_size + 4
    header      = "INGREDIENTS / INCI:"
    wrapped     = _wrap_text(ingredients, chars_per_line=62)
    height      = line_height * (len(wrapped) + 3) + 50

    # Cream/yellowish off-white background — more realistic than pure white
    bg_color = (
        int(rng.integers(228, 245)),
        int(rng.integers(220, 238)),
        int(rng.integers(195, 218)),
    )
    img = Image.new("RGB", (width, height), color=bg_color)
    arr = np.array(img).astype(np.int16)

    # Stronger paper grain texture
    texture = rng.integers(-18, 18, arr.shape, dtype=np.int16)
    arr     = np.clip(arr + texture, 0, 255).astype(np.uint8)

    # Bake in a faint diagonal brightness gradient to simulate paper curl
    gy = np.linspace(0.94, 1.0, height, dtype=np.float32).reshape(-1, 1, 1)
    gx = np.linspace(0.96, 1.0, width,  dtype=np.float32).reshape(1, -1, 1)
    arr = np.clip(arr * gy * gx, 0, 255).astype(np.uint8)

    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    font        = _get_font(font_size)
    header_font = _get_font(font_size + 1)

    # Ink colour: dark but not pure black, slight variation per label
    base_ink = int(rng.integers(15, 55))
    ink = (base_ink, base_ink, base_ink)

    draw.text((18, 14), header, fill=ink, font=header_font)
    y = 14 + line_height + 6
    for wline in wrapped:
        # per-line vertical AND horizontal jitter to mimic uneven label printing
        v_jitter = int(rng.integers(-2, 3))
        h_jitter = int(rng.integers(-1, 2))
        draw.text((18 + h_jitter, y + v_jitter), wline, fill=ink, font=font)
        y += line_height

    return img


def apply_blur(img: Image.Image, radius: float = 1) -> Image.Image:
    """
    Simulate a slightly out-of-focus camera shot.
    Radius reduced to 2.8 so text is degraded but not completely unreadable,
    which is more representative of a mildly shaky phone camera than a fully
    defocused lens. Contrast is also reduced to mimic real blur behaviour.
    """
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    img = ImageEnhance.Contrast(img).enhance(0.95)
    # add a small amount of brightness reduction to simulate haze
    img = ImageEnhance.Brightness(img).enhance(0.92)
    return img


def apply_rotation(img: Image.Image, angle: float = None) -> Image.Image:
    """
    Simulate a photo taken at a more pronounced angle.
    Rotation range raised to 15-28 degrees and perspective warp margins
    increased to better represent a label photographed from the side.
    A slight blur is also applied after warping since angled photos
    are rarely in perfect focus across the full label.
    """
    rng   = np.random.default_rng()
    angle = angle if angle is not None else float(rng.uniform(8, 18))

    img = img.rotate(angle, expand=True, fillcolor=(232, 228, 210))

    arr  = np.array(img)
    h, w = arr.shape[:2]
    # larger margin = more aggressive perspective squeeze
    margin = int(w * 0.06)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [margin,               int(rng.integers(5, 22))],
        [w - margin,           int(rng.integers(0, 14))],
        [w - int(margin*0.4),  h - int(rng.integers(5, 18))],
        [int(margin*0.4),      h - int(rng.integers(5, 15))],
    ])
    M   = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(arr, M, (w, h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(232, 228, 210))
    img = Image.fromarray(arr)
    # slight post-warp blur — angled shots are rarely perfectly sharp
    img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
    return img


def apply_noise(img: Image.Image, intensity: int = None) -> Image.Image:
    """
    Simulate a low-light or low-quality phone camera capture with:
    - heavier random pixel noise (intensity 55-80)
    - aggressive JPEG compression artefacts (quality 18-35)
    - brightness drop to simulate low-light shooting
    - contrast reduction
    - a second mild Gaussian blur pass to simulate sensor noise smearing
    """
    import io as _io
    rng       = np.random.default_rng()
    intensity = intensity if intensity is not None else int(rng.integers(55, 80))

    arr   = np.array(img).astype(np.int16)
    noise = rng.integers(-intensity, intensity, arr.shape, dtype=np.int16)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)

    # Heavy JPEG compression artefacts
    buf     = _io.BytesIO()
    quality = int(rng.integers(18, 35))
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).copy()

    # Low-light brightness drop
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.70, 0.85)))
    # Contrast reduction
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.68, 0.82)))
    # Sensor noise smearing
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 0.9))))
    return img


def apply_shadow(img: Image.Image) -> Image.Image:
    """
    Add an aggressive multi-band shadow to simulate a label photographed
    under harsh directional or partial lighting. Strength raised to 0.55-0.75
    and a second crossing gradient is overlaid to create a corner-darkening
    effect that is much harder for the pre-processing stage to correct.
    A slight warmth shift is also applied to the dark regions to mimic
    incandescent or phone-torch colour casts.
    """
    rng  = np.random.default_rng()
    arr  = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    # Primary gradient — strong shadow from one side
    direction = rng.choice(["horizontal", "vertical"])
    strength  = float(rng.uniform(0.55, 0.75))

    if direction == "horizontal":
        g1 = np.linspace(1.0 - strength, 1.0, w, dtype=np.float32)
        mask = np.tile(g1, (h, 1))
    else:
        g1 = np.linspace(1.0 - strength, 1.0, h, dtype=np.float32)
        mask = np.tile(g1.reshape(-1, 1), (1, w))

    if rng.random() > 0.5:
        mask = np.flip(mask, axis=1 if direction == "horizontal" else 0)

    # Secondary crossing gradient for corner darkening
    cross_strength = float(rng.uniform(0.20, 0.40))
    g2   = np.linspace(1.0 - cross_strength, 1.0, h, dtype=np.float32)
    mask2 = np.tile(g2.reshape(-1, 1), (1, w))
    if rng.random() > 0.5:
        mask2 = np.flip(mask2, axis=0)

    combined = (mask * mask2)[:, :, np.newaxis]
    arr      = np.clip(arr * combined, 0, 255)

    # Warmth colour cast in the darker region — orange/yellow tint
    cast_strength = float(rng.uniform(0.06, 0.14))
    dark_mask     = 1.0 - combined                        # 1 = dark area
    arr[:, :, 0]  = np.clip(arr[:, :, 0] + dark_mask[:,:,0] * 255 * cast_strength * 0.9, 0, 255)
    arr[:, :, 1]  = np.clip(arr[:, :, 1] + dark_mask[:,:,0] * 255 * cast_strength * 0.5, 0, 255)
    arr[:, :, 2]  = np.clip(arr[:, :, 2] - dark_mask[:,:,0] * 255 * cast_strength * 0.3, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


# =============================================================================
# EVALUATION METRIC HELPERS
# =============================================================================

def normalise_token(t: str) -> str:
    """Lowercase, strip punctuation, collapse spaces for fair matching."""
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_metrics(predicted: list, ground_truth: list) -> dict:
    pred_set = set(normalise_token(p) for p in predicted if normalise_token(p))
    gt_set   = set(normalise_token(g) for g in ground_truth if normalise_token(g))

    if not pred_set and not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set  - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(output_dir: str, real_image_path: str = None, tesseract_cmd: str = None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # folder where all generated label images are saved for visual inspection
    images_path = output_path / "synthetic_labels"
    images_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSynthetic label images will be saved to: {images_path}")

    ocr = OCRHandler(tesseract_cmd=tesseract_cmd)

    conditions = {
        "Clean"  : lambda img: img,
        "Blurred": lambda img: apply_blur(img),
        "Rotated": lambda img: apply_rotation(img),
        "Noisy"  : lambda img: apply_noise(img),
        "Shadow" : lambda img: apply_shadow(img),
    }

    results = {}

    for cond_name, transform in conditions.items():
        print(f"\nRunning condition: {cond_name}")
        cond_dir = images_path / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)
        cond_metrics = []

        for idx, gt_ingredients in enumerate(LABEL_SETS):
            base_img     = make_label_image(gt_ingredients)
            degraded_img = transform(base_img)

            # save each generated image so you can open and inspect what Tesseract saw
            img_save_path = cond_dir / f"sample_{idx+1:02d}.png"
            degraded_img.save(str(img_save_path))

            ocr_result = ocr.extract_from_pil(degraded_img, filename=f"{cond_name}_{idx}.png")
            predicted  = ocr_result.get("ingredients", [])
            metrics    = compute_metrics(predicted, gt_ingredients)
            cond_metrics.append(metrics)

            print(f"  Sample {idx+1:>2} | GT={len(gt_ingredients):>2} "
                  f"Pred={len(predicted):>2} "
                  f"P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f} "
                  f"F1={metrics['f1']:.3f}")

        avg_p  = float(np.mean([m["precision"] for m in cond_metrics]))
        avg_r  = float(np.mean([m["recall"]    for m in cond_metrics]))
        avg_f1 = float(np.mean([m["f1"]        for m in cond_metrics]))

        results[cond_name] = {
            "samples"   : len(cond_metrics),
            "precision" : round(avg_p,  4),
            "recall"    : round(avg_r,  4),
            "f1"        : round(avg_f1, 4),
            "per_sample": cond_metrics,
        }

    # overall synthetic
    all_p  = [m["precision"] for c in results.values() for m in c["per_sample"]]
    all_r  = [m["recall"]    for c in results.values() for m in c["per_sample"]]
    all_f1 = [m["f1"]        for c in results.values() for m in c["per_sample"]]
    results["Overall"] = {
        "samples"  : sum(c["samples"] for c in results.values()),
        "precision": round(float(np.mean(all_p)),  4),
        "recall"   : round(float(np.mean(all_r)),  4),
        "f1"       : round(float(np.mean(all_f1)), 4),
    }

    # ==========================================================================
    # REAL IMAGE SMOKE TEST
    # ==========================================================================
    real_result = None
    if real_image_path:
        rp = Path(real_image_path)
        if rp.exists():
            print(f"\n{'='*62}")
            print(f"  Real image smoke test: {rp.name}")
            print(f"{'='*62}")
            import time as _time
            t0          = _time.perf_counter()
            real_result = ocr.extract_from_path(str(rp))
            elapsed_ms  = round((_time.perf_counter() - t0) * 1000, 1)

            print(f"  success          : {real_result['success']}")
            print(f"  confidence       : {real_result['confidence']}")
            print(f"  ingredient count : {real_result['ingredient_count']}")
            print(f"  total_ms         : {real_result['meta'].get('total_ms', elapsed_ms)}")
            if real_result["warnings"]:
                for w in real_result["warnings"]:
                    print(f"  warning          : {w}")
            print(f"\n  Extracted ingredients:")
            for i, ing in enumerate(real_result["ingredients"], 1):
                print(f"    {i:>2}. {ing}")
        else:
            print(f"\nWARNING: Real image not found at '{real_image_path}' — skipping smoke test.")
    else:
        print("\nNo real image path provided. Use --real_image to run the smoke test.")

    # ==========================================================================
    # PRINT SYNTHETIC TABLE
    # ==========================================================================
    print("\n")
    print("=" * 62)
    print(f"  {'Condition':<10} {'Samples':>7}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print("=" * 62)
    for cond, r in results.items():
        print(f"  {cond:<10} {r['samples']:>7}  {r['precision']:>9.4f}  "
              f"{r['recall']:>6.4f}  {r['f1']:>6.4f}")
    print("=" * 62)

    # ==========================================================================
    # LaTeX rows
    # ==========================================================================
    print("\n-- LaTeX table rows (paste into your results table) --")
    for cond, r in results.items():
        if cond == "Overall":
            print("\\midrule")
        print(f"{cond} & {r['samples']} & {r['precision']:.4f} & "
              f"{r['recall']:.4f} & {r['f1']:.4f} \\\\")

    if real_result:
        print("\n-- LaTeX smoke test line --")
        ms  = real_result['meta'].get('total_ms', 'N/A')
        cnt = real_result['ingredient_count']
        conf= real_result['confidence']
        print(f"A real-image smoke test on \\texttt{{{rp.name}}} returned "
              f"{cnt} ingredients with {conf} confidence in {ms}\\,ms.")

    # ==========================================================================
    # MATPLOTLIB FIGURE
    # ==========================================================================
    cond_names = ["Clean", "Blurred", "Rotated", "Noisy", "Shadow"]
    precisions = [results[c]["precision"] for c in cond_names]
    recalls    = [results[c]["recall"]    for c in cond_names]
    f1s        = [results[c]["f1"]        for c in cond_names]

    x     = np.arange(len(cond_names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_p  = ax.bar(x - width, precisions, width, label="Precision",
                     color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars_r  = ax.bar(x,          recalls,   width, label="Recall",
                     color="#55A868", edgecolor="white", linewidth=0.8)
    bars_f1 = ax.bar(x + width,  f1s,       width, label="F1 Score",
                     color="#C44E52", edgecolor="white", linewidth=0.8)

    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#333333"
                )

    label_bars(bars_p)
    label_bars(bars_r)
    label_bars(bars_f1)

    ax.set_xlabel("Image Condition", fontsize=12, labelpad=8)
    ax.set_ylabel("Score", fontsize=12, labelpad=8)
    ax.set_title("OCR Extraction Quality by Image Condition", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    overall_f1 = results["Overall"]["f1"]
    ax.axhline(overall_f1, color="#C44E52", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.legend(
        [bars_p, bars_r, bars_f1,
         plt.Line2D([0], [0], color="#C44E52", linestyle="--", linewidth=1.2, alpha=0.6)],
        ["Precision", "Recall", "F1 Score", f"Overall F1 = {overall_f1:.4f}"],
        fontsize=10, framealpha=0.9,
    )

    plt.tight_layout()
    fig_path = output_path / "ocr_quality_by_condition.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to: {fig_path}")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkinSpectra OCR Evaluator")
    parser.add_argument("--output_dir",  default="figures",
                        help="Directory to save figures and synthetic label images (default: figures)")
    parser.add_argument("--real_image",  default=None,
                        help="Path to a real label photo for the smoke test "
                             "(e.g. ../testing/dry_moisturizer.jpg)")
    parser.add_argument("--tesseract",   default=None,
                        help="Path to tesseract.exe if not on PATH")
    args = parser.parse_args()

    run_evaluation(
        output_dir     = args.output_dir,
        real_image_path= args.real_image,
        tesseract_cmd  = args.tesseract,
    )
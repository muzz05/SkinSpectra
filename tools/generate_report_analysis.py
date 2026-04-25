import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from components.calculation_individual_layer import CFG as IND_CFG
from components.calculation_layering_layer import CFG as LAY_CFG
from components.facial_analysis import FacialAnalyzer
from components.model_comparison import run_individual, run_layering
from components.nlp_layer import INCIMapper
from components.ocr_handler import OCRHandler

ROOT = Path(__file__).resolve().parent.parent
REPORT_DATA = ROOT / "report_data"
REPORT_FIGS = ROOT / "report" / "figures"
ANALYSIS_JSON = REPORT_DATA / "analysis_results.json"
COMP_JSON = ROOT / "models" / "comparison_results.json"

RNG = random.Random(42)


def _ensure_dirs() -> None:
    REPORT_DATA.mkdir(parents=True, exist_ok=True)
    REPORT_FIGS.mkdir(parents=True, exist_ok=True)


def _norm_token(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def evaluate_nlp() -> dict:
    mapper = INCIMapper.load(ROOT / "models" / "nlp")
    df = pd.read_csv(ROOT / "data" / "ingredient_mapping.csv")

    alias_columns = [
        "inci_name",
        "common_names",
        "trade_names",
        "chemical_aliases",
        "language_variants",
    ]

    cases = []
    for _, row in df.iterrows():
        target = str(row.get("inci_name", "")).strip()
        if not target:
            continue
        for col in alias_columns:
            raw = row.get(col, "")
            if pd.isna(raw):
                continue
            for part in str(raw).split("|"):
                q = part.strip()
                if q:
                    cases.append((q, target, col))

    by_col = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    conf = Counter()
    latencies = []
    top1 = 0
    top3 = 0

    for query, target, col in cases:
        res = mapper.map(query)
        pred = _norm_token(res.get("inci_name", ""))
        tgt = _norm_token(target)

        alt_names = [_norm_token(a.get("inci_name", "")) for a in res.get("alternatives", [])]

        match_top1 = pred == tgt
        match_top3 = match_top1 or (tgt in alt_names)

        top1 += int(match_top1)
        top3 += int(match_top3)
        by_col[col]["n"] += 1
        by_col[col]["top1"] += int(match_top1)
        by_col[col]["top3"] += int(match_top3)
        conf[res.get("confidence", "uncertain")] += 1
        latencies.append(float(res.get("latency_ms", 0.0)))

    n = len(cases)
    out = {
        "cases": n,
        "top1_accuracy": round(_safe_div(top1, n), 4),
        "top3_recall": round(_safe_div(top3, n), 4),
        "confidence_distribution": dict(conf),
        "latency_ms": {
            "mean": round(mean(latencies), 3) if latencies else 0.0,
            "median": round(median(latencies), 3) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        },
        "by_source_column": {
            k: {
                "cases": v["n"],
                "top1_accuracy": round(_safe_div(v["top1"], v["n"]), 4),
                "top3_recall": round(_safe_div(v["top3"], v["n"]), 4),
            }
            for k, v in by_col.items()
        },
    }
    return out


def evaluate_regression_models() -> dict:
    ind_metrics = json.loads((ROOT / "models" / "calculation_individual" / "metrics.json").read_text(encoding="utf-8"))
    lay_metrics = json.loads((ROOT / "models" / "calculation_layering" / "metrics.json").read_text(encoding="utf-8"))

    return {
        "individual": ind_metrics,
        "layering": lay_metrics,
        "ensemble_weights": {
            "individual": {"rule_weight": IND_CFG["rule_weight"], "ml_weight": IND_CFG["ml_weight"]},
            "layering": {"rule_weight": LAY_CFG["rule_weight"], "ml_weight": LAY_CFG["ml_weight"]},
        },
    }


def evaluate_model_comparison(samples: int = 2500, seed: int = 42) -> dict:
    payload = {
        "generated_at_epoch": int(time.time()),
        "source": "tools/generate_report_analysis.py",
        "comparison_samples": samples,
        "comparison_seed": seed,
        "individual": run_individual(samples, seed),
        "layering": run_layering(samples, seed),
    }
    COMP_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _augment_face_image(img: np.ndarray, mode: str, idx: int) -> np.ndarray:
    out = img.copy()
    if mode == "clean":
        return out
    if mode == "gaussian_blur":
        k = 3 + (idx % 3) * 2
        return cv2.GaussianBlur(out, (k, k), 0)
    if mode == "brightness":
        alpha = 0.7 + 0.1 * (idx % 5)
        return cv2.convertScaleAbs(out, alpha=alpha, beta=0)
    if mode == "noise":
        noise = np.random.default_rng(42 + idx).normal(0, 10 + (idx % 5) * 3, out.shape).astype(np.int16)
        nimg = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return nimg
    if mode == "rotation":
        angle = [-10, -6, -3, 3, 6, 10][idx % 6]
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def evaluate_facial() -> dict:
    analyzer = FacialAnalyzer.load()

    oily_path = ROOT / "testing" / "oily-face.webp"
    blur_path = ROOT / "testing" / "blur.jpg"
    noface_path = ROOT / "testing" / "no-face.webp"

    pos_modes = ["clean", "gaussian_blur", "brightness", "noise", "rotation"]
    per_mode = defaultdict(list)

    img = cv2.imread(str(oily_path))
    if img is None:
        raise FileNotFoundError(f"Missing test image: {oily_path}")

    tmp_dir = ROOT / "report_data" / "_tmp_face"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    positive_total = 0
    positive_correct = 0
    valid_predictions = 0
    latencies = []

    for mode in pos_modes:
        for i in range(10):
            positive_total += 1
            aug = _augment_face_image(img, mode, i)
            p = tmp_dir / f"face_{mode}_{i}.png"
            cv2.imwrite(str(p), aug)
            res = analyzer.predict(str(p))

            if "error" not in res:
                valid_predictions += 1
                per_mode[mode].append(float(res.get("confidence", 0.0)))
                latencies.append(float(res.get("latency_ms", 0.0)))
                if str(res.get("skin_type", "")).lower() == "oily":
                    positive_correct += 1
            else:
                per_mode[mode].append(0.0)

    negatives = [
        (blur_path, "blurry"),
        (noface_path, "no_face"),
    ]
    negative_total = 0
    negative_correct = 0
    negative_details = []
    for path, expected_err in negatives:
        negative_total += 1
        res = analyzer.predict(str(path))
        got = res.get("error", "none")
        is_correct = got == expected_err
        negative_correct += int(is_correct)
        negative_details.append({"file": path.name, "expected": expected_err, "predicted": got, "correct": is_correct})

    for p in tmp_dir.glob("*.png"):
        try:
            p.unlink()
        except OSError:
            pass

    total = positive_total + negative_total
    correct = positive_correct + negative_correct

    return {
        "positive_samples": positive_total,
        "positive_oily_consistency": round(_safe_div(positive_correct, positive_total), 4),
        "positive_valid_prediction_rate": round(_safe_div(valid_predictions, positive_total), 4),
        "negative_samples": negative_total,
        "negative_rejection_accuracy": round(_safe_div(negative_correct, negative_total), 4),
        "overall_operational_accuracy": round(_safe_div(correct, total), 4),
        "latency_ms": {
            "mean": round(mean(latencies), 3) if latencies else 0.0,
            "median": round(median(latencies), 3) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        },
        "mode_confidence": {m: [round(v, 4) for v in vals] for m, vals in per_mode.items()},
        "negative_checks": negative_details,
    }


def _generate_label_image(ingredients: list[str], condition: str) -> Image.Image:
    text = "Ingredients: " + ", ".join(ingredients)
    w, h = 1450, 420
    img = Image.new("RGB", (w, h), color=(248, 248, 246))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    lines = []
    words = text.split(" ")
    cur = []
    for wtoken in words:
        candidate = " ".join(cur + [wtoken])
        if len(candidate) > 95:
            lines.append(" ".join(cur))
            cur = [wtoken]
        else:
            cur.append(wtoken)
    if cur:
        lines.append(" ".join(cur))

    y = 40
    for line in lines:
        draw.text((30, y), line, fill=(20, 20, 20), font=font)
        y += 24

    if condition == "clean":
        return img
    if condition == "blurred":
        return img.filter(ImageFilter.GaussianBlur(radius=1.2))
    if condition == "rotated":
        return img.rotate(2.5, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(248, 248, 246))
    if condition == "noisy":
        arr = np.array(img)
        noise = np.random.default_rng(123).normal(0, 12, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return img


def _precision_recall_f1(pred: set[str], gt: set[str]) -> tuple[float, float, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_ocr() -> dict:
    handler = OCRHandler()
    inci_df = pd.read_csv(ROOT / "data" / "ingredient_profiles.csv")
    inci_names = [str(v).strip() for v in inci_df["inci_name"].dropna().tolist() if str(v).strip()]

    conditions = ["clean", "blurred", "rotated", "noisy"]
    rows = []

    for condition in conditions:
        for i in range(10):
            picked = RNG.sample(inci_names, 10)
            gt = {_norm_token(x) for x in picked}
            img = _generate_label_image(picked, condition)
            res = handler.extract_from_pil(img, filename=f"synthetic_{condition}_{i}.png")
            pred = {_norm_token(x) for x in res.get("ingredients", [])}
            p, r, f1 = _precision_recall_f1(pred, gt)
            rows.append(
                {
                    "condition": condition,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "latency_ms": float(res.get("meta", {}).get("total_ms", 0.0)),
                    "confidence": res.get("confidence", "very_low"),
                    "pred_count": len(pred),
                    "gt_count": len(gt),
                }
            )

    ocr_df = pd.DataFrame(rows)

    by_condition = {}
    for cond, g in ocr_df.groupby("condition"):
        by_condition[cond] = {
            "samples": int(len(g)),
            "precision": round(float(g["precision"].mean()), 4),
            "recall": round(float(g["recall"].mean()), 4),
            "f1": round(float(g["f1"].mean()), 4),
            "latency_ms_mean": round(float(g["latency_ms"].mean()), 3),
        }

    real_img_path = ROOT / "testing" / "dry_moisturizer.jpg"
    real_img_result = None
    if real_img_path.exists():
        real_img_result = handler.extract_from_path(str(real_img_path))

    return {
        "synthetic_samples": int(len(ocr_df)),
        "overall": {
            "precision": round(float(ocr_df["precision"].mean()), 4),
            "recall": round(float(ocr_df["recall"].mean()), 4),
            "f1": round(float(ocr_df["f1"].mean()), 4),
            "latency_ms_mean": round(float(ocr_df["latency_ms"].mean()), 3),
        },
        "by_condition": by_condition,
        "real_image_smoke": {
            "file": real_img_path.name if real_img_result else None,
            "success": bool(real_img_result and real_img_result.get("success")),
            "ingredient_count": int(real_img_result.get("ingredient_count", 0)) if real_img_result else 0,
            "confidence": real_img_result.get("confidence", "very_low") if real_img_result else "very_low",
            "latency_ms": float(real_img_result.get("meta", {}).get("total_ms", 0.0)) if real_img_result else 0.0,
        },
        "raw_rows": rows,
    }


def plot_nlp_confidence(nlp: dict) -> None:
    dist = nlp.get("confidence_distribution", {})
    keys = ["high", "medium", "low", "uncertain"]
    vals = [dist.get(k, 0) for k in keys]

    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(keys, vals, color=["#2a9d8f", "#457b9d", "#f4a261", "#e76f51"])
    plt.title("NLP Mapping Confidence Distribution")
    plt.ylabel("Count")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "nlp_confidence_distribution.png", dpi=180)
    plt.close()


def plot_model_comparison(comp: dict) -> None:
    ind_rows = comp.get("individual", {}).get("results", [])
    lay_rows = comp.get("layering", {}).get("results", [])

    names = [r["model"] for r in ind_rows]
    ind_r2 = [r["r2"] for r in ind_rows]
    lay_map = {r["model"]: r["r2"] for r in lay_rows}
    lay_r2 = [lay_map.get(n, np.nan) for n in names]

    x = np.arange(len(names))
    w = 0.38

    plt.figure(figsize=(12, 5.2))
    plt.bar(x - w / 2, ind_r2, width=w, label="Single-Product Task", color="#457b9d")
    plt.bar(x + w / 2, lay_r2, width=w, label="Layering Task", color="#2a9d8f")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylim(0.0, 1.02)
    plt.ylabel("R-squared")
    plt.title("Algorithm Comparison Across Regression Tasks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "model_comparison_r2.png", dpi=180)
    plt.close()


def plot_facial_robustness(facial: dict) -> None:
    mode_conf = facial.get("mode_confidence", {})
    modes = list(mode_conf.keys())
    if not modes:
        return

    data = [mode_conf[m] for m in modes]
    plt.figure(figsize=(9, 4.8))
    plt.boxplot(data, labels=modes, showmeans=True)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Predicted Oily Confidence")
    plt.title("Facial Model Robustness Under Input Perturbations")
    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "facial_robustness_confidence.png", dpi=180)
    plt.close()


def plot_ocr_quality(ocr: dict) -> None:
    by_cond = ocr.get("by_condition", {})
    conditions = ["clean", "blurred", "rotated", "noisy"]
    prec = [by_cond.get(c, {}).get("precision", 0.0) for c in conditions]
    rec = [by_cond.get(c, {}).get("recall", 0.0) for c in conditions]
    f1 = [by_cond.get(c, {}).get("f1", 0.0) for c in conditions]

    x = np.arange(len(conditions))
    w = 0.25

    plt.figure(figsize=(9.5, 4.8))
    plt.bar(x - w, prec, width=w, label="Precision", color="#264653")
    plt.bar(x, rec, width=w, label="Recall", color="#2a9d8f")
    plt.bar(x + w, f1, width=w, label="F1", color="#e9c46a")
    plt.xticks(x, conditions)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Score")
    plt.title("OCR Extraction Quality by Image Condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "ocr_quality_by_condition.png", dpi=180)
    plt.close()


def main() -> None:
    _ensure_dirs()

    print("[analysis] Running NLP evaluation...")
    nlp = evaluate_nlp()

    print("[analysis] Reading regression model metrics...")
    regression = evaluate_regression_models()

    print("[analysis] Running algorithm comparisons...")
    comparison = evaluate_model_comparison(samples=2500, seed=42)

    print("[analysis] Running facial robustness evaluation...")
    facial = evaluate_facial()

    print("[analysis] Running OCR quality evaluation...")
    ocr = evaluate_ocr()

    summary = {
        "generated_at_epoch": int(time.time()),
        "nlp": nlp,
        "regression_models": regression,
        "model_comparison": comparison,
        "facial": facial,
        "ocr": {
            "synthetic_samples": ocr["synthetic_samples"],
            "overall": ocr["overall"],
            "by_condition": ocr["by_condition"],
            "real_image_smoke": ocr["real_image_smoke"],
        },
    }

    print("[analysis] Saving plots...")
    plot_nlp_confidence(nlp)
    plot_model_comparison(comparison)
    plot_facial_robustness(facial)
    plot_ocr_quality(ocr)

    ANALYSIS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[analysis] Saved analysis JSON to: {ANALYSIS_JSON}")
    print(f"[analysis] Figures directory: {REPORT_FIGS}")


if __name__ == "__main__":
    main()

"""
SkinSpectra Comparative Analysis Script
=========================================
Benchmarks multiple ML algorithms against the SkinSpectra XGBoost (single-product)
and LightGBM (layering) models on identical synthetic datasets. Reports MAE, RMSE,
R2 and per-sample inference latency for each model. Saves two bar charts and prints
LaTeX table rows.

Usage
-----
    python eval_comparative.py
        --dataset2   ../data/ingredient_profiles.csv
        --dataset3   ../data/layering_compatibility.csv
        --model_dir1 ../models/calculation_individual
        --model_dir2 ../models/calculation_layering
        --output_dir figures
        --samples    2500
        --seed       42

Requirements
------------
    pip install xgboost lightgbm scikit-learn numpy pandas matplotlib tqdm joblib
    Both calculation modules must be importable from the same directory.
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import sys
# Ensure repository root is on sys.path so `components` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── import SkinSpectra modules ────────────────────────────────────────────────
try:
    from components.calculation_individual_layer import (
        IngredientProfileDB as ProfileDB1,
        RuleEngine,
        generate_synthetic_data as gen_individual,
        CFG as CFG1,
    )
except ImportError:
    print("ERROR: Could not import calculate_individual.py. "
          "Make sure it is in the same directory.")
    sys.exit(1)

try:
    from components.calculation_layering_layer import (
        IngredientProfileDB as ProfileDB2,
        LayeringPairDB,
        LayeringRuleEngine,
        generate_synthetic_data as gen_layering,
        CFG as CFG2,
    )
except ImportError:
    print("ERROR: Could not import calculate_layering.py. "
          "Make sure it is in the same directory.")
    sys.exit(1)


# =============================================================================
# CANDIDATE MODELS
# =============================================================================

def get_candidate_models():
    """
    Returns a list of (label, model) tuples.
    Only tree-ensemble and distance-based models are included to keep
    the comparison focused on non-linear regressors comparable in
    architecture to XGBoost and LightGBM.
    """
    return [
        ("Random Forest",  RandomForestRegressor(n_estimators=200, max_depth=8,
                                                  random_state=42, n_jobs=-1)),
        ("SVR (RBF)",      SVR(kernel="rbf", C=10, epsilon=0.5)),
        ("KNN (k=5)",      KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
    ]


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def benchmark(X_train, X_val, y_train, y_val,
              skinspectra_label: str,
              skinspectra_params: dict,
              use_lightgbm: bool = False) -> list:
    """
    Train ALL models including SkinSpectra on the same X_train/y_train
    and evaluate on the same X_val/y_val. This is the only fair comparison
    because the SkinSpectra model sees the same data distribution as competitors.
    A fresh StandardScaler is fitted on X_train for all models.
    """
    results = []
    candidates = get_candidate_models()

    # Fit a shared scaler on training data
    shared_scaler = StandardScaler()
    X_train_sc    = shared_scaler.fit_transform(X_train)
    X_val_sc      = shared_scaler.transform(X_val)

    # ── Candidate models ──────────────────────────────────────────────────
    for label, model in candidates:
        print(f"  Training: {label} ...")
        model.fit(X_train_sc, y_train)

        t_inf  = time.perf_counter()
        y_pred = np.clip(model.predict(X_val_sc), 0, 100)
        inf_us = (time.perf_counter() - t_inf) * 1e6 / len(X_val)

        mae  = mean_absolute_error(y_val, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2   = r2_score(y_val, y_pred)

        results.append({
            "label"      : label,
            "mae"        : round(mae,  4),
            "rmse"       : round(rmse, 4),
            "r2"         : round(r2,   4),
            "infer_us"   : round(inf_us, 2),
            "skinspectra": False,
        })
        print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  "
              f"Infer={inf_us:.2f}µs")

    # ── SkinSpectra model — trained fresh on same split ───────────────────
    print(f"  Training: {skinspectra_label} (SkinSpectra) ...")
    if use_lightgbm:
        import lightgbm as lgb_fresh
        ss_model = lgb_fresh.LGBMRegressor(**skinspectra_params)
        ss_model.fit(
            X_train_sc, y_train,
            eval_set=[(X_val_sc, y_val)],
            callbacks=[lgb_fresh.early_stopping(50, verbose=False),
                       lgb_fresh.log_evaluation(-1)],
        )
    else:
        import xgboost as xgb_fresh
        ss_params = {k: v for k, v in skinspectra_params.items()
                     if k not in ("objective", "eval_metric")}
        ss_model = xgb_fresh.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **ss_params,
        )
        ss_model.fit(X_train_sc, y_train,
                     eval_set=[(X_val_sc, y_val)],
                     verbose=False)

    t_inf  = time.perf_counter()
    y_pred = np.clip(ss_model.predict(X_val_sc), 0, 100)
    inf_us = (time.perf_counter() - t_inf) * 1e6 / len(X_val)

    mae  = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2   = r2_score(y_val, y_pred)

    results.append({
        "label"      : skinspectra_label,
        "mae"        : round(mae,  4),
        "rmse"       : round(rmse, 4),
        "r2"         : round(r2,   4),
        "infer_us"   : round(inf_us, 2),
        "skinspectra": True,
    })
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  "
          f"Infer={inf_us:.2f}µs")

    results.sort(key=lambda x: x["r2"], reverse=True)
    return results


# =============================================================================
# PRINTING AND LATEX
# =============================================================================

def print_table(results: list, title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"  {'Model':<32} {'MAE':>7}  {'RMSE':>7}  {'R2':>7}  {'Infer(µs)':>10}")
    print(f"  {'-'*32} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}")
    for r in results:
        marker = " *" if r["skinspectra"] else "  "
        print(f"{marker} {r['label']:<32} {r['mae']:>7.4f}  {r['rmse']:>7.4f}  "
              f"{r['r2']:>7.4f}  {r['infer_us']:>10.2f}")
    print(f"{'='*72}")
    print("  (* = SkinSpectra production model)")


def print_latex(results: list, label: str, caption: str):
    print(f"\n-- LaTeX table: {caption} --")
    print(f"\\captionof{{table}}{{{caption}}}")
    print(f"\\label{{{label}}}")
    print("\\begin{tabular}{@{}lllll@{}}")
    print("\\toprule")
    print("Model & MAE & RMSE & $R^2$ & Infer ($\\mu$s) \\\\")
    print("\\midrule")
    for r in results:
        name = r["label"].replace("(", "\\textbf{(").replace(")", ")}")  \
               if r["skinspectra"] else r["label"]
        if r["skinspectra"]:
            name = f"\\textbf{{{r['label']}}}"
        print(f"{name} & {r['mae']} & {r['rmse']} & {r['r2']} & {r['infer_us']} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


# =============================================================================
# FIGURES — SINGLE-PRODUCT (3 individual figures)
# =============================================================================

def plot_single_r2_mae(results: list, output_dir: str):
    """Bar chart: R2 and MAE for single-product task."""
    labels      = [r["label"] for r in results]
    r2_vals     = [r["r2"]    for r in results]
    mae_vals    = [r["mae"]   for r in results]
    colors_r2   = ["#C44E52" if r["skinspectra"] else "#4C72B0" for r in results]
    colors_mae  = ["#e88a8c" if r["skinspectra"] else "#8aaed4" for r in results]

    x     = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_r2  = ax.bar(x - width/2, r2_vals,  width, color=colors_r2,
                      edgecolor="white", linewidth=0.9, alpha=0.92, zorder=3)
    bars_mae = ax.bar(x + width/2, mae_vals, width, color=colors_mae,
                      edgecolor="white", linewidth=0.9, alpha=0.92, zorder=3)

    for bar in list(bars_r2) + list(bars_mae):
        h = bar.get_height()
        if h > 0.005:
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="#222222")

    for i, r in enumerate(results):
        if r["skinspectra"]:
            for bg in [bars_r2, bars_mae]:
                bg[i].set_edgecolor("#222222")
                bg[i].set_linewidth(1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Single-Product Task: $R^2$ and MAE Comparison",
                 fontsize=13, pad=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#4C72B0", alpha=0.92, label="$R^2$ (competitor)"),
        Patch(facecolor="#8aaed4", alpha=0.92, label="MAE (competitor)"),
        Patch(facecolor="#C44E52", alpha=0.92, edgecolor="#222222",
              linewidth=1.4, label="$R^2$ (SkinSpectra)"),
        Patch(facecolor="#e88a8c", alpha=0.92, edgecolor="#222222",
              linewidth=1.4, label="MAE (SkinSpectra)"),
    ], fontsize=9, framealpha=0.92, loc="upper right", edgecolor="#cccccc")

    plt.tight_layout()
    path = Path(output_dir) / "single_r2_mae.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {path}")


def plot_single_latency(results: list, output_dir: str):
    """Horizontal log-scale latency chart for single-product task."""
    sorted_r = sorted(results, key=lambda x: x["infer_us"])
    labels   = [r["label"]    for r in sorted_r]
    lats     = [r["infer_us"] for r in sorted_r]
    colors   = ["#C44E52" if r["skinspectra"] else "#55A868" for r in sorted_r]
    ec       = ["#222222" if r["skinspectra"] else "white"   for r in sorted_r]
    lw       = [1.8       if r["skinspectra"] else 0.8       for r in sorted_r]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, lats, color=colors, edgecolor=ec,
                   linewidth=lw, alpha=0.88)
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency per sample (µs, log scale)", fontsize=10)
    ax.set_title("Single-Product Task: Inference Latency",
                 fontsize=13, pad=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, v in zip(bars, lats):
        ax.text(v * 1.12, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} µs", va="center", ha="left", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#55A868", alpha=0.88, label="Competitor models"),
        Patch(facecolor="#C44E52", alpha=0.88, edgecolor="#222222",
              linewidth=1.4, label="SkinSpectra model"),
    ], fontsize=9, framealpha=0.92, loc="lower right", edgecolor="#cccccc")

    plt.tight_layout()
    path = Path(output_dir) / "single_latency.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {path}")


def plot_single_heatmap(results: list, output_dir: str):
    """Heatmap of all metrics for single-product task."""
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.colors as mcolors

    metrics = ["MAE", "RMSE", "R²", "Infer (µs)"]
    labels  = [r["label"] for r in results]
    mat     = np.array([[r["mae"], r["rmse"], r["r2"], r["infer_us"]]
                        for r in results], dtype=float)

    def norm_col(col, higher_better):
        mn, mx = col.min(), col.max()
        if mx == mn: return np.ones_like(col) * 0.5
        n = (col - mn) / (mx - mn)
        return n if higher_better else 1.0 - n

    norm = np.column_stack([
        norm_col(mat[:, 0], False),
        norm_col(mat[:, 1], False),
        norm_col(mat[:, 2], True),
        norm_col(mat[:, 3], False),
    ])

    n_models = len(labels)
    fig, ax  = plt.subplots(figsize=(8, n_models * 0.85 + 1.8))
    im = ax.imshow(norm, cmap=plt.cm.RdYlGn, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Single-Product Task: Model Comparison Heatmap",
                 fontsize=13, pad=12, fontweight="bold")

    for i in range(n_models):
        for j in range(len(metrics)):
            val = mat[i, j]
            txt = f"{val:.4f}" if j < 3 else f"{val:.1f}"
            brightness  = norm[i, j]
            text_color  = "white" if brightness < 0.35 or brightness > 0.85 else "#222222"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    for i, r in enumerate(results):
        if r["skinspectra"]:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color("#C44E52")
            rect = FancyBboxPatch((-0.5, i - 0.5), len(metrics), 1.0,
                                  linewidth=2, edgecolor="#C44E52",
                                  facecolor="none", boxstyle="round,pad=0.05")
            ax.add_patch(rect)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(bottom=0.20)
    cbar_ax = fig.add_axes([0.15, 0.07, 0.70, 0.04])
    cbar    = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalised score (green = better)", fontsize=10)

    path = Path(output_dir) / "single_heatmap.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {path}")


# =============================================================================
# FIGURES — LAYERING (3 individual figures)
# =============================================================================

def plot_layering_r2_mae(results: list, output_dir: str):
    """Bar chart: R2 and MAE for layering task."""
    labels     = [r["label"] for r in results]
    r2_vals    = [r["r2"]    for r in results]
    mae_vals   = [r["mae"]   for r in results]
    colors_r2  = ["#C44E52" if r["skinspectra"] else "#4C72B0" for r in results]
    colors_mae = ["#e88a8c" if r["skinspectra"] else "#8aaed4" for r in results]

    x     = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_r2  = ax.bar(x - width/2, r2_vals,  width, color=colors_r2,
                      edgecolor="white", linewidth=0.9, alpha=0.92, zorder=3)
    bars_mae = ax.bar(x + width/2, mae_vals, width, color=colors_mae,
                      edgecolor="white", linewidth=0.9, alpha=0.92, zorder=3)

    for bar in list(bars_r2) + list(bars_mae):
        h = bar.get_height()
        if h > 0.005:
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="#222222")

    for i, r in enumerate(results):
        if r["skinspectra"]:
            for bg in [bars_r2, bars_mae]:
                bg[i].set_edgecolor("#222222")
                bg[i].set_linewidth(1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Layering Task: $R^2$ and MAE Comparison",
                 fontsize=13, pad=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#4C72B0", alpha=0.92, label="$R^2$ (competitor)"),
        Patch(facecolor="#8aaed4", alpha=0.92, label="MAE (competitor)"),
        Patch(facecolor="#C44E52", alpha=0.92, edgecolor="#222222",
              linewidth=1.4, label="$R^2$ (SkinSpectra)"),
        Patch(facecolor="#e88a8c", alpha=0.92, edgecolor="#222222",
              linewidth=1.4, label="MAE (SkinSpectra)"),
    ], fontsize=9, framealpha=0.92, loc="upper right", edgecolor="#cccccc")

    plt.tight_layout()
    path = Path(output_dir) / "layering_r2_mae.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {path}")


def plot_layering_latency(results: list, output_dir: str):
    """Horizontal log-scale latency chart for layering task."""
    sorted_r = sorted(results, key=lambda x: x["infer_us"])
    labels   = [r["label"]    for r in sorted_r]
    lats     = [r["infer_us"] for r in sorted_r]
    colors   = ["#C44E52" if r["skinspectra"] else "#55A868" for r in sorted_r]
    ec       = ["#222222" if r["skinspectra"] else "white"   for r in sorted_r]
    lw       = [1.8       if r["skinspectra"] else 0.8       for r in sorted_r]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, lats, color=colors, edgecolor=ec,
                   linewidth=lw, alpha=0.88)
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency per sample (µs, log scale)", fontsize=10)
    ax.set_title("Layering Task: Inference Latency",
                 fontsize=13, pad=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, v in zip(bars, lats):
        ax.text(v * 1.12, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} µs", va="center", ha="left", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#55A868", alpha=0.88, label="Competitor models"),
        Patch(facecolor="#C44E52", alpha=0.88, edgecolor="#222222",
              linewidth=1.4, label="SkinSpectra model"),
    ], fontsize=9, framealpha=0.92, loc="lower right", edgecolor="#cccccc")

    plt.tight_layout()
    path = Path(output_dir) / "layering_latency.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {path}")


def plot_layering_heatmap(results: list, output_dir: str):
    """Heatmap of all metrics for layering task."""
    from matplotlib.patches import FancyBboxPatch

    metrics = ["MAE", "RMSE", "R²", "Infer (µs)"]
    labels  = [r["label"] for r in results]
    mat     = np.array([[r["mae"], r["rmse"], r["r2"], r["infer_us"]]
                        for r in results], dtype=float)

    def norm_col(col, higher_better):
        mn, mx = col.min(), col.max()
        if mx == mn: return np.ones_like(col) * 0.5
        n = (col - mn) / (mx - mn)
        return n if higher_better else 1.0 - n

    norm = np.column_stack([
        norm_col(mat[:, 0], False),
        norm_col(mat[:, 1], False),
        norm_col(mat[:, 2], True),
        norm_col(mat[:, 3], False),
    ])

    n_models = len(labels)
    fig, ax  = plt.subplots(figsize=(8, n_models * 0.85 + 1.8))
    im = ax.imshow(norm, cmap=plt.cm.RdYlGn, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Layering Task: Model Comparison Heatmap",
                 fontsize=13, pad=12, fontweight="bold")

    for i in range(n_models):
        for j in range(len(metrics)):
            val = mat[i, j]
            txt = f"{val:.4f}" if j < 3 else f"{val:.1f}"
            brightness = norm[i, j]
            text_color = "white" if brightness < 0.35 or brightness > 0.85 else "#222222"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    for i, r in enumerate(results):
        if r["skinspectra"]:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color("#C44E52")
            rect = FancyBboxPatch((-0.5, i - 0.5), len(metrics), 1.0,
                                  linewidth=2, edgecolor="#C44E52",
                                  facecolor="none", boxstyle="round,pad=0.05")
            ax.add_patch(rect)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(bottom=0.20)
    cbar_ax = fig.add_axes([0.15, 0.07, 0.70, 0.04])
    cbar    = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalised score (green = better)", fontsize=10)

    path = Path(output_dir) / "layering_heatmap.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SkinSpectra Comparative Model Benchmarking"
    )
    parser.add_argument("--dataset2",   default="../data/ingredient_profiles.csv")
    parser.add_argument("--dataset3",   default="../data/layering_compatibility.csv")
    parser.add_argument("--model_dir1", default="../models/calculation_individual",
                        help="Directory containing xgb_model.pkl and scaler.pkl")
    parser.add_argument("--model_dir2", default="../models/calculation_layering",
                        help="Directory containing lgb_model.pkl and scaler.pkl")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--samples",    type=int, default=2500)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── SINGLE-PRODUCT TASK ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SINGLE-PRODUCT TASK")
    print("=" * 60)

    print("Loading ingredient profile DB and rule engine...")
    db1         = ProfileDB1(args.dataset2)
    rule_eng1   = RuleEngine(CFG1)

    print(f"Generating {args.samples} synthetic samples...")
    X1, y1 = gen_individual(db1, rule_eng1, args.samples, args.seed)
    print(f"Generated {len(X1)} samples | score range [{y1.min():.1f}, {y1.max():.1f}]")

    X1_tr, X1_val, y1_tr, y1_val = train_test_split(
        X1, y1, test_size=0.20, random_state=args.seed
    )
    print(f"\nBenchmarking on {len(X1_val)} validation samples...")
    # strip keys that XGBRegressor does not accept as constructor args
    xgb_params_clean = {k: v for k, v in CFG1["xgb_params"].items()
                        if k not in ("objective", "eval_metric")}
    results_ind = benchmark(
        X1_tr, X1_val, y1_tr, y1_val,
        "XGBoost (SkinSpectra)",
        xgb_params_clean,
        use_lightgbm=False,
    )

    print_table(results_ind, "Single-Product Task Results")
    print_latex(results_ind, "tbl:cmp_individual",
                "Comparative Results: Single-Product Task (2500 Samples)")

    # ── LAYERING TASK ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LAYERING TASK")
    print("=" * 60)

    print("Loading layering pair DB and rule engine...")
    db2       = ProfileDB2(args.dataset2)
    pair_db   = LayeringPairDB(args.dataset3)
    rule_eng2 = LayeringRuleEngine(pair_db, CFG2)

    print(f"Generating {args.samples} synthetic samples...")
    X2, y2 = gen_layering(pair_db, db2, rule_eng2, args.samples, args.seed)
    print(f"Generated {len(X2)} samples | score range [{y2.min():.1f}, {y2.max():.1f}]")

    X2_tr, X2_val, y2_tr, y2_val = train_test_split(
        X2, y2, test_size=0.20, random_state=args.seed
    )
    print(f"\nBenchmarking on {len(X2_val)} validation samples...")
    results_lay = benchmark(
        X2_tr, X2_val, y2_tr, y2_val,
        "LightGBM (SkinSpectra)",
        CFG2["lgb_params"],
        use_lightgbm=True,
    )

    print_table(results_lay, "Layering Task Results")
    print_latex(results_lay, "tbl:cmp_layering",
                "Comparative Results: Layering Task (2500 Samples)")

    # ── FIGURES ───────────────────────────────────────────────────────────
    # Single-product figures
    plot_single_r2_mae(results_ind, args.output_dir)
    plot_single_latency(results_ind, args.output_dir)
    plot_single_heatmap(results_ind, args.output_dir)

    # Layering figures
    plot_layering_r2_mae(results_lay, args.output_dir)
    plot_layering_latency(results_lay, args.output_dir)
    plot_layering_heatmap(results_lay, args.output_dir)

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("\n-- Summary --")
    for task, results in [("Single-product", results_ind), ("Layering", results_lay)]:
        ss      = next(r for r in results if r["skinspectra"])
        best_r2 = results[0]   # already sorted desc
        print(f"\n  {task}:")
        print(f"    SkinSpectra : R2={ss['r2']}  MAE={ss['mae']}  Infer={ss['infer_us']}µs")
        print(f"    Best R2     : {best_r2['label']} R2={best_r2['r2']}")
        if ss["r2"] == best_r2["r2"]:
            print(f"    SkinSpectra is the best model on this task.")


if __name__ == "__main__":
    main()
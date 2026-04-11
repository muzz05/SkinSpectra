"""
SkinSpectra - Model Comparison Module
Compares production models against baseline regressors for CEP analysis.
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from components.calculation_individual_layer import (
    CFG as IND_CFG,
    IngredientProfileDB as IndividualProfileDB,
    RuleEngine as IndividualRuleEngine,
    generate_synthetic_data as generate_individual_synthetic,
)
from components.calculation_layering_layer import (
    CFG as LAY_CFG,
    IngredientProfileDB as LayeringProfileDB,
    LayeringPairDB,
    LayeringRuleEngine,
    generate_synthetic_data as generate_layering_synthetic,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "models" / "comparison_results.json"


def _safe_predict(model, x_val: np.ndarray) -> np.ndarray:
    y_pred = model.predict(x_val)
    return np.clip(np.asarray(y_pred, dtype=np.float32), 0, 100)


def evaluate_models(
    x: np.ndarray,
    y: np.ndarray,
    models: Dict[str, object],
    test_size: float = 0.15,
    random_state: int = 42,
) -> List[Dict]:
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    results = []
    for name, model in models.items():
        t_train = time.perf_counter()
        model.fit(x_tr, y_tr)
        train_time = time.perf_counter() - t_train

        t_infer = time.perf_counter()
        y_pred = _safe_predict(model, x_val)
        infer_time_us = (time.perf_counter() - t_infer) / max(len(x_val), 1) * 1e6

        mae = round(float(mean_absolute_error(y_val, y_pred)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 4)
        r2 = round(float(r2_score(y_val, y_pred)), 4)

        row = {
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "train_time_s": round(train_time, 3),
            "infer_us_per_sample": round(infer_time_us, 2),
        }
        results.append(row)
        print(
            f"  {name:35s} MAE={mae:.4f}  RMSE={rmse:.4f}  "
            f"R2={r2:.4f}  Train={train_time:.2f}s"
        )

    return sorted(results, key=lambda r: r["r2"], reverse=True)


def _individual_models(seed: int) -> Dict[str, object]:
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed),
        "Gradient Boosting (sklearn)": GradientBoostingRegressor(
            n_estimators=200, random_state=seed
        ),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, gamma="scale")),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "XGBoost (SkinSpectra)": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
            n_jobs=-1,
        ),
    }


def _layering_models(seed: int) -> Dict[str, object]:
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed),
        "Gradient Boosting (sklearn)": GradientBoostingRegressor(
            n_estimators=200, random_state=seed
        ),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, gamma="scale")),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "LightGBM (SkinSpectra)": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            verbose=-1,
        ),
    }


def run_individual(samples: int, seed: int) -> Dict:
    print("\n[comparison] Building synthetic dataset for individual scoring...")
    cfg = copy.deepcopy(IND_CFG)
    data_path = str((ROOT / "data" / "ingredient_profiles.csv").resolve())
    db = IndividualProfileDB(data_path)
    rule = IndividualRuleEngine(cfg)
    x, y = generate_individual_synthetic(db, rule, samples, seed=seed)

    print("[comparison] Evaluating regressors for individual scoring:")
    rows = evaluate_models(x, y, _individual_models(seed), random_state=seed)
    return {
        "task": "individual",
        "samples": int(len(x)),
        "random_seed": seed,
        "results": rows,
    }


def run_layering(samples: int, seed: int) -> Dict:
    print("\n[comparison] Building synthetic dataset for layering scoring...")
    lay_cfg = copy.deepcopy(LAY_CFG)
    d2 = str((ROOT / "data" / "ingredient_profiles.csv").resolve())
    d3 = str((ROOT / "data" / "layering_compatibility.csv").resolve())

    profile_db = LayeringProfileDB(d2)
    pair_db = LayeringPairDB(d3)
    rule = LayeringRuleEngine(pair_db, lay_cfg)
    x, y = generate_layering_synthetic(pair_db, profile_db, rule, samples, seed=seed)

    print("[comparison] Evaluating regressors for layering scoring:")
    rows = evaluate_models(x, y, _layering_models(seed), random_state=seed)
    return {
        "task": "layering",
        "samples": int(len(x)),
        "random_seed": seed,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SkinSpectra model comparison")
    parser.add_argument("--task", choices=["individual", "layering", "both"], default="both")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = {
        "generated_at_epoch": int(time.time()),
        "source": "components/model_comparison.py",
    }

    if args.task in ("individual", "both"):
        payload["individual"] = run_individual(args.samples, args.seed)

    if args.task in ("layering", "both"):
        payload["layering"] = run_layering(args.samples, args.seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[comparison] Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

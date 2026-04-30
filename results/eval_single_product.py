"""
Run this from the project root after training:
    python eval_single_product.py --data ../data/ingredient_profiles.csv \
                                   --model_dir ../models/calculation_individual

Prints verified metrics and the corrected LaTeX table row.
"""

import argparse
import json
import numpy as np
import joblib
from pathlib import Path

import sys
# Ensure repository root is on sys.path so `components` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── adjust this import to match your actual module filename ──────────────────
from components.calculation_individual_layer import (
    IngredientProfileDB, RuleEngine, generate_synthetic_data, CFG
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default=CFG["data_path"])
    parser.add_argument("--model_dir", default=CFG["output_dir"])
    parser.add_argument("--samples",   type=int, default=CFG["n_synthetic"])
    parser.add_argument("--seed",      type=int, default=CFG["random_seed"])
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    print("Loading model and scaler...")
    model  = joblib.load(model_dir / "xgb_model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")

    # check if saved metrics exist
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            saved = json.load(f)
        print("\nSaved metrics from training run:")
        print(f"  MAE    : {saved['mae']}")
        print(f"  RMSE   : {saved['rmse']}")
        print(f"  R2     : {saved['r2']}")
        print(f"  n_train: {saved['n_train']}")
        print(f"  n_val  : {saved['n_val']}")
    else:
        print("No saved metrics.json found, regenerating validation set...")

    # regenerate the exact same val split to verify
    print("\nRegenerating synthetic data to reproduce val split...")
    db          = IngredientProfileDB(args.data)
    rule_engine = RuleEngine(CFG)
    X, y        = generate_synthetic_data(db, rule_engine, args.samples, args.seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=args.seed
    )

    X_val_sc = scaler.transform(X_val)
    y_pred   = np.clip(model.predict(X_val_sc), 0, 100)

    mae  = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2   = float(r2_score(y_val, y_pred))

    print("\nVerified metrics on reproduced val split:")
    print(f"  MAE    : {mae:.4f}")
    print(f"  RMSE   : {rmse:.4f}")
    print(f"  R2     : {r2:.4f}")
    print(f"  n_train: {len(X_train)}")
    print(f"  n_val  : {len(X_val)}")

if __name__ == "__main__":
    main()
"""
SkinSpectra Calculation Layer - Feature 1: Individual Product Compatibility
===========================================================================
Architecture:
  - Rule Engine  : Deterministic scoring using dataset2 ingredient profiles
                   Handles skin type fit, concern matching, comedogenicity,
                   irritancy, pregnancy, age group, and ingredient penalties
  - ML Model     : Gradient Boosted Trees (XGBoost) trained on synthetic
                   ground-truth samples derived from dermatological rules
                   Refines the rule score with learned non-linear patterns
  - Ensemble     : Final Score = 0.45 * rule_score + 0.55 * ml_score
  - Output       : 0-100 compatibility score + detailed pros/cons report

Scoring Dimensions
------------------
  1. Skin Type Compatibility      (0-25 pts)
  2. Skin Concern Matching        (0-25 pts)
  3. Ingredient Safety Profile    (0-20 pts)  [irritancy + comedogenicity]
  4. Age Group Suitability        (0-10 pts)
  5. Pregnancy Safety             (0-10 pts)
  6. Beneficial Ingredient Bonus  (0-10 pts)  [actives that help concerns]
  Penalties applied for: worsened concerns, unsafe pregnancy, wrong age,
                         high irritancy stacking, high comedogenicity
"""

import os
import re
import json
import time
import pickle
import logging
import warnings
import argparse
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ML
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.calc")

# =============================================================================
# CONFIGURATION
# =============================================================================
CFG = {
    "data_path"      : "../data/ingredient_profiles.csv",
    "output_dir"     : "../models/calculation-individual",
    "model_path"     : "../models/calculation-individual/xgb_model.pkl",
    "scaler_path"    : "../models/calculation-individual/scaler.pkl",
    "encoder_path"   : "../models/calculation-individual/encoders.pkl",
    "profiles_path"  : "../models/calculation-individual/ingredient_profiles.pkl",

    # Ensemble weights
    "rule_weight"    : 0.45,
    "ml_weight"      : 0.55,

    # Synthetic data generation
    "n_synthetic"    : 8000,
    "random_seed"    : 42,

    # XGBoost params
    "xgb_params": {
        "n_estimators"      : 500,
        "max_depth"         : 6,
        "learning_rate"     : 0.05,
        "subsample"         : 0.8,
        "colsample_bytree"  : 0.8,
        "min_child_weight"  : 3,
        "reg_alpha"         : 0.1,
        "reg_lambda"        : 1.0,
        "objective"         : "reg:squarederror",
        "eval_metric"       : "rmse",
        "random_state"      : 42,
        "n_jobs"            : -1,
    },

    # Scoring weights
    "weights": {
        "skin_type"     : 25,
        "skin_concern"  : 25,
        "safety"        : 20,
        "age"           : 10,
        "pregnancy"     : 10,
        "bonus"         : 10,
    },

    # Penalty values
    "penalties": {
        "concern_worsened"      : -12,   # ingredient worsens a user concern
        "high_irritancy"        : -8,    # high irritancy for sensitive skin
        "medium_irritancy_sens" : -4,    # medium irritancy on sensitive
        "comedogenic_4_5"       : -10,   # high comedogenicity on oily/acne
        "comedogenic_3"         : -5,    # moderate comedogenicity
        "comedogenic_2_oily"    : -2,    # mild comedogenicity on oily skin
        "pregnancy_unsafe"      : -20,   # definite no for pregnancy
        "pregnancy_consult"     : -5,    # needs consult for pregnancy
        "age_mismatch"          : -8,    # ingredient not for user age
        "skin_type_no"          : -15,   # ingredient explicitly bad for skin type
        "skin_type_caution"     : -5,    # caution for skin type
    },
}

# =============================================================================
# USER PROFILE SCHEMA
# =============================================================================

VALID_SKIN_TYPES    = ["oily", "dry", "combination", "normal", "sensitive", "mature"]
VALID_CONCERNS      = ["acne", "dryness", "aging", "hyperpigmentation",
                       "redness", "sensitivity", "pores", "texture", "dullness", "barrier"]
VALID_AGE_GROUPS    = ["teen", "adult", "mature"]
VALID_PREGNANCY     = ["yes", "no"]

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

class IngredientProfileDB:
    """
    Loads dataset2 and exposes fast O(1) profile lookup by INCI name.
    """
    def __init__(self, csv_path: str):
        self.df      = pd.read_csv(csv_path)
        self.lookup  = {}
        self._build_lookup()
        log.info(f"Loaded {len(self.df)} ingredient profiles from {csv_path}")

    def _parse_list(self, val: str) -> List[str]:
        if not val or str(val).strip().lower() in ("none", "nan", ""):
            return []
        return [v.strip().lower() for v in str(val).split("|") if v.strip()]

    def _parse_suitability(self, val: str) -> str:
        v = str(val).strip().lower()
        if v in ("yes", "1", "true"):    return "yes"
        if v in ("no",  "0", "false"):   return "no"
        if v == "moderate":              return "moderate"
        if v == "caution":               return "caution"
        return "yes"

    def _build_lookup(self):
        for _, row in self.df.iterrows():
            inci = str(row["inci_name"]).strip()
            profile = {
                "inci_name"         : inci,
                "category"          : str(row.get("ingredient_category", "")),
                "function"          : str(row.get("primary_function", "")),
                "skin_type"         : {
                    "oily"       : self._parse_suitability(row.get("suitable_oily",        "yes")),
                    "dry"        : self._parse_suitability(row.get("suitable_dry",         "yes")),
                    "combination": self._parse_suitability(row.get("suitable_combination", "yes")),
                    "normal"     : self._parse_suitability(row.get("suitable_normal",      "yes")),
                    "sensitive"  : self._parse_suitability(row.get("suitable_sensitive",   "yes")),
                    "mature"     : self._parse_suitability(row.get("suitable_mature",      "yes")),
                },
                "concerns_helps"    : self._parse_list(row.get("skin_concerns_helps", "")),
                "concerns_worsens"  : self._parse_list(row.get("skin_concerns_worsens", "")),
                "age_group"         : self._parse_list(row.get("age_group_suitable", "all")),
                "pregnancy_safe"    : str(row.get("pregnancy_safe", "yes")).strip().lower(),
                "irritancy"         : str(row.get("irritancy_potential", "low")).strip().lower(),
                "comedogenicity"    : self._safe_int(row.get("comedogenicity_0_to_5", 0)),
                "conc_min"          : self._safe_float(row.get("concentration_min_percent", 0)),
                "conc_max"          : self._safe_float(row.get("concentration_max_percent", 100)),
                "avoid_with"        : self._parse_list(row.get("avoid_combining_with", "")),
                "notes"             : str(row.get("usage_notes", "")),
            }
            self.lookup[inci.lower()] = profile

    def _safe_int(self, val) -> int:
        try:   return int(float(str(val)))
        except: return 0

    def _safe_float(self, val) -> float:
        try:   return float(str(val))
        except: return 0.0

    def get(self, inci_name: str) -> Optional[Dict]:
        return self.lookup.get(inci_name.strip().lower())

    def all_profiles(self) -> List[Dict]:
        return list(self.lookup.values())

    def all_names(self) -> List[str]:
        return list(self.lookup.keys())


# =============================================================================
# 2. RULE-BASED SCORING ENGINE
# =============================================================================

class RuleEngine:
    """
    Deterministic scoring based on dermatological rules.
    Returns a score 0-100 and structured breakdown.
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.weights  = cfg["weights"]
        self.penalties= cfg["penalties"]

    # ------------------------------------------------------------------
    def score_ingredient(
        self,
        profile     : Dict,
        skin_type   : str,
        concerns    : List[str],
        age_group   : str,
        is_pregnant : bool,
    ) -> Dict:
        """Score a single ingredient against user profile."""

        breakdown = {
            "skin_type_score"   : 0.0,
            "concern_score"     : 0.0,
            "safety_score"      : 0.0,
            "age_score"         : 0.0,
            "pregnancy_score"   : 0.0,
            "bonus_score"       : 0.0,
            "penalties"         : 0.0,
            "penalty_reasons"   : [],
            "pros"              : [],
            "cons"              : [],
        }

        # ── 1. Skin Type Compatibility (0-25) ─────────────────────────
        st_val = profile["skin_type"].get(skin_type, "yes")
        if st_val == "yes":
            breakdown["skin_type_score"] = self.weights["skin_type"]
            breakdown["pros"].append(
                f"Well-suited for {skin_type} skin ({profile['function']})"
            )
        elif st_val == "moderate":
            breakdown["skin_type_score"] = self.weights["skin_type"] * 0.65
        elif st_val == "caution":
            breakdown["skin_type_score"] = self.weights["skin_type"] * 0.35
            breakdown["cons"].append(f"Use with caution on {skin_type} skin")
            breakdown["penalties"] += self.penalties["skin_type_caution"]
            breakdown["penalty_reasons"].append(f"Caution flag for {skin_type} skin")
        elif st_val == "no":
            breakdown["skin_type_score"] = 0.0
            breakdown["cons"].append(
                f"Not recommended for {skin_type} skin — "
                f"may worsen skin condition"
            )
            breakdown["penalties"] += self.penalties["skin_type_no"]
            breakdown["penalty_reasons"].append(
                f"Explicitly unsuitable for {skin_type} skin"
            )

        # ── 2. Skin Concern Matching (0-25) ───────────────────────────
        helps_count   = 0
        worsens_count = 0
        concern_bonus = 0.0

        helps_raw    = profile["concerns_helps"]
        worsens_raw  = profile["concerns_worsens"]

        # Flatten worsens — strip parenthetical notes
        worsens_clean = []
        for w in worsens_raw:
            base = re.split(r"\s*\(", w)[0].strip()
            if base and base != "none":
                worsens_clean.append(base)

        helps_clean = []
        for h in helps_raw:
            base = re.split(r"\s*\(", h)[0].strip()
            if base and base != "none":
                helps_clean.append(base)

        for concern in concerns:
            c = concern.lower()
            if any(c in h for h in helps_clean):
                helps_count += 1
                breakdown["pros"].append(
                    f"Helps with {concern} ({profile['function']})"
                )
            if any(c in w for w in worsens_clean):
                worsens_count += 1
                breakdown["cons"].append(
                    f"May worsen {concern} — monitor carefully"
                )
                breakdown["penalties"] += self.penalties["concern_worsened"]
                breakdown["penalty_reasons"].append(
                    f"Ingredient may worsen {concern}"
                )

        if concerns:
            match_ratio = helps_count / len(concerns)
            breakdown["concern_score"] = self.weights["skin_concern"] * match_ratio
            concern_bonus = min(5.0, helps_count * 1.5)
        else:
            breakdown["concern_score"] = self.weights["skin_concern"] * 0.5

        # ── 3. Safety Profile (0-20) ──────────────────────────────────
        irritancy    = profile["irritancy"]
        comedogenic  = profile["comedogenicity"]

        # Irritancy scoring
        irritancy_score = {
            "low"   : self.weights["safety"],
            "medium": self.weights["safety"] * 0.6,
            "high"  : self.weights["safety"] * 0.25,
        }.get(irritancy, self.weights["safety"])
        breakdown["safety_score"] = irritancy_score

        if skin_type == "sensitive":
            if irritancy == "high":
                breakdown["penalties"]     += self.penalties["high_irritancy"]
                breakdown["penalty_reasons"].append("High irritancy ingredient on sensitive skin")
                breakdown["cons"].append(
                    "High irritancy potential — not ideal for sensitive skin"
                )
            elif irritancy == "medium":
                breakdown["penalties"]     += self.penalties["medium_irritancy_sens"]
                breakdown["penalty_reasons"].append("Medium irritancy on sensitive skin")
                breakdown["cons"].append("Moderate irritancy — patch test recommended")

        # Comedogenicity scoring
        acne_relevant = skin_type in ("oily", "combination") or "acne" in concerns
        if comedogenic >= 4:
            if acne_relevant:
                breakdown["penalties"]     += self.penalties["comedogenic_4_5"]
                breakdown["penalty_reasons"].append(
                    f"High comedogenicity ({comedogenic}/5) on acne/oily skin"
                )
                breakdown["cons"].append(
                    f"Comedogenicity rating {comedogenic}/5 — high risk of clogged pores"
                )
            else:
                breakdown["penalties"] += self.penalties["comedogenic_4_5"] * 0.4
                breakdown["cons"].append(
                    f"Comedogenicity rating {comedogenic}/5 — use sparingly"
                )
        elif comedogenic == 3:
            if acne_relevant:
                breakdown["penalties"]     += self.penalties["comedogenic_3"]
                breakdown["penalty_reasons"].append(f"Moderate comedogenicity ({comedogenic}/5)")
                breakdown["cons"].append(
                    f"Comedogenicity rating {comedogenic}/5 — moderate pore-clogging risk"
                )
        elif comedogenic == 2 and skin_type == "oily":
            breakdown["penalties"]     += self.penalties["comedogenic_2_oily"]
            breakdown["penalty_reasons"].append(f"Mild comedogenicity ({comedogenic}/5) on oily skin")

        if comedogenic == 0:
            breakdown["pros"].append("Non-comedogenic (0/5) — safe for pore-prone skin")
        elif comedogenic == 1:
            breakdown["pros"].append("Very low comedogenicity (1/5)")

        # ── 4. Age Group Suitability (0-10) ───────────────────────────
        age_groups = profile["age_group"]
        if "all" in age_groups or age_group in age_groups:
            breakdown["age_score"] = self.weights["age"]
            if age_group == "teen" and "teen" in age_groups:
                breakdown["pros"].append("Safe and appropriate for teen skin")
            elif age_group == "mature" and "mature" in age_groups:
                breakdown["pros"].append("Formulated for mature skin needs")
        else:
            breakdown["age_score"] = 0.0
            breakdown["penalties"]     += self.penalties["age_mismatch"]
            breakdown["penalty_reasons"].append(
                f"Not recommended for {age_group} age group "
                f"(suitable for: {', '.join(age_groups)})"
            )
            breakdown["cons"].append(
                f"Not typically recommended for {age_group} users"
            )

        # ── 5. Pregnancy Safety (0-10) ────────────────────────────────
        preg_safe = profile["pregnancy_safe"]
        if is_pregnant:
            if preg_safe == "yes":
                breakdown["pregnancy_score"] = self.weights["pregnancy"]
                breakdown["pros"].append("Pregnancy safe")
            elif preg_safe == "consult":
                breakdown["pregnancy_score"] = self.weights["pregnancy"] * 0.4
                breakdown["penalties"]     += self.penalties["pregnancy_consult"]
                breakdown["penalty_reasons"].append("Ingredient requires pregnancy consultation")
                breakdown["cons"].append(
                    "Consult your doctor before using during pregnancy"
                )
            elif preg_safe == "no":
                breakdown["pregnancy_score"] = 0.0
                breakdown["penalties"]     += self.penalties["pregnancy_unsafe"]
                breakdown["penalty_reasons"].append("Ingredient is NOT safe during pregnancy")
                breakdown["cons"].append(
                    "NOT recommended during pregnancy — avoid completely"
                )
        else:
            # Not pregnant — small bonus for generally safe ingredients
            if preg_safe == "yes":
                breakdown["pregnancy_score"] = self.weights["pregnancy"]
            elif preg_safe == "consult":
                breakdown["pregnancy_score"] = self.weights["pregnancy"] * 0.7
            else:
                breakdown["pregnancy_score"] = self.weights["pregnancy"] * 0.5

        # ── 6. Beneficial Bonus (0-10) ────────────────────────────────
        breakdown["bonus_score"] = min(self.weights["bonus"], concern_bonus)

        # ── Final Rule Score ──────────────────────────────────────────
        raw = (
            breakdown["skin_type_score"]
            + breakdown["concern_score"]
            + breakdown["safety_score"]
            + breakdown["age_score"]
            + breakdown["pregnancy_score"]
            + breakdown["bonus_score"]
            + breakdown["penalties"]
        )
        breakdown["raw_score"] = max(0.0, min(100.0, raw))
        return breakdown

    # ------------------------------------------------------------------
    def score_product(
        self,
        ingredients : List[Dict],           # list of ingredient profiles
        skin_type   : str,
        concerns    : List[str],
        age_group   : str,
        is_pregnant : bool,
    ) -> Dict:
        """Aggregate ingredient scores into a product-level rule score."""

        if not ingredients:
            return {"rule_score": 0.0, "details": [], "all_pros": [], "all_cons": []}

        scores       = []
        all_pros     = []
        all_cons     = []
        all_penalties= []
        details      = []

        for profile in ingredients:
            bd = self.score_ingredient(
                profile, skin_type, concerns, age_group, is_pregnant
            )
            scores.append(bd["raw_score"])
            all_pros.extend(bd["pros"])
            all_cons.extend(bd["cons"])
            all_penalties.extend(bd["penalty_reasons"])
            details.append({
                "ingredient"  : profile["inci_name"],
                "score"       : round(bd["raw_score"], 2),
                "pros"        : bd["pros"],
                "cons"        : bd["cons"],
                "penalties"   : bd["penalty_reasons"],
            })

        # Product score = weighted average
        # Key actives (non-humectants, non-preservatives) get higher weight
        weights = []
        for profile in ingredients:
            cat = profile.get("category", "").lower()
            if any(k in cat for k in ["active", "retinoid", "brightening", "anti-acne",
                                       "exfoliant", "peptide", "barrier", "uv filter"]):
                weights.append(2.0)
            elif any(k in cat for k in ["humectant", "emollient", "thickener",
                                         "preservative", "emulsifier"]):
                weights.append(0.7)
            else:
                weights.append(1.0)

        if sum(weights) > 0:
            rule_score = np.average(scores, weights=weights)
        else:
            rule_score = np.mean(scores)

        # Penalty stacking: each bad ingredient compounds from the first one.
        # Uses diminishing returns (6 / sqrt(i)) so that the first bad
        # ingredient is penalised hardest and additional ones add progressively
        # less — preventing a single outlier from being treated the same as
        # three genuinely incompatible ingredients.
        # Approx penalties:  1 bad→-6,  2 bad→-10,  3 bad→-14,  5 bad→-19
        bad_count = sum(1 for s in scores if s < 40)
        if bad_count > 0:
            stack_penalty = sum(6.0 / (i ** 0.5) for i in range(1, bad_count + 1))
            rule_score = max(0.0, rule_score - stack_penalty)

        return {
            "rule_score"  : float(np.clip(rule_score, 0, 100)),
            "details"     : details,
            "all_pros"    : list(dict.fromkeys(all_pros)),    # deduplicate
            "all_cons"    : list(dict.fromkeys(all_cons)),
            "all_penalties": list(dict.fromkeys(all_penalties)),
        }


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def encode_skin_type(skin_type: str) -> List[int]:
    return [int(skin_type == st) for st in VALID_SKIN_TYPES]

def encode_age(age_group: str) -> List[int]:
    return [int(age_group == ag) for ag in VALID_AGE_GROUPS]

def encode_concerns(concerns: List[str]) -> List[int]:
    return [int(c in concerns) for c in VALID_CONCERNS]

def extract_product_features(
    ingredients  : List[Dict],
    skin_type    : str,
    concerns     : List[str],
    age_group    : str,
    is_pregnant  : bool,
    rule_score   : float,
) -> np.ndarray:
    """
    Build a flat feature vector for the XGBoost model.
    """
    n = len(ingredients) if ingredients else 1

    # Aggregate ingredient statistics
    comedogenicity_vals = [p.get("comedogenicity", 0) for p in ingredients]
    irritancy_map       = {"low": 0, "medium": 1, "high": 2}
    irritancy_vals      = [irritancy_map.get(p.get("irritancy","low"), 0) for p in ingredients]

    # Skin type suitability counts
    st_yes  = sum(1 for p in ingredients if p["skin_type"].get(skin_type) == "yes")
    st_mod  = sum(1 for p in ingredients if p["skin_type"].get(skin_type) == "moderate")
    st_caut = sum(1 for p in ingredients if p["skin_type"].get(skin_type) == "caution")
    st_no   = sum(1 for p in ingredients if p["skin_type"].get(skin_type) == "no")

    # Concern matching
    helps_count   = 0
    worsens_count = 0
    for p in ingredients:
        helps_raw   = [re.split(r"\s*\(", h)[0].strip() for h in p.get("concerns_helps", [])]
        worsens_raw = [re.split(r"\s*\(", w)[0].strip() for w in p.get("concerns_worsens", [])]
        for c in concerns:
            if any(c.lower() in h for h in helps_raw):   helps_count   += 1
            if any(c.lower() in w for w in worsens_raw): worsens_count += 1

    # Pregnancy counts
    preg_yes     = sum(1 for p in ingredients if p.get("pregnancy_safe") == "yes")
    preg_consult = sum(1 for p in ingredients if p.get("pregnancy_safe") == "consult")
    preg_no      = sum(1 for p in ingredients if p.get("pregnancy_safe") == "no")

    # Age group matching
    age_match = sum(
        1 for p in ingredients
        if "all" in p.get("age_group", []) or age_group in p.get("age_group", [])
    )

    # Category distribution
    categories = [p.get("category", "").lower() for p in ingredients]
    n_actives      = sum(1 for c in categories if "active" in c)
    n_humectants   = sum(1 for c in categories if "humectant" in c)
    n_emollients   = sum(1 for c in categories if "emollient" in c)
    n_preservatives= sum(1 for c in categories if "preservative" in c)
    n_sunscreen    = sum(1 for c in categories if "uv filter" in c or "sunscreen" in c)

    feats = [
        # Rule score (most important)
        rule_score / 100.0,

        # Ingredient count features (normalized)
        n,
        n / max(n, 1),

        # Skin type
        st_yes   / n,
        st_mod   / n,
        st_caut  / n,
        st_no    / n,

        # Concern features
        helps_count   / max(n * max(len(concerns), 1), 1),
        worsens_count / max(n * max(len(concerns), 1), 1),
        len(concerns),
        helps_count,
        worsens_count,

        # Safety
        np.mean(comedogenicity_vals)  if comedogenicity_vals else 0,
        np.max(comedogenicity_vals)   if comedogenicity_vals else 0,
        np.mean(irritancy_vals)       if irritancy_vals     else 0,
        np.max(irritancy_vals)        if irritancy_vals     else 0,
        sum(1 for v in irritancy_vals if v == 2) / n,    # high irritancy ratio
        sum(1 for v in comedogenicity_vals if v >= 4) / n,  # high comed ratio

        # Pregnancy
        int(is_pregnant),
        preg_yes     / n,
        preg_consult / n,
        preg_no      / n,

        # Age
        age_match / n,

        # Category ratios
        n_actives       / n,
        n_humectants    / n,
        n_emollients    / n,
        n_preservatives / n,
        n_sunscreen     / n,

        # Skin type one-hot
        *encode_skin_type(skin_type),

        # Age one-hot
        *encode_age(age_group),

        # Concern one-hot
        *encode_concerns(concerns),
    ]

    return np.array(feats, dtype=np.float32)


# =============================================================================
# 4. SYNTHETIC TRAINING DATA GENERATION
# =============================================================================

def generate_synthetic_data(
    db         : IngredientProfileDB,
    rule_engine: RuleEngine,
    n_samples  : int,
    seed       : int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (features, score) pairs using the rule engine as ground truth.
    The ML model learns to refine rule scores with non-linear learned patterns.
    """
    rng     = np.random.default_rng(seed)
    names   = db.all_names()
    X_list  = []
    y_list  = []

    log.info(f"Generating {n_samples} synthetic training samples...")

    for i in tqdm(range(n_samples), desc="Synthetic data"):
        # Random user profile
        skin_type   = rng.choice(VALID_SKIN_TYPES)
        age_group   = rng.choice(VALID_AGE_GROUPS)
        is_pregnant = bool(rng.choice([True, False], p=[0.15, 0.85]))
        n_concerns  = int(rng.integers(0, 5))
        concerns    = list(rng.choice(VALID_CONCERNS, size=n_concerns, replace=False))

        # Random ingredient list (2-20 ingredients per product)
        n_ings    = int(rng.integers(2, 21))
        ing_names = rng.choice(names, size=min(n_ings, len(names)), replace=False)
        profiles  = [p for n in ing_names if (p := db.get(n)) is not None]

        if not profiles:
            continue

        # Rule score
        result      = rule_engine.score_product(
            profiles, skin_type, concerns, age_group, is_pregnant
        )
        rule_score  = result["rule_score"]

        # Add calibrated noise to create realistic variance
        # (real-world scores have uncertainty)
        noise = rng.normal(0, 3.0)
        label = float(np.clip(rule_score + noise, 0, 100))

        # Feature vector
        feat = extract_product_features(
            profiles, skin_type, concerns, age_group, is_pregnant, rule_score
        )

        X_list.append(feat)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    log.info(f"Generated {len(X)} samples | score range: [{y.min():.1f}, {y.max():.1f}]")
    return X, y


# =============================================================================
# 5. MODEL TRAINING
# =============================================================================

def train_ml_model(
    X     : np.ndarray,
    y     : np.ndarray,
    cfg   : dict,
) -> Tuple[xgb.XGBRegressor, StandardScaler, Dict]:
    """
    Train XGBoost regressor on synthetic data.
    Returns trained model, scaler, and evaluation metrics.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=cfg["random_seed"]
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    log.info(f"Training XGBoost on {len(X_train)} samples, validating on {len(X_val)}...")

    model = xgb.XGBRegressor(**cfg["xgb_params"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    y_pred     = model.predict(X_val)
    y_pred     = np.clip(y_pred, 0, 100)
    mae        = mean_absolute_error(y_val, y_pred)
    rmse       = np.sqrt(mean_squared_error(y_val, y_pred))
    r2         = r2_score(y_val, y_pred)

    metrics = {
        "mae"   : round(mae,  4),
        "rmse"  : round(rmse, 4),
        "r2"    : round(r2,   4),
        "n_train": len(X_train),
        "n_val"  : len(X_val),
    }

    log.info(f"XGBoost | MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

    # Feature importance
    importances = model.feature_importances_
    log.info(f"Top-5 feature importances: {sorted(enumerate(importances), key=lambda x: -x[1])[:5]}")

    return model, scaler, metrics


# =============================================================================
# 6. COMPATIBILITY SCORER (INFERENCE ENGINE)
# =============================================================================

class CompatibilityScorer:
    """
    Full inference pipeline: Rule Engine + XGBoost ensemble.

    Usage
    -----
    scorer = CompatibilityScorer.load("skinspectra_calc_model")
    result = scorer.score(
        ingredient_names=["Niacinamide", "Hyaluronic Acid", "Retinol"],
        skin_type="sensitive",
        concerns=["acne", "aging"],
        age_group="adult",
        is_pregnant=False
    )
    """

    def __init__(
        self,
        model       : xgb.XGBRegressor,
        scaler      : StandardScaler,
        db          : IngredientProfileDB,
        rule_engine : RuleEngine,
        cfg         : dict,
    ):
        self.model       = model
        self.scaler      = scaler
        self.db          = db
        self.rule_engine = rule_engine
        self.cfg         = cfg

    # ------------------------------------------------------------------
    def score(
        self,
        ingredient_names : List[str],
        skin_type        : str,
        concerns         : List[str],
        age_group        : str,
        is_pregnant      : bool,
    ) -> Dict:
        """
        Full compatibility score for a product against a user profile.

        Returns
        -------
        {
          "compatibility_score" : float  (0-100),
          "grade"               : str    (A+/A/B+/B/C+/C/D/F),
          "verdict"             : str,
          "pros"                : list,
          "cons"                : list,
          "ingredient_details"  : list,
          "warnings"            : list,
          "not_found"           : list,
          "rule_score"          : float,
          "ml_score"            : float,
          "latency_ms"          : float,
        }
        """
        t0          = time.perf_counter()
        skin_type   = skin_type.lower().strip()
        age_group   = age_group.lower().strip()
        concerns    = [c.lower().strip() for c in concerns]

        # Validate inputs
        self._validate(skin_type, concerns, age_group, is_pregnant)

        # Look up ingredient profiles
        profiles  = []
        not_found = []
        for name in ingredient_names:
            p = self.db.get(name)
            if p:
                profiles.append(p)
            else:
                not_found.append(name)

        if not profiles:
            return {
                "compatibility_score" : 0.0,
                "grade"               : "F",
                "verdict"             : "No recognisable ingredients found in database.",
                "pros"                : [],
                "cons"                : ["No ingredients could be evaluated"],
                "ingredient_details"  : [],
                "warnings"            : [],
                "not_found"           : not_found,
                "rule_score"          : 0.0,
                "ml_score"            : 0.0,
                "latency_ms"          : 0.0,
            }

        # Rule engine score
        rule_result = self.rule_engine.score_product(
            profiles, skin_type, concerns, age_group, is_pregnant
        )
        rule_score  = rule_result["rule_score"]

        # ML score
        feat     = extract_product_features(
            profiles, skin_type, concerns, age_group, is_pregnant, rule_score
        )
        feat_sc  = self.scaler.transform(feat.reshape(1, -1))
        ml_raw   = float(self.model.predict(feat_sc)[0])
        ml_score = float(np.clip(ml_raw, 0, 100))

        # Ensemble
        rw    = self.cfg["rule_weight"]
        mw    = self.cfg["ml_weight"]
        final = float(np.clip(rw * rule_score + mw * ml_score, 0, 100))
        final = round(final, 1)

        # Warnings
        warnings_list = self._build_warnings(
            profiles, skin_type, concerns, age_group, is_pregnant,
            rule_result["all_penalties"]
        )

        # Pros / Cons (deduplicated, limited)
        pros = rule_result["all_pros"][:8]
        cons = rule_result["all_cons"][:8]

        # Grade
        grade   = self._grade(final)
        verdict = self._verdict(final, skin_type, concerns)

        latency = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "compatibility_score" : final,
            "grade"               : grade,
            "verdict"             : verdict,
            "pros"                : pros,
            "cons"                : cons,
            "ingredient_details"  : rule_result["details"],
            "warnings"            : warnings_list,
            "not_found"           : not_found,
            "rule_score"          : round(rule_score, 1),
            "ml_score"            : round(ml_score,   1),
            "latency_ms"          : latency,
        }

    # ------------------------------------------------------------------
    def _validate(self, skin_type, concerns, age_group, is_pregnant):
        if skin_type not in VALID_SKIN_TYPES:
            raise ValueError(f"skin_type must be one of {VALID_SKIN_TYPES}")
        if age_group not in VALID_AGE_GROUPS:
            raise ValueError(f"age_group must be one of {VALID_AGE_GROUPS}")
        for c in concerns:
            if c not in VALID_CONCERNS:
                raise ValueError(f"concern '{c}' must be one of {VALID_CONCERNS}")

    def _grade(self, score: float) -> str:
        if score >= 92: return "A+"
        if score >= 85: return "A"
        if score >= 78: return "B+"
        if score >= 70: return "B"
        if score >= 62: return "C+"
        if score >= 55: return "C"
        if score >= 40: return "D"
        return "F"

    def _verdict(self, score: float, skin_type: str, concerns: List[str]) -> str:
        concern_str = ", ".join(concerns) if concerns else "general skin health"
        if score >= 85:
            return (f"Excellent compatibility ({score}/100). This product is highly "
                    f"recommended for your {skin_type} skin and {concern_str} concerns.")
        if score >= 70:
            return (f"Good compatibility ({score}/100). This product works well for "
                    f"your {skin_type} skin with minor cautions noted.")
        if score >= 55:
            return (f"Moderate compatibility ({score}/100). This product is acceptable "
                    f"but has some concerns for your {skin_type} skin — see details.")
        if score >= 40:
            return (f"Poor compatibility ({score}/100). This product has significant "
                    f"incompatibilities with your {skin_type} skin profile.")
        return (f"Very poor compatibility ({score}/100). This product is not "
                f"recommended for your skin type and concerns.")

    def _build_warnings(
        self, profiles, skin_type, concerns, age_group, is_pregnant, penalties
    ) -> List[str]:
        warnings_list = []

        # Pregnancy warnings
        if is_pregnant:
            unsafe = [p["inci_name"] for p in profiles if p.get("pregnancy_safe") == "no"]
            consult= [p["inci_name"] for p in profiles if p.get("pregnancy_safe") == "consult"]
            if unsafe:
                warnings_list.append(
                    f"PREGNANCY WARNING: The following ingredients should be "
                    f"AVOIDED during pregnancy: {', '.join(unsafe)}"
                )
            if consult:
                warnings_list.append(
                    f"PREGNANCY CAUTION: Consult your OB-GYN before using "
                    f"products containing: {', '.join(consult)}"
                )

        # High irritancy on sensitive
        if skin_type == "sensitive":
            irritants = [p["inci_name"] for p in profiles if p.get("irritancy") == "high"]
            if irritants:
                warnings_list.append(
                    f"SENSITIVITY WARNING: High-irritancy ingredients detected "
                    f"for sensitive skin: {', '.join(irritants)}"
                )

        # Highly comedogenic on oily/acne skin
        if skin_type in ("oily", "combination") or "acne" in concerns:
            high_comed = [
                p["inci_name"] for p in profiles
                if p.get("comedogenicity", 0) >= 4
            ]
            if high_comed:
                warnings_list.append(
                    f"ACNE WARNING: High-comedogenicity ingredients (4-5/5) "
                    f"found: {', '.join(high_comed)} — may clog pores"
                )

        # Teen warnings for strong actives
        if age_group == "teen":
            strong = [
                p["inci_name"] for p in profiles
                if p.get("irritancy") == "high" and "teen" not in p.get("age_group", [])
            ]
            if strong:
                warnings_list.append(
                    f"AGE WARNING: Strong actives not typically recommended "
                    f"for teen skin: {', '.join(strong)}"
                )

        return warnings_list

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_dir: str, data_path: str = None, cfg: dict = CFG) -> "CompatibilityScorer":
        model_dir  = Path(model_dir)
        model      = joblib.load(model_dir / "xgb_model.pkl")
        scaler     = joblib.load(model_dir / "scaler.pkl")
        data_path  = data_path or cfg["data_path"]
        db         = IngredientProfileDB(data_path)
        rule_engine= RuleEngine(cfg)
        log.info(f"CompatibilityScorer loaded from {model_dir}")
        return cls(model, scaler, db, rule_engine, cfg)


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SkinSpectra Calculation Layer Trainer")
    parser.add_argument("--data",      default=CFG["data_path"])
    parser.add_argument("--output",    default=CFG["output_dir"])
    parser.add_argument("--samples",   type=int, default=CFG["n_synthetic"])
    parser.add_argument("--seed",      type=int, default=CFG["random_seed"])
    args = parser.parse_args()

    CFG["data_path"]    = args.data
    CFG["output_dir"]   = args.output
    CFG["model_path"]   = f"{args.output}/xgb_model.pkl"
    CFG["scaler_path"]  = f"{args.output}/scaler.pkl"
    CFG["n_synthetic"]  = args.samples
    CFG["random_seed"]  = args.seed

    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  SkinSpectra -- Calculation Layer Training (Feature 1)")
    log.info("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────
    db          = IngredientProfileDB(CFG["data_path"])
    rule_engine = RuleEngine(CFG)

    # ── Step 2: Generate synthetic training data ───────────────────────
    X, y = generate_synthetic_data(db, rule_engine, CFG["n_synthetic"], CFG["random_seed"])

    # ── Step 3: Train XGBoost model ───────────────────────────────────
    model, scaler, metrics = train_ml_model(X, y, CFG)

    # ── Step 4: Save model artifacts ──────────────────────────────────
    joblib.dump(model,  output_dir / "xgb_model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Model artifacts saved to {output_dir}")

    # ── Step 5: Smoke test ────────────────────────────────────────────
    scorer = CompatibilityScorer(model, scaler, db, rule_engine, CFG)

    smoke_tests = [
        {
            "label"     : "Sensitive skin + acne-prone using harsh product",
            "ingredients": ["Coconut Oil", "Isopropyl Myristate", "Benzoyl Peroxide"],
            "skin_type" : "sensitive",
            "concerns"  : ["acne", "redness"],
            "age_group" : "adult",
            "is_pregnant": False,
        },
        {
            "label"     : "Dry mature skin ideal moisturizer",
            "ingredients": ["Hyaluronic Acid", "Ceramide NP", "Squalane", "Panthenol", "Niacinamide"],
            "skin_type" : "dry",
            "concerns"  : ["dryness", "aging"],
            "age_group" : "mature",
            "is_pregnant": False,
        },
        {
            "label"     : "Pregnancy-safe brightening routine",
            "ingredients": ["Niacinamide", "Azelaic Acid", "Aloe Barbadensis Leaf Juice", "Glycerin"],
            "skin_type" : "normal",
            "concerns"  : ["hyperpigmentation", "dullness"],
            "age_group" : "adult",
            "is_pregnant": True,
        },
        {
            "label"     : "Teen acne routine",
            "ingredients": ["Salicylic Acid", "Niacinamide", "Zinc PCA", "Aloe Barbadensis Leaf Juice"],
            "skin_type" : "oily",
            "concerns"  : ["acne", "pores"],
            "age_group" : "teen",
            "is_pregnant": False,
        },
        {
            "label"     : "Retinol on pregnant user (should flag warning)",
            "ingredients": ["Retinol", "Glycolic Acid", "Ascorbic Acid"],
            "skin_type" : "normal",
            "concerns"  : ["aging"],
            "age_group" : "adult",
            "is_pregnant": True,
        },
    ]

    log.info("\n-- Smoke Tests " + "-" * 45)
    for t in smoke_tests:
        r = scorer.score(
            ingredient_names = t["ingredients"],
            skin_type        = t["skin_type"],
            concerns         = t["concerns"],
            age_group        = t["age_group"],
            is_pregnant      = t["is_pregnant"],
        )
        log.info(f"\n  [{t['label']}]")
        log.info(f"  Score : {r['compatibility_score']}/100  Grade: {r['grade']}")
        log.info(f"  Rule  : {r['rule_score']}  ML: {r['ml_score']}")
        log.info(f"  Verdict: {r['verdict']}")
        log.info(f"  Pros  : {r['pros'][:3]}")
        log.info(f"  Cons  : {r['cons'][:3]}")
        if r["warnings"]:
            log.info(f"  WARNINGS: {r['warnings']}")
        if r["not_found"]:
            log.info(f"  Not Found: {r['not_found']}")

    log.info("\n Calculation Layer training complete.")
    log.info(f"  Metrics : {metrics}")
    log.info(f"  Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
"""
SkinSpectra Calculation Layer - Feature 2: Product Layering Compatibility
=========================================================================
Architecture:
  - Rule Engine  : Deterministic scoring engine using dataset3 layering pairs
                   + dataset2 individual ingredient profiles.
                   Evaluates interaction type, layering order, wait time,
                   time of day, skin type fit, concern impact, pregnancy,
                   age group, pH conflicts, and irritancy stacking.
  - ML Model     : LightGBM Regressor (faster than XGBoost for tabular data
                   with many categorical features; excels at interaction tasks)
                   Trained on synthetic ground-truth from rule engine.
  - Ensemble     : Final Score = 0.40 * rule_score + 0.60 * ml_score
  - Output       : 0-100 layering compatibility score + full report

Scoring Dimensions (per product pair)
--------------------------------------
  1.  Ingredient-pair interactions    (0-30 pts)  [synergistic/neutral/caution/conflict]
  2.  Layering order correctness      (0-15 pts)  [order enforced or flexible]
  3.  Skin type compatibility         (0-15 pts)  [both products suit skin type]
  4.  Skin concern alignment          (0-15 pts)  [combined impact on user concerns]
  5.  Time of day suitability         (0-10 pts)  [AM/PM/both]
  6.  Pregnancy & age safety          (0-10 pts)  [combined safety for user]
  7.  Wait time feasibility           (0-5 pts)   [wait time vs user patience]

Penalties
---------
  - Conflicting pair           : -25 pts
  - Avoid-same-time pair       : -18 pts
  - Caution pair (unmanaged)   : -8  pts
  - pH conflict stacking       : -10 pts
  - Irritancy double-stack     : -12 pts
  - Wrong layering order       : -10 pts
  - Pregnancy unsafe combo     : -20 pts
  - Age group mismatch         : -8  pts
  - Concern worsened by combo  : -10 pts each
  - Both products high irritancy on sensitive: -15 pts
"""

import re
import os
import json
import math
import time
import logging
import warnings
import argparse
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.layering")

# =============================================================================
# CONFIGURATION
# =============================================================================
CFG = {
    "dataset2_path"  : "../data/ingredient_profiles.csv",
    "dataset3_path"  : "../data/layering_compatibility.csv",
    "output_dir"     : "../models/calculation_layering",

    # Ensemble weights
    "rule_weight"    : 0.40,
    "ml_weight"      : 0.60,

    # Synthetic data
    "n_synthetic"    : 10000,
    "random_seed"    : 42,

    # LightGBM params
    "lgb_params": {
        "objective"        : "regression",
        "metric"           : "rmse",
        "n_estimators"     : 600,
        "learning_rate"    : 0.04,
        "max_depth"        : 7,
        "num_leaves"       : 63,
        "min_child_samples": 20,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 1.0,
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbose"          : -1,
    },

    # Scoring weights (must sum to 100)
    "weights": {
        "interaction"   : 30,
        "order"         : 15,
        "skin_type"     : 15,
        "concern"       : 15,
        "time_of_day"   : 10,
        "safety"        : 10,
        "wait_time"     : 5,
    },

    # Penalty values
    "penalties": {
        "conflicting"              : -30,   # inactivates both ingredients
        "avoid_same_time"          : -25,   # must be in different routines
        "caution_unmanaged"        : -12,   # caution pair layered anyway
        "ph_conflict"              : -12,   # pH renders ingredients ineffective
        "irritancy_double_stack"   : -15,   # two high-irritancy products stacked
        "wrong_order"              : -10,   # heavier before lighter
        "pregnancy_unsafe"         : -35,   # definite no for pregnancy (applied once, globally)
        "pregnancy_consult"        : -8,    # needs OB-GYN sign-off
        "age_mismatch"             : -18,   # ingredient not for this age group
        "concern_worsened"         : -12,   # combination worsens a user concern
        "both_high_irritancy_sens" : -20,   # double high irritancy on sensitive
        "skin_type_no_product"     : -15,   # product explicitly bad for skin type
        "skin_type_caution_product": -6,    # caution for skin type
    },
}

# Valid values
VALID_SKIN_TYPES = ["oily", "dry", "combination", "normal", "sensitive", "mature"]
VALID_CONCERNS   = ["acne", "dryness", "aging", "hyperpigmentation",
                    "redness", "sensitivity", "pores", "texture", "dullness", "barrier"]
VALID_AGE_GROUPS = ["teen", "adult", "mature"]
VALID_TOD        = ["AM", "PM", "both"]

INTERACTION_RANK = {
    "synergistic"          : 1.0,
    "complementary routines": 0.85,
    "neutral"              : 0.65,
    "caution"              : 0.35,
    "avoid same time"      : 0.15,
    "avoid"                : 0.10,
    "conflicting"          : 0.0,
}

# =============================================================================
# 1. DATABASE CLASSES
# =============================================================================

class IngredientProfileDB:
    """Loads dataset2 - individual ingredient profiles."""

    def __init__(self, csv_path: str):
        self.df     = pd.read_csv(csv_path)
        self.lookup : Dict[str, Dict] = {}
        self._build()
        log.info(f"IngredientProfileDB: {len(self.lookup)} profiles from {csv_path}")

    def _parse_list(self, val) -> List[str]:
        if not val or str(val).strip().lower() in ("none", "nan", ""):
            return []
        return [v.strip().lower() for v in str(val).split("|") if v.strip()]

    def _suit(self, val) -> str:
        v = str(val).strip().lower()
        if v in ("yes", "1", "true"):  return "yes"
        if v in ("no",  "0", "false"): return "no"
        if v == "moderate":            return "moderate"
        if v == "caution":             return "caution"
        return "yes"

    def _safe_int(self, v) -> int:
        try:   return int(float(str(v)))
        except: return 0

    def _build(self):
        for _, row in self.df.iterrows():
            inci = str(row["inci_name"]).strip()
            self.lookup[inci.lower()] = {
                "inci_name"      : inci,
                "category"       : str(row.get("ingredient_category", "")),
                "function"       : str(row.get("primary_function", "")),
                "skin_type"      : {
                    st: self._suit(row.get(f"suitable_{st}", "yes"))
                    for st in VALID_SKIN_TYPES
                },
                "concerns_helps"  : self._parse_list(row.get("skin_concerns_helps", "")),
                "concerns_worsens": self._parse_list(row.get("skin_concerns_worsens", "")),
                "age_group"       : self._parse_list(row.get("age_group_suitable", "all")),
                "pregnancy_safe"  : str(row.get("pregnancy_safe", "yes")).strip().lower(),
                "irritancy"       : str(row.get("irritancy_potential", "low")).strip().lower(),
                "comedogenicity"  : self._safe_int(row.get("comedogenicity_0_to_5", 0)),
            }

    def get(self, name: str) -> Optional[Dict]:
        return self.lookup.get(name.strip().lower())


class LayeringPairDB:
    """
    Loads dataset3 - ingredient pair layering data.
    Builds a bidirectional lookup: (ing1, ing2) -> pair_data
    """

    def __init__(self, csv_path: str):
        self.df   = pd.read_csv(csv_path)
        self.pairs: Dict[Tuple[str, str], Dict] = {}
        self._build()
        log.info(f"LayeringPairDB: {len(self.pairs)} directed pairs from {csv_path}")

    def _parse_list(self, val) -> List[str]:
        if not val or str(val).strip().lower() in ("none", "nan", ""):
            return []
        return [v.strip().lower() for v in str(val).split("|") if v.strip()]

    def _build(self):
        for _, row in self.df.iterrows():
            i1 = str(row["ingredient_1_inci"]).strip().lower()
            i2 = str(row["ingredient_2_inci"]).strip().lower()
            interaction = str(row.get("interaction_type", "neutral")).strip().lower()
            wait_raw    = str(row.get("wait_time_minutes", "0"))
            try:
                wait = int(float(wait_raw))
            except:
                wait = 0

            tod_raw = str(row.get("time_of_day", "both")).strip()

            data = {
                "ingredient_1"      : i1,
                "ingredient_2"      : i2,
                "interaction_type"  : interaction,
                "layering_order"    : str(row.get("layering_order", "either")).strip(),
                "wait_time_minutes" : wait,
                "time_of_day"       : tod_raw,
                "skin_oily"         : str(row.get("skin_type_oily",        "yes")).strip().lower(),
                "skin_dry"          : str(row.get("skin_type_dry",         "yes")).strip().lower(),
                "skin_sensitive"    : str(row.get("skin_type_sensitive",   "yes")).strip().lower(),
                "skin_combination"  : str(row.get("skin_type_combination", "yes")).strip().lower(),
                "skin_mature"       : str(row.get("skin_type_mature",      "yes")).strip().lower(),
                "concern_impact"    : str(row.get("skin_concern_impact",   "")).strip(),
                "age_group"         : self._parse_list(row.get("age_group", "all")),
                "pregnancy_notes"   : str(row.get("pregnancy_notes", "")).strip().lower(),
                "conflict_reason"   : str(row.get("conflict_reason",  "none")).strip(),
                "synergy_reason"    : str(row.get("synergy_reason",   "")).strip(),
                "application_notes" : str(row.get("application_notes","")).strip(),
            }
            # Store bidirectionally
            self.pairs[(i1, i2)] = data
            self.pairs[(i2, i1)] = data

    def get_pair(self, a: str, b: str) -> Optional[Dict]:
        a, b = a.strip().lower(), b.strip().lower()
        return self.pairs.get((a, b)) or self.pairs.get((b, a))

    def all_pairs(self) -> List[Dict]:
        seen: Set[Tuple] = set()
        result = []
        for (i1, i2), data in self.pairs.items():
            key = tuple(sorted([i1, i2]))
            if key not in seen:
                seen.add(key)
                result.append(data)
        return result


# =============================================================================
# 2. PRODUCT REPRESENTATION
# =============================================================================

class Product:
    """
    Represents a skincare product as a list of INCI ingredient names.
    Enriched with profile data from dataset2.
    """

    def __init__(self, name: str, ingredients: List[str], db: IngredientProfileDB):
        self.name        = name
        self.raw_ings    = ingredients
        self.profiles    : List[Dict]  = []
        self.not_found   : List[str]   = []

        for ing in ingredients:
            p = db.get(ing)
            if p:
                self.profiles.append(p)
            else:
                self.not_found.append(ing)

    # ------------------------------------------------------------------
    def get_skin_type_fit(self, skin_type: str) -> str:
        """
        Aggregate skin type suitability across all ingredients.
        Returns: 'yes' / 'moderate' / 'caution' / 'no'
        """
        rank = {"yes": 3, "moderate": 2, "caution": 1, "no": 0}
        if not self.profiles:
            return "yes"
        vals = [p["skin_type"].get(skin_type, "yes") for p in self.profiles]
        # Worst ingredient dominates
        worst = min(vals, key=lambda v: rank.get(v, 3))
        return worst

    def get_irritancy_level(self) -> str:
        irr_rank = {"low": 0, "medium": 1, "high": 2}
        if not self.profiles:
            return "low"
        levels = [irr_rank.get(p.get("irritancy","low"), 0) for p in self.profiles]
        mx = max(levels)
        return ["low","medium","high"][mx]

    def get_max_comedogenicity(self) -> int:
        if not self.profiles:
            return 0
        return max(p.get("comedogenicity", 0) for p in self.profiles)

    def get_pregnancy_safety(self) -> str:
        """Returns 'no' if any ingredient is unsafe, 'consult' if any needs consult."""
        vals = [p.get("pregnancy_safe","yes") for p in self.profiles]
        if "no" in vals:      return "no"
        if "consult" in vals: return "consult"
        return "yes"

    def get_age_groups(self) -> Set[str]:
        if not self.profiles:
            return {"all"}
        groups: Set[str] = set()
        for p in self.profiles:
            ag = p.get("age_group", ["all"])
            for g in ag:
                groups.add(g)
        return groups

    def get_concerns_helped(self) -> List[str]:
        helped: Set[str] = set()
        for p in self.profiles:
            for c in p.get("concerns_helps", []):
                base = re.split(r"\s*\(", c)[0].strip()
                if base and base != "none":
                    helped.add(base)
        return list(helped)

    def get_concerns_worsened(self) -> List[str]:
        worsened: Set[str] = set()
        for p in self.profiles:
            for c in p.get("concerns_worsens", []):
                base = re.split(r"\s*\(", c)[0].strip()
                if base and base != "none":
                    worsened.add(base)
        return list(worsened)


# =============================================================================
# 3. RULE ENGINE
# =============================================================================

class LayeringRuleEngine:
    """
    Deterministic scoring of two-product layering compatibility
    against a user skin profile.
    """

    def __init__(self, pair_db: LayeringPairDB, cfg: dict):
        self.pair_db = pair_db
        self.cfg     = cfg
        self.W       = cfg["weights"]
        self.P       = cfg["penalties"]

    # ------------------------------------------------------------------
    def score(
        self,
        product_a   : Product,
        product_b   : Product,
        skin_type   : str,
        concerns    : List[str],
        age_group   : str,
        is_pregnant : bool,
        time_of_day : str = "both",   # "AM" / "PM" / "both"
    ) -> Dict:
        """
        Score two products layered together against user profile.
        product_a is applied FIRST, product_b is applied SECOND.
        """
        breakdown = {
            "interaction_score"  : 0.0,
            "order_score"        : 0.0,
            "skin_type_score"    : 0.0,
            "concern_score"      : 0.0,
            "time_of_day_score"  : 0.0,
            "safety_score"       : 0.0,
            "wait_time_score"    : 0.0,
            "penalties"          : 0.0,
            "penalty_reasons"    : [],
            "pros"               : [],
            "cons"               : [],
            "pair_interactions"  : [],
            "wait_time_required" : 0,
            "layering_order_note": "",
            "application_steps"  : [],
        }

        # ── 1. Ingredient-pair interaction analysis (0-30) ─────────────
        inter_score, inter_details, max_wait, order_note, app_steps = \
            self._analyse_pairs(product_a, product_b, breakdown)
        breakdown["interaction_score"]   = inter_score
        breakdown["pair_interactions"]   = inter_details
        breakdown["wait_time_required"]  = max_wait
        breakdown["layering_order_note"] = order_note
        breakdown["application_steps"]   = app_steps

        # ── 2. Layering order score (0-15) ─────────────────────────────
        order_score = self._score_order(product_a, product_b, breakdown)
        breakdown["order_score"] = order_score

        # ── 3. Skin type compatibility (0-15) ──────────────────────────
        st_score = self._score_skin_type(product_a, product_b, skin_type, breakdown)
        breakdown["skin_type_score"] = st_score

        # ── 4. Skin concern alignment (0-15) ───────────────────────────
        concern_score = self._score_concerns(product_a, product_b, concerns, breakdown)
        breakdown["concern_score"] = concern_score

        # ── 5. Time of day (0-10) ──────────────────────────────────────
        tod_score = self._score_time_of_day(product_a, product_b, time_of_day, breakdown)
        breakdown["time_of_day_score"] = tod_score

        # ── 6. Safety: pregnancy + age (0-10) ──────────────────────────
        safety_score = self._score_safety(
            product_a, product_b, age_group, is_pregnant, breakdown
        )
        breakdown["safety_score"] = safety_score

        # ── 7. Wait time feasibility (0-5) ─────────────────────────────
        wt_score = self.W["wait_time"] if max_wait <= 20 else (
            self.W["wait_time"] * 0.5 if max_wait <= 45 else 0.0
        )
        breakdown["wait_time_score"] = wt_score

        # ── Irritancy double-stack penalty ─────────────────────────────
        irr_a = product_a.get_irritancy_level()
        irr_b = product_b.get_irritancy_level()
        if irr_a == "high" and irr_b == "high":
            breakdown["penalties"]       += self.P["irritancy_double_stack"]
            breakdown["penalty_reasons"].append(
                "Both products contain high-irritancy ingredients — stacking risk"
            )
            breakdown["cons"].append(
                "High irritancy stacking: both products have harsh ingredients. "
                "Risk of barrier disruption and inflammation."
            )
        elif skin_type == "sensitive" and irr_a == "high" and irr_b == "high":
            breakdown["penalties"]       += self.P["both_high_irritancy_sens"]
            breakdown["penalty_reasons"].append(
                "Double high-irritancy on sensitive skin — severe stacking risk"
            )

        # ── Final rule score ───────────────────────────────────────────
        raw = (
            breakdown["interaction_score"]
            + breakdown["order_score"]
            + breakdown["skin_type_score"]
            + breakdown["concern_score"]
            + breakdown["time_of_day_score"]
            + breakdown["safety_score"]
            + breakdown["wait_time_score"]
            + breakdown["penalties"]
        )
        breakdown["rule_score"] = float(np.clip(raw, 0.0, 100.0))
        return breakdown

    # ------------------------------------------------------------------
    def _analyse_pairs(
        self,
        product_a : Product,
        product_b : Product,
        breakdown : Dict,
    ) -> Tuple[float, List[Dict], int, str, List[str]]:
        """
        Cross-match all ingredient pairs between product_a and product_b.
        Returns (interaction_score, details, max_wait, order_note, app_steps)
        """
        details    : List[Dict] = []
        max_wait   : int        = 0
        order_note : str        = ""
        app_steps  : List[str]  = []

        ings_a = [p["inci_name"] for p in product_a.profiles]
        ings_b = [p["inci_name"] for p in product_b.profiles]

        # Cross-product ingredient pairs
        pair_scores  : List[float] = []
        best_pairs   : List[Dict]  = []
        worst_pairs  : List[Dict]  = []
        has_conflict : int = 0
        has_avoid    : int = 0
        has_caution  : int = 0

        for ia in ings_a:
            for ib in ings_b:
                pair = self.pair_db.get_pair(ia, ib)
                if not pair:
                    continue

                itype = pair["interaction_type"]
                rank  = INTERACTION_RANK.get(itype, 0.65)
                wait  = pair["wait_time_minutes"]
                max_wait = max(max_wait, wait)
                order  = pair["layering_order"]
                notes  = pair["application_notes"]
                syn    = pair["synergy_reason"]
                conf   = pair["conflict_reason"]

                pair_scores.append(rank)
                detail = {
                    "ingredient_a"      : ia,
                    "ingredient_b"      : ib,
                    "interaction_type"  : itype,
                    "interaction_rank"  : rank,
                    "wait_time_minutes" : wait,
                    "layering_order"    : order,
                    "notes"             : notes,
                }
                details.append(detail)

                # Pros / cons
                if itype == "synergistic":
                    best_pairs.append(detail)
                    breakdown["pros"].append(
                        f"{ia} + {ib}: Synergistic — {syn[:80] if syn else 'Enhanced combined effect'}"
                    )
                    if notes:
                        app_steps.append(notes)

                elif itype == "conflicting":
                    has_conflict += 1
                    worst_pairs.append(detail)
                    breakdown["penalties"]       += self.P["conflicting"]
                    breakdown["penalty_reasons"].append(
                        f"{ia} + {ib} conflict: {conf[:80] if conf else 'Incompatible combination'}"
                    )
                    breakdown["cons"].append(
                        f"CONFLICT: {ia} + {ib} — {conf[:80] if conf else 'These ingredients inactivate each other'}"
                    )

                elif itype == "avoid same time":
                    has_avoid += 1
                    worst_pairs.append(detail)
                    breakdown["penalties"]       += self.P["avoid_same_time"]
                    breakdown["penalty_reasons"].append(
                        f"{ia} + {ib}: Must be used at different times"
                    )
                    breakdown["cons"].append(
                        f"AVOID TOGETHER: {ia} + {ib} — use on alternate routines"
                    )

                elif itype == "caution":
                    has_caution += 1
                    breakdown["penalties"]       += self.P["caution_unmanaged"]
                    breakdown["penalty_reasons"].append(
                        f"{ia} + {ib}: Use with caution — {conf[:60] if conf else ''}"
                    )
                    breakdown["cons"].append(
                        f"Caution: {ia} + {ib} — {conf[:60] if conf else 'Use carefully'}"
                    )

                # pH conflict detection
                if "ph" in conf.lower() or "ph" in (pair.get("conflict_reason","").lower()):
                    breakdown["penalties"]       += self.P["ph_conflict"]
                    breakdown["penalty_reasons"].append(f"pH conflict: {ia} vs {ib}")

                # Order note from most important pair
                if order and order.lower() not in ("either", "same formula preferred", ""):
                    order_note = f"{order} — {notes[:80] if notes else ''}"

        # Compute overall interaction score
        if pair_scores:
            avg_rank = np.mean(pair_scores)
            min_rank = min(pair_scores)
            # Worst pair drives score: 45% avg + 55% min
            blended = 0.45 * avg_rank + 0.55 * min_rank

            # Hard caps based on worst interaction type found
            if has_conflict > 0:
                blended = min(blended, 0.20)   # conflict = hard cap at 20%
            elif has_avoid > 0:
                blended = min(blended, 0.30)   # avoid-same-time = cap at 30%
            elif has_caution > 0:
                blended = min(blended, 0.55)   # caution = cap at 55%

            inter_score = self.W["interaction"] * blended
        else:
            # No known pairs found — neutral assumption
            inter_score = self.W["interaction"] * 0.60
            breakdown["pros"].append(
                "No known ingredient conflicts detected between these products"
            )

        return inter_score, details, max_wait, order_note, app_steps

    # ------------------------------------------------------------------
    def _score_order(
        self,
        product_a : Product,
        product_b : Product,
        breakdown : Dict,
    ) -> float:
        """
        Score layering order based on texture/category rules:
        Correct order: lightest/wateriest first → heaviest last.
        Serum → Moisturizer → Oil → Occlusive
        """
        texture_rank = {
            "humectant"          : 1,
            "antioxidant"        : 2,
            "brightening"        : 2,
            "exfoliant"          : 2,
            "aha"                : 2,
            "bha"                : 2,
            "pha"                : 2,
            "retinoid"           : 3,
            "peptide"            : 3,
            "anti-aging"         : 3,
            "anti-acne"          : 3,
            "uv filter"          : 4,
            "sunscreen"          : 4,
            "emollient"          : 5,
            "barrier"            : 5,
            "occlusive"          : 6,
        }

        def product_rank(product: Product) -> float:
            if not product.profiles:
                return 3.0
            ranks = []
            for p in product.profiles:
                cat  = p.get("category", "").lower()
                func = p.get("function", "").lower()
                found = False
                for key, r in texture_rank.items():
                    if key in cat or key in func:
                        ranks.append(r)
                        found = True
                        break
                if not found:
                    ranks.append(3)
            return np.mean(ranks)

        rank_a = product_rank(product_a)
        rank_b = product_rank(product_b)

        if rank_a <= rank_b:
            # Correct order: A (lighter) then B (heavier)
            order_score = self.W["order"]
            breakdown["pros"].append(
                f"Correct layering order: {product_a.name} (lighter) "
                f"applied before {product_b.name} (heavier)"
            )
            breakdown["application_steps"].insert(
                0,
                f"Step 1: Apply {product_a.name} first "
                f"Step 2: Wait {max(breakdown.get('wait_time_required',0), 0)} min "
                f"Step 3: Apply {product_b.name}"
            )
        else:
            # Wrong order
            order_score = self.W["order"] * 0.3
            breakdown["penalties"]       += self.P["wrong_order"]
            breakdown["penalty_reasons"].append(
                f"Layering order may be incorrect: {product_a.name} appears heavier "
                f"than {product_b.name} — consider swapping"
            )
            breakdown["cons"].append(
                f"Order concern: {product_a.name} may be too heavy to layer before "
                f"{product_b.name}. Heavier products block absorption of lighter ones."
            )

        return order_score

    # ------------------------------------------------------------------
    def _score_skin_type(
        self,
        product_a : Product,
        product_b : Product,
        skin_type : str,
        breakdown : Dict,
    ) -> float:
        rank = {"yes": 1.0, "moderate": 0.65, "caution": 0.30, "no": 0.0}
        fit_a = product_a.get_skin_type_fit(skin_type)
        fit_b = product_b.get_skin_type_fit(skin_type)
        r_a   = rank.get(fit_a, 1.0)
        r_b   = rank.get(fit_b, 1.0)
        combined = (r_a + r_b) / 2.0
        score    = self.W["skin_type"] * combined

        # Penalties
        for pname, fit, rank_val in [
            (product_a.name, fit_a, r_a),
            (product_b.name, fit_b, r_b),
        ]:
            if fit == "no":
                breakdown["penalties"]       += self.P["skin_type_no_product"]
                breakdown["penalty_reasons"].append(
                    f"{pname} is NOT suitable for {skin_type} skin"
                )
                breakdown["cons"].append(
                    f"{pname} is not recommended for {skin_type} skin"
                )
            elif fit == "caution":
                breakdown["penalties"]       += self.P["skin_type_caution_product"]
                breakdown["penalty_reasons"].append(
                    f"{pname} requires caution on {skin_type} skin"
                )
                breakdown["cons"].append(
                    f"Use {pname} with caution on {skin_type} skin"
                )
            elif fit == "yes":
                breakdown["pros"].append(
                    f"{pname} is well-suited for {skin_type} skin"
                )

        return score

    # ------------------------------------------------------------------
    def _score_concerns(
        self,
        product_a : Product,
        product_b : Product,
        concerns  : List[str],
        breakdown : Dict,
    ) -> float:
        if not concerns:
            return self.W["concern"] * 0.5

        helped_a   = set(product_a.get_concerns_helped())
        helped_b   = set(product_b.get_concerns_helped())
        worsened_a = set(product_a.get_concerns_worsened())
        worsened_b = set(product_b.get_concerns_worsened())

        combined_helped   = helped_a | helped_b
        combined_worsened = worsened_a | worsened_b

        helps_count   = 0
        worsens_count = 0

        for c in concerns:
            c_lo = c.lower()
            if any(c_lo in h for h in combined_helped):
                helps_count += 1
                # Check if both products help
                both = (any(c_lo in h for h in helped_a) and
                        any(c_lo in h for h in helped_b))
                if both:
                    breakdown["pros"].append(
                        f"Both products address {c} — enhanced efficacy"
                    )
                else:
                    breakdown["pros"].append(
                        f"Combined routine addresses {c}"
                    )
            if any(c_lo in w for w in combined_worsened):
                worsens_count += 1
                breakdown["penalties"]       += self.P["concern_worsened"]
                breakdown["penalty_reasons"].append(
                    f"Combination may worsen {c}"
                )
                breakdown["cons"].append(
                    f"Caution: this combination may aggravate {c}"
                )

        match_ratio  = helps_count / len(concerns)
        concern_score = self.W["concern"] * match_ratio
        return concern_score

    # ------------------------------------------------------------------
    def _score_time_of_day(
        self,
        product_a   : Product,
        product_b   : Product,
        time_of_day : str,
        breakdown   : Dict,
    ) -> float:
        """
        Score based on whether the combined products are appropriate
        for the chosen time of day (AM/PM/both).
        """
        # Check pairs for explicit time-of-day conflicts
        ings_a = [p["inci_name"] for p in product_a.profiles]
        ings_b = [p["inci_name"] for p in product_b.profiles]

        tod_conflicts    : List[str] = []
        tod_requirements : Set[str]  = set()

        for ia in ings_a:
            for ib in ings_b:
                pair = self.pair_db.get_pair(ia, ib)
                if not pair:
                    continue
                pair_tod = pair.get("time_of_day", "both").upper()

                if "AM AND PM" in pair_tod or pair_tod == "BOTH":
                    tod_requirements.add("both")
                elif "SEPARATELY" in pair_tod or "SEPARATE" in pair_tod:
                    tod_conflicts.append(f"{ia} + {ib}: must be used in separate AM/PM routines")
                elif "AM" in pair_tod and "PM" not in pair_tod:
                    tod_requirements.add("AM")
                elif "PM" in pair_tod and "AM" not in pair_tod:
                    tod_requirements.add("PM")

        if tod_conflicts:
            breakdown["cons"].append(
                f"Time-of-day conflict: {'; '.join(tod_conflicts[:2])}"
            )
            # Partial penalty already captured in interaction analysis
            tod_score = self.W["time_of_day"] * 0.3
        elif not tod_requirements or "both" in tod_requirements:
            tod_score = self.W["time_of_day"]
            breakdown["pros"].append(
                f"This combination is suitable for {time_of_day} application"
            )
        elif time_of_day.upper() in tod_requirements:
            tod_score = self.W["time_of_day"]
            breakdown["pros"].append(
                f"Products correctly used in {time_of_day} routine"
            )
        else:
            tod_score = self.W["time_of_day"] * 0.5
            req_str   = " / ".join(tod_requirements)
            breakdown["cons"].append(
                f"These products are best used in {req_str} — not {time_of_day}"
            )

        return tod_score

    # ------------------------------------------------------------------
    def _score_safety(
        self,
        product_a   : Product,
        product_b   : Product,
        age_group   : str,
        is_pregnant : bool,
        breakdown   : Dict,
    ) -> float:
        score = self.W["safety"]

        # Pregnancy
        preg_a = product_a.get_pregnancy_safety()
        preg_b = product_b.get_pregnancy_safety()

        if is_pregnant:
            if preg_a == "no" or preg_b == "no":
                # Apply flat global penalty once (do NOT also modify score)
                breakdown["penalties"]        += self.P["pregnancy_unsafe"]
                unsafe = []
                if preg_a == "no": unsafe.append(product_a.name)
                if preg_b == "no": unsafe.append(product_b.name)
                # Extra penalty if BOTH products are unsafe
                if preg_a == "no" and preg_b == "no":
                    breakdown["penalties"]    += self.P["pregnancy_unsafe"] * 0.5
                breakdown["penalty_reasons"].append(
                    f"Pregnancy UNSAFE: {', '.join(unsafe)}"
                )
                breakdown["cons"].append(
                    f"PREGNANCY WARNING: {', '.join(unsafe)} should NOT be used "
                    f"during pregnancy. Discontinue immediately."
                )
            elif preg_a == "consult" or preg_b == "consult":
                breakdown["penalties"]        += self.P["pregnancy_consult"]
                consult = []
                if preg_a == "consult": consult.append(product_a.name)
                if preg_b == "consult": consult.append(product_b.name)
                # Both need consult = extra penalty
                if preg_a == "consult" and preg_b == "consult":
                    breakdown["penalties"]    += self.P["pregnancy_consult"]
                breakdown["penalty_reasons"].append(
                    f"Pregnancy CONSULT required: {', '.join(consult)}"
                )
                breakdown["cons"].append(
                    f"Pregnancy caution: Consult your OB-GYN before using "
                    f"{', '.join(consult)} during pregnancy."
                )
            else:
                breakdown["pros"].append(
                    "Both products are pregnancy-safe as a combination"
                )
        else:
            if preg_a == "yes" and preg_b == "yes":
                breakdown["pros"].append("Both products have excellent general safety profiles")

        # Age group — check at ingredient level (stricter)
        def age_ok_product(product, ag: str) -> Tuple[bool, List[str]]:
            """Returns (ok, list_of_mismatched_ingredient_names)"""
            mismatched = []
            for p in product.profiles:
                groups = set(p.get("age_group", ["all"]))
                if "all" not in groups and ag not in groups:
                    mismatched.append(p["inci_name"])
            return (len(mismatched) == 0), mismatched

        ok_a, mismatch_a = age_ok_product(product_a, age_group)
        ok_b, mismatch_b = age_ok_product(product_b, age_group)

        if not ok_a:
            n_mismatch = len(mismatch_a)
            # Scale penalty by how many ingredients mismatch
            penalty = self.P["age_mismatch"] * min(n_mismatch, 2)
            breakdown["penalties"]       += penalty
            breakdown["penalty_reasons"].append(
                f"{product_a.name}: {n_mismatch} ingredient(s) not for {age_group} "
                f"({', '.join(mismatch_a[:3])})"
            )
            breakdown["cons"].append(
                f"{product_a.name} contains ingredients not recommended for {age_group} users: "
                f"{', '.join(mismatch_a[:3])}"
            )

        if not ok_b:
            n_mismatch = len(mismatch_b)
            penalty = self.P["age_mismatch"] * min(n_mismatch, 2)
            breakdown["penalties"]       += penalty
            breakdown["penalty_reasons"].append(
                f"{product_b.name}: {n_mismatch} ingredient(s) not for {age_group} "
                f"({', '.join(mismatch_b[:3])})"
            )
            breakdown["cons"].append(
                f"{product_b.name} contains ingredients not recommended for {age_group} users: "
                f"{', '.join(mismatch_b[:3])}"
            )

        return max(0.0, min(self.W["safety"], score))


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def encode_st(st: str)  -> List[int]: return [int(st == s) for s in VALID_SKIN_TYPES]
def encode_ag(ag: str)  -> List[int]: return [int(ag == a) for a in VALID_AGE_GROUPS]
def encode_co(co: List[str]) -> List[int]: return [int(c in co) for c in VALID_CONCERNS]
def encode_tod(tod: str) -> List[int]: return [int(tod.upper() == t) for t in VALID_TOD]


def extract_features(
    product_a   : Product,
    product_b   : Product,
    pair_db     : LayeringPairDB,
    skin_type   : str,
    concerns    : List[str],
    age_group   : str,
    is_pregnant : bool,
    time_of_day : str,
    rule_score  : float,
) -> np.ndarray:
    n_a = max(len(product_a.profiles), 1)
    n_b = max(len(product_b.profiles), 1)

    irr_map = {"low": 0, "medium": 1, "high": 2}
    irr_a   = irr_map.get(product_a.get_irritancy_level(), 0)
    irr_b   = irr_map.get(product_b.get_irritancy_level(), 0)

    # Pair statistics
    all_pairs      = []
    wait_times     = []
    has_conflict   = 0
    has_synergy    = 0
    has_caution    = 0
    has_avoid      = 0
    has_ph_conflict= 0

    ings_a = [p["inci_name"] for p in product_a.profiles]
    ings_b = [p["inci_name"] for p in product_b.profiles]

    for ia in ings_a:
        for ib in ings_b:
            pair = pair_db.get_pair(ia, ib)
            if not pair:
                continue
            itype = pair["interaction_type"]
            all_pairs.append(INTERACTION_RANK.get(itype, 0.65))
            wait_times.append(pair.get("wait_time_minutes", 0))

            if itype == "conflicting":              has_conflict    += 1
            elif itype == "synergistic":            has_synergy     += 1
            elif itype == "caution":                has_caution     += 1
            elif "avoid" in itype:                  has_avoid       += 1

            conf = pair.get("conflict_reason","").lower()
            if "ph" in conf:                        has_ph_conflict += 1

    n_pairs    = max(len(all_pairs), 1)
    avg_rank   = np.mean(all_pairs)   if all_pairs  else 0.65
    min_rank   = min(all_pairs)       if all_pairs  else 0.65
    max_wait   = max(wait_times)      if wait_times else 0

    # Concern overlap
    helped_a  = set(product_a.get_concerns_helped())
    helped_b  = set(product_b.get_concerns_helped())
    both_help = sum(
        1 for c in concerns
        if any(c in h for h in helped_a) and any(c in h for h in helped_b)
    )
    any_help  = sum(
        1 for c in concerns
        if any(c in h for h in (helped_a | helped_b))
    )

    preg_a    = {"yes": 2, "consult": 1, "no": 0}.get(product_a.get_pregnancy_safety(), 2)
    preg_b    = {"yes": 2, "consult": 1, "no": 0}.get(product_b.get_pregnancy_safety(), 2)

    st_fit_a  = {"yes": 1.0, "moderate": 0.65, "caution": 0.30, "no": 0.0}.get(
        product_a.get_skin_type_fit(skin_type), 1.0)
    st_fit_b  = {"yes": 1.0, "moderate": 0.65, "caution": 0.30, "no": 0.0}.get(
        product_b.get_skin_type_fit(skin_type), 1.0)

    feats = [
        rule_score / 100.0,           # rule score (normalised)
        n_a, n_b,                      # ingredient counts
        avg_rank,                      # avg interaction rank
        min_rank,                      # worst pair rank
        n_pairs,                       # known pairs found
        has_conflict,                  # conflict count
        has_synergy,                   # synergy count
        has_caution,                   # caution count
        has_avoid,                     # avoid count
        has_ph_conflict,               # pH conflict count
        max_wait,                      # max wait time needed
        irr_a, irr_b,                  # irritancy levels
        irr_a + irr_b,                 # combined irritancy
        int(irr_a == 2 and irr_b == 2),# both high irritancy flag
        product_a.get_max_comedogenicity(),
        product_b.get_max_comedogenicity(),
        st_fit_a, st_fit_b,            # skin type fit
        (st_fit_a + st_fit_b) / 2,     # avg skin type fit
        preg_a, preg_b,                # pregnancy safety
        int(is_pregnant),
        int(is_pregnant and (preg_a == 0 or preg_b == 0)),  # pregnant + unsafe
        len(concerns),
        both_help,                     # concerns both products help
        any_help,                      # concerns any product helps
        both_help / max(len(concerns), 1),
        any_help  / max(len(concerns), 1),
        *encode_st(skin_type),         # 6 features
        *encode_ag(age_group),         # 3 features
        *encode_co(concerns),          # 10 features
        *encode_tod(time_of_day),      # 3 features
    ]

    return np.array(feats, dtype=np.float32)


# =============================================================================
# 5. SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_data(
    pair_db     : LayeringPairDB,
    profile_db  : IngredientProfileDB,
    rule_engine : LayeringRuleEngine,
    n_samples   : int,
    seed        : int = 42,
) -> Tuple[np.ndarray, np.ndarray]:

    rng        = np.random.default_rng(seed)
    all_names  = list(profile_db.lookup.keys())
    tod_opts   = ["AM", "PM", "both"]

    X_list, y_list = [], []
    log.info(f"Generating {n_samples} synthetic layering samples...")

    for _ in tqdm(range(n_samples), desc="Synthetic layering data"):
        skin_type   = rng.choice(VALID_SKIN_TYPES)
        age_group   = rng.choice(VALID_AGE_GROUPS)
        is_pregnant = bool(rng.choice([True, False], p=[0.12, 0.88]))
        n_concerns  = int(rng.integers(0, 5))
        concerns    = list(rng.choice(VALID_CONCERNS, size=n_concerns, replace=False))
        time_of_day = rng.choice(tod_opts)

        n_a = int(rng.integers(1, 10))
        n_b = int(rng.integers(1, 10))
        ings_a = list(rng.choice(all_names, size=min(n_a, len(all_names)), replace=False))
        ings_b = list(rng.choice(all_names, size=min(n_b, len(all_names)), replace=False))

        prod_a = Product("ProductA", [profile_db.lookup[n]["inci_name"] for n in ings_a], profile_db)
        prod_b = Product("ProductB", [profile_db.lookup[n]["inci_name"] for n in ings_b], profile_db)

        if not prod_a.profiles and not prod_b.profiles:
            continue

        result     = rule_engine.score(
            prod_a, prod_b, skin_type, concerns, age_group, is_pregnant, time_of_day
        )
        rule_score = result["rule_score"]
        noise      = rng.normal(0, 3.5)
        label      = float(np.clip(rule_score + noise, 0, 100))

        feat = extract_features(
            prod_a, prod_b, pair_db, skin_type, concerns,
            age_group, is_pregnant, time_of_day, rule_score
        )
        X_list.append(feat)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    log.info(f"Generated {len(X)} samples | score range [{y.min():.1f}, {y.max():.1f}]")
    return X, y


# =============================================================================
# 6. MODEL TRAINING
# =============================================================================

def train_lgb_model(
    X   : np.ndarray,
    y   : np.ndarray,
    cfg : dict,
) -> Tuple[lgb.LGBMRegressor, StandardScaler, Dict]:

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=cfg["random_seed"]
    )
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_tr)
    X_val   = scaler.transform(X_val)

    log.info(f"Training LightGBM on {len(X_tr)} samples | validating on {len(X_val)}...")
    params  = {**cfg["lgb_params"]}
    model   = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    y_pred  = np.clip(model.predict(X_val), 0, 100)
    mae     = mean_absolute_error(y_val, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_val, y_pred))
    r2      = r2_score(y_val, y_pred)
    metrics = {"mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4),
               "n_train": len(X_tr), "n_val": len(X_val)}
    log.info(f"LightGBM | MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")
    return model, scaler, metrics


# =============================================================================
# 7. LAYERING SCORER (INFERENCE ENGINE)
# =============================================================================

class LayeringScorer:
    """
    Full two-product layering compatibility scorer.

    Usage
    -----
    scorer = LayeringScorer.load("skinspectra_layering_model")
    result = scorer.score(
        product_a_name       = "Vitamin C Serum",
        product_a_ingredients= ["Ascorbic Acid", "Ferulic Acid", "Glycerin"],
        product_b_name       = "Retinol Night Cream",
        product_b_ingredients= ["Retinol", "Ceramide NP", "Hyaluronic Acid"],
        skin_type   = "normal",
        concerns    = ["aging", "hyperpigmentation"],
        age_group   = "adult",
        is_pregnant = False,
        time_of_day = "PM",
    )
    """

    def __init__(
        self,
        model        : lgb.LGBMRegressor,
        scaler       : StandardScaler,
        profile_db   : IngredientProfileDB,
        pair_db      : LayeringPairDB,
        rule_engine  : LayeringRuleEngine,
        cfg          : dict,
    ):
        self.model       = model
        self.scaler      = scaler
        self.profile_db  = profile_db
        self.pair_db     = pair_db
        self.rule_engine = rule_engine
        self.cfg         = cfg

    # ------------------------------------------------------------------
    def score(
        self,
        product_a_name        : str,
        product_a_ingredients : List[str],
        product_b_name        : str,
        product_b_ingredients : List[str],
        skin_type             : str,
        concerns              : List[str],
        age_group             : str,
        is_pregnant           : bool,
        time_of_day           : str = "both",
    ) -> Dict:
        """
        Returns
        -------
        {
          "layering_score"      : float (0-100),
          "grade"               : str,
          "verdict"             : str,
          "layering_order"      : str,  (which product goes first)
          "wait_time_minutes"   : int,
          "application_steps"   : list,
          "pros"                : list,
          "cons"                : list,
          "warnings"            : list,
          "pair_interactions"   : list,
          "product_a_not_found" : list,
          "product_b_not_found" : list,
          "rule_score"          : float,
          "ml_score"            : float,
          "latency_ms"          : float,
        }
        """
        t0          = time.perf_counter()
        skin_type   = skin_type.lower().strip()
        age_group   = age_group.lower().strip()
        concerns    = [c.lower().strip() for c in concerns]
        time_of_day = time_of_day.upper().strip()

        # Validate
        self._validate(skin_type, concerns, age_group)

        prod_a = Product(product_a_name, product_a_ingredients, self.profile_db)
        prod_b = Product(product_b_name, product_b_ingredients, self.profile_db)

        # Rule engine
        rule_result  = self.rule_engine.score(
            prod_a, prod_b, skin_type, concerns, age_group, is_pregnant, time_of_day
        )
        rule_score   = rule_result["rule_score"]

        # ML score
        feat         = extract_features(
            prod_a, prod_b, self.pair_db, skin_type, concerns,
            age_group, is_pregnant, time_of_day, rule_score
        )
        feat_sc      = self.scaler.transform(feat.reshape(1, -1))
        ml_raw       = float(self.model.predict(feat_sc)[0])
        ml_score     = float(np.clip(ml_raw, 0, 100))

        # Ensemble
        rw     = self.cfg["rule_weight"]
        mw     = self.cfg["ml_weight"]
        final  = float(np.clip(rw * rule_score + mw * ml_score, 0, 100))
        final  = round(final, 1)

        # Build warnings
        warnings_list = self._build_warnings(
            prod_a, prod_b, skin_type, concerns, is_pregnant, rule_result
        )

        # Deduplicate pros/cons
        pros = list(dict.fromkeys(rule_result["pros"]))[:8]
        cons = list(dict.fromkeys(rule_result["cons"]))[:8]

        # Layering order recommendation
        order_note = rule_result.get("layering_order_note", "")
        if not order_note:
            order_note = (
                f"Apply {product_a_name} first, then {product_b_name}. "
                f"Wait {rule_result['wait_time_required']} minutes between applications."
                if rule_result["wait_time_required"] > 0
                else f"Apply {product_a_name} first, then {product_b_name}."
            )

        # Application steps
        steps = rule_result.get("application_steps", [])
        if not steps:
            wait = rule_result["wait_time_required"]
            steps = [
                f"1. Cleanse and prep skin.",
                f"2. Apply {product_a_name}.",
                f"3. {'Wait ' + str(wait) + ' minutes.' if wait > 0 else 'Allow to absorb briefly.'}",
                f"4. Apply {product_b_name}.",
                f"5. Follow with SPF if using AM routine.",
            ]

        grade   = self._grade(final)
        verdict = self._verdict(
            final, product_a_name, product_b_name, skin_type, concerns
        )
        latency = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "layering_score"       : final,
            "grade"                : grade,
            "verdict"              : verdict,
            "layering_order"       : order_note,
            "wait_time_minutes"    : rule_result["wait_time_required"],
            "application_steps"    : steps,
            "pros"                 : pros,
            "cons"                 : cons,
            "warnings"             : warnings_list,
            "pair_interactions"    : rule_result["pair_interactions"][:10],
            "product_a_not_found"  : prod_a.not_found,
            "product_b_not_found"  : prod_b.not_found,
            "rule_score"           : round(rule_score, 1),
            "ml_score"             : round(ml_score,   1),
            "latency_ms"           : latency,
        }

    # ------------------------------------------------------------------
    def _validate(self, skin_type, concerns, age_group):
        if skin_type not in VALID_SKIN_TYPES:
            raise ValueError(f"skin_type must be one of {VALID_SKIN_TYPES}")
        if age_group not in VALID_AGE_GROUPS:
            raise ValueError(f"age_group must be one of {VALID_AGE_GROUPS}")
        for c in concerns:
            if c not in VALID_CONCERNS:
                raise ValueError(f"concern '{c}' not in {VALID_CONCERNS}")

    def _grade(self, score: float) -> str:
        if score >= 92: return "A+"
        if score >= 85: return "A"
        if score >= 78: return "B+"
        if score >= 70: return "B"
        if score >= 62: return "C+"
        if score >= 55: return "C"
        if score >= 40: return "D"
        return "F"

    def _verdict(self, score, pa, pb, skin_type, concerns) -> str:
        c_str = ", ".join(concerns) if concerns else "general skin health"
        if score >= 85:
            return (f"Excellent layering compatibility ({score}/100). "
                    f"{pa} and {pb} work exceptionally well together for "
                    f"{skin_type} skin targeting {c_str}.")
        if score >= 70:
            return (f"Good compatibility ({score}/100). "
                    f"{pa} and {pb} layer well together with minor cautions.")
        if score >= 55:
            return (f"Moderate compatibility ({score}/100). "
                    f"These products can be layered but require careful application — "
                    f"see recommendations below.")
        if score >= 40:
            return (f"Poor layering compatibility ({score}/100). "
                    f"Significant conflicts exist between {pa} and {pb}. "
                    f"Consider using on alternate days instead.")
        return (f"Very poor compatibility ({score}/100). "
                f"Do NOT layer {pa} and {pb} together. "
                f"Use in completely separate AM/PM routines or on alternate days.")

    def _build_warnings(
        self, prod_a, prod_b, skin_type, concerns, is_pregnant, rule_result
    ) -> List[str]:
        warnings = []

        # Pregnancy
        if is_pregnant:
            for prod in [prod_a, prod_b]:
                if prod.get_pregnancy_safety() == "no":
                    warnings.append(
                        f"PREGNANCY DANGER: {prod.name} contains ingredients "
                        f"unsafe during pregnancy — STOP USE IMMEDIATELY."
                    )
                elif prod.get_pregnancy_safety() == "consult":
                    warnings.append(
                        f"PREGNANCY CAUTION: Consult your OB-GYN before using "
                        f"{prod.name} during pregnancy."
                    )

        # Conflicting pairs
        conflicts = [
            d for d in rule_result.get("pair_interactions", [])
            if d["interaction_type"] == "conflicting"
        ]
        if conflicts:
            pairs_str = ", ".join(
                f"{d['ingredient_a']} + {d['ingredient_b']}" for d in conflicts[:3]
            )
            warnings.append(
                f"INGREDIENT CONFLICT: {pairs_str} — these pairs inactivate "
                f"each other. Do NOT use same application."
            )

        # Avoid-same-time pairs
        avoids = [
            d for d in rule_result.get("pair_interactions", [])
            if "avoid" in d["interaction_type"]
        ]
        if avoids:
            warnings.append(
                f"TIMING WARNING: Some ingredients require separate AM/PM applications. "
                f"Do not apply both products in the same routine step."
            )

        # Irritancy on sensitive
        irr_a = prod_a.get_irritancy_level()
        irr_b = prod_b.get_irritancy_level()
        if skin_type == "sensitive" and irr_a == "high" and irr_b == "high":
            warnings.append(
                f"SENSITIVITY ALERT: Both products contain high-irritancy ingredients. "
                f"Layering on sensitive skin may cause irritation, redness, or burns."
            )

        # Wait time
        wait = rule_result.get("wait_time_required", 0)
        if wait > 0:
            warnings.append(
                f"WAIT TIME: Allow {wait} minutes between applying these products "
                f"to allow pH normalisation and absorption."
            )

        return warnings

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        model_dir      : str,
        dataset2_path  : str = None,
        dataset3_path  : str = None,
        cfg            : dict = CFG,
    ) -> "LayeringScorer":
        model_dir     = Path(model_dir)
        model         = joblib.load(model_dir / "lgb_model.pkl")
        scaler        = joblib.load(model_dir / "scaler.pkl")
        d2            = dataset2_path or cfg["dataset2_path"]
        d3            = dataset3_path or cfg["dataset3_path"]
        profile_db    = IngredientProfileDB(d2)
        pair_db       = LayeringPairDB(d3)
        rule_engine   = LayeringRuleEngine(pair_db, cfg)
        log.info(f"LayeringScorer loaded from {model_dir}")
        return cls(model, scaler, profile_db, pair_db, rule_engine, cfg)


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SkinSpectra Layering Layer Trainer")
    parser.add_argument("--dataset2", default=CFG["dataset2_path"])
    parser.add_argument("--dataset3", default=CFG["dataset3_path"])
    parser.add_argument("--output",   default=CFG["output_dir"])
    parser.add_argument("--samples",  type=int, default=CFG["n_synthetic"])
    parser.add_argument("--seed",     type=int, default=CFG["random_seed"])
    args = parser.parse_args()

    CFG["dataset2_path"] = args.dataset2
    CFG["dataset3_path"] = args.dataset3
    CFG["output_dir"]    = args.output
    CFG["n_synthetic"]   = args.samples
    CFG["random_seed"]   = args.seed

    out = Path(CFG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  SkinSpectra -- Layering Layer Training (Feature 2)")
    log.info("=" * 60)

    # Step 1: Load data
    profile_db  = IngredientProfileDB(CFG["dataset2_path"])
    pair_db     = LayeringPairDB(CFG["dataset3_path"])
    rule_engine = LayeringRuleEngine(pair_db, CFG)

    # Step 2: Synthetic data
    X, y = generate_synthetic_data(
        pair_db, profile_db, rule_engine, CFG["n_synthetic"], CFG["random_seed"]
    )

    # Step 3: Train LightGBM
    model, scaler, metrics = train_lgb_model(X, y, CFG)

    # Step 4: Save
    joblib.dump(model,  out / "lgb_model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved to {out}")

    # Step 5: Smoke tests
    scorer = LayeringScorer(model, scaler, profile_db, pair_db, rule_engine, CFG)
    smoke_tests = [
        {
            "label"  : "Classic conflict: Vitamin C + Retinol same routine",
            "a_name" : "Vitamin C Serum",
            "a_ings" : ["Ascorbic Acid", "Ferulic Acid", "Glycerin"],
            "b_name" : "Retinol Night Cream",
            "b_ings" : ["Retinol", "Ceramide NP", "Hyaluronic Acid"],
            "st"     : "normal", "concerns": ["aging", "hyperpigmentation"],
            "ag"     : "adult", "preg": False, "tod": "PM",
        },
        {
            "label"  : "Ideal synergistic layering: HA serum + Ceramide moisturizer",
            "a_name" : "HA Serum",
            "a_ings" : ["Hyaluronic Acid", "Sodium PCA", "Glycerin"],
            "b_name" : "Ceramide Moisturizer",
            "b_ings" : ["Ceramide NP", "Ceramide AP", "Cholesterol", "Squalane"],
            "st"     : "dry", "concerns": ["dryness", "barrier"],
            "ag"     : "adult", "preg": False, "tod": "both",
        },
        {
            "label"  : "Pregnancy safe brightening duo",
            "a_name" : "Niacinamide Serum",
            "a_ings" : ["Niacinamide", "Tranexamic Acid", "Hyaluronic Acid"],
            "b_name" : "Azelaic Acid Treatment",
            "b_ings" : ["Azelaic Acid", "Glycerin", "Allantoin"],
            "st"     : "sensitive", "concerns": ["hyperpigmentation", "redness"],
            "ag"     : "adult", "preg": True, "tod": "both",
        },
        {
            "label"  : "Conflicting acids + retinol on sensitive (worst case)",
            "a_name" : "Glycolic Acid Toner",
            "a_ings" : ["Glycolic Acid", "Salicylic Acid"],
            "b_name" : "Retinol Serum",
            "b_ings" : ["Retinol", "Benzoyl Peroxide"],
            "st"     : "sensitive", "concerns": ["acne", "aging"],
            "ag"     : "teen", "preg": False, "tod": "PM",
        },
        {
            "label"  : "Teen acne routine: BHA toner + niacinamide moisturizer",
            "a_name" : "BHA Toner",
            "a_ings" : ["Salicylic Acid", "Zinc PCA", "Aloe Barbadensis Leaf Juice"],
            "b_name" : "Niacinamide Moisturizer",
            "b_ings" : ["Niacinamide", "Ceramide NP", "Hyaluronic Acid"],
            "st"     : "oily", "concerns": ["acne", "pores"],
            "ag"     : "teen", "preg": False, "tod": "AM",
        },
    ]

    log.info("\n-- Smoke Tests " + "-" * 45)
    for t in smoke_tests:
        r = scorer.score(
            t["a_name"], t["a_ings"], t["b_name"], t["b_ings"],
            t["st"], t["concerns"], t["ag"], t["preg"], t["tod"]
        )
        log.info(f"\n  [{t['label']}]")
        log.info(f"  Score : {r['layering_score']}/100  Grade: {r['grade']}")
        log.info(f"  Rule  : {r['rule_score']}  ML: {r['ml_score']}")
        log.info(f"  Order : {r['layering_order']}")
        log.info(f"  Wait  : {r['wait_time_minutes']} min")
        log.info(f"  Verdict: {r['verdict']}")
        log.info(f"  Pros  : {r['pros'][:2]}")
        log.info(f"  Cons  : {r['cons'][:2]}")
        if r["warnings"]:
            log.info(f"  WARNINGS: {r['warnings'][:2]}")

    log.info(f"\n Layering Layer training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
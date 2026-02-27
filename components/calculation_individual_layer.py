import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

CFG = {
    "data_path"      : "../data/ingredient_profiles.csv",
    "output_dir"     : "../models/calculation_individual",
    "model_path"     : "../models/calculation_individual/xgb_model.pkl",
    "scaler_path"    : "../models/calculation_individual/scaler.pkl",
    "encoder_path"   : "../models/calculation_individual/encoders.pkl",
    "profiles_path"  : "../models/calculation_individual/ingredient_profiles.pkl",
    "rule_weight"    : 0.45,
    "ml_weight"      : 0.55,
    "n_synthetic"    : 8000,
    "random_seed"    : 42,
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
    "weights": {
        "skin_type"     : 25,
        "skin_concern"  : 25,
        "safety"        : 20,
        "age"           : 10,
        "pregnancy"     : 10,
        "bonus"         : 10,
    },
    "penalties": {
        "concern_worsened"      : -12,
        "high_irritancy"        : -8,
        "medium_irritancy_sens" : -4,
        "comedogenic_4_5"       : -10,
        "comedogenic_3"         : -5,
        "comedogenic_2_oily"    : -2,
        "pregnancy_unsafe"      : -20,
        "pregnancy_consult"     : -5,
        "age_mismatch"          : -8,
        "skin_type_no"          : -15,
        "skin_type_caution"     : -5,
    },
}

VALID_SKIN_TYPES = ["oily", "dry", "combination", "normal", "sensitive", "mature"]
VALID_CONCERNS   = ["acne", "dryness", "aging", "hyperpigmentation",
                    "redness", "sensitivity", "pores", "texture", "dullness", "barrier"]
VALID_AGE_GROUPS = ["teen", "adult", "mature"]
VALID_PREGNANCY  = ["yes", "no"]


class IngredientProfileDB:
    def __init__(self, csv_path):
        self.df     = pd.read_csv(csv_path)
        self.lookup = {}
        self._build_lookup()

    def _parse_list(self, val):
        if not val or str(val).strip().lower() in ("none", "nan", ""):
            return []
        return [v.strip().lower() for v in str(val).split("|") if v.strip()]

    def _parse_suitability(self, val):
        v = str(val).strip().lower()
        if v in ("yes", "1", "true"):  return "yes"
        if v in ("no",  "0", "false"): return "no"
        if v == "moderate":            return "moderate"
        if v == "caution":             return "caution"
        return "yes"

    def _build_lookup(self):
        for _, row in self.df.iterrows():
            inci = str(row["inci_name"]).strip()
            profile = {
                "inci_name"        : inci,
                "category"         : str(row.get("ingredient_category", "")),
                "function"         : str(row.get("primary_function", "")),
                "skin_type"        : {
                    "oily"        : self._parse_suitability(row.get("suitable_oily",        "yes")),
                    "dry"         : self._parse_suitability(row.get("suitable_dry",         "yes")),
                    "combination" : self._parse_suitability(row.get("suitable_combination", "yes")),
                    "normal"      : self._parse_suitability(row.get("suitable_normal",      "yes")),
                    "sensitive"   : self._parse_suitability(row.get("suitable_sensitive",   "yes")),
                    "mature"      : self._parse_suitability(row.get("suitable_mature",      "yes")),
                },
                "concerns_helps"   : self._parse_list(row.get("skin_concerns_helps", "")),
                "concerns_worsens" : self._parse_list(row.get("skin_concerns_worsens", "")),
                "age_group"        : self._parse_list(row.get("age_group_suitable", "all")),
                "pregnancy_safe"   : str(row.get("pregnancy_safe", "yes")).strip().lower(),
                "irritancy"        : str(row.get("irritancy_potential", "low")).strip().lower(),
                "comedogenicity"   : self._safe_int(row.get("comedogenicity_0_to_5", 0)),
                "conc_min"         : self._safe_float(row.get("concentration_min_percent", 0)),
                "conc_max"         : self._safe_float(row.get("concentration_max_percent", 100)),
                "avoid_with"       : self._parse_list(row.get("avoid_combining_with", "")),
                "notes"            : str(row.get("usage_notes", "")),
            }
            self.lookup[inci.lower()] = profile

    def _safe_int(self, val):
        try:   return int(float(str(val)))
        except: return 0

    def _safe_float(self, val):
        try:   return float(str(val))
        except: return 0.0

    def get(self, inci_name):
        return self.lookup.get(inci_name.strip().lower())

    def all_profiles(self):
        return list(self.lookup.values())

    def all_names(self):
        return list(self.lookup.keys())


class RuleEngine:
    def __init__(self, cfg):
        self.cfg      = cfg
        self.weights  = cfg["weights"]
        self.penalties= cfg["penalties"]

    def score_ingredient(self, profile, skin_type, concerns, age_group, is_pregnant):
        breakdown = {
            "skin_type_score" : 0.0,
            "concern_score"   : 0.0,
            "safety_score"    : 0.0,
            "age_score"       : 0.0,
            "pregnancy_score" : 0.0,
            "bonus_score"     : 0.0,
            "penalties"       : 0.0,
            "penalty_reasons" : [],
            "pros"            : [],
            "cons"            : [],
        }

        st_val = profile["skin_type"].get(skin_type, "yes")
        if st_val == "yes":
            breakdown["skin_type_score"] = self.weights["skin_type"]
            breakdown["pros"].append(f"Well-suited for {skin_type} skin ({profile['function']})")
        elif st_val == "moderate":
            breakdown["skin_type_score"] = self.weights["skin_type"] * 0.65
        elif st_val == "caution":
            breakdown["skin_type_score"] = self.weights["skin_type"] * 0.35
            breakdown["cons"].append(f"Use with caution on {skin_type} skin")
            breakdown["penalties"] += self.penalties["skin_type_caution"]
            breakdown["penalty_reasons"].append(f"Caution flag for {skin_type} skin")
        elif st_val == "no":
            breakdown["skin_type_score"] = 0.0
            breakdown["cons"].append(f"Not recommended for {skin_type} skin — may worsen skin condition")
            breakdown["penalties"] += self.penalties["skin_type_no"]
            breakdown["penalty_reasons"].append(f"Explicitly unsuitable for {skin_type} skin")

        helps_count   = 0
        worsens_count = 0
        concern_bonus = 0.0

        worsens_clean = [re.split(r"\s*\(", w)[0].strip() for w in profile["concerns_worsens"]
                         if re.split(r"\s*\(", w)[0].strip() and re.split(r"\s*\(", w)[0].strip() != "none"]
        helps_clean   = [re.split(r"\s*\(", h)[0].strip() for h in profile["concerns_helps"]
                         if re.split(r"\s*\(", h)[0].strip() and re.split(r"\s*\(", h)[0].strip() != "none"]

        for concern in concerns:
            c = concern.lower()
            if any(c in h for h in helps_clean):
                helps_count += 1
                breakdown["pros"].append(f"Helps with {concern} ({profile['function']})")
            if any(c in w for w in worsens_clean):
                worsens_count += 1
                breakdown["cons"].append(f"May worsen {concern} — monitor carefully")
                breakdown["penalties"] += self.penalties["concern_worsened"]
                breakdown["penalty_reasons"].append(f"Ingredient may worsen {concern}")

        if concerns:
            breakdown["concern_score"] = self.weights["skin_concern"] * (helps_count / len(concerns))
            concern_bonus = min(5.0, helps_count * 1.5)
        else:
            breakdown["concern_score"] = self.weights["skin_concern"] * 0.5

        irritancy   = profile["irritancy"]
        comedogenic = profile["comedogenicity"]

        breakdown["safety_score"] = {
            "low"   : self.weights["safety"],
            "medium": self.weights["safety"] * 0.6,
            "high"  : self.weights["safety"] * 0.25,
        }.get(irritancy, self.weights["safety"])

        if skin_type == "sensitive":
            if irritancy == "high":
                breakdown["penalties"]      += self.penalties["high_irritancy"]
                breakdown["penalty_reasons"].append("High irritancy ingredient on sensitive skin")
                breakdown["cons"].append("High irritancy potential — not ideal for sensitive skin")
            elif irritancy == "medium":
                breakdown["penalties"]      += self.penalties["medium_irritancy_sens"]
                breakdown["penalty_reasons"].append("Medium irritancy on sensitive skin")
                breakdown["cons"].append("Moderate irritancy — patch test recommended")

        acne_relevant = skin_type in ("oily", "combination") or "acne" in concerns
        if comedogenic >= 4:
            if acne_relevant:
                breakdown["penalties"]      += self.penalties["comedogenic_4_5"]
                breakdown["penalty_reasons"].append(f"High comedogenicity ({comedogenic}/5) on acne/oily skin")
                breakdown["cons"].append(f"Comedogenicity rating {comedogenic}/5 — high risk of clogged pores")
            else:
                breakdown["penalties"] += self.penalties["comedogenic_4_5"] * 0.4
                breakdown["cons"].append(f"Comedogenicity rating {comedogenic}/5 — use sparingly")
        elif comedogenic == 3:
            if acne_relevant:
                breakdown["penalties"]      += self.penalties["comedogenic_3"]
                breakdown["penalty_reasons"].append(f"Moderate comedogenicity ({comedogenic}/5)")
                breakdown["cons"].append(f"Comedogenicity rating {comedogenic}/5 — moderate pore-clogging risk")
        elif comedogenic == 2 and skin_type == "oily":
            breakdown["penalties"]      += self.penalties["comedogenic_2_oily"]
            breakdown["penalty_reasons"].append(f"Mild comedogenicity ({comedogenic}/5) on oily skin")

        if comedogenic == 0:
            breakdown["pros"].append("Non-comedogenic (0/5) — safe for pore-prone skin")
        elif comedogenic == 1:
            breakdown["pros"].append("Very low comedogenicity (1/5)")

        age_groups = profile["age_group"]
        if "all" in age_groups or age_group in age_groups:
            breakdown["age_score"] = self.weights["age"]
            if age_group == "teen" and "teen" in age_groups:
                breakdown["pros"].append("Safe and appropriate for teen skin")
            elif age_group == "mature" and "mature" in age_groups:
                breakdown["pros"].append("Formulated for mature skin needs")
        else:
            breakdown["age_score"] = 0.0
            breakdown["penalties"]      += self.penalties["age_mismatch"]
            breakdown["penalty_reasons"].append(
                f"Not recommended for {age_group} age group (suitable for: {', '.join(age_groups)})"
            )
            breakdown["cons"].append(f"Not typically recommended for {age_group} users")

        preg_safe = profile["pregnancy_safe"]
        if is_pregnant:
            if preg_safe == "yes":
                breakdown["pregnancy_score"] = self.weights["pregnancy"]
                breakdown["pros"].append("Pregnancy safe")
            elif preg_safe == "consult":
                breakdown["pregnancy_score"] = self.weights["pregnancy"] * 0.4
                breakdown["penalties"]      += self.penalties["pregnancy_consult"]
                breakdown["penalty_reasons"].append("Ingredient requires pregnancy consultation")
                breakdown["cons"].append("Consult your doctor before using during pregnancy")
            elif preg_safe == "no":
                breakdown["pregnancy_score"] = 0.0
                breakdown["penalties"]      += self.penalties["pregnancy_unsafe"]
                breakdown["penalty_reasons"].append("Ingredient is NOT safe during pregnancy")
                breakdown["cons"].append("NOT recommended during pregnancy — avoid completely")
        else:
            if preg_safe == "yes":
                breakdown["pregnancy_score"] = self.weights["pregnancy"]
            elif preg_safe == "consult":
                breakdown["pregnancy_score"] = self.weights["pregnancy"] * 0.7
            else:
                breakdown["pregnancy_score"] = self.weights["pregnancy"] * 0.5

        breakdown["bonus_score"] = min(self.weights["bonus"], concern_bonus)

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

    def score_product(self, ingredients, skin_type, concerns, age_group, is_pregnant):
        if not ingredients:
            return {"rule_score": 0.0, "details": [], "all_pros": [], "all_cons": []}

        scores        = []
        all_pros      = []
        all_cons      = []
        all_penalties = []
        details       = []

        for profile in ingredients:
            bd = self.score_ingredient(profile, skin_type, concerns, age_group, is_pregnant)
            scores.append(bd["raw_score"])
            all_pros.extend(bd["pros"])
            all_cons.extend(bd["cons"])
            all_penalties.extend(bd["penalty_reasons"])
            details.append({
                "ingredient" : profile["inci_name"],
                "score"      : round(bd["raw_score"], 2),
                "pros"       : bd["pros"],
                "cons"       : bd["cons"],
                "penalties"  : bd["penalty_reasons"],
            })

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

        rule_score = np.average(scores, weights=weights) if sum(weights) > 0 else np.mean(scores)

        bad_count = sum(1 for s in scores if s < 40)
        if bad_count > 0:
            stack_penalty = sum(6.0 / (i ** 0.5) for i in range(1, bad_count + 1))
            rule_score = max(0.0, rule_score - stack_penalty)

        return {
            "rule_score"   : float(np.clip(rule_score, 0, 100)),
            "details"      : details,
            "all_pros"     : list(dict.fromkeys(all_pros)),
            "all_cons"     : list(dict.fromkeys(all_cons)),
            "all_penalties": list(dict.fromkeys(all_penalties)),
        }


def encode_skin_type(skin_type):
    return [int(skin_type == st) for st in VALID_SKIN_TYPES]

def encode_age(age_group):
    return [int(age_group == ag) for ag in VALID_AGE_GROUPS]

def encode_concerns(concerns):
    return [int(c in concerns) for c in VALID_CONCERNS]

def extract_product_features(ingredients, skin_type, concerns, age_group, is_pregnant, rule_score):
    n = len(ingredients) if ingredients else 1

    comedogenicity_vals = [p.get("comedogenicity", 0) for p in ingredients]
    irritancy_map       = {"low": 0, "medium": 1, "high": 2}
    irritancy_vals      = [irritancy_map.get(p.get("irritancy", "low"), 0) for p in ingredients]

    st_yes  = sum(1 for p in ingredients if p["skin_type"].get(skin_type) == "yes")
    st_mod  = sum(1 for p in ingredients if p["skin_type"].get(skin_type) == "moderate")
    st_caut = sum(1 for p in ingredients if p["skin_type"].get(skin_type) == "caution")
    st_no   = sum(1 for p in ingredients if p["skin_type"].get(skin_type) == "no")

    helps_count   = 0
    worsens_count = 0
    for p in ingredients:
        helps_raw   = [re.split(r"\s*\(", h)[0].strip() for h in p.get("concerns_helps", [])]
        worsens_raw = [re.split(r"\s*\(", w)[0].strip() for w in p.get("concerns_worsens", [])]
        for c in concerns:
            if any(c.lower() in h for h in helps_raw):   helps_count   += 1
            if any(c.lower() in w for w in worsens_raw): worsens_count += 1

    preg_yes     = sum(1 for p in ingredients if p.get("pregnancy_safe") == "yes")
    preg_consult = sum(1 for p in ingredients if p.get("pregnancy_safe") == "consult")
    preg_no      = sum(1 for p in ingredients if p.get("pregnancy_safe") == "no")

    age_match = sum(
        1 for p in ingredients
        if "all" in p.get("age_group", []) or age_group in p.get("age_group", [])
    )

    categories      = [p.get("category", "").lower() for p in ingredients]
    n_actives       = sum(1 for c in categories if "active" in c)
    n_humectants    = sum(1 for c in categories if "humectant" in c)
    n_emollients    = sum(1 for c in categories if "emollient" in c)
    n_preservatives = sum(1 for c in categories if "preservative" in c)
    n_sunscreen     = sum(1 for c in categories if "uv filter" in c or "sunscreen" in c)

    feats = [
        rule_score / 100.0,
        n,
        n / max(n, 1),
        st_yes   / n,
        st_mod   / n,
        st_caut  / n,
        st_no    / n,
        helps_count   / max(n * max(len(concerns), 1), 1),
        worsens_count / max(n * max(len(concerns), 1), 1),
        len(concerns),
        helps_count,
        worsens_count,
        np.mean(comedogenicity_vals)  if comedogenicity_vals else 0,
        np.max(comedogenicity_vals)   if comedogenicity_vals else 0,
        np.mean(irritancy_vals)       if irritancy_vals      else 0,
        np.max(irritancy_vals)        if irritancy_vals      else 0,
        sum(1 for v in irritancy_vals if v == 2) / n,
        sum(1 for v in comedogenicity_vals if v >= 4) / n,
        int(is_pregnant),
        preg_yes     / n,
        preg_consult / n,
        preg_no      / n,
        age_match / n,
        n_actives       / n,
        n_humectants    / n,
        n_emollients    / n,
        n_preservatives / n,
        n_sunscreen     / n,
        *encode_skin_type(skin_type),
        *encode_age(age_group),
        *encode_concerns(concerns),
    ]

    return np.array(feats, dtype=np.float32)


def generate_synthetic_data(db, rule_engine, n_samples, seed=42):
    rng    = np.random.default_rng(seed)
    names  = db.all_names()
    X_list = []
    y_list = []

    for _ in tqdm(range(n_samples), desc="Synthetic data"):
        skin_type   = rng.choice(VALID_SKIN_TYPES)
        age_group   = rng.choice(VALID_AGE_GROUPS)
        is_pregnant = bool(rng.choice([True, False], p=[0.15, 0.85]))
        n_concerns  = int(rng.integers(0, 5))
        concerns    = list(rng.choice(VALID_CONCERNS, size=n_concerns, replace=False))

        n_ings    = int(rng.integers(2, 21))
        ing_names = rng.choice(names, size=min(n_ings, len(names)), replace=False)
        profiles  = [p for n in ing_names if (p := db.get(n)) is not None]

        if not profiles:
            continue

        result     = rule_engine.score_product(profiles, skin_type, concerns, age_group, is_pregnant)
        rule_score = result["rule_score"]
        noise      = rng.normal(0, 3.0)
        label      = float(np.clip(rule_score + noise, 0, 100))
        feat       = extract_product_features(profiles, skin_type, concerns, age_group, is_pregnant, rule_score)

        X_list.append(feat)
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_ml_model(X, y, cfg):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=cfg["random_seed"]
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    model = xgb.XGBRegressor(**cfg["xgb_params"])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    y_pred = np.clip(model.predict(X_val), 0, 100)
    metrics = {
        "mae"    : round(float(mean_absolute_error(y_val, y_pred)), 4),
        "rmse"   : round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 4),
        "r2"     : round(float(r2_score(y_val, y_pred)), 4),
        "n_train": len(X_train),
        "n_val"  : len(X_val),
    }

    return model, scaler, metrics


class CompatibilityScorer:
    def __init__(self, model, scaler, db, rule_engine, cfg):
        self.model       = model
        self.scaler      = scaler
        self.db          = db
        self.rule_engine = rule_engine
        self.cfg         = cfg

    def score(self, ingredient_names, skin_type, concerns, age_group, is_pregnant):
        t0        = time.perf_counter()
        skin_type = skin_type.lower().strip()
        age_group = age_group.lower().strip()
        concerns  = [c.lower().strip() for c in concerns]

        self._validate(skin_type, concerns, age_group, is_pregnant)

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

        rule_result = self.rule_engine.score_product(profiles, skin_type, concerns, age_group, is_pregnant)
        rule_score  = rule_result["rule_score"]

        feat     = extract_product_features(profiles, skin_type, concerns, age_group, is_pregnant, rule_score)
        feat_sc  = self.scaler.transform(feat.reshape(1, -1))
        ml_score = float(np.clip(self.model.predict(feat_sc)[0], 0, 100))

        final = round(float(np.clip(
            self.cfg["rule_weight"] * rule_score + self.cfg["ml_weight"] * ml_score, 0, 100
        )), 1)

        return {
            "compatibility_score" : final,
            "grade"               : self._grade(final),
            "verdict"             : self._verdict(final, skin_type, concerns),
            "pros"                : rule_result["all_pros"][:8],
            "cons"                : rule_result["all_cons"][:8],
            "ingredient_details"  : rule_result["details"],
            "warnings"            : self._build_warnings(profiles, skin_type, concerns, age_group, is_pregnant, rule_result["all_penalties"]),
            "not_found"           : not_found,
            "rule_score"          : round(rule_score, 1),
            "ml_score"            : round(ml_score,   1),
            "latency_ms"          : round((time.perf_counter() - t0) * 1000, 2),
        }

    def _validate(self, skin_type, concerns, age_group, is_pregnant):
        if skin_type not in VALID_SKIN_TYPES:
            raise ValueError(f"skin_type must be one of {VALID_SKIN_TYPES}")
        if age_group not in VALID_AGE_GROUPS:
            raise ValueError(f"age_group must be one of {VALID_AGE_GROUPS}")
        for c in concerns:
            if c not in VALID_CONCERNS:
                raise ValueError(f"concern '{c}' must be one of {VALID_CONCERNS}")

    def _grade(self, score):
        if score >= 92: return "A+"
        if score >= 85: return "A"
        if score >= 78: return "B+"
        if score >= 70: return "B"
        if score >= 62: return "C+"
        if score >= 55: return "C"
        if score >= 40: return "D"
        return "F"

    def _verdict(self, score, skin_type, concerns):
        concern_str = ", ".join(concerns) if concerns else "general skin health"
        if score >= 85:
            return f"Excellent compatibility ({score}/100). This product is highly recommended for your {skin_type} skin and {concern_str} concerns."
        if score >= 70:
            return f"Good compatibility ({score}/100). This product works well for your {skin_type} skin with minor cautions noted."
        if score >= 55:
            return f"Moderate compatibility ({score}/100). This product is acceptable but has some concerns for your {skin_type} skin — see details."
        if score >= 40:
            return f"Poor compatibility ({score}/100). This product has significant incompatibilities with your {skin_type} skin profile."
        return f"Very poor compatibility ({score}/100). This product is not recommended for your skin type and concerns."

    def _build_warnings(self, profiles, skin_type, concerns, age_group, is_pregnant, penalties):
        warnings_list = []

        if is_pregnant:
            unsafe  = [p["inci_name"] for p in profiles if p.get("pregnancy_safe") == "no"]
            consult = [p["inci_name"] for p in profiles if p.get("pregnancy_safe") == "consult"]
            if unsafe:
                warnings_list.append(f"PREGNANCY WARNING: The following ingredients should be AVOIDED during pregnancy: {', '.join(unsafe)}")
            if consult:
                warnings_list.append(f"PREGNANCY CAUTION: Consult your OB-GYN before using products containing: {', '.join(consult)}")

        if skin_type == "sensitive":
            irritants = [p["inci_name"] for p in profiles if p.get("irritancy") == "high"]
            if irritants:
                warnings_list.append(f"SENSITIVITY WARNING: High-irritancy ingredients detected for sensitive skin: {', '.join(irritants)}")

        if skin_type in ("oily", "combination") or "acne" in concerns:
            high_comed = [p["inci_name"] for p in profiles if p.get("comedogenicity", 0) >= 4]
            if high_comed:
                warnings_list.append(f"ACNE WARNING: High-comedogenicity ingredients (4-5/5) found: {', '.join(high_comed)} — may clog pores")

        if age_group == "teen":
            strong = [p["inci_name"] for p in profiles
                      if p.get("irritancy") == "high" and "teen" not in p.get("age_group", [])]
            if strong:
                warnings_list.append(f"AGE WARNING: Strong actives not typically recommended for teen skin: {', '.join(strong)}")

        return warnings_list

    @classmethod
    def load(cls, model_dir, data_path=None, cfg=CFG):
        model_dir  = Path(model_dir)
        model      = joblib.load(model_dir / "xgb_model.pkl")
        scaler     = joblib.load(model_dir / "scaler.pkl")
        db         = IngredientProfileDB(data_path or cfg["data_path"])
        rule_engine= RuleEngine(cfg)
        return cls(model, scaler, db, rule_engine, cfg)


def main():
    parser = argparse.ArgumentParser(description="SkinSpectra Calculation Layer Trainer")
    parser.add_argument("--data",    default=CFG["data_path"])
    parser.add_argument("--output",  default=CFG["output_dir"])
    parser.add_argument("--samples", type=int, default=CFG["n_synthetic"])
    parser.add_argument("--seed",    type=int, default=CFG["random_seed"])
    args = parser.parse_args()

    CFG["data_path"]   = args.data
    CFG["output_dir"]  = args.output
    CFG["model_path"]  = f"{args.output}/xgb_model.pkl"
    CFG["scaler_path"] = f"{args.output}/scaler.pkl"
    CFG["n_synthetic"] = args.samples
    CFG["random_seed"] = args.seed

    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    db          = IngredientProfileDB(CFG["data_path"])
    rule_engine = RuleEngine(CFG)
    X, y        = generate_synthetic_data(db, rule_engine, CFG["n_synthetic"], CFG["random_seed"])
    model, scaler, metrics = train_ml_model(X, y, CFG)

    joblib.dump(model,  output_dir / "xgb_model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

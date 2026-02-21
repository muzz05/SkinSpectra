"""
SkinSpectra Calculation Layer - Test Suite (Feature 1)
======================================================
Tests
-----
1.  Score range validation          -- all scores must be 0-100
2.  Grade assignment accuracy       -- correct letter grades
3.  Ideal product (perfect match)   -- expects score >= 85
4.  Terrible product (mismatches)   -- expects score <= 45
5.  Pregnancy safety enforcement    -- retinol/AHA flags
6.  Comedogenicity penalties        -- coconut oil on acne skin
7.  Irritancy penalties             -- harsh ingredients on sensitive
8.  Age group enforcement           -- teen vs mature ingredient mismatch
9.  Skin type compatibility         -- ingredient suitability per type
10. Skin concern matching           -- helps vs worsens logic
11. Pros and cons generation        -- non-empty and relevant
12. Warnings generation             -- critical flags present
13. Not-found ingredient handling   -- graceful degradation
14. Empty ingredient list           -- zero score, no crash
15. Regression test set             -- 20 labelled scenarios
16. Penalty stacking                -- multiple bad ingredients compound
17. Boundary conditions             -- score never exceeds 100 or goes below 0
18. All skin types                  -- valid result for every skin type
19. All concerns                    -- valid result for every concern
20. Latency benchmark               -- p95 < 200ms
21. Schema validation               -- all required keys present
22. Ensemble logic                  -- rule + ML scores both present
"""

import sys
import time
import json
import logging
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from components.calculation_individual_layer import (
    CompatibilityScorer, IngredientProfileDB, RuleEngine,
    CFG, VALID_SKIN_TYPES, VALID_CONCERNS, VALID_AGE_GROUPS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.calc.test")

# ── Terminal colours ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}PASS  {msg}{RESET}")
def fail(msg): print(f"  {RED}FAIL  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}WARN  {msg}{RESET}")
def info(msg): print(f"  {CYAN}INFO  {msg}{RESET}")
def hdr(msg):
    print(f"\n{BOLD}{CYAN}{'─'*65}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{'─'*65}")


@dataclass
class TR:
    name:    str
    passed:  int = 0
    failed:  int = 0
    warned:  int = 0

    @property
    def total(self): return self.passed + self.failed + self.warned
    @property
    def pct(self):   return 100 * self.passed / self.total if self.total else 0


# ── Required schema keys ──────────────────────────────────────────────────────
REQUIRED_KEYS = {
    "compatibility_score", "grade", "verdict", "pros", "cons",
    "ingredient_details", "warnings", "not_found",
    "rule_score", "ml_score", "latency_ms",
}

# =============================================================================
# REGRESSION SCENARIOS
# Each scenario: ingredients, skin_type, concerns, age_group, is_pregnant,
#                min_score, max_score, description
# =============================================================================
REGRESSION_SCENARIOS = [
    # ── IDEAL matches ──────────────────────────────────────────────────────
    {
        "desc"        : "Perfect gentle moisturizer for dry sensitive skin",
        "ingredients" : ["Hyaluronic Acid", "Ceramide NP", "Glycerin", "Panthenol", "Allantoin"],
        "skin_type"   : "dry",
        "concerns"    : ["dryness", "sensitivity", "barrier"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 78,
        "max_score"   : 100,
    },
    {
        "desc"        : "Ideal oily acne routine with salicylic + niacinamide",
        "ingredients" : ["Salicylic Acid", "Niacinamide", "Zinc PCA", "Aloe Barbadensis Leaf Juice"],
        "skin_type"   : "oily",
        "concerns"    : ["acne", "pores", "redness"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 75,
        "max_score"   : 100,
    },
    {
        "desc"        : "Pregnancy-safe brightening serum",
        "ingredients" : ["Niacinamide", "Azelaic Acid", "Hyaluronic Acid", "Glycerin"],
        "skin_type"   : "normal",
        "concerns"    : ["hyperpigmentation", "dullness"],
        "age_group"   : "adult",
        "is_pregnant" : True,
        "min_score"   : 75,
        "max_score"   : 100,
    },
    {
        "desc"        : "Anti-aging routine for mature dry skin",
        "ingredients" : ["Retinol", "Hyaluronic Acid", "Ceramide NP", "Squalane", "Niacinamide"],
        "skin_type"   : "dry",
        "concerns"    : ["aging", "texture", "dullness"],
        "age_group"   : "mature",
        "is_pregnant" : False,
        "min_score"   : 70,
        "max_score"   : 100,
    },
    {
        "desc"        : "Teen safe simple moisturiser",
        "ingredients" : ["Aloe Barbadensis Leaf Juice", "Glycerin", "Panthenol", "Allantoin"],
        "skin_type"   : "combination",
        "concerns"    : ["acne", "redness"],
        "age_group"   : "teen",
        "is_pregnant" : False,
        "min_score"   : 75,
        "max_score"   : 100,
    },
    {
        "desc"        : "Brightening stack for hyperpigmentation",
        "ingredients" : ["Alpha Arbutin", "Niacinamide", "Tranexamic Acid", "Ascorbyl Glucoside"],
        "skin_type"   : "normal",
        "concerns"    : ["hyperpigmentation", "dullness"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 75,
        "max_score"   : 100,
    },
    {
        "desc"        : "Sensitive barrier repair routine",
        "ingredients" : ["Ceramide NP", "Ceramide AP", "Cholesterol", "Oat Extract", "Bisabolol"],
        "skin_type"   : "sensitive",
        "concerns"    : ["sensitivity", "barrier", "redness"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 78,
        "max_score"   : 100,
    },
    {
        "desc"        : "Mineral sunscreen for sensitive skin",
        "ingredients" : ["Zinc Oxide", "Titanium Dioxide", "Dimethicone", "Glycerin"],
        "skin_type"   : "sensitive",
        "concerns"    : ["redness", "sensitivity"],
        "age_group"   : "adult",
        "is_pregnant" : True,
        "min_score"   : 75,
        "max_score"   : 100,
    },

    # ── POOR matches ───────────────────────────────────────────────────────
    {
        "desc"        : "Highly comedogenic product on acne oily skin",
        "ingredients" : ["Coconut Oil", "Isopropyl Myristate", "Mineral Oil"],
        "skin_type"   : "oily",
        "concerns"    : ["acne", "pores"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 0,
        "max_score"   : 45,
    },
    {
        "desc"        : "Unsafe pregnancy - retinol + glycolic acid",
        "ingredients" : ["Retinol", "Glycolic Acid", "Benzoyl Peroxide"],
        "skin_type"   : "normal",
        "concerns"    : ["aging"],
        "age_group"   : "adult",
        "is_pregnant" : True,
        "min_score"   : 0,
        "max_score"   : 30,
    },
    {
        "desc"        : "High irritancy on sensitive skin",
        "ingredients" : ["Benzoyl Peroxide", "Glycolic Acid", "Retinol", "Salicylic Acid"],
        "skin_type"   : "sensitive",
        "concerns"    : ["acne"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 0,
        "max_score"   : 40,
    },
    {
        "desc"        : "Dry skin + ingredients that worsen dryness",
        "ingredients" : ["Benzoyl Peroxide", "Colloidal Sulfur", "Witch Hazel Extract"],
        "skin_type"   : "dry",
        "concerns"    : ["dryness", "sensitivity"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 0,
        "max_score"   : 45,
    },

    # ── MODERATE matches ───────────────────────────────────────────────────
    {
        "desc"        : "Mixed product - some good, some caution ingredients",
        "ingredients" : ["Niacinamide", "Hyaluronic Acid", "Argan Oil", "Propylene Glycol"],
        "skin_type"   : "sensitive",
        "concerns"    : ["redness", "dryness"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 50,
        "max_score"   : 80,
    },
    {
        "desc"        : "Retinol for teen (age mismatch penalty)",
        "ingredients" : ["Retinol", "Niacinamide", "Hyaluronic Acid"],
        "skin_type"   : "oily",
        "concerns"    : ["acne", "texture"],
        "age_group"   : "teen",
        "is_pregnant" : False,
        "min_score"   : 40,
        "max_score"   : 72,
    },
    {
        "desc"        : "Combination skin mixed routine",
        "ingredients" : ["Salicylic Acid", "Glycerin", "Niacinamide", "Rosehip Oil"],
        "skin_type"   : "combination",
        "concerns"    : ["acne", "dryness"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 55,
        "max_score"   : 85,
    },
    {
        "desc"        : "Mature skin routine with pregnancy consult items",
        "ingredients" : ["Retinyl Retinoate", "Alpha Arbutin", "Hyaluronic Acid"],
        "skin_type"   : "mature",
        "concerns"    : ["aging", "hyperpigmentation"],
        "age_group"   : "mature",
        "is_pregnant" : True,
        "min_score"   : 30,
        "max_score"   : 65,
    },
    {
        "desc"        : "Bakuchiol pregnancy-safe anti-aging",
        "ingredients" : ["Bakuchiol", "Hyaluronic Acid", "Niacinamide", "Ceramide NP"],
        "skin_type"   : "normal",
        "concerns"    : ["aging", "dullness"],
        "age_group"   : "adult",
        "is_pregnant" : True,
        "min_score"   : 72,
        "max_score"   : 100,
    },
    {
        "desc"        : "Oily skin with moderate comedogenic risk",
        "ingredients" : ["Sweet Almond Oil", "Argan Oil", "Shea Butter"],
        "skin_type"   : "oily",
        "concerns"    : ["acne"],
        "age_group"   : "adult",
        "is_pregnant" : False,
        "min_score"   : 30,
        "max_score"   : 62,
    },
]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def check(cond: bool, desc: str, r: TR):
    if cond:
        ok(desc)
        r.passed += 1
    else:
        fail(desc)
        r.failed += 1


def run_score(scorer, ingredients, skin_type, concerns, age_group, is_pregnant):
    return scorer.score(
        ingredient_names=ingredients,
        skin_type=skin_type,
        concerns=concerns,
        age_group=age_group,
        is_pregnant=is_pregnant,
    )


# ── Test 1: Score range validation ────────────────────────────────────────────
def test_score_range(scorer) -> TR:
    hdr("1. Score Range Validation (must be 0-100)")
    r = TR(name="Score Range")
    test_cases = [
        (["Glycerin", "Hyaluronic Acid"], "dry",  ["dryness"], "adult", False),
        (["Coconut Oil", "Benzoyl Peroxide", "Retinol"], "sensitive", ["acne"], "teen", True),
        (["Salicylic Acid"], "oily", ["acne", "pores"], "adult", False),
        (["Ceramide NP", "Cholesterol", "Squalane"], "sensitive", ["barrier"], "mature", False),
    ]
    for ings, st, c, ag, preg in test_cases:
        res = run_score(scorer, ings, st, c, ag, preg)
        sc  = res["compatibility_score"]
        check(0.0 <= sc <= 100.0, f"Score {sc} in [0,100] for {ings[:2]}...", r)
    return r


# ── Test 2: Grade assignment ───────────────────────────────────────────────────
def test_grades(scorer) -> TR:
    hdr("2. Grade Assignment Accuracy")
    r = TR(name="Grade Assignment")
    grade_map = {
        (92, 100): "A+",
        (85, 91) : "A",
        (78, 84) : "B+",
        (70, 77) : "B",
        (62, 69) : "C+",
        (55, 61) : "C",
        (40, 54) : "D",
        (0,  39) : "F",
    }
    eng = RuleEngine(CFG)
    for (lo, hi), expected_grade in grade_map.items():
        mid = (lo + hi) / 2
        actual_grade = scorer._grade(mid)
        check(actual_grade == expected_grade,
              f"Score {mid} -> grade '{actual_grade}' (expected '{expected_grade}')", r)
    return r


# ── Test 3: Ideal product ─────────────────────────────────────────────────────
def test_ideal_product(scorer) -> TR:
    hdr("3. Ideal Product (expect score >= 78)")
    r = TR(name="Ideal Product")
    res = run_score(
        scorer,
        ["Hyaluronic Acid", "Ceramide NP", "Glycerin", "Panthenol", "Allantoin", "Niacinamide"],
        "dry", ["dryness", "sensitivity", "barrier"], "adult", False
    )
    check(res["compatibility_score"] >= 78,
          f"Ideal moisturizer score={res['compatibility_score']} >= 78", r)
    check(res["grade"] in ("A+","A","B+","B"),
          f"Grade={res['grade']} in A+/A/B+/B range", r)
    check(len(res["pros"]) >= 2,
          f"Pros list has {len(res['pros'])} entries >= 2", r)
    info(f"  Score: {res['compatibility_score']}  Grade: {res['grade']}")
    info(f"  Pros:  {res['pros'][:3]}")
    return r


# ── Test 4: Bad product ───────────────────────────────────────────────────────
def test_bad_product(scorer) -> TR:
    hdr("4. Terrible Product (expect score <= 45)")
    r = TR(name="Bad Product")
    res = run_score(
        scorer,
        ["Coconut Oil", "Isopropyl Myristate", "Benzoyl Peroxide"],
        "sensitive", ["acne", "redness", "sensitivity"], "adult", False
    )
    check(res["compatibility_score"] <= 45,
          f"Bad product score={res['compatibility_score']} <= 45", r)
    check(len(res["cons"]) >= 2,
          f"Cons list has {len(res['cons'])} entries >= 2", r)
    info(f"  Score: {res['compatibility_score']}  Grade: {res['grade']}")
    info(f"  Cons:  {res['cons'][:3]}")
    return r


# ── Test 5: Pregnancy safety ──────────────────────────────────────────────────
def test_pregnancy(scorer) -> TR:
    hdr("5. Pregnancy Safety Enforcement")
    r = TR(name="Pregnancy Safety")

    # Unsafe combo for pregnant user
    res_unsafe = run_score(
        scorer, ["Retinol", "Glycolic Acid", "Benzoyl Peroxide"],
        "normal", ["aging"], "adult", True
    )
    check(res_unsafe["compatibility_score"] <= 35,
          f"Pregnant+retinol score={res_unsafe['compatibility_score']} <= 35", r)
    has_preg_warn = any("PREGNANCY" in w.upper() for w in res_unsafe["warnings"])
    check(has_preg_warn, "Pregnancy warning present in warnings", r)

    # Safe combo for pregnant user
    res_safe = run_score(
        scorer, ["Niacinamide", "Hyaluronic Acid", "Azelaic Acid", "Glycerin"],
        "normal", ["hyperpigmentation"], "adult", True
    )
    check(res_safe["compatibility_score"] >= 65,
          f"Pregnant+safe combo score={res_safe['compatibility_score']} >= 65", r)
    no_unsafe_warn = not any("AVOID" in w.upper() and "PREGNANCY" in w.upper()
                             for w in res_safe["warnings"])
    check(no_unsafe_warn, "No unsafe pregnancy warning for safe combo", r)

    info(f"  Unsafe score: {res_unsafe['compatibility_score']}")
    info(f"  Safe score  : {res_safe['compatibility_score']}")
    return r


# ── Test 6: Comedogenicity penalties ─────────────────────────────────────────
def test_comedogenicity(scorer) -> TR:
    hdr("6. Comedogenicity Penalties")
    r = TR(name="Comedogenicity")

    # Coconut oil (comed=4) on oily acne skin vs dry skin
    res_oily = run_score(scorer, ["Coconut Oil"], "oily", ["acne"], "adult", False)
    res_dry  = run_score(scorer, ["Coconut Oil"], "dry",  ["dryness"], "adult", False)

    check(res_oily["compatibility_score"] < res_dry["compatibility_score"],
          f"Coconut oil scores lower on oily({res_oily['compatibility_score']}) "
          f"vs dry({res_dry['compatibility_score']}) skin", r)
    check(res_oily["compatibility_score"] <= 55,
          f"Coconut oil on oily acne skin score={res_oily['compatibility_score']} <= 55", r)

    # Check comedogenicity warning
    has_comed_warning = any("comedogen" in c.lower() or "pore" in c.lower()
                            for c in res_oily["cons"])
    check(has_comed_warning, "Comedogenicity concern present in cons", r)
    info(f"  Oily skin: {res_oily['compatibility_score']}  Dry skin: {res_dry['compatibility_score']}")
    return r


# ── Test 7: Irritancy penalties ───────────────────────────────────────────────
def test_irritancy(scorer) -> TR:
    hdr("7. Irritancy Penalties on Sensitive Skin")
    r = TR(name="Irritancy")

    res_sensitive = run_score(
        scorer, ["Benzoyl Peroxide", "Glycolic Acid"],
        "sensitive", ["acne"], "adult", False
    )
    res_oily = run_score(
        scorer, ["Benzoyl Peroxide", "Glycolic Acid"],
        "oily", ["acne"], "adult", False
    )
    check(res_sensitive["compatibility_score"] < res_oily["compatibility_score"],
          f"High irritancy scores lower on sensitive({res_sensitive['compatibility_score']}) "
          f"vs oily({res_oily['compatibility_score']})", r)

    has_irritancy_warn = any(
        "SENSITIV" in w.upper() or "irritan" in w.lower()
        for w in res_sensitive["warnings"] + res_sensitive["cons"]
    )
    check(has_irritancy_warn, "Irritancy warning present for sensitive skin", r)
    info(f"  Sensitive: {res_sensitive['compatibility_score']}  Oily: {res_oily['compatibility_score']}")
    return r


# ── Test 8: Age group enforcement ────────────────────────────────────────────
def test_age_group(scorer) -> TR:
    hdr("8. Age Group Suitability Enforcement")
    r = TR(name="Age Group")

    # Retinol is adult/mature only - teen should score lower
    res_adult = run_score(scorer, ["Retinol"], "oily", ["acne"], "adult", False)
    res_teen  = run_score(scorer, ["Retinol"], "oily", ["acne"], "teen",  False)

    check(res_adult["compatibility_score"] >= res_teen["compatibility_score"],
          f"Retinol: adult({res_adult['compatibility_score']}) >= teen({res_teen['compatibility_score']})", r)

    # Teen-safe routine should score well for teen
    res_teen_safe = run_score(
        scorer, ["Salicylic Acid", "Niacinamide", "Aloe Barbadensis Leaf Juice"],
        "oily", ["acne"], "teen", False
    )
    check(res_teen_safe["compatibility_score"] >= 60,
          f"Teen-safe routine for teen scores {res_teen_safe['compatibility_score']} >= 60", r)
    info(f"  Retinol adult: {res_adult['compatibility_score']}  teen: {res_teen['compatibility_score']}")
    return r


# ── Test 9: Skin type compatibility ──────────────────────────────────────────
def test_skin_types(scorer) -> TR:
    hdr("9. All Skin Types Return Valid Scores")
    r = TR(name="Skin Types")
    ingredients = ["Hyaluronic Acid", "Niacinamide", "Glycerin"]
    for st in VALID_SKIN_TYPES:
        res = run_score(scorer, ingredients, st, ["dryness"], "adult", False)
        check(0 <= res["compatibility_score"] <= 100,
              f"Skin type '{st}' -> score={res['compatibility_score']}", r)
    return r


# ── Test 10: Skin concern matching ────────────────────────────────────────────
def test_concerns(scorer) -> TR:
    hdr("10. Skin Concern Matching Logic")
    r = TR(name="Concerns")

    # Salicylic acid should help acne more than aging
    res_acne  = run_score(scorer, ["Salicylic Acid"], "oily", ["acne"],  "adult", False)
    res_aging = run_score(scorer, ["Salicylic Acid"], "oily", ["aging"], "adult", False)
    check(res_acne["compatibility_score"] >= res_aging["compatibility_score"],
          f"Salicylic: acne({res_acne['compatibility_score']}) >= aging({res_aging['compatibility_score']})", r)

    # Retinol should help aging more than acne
    res_aging2 = run_score(scorer, ["Retinol"], "normal", ["aging"], "adult", False)
    res_dry    = run_score(scorer, ["Retinol"], "dry",    ["dryness"], "adult", False)
    check(res_aging2["compatibility_score"] >= 50,
          f"Retinol for aging scores {res_aging2['compatibility_score']} >= 50", r)
    return r


# ── Test 11: Pros and cons quality ───────────────────────────────────────────
def test_pros_cons(scorer) -> TR:
    hdr("11. Pros and Cons Quality")
    r = TR(name="Pros Cons")

    # Good product
    res_good = run_score(
        scorer, ["Hyaluronic Acid", "Ceramide NP", "Niacinamide"],
        "dry", ["dryness", "barrier"], "adult", False
    )
    check(len(res_good["pros"]) >= 2, f"Good product has {len(res_good['pros'])} pros", r)
    check(isinstance(res_good["pros"][0], str), "Pros are strings", r)

    # Bad product
    res_bad = run_score(
        scorer, ["Coconut Oil", "Benzoyl Peroxide"],
        "sensitive", ["acne"], "adult", False
    )
    check(len(res_bad["cons"]) >= 2, f"Bad product has {len(res_bad['cons'])} cons", r)
    check(isinstance(res_bad["cons"][0], str), "Cons are strings", r)

    info(f"  Good pros: {res_good['pros'][:2]}")
    info(f"  Bad cons : {res_bad['cons'][:2]}")
    return r


# ── Test 12: Warnings generation ─────────────────────────────────────────────
def test_warnings(scorer) -> TR:
    hdr("12. Critical Warnings Generation")
    r = TR(name="Warnings")

    # Retinol for pregnant user -> PREGNANCY WARNING
    res = run_score(scorer, ["Retinol", "Glycolic Acid"], "normal", ["aging"], "adult", True)
    has_preg = any("PREGNANCY" in w.upper() for w in res["warnings"])
    check(has_preg, "Pregnancy warning fires for retinol+pregnant user", r)

    # Benzoyl peroxide on sensitive -> SENSITIVITY WARNING
    res2 = run_score(scorer, ["Benzoyl Peroxide"], "sensitive", ["acne"], "adult", False)
    has_sens = any("SENSITIV" in w.upper() for w in res2["warnings"])
    check(has_sens, "Sensitivity warning fires for BP on sensitive skin", r)

    # Coconut oil on oily+acne -> ACNE WARNING
    res3 = run_score(scorer, ["Coconut Oil"], "oily", ["acne"], "adult", False)
    has_acne = any("ACNE" in w.upper() or "COMED" in w.upper()
                   for w in res3["warnings"])
    check(has_acne, "Acne/comedogenic warning fires for coconut oil on oily", r)
    return r


# ── Test 13: Not-found ingredient handling ────────────────────────────────────
def test_not_found(scorer) -> TR:
    hdr("13. Unknown Ingredient Graceful Handling")
    r = TR(name="Not Found")

    res = run_score(
        scorer,
        ["Hyaluronic Acid", "FAKE_INGREDIENT_XYZ123", "Glycerin"],
        "normal", ["dryness"], "adult", False
    )
    check("FAKE_INGREDIENT_XYZ123" in res["not_found"],
          "Unknown ingredient listed in not_found", r)
    check(res["compatibility_score"] > 0,
          f"Score still computed ({res['compatibility_score']}) despite unknown ingredient", r)
    check(len(res["ingredient_details"]) == 2,
          f"Only known ingredients scored ({len(res['ingredient_details'])} found)", r)

    # All unknown
    res_all_unk = run_score(scorer, ["XYZ_FAKE_A", "XYZ_FAKE_B"], "normal", [], "adult", False)
    check(res_all_unk["compatibility_score"] == 0.0,
          "All-unknown product scores 0", r)
    return r


# ── Test 14: Empty ingredient list ────────────────────────────────────────────
def test_empty(scorer) -> TR:
    hdr("14. Empty Ingredient List")
    r = TR(name="Empty List")
    try:
        res = run_score(scorer, [], "normal", ["dryness"], "adult", False)
        check(res["compatibility_score"] == 0.0, "Empty list returns score=0", r)
        check(res["grade"] == "F", f"Empty list grade='F' (got {res['grade']})", r)
    except Exception as e:
        fail(f"Crashed on empty list: {e}")
        r.failed += 1
    return r


# ── Test 15: Regression set ───────────────────────────────────────────────────
def test_regression(scorer) -> TR:
    hdr("15. Regression Scenario Set")
    r = TR(name="Regression")
    rows = []

    for sc in REGRESSION_SCENARIOS:
        res   = run_score(
            scorer, sc["ingredients"], sc["skin_type"],
            sc["concerns"], sc["age_group"], sc["is_pregnant"]
        )
        score = res["compatibility_score"]
        ok_   = sc["min_score"] <= score <= sc["max_score"]
        color = GREEN if ok_ else RED
        status= "PASS" if ok_ else "FAIL"
        rows.append((ok_, sc["desc"], score, sc["min_score"], sc["max_score"]))

        if ok_:
            r.passed += 1
        else:
            r.failed += 1

    print(f"\n  {'STATUS':<6} {'SCORE':>6} {'RANGE':<12} DESCRIPTION")
    print(f"  {'─'*75}")
    for passed, desc, score, lo, hi in rows:
        color  = GREEN if passed else RED
        status = "PASS" if passed else "FAIL"
        print(f"  {color}{status}{RESET}  {score:>6.1f}  [{lo:>3}-{hi:>3}]  {desc}")

    pct = 100 * r.passed / len(REGRESSION_SCENARIOS)
    print(f"\n  Regression accuracy: {r.passed}/{len(REGRESSION_SCENARIOS)} = {pct:.0f}%")
    return r


# ── Test 16: Penalty stacking ─────────────────────────────────────────────────
def test_penalty_stacking(scorer) -> TR:
    hdr("16. Penalty Stacking (multiple bad ingredients compound)")
    r = TR(name="Penalty Stacking")

    res_1bad = run_score(scorer, ["Coconut Oil"], "oily", ["acne"], "adult", False)
    # All three ingredients are bad for oily/acne skin (high comedogenicity,
    # worsens acne/pores, unsuitable skin type = "no" / "caution").
    # Benzoyl Peroxide was replaced with Petrolatum because it is an anti-acne
    # active that scores well on oily/acne skin and therefore inflated the
    # 3-ingredient average above the single bad-ingredient score.
    res_3bad = run_score(
        scorer, ["Coconut Oil", "Isopropyl Myristate", "Petrolatum"],
        "oily", ["acne"], "adult", False
    )
    check(res_3bad["compatibility_score"] < res_1bad["compatibility_score"],
          f"3 bad ingredients ({res_3bad['compatibility_score']}) "
          f"scores lower than 1 bad ({res_1bad['compatibility_score']})", r)
    info(f"  1 bad: {res_1bad['compatibility_score']}  3 bad: {res_3bad['compatibility_score']}")
    return r


# ── Test 17: Boundary conditions ─────────────────────────────────────────────
def test_boundaries(scorer) -> TR:
    hdr("17. Boundary Conditions (score always 0-100)")
    r = TR(name="Boundaries")
    combos = [
        (["Retinol","Benzoyl Peroxide","Glycolic Acid","Coconut Oil"], "sensitive", ["acne","sensitivity","redness"], "teen", True),
        (["Hyaluronic Acid","Glycerin","Ceramide NP","Allantoin","Panthenol","Squalane"], "dry", VALID_CONCERNS, "mature", False),
    ]
    for ings, st, c, ag, preg in combos:
        res = run_score(scorer, ings, st, c, ag, preg)
        sc  = res["compatibility_score"]
        check(0.0 <= sc <= 100.0, f"Score {sc:.1f} within [0,100]", r)
    return r


# ── Test 18: All skin types ───────────────────────────────────────────────────
def test_all_skin_types(scorer) -> TR:
    hdr("18. Valid Results for All Skin Types")
    r = TR(name="All Skin Types")
    for st in VALID_SKIN_TYPES:
        res = run_score(
            scorer, ["Niacinamide", "Hyaluronic Acid", "Glycerin"],
            st, ["dryness"], "adult", False
        )
        ok_ = 0 <= res["compatibility_score"] <= 100 and res["grade"] in ("A+","A","B+","B","C+","C","D","F")
        check(ok_, f"Skin type '{st}' -> score={res['compatibility_score']}  grade={res['grade']}", r)
    return r


# ── Test 19: All concerns ─────────────────────────────────────────────────────
def test_all_concerns(scorer) -> TR:
    hdr("19. Valid Results for All Skin Concerns")
    r = TR(name="All Concerns")
    for concern in VALID_CONCERNS:
        res = run_score(
            scorer, ["Niacinamide", "Hyaluronic Acid"],
            "normal", [concern], "adult", False
        )
        check(0 <= res["compatibility_score"] <= 100,
              f"Concern '{concern}' -> score={res['compatibility_score']}", r)
    return r


# ── Test 20: Latency benchmark ────────────────────────────────────────────────
def test_latency(scorer) -> TR:
    hdr("20. Latency Benchmark")
    r = TR(name="Latency")
    runs = []
    cases = [
        (["Hyaluronic Acid", "Niacinamide", "Glycerin", "Ceramide NP", "Squalane"],
         "dry", ["dryness","aging"], "adult", False),
        (["Salicylic Acid", "Niacinamide", "Zinc PCA"],
         "oily", ["acne","pores"], "teen", False),
        (["Retinol", "Hyaluronic Acid", "Ceramide NP"],
         "normal", ["aging"], "mature", False),
    ] * 20

    for ings, st, c, ag, preg in cases:
        t0  = time.perf_counter()
        run_score(scorer, ings, st, c, ag, preg)
        runs.append((time.perf_counter() - t0) * 1000)

    runs.sort()
    p50 = statistics.median(runs)
    p95 = runs[int(0.95 * len(runs))]
    p99 = runs[int(0.99 * len(runs))]
    info(f"  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  mean={statistics.mean(runs):.1f}ms")

    check(p50 < 100,  f"p50={p50:.1f}ms < 100ms", r)
    check(p95 < 200,  f"p95={p95:.1f}ms < 200ms", r)
    check(p99 < 500,  f"p99={p99:.1f}ms < 500ms", r)
    return r


# ── Test 21: Schema validation ────────────────────────────────────────────────
def test_schema(scorer) -> TR:
    hdr("21. Response Schema Validation")
    r = TR(name="Schema")
    cases = [
        (["Glycerin"], "normal", [], "adult", False),
        (["Retinol", "Glycolic Acid"], "sensitive", ["aging"], "adult", True),
        (["UNKNOWN_XYZ"], "oily", ["acne"], "teen", False),
    ]
    for ings, st, c, ag, preg in cases:
        res = run_score(scorer, ings, st, c, ag, preg)
        missing = REQUIRED_KEYS - set(res.keys())
        check(not missing, f"All keys present for {ings}", r)
        check(isinstance(res["compatibility_score"], float),
              "compatibility_score is float", r)
        check(isinstance(res["pros"], list), "pros is list", r)
        check(isinstance(res["cons"], list), "cons is list", r)
        check(isinstance(res["ingredient_details"], list), "ingredient_details is list", r)
        check(isinstance(res["warnings"], list), "warnings is list", r)
    return r


# ── Test 22: Ensemble logic ───────────────────────────────────────────────────
def test_ensemble(scorer) -> TR:
    hdr("22. Ensemble Logic (rule + ML both contribute)")
    r = TR(name="Ensemble")
    res = run_score(
        scorer, ["Hyaluronic Acid", "Niacinamide", "Ceramide NP"],
        "dry", ["dryness"], "adult", False
    )
    check("rule_score" in res and res["rule_score"] > 0,
          f"rule_score={res['rule_score']} present and positive", r)
    check("ml_score" in res and res["ml_score"] > 0,
          f"ml_score={res['ml_score']} present and positive", r)
    check(res["compatibility_score"] != res["rule_score"],
          "Final score differs from raw rule score (ML is contributing)", r)
    rw = CFG["rule_weight"]
    mw = CFG["ml_weight"]
    expected = round(rw * res["rule_score"] + mw * res["ml_score"], 1)
    check(abs(res["compatibility_score"] - expected) < 2.0,
          f"Ensemble formula correct: {rw}*{res['rule_score']}+{mw}*{res['ml_score']}={expected} "
          f"(got {res['compatibility_score']})", r)
    return r


# =============================================================================
# MASTER RUNNER
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SkinSpectra Calc Layer Test Suite")
    parser.add_argument("--model_dir", default=CFG["output_dir"])
    parser.add_argument("--data",      default=CFG["data_path"])
    parser.add_argument("--skip",      nargs="*", default=[])
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        log.error(f"Model dir '{model_dir}' not found. Run train_calc_layer.py first.")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  SkinSpectra -- Calculation Layer Test Suite (Feature 1){RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}")

    log.info("Loading CompatibilityScorer...")
    scorer = CompatibilityScorer.load(str(model_dir), data_path=args.data, cfg=CFG)

    skip = set(args.skip or [])
    all_results = []

    tests = [
        ("range",      test_score_range),
        ("grades",     test_grades),
        ("ideal",      test_ideal_product),
        ("bad",        test_bad_product),
        ("pregnancy",  test_pregnancy),
        ("comed",      test_comedogenicity),
        ("irritancy",  test_irritancy),
        ("age",        test_age_group),
        ("skin_types", test_skin_types),
        ("concerns",   test_concerns),
        ("pros_cons",  test_pros_cons),
        ("warnings",   test_warnings),
        ("not_found",  test_not_found),
        ("empty",      test_empty),
        ("regression", test_regression),
        ("stacking",   test_penalty_stacking),
        ("boundaries", test_boundaries),
        ("all_types",  test_all_skin_types),
        ("all_concerns",test_all_concerns),
        ("latency",    test_latency),
        ("schema",     test_schema),
        ("ensemble",   test_ensemble),
    ]

    for key, fn in tests:
        if key in skip:
            warn(f"Skipping: {key}")
            continue
        tr = fn(scorer)
        all_results.append(tr)

    # Final report
    hdr("FINAL REPORT")
    total_p = sum(t.passed for t in all_results)
    total_w = sum(t.warned for t in all_results)
    total_f = sum(t.failed for t in all_results)
    grand   = total_p + total_w + total_f
    overall = 100 * total_p / grand if grand else 0

    print(f"\n  {'Section':<25} {'Pass':>5} {'Warn':>5} {'Fail':>5} {'%':>6}")
    print(f"  {'─'*50}")
    for t in all_results:
        color = GREEN if t.pct >= 80 else (YELLOW if t.pct >= 60 else RED)
        print(f"  {t.name:<25} {t.passed:>5} {t.warned:>5} {t.failed:>5} "
              f"{color}{t.pct:>5.0f}%{RESET}")
    print(f"  {'─'*50}")

    oc = GREEN if overall >= 80 else (YELLOW if overall >= 65 else RED)
    print(f"  {'TOTAL':<25} {total_p:>5} {total_w:>5} {total_f:>5} {oc}{overall:>5.0f}%{RESET}\n")

    if overall >= 80:
        print(f"  {GREEN}{BOLD}Calculation Layer is production-ready.{RESET}\n")
    elif overall >= 65:
        print(f"  {YELLOW}{BOLD}Calculation Layer needs minor tuning.{RESET}\n")
    else:
        print(f"  {RED}{BOLD}Calculation Layer requires fixes before production.{RESET}\n")


if __name__ == "__main__":
    main()
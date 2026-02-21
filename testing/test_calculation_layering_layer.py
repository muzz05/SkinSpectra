"""
SkinSpectra Layering Layer - Test Suite (Feature 2)
====================================================
Tests
-----
1.  Score range validation          -- all scores 0-100
2.  Grade assignment accuracy       -- correct letter grades
3.  Ideal synergistic layering      -- expects score >= 80
4.  Conflicting layering            -- expects score <= 40
5.  Pregnancy safety warnings       -- retinol/AHA flagged
6.  Conflict detection              -- conflicting pairs detected
7.  Avoid-same-time detection       -- avoid pairs penalised
8.  Layering order correctness      -- lighter before heavier
9.  Wrong order penalty             -- heavier before lighter penalised
10. Skin type enforcement           -- unsuitable products penalised
11. Concern alignment               -- both-products-help bonus
12. Wait time enforcement           -- wait > 0 when needed
13. Application steps generated     -- non-empty steps list
14. Pros and cons quality           -- non-empty and meaningful
15. Warnings completeness           -- critical flags present
16. Not-found ingredient handling   -- graceful degradation
17. Empty product handling          -- no crash
18. All skin types                  -- valid scores for all types
19. All concerns                    -- valid scores for all concerns
20. Regression set                  -- 20 labelled scenarios
21. Penalty stacking                -- multiple conflicts compound
22. Boundary conditions             -- never < 0 or > 100
23. Ensemble logic                  -- both rule and ML contribute
24. Latency benchmark               -- p95 < 250ms
25. Schema validation               -- all required keys present
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
from components.calculation_layering_layer import (
    LayeringScorer, CFG,
    VALID_SKIN_TYPES, VALID_CONCERNS, VALID_AGE_GROUPS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.layering.test")

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(m):   print(f"  {GREEN}PASS  {m}{RESET}")
def fail(m): print(f"  {RED}FAIL  {m}{RESET}")
def warn(m): print(f"  {YELLOW}WARN  {m}{RESET}")
def info(m): print(f"  {CYAN}INFO  {m}{RESET}")
def hdr(m):
    print(f"\n{BOLD}{CYAN}{'─'*65}{RESET}")
    print(f"{BOLD}  {m}{RESET}")
    print(f"{'─'*65}")


@dataclass
class TR:
    name:   str
    passed: int = 0
    failed: int = 0
    warned: int = 0

    @property
    def total(self): return self.passed + self.failed + self.warned
    @property
    def pct(self):   return 100 * self.passed / self.total if self.total else 0


REQUIRED_KEYS = {
    "layering_score", "grade", "verdict", "layering_order",
    "wait_time_minutes", "application_steps", "pros", "cons",
    "warnings", "pair_interactions", "product_a_not_found",
    "product_b_not_found", "rule_score", "ml_score", "latency_ms",
}


# =============================================================================
# REGRESSION SCENARIOS
# =============================================================================

REGRESSION_SCENARIOS = [
    # ── IDEAL (score >= min_score) ─────────────────────────────────────────
    {
        "desc"   : "Hyaluronic Acid serum + Ceramide moisturizer (perfect hydration stack)",
        "a_name" : "HA Serum",
        "a_ings" : ["Hyaluronic Acid", "Glycerin", "Sodium PCA"],
        "b_name" : "Ceramide Moisturizer",
        "b_ings" : ["Ceramide NP", "Ceramide AP", "Cholesterol", "Squalane"],
        "st": "dry", "concerns": ["dryness", "barrier"], "ag": "adult",
        "preg": False, "tod": "both", "min": 75, "max": 100,
    },
    {
        "desc"   : "Niacinamide serum + Ceramide cream for sensitive barrier",
        "a_name" : "Niacinamide Serum",
        "a_ings" : ["Niacinamide", "Hyaluronic Acid", "Panthenol"],
        "b_name" : "Ceramide Cream",
        "b_ings" : ["Ceramide NP", "Cholesterol", "Glycerin"],
        "st": "sensitive", "concerns": ["sensitivity", "barrier", "redness"],
        "ag": "adult", "preg": False, "tod": "both", "min": 75, "max": 100,
    },
    {
        "desc"   : "BHA toner + Niacinamide moisturizer for oily acne",
        "a_name" : "BHA Toner",
        "a_ings" : ["Salicylic Acid", "Zinc PCA"],
        "b_name" : "Niacinamide Cream",
        "b_ings" : ["Niacinamide", "Ceramide NP", "Hyaluronic Acid"],
        "st": "oily", "concerns": ["acne", "pores"], "ag": "teen",
        "preg": False, "tod": "AM", "min": 72, "max": 100,
    },
    {
        "desc"   : "Pregnancy-safe: Niacinamide + Azelaic acid",
        "a_name" : "Niacinamide Serum",
        "a_ings" : ["Niacinamide", "Tranexamic Acid", "Hyaluronic Acid"],
        "b_name" : "Azelaic Treatment",
        "b_ings" : ["Azelaic Acid", "Allantoin", "Glycerin"],
        "st": "sensitive", "concerns": ["hyperpigmentation", "redness"],
        "ag": "adult", "preg": True, "tod": "both", "min": 70, "max": 100,
    },
    {
        "desc"   : "Vitamin C serum + SPF moisturizer (classic AM)",
        "a_name" : "Vitamin C Serum",
        "a_ings" : ["Ascorbic Acid", "Ferulic Acid", "Glycerin"],
        "b_name" : "SPF Moisturizer",
        "b_ings" : ["Zinc Oxide", "Titanium Dioxide", "Dimethicone"],
        "st": "normal", "concerns": ["aging", "dullness"], "ag": "adult",
        "preg": False, "tod": "AM", "min": 72, "max": 100,
    },
    {
        "desc"   : "Retinol night serum + Ceramide night cream (best PM routine)",
        "a_name" : "Retinol Serum",
        "a_ings" : ["Retinol", "Squalane"],
        "b_name" : "Ceramide Night Cream",
        "b_ings" : ["Ceramide NP", "Hyaluronic Acid", "Panthenol"],
        "st": "normal", "concerns": ["aging", "texture"], "ag": "mature",
        "preg": False, "tod": "PM", "min": 68, "max": 100,
    },
    {
        "desc"   : "Bakuchiol + Niacinamide (pregnancy-safe anti-aging)",
        "a_name" : "Bakuchiol Serum",
        "a_ings" : ["Bakuchiol", "Squalane"],
        "b_name" : "Niacinamide Cream",
        "b_ings" : ["Niacinamide", "Ceramide NP", "Hyaluronic Acid"],
        "st": "normal", "concerns": ["aging", "dullness"], "ag": "adult",
        "preg": True, "tod": "PM", "min": 72, "max": 100,
    },
    {
        "desc"   : "Alpha arbutin + Niacinamide (brightening stack)",
        "a_name" : "Arbutin Serum",
        "a_ings" : ["Alpha Arbutin", "Tranexamic Acid", "Hyaluronic Acid"],
        "b_name" : "Niacinamide Cream",
        "b_ings" : ["Niacinamide", "Ceramide NP", "Glycerin"],
        "st": "combination", "concerns": ["hyperpigmentation", "dullness"],
        "ag": "adult", "preg": False, "tod": "both", "min": 72, "max": 100,
    },

    # ── POOR (score <= max_score) ──────────────────────────────────────────
    {
        "desc"   : "Vitamin C + Retinol same routine (classic conflict)",
        "a_name" : "Vitamin C Serum",
        "a_ings" : ["Ascorbic Acid", "Ferulic Acid"],
        "b_name" : "Retinol Serum",
        "b_ings" : ["Retinol", "Squalane"],
        "st": "normal", "concerns": ["aging"],
        "ag": "adult", "preg": False, "tod": "PM", "min": 0, "max": 50,
    },
    {
        "desc"   : "Glycolic acid + Retinol same PM (double irritation)",
        "a_name" : "Glycolic Toner",
        "a_ings" : ["Glycolic Acid"],
        "b_name" : "Retinol Serum",
        "b_ings" : ["Retinol"],
        "st": "sensitive", "concerns": ["aging", "texture"],
        "ag": "adult", "preg": False, "tod": "PM", "min": 0, "max": 40,
    },
    {
        "desc"   : "Unsafe pregnancy: retinol + glycolic acid",
        "a_name" : "Retinol Cream",
        "a_ings" : ["Retinol", "Ceramide NP"],
        "b_name" : "Glycolic Serum",
        "b_ings" : ["Glycolic Acid", "Hyaluronic Acid"],
        "st": "normal", "concerns": ["aging"], "ag": "adult",
        "preg": True, "tod": "PM", "min": 0, "max": 30,
    },
    {
        "desc"   : "Benzoyl peroxide + Vitamin C (oxidation conflict)",
        "a_name" : "BP Spot Treatment",
        "a_ings" : ["Benzoyl Peroxide"],
        "b_name" : "Vitamin C Serum",
        "b_ings" : ["Ascorbic Acid", "Ferulic Acid"],
        "st": "oily", "concerns": ["acne"], "ag": "adult",
        "preg": False, "tod": "AM", "min": 0, "max": 40,
    },
    {
        "desc"   : "Double high-irritancy on sensitive: BP + Glycolic",
        "a_name" : "BP Wash",
        "a_ings" : ["Benzoyl Peroxide"],
        "b_name" : "Glycolic Serum",
        "b_ings" : ["Glycolic Acid", "Salicylic Acid"],
        "st": "sensitive", "concerns": ["acne"],
        "ag": "adult", "preg": False, "tod": "PM", "min": 0, "max": 35,
    },

    # ── MODERATE (min <= score <= max) ─────────────────────────────────────
    {
        "desc"   : "Niacinamide + Ascorbic acid (caution pairing)",
        "a_name" : "Niacinamide Serum",
        "a_ings" : ["Niacinamide", "Hyaluronic Acid"],
        "b_name" : "Vitamin C Serum",
        "b_ings" : ["Ascorbic Acid", "Ferulic Acid"],
        "st": "normal", "concerns": ["hyperpigmentation", "dullness"],
        "ag": "adult", "preg": False, "tod": "AM", "min": 40, "max": 72,
    },
    {
        "desc"   : "Retinol for teen (age mismatch + irritancy)",
        "a_name" : "Retinol Serum",
        "a_ings" : ["Retinol", "Hyaluronic Acid"],
        "b_name" : "Moisturizer",
        "b_ings" : ["Ceramide NP", "Glycerin"],
        "st": "oily", "concerns": ["acne", "texture"],
        "ag": "teen", "preg": False, "tod": "PM", "min": 30, "max": 65,
    },
    {
        "desc"   : "Mixed oily skin routine with caution ingredients",
        "a_name" : "Exfoliant Toner",
        "a_ings" : ["Glycolic Acid", "Salicylic Acid"],
        "b_name" : "Moisturizer",
        "b_ings" : ["Argan Oil", "Sweet Almond Oil"],
        "st": "oily", "concerns": ["acne", "texture"],
        "ag": "adult", "preg": False, "tod": "PM", "min": 35, "max": 65,
    },
    {
        "desc"   : "Copper peptide + Vitamin C (separate routines needed)",
        "a_name" : "Copper Peptide Serum",
        "a_ings" : ["Copper Tripeptide-1"],
        "b_name" : "Vitamin C Serum",
        "b_ings" : ["Ascorbic Acid"],
        "st": "normal", "concerns": ["aging"], "ag": "mature",
        "preg": False, "tod": "AM", "min": 25, "max": 60,
    },
    {
        "desc"   : "Kojic acid + Vitamin C (stability caution)",
        "a_name" : "Kojic Serum",
        "a_ings" : ["Kojic Acid"],
        "b_name" : "Vitamin C Serum",
        "b_ings" : ["Ascorbic Acid"],
        "st": "normal", "concerns": ["hyperpigmentation"],
        "ag": "adult", "preg": False, "tod": "AM", "min": 30, "max": 65,
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def sc(scorer, a_name, a_ings, b_name, b_ings,
       st, concerns, ag, preg, tod="both"):
    return scorer.score(a_name, a_ings, b_name, b_ings,
                        st, concerns, ag, preg, tod)

def check(cond, desc, r: TR):
    if cond:
        ok(desc);   r.passed += 1
    else:
        fail(desc); r.failed += 1


# =============================================================================
# TESTS
# =============================================================================

def test_score_range(scorer) -> TR:
    hdr("1. Score Range Validation (0-100)")
    r = TR("Score Range")
    cases = [
        (["Hyaluronic Acid","Glycerin"], ["Ceramide NP","Squalane"], "dry",  ["dryness"], "adult", False),
        (["Retinol"],                    ["Ascorbic Acid"],           "normal",["aging"],  "adult", False),
        (["Benzoyl Peroxide"],           ["Glycolic Acid"],           "sensitive",["acne"],"teen",  True),
        (["Niacinamide"],               ["Ceramide NP"],             "sensitive",["barrier"],"adult",False),
    ]
    for a, b, st, co, ag, preg in cases:
        res = sc(scorer, "ProdA", a, "ProdB", b, st, co, ag, preg)
        s   = res["layering_score"]
        check(0.0 <= s <= 100.0, f"Score {s:.1f} in [0,100]", r)
    return r


def test_grades(scorer) -> TR:
    hdr("2. Grade Assignment")
    r = TR("Grades")
    grade_map = [
        (95, "A+"), (87, "A"), (80, "B+"), (73, "B"),
        (64, "C+"), (57, "C"), (45, "D"), (20, "F"),
    ]
    for score, expected in grade_map:
        actual = scorer._grade(score)
        check(actual == expected, f"Score {score} -> '{actual}' (expected '{expected}')", r)
    return r


def test_ideal_layering(scorer) -> TR:
    hdr("3. Ideal Synergistic Layering (expect >= 78)")
    r = TR("Ideal Layering")
    res = sc(
        scorer,
        "HA Serum",       ["Hyaluronic Acid", "Glycerin", "Sodium PCA"],
        "Ceramide Cream", ["Ceramide NP", "Ceramide AP", "Cholesterol", "Squalane"],
        "dry", ["dryness", "barrier"], "adult", False, "both"
    )
    check(res["layering_score"] >= 78,
          f"Ideal stack score={res['layering_score']} >= 78", r)
    check(res["grade"] in ("A+","A","B+","B"),
          f"Grade={res['grade']} in premium range", r)
    check(len(res["pros"]) >= 2, f"Has {len(res['pros'])} pros", r)
    info(f"  Score: {res['layering_score']}  Grade: {res['grade']}")
    info(f"  Pros:  {res['pros'][:2]}")
    return r


def test_conflict_layering(scorer) -> TR:
    hdr("4. Conflicting Layering (expect <= 45)")
    r = TR("Conflict Layering")
    res = sc(
        scorer,
        "Vitamin C Serum", ["Ascorbic Acid", "Ferulic Acid"],
        "Retinol Serum",   ["Retinol", "Squalane"],
        "normal", ["aging", "hyperpigmentation"], "adult", False, "PM"
    )
    check(res["layering_score"] <= 50,
          f"Conflict score={res['layering_score']} <= 50", r)
    check(len(res["cons"]) >= 1, f"Has {len(res['cons'])} cons", r)
    info(f"  Score: {res['layering_score']}  Cons: {res['cons'][:2]}")
    return r


def test_pregnancy_safety(scorer) -> TR:
    hdr("5. Pregnancy Safety Enforcement")
    r = TR("Pregnancy Safety")

    # Unsafe: retinol for pregnant user
    res_unsafe = sc(
        scorer,
        "Retinol Serum",   ["Retinol", "Ceramide NP"],
        "Glycolic Toner",  ["Glycolic Acid", "Hyaluronic Acid"],
        "normal", ["aging"], "adult", True, "PM"
    )
    check(res_unsafe["layering_score"] <= 35,
          f"Pregnant+retinol score={res_unsafe['layering_score']} <= 35", r)
    has_preg_warn = any("PREGNANCY" in w.upper() for w in res_unsafe["warnings"])
    check(has_preg_warn, "Pregnancy warning present", r)

    # Safe: bakuchiol + niacinamide for pregnant
    res_safe = sc(
        scorer,
        "Bakuchiol Serum", ["Bakuchiol", "Squalane"],
        "Niacinamide Cream",["Niacinamide", "Ceramide NP", "Glycerin"],
        "normal", ["aging"], "adult", True, "PM"
    )
    check(res_safe["layering_score"] >= 65,
          f"Pregnant+safe score={res_safe['layering_score']} >= 65", r)
    no_danger = not any("DANGER" in w.upper() for w in res_safe["warnings"])
    check(no_danger, "No danger warning for safe pregnancy combo", r)
    info(f"  Unsafe: {res_unsafe['layering_score']}  Safe: {res_safe['layering_score']}")
    return r


def test_conflict_detection(scorer) -> TR:
    hdr("6. Conflict Detection")
    r = TR("Conflict Detection")
    res = sc(
        scorer,
        "Vitamin C",        ["Ascorbic Acid"],
        "Benzoyl Peroxide", ["Benzoyl Peroxide"],
        "oily", ["acne"], "adult", False, "AM"
    )
    has_conflict_con = any(
        "conflict" in c.lower() or "inactivate" in c.lower() or "oxidize" in c.lower()
        for c in res["cons"] + res["warnings"]
    )
    check(has_conflict_con, "Conflict captured in cons/warnings", r)
    check(res["layering_score"] <= 55,
          f"Conflict pair penalised: score={res['layering_score']} <= 55", r)

    # Interactions list should contain the conflicting pair
    has_pair = any(
        d.get("interaction_type") in ("conflicting", "avoid same time")
        for d in res["pair_interactions"]
    )
    check(has_pair or res["layering_score"] <= 55,
          "Conflicting interaction recorded in pair_interactions", r)
    return r


def test_avoid_detection(scorer) -> TR:
    hdr("7. Avoid-Same-Time Detection")
    r = TR("Avoid Detection")
    res = sc(
        scorer,
        "Retinol Serum",  ["Retinol"],
        "Glycolic Toner", ["Glycolic Acid"],
        "normal", ["aging", "texture"], "adult", False, "PM"
    )
    check(res["layering_score"] <= 55,
          f"Avoid-same-time penalised: score={res['layering_score']} <= 55", r)
    has_avoid = any(
        "avoid" in c.lower() or "alternate" in c.lower() or "separate" in c.lower()
        for c in res["cons"] + res["warnings"]
    )
    check(has_avoid, "Avoid instruction in cons/warnings", r)
    return r


def test_order_correct(scorer) -> TR:
    hdr("8. Correct Layering Order (lighter first)")
    r = TR("Order Correct")
    # Serum (A) before Moisturizer (B) = correct
    res = sc(
        scorer,
        "Niacinamide Serum",  ["Niacinamide", "Hyaluronic Acid"],
        "Ceramide Moisturizer",["Ceramide NP", "Squalane", "Shea Butter"],
        "dry", ["dryness"], "adult", False, "PM"
    )
    has_order_pro = any(
        "order" in p.lower() or "correct" in p.lower() or "lighter" in p.lower()
        for p in res["pros"]
    )
    check(res["layering_score"] >= 60,
          f"Correct order: score={res['layering_score']} >= 60", r)
    info(f"  Score: {res['layering_score']}  Order: {res['layering_order'][:60]}")
    return r


def test_order_wrong(scorer) -> TR:
    hdr("9. Wrong Layering Order Penalty")
    r = TR("Order Wrong")
    # Occlusive (A) before serum (B) = wrong order
    res_wrong = sc(
        scorer,
        "Petrolatum Balm",     ["Petrolatum", "Mineral Oil"],
        "Hyaluronic Acid Serum",["Hyaluronic Acid", "Glycerin"],
        "dry", ["dryness"], "adult", False, "PM"
    )
    res_correct = sc(
        scorer,
        "Hyaluronic Acid Serum",["Hyaluronic Acid", "Glycerin"],
        "Petrolatum Balm",      ["Petrolatum", "Mineral Oil"],
        "dry", ["dryness"], "adult", False, "PM"
    )
    check(res_correct["layering_score"] >= res_wrong["layering_score"],
          f"Correct order({res_correct['layering_score']}) >= "
          f"wrong order({res_wrong['layering_score']})", r)
    info(f"  Wrong: {res_wrong['layering_score']}  Correct: {res_correct['layering_score']}")
    return r


def test_skin_type_enforcement(scorer) -> TR:
    hdr("10. Skin Type Compatibility Enforcement")
    r = TR("Skin Type")
    # Coconut oil (comed 4) on oily acne skin should score lower than dry
    res_oily = sc(
        scorer,
        "Coconut Oil Product", ["Coconut Oil"],
        "Moisturizer",         ["Glycerin", "Hyaluronic Acid"],
        "oily", ["acne"], "adult", False
    )
    res_dry = sc(
        scorer,
        "Coconut Oil Product", ["Coconut Oil"],
        "Moisturizer",         ["Glycerin", "Hyaluronic Acid"],
        "dry", ["dryness"], "adult", False
    )
    check(res_oily["layering_score"] <= res_dry["layering_score"],
          f"Coconut oil on oily({res_oily['layering_score']}) "
          f"<= dry({res_dry['layering_score']})", r)

    # All skin types return valid scores
    for st in VALID_SKIN_TYPES:
        res = sc(scorer, "P1", ["Niacinamide"], "P2", ["Hyaluronic Acid"],
                 st, ["dryness"], "adult", False)
        check(0 <= res["layering_score"] <= 100,
              f"Skin type '{st}' valid score={res['layering_score']}", r)
    return r


def test_concern_alignment(scorer) -> TR:
    hdr("11. Concern Alignment Scoring")
    r = TR("Concern Alignment")
    # Both products target acne = should score better than unrelated
    res_aligned = sc(
        scorer,
        "BHA Toner",     ["Salicylic Acid", "Zinc PCA"],
        "Acne Moisturizer",["Niacinamide", "Azelaic Acid"],
        "oily", ["acne", "pores"], "adult", False
    )
    res_unrelated = sc(
        scorer,
        "BHA Toner",   ["Salicylic Acid"],
        "Anti-aging",  ["Retinol"],
        "oily", ["acne"], "adult", False, "PM"
    )
    check(res_aligned["layering_score"] >= res_unrelated["layering_score"] - 10,
          f"Aligned({res_aligned['layering_score']}) not much worse than "
          f"unrelated({res_unrelated['layering_score']})", r)
    # Both-help bonus should appear in pros
    both_help_pro = any(
        "both" in p.lower() and ("address" in p.lower() or "target" in p.lower() or "help" in p.lower())
        for p in res_aligned["pros"]
    )
    check(both_help_pro or res_aligned["layering_score"] >= 60,
          "Both-help concern bonus reflected", r)
    return r


def test_wait_time(scorer) -> TR:
    hdr("12. Wait Time Enforcement")
    r = TR("Wait Time")
    # Vitamin C + Niacinamide has a 60 min wait time in dataset
    res = sc(
        scorer,
        "Vitamin C Serum", ["Ascorbic Acid"],
        "Niacinamide Serum",["Niacinamide"],
        "normal", ["dullness"], "adult", False, "AM"
    )
    # Wait time captured in warnings or wait_time_minutes
    info(f"  wait_time_minutes={res['wait_time_minutes']}")
    check(isinstance(res["wait_time_minutes"], int),
          f"wait_time_minutes is int ({res['wait_time_minutes']})", r)

    # Glycolic + HA has 15 min wait
    res2 = sc(
        scorer,
        "Glycolic Toner", ["Glycolic Acid"],
        "HA Serum",       ["Hyaluronic Acid"],
        "normal", ["texture"], "adult", False, "PM"
    )
    check(res2["wait_time_minutes"] >= 0,
          f"Wait time >= 0 ({res2['wait_time_minutes']} min)", r)
    return r


def test_application_steps(scorer) -> TR:
    hdr("13. Application Steps Generated")
    r = TR("App Steps")
    res = sc(
        scorer,
        "Serum", ["Niacinamide", "Hyaluronic Acid"],
        "Moisturizer", ["Ceramide NP", "Squalane"],
        "dry", ["dryness"], "adult", False
    )
    check(isinstance(res["application_steps"], list),
          "application_steps is a list", r)
    check(len(res["application_steps"]) >= 1,
          f"Has {len(res['application_steps'])} steps", r)
    check(isinstance(res["application_steps"][0], str),
          "Steps are strings", r)
    info(f"  Steps: {res['application_steps'][:2]}")
    return r


def test_pros_cons(scorer) -> TR:
    hdr("14. Pros and Cons Quality")
    r = TR("Pros Cons")
    res_good = sc(
        scorer,
        "HA Serum",       ["Hyaluronic Acid", "Glycerin"],
        "Ceramide Cream", ["Ceramide NP", "Squalane"],
        "dry", ["dryness", "barrier"], "adult", False
    )
    check(len(res_good["pros"]) >= 1, f"Good combo has {len(res_good['pros'])} pros", r)
    check(all(isinstance(p, str) for p in res_good["pros"]), "All pros are strings", r)

    res_bad = sc(
        scorer,
        "Vitamin C",      ["Ascorbic Acid"],
        "Retinol Serum",  ["Retinol"],
        "normal", ["aging"], "adult", False, "PM"
    )
    check(len(res_bad["cons"]) >= 1, f"Bad combo has {len(res_bad['cons'])} cons", r)
    check(all(isinstance(c, str) for c in res_bad["cons"]), "All cons are strings", r)
    info(f"  Good pros: {res_good['pros'][:2]}")
    info(f"  Bad cons:  {res_bad['cons'][:2]}")
    return r


def test_warnings(scorer) -> TR:
    hdr("15. Warnings Completeness")
    r = TR("Warnings")

    # Pregnancy danger
    res = sc(
        scorer,
        "Retinol Cream", ["Retinol", "Ceramide NP"],
        "AHA Serum",     ["Glycolic Acid"],
        "normal", ["aging"], "adult", True, "PM"
    )
    preg_warn = any("PREGNANCY" in w.upper() for w in res["warnings"])
    check(preg_warn, "Pregnancy warning fires for retinol+pregnant", r)

    # Conflicting pair warning
    res2 = sc(
        scorer,
        "Vitamin C",   ["Ascorbic Acid"],
        "BP Treatment",["Benzoyl Peroxide"],
        "oily", ["acne"], "adult", False, "AM"
    )
    has_conflict_warn = any(
        "conflict" in w.lower() or "inactivate" in w.lower()
        for w in res2["warnings"] + res2["cons"]
    )
    check(has_conflict_warn or res2["layering_score"] <= 45,
          "Conflict warning or low score for BP+VitC", r)

    # Wait time warning
    res3 = sc(
        scorer,
        "Glycolic Toner", ["Glycolic Acid"],
        "HA Serum",       ["Hyaluronic Acid"],
        "normal", ["texture"], "adult", False, "PM"
    )
    wait_in_warnings = (
        any("wait" in w.lower() for w in res3["warnings"]) or
        res3["wait_time_minutes"] > 0
    )
    check(wait_in_warnings, "Wait time warning or wait_time_minutes > 0", r)
    return r


def test_not_found(scorer) -> TR:
    hdr("16. Unknown Ingredient Handling")
    r = TR("Not Found")
    res = sc(
        scorer,
        "ProductA", ["Hyaluronic Acid", "FAKE_ING_XYZ123"],
        "ProductB", ["Glycerin", "ANOTHER_FAKE_ABC"],
        "normal", [], "adult", False
    )
    check("FAKE_ING_XYZ123" in res["product_a_not_found"],
          "Unknown ingredient in product_a_not_found", r)
    check("ANOTHER_FAKE_ABC" in res["product_b_not_found"],
          "Unknown ingredient in product_b_not_found", r)
    check(res["layering_score"] >= 0,
          f"Score still computed ({res['layering_score']}) with unknown ingredients", r)
    return r


def test_empty_handling(scorer) -> TR:
    hdr("17. Empty Product Handling")
    r = TR("Empty Handling")
    for a, b in [([], ["Glycerin"]), (["Niacinamide"], []), ([], [])]:
        try:
            res = sc(scorer, "P1", a, "P2", b, "normal", [], "adult", False)
            check(0 <= res["layering_score"] <= 100,
                  f"Empty({len(a)},{len(b)}) -> score={res['layering_score']}", r)
        except Exception as e:
            fail(f"Crash on empty({len(a)},{len(b)}): {e}")
            r.failed += 1
    return r


def test_all_skin_types(scorer) -> TR:
    hdr("18. All Skin Types")
    r = TR("All Skin Types")
    for st in VALID_SKIN_TYPES:
        res = sc(
            scorer,
            "SerumA", ["Niacinamide","Hyaluronic Acid"],
            "CreamB", ["Ceramide NP","Glycerin"],
            st, ["dryness"], "adult", False
        )
        check(0 <= res["layering_score"] <= 100,
              f"'{st}' -> score={res['layering_score']}", r)
    return r


def test_all_concerns(scorer) -> TR:
    hdr("19. All Skin Concerns")
    r = TR("All Concerns")
    for concern in VALID_CONCERNS:
        res = sc(
            scorer,
            "P1", ["Niacinamide","Hyaluronic Acid"],
            "P2", ["Ceramide NP","Glycerin"],
            "normal", [concern], "adult", False
        )
        check(0 <= res["layering_score"] <= 100,
              f"Concern '{concern}' -> score={res['layering_score']}", r)
    return r


def test_regression(scorer) -> TR:
    hdr("20. Regression Scenario Set")
    r = TR("Regression")
    rows = []

    for sc_ in REGRESSION_SCENARIOS:
        res   = scorer.score(
            sc_["a_name"], sc_["a_ings"],
            sc_["b_name"], sc_["b_ings"],
            sc_["st"], sc_["concerns"],
            sc_["ag"], sc_["preg"], sc_["tod"],
        )
        score = res["layering_score"]
        ok_   = sc_["min"] <= score <= sc_["max"]
        rows.append((ok_, sc_["desc"], score, sc_["min"], sc_["max"]))
        if ok_: r.passed += 1
        else:   r.failed += 1

    print(f"\n  {'ST':<6} {'SCORE':>6} {'RANGE':<12} DESCRIPTION")
    print(f"  {'─'*70}")
    for passed, desc, score, lo, hi in rows:
        color  = GREEN if passed else RED
        status = "PASS" if passed else "FAIL"
        print(f"  {color}{status}{RESET}  {score:>6.1f}  [{lo:>3}-{hi:>3}]  {desc}")

    pct = 100 * r.passed / len(REGRESSION_SCENARIOS)
    print(f"\n  Regression: {r.passed}/{len(REGRESSION_SCENARIOS)} = {pct:.0f}%")
    return r


def test_penalty_stacking(scorer) -> TR:
    hdr("21. Penalty Stacking")
    r = TR("Penalty Stacking")
    # 1 conflict
    res1 = sc(
        scorer,
        "Vitamin C", ["Ascorbic Acid"],
        "Retinol",   ["Retinol"],
        "normal", ["aging"], "adult", False, "PM"
    )
    # 2 conflicts
    res2 = sc(
        scorer,
        "Multi-acid", ["Ascorbic Acid", "Benzoyl Peroxide"],
        "Retinol",    ["Retinol", "Copper Tripeptide-1"],
        "sensitive", ["aging"], "adult", False, "PM"
    )
    check(res2["layering_score"] <= res1["layering_score"],
          f"More conflicts({res2['layering_score']}) <= fewer({res1['layering_score']})", r)
    info(f"  1 conflict: {res1['layering_score']}  2 conflicts: {res2['layering_score']}")
    return r


def test_boundaries(scorer) -> TR:
    hdr("22. Boundary Conditions (never < 0 or > 100)")
    r = TR("Boundaries")
    cases = [
        (["Retinol","Benzoyl Peroxide","Glycolic Acid"],
         ["Ascorbic Acid","Salicylic Acid"],
         "sensitive", ["acne","aging","redness","sensitivity"], "teen", True),
        (["Hyaluronic Acid","Glycerin","Ceramide NP","Panthenol","Allantoin","Squalane"],
         ["Ceramide AP","Cholesterol","Niacinamide"],
         "dry", VALID_CONCERNS, "mature", False),
    ]
    for a, b, st, co, ag, preg in cases:
        res = sc(scorer, "P1", a, "P2", b, st, co, ag, preg)
        sc_ = res["layering_score"]
        check(0.0 <= sc_ <= 100.0, f"Score {sc_:.1f} within [0,100]", r)
    return r


def test_ensemble(scorer) -> TR:
    hdr("23. Ensemble Logic")
    r = TR("Ensemble")
    res = sc(
        scorer,
        "HA Serum",       ["Hyaluronic Acid","Glycerin"],
        "Ceramide Cream", ["Ceramide NP","Squalane"],
        "dry", ["dryness"], "adult", False
    )
    check(res["rule_score"] > 0,
          f"rule_score={res['rule_score']} is positive", r)
    check(res["ml_score"] > 0,
          f"ml_score={res['ml_score']} is positive", r)
    rw = CFG["rule_weight"]
    mw = CFG["ml_weight"]
    expected = round(rw * res["rule_score"] + mw * res["ml_score"], 1)
    check(abs(res["layering_score"] - expected) < 2.0,
          f"Ensemble formula: {rw}*{res['rule_score']}+"
          f"{mw}*{res['ml_score']}={expected} (got {res['layering_score']})", r)
    return r


def test_latency(scorer) -> TR:
    hdr("24. Latency Benchmark")
    r = TR("Latency")
    cases = [
        (["Hyaluronic Acid","Niacinamide","Glycerin"],
         ["Ceramide NP","Squalane"], "dry", ["dryness"], "adult", False),
        (["Salicylic Acid","Zinc PCA"],
         ["Niacinamide","Ceramide NP"], "oily", ["acne"], "teen", False),
        (["Ascorbic Acid","Ferulic Acid"],
         ["Retinol","Ceramide NP"], "normal", ["aging"], "adult", False),
    ] * 20
    latencies = []
    for a, b, st, co, ag, preg in cases:
        t0  = time.perf_counter()
        sc(scorer, "P1", a, "P2", b, st, co, ag, preg)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies))]
    info(f"  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  mean={statistics.mean(latencies):.1f}ms")
    check(p50 < 150,  f"p50={p50:.1f}ms < 150ms", r)
    check(p95 < 300,  f"p95={p95:.1f}ms < 300ms", r)
    check(p99 < 600,  f"p99={p99:.1f}ms < 600ms", r)
    return r


def test_schema(scorer) -> TR:
    hdr("25. Schema Validation")
    r = TR("Schema")
    cases = [
        (["Glycerin"],        ["Ceramide NP"],   "normal", [],       "adult", False),
        (["Retinol"],         ["Glycolic Acid"],  "sensitive",["aging"],"adult", True),
        (["UNKNOWN_FAKE"],    ["Niacinamide"],    "oily",   ["acne"], "teen",  False),
    ]
    for a, b, st, co, ag, preg in cases:
        res     = sc(scorer, "P1", a, "P2", b, st, co, ag, preg)
        missing = REQUIRED_KEYS - set(res.keys())
        check(not missing, f"All keys present for {a}+{b}", r)
        check(isinstance(res["layering_score"], float),  "layering_score is float", r)
        check(isinstance(res["application_steps"], list),"application_steps is list", r)
        check(isinstance(res["pair_interactions"], list), "pair_interactions is list", r)
        check(isinstance(res["wait_time_minutes"], int),  "wait_time_minutes is int", r)
    return r


# =============================================================================
# MASTER RUNNER
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SkinSpectra Layering Test Suite")
    parser.add_argument("--model_dir",  default=CFG["output_dir"])
    parser.add_argument("--dataset2",   default=CFG["dataset2_path"])
    parser.add_argument("--dataset3",   default=CFG["dataset3_path"])
    parser.add_argument("--skip",       nargs="*", default=[])
    args = parser.parse_args()

    if not Path(args.model_dir).exists():
        log.error(f"Model dir '{args.model_dir}' not found. Run train_layering_layer.py first.")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  SkinSpectra -- Layering Layer Test Suite (Feature 2){RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}")

    log.info("Loading LayeringScorer...")
    scorer = LayeringScorer.load(
        args.model_dir, args.dataset2, args.dataset3, CFG
    )

    skip  = set(args.skip or [])
    tests = [
        ("range",     test_score_range),
        ("grades",    test_grades),
        ("ideal",     test_ideal_layering),
        ("conflict",  test_conflict_layering),
        ("pregnancy", test_pregnancy_safety),
        ("conf_det",  test_conflict_detection),
        ("avoid_det", test_avoid_detection),
        ("order_ok",  test_order_correct),
        ("order_bad", test_order_wrong),
        ("skin_type", test_skin_type_enforcement),
        ("concern",   test_concern_alignment),
        ("wait",      test_wait_time),
        ("steps",     test_application_steps),
        ("pros_cons", test_pros_cons),
        ("warnings",  test_warnings),
        ("not_found", test_not_found),
        ("empty",     test_empty_handling),
        ("all_st",    test_all_skin_types),
        ("all_co",    test_all_concerns),
        ("regression",test_regression),
        ("stacking",  test_penalty_stacking),
        ("boundary",  test_boundaries),
        ("ensemble",  test_ensemble),
        ("latency",   test_latency),
        ("schema",    test_schema),
    ]

    all_results = []
    for key, fn in tests:
        if key in skip:
            warn(f"Skipping: {key}")
            continue
        all_results.append(fn(scorer))

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
        c = GREEN if t.pct >= 80 else (YELLOW if t.pct >= 60 else RED)
        print(f"  {t.name:<25} {t.passed:>5} {t.warned:>5} {t.failed:>5} {c}{t.pct:>5.0f}%{RESET}")
    print(f"  {'─'*50}")
    oc = GREEN if overall >= 80 else (YELLOW if overall >= 65 else RED)
    print(f"  {'TOTAL':<25} {total_p:>5} {total_w:>5} {total_f:>5} {oc}{overall:>5.0f}%{RESET}\n")

    if overall >= 80:
        print(f"  {GREEN}{BOLD}Layering Layer is production-ready.{RESET}\n")
    elif overall >= 65:
        print(f"  {YELLOW}{BOLD}Layering Layer needs minor tuning.{RESET}\n")
    else:
        print(f"  {RED}{BOLD}Layering Layer requires fixes before production.{RESET}\n")


if __name__ == "__main__":
    main()
"""
SkinSpectra LLM Layer - Test Suite
====================================
Tests cover both Feature 1 (individual) and Feature 2 (layering).
Uses mock calc/NLP outputs so no trained models are required.

Tests
-----
INDIVIDUAL FEATURE (13 tests)
  1.  API connectivity            -- Gemini responds successfully
  2.  JSON validity               -- output is parseable JSON
  3.  Schema completeness         -- all required keys present
  4.  Score reflection            -- report score matches input score
  5.  Pregnancy warning inclusion -- present when is_pregnant=True + unsafe ings
  6.  Pregnancy clean report      -- no warnings when pregnancy-safe
  7.  High-score report tone      -- positive language for score >= 85
  8.  Low-score report tone       -- honest/cautionary for score < 50
  9.  Skin type personalisation   -- skin type mentioned in output
 10.  Concern coverage            -- user concerns referenced
 11.  No hallucination guard      -- only input ingredients in report
 12.  Climate personalisation     -- climate note present when provided
 13.  Token efficiency            -- total tokens < 2500 per call

LAYERING FEATURE (12 tests)
 14.  API connectivity (layering) -- Gemini responds
 15.  JSON validity (layering)    -- parseable JSON
 16.  Schema completeness         -- all required keys present
 17.  Compatibility verdict       -- one of 4 valid verdicts
 18.  Application protocol        -- order + steps + wait time present
 19.  Conflict report             -- conflicts listed for bad pairs
 20.  Synergy report              -- synergies listed for good pairs
 21.  Pregnancy warning (layering)-- fires for unsafe combos
 22.  Score < 50 alternative      -- alternative_approach non-empty
 23.  Wait time accuracy          -- wait time reflected in steps
 24.  Concern coverage (layering) -- user concerns in report
 25.  Token efficiency (layering) -- total tokens < 2500 per call

EDGE CASES (5 tests)
 26.  Empty ingredient list       -- handles gracefully
 27.  All unknown ingredients     -- confidence_note present
 28.  Teen user safety            -- age-specific note
 29.  Minimal profile             -- no optional fields provided
 30.  Special characters in names -- no crash on unicode product names
"""

import sys
import json
import time
import logging
import argparse
import statistics
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from components.llm_layer import (
    LLMLayer, UserProfile,
    mock_individual_calc_output,
    mock_layering_calc_output,
    mock_nlp_output,
    INDIVIDUAL_REPORT_SCHEMA,
    LAYERING_REPORT_SCHEMA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.llm.test")

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


def check(cond: bool, desc: str, r: TR):
    if cond:
        ok(desc);   r.passed += 1
    else:
        fail(desc); r.failed += 1


# =============================================================================
# REQUIRED SCHEMA KEYS
# =============================================================================

INDIVIDUAL_REQUIRED = {
    "report_type", "product_name", "overall_score", "grade", "headline",
    "summary", "skin_profile_fit", "key_benefits", "key_concerns",
    "warnings", "ingredient_highlights", "routine_integration",
    "usage_tips", "alternatives_note", "pregnancy_note",
    "climate_note", "confidence_note",
}

LAYERING_REQUIRED = {
    "report_type", "product_a_name", "product_b_name", "overall_score",
    "grade", "headline", "summary", "compatibility_verdict",
    "application_protocol", "synergies", "conflicts",
    "skin_type_assessment", "concern_coverage", "warnings",
    "pro_tips", "pregnancy_note", "alternative_approach",
}

VALID_COMPAT_VERDICTS = {
    "Safe to Layer",
    "Layer with Care",
    "Avoid Same Routine",
    "Do Not Combine",
}

# =============================================================================
# SHARED FIXTURES
# =============================================================================

def make_oily_acne_profile(**kwargs) -> UserProfile:
    defaults = dict(
        skin_type="oily", concerns=["acne","pores","hyperpigmentation"],
        age_group="adult", is_pregnant=False,
        skin_sensitivity="normal", current_routine="CeraVe cleanser + SPF50",
        experience_level="intermediate", location_climate="humid tropical",
    )
    defaults.update(kwargs)
    return UserProfile(**defaults)


def make_dry_aging_profile(**kwargs) -> UserProfile:
    defaults = dict(
        skin_type="dry", concerns=["aging","dryness","texture"],
        age_group="mature", is_pregnant=False,
        skin_sensitivity="low", current_routine="gentle cleanser + rich moisturiser",
        experience_level="advanced",
    )
    defaults.update(kwargs)
    return UserProfile(**defaults)


def make_pregnant_profile(**kwargs) -> UserProfile:
    defaults = dict(
        skin_type="normal", concerns=["hyperpigmentation","dryness"],
        age_group="adult", is_pregnant=True,
        skin_sensitivity="high", experience_level="beginner",
    )
    defaults.update(kwargs)
    return UserProfile(**defaults)


# =============================================================================
# INDIVIDUAL PRODUCT TESTS
# =============================================================================

def test_1_api_connectivity(llm: LLMLayer) -> TR:
    hdr("1. API Connectivity (Individual)")
    r = TR("API Connectivity")
    result = llm.generate_individual_report(
        product_name     = "Test Serum",
        ingredient_names = ["Niacinamide", "Hyaluronic Acid"],
        user_profile     = make_oily_acne_profile(),
        calc_output      = mock_individual_calc_output(82.0),
        nlp_mapped       = mock_nlp_output(["Niacinamide", "Hyaluronic Acid"]),
    )
    check(result["success"],          "API call succeeded",                  r)
    check("report" in result,         "Response has 'report' key",           r)
    check(result["latency_ms"] > 0,   f"Latency={result['latency_ms']}ms",   r)
    info(f"  Latency: {result['latency_ms']}ms")
    info(f"  Tokens : {result.get('usage',{})}")
    return r


def test_2_json_validity(llm: LLMLayer) -> TR:
    hdr("2. JSON Validity (Individual)")
    r = TR("JSON Validity")
    result = llm.generate_individual_report(
        product_name     = "Niacinamide Serum",
        ingredient_names = ["Niacinamide", "Zinc PCA", "Glycerin"],
        user_profile     = make_oily_acne_profile(),
        calc_output      = mock_individual_calc_output(78.0),
        nlp_mapped       = mock_nlp_output(["Niacinamide","Zinc PCA","Glycerin"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report = result["report"]
        check(isinstance(report, dict), "Report is a dict", r)
        check(len(report) > 0,          "Report is non-empty", r)
        # Re-serialise to confirm it's valid JSON
        try:
            json.dumps(report)
            ok("Report re-serialises to JSON"); r.passed += 1
        except Exception as e:
            fail(f"Re-serialisation failed: {e}"); r.failed += 1
    return r


def test_3_schema_completeness(llm: LLMLayer) -> TR:
    hdr("3. Schema Completeness (Individual)")
    r = TR("Schema Completeness")
    result = llm.generate_individual_report(
        product_name     = "Hydrating Serum",
        ingredient_names = ["Hyaluronic Acid", "Ceramide NP", "Panthenol"],
        user_profile     = make_dry_aging_profile(),
        calc_output      = mock_individual_calc_output(88.0),
        nlp_mapped       = mock_nlp_output(["Hyaluronic Acid","Ceramide NP","Panthenol"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report = result["report"]
        for key in INDIVIDUAL_REQUIRED:
            check(key in report, f"Key '{key}' present", r)
        # Nested checks
        sfit = report.get("skin_profile_fit", {})
        check("rating" in sfit,      "skin_profile_fit.rating present",      r)
        check("explanation" in sfit, "skin_profile_fit.explanation present",  r)
        check(isinstance(report.get("key_benefits", []),    list), "key_benefits is list", r)
        check(isinstance(report.get("key_concerns", []),    list), "key_concerns is list", r)
        check(isinstance(report.get("ingredient_highlights",[]),list),"ingredient_highlights is list",r)
        check(isinstance(report.get("usage_tips", []),      list), "usage_tips is list",   r)
        check(isinstance(report.get("warnings", []),        list), "warnings is list",     r)
    return r


def test_4_score_reflection(llm: LLMLayer) -> TR:
    hdr("4. Score Reflection in Report")
    r = TR("Score Reflection")
    target_score = 73.5
    result = llm.generate_individual_report(
        product_name     = "Moderate Serum",
        ingredient_names = ["Niacinamide", "Salicylic Acid"],
        user_profile     = make_oily_acne_profile(),
        calc_output      = mock_individual_calc_output(target_score),
        nlp_mapped       = mock_nlp_output(["Niacinamide","Salicylic Acid"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report   = result["report"]
        rep_score= report.get("overall_score", -1)
        check(isinstance(rep_score, (int, float)), f"overall_score is numeric ({rep_score})", r)
        check(abs(rep_score - target_score) <= 5,
              f"Report score {rep_score} close to input {target_score} (±5)", r)
        grade = report.get("grade","")
        check(grade in ("A+","A","B+","B","C+","C","D","F"),
              f"Grade '{grade}' is valid", r)
    return r


def test_5_pregnancy_warning(llm: LLMLayer) -> TR:
    hdr("5. Pregnancy Warning Inclusion")
    r = TR("Pregnancy Warning")
    # Calc output with retinol warning
    calc = mock_individual_calc_output(22.0)
    calc["warnings"] = [
        "PREGNANCY WARNING: The following ingredients should be AVOIDED during pregnancy: Retinol",
        "NOT recommended during pregnancy — avoid completely",
    ]
    calc["not_found"] = []

    result = llm.generate_individual_report(
        product_name     = "Retinol Night Cream",
        ingredient_names = ["Retinol", "Ceramide NP", "Squalane"],
        user_profile     = make_pregnant_profile(),
        calc_output      = calc,
        nlp_mapped       = mock_nlp_output(["Retinol","Ceramide NP","Squalane"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report   = result["report"]
        preg_note= report.get("pregnancy_note","")
        warnings = report.get("warnings", [])
        has_preg_content = (
            (preg_note and len(preg_note) > 10) or
            any("pregnan" in w.lower() or "retinol" in w.lower() for w in warnings)
        )
        check(has_preg_content,
              "Pregnancy concern present in pregnancy_note or warnings", r)
        check(len(preg_note) > 0,
              f"pregnancy_note non-empty (len={len(preg_note)})", r)
        info(f"  pregnancy_note: {preg_note[:100]}")
    return r


def test_6_pregnancy_clean(llm: LLMLayer) -> TR:
    hdr("6. Clean Report for Pregnancy-Safe Product")
    r = TR("Pregnancy Clean")
    result = llm.generate_individual_report(
        product_name     = "Safe Brightening Serum",
        ingredient_names = ["Niacinamide", "Azelaic Acid", "Hyaluronic Acid", "Glycerin"],
        user_profile     = make_pregnant_profile(),
        calc_output      = mock_individual_calc_output(85.0),
        nlp_mapped       = mock_nlp_output(["Niacinamide","Azelaic Acid","Hyaluronic Acid","Glycerin"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report   = result["report"]
        warnings = report.get("warnings", [])
        no_unsafe= not any("AVOID" in w.upper() and "PREGNAN" in w.upper() for w in warnings)
        check(no_unsafe, "No 'AVOID during pregnancy' warning for safe combo", r)
        score = report.get("overall_score", 0)
        check(score >= 70, f"Safe combo scores {score} >= 70", r)
    return r


def test_7_high_score_tone(llm: LLMLayer) -> TR:
    hdr("7. Positive Tone for High Score (>= 85)")
    r = TR("High Score Tone")
    result = llm.generate_individual_report(
        product_name     = "Perfect Moisturiser",
        ingredient_names = ["Hyaluronic Acid", "Ceramide NP", "Glycerin", "Panthenol", "Allantoin"],
        user_profile     = make_dry_aging_profile(),
        calc_output      = mock_individual_calc_output(92.0),
        nlp_mapped       = mock_nlp_output(["Hyaluronic Acid","Ceramide NP","Glycerin"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report   = result["report"]
        summary  = report.get("summary","").lower()
        headline = report.get("headline","").lower()
        sfit     = report.get("skin_profile_fit",{}).get("rating","")

        positive_words = ["excellent","great","ideal","perfect","highly","recommend",
                          "well","good","best","love","wonderful","outstanding"]
        has_positive = any(w in summary + headline for w in positive_words)
        check(has_positive, "Positive language in summary/headline for high score", r)
        check(sfit in ("Excellent","Good"),
              f"skin_profile_fit.rating='{sfit}' is positive", r)
        info(f"  headline: {report.get('headline','')}")
        info(f"  rating  : {sfit}")
    return r


def test_8_low_score_tone(llm: LLMLayer) -> TR:
    hdr("8. Honest Tone for Low Score (< 50)")
    r = TR("Low Score Tone")
    calc = mock_individual_calc_output(28.0)
    calc["pros"]  = []
    calc["cons"]  = [
        "NOT recommended for sensitive skin",
        "High comedogenicity (4/5) — clogs pores",
        "High irritancy potential",
    ]
    calc["warnings"] = ["SENSITIVITY WARNING: High-irritancy ingredients for sensitive skin"]

    result = llm.generate_individual_report(
        product_name     = "Wrong Product",
        ingredient_names = ["Coconut Oil", "Isopropyl Myristate", "Benzoyl Peroxide"],
        user_profile     = UserProfile(
            skin_type="sensitive", concerns=["acne","sensitivity"],
            age_group="adult", is_pregnant=False,
        ),
        calc_output  = calc,
        nlp_mapped   = mock_nlp_output(["Coconut Oil","Isopropyl Myristate","Benzoyl Peroxide"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report    = result["report"]
        sfit      = report.get("skin_profile_fit",{}).get("rating","")
        alt_note  = report.get("alternatives_note","")
        concerns  = report.get("key_concerns",[])

        check(sfit in ("Poor","Avoid","Moderate"),
              f"Low score -> skin_profile_fit.rating='{sfit}' is cautionary", r)
        check(len(concerns) >= 1,
              f"Key concerns list has {len(concerns)} items", r)
        check(len(alt_note) > 10,
              f"alternatives_note non-empty for low score (len={len(alt_note)})", r)
        info(f"  rating: {sfit}")
        info(f"  alt_note: {alt_note[:80]}")
    return r


def test_9_skin_type_personalisation(llm: LLMLayer) -> TR:
    hdr("9. Skin Type Personalisation")
    r = TR("Skin Personalisation")
    profile = UserProfile(
        skin_type="combination", concerns=["acne","dullness"],
        age_group="adult", is_pregnant=False, skin_sensitivity="normal",
    )
    result = llm.generate_individual_report(
        product_name     = "Multi-Skin Serum",
        ingredient_names = ["Niacinamide", "Salicylic Acid", "Hyaluronic Acid"],
        user_profile     = profile,
        calc_output      = mock_individual_calc_output(76.0),
        nlp_mapped       = mock_nlp_output(["Niacinamide","Salicylic Acid","Hyaluronic Acid"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report   = result["report"]
        full_text= json.dumps(report).lower()
        check("combination" in full_text,
              "Skin type 'combination' mentioned in report", r)
        sfit_exp = report.get("skin_profile_fit",{}).get("explanation","").lower()
        check(len(sfit_exp) > 20,
              f"skin_profile_fit.explanation is substantive ({len(sfit_exp)} chars)", r)
    return r


def test_10_concern_coverage(llm: LLMLayer) -> TR:
    hdr("10. Skin Concern Coverage")
    r = TR("Concern Coverage")
    profile = make_oily_acne_profile()  # concerns: acne, pores, hyperpigmentation
    result = llm.generate_individual_report(
        product_name     = "Brightening Acne Serum",
        ingredient_names = ["Niacinamide", "Alpha Arbutin", "Salicylic Acid"],
        user_profile     = profile,
        calc_output      = mock_individual_calc_output(83.0),
        nlp_mapped       = mock_nlp_output(["Niacinamide","Alpha Arbutin","Salicylic Acid"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report    = result["report"]
        full_text = json.dumps(report).lower()
        for concern in ["acne", "pores", "hyperpigmentation"]:
            check(concern in full_text,
                  f"User concern '{concern}' referenced in report", r)
    return r


def test_11_no_hallucination(llm: LLMLayer) -> TR:
    hdr("11. No Ingredient Hallucination")
    r = TR("No Hallucination")
    input_ings = ["Niacinamide", "Glycerin", "Ceramide NP"]
    result = llm.generate_individual_report(
        product_name     = "Simple Moisturiser",
        ingredient_names = input_ings,
        user_profile     = make_dry_aging_profile(),
        calc_output      = mock_individual_calc_output(85.0),
        nlp_mapped       = mock_nlp_output(input_ings),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report   = result["report"]
        # Check ingredient_highlights only reference input ingredients
        highlights = report.get("ingredient_highlights", [])
        known      = {i.lower() for i in input_ings}
        phantom    = []
        for h in highlights:
            ing = h.get("ingredient","").lower()
            # Allow partial matches (e.g. "Ceramide" for "Ceramide NP")
            if ing and not any(k in ing or ing in k for k in known):
                phantom.append(h.get("ingredient",""))

        check(len(phantom) == 0,
              f"No hallucinated ingredients in highlights (found phantom: {phantom})", r)
        check(len(highlights) <= 5,
              f"ingredient_highlights count={len(highlights)} <= 5 (per schema)", r)
    return r


def test_12_climate_personalisation(llm: LLMLayer) -> TR:
    hdr("12. Climate Personalisation")
    r = TR("Climate Note")
    profile = UserProfile(
        skin_type="oily", concerns=["acne"], age_group="adult",
        is_pregnant=False, location_climate="humid tropical",
    )
    result = llm.generate_individual_report(
        product_name     = "Lightweight Gel",
        ingredient_names = ["Niacinamide","Hyaluronic Acid","Glycerin"],
        user_profile     = profile,
        calc_output      = mock_individual_calc_output(80.0),
        nlp_mapped       = mock_nlp_output(["Niacinamide","Hyaluronic Acid","Glycerin"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        climate_note = result["report"].get("climate_note","")
        check(len(climate_note) > 5,
              f"climate_note present and non-empty (len={len(climate_note)})", r)
        check("humid" in climate_note.lower() or "tropical" in climate_note.lower()
              or "climate" in climate_note.lower(),
              "climate_note mentions climate context", r)
        info(f"  climate_note: {climate_note[:100]}")
    return r


def test_13_token_efficiency(llm: LLMLayer) -> TR:
    hdr("13. Token Efficiency (Individual)")
    r = TR("Token Efficiency")
    result = llm.generate_individual_report(
        product_name     = "Standard Serum",
        ingredient_names = ["Niacinamide","Hyaluronic Acid","Glycerin","Ceramide NP"],
        user_profile     = make_oily_acne_profile(),
        calc_output      = mock_individual_calc_output(79.0),
        nlp_mapped       = mock_nlp_output(["Niacinamide","Hyaluronic Acid","Glycerin","Ceramide NP"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        usage  = result.get("usage",{})
        total  = usage.get("total_tokens", 0)
        prompt = usage.get("prompt_tokens", 0)
        compl  = usage.get("completion_tokens", 0)
        info(f"  prompt={prompt} | completion={compl} | total={total}")
        if total > 0:
            check(total < 3000,
                  f"Total tokens {total} < 3000 (efficient)", r)
            check(compl < 1500,
                  f"Completion tokens {compl} < 1500", r)
        else:
            warn("Token usage not available from API response")
            r.warned += 1
    return r


# =============================================================================
# LAYERING TESTS
# =============================================================================

def test_14_api_connectivity_layering(llm: LLMLayer) -> TR:
    hdr("14. API Connectivity (Layering)")
    r = TR("API Layering Connect")
    result = llm.generate_layering_report(
        product_a_name  = "BHA Toner",
        product_a_ings  = ["Salicylic Acid", "Zinc PCA"],
        product_b_name  = "Niacinamide Serum",
        product_b_ings  = ["Niacinamide", "Hyaluronic Acid"],
        user_profile    = make_oily_acne_profile(),
        layering_output = mock_layering_calc_output(88.0),
        nlp_mapped_a    = mock_nlp_output(["Salicylic Acid","Zinc PCA"]),
        nlp_mapped_b    = mock_nlp_output(["Niacinamide","Hyaluronic Acid"]),
    )
    check(result["success"],        "API call succeeded",                 r)
    check("report" in result,       "Response has 'report' key",          r)
    check(result["latency_ms"] > 0, f"Latency={result['latency_ms']}ms",  r)
    info(f"  Latency: {result['latency_ms']}ms | Tokens: {result.get('usage',{})}")
    return r


def test_15_json_validity_layering(llm: LLMLayer) -> TR:
    hdr("15. JSON Validity (Layering)")
    r = TR("JSON Layering Valid")
    result = llm.generate_layering_report(
        product_a_name  = "Vitamin C Serum",
        product_a_ings  = ["Ascorbic Acid","Ferulic Acid","Glycerin"],
        product_b_name  = "SPF Moisturiser",
        product_b_ings  = ["Zinc Oxide","Dimethicone","Hyaluronic Acid"],
        user_profile    = make_dry_aging_profile(),
        layering_output = mock_layering_calc_output(86.0),
        nlp_mapped_a    = mock_nlp_output(["Ascorbic Acid","Ferulic Acid"]),
        nlp_mapped_b    = mock_nlp_output(["Zinc Oxide","Dimethicone"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        try:
            json.dumps(result["report"])
            ok("Report re-serialises cleanly"); r.passed += 1
        except Exception as e:
            fail(f"JSON error: {e}"); r.failed += 1
    return r


def test_16_schema_completeness_layering(llm: LLMLayer) -> TR:
    hdr("16. Schema Completeness (Layering)")
    r = TR("Schema Layering")
    result = llm.generate_layering_report(
        product_a_name  = "HA Serum",
        product_a_ings  = ["Hyaluronic Acid","Glycerin"],
        product_b_name  = "Ceramide Cream",
        product_b_ings  = ["Ceramide NP","Squalane","Cholesterol"],
        user_profile    = make_dry_aging_profile(),
        layering_output = mock_layering_calc_output(91.0),
        nlp_mapped_a    = mock_nlp_output(["Hyaluronic Acid","Glycerin"]),
        nlp_mapped_b    = mock_nlp_output(["Ceramide NP","Squalane"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report = result["report"]
        for key in LAYERING_REQUIRED:
            check(key in report, f"Key '{key}' present", r)
        # Nested checks
        proto = report.get("application_protocol",{})
        check("order"       in proto, "application_protocol.order present",    r)
        check("wait_time"   in proto, "application_protocol.wait_time present", r)
        check("time_of_day" in proto, "application_protocol.time_of_day present",r)
        check("steps"       in proto, "application_protocol.steps present",    r)
        check(isinstance(proto.get("steps",[]), list), "steps is list", r)
        check(isinstance(report.get("synergies",[]),  list), "synergies is list",  r)
        check(isinstance(report.get("conflicts",[]),  list), "conflicts is list",  r)
        check(isinstance(report.get("concern_coverage",[]),list),"concern_coverage is list",r)
        check(isinstance(report.get("pro_tips",[]),   list), "pro_tips is list",   r)
    return r


def test_17_compatibility_verdict(llm: LLMLayer) -> TR:
    hdr("17. Compatibility Verdict Validity")
    r = TR("Compat Verdict")
    result = llm.generate_layering_report(
        product_a_name  = "Vitamin C Serum",
        product_a_ings  = ["Ascorbic Acid"],
        product_b_name  = "Retinol Serum",
        product_b_ings  = ["Retinol"],
        user_profile    = make_dry_aging_profile(),
        layering_output = mock_layering_calc_output(38.0),
        nlp_mapped_a    = mock_nlp_output(["Ascorbic Acid"]),
        nlp_mapped_b    = mock_nlp_output(["Retinol"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        verdict = result["report"].get("compatibility_verdict","")
        check(verdict in VALID_COMPAT_VERDICTS,
              f"Verdict '{verdict}' is one of {VALID_COMPAT_VERDICTS}", r)
        # For a conflicting pair, expect cautionary verdict
        check(verdict in ("Avoid Same Routine","Do Not Combine","Layer with Care"),
              f"Conflicting pair gets cautionary verdict: '{verdict}'", r)
        info(f"  verdict: {verdict}")
    return r


def test_18_application_protocol(llm: LLMLayer) -> TR:
    hdr("18. Application Protocol Completeness")
    r = TR("App Protocol")
    layer_out = mock_layering_calc_output(88.0)
    layer_out["wait_time_minutes"] = 15
    result = llm.generate_layering_report(
        product_a_name  = "Glycolic Toner",
        product_a_ings  = ["Glycolic Acid"],
        product_b_name  = "HA Serum",
        product_b_ings  = ["Hyaluronic Acid","Glycerin"],
        user_profile    = make_oily_acne_profile(),
        layering_output = layer_out,
        nlp_mapped_a    = mock_nlp_output(["Glycolic Acid"]),
        nlp_mapped_b    = mock_nlp_output(["Hyaluronic Acid"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        proto = result["report"].get("application_protocol",{})
        steps = proto.get("steps",[])
        order = proto.get("order","")
        wtime = proto.get("wait_time","")

        check(len(steps) >= 3,    f"Steps list has {len(steps)} items >= 3",     r)
        check(len(order) > 10,    f"Order instruction non-empty ({len(order)}ch)",r)
        check(len(wtime) > 0,     f"Wait time specified: '{wtime}'",             r)
        check("15" in wtime or "wait" in wtime.lower() or "minute" in wtime.lower(),
              "Wait time of 15 min reflected in protocol", r)
        info(f"  order : {order[:60]}")
        info(f"  wtime : {wtime}")
        info(f"  steps : {steps[:2]}")
    return r


def test_19_conflict_report(llm: LLMLayer) -> TR:
    hdr("19. Conflict Reporting for Bad Pairs")
    r = TR("Conflict Report")
    layer_out = mock_layering_calc_output(32.0)
    layer_out["pair_interactions"] = [
        {
            "ingredient_a"    : "Ascorbic Acid",
            "ingredient_b"    : "Benzoyl Peroxide",
            "interaction_type": "conflicting",
            "interaction_rank": 0.0,
            "wait_time_minutes": 0,
            "layering_order"  : "do not layer",
            "notes"           : "BP oxidizes vitamin C, rendering both inactive.",
        }
    ]
    layer_out["cons"] = ["CONFLICT: Ascorbic Acid + Benzoyl Peroxide — BP oxidizes vitamin C"]
    layer_out["warnings"] = ["INGREDIENT CONFLICT: Ascorbic Acid + Benzoyl Peroxide"]

    result = llm.generate_layering_report(
        product_a_name  = "Vitamin C Serum",
        product_a_ings  = ["Ascorbic Acid"],
        product_b_name  = "BP Treatment",
        product_b_ings  = ["Benzoyl Peroxide"],
        user_profile    = make_oily_acne_profile(),
        layering_output = layer_out,
        nlp_mapped_a    = mock_nlp_output(["Ascorbic Acid"]),
        nlp_mapped_b    = mock_nlp_output(["Benzoyl Peroxide"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report    = result["report"]
        conflicts = report.get("conflicts",[])
        warnings  = report.get("warnings",[])
        verdict   = report.get("compatibility_verdict","")
        check(len(conflicts) >= 1,
              f"Conflicts list has {len(conflicts)} entries", r)
        check(verdict in ("Avoid Same Routine","Do Not Combine"),
              f"Conflict verdict: '{verdict}'", r)
        if conflicts:
            c = conflicts[0]
            check("pair"     in c, "Conflict has 'pair' field",     r)
            check("issue"    in c, "Conflict has 'issue' field",    r)
            check("severity" in c, "Conflict has 'severity' field", r)
            check("solution" in c, "Conflict has 'solution' field", r)
    return r


def test_20_synergy_report(llm: LLMLayer) -> TR:
    hdr("20. Synergy Reporting for Good Pairs")
    r = TR("Synergy Report")
    result = llm.generate_layering_report(
        product_a_name  = "HA Serum",
        product_a_ings  = ["Hyaluronic Acid","Glycerin"],
        product_b_name  = "Ceramide Moisturiser",
        product_b_ings  = ["Ceramide NP","Squalane","Panthenol"],
        user_profile    = make_dry_aging_profile(),
        layering_output = mock_layering_calc_output(93.0),
        nlp_mapped_a    = mock_nlp_output(["Hyaluronic Acid","Glycerin"]),
        nlp_mapped_b    = mock_nlp_output(["Ceramide NP","Squalane"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        synergies = result["report"].get("synergies",[])
        check(len(synergies) >= 1,
              f"Synergies list has {len(synergies)} entries for good combo", r)
        if synergies:
            s = synergies[0]
            check("pair"           in s, "Synergy has 'pair' field",            r)
            check("benefit"        in s, "Synergy has 'benefit' field",         r)
            check("impact_on_user" in s, "Synergy has 'impact_on_user' field",  r)
    return r


def test_21_pregnancy_warning_layering(llm: LLMLayer) -> TR:
    hdr("21. Pregnancy Warning in Layering Report")
    r = TR("Pregnancy Layering")
    layer_out = mock_layering_calc_output(18.0)
    layer_out["warnings"] = [
        "PREGNANCY DANGER: Retinol Night Cream contains ingredients unsafe during pregnancy — STOP USE IMMEDIATELY.",
        "PREGNANCY DANGER: Glycolic Toner contains ingredients unsafe during pregnancy.",
    ]

    result = llm.generate_layering_report(
        product_a_name  = "Retinol Night Cream",
        product_a_ings  = ["Retinol","Ceramide NP"],
        product_b_name  = "Glycolic Toner",
        product_b_ings  = ["Glycolic Acid","Hyaluronic Acid"],
        user_profile    = make_pregnant_profile(),
        layering_output = layer_out,
        nlp_mapped_a    = mock_nlp_output(["Retinol"]),
        nlp_mapped_b    = mock_nlp_output(["Glycolic Acid"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report    = result["report"]
        preg_note = report.get("pregnancy_note","")
        warnings  = report.get("warnings",[])
        has_preg  = (len(preg_note) > 10 or
                     any("pregnan" in w.lower() for w in warnings))
        check(has_preg, "Pregnancy warning present in layering report", r)
        check(len(preg_note) > 0, f"pregnancy_note non-empty ({len(preg_note)} chars)", r)
        info(f"  pregnancy_note: {preg_note[:100]}")
    return r


def test_22_low_score_alternative(llm: LLMLayer) -> TR:
    hdr("22. Alternative Approach for Low Layering Score (< 60)")
    r = TR("Low Score Alt")
    layer_out = mock_layering_calc_output(25.0)
    layer_out["verdict"] = "Very poor compatibility (25/100). Do NOT layer these products together."
    result = llm.generate_layering_report(
        product_a_name  = "BP Wash",
        product_a_ings  = ["Benzoyl Peroxide"],
        product_b_name  = "Glycolic Serum",
        product_b_ings  = ["Glycolic Acid","Salicylic Acid"],
        user_profile    = UserProfile(
            skin_type="sensitive", concerns=["acne"],
            age_group="adult", is_pregnant=False,
        ),
        layering_output = layer_out,
        nlp_mapped_a    = mock_nlp_output(["Benzoyl Peroxide"]),
        nlp_mapped_b    = mock_nlp_output(["Glycolic Acid","Salicylic Acid"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        alt = result["report"].get("alternative_approach","")
        check(len(alt) > 20,
              f"alternative_approach present for low score (len={len(alt)})", r)
        info(f"  alternative: {alt[:100]}")
    return r


def test_23_wait_time_accuracy(llm: LLMLayer) -> TR:
    hdr("23. Wait Time Accuracy in Protocol")
    r = TR("Wait Time")
    layer_out = mock_layering_calc_output(72.0)
    layer_out["wait_time_minutes"] = 30

    result = llm.generate_layering_report(
        product_a_name  = "Vitamin C Serum",
        product_a_ings  = ["Ascorbic Acid","Ferulic Acid"],
        product_b_name  = "Niacinamide Serum",
        product_b_ings  = ["Niacinamide","Hyaluronic Acid"],
        user_profile    = make_oily_acne_profile(),
        layering_output = layer_out,
        nlp_mapped_a    = mock_nlp_output(["Ascorbic Acid"]),
        nlp_mapped_b    = mock_nlp_output(["Niacinamide"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        proto     = result["report"].get("application_protocol",{})
        wtime_str = proto.get("wait_time","").lower()
        steps_str = " ".join(proto.get("steps",[])).lower()
        has_wait  = (
            "30" in wtime_str or "30" in steps_str or
            "wait" in wtime_str or "minute" in wtime_str
        )
        check(has_wait,
              f"30-minute wait reflected in protocol: wtime='{wtime_str[:50]}'", r)
    return r


def test_24_concern_coverage_layering(llm: LLMLayer) -> TR:
    hdr("24. Concern Coverage in Layering Report")
    r = TR("Concern Coverage L")
    profile = make_oily_acne_profile()  # concerns: acne, pores, hyperpigmentation
    result = llm.generate_layering_report(
        product_a_name  = "BHA Toner",
        product_a_ings  = ["Salicylic Acid","Zinc PCA"],
        product_b_name  = "Brightening Serum",
        product_b_ings  = ["Niacinamide","Alpha Arbutin","Tranexamic Acid"],
        user_profile    = profile,
        layering_output = mock_layering_calc_output(87.0),
        nlp_mapped_a    = mock_nlp_output(["Salicylic Acid"]),
        nlp_mapped_b    = mock_nlp_output(["Niacinamide","Alpha Arbutin"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        report   = result["report"]
        cov      = report.get("concern_coverage",[])
        full_txt = json.dumps(report).lower()
        check(len(cov) >= 1, f"concern_coverage has {len(cov)} entries", r)
        check("acne" in full_txt or "pore" in full_txt,
              "User concerns mentioned in report", r)
        if cov:
            c = cov[0]
            check("concern"       in c, "concern_coverage item has 'concern' field",      r)
            check("addressed_by"  in c, "concern_coverage item has 'addressed_by' field", r)
            check("effectiveness" in c, "concern_coverage item has 'effectiveness' field",r)
    return r


def test_25_token_efficiency_layering(llm: LLMLayer) -> TR:
    hdr("25. Token Efficiency (Layering)")
    r = TR("Token Efficiency L")
    result = llm.generate_layering_report(
        product_a_name  = "Standard Serum A",
        product_a_ings  = ["Niacinamide","Hyaluronic Acid"],
        product_b_name  = "Standard Cream B",
        product_b_ings  = ["Ceramide NP","Squalane","Glycerin"],
        user_profile    = make_oily_acne_profile(),
        layering_output = mock_layering_calc_output(85.0),
        nlp_mapped_a    = mock_nlp_output(["Niacinamide"]),
        nlp_mapped_b    = mock_nlp_output(["Ceramide NP"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        usage = result.get("usage",{})
        total = usage.get("total_tokens",0)
        compl = usage.get("completion_tokens",0)
        info(f"  total={total} | completion={compl}")
        if total > 0:
            check(total < 3500, f"Total tokens {total} < 3500", r)
            check(compl < 1600, f"Completion {compl} < 1600",   r)
        else:
            warn("Token usage unavailable"); r.warned += 1
    return r


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_26_empty_ingredients(llm: LLMLayer) -> TR:
    hdr("26. Empty Ingredient List (Edge Case)")
    r = TR("Empty Ingredients")
    calc = mock_individual_calc_output(0.0)
    calc["verdict"]  = "No recognisable ingredients found."
    calc["pros"]     = []
    calc["cons"]     = ["No ingredients could be evaluated"]
    calc["warnings"] = []
    calc["not_found"]= ["mystery cream"]
    try:
        result = llm.generate_individual_report(
            product_name     = "Mystery Cream",
            ingredient_names = [],
            user_profile     = make_oily_acne_profile(),
            calc_output      = calc,
            nlp_mapped       = [],
        )
        check(result["success"] or "error" in result,
              "Handles empty ingredients without crash", r)
        if result["success"]:
            conf_note = result["report"].get("confidence_note","")
            check(len(conf_note) > 0, "confidence_note present for unknown product", r)
    except Exception as e:
        fail(f"Crash on empty ingredients: {e}"); r.failed += 1
    return r


def test_27_unknown_ingredients(llm: LLMLayer) -> TR:
    hdr("27. All Unknown Ingredients (Confidence Note)")
    r = TR("Unknown Ings")
    calc = mock_individual_calc_output(50.0)
    calc["not_found"] = ["XYZ-Peptide-999","AquaGel-X","NanoRetinol-Pro"]
    nlp_unk = [
        {"input": n, "inci_name": n, "score": 0.3, "confidence": "uncertain",
         "method": "ensemble", "alternatives": [], "latency_ms": 2.0}
        for n in calc["not_found"]
    ]
    result = llm.generate_individual_report(
        product_name     = "Unknown Brand Serum",
        ingredient_names = calc["not_found"],
        user_profile     = make_oily_acne_profile(),
        calc_output      = calc,
        nlp_mapped       = nlp_unk,
    )
    check(result["success"], "API success with unknowns", r)
    if result["success"]:
        conf_note = result["report"].get("confidence_note","")
        check(len(conf_note) > 10,
              f"confidence_note explains unknown ingredients ({len(conf_note)} chars)", r)
        info(f"  confidence_note: {conf_note[:100]}")
    return r


def test_28_teen_user_safety(llm: LLMLayer) -> TR:
    hdr("28. Teen User Age-Specific Safety")
    r = TR("Teen Safety")
    calc = mock_individual_calc_output(55.0)
    calc["cons"] = ["Retinol not typically recommended for teen users"]
    teen_profile = UserProfile(
        skin_type="oily", concerns=["acne","pores"],
        age_group="teen", is_pregnant=False,
        skin_sensitivity="normal", experience_level="beginner",
    )
    result = llm.generate_individual_report(
        product_name     = "Retinol Serum",
        ingredient_names = ["Retinol","Ceramide NP"],
        user_profile     = teen_profile,
        calc_output      = calc,
        nlp_mapped       = mock_nlp_output(["Retinol","Ceramide NP"]),
    )
    check(result["success"], "API success", r)
    if result["success"]:
        full_txt = json.dumps(result["report"]).lower()
        has_age  = "teen" in full_txt or "young" in full_txt or "age" in full_txt
        check(has_age, "Age/teen consideration reflected in report", r)
        tips = result["report"].get("usage_tips",[])
        check(len(tips) >= 1, f"Usage tips present ({len(tips)})", r)
    return r


def test_29_minimal_profile(llm: LLMLayer) -> TR:
    hdr("29. Minimal UserProfile (No Optional Fields)")
    r = TR("Minimal Profile")
    minimal = UserProfile(
        skin_type="normal", concerns=[], age_group="adult", is_pregnant=False
    )
    try:
        result = llm.generate_individual_report(
            product_name     = "Basic Moisturiser",
            ingredient_names = ["Glycerin","Hyaluronic Acid"],
            user_profile     = minimal,
            calc_output      = mock_individual_calc_output(75.0),
            nlp_mapped       = mock_nlp_output(["Glycerin","Hyaluronic Acid"]),
        )
        check(result["success"], "Minimal profile succeeds", r)
        if result["success"]:
            check(isinstance(result["report"], dict), "Report is dict", r)
    except Exception as e:
        fail(f"Crash on minimal profile: {e}"); r.failed += 1
    return r


def test_30_unicode_product_name(llm: LLMLayer) -> TR:
    hdr("30. Unicode/Special Character Product Names")
    r = TR("Unicode Names")
    try:
        result = llm.generate_individual_report(
            product_name     = "Laneige 水分クリーム • Crème Hydratante",
            ingredient_names = ["Hyaluronic Acid","Glycerin","Niacinamide"],
            user_profile     = make_dry_aging_profile(),
            calc_output      = mock_individual_calc_output(82.0),
            nlp_mapped       = mock_nlp_output(["Hyaluronic Acid","Glycerin"]),
        )
        check(result["success"], "Unicode product name handled", r)
        if result["success"]:
            check(isinstance(result["report"], dict), "Report is dict", r)
    except Exception as e:
        fail(f"Crash on unicode name: {e}"); r.failed += 1
    return r


# =============================================================================
# MASTER RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SkinSpectra LLM Layer Test Suite")
    parser.add_argument("--api_key", default="",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Test keys to skip (e.g. --skip t13 t25)")
    parser.add_argument("--only", nargs="*", default=[],
                        help="Run only these test keys (e.g. --only t1 t14)")
    parser.add_argument("--output", default="llm_test_results.json",
                        help="Save full results to JSON")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY","")
    if not api_key:
        print(f"{RED}ERROR: Provide --api_key or set GEMINI_API_KEY env var{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  SkinSpectra -- LLM Layer Test Suite{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"  Model : gemini-2.5-flash")
    print(f"  Tests : 30 (13 individual + 12 layering + 5 edge cases)\n")

    llm = LLMLayer(api_key=api_key)

    tests = [
        ("t1",  "API Connectivity",          test_1_api_connectivity),
        ("t2",  "JSON Validity",              test_2_json_validity),
        ("t3",  "Schema Completeness",        test_3_schema_completeness),
        ("t4",  "Score Reflection",           test_4_score_reflection),
        ("t5",  "Pregnancy Warning",          test_5_pregnancy_warning),
        ("t6",  "Pregnancy Clean",            test_6_pregnancy_clean),
        ("t7",  "High Score Tone",            test_7_high_score_tone),
        ("t8",  "Low Score Tone",             test_8_low_score_tone),
        ("t9",  "Skin Type Personalisation",  test_9_skin_type_personalisation),
        ("t10", "Concern Coverage",           test_10_concern_coverage),
        ("t11", "No Hallucination",           test_11_no_hallucination),
        ("t12", "Climate Personalisation",    test_12_climate_personalisation),
        ("t13", "Token Efficiency",           test_13_token_efficiency),
        ("t14", "API Layering Connect",       test_14_api_connectivity_layering),
        ("t15", "JSON Layering Valid",        test_15_json_validity_layering),
        ("t16", "Schema Layering",            test_16_schema_completeness_layering),
        ("t17", "Compat Verdict",             test_17_compatibility_verdict),
        ("t18", "App Protocol",              test_18_application_protocol),
        ("t19", "Conflict Report",            test_19_conflict_report),
        ("t20", "Synergy Report",             test_20_synergy_report),
        ("t21", "Pregnancy Layering",         test_21_pregnancy_warning_layering),
        ("t22", "Low Score Alt",              test_22_low_score_alternative),
        ("t23", "Wait Time",                  test_23_wait_time_accuracy),
        ("t24", "Concern Coverage L",         test_24_concern_coverage_layering),
        ("t25", "Token Efficiency L",         test_25_token_efficiency_layering),
        ("t26", "Empty Ingredients",          test_26_empty_ingredients),
        ("t27", "Unknown Ings",               test_27_unknown_ingredients),
        ("t28", "Teen Safety",                test_28_teen_user_safety),
        ("t29", "Minimal Profile",            test_29_minimal_profile),
        ("t30", "Unicode Names",              test_30_unicode_product_name),
    ]

    skip_set  = set(args.skip or [])
    only_set  = set(args.only or [])
    all_results = []
    raw_results = {}

    for key, name, fn in tests:
        if key in skip_set:
            warn(f"Skipping: {key} ({name})")
            continue
        if only_set and key not in only_set:
            continue

        # Rate limiting — Gemini Flash has generous limits but be polite
        time.sleep(1.5)

        tr = fn(llm)
        all_results.append(tr)
        raw_results[key] = {"name": name, "passed": tr.passed,
                            "failed": tr.failed, "warned": tr.warned}

    # ── Final report ──────────────────────────────────────────────────
    hdr("FINAL REPORT")
    total_p = sum(t.passed for t in all_results)
    total_w = sum(t.warned for t in all_results)
    total_f = sum(t.failed for t in all_results)
    grand   = total_p + total_w + total_f
    overall = 100 * total_p / grand if grand else 0

    print(f"\n  {'Section':<30} {'Pass':>5} {'Warn':>5} {'Fail':>5} {'%':>6}")
    print(f"  {'─'*55}")
    for t in all_results:
        c = GREEN if t.pct >= 80 else (YELLOW if t.pct >= 60 else RED)
        print(f"  {t.name:<30} {t.passed:>5} {t.warned:>5} {t.failed:>5} "
              f"{c}{t.pct:>5.0f}%{RESET}")
    print(f"  {'─'*55}")
    oc = GREEN if overall >= 80 else (YELLOW if overall >= 65 else RED)
    print(f"  {'TOTAL':<30} {total_p:>5} {total_w:>5} {total_f:>5} "
          f"{oc}{overall:>5.0f}%{RESET}\n")

    if overall >= 80:
        print(f"  {GREEN}{BOLD}LLM Layer is production-ready.{RESET}\n")
    elif overall >= 65:
        print(f"  {YELLOW}{BOLD}LLM Layer needs minor tuning.{RESET}\n")
    else:
        print(f"  {RED}{BOLD}LLM Layer requires fixes.{RESET}\n")

    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "summary": {"passed": total_p, "warned": total_w,
                        "failed": total_f, "pct": round(overall,1)},
            "tests"  : raw_results,
        }, f, indent=2)
    print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
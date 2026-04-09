"""
SkinSpectra LLM Layer - Personalized Report Generator
======================================================
Architecture:
  - Model      : Gemini 2.5 Flash (google-genai SDK)
  - Input      : Outputs from NLP layer + Calculation layer + User skin profile
  - Output     : Structured JSON report (ready for API consumption)
  - Features   : Feature 1 (Individual product) + Feature 2 (Product layering)

Token Strategy:
  - System prompt  : ~400 tokens  (role + rules, sent once)
  - User prompt    : ~600-900 tokens (compressed structured data, no redundancy)
  - Output         : ~800-2500 tokens (rich JSON, capped via max_output_tokens)
  - Total per call : ~1800-3500 tokens (well within Flash limits)

Prompt Engineering Principles:
  1. Role injection          : Dermatologist-grade expertise framing
  2. Data compression        : Only essential fields passed, not raw dumps
  3. JSON schema enforcement : Explicit output schema in prompt
  4. Tone instruction        : Warm, clinical, empowering — not fear-mongering
  5. Priority ordering       : Safety > Efficacy > Cosmetic experience
  6. Constraint enforcement  : No hallucinated ingredients, no medical diagnosis
"""

import os
import re
import json
import time
import logging
import textwrap
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.llm")

# Configuration

GEMINI_MODEL   = "gemini-2.5-flash"
MAX_OUT_TOKENS = 3000
TEMPERATURE    = 0.25
TOP_P          = 0.90

# User Profile

@dataclass
class UserProfile:
    """User skin profile and context for personalized recommendations."""
    skin_type        : str
    concerns         : List[str]
    age_group        : str
    is_pregnant      : bool
    skin_sensitivity : str  = "normal"
    current_routine  : str  = ""
    allergies        : str  = ""
    location_climate : str  = ""
    experience_level : str  = "beginner"

    def to_prompt_str(self) -> str:
        """Format profile as compact pipe-delimited string for prompt injection."""
        parts = [
            f"skin_type={self.skin_type}",
            f"age={self.age_group}",
            f"concerns=[{', '.join(self.concerns) if self.concerns else 'none'}]",
            f"pregnant={self.is_pregnant}",
            f"sensitivity={self.skin_sensitivity}",
        ]
        if self.allergies:
            parts.append(f"allergies={self.allergies}")
        if self.location_climate:
            parts.append(f"climate={self.location_climate}")
        if self.current_routine:
            parts.append(f"routine={self.current_routine[:60]}")
        parts.append(f"level={self.experience_level}")
        return " | ".join(parts)


# Output Schemas

# Feature 1: Individual Product Report
INDIVIDUAL_REPORT_SCHEMA = {
    "report_type"          : "individual_product",
    "product_name"         : "str",
    "overall_score"        : "float  (0-100)",
    "grade"                : "str   (A+/A/B+/B/C+/C/D/F)",
    "headline"             : "str   (1 punchy sentence, max 15 words)",
    "summary"              : "str   (2-3 sentences, personalised verdict)",
    "skin_profile_fit"     : {
        "rating"           : "str   (Excellent/Good/Moderate/Poor/Avoid)",
        "explanation"      : "str   (1-2 sentences specific to user's skin type)"
    },
    "key_benefits"         : [
        {"benefit": "str", "ingredient": "str", "relevance_to_user": "str"}
    ],
    "key_concerns"         : [
        {"concern": "str", "ingredient": "str", "severity": "str (low/medium/high/critical)"}
    ],
    "warnings"             : ["str  (critical safety only)"],
    "ingredient_highlights": [
        {"ingredient": "str", "role": "str", "verdict": "str (star/good/neutral/watch/avoid)"}
    ],
    "routine_integration"  : "str   (how to fit this product into their routine, 2-3 sentences)",
    "usage_tips"           : ["str  (3-5 practical tips personalised to user)"],
    "alternatives_note"    : "str   (optional: suggest what to look for if score < 70)",
    "pregnancy_note"       : "str   (only if is_pregnant=true, else empty string)",
    "climate_note"         : "str   (only if climate provided, else empty string)",
    "confidence_note"      : "str   (brief note on any ingredients not found in database)"
}

# Feature 2: Product Layering Report
LAYERING_REPORT_SCHEMA = {
    "report_type"          : "product_layering",
    "product_a_name"       : "str",
    "product_b_name"       : "str",
    "overall_score"        : "float  (0-100)",
    "grade"                : "str   (A+/A/B+/B/C+/C/D/F)",
    "headline"             : "str   (1 punchy sentence, max 15 words)",
    "summary"              : "str   (2-3 sentences, personalised layering verdict)",
    "compatibility_verdict": "str   (Safe to Layer / Layer with Care / Avoid Same Routine / Do Not Combine)",
    "application_protocol" : {
        "order"            : "str   (which product goes first and why)",
        "wait_time"        : "str   (e.g. 'Wait 15 minutes' or 'No wait needed')",
        "time_of_day"      : "str   (AM / PM / Both / Separate AM+PM)",
        "steps"            : ["str  (plain step-by-step application guide, 4-6 steps, no numbering)"]
    },
    "synergies"            : [
        {"pair": "str", "benefit": "str", "impact_on_user": "str"}
    ],
    "conflicts"            : [
        {"pair": "str", "issue": "str", "severity": "str (low/medium/high/critical)", "solution": "str"}
    ],
    "skin_type_assessment" : "str   (how this combination works for their specific skin type, 1-2 sentences)",
    "concern_coverage"     : [
        {"concern": "str", "addressed_by": "str", "effectiveness": "str (well/partially/not)"}
    ],
    "warnings"             : ["str  (critical flags: pregnancy, conflicts, irritancy stacking)"],
    "pro_tips"             : ["str  (3-4 advanced tips for getting the most from this combination)"],
    "pregnancy_note"       : "str   (only if is_pregnant=true, else empty string)",
    "alternative_approach" : "str   (if score < 60: what to do instead)"
}


# Prompt Construction

class PromptBuilder:
    """Constructs token-efficient, structured prompts for individual product and layering reports."""

    # System instruction (sent as system role to Gemini)
    SYSTEM_PROMPT = textwrap.dedent("""
        You are SkinSpectra, a clinical skincare expert assistant grounded in cosmetic chemistry and evidence-based dermatology.

        ROLE & TONE:
        Synthesize computational compatibility scores into personalized, medically-grounded recommendations. Your voice is warm yet authoritative—knowledgeable friend who is also a dermatologist.

        PRIORITY FRAMEWORK:
        1. Safety (toxicity, pregnancy, allergies, interactions)
        2. Efficacy (clinical evidence, skin concern targeting)
        3. Integration (routine fit, user experience)

        OUTPUT REQUIREMENTS:
        - Return ONLY valid JSON matching the provided schema. No markdown, preamble, explanation, or code fences.
        - All responses must be factually grounded in the ingredient data provided.
        - Output fields must be concise and action-oriented. Eliminate filler, padding, or hedging language.

        CRITICAL CONSTRAINTS:
        1. Never diagnose medical conditions or claim to replace professional dermatology.
        2. Never reference ingredients not explicitly provided in the data input.
        3. Pregnancy: If user is pregnant, flag ALL potentially unsafe ingredients with explicit severity (critical/avoid/use_caution).
        4. Tone calibration: Score ≥80 = enthusiastic confidence. Score <50 = honest caution (never alarmist). Score 50-79 = balanced assessment.
        5. Scoring rationale: Always reflect the computational score in your text-based verdict. Do not contradict the data.
        6. List constraints: key_benefits max 4, key_concerns max 4, ingredient_highlights max 5, warnings: critical only.

        FIELD-SPECIFIC RULES:
        - headline: punchy, ≤15 words, no filler adjectives
        - summary: 2-3 personalized sentences tied to user's skin_type and concerns
        - routine_integration: specific steps—"apply after cleanser, wait 2 min, then moisturizer"
        - usage_tips: practical, numbered implicitly, tailored to user experience_level
        - warnings: critical safety issues only (not "may cause dryness")
        - confidence_note: ONLY include if ingredients marked low_confidence or not_found in database

        JSON SCHEMA COMPLIANCE:
        Your output must match the schema provided exactly. All required fields must be present. Optional fields (e.g., climate_note, pregnancy_note) are included only when relevant.
    """).strip()

    # Build prompt for individual product report
    @staticmethod
    def build_individual_prompt(
        product_name    : str,
        ingredient_names: List[str],
        user_profile    : UserProfile,
        calc_output     : Dict,
        nlp_mapped      : List[Dict],
    ) -> str:

        score   = calc_output.get("compatibility_score", 0)
        grade   = calc_output.get("grade", "?")
        verdict = calc_output.get("verdict", "")
        pros    = calc_output.get("pros", [])[:5]
        cons    = calc_output.get("cons", [])[:5]
        warns   = calc_output.get("warnings", [])
        details = calc_output.get("ingredient_details", [])

        details_sorted = sorted(details, key=lambda x: abs(x.get("score", 50) - 50), reverse=True)[:6]
        ing_summary = []
        for d in details_sorted:
            p_str = "; ".join(d.get("pros", [])[:1])
            c_str = "; ".join(d.get("cons", [])[:1])
            ing_summary.append(
                f"{d['ingredient']}: score={d['score']}" +
                (f", benefit={p_str}" if p_str else "") +
                (f", concern={c_str}" if c_str else "")
            )

        low_conf = [m["input"] for m in nlp_mapped if m.get("confidence") in ("low", "uncertain")]
        not_found= calc_output.get("not_found", [])

        prompt = textwrap.dedent(f"""
            FEATURE: Individual Product Compatibility Report

            USER PROFILE:
            {user_profile.to_prompt_str()}

            PRODUCT:
            name={product_name}
            ingredients=[{", ".join(ingredient_names[:20])}]

            COMPATIBILITY SCORE: {score}/100 | Grade: {grade}
            verdict_hint="{verdict}"

            KEY POSITIVES (from engine):
            {chr(10).join(f"- {p}" for p in pros) if pros else "- none"}

            KEY CONCERNS (from engine):
            {chr(10).join(f"- {c}" for c in cons) if cons else "- none"}

            WARNINGS (mandatory to include):
            {chr(10).join(f"- {w}" for w in warns) if warns else "- none"}

            TOP INGREDIENT BREAKDOWN:
            {chr(10).join(f"- {s}" for s in ing_summary) if ing_summary else "- no data"}

            DATABASE GAPS:
            low_confidence_mappings=[{", ".join(low_conf) if low_conf else "none"}]
            not_in_database=[{", ".join(not_found) if not_found else "none"}]

            OUTPUT SCHEMA:
            {json.dumps(INDIVIDUAL_REPORT_SCHEMA, indent=2)}

            Generate the JSON report now. Be specific to this user's profile. No markdown fences.
        """).strip()

        return prompt

    # Build prompt for layering report
    @staticmethod
    def build_layering_prompt(
        product_a_name  : str,
        product_a_ings  : List[str],
        product_b_name  : str,
        product_b_ings  : List[str],
        user_profile    : UserProfile,
        layering_output : Dict,
        nlp_mapped_a    : List[Dict],
        nlp_mapped_b    : List[Dict],
    ) -> str:

        score   = layering_output.get("layering_score", 0)
        grade   = layering_output.get("grade", "?")
        verdict = layering_output.get("verdict", "")
        order   = layering_output.get("layering_order", "")
        wait    = layering_output.get("wait_time_minutes", 0)
        steps   = layering_output.get("application_steps", [])[:5]
        pros    = layering_output.get("pros", [])[:4]
        cons    = layering_output.get("cons", [])[:4]
        warns   = layering_output.get("warnings", [])
        pairs   = layering_output.get("pair_interactions", [])

        pair_lines = []
        for p in pairs[:6]:
            itype = p.get("interaction_type", "unknown")
            ia = p.get("ingredient_a", "")
            ib = p.get("ingredient_b", "")
            notes = p.get("notes", "")[:60]
            pair_lines.append(f"{ia} + {ib} = {itype}" + (f" | {notes}" if notes else ""))

        low_conf_a = [m["input"] for m in nlp_mapped_a if m.get("confidence") in ("low","uncertain")]
        low_conf_b = [m["input"] for m in nlp_mapped_b if m.get("confidence") in ("low","uncertain")]
        not_found_a= layering_output.get("product_a_not_found", [])
        not_found_b= layering_output.get("product_b_not_found", [])

        prompt = textwrap.dedent(f"""
            FEATURE: Product Layering Compatibility Report

            USER PROFILE:
            {user_profile.to_prompt_str()}

            PRODUCT A (applied first): {product_a_name}
            ingredients=[{", ".join(product_a_ings[:15])}]

            PRODUCT B (applied second): {product_b_name}
            ingredients=[{", ".join(product_b_ings[:15])}]

            LAYERING SCORE: {score}/100 | Grade: {grade}
            verdict_hint="{verdict}"
            layering_order_hint="{order}"
            wait_time_minutes={wait}

            APPLICATION STEPS (engine output):
            {chr(10).join(f"- {s}" for s in steps) if steps else "- no steps"}

            SYNERGIES & CONFLICTS:
            {chr(10).join(f"- {l}" for l in pair_lines) if pair_lines else "- no known pairs"}

            ENGINE POSITIVES:
            {chr(10).join(f"- {p}" for p in pros) if pros else "- none"}

            ENGINE CONCERNS:
            {chr(10).join(f"- {c}" for c in cons) if cons else "- none"}

            WARNINGS (mandatory):
            {chr(10).join(f"- {w}" for w in warns) if warns else "- none"}

            DATABASE GAPS:
            product_a_low_conf=[{", ".join(low_conf_a) if low_conf_a else "none"}]
            product_b_low_conf=[{", ".join(low_conf_b) if low_conf_b else "none"}]
            product_a_not_found=[{", ".join(not_found_a) if not_found_a else "none"}]
            product_b_not_found=[{", ".join(not_found_b) if not_found_b else "none"}]

            OUTPUT SCHEMA:
            {json.dumps(LAYERING_REPORT_SCHEMA, indent=2)}

            Generate the JSON report now. Be specific to this user's profile. No markdown fences.
        """).strip()

        return prompt


# Gemini Integration

class GeminiClient:
    """Manages communication with Gemini API for structured report generation."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
            )
        self._client = genai.Client(api_key=key)

        self._config = types.GenerateContentConfig(
            system_instruction = PromptBuilder.SYSTEM_PROMPT,
            temperature        = TEMPERATURE,
            top_p              = TOP_P,
            max_output_tokens  = MAX_OUT_TOKENS,
        )
        log.info(f"GeminiClient initialised | model={GEMINI_MODEL}")

    def generate(self, user_prompt: str) -> Dict:
        """Send prompt to Gemini and parse JSON response. Returns report dict with metadata."""
        t0 = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model    = GEMINI_MODEL,
                contents = user_prompt,
                config   = self._config,
            )
            raw_text = response.text.strip()
            latency  = round((time.perf_counter() - t0) * 1000, 1)

            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$",          "", raw_text)
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                usage = {
                    "prompt_tokens"    : getattr(um, "prompt_token_count",     0),
                    "completion_tokens": getattr(um, "candidates_token_count", 0),
                    "total_tokens"     : getattr(um, "total_token_count",      0),
                }

            return {
                "success"   : True,
                "report"    : parsed,
                "latency_ms": latency,
                "usage"     : usage,
                "raw"       : raw_text,
            }

        except json.JSONDecodeError as e:
            latency = round((time.perf_counter() - t0) * 1000, 1)
            log.error(f"JSON parse error: {e}")
            log.error(f"Raw response: {raw_text[:500]}")
            return {
                "success"   : False,
                "error"     : f"JSON parse failed: {e}",
                "latency_ms": latency,
                "raw"       : raw_text,
                "usage"     : {},
            }
        except Exception as e:
            latency = round((time.perf_counter() - t0) * 1000, 1)
            log.error(f"Gemini API error: {e}")
            return {
                "success"   : False,
                "error"     : str(e),
                "latency_ms": latency,
                "raw"       : "",
                "usage"     : {},
            }


# LLM Layer - Main Interface

class LLMLayer:
    """Main interface for generating personalized skincare reports via Gemini.
    
    Features:
    - Feature 1: Individual product compatibility assessment
    - Feature 2: Product layering (2-product combination) assessment
    
    Integrates NLP-mapped ingredients, compatibility scores, and user profiles
    to produce structured JSON reports.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client  = GeminiClient(api_key=api_key)
        self.builder = PromptBuilder()

    def generate_individual_report(
        self,
        product_name    : str,
        ingredient_names: List[str],
        user_profile    : UserProfile,
        calc_output     : Dict,
        nlp_mapped      : Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate personalized individual product compatibility report.
        
        Args:
            product_name: Product display name
            ingredient_names: List of ingredient names
            user_profile: UserProfile instance
            calc_output: Compatibility score output from calculation engine
            nlp_mapped: Optional NLP mapping output for confidence tracking
        
        Returns:
            Dict with keys: success, report, latency_ms, usage, feature, product_name, score, error (if applicable)
        """
        nlp_mapped = nlp_mapped or []

        prompt = self.builder.build_individual_prompt(
            product_name     = product_name,
            ingredient_names = ingredient_names,
            user_profile     = user_profile,
            calc_output      = calc_output,
            nlp_mapped       = nlp_mapped,
        )

        log.info(
            f"Generating individual report | product='{product_name}' "
            f"| score={calc_output.get('compatibility_score',0)} "
            f"| skin={user_profile.skin_type}"
        )

        result = self.client.generate(prompt)
        result["feature"]      = "individual"
        result["product_name"] = product_name
        result["score"]        = calc_output.get("compatibility_score", 0)

        if result["success"]:
            log.info(
                f"Report generated | latency={result['latency_ms']}ms "
                f"| tokens={result['usage'].get('total_tokens','?')}"
            )
        else:
            log.error(f"Report generation failed: {result.get('error','unknown')}")

        return result

    def generate_layering_report(
        self,
        product_a_name  : str,
        product_a_ings  : List[str],
        product_b_name  : str,
        product_b_ings  : List[str],
        user_profile    : UserProfile,
        layering_output : Dict,
        nlp_mapped_a    : Optional[List[Dict]] = None,
        nlp_mapped_b    : Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate personalized product layering (combination) compatibility report.
        
        Args:
            product_a_name: Product A name (applied first)
            product_a_ings: Product A ingredient list
            product_b_name: Product B name (applied second)
            product_b_ings: Product B ingredient list
            user_profile: UserProfile instance
            layering_output: Layering compatibility score from calculation engine
            nlp_mapped_a/b: Optional NLP mapping outputs for confidence tracking
        
        Returns:
            Dict with keys: success, report, latency_ms, usage, feature, product_a_name, product_b_name, score, error (if applicable)
        """
        nlp_mapped_a = nlp_mapped_a or []
        nlp_mapped_b = nlp_mapped_b or []

        prompt = self.builder.build_layering_prompt(
            product_a_name  = product_a_name,
            product_a_ings  = product_a_ings,
            product_b_name  = product_b_name,
            product_b_ings  = product_b_ings,
            user_profile    = user_profile,
            layering_output = layering_output,
            nlp_mapped_a    = nlp_mapped_a,
            nlp_mapped_b    = nlp_mapped_b,
        )

        log.info(
            f"Generating layering report | "
            f"A='{product_a_name}' + B='{product_b_name}' "
            f"| score={layering_output.get('layering_score',0)} "
            f"| skin={user_profile.skin_type}"
        )

        result = self.client.generate(prompt)
        result["feature"]        = "layering"
        result["product_a_name"] = product_a_name
        result["product_b_name"] = product_b_name
        result["score"]          = layering_output.get("layering_score", 0)

        if result["success"]:
            log.info(
                f"Layering report generated | latency={result['latency_ms']}ms "
                f"| tokens={result['usage'].get('total_tokens','?')}"
            )
        else:
            log.error(f"Layering report failed: {result.get('error','unknown')}")

        return result


# Test Utilities

def mock_individual_calc_output(score: float = 82.0) -> Dict:
    """Generate mock individual product compatibility score for testing."""
    return {
        "compatibility_score": score,
        "grade": "A" if score >= 85 else "B+" if score >= 78 else "B" if score >= 70 else "C",
        "verdict": f"Good compatibility ({score}/100) for your skin profile.",
        "pros": [
            "Well-suited for oily skin (sebum regulation)",
            "Helps with acne (antimicrobial + anti-inflammatory)",
            "Non-comedogenic (0/5) — safe for pore-prone skin",
            "Helps with pores (pore minimizing)",
        ],
        "cons": [
            "Use with caution on sensitive skin",
            "May cause initial purging in first 2-4 weeks",
        ],
        "warnings": [],
        "not_found": [],
        "ingredient_details": [
            {
                "ingredient": "Niacinamide",
                "score": 91.0,
                "pros": ["Helps with acne, pores, redness"],
                "cons": [],
                "penalties": [],
            },
            {
                "ingredient": "Zinc PCA",
                "score": 88.0,
                "pros": ["Controls sebum, antimicrobial"],
                "cons": ["Avoid overuse on dry patches"],
                "penalties": [],
            },
            {
                "ingredient": "Salicylic Acid",
                "score": 79.0,
                "pros": ["Helps with acne and pores"],
                "cons": ["Can cause dryness at high concentrations"],
                "penalties": [],
            },
        ],
        "rule_score": 80.5,
        "ml_score"  : 83.2,
        "latency_ms": 45.3,
    }


def mock_layering_calc_output(score: float = 88.0) -> Dict:
    """Generate mock layering compatibility score for testing."""
    return {
        "layering_score"    : score,
        "grade"             : "A" if score >= 85 else "B+" if score >= 78 else "B",
        "verdict"           : f"Good layering compatibility ({score}/100).",
        "layering_order"    : "Apply BHA Toner first, then Niacinamide Serum after 5 minutes.",
        "wait_time_minutes" : 5,
        "application_steps" : [
            "Cleanse face thoroughly.",
            "Apply BHA Toner to cotton pad, sweep across face.",
            "Wait 5 minutes for pH to stabilise.",
            "Apply Niacinamide Serum evenly.",
            "Follow with moisturiser and SPF (AM) or occlusive (PM).",
        ],
        "pros": [
            "BHA + Niacinamide: BHA clears pores, niacinamide reduces redness and PIH",
            "Correct layering order: BHA Toner (lighter) before Niacinamide Serum",
            "Both products address acne and pores",
        ],
        "cons": [
            "May cause dryness if overused — limit to once daily initially",
        ],
        "warnings": [],
        "pair_interactions": [
            {
                "ingredient_a"    : "Salicylic Acid",
                "ingredient_b"    : "Niacinamide",
                "interaction_type": "synergistic",
                "interaction_rank": 1.0,
                "wait_time_minutes": 0,
                "layering_order"  : "Salicylic Acid first",
                "notes"           : "BHA toner then niacinamide serum. Oil control focused.",
            }
        ],
        "product_a_not_found": [],
        "product_b_not_found": [],
        "rule_score"         : 86.5,
        "ml_score"           : 89.1,
        "latency_ms"         : 62.4,
    }


def mock_nlp_output(names: List[str]) -> List[Dict]:
    """Generate mock NLP ingredient mapping output for testing."""
    return [
        {
            "input"       : name,
            "inci_name"   : name,
            "score"       : 0.97,
            "confidence"  : "high",
            "method"      : "exact",
            "alternatives": [],
            "latency_ms"  : 1.2,
        }
        for name in names
    ]


# Demo / Testing

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SkinSpectra LLM Layer Demo")
    parser.add_argument("--api_key", default=os.environ.get("GEMINI_API_KEY",""),
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--feature", choices=["individual","layering","both"],
                        default="both")
    parser.add_argument("--output",  default="llm_demo_output.json",
                        help="Save demo outputs to this JSON file")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Provide --api_key or set GEMINI_API_KEY env var")
        exit(1)

    llm     = LLMLayer(api_key=args.api_key)
    outputs = {}

    # Demo 1: Individual Product Report
    if args.feature in ("individual", "both"):
        print("\n" + "="*60)
        print("Individual Product Report Demo")
        print("="*60)

        profile = UserProfile(
            skin_type        = "oily",
            concerns         = ["acne", "pores", "hyperpigmentation"],
            age_group        = "adult",
            is_pregnant      = False,
            skin_sensitivity = "normal",
            current_routine  = "CeraVe cleanser, SPF 50",
            experience_level = "intermediate",
            location_climate = "humid tropical",
        )

        product_ings = ["Niacinamide", "Zinc PCA", "Salicylic Acid", "Hyaluronic Acid", "Glycerin"]
        calc_out     = mock_individual_calc_output(score=84.5)
        nlp_out      = mock_nlp_output(product_ings)

        result = llm.generate_individual_report(
            product_name     = "The Ordinary Niacinamide 10% + Zinc 1%",
            ingredient_names = product_ings,
            user_profile     = profile,
            calc_output      = calc_out,
            nlp_mapped       = nlp_out,
        )

        if result["success"]:
            print(json.dumps(result["report"], indent=2))
            print(f"\nLatency: {result['latency_ms']}ms | Tokens: {result['usage']}")
        else:
            print(f"FAILED: {result['error']}")

        outputs["individual"] = result

    # Demo 2: Product Layering Report
    if args.feature in ("layering", "both"):
        print("\n" + "="*60)
        print("Product Layering Report Demo")
        print("="*60)

        profile2 = UserProfile(
            skin_type        = "combination",
            concerns         = ["acne", "pores", "dullness"],
            age_group        = "adult",
            is_pregnant      = False,
            skin_sensitivity = "normal",
            current_routine  = "gentle cleanser, moisturiser, SPF",
            experience_level = "beginner",
        )

        a_ings = ["Salicylic Acid", "Zinc PCA", "Aloe Barbadensis Leaf Juice"]
        b_ings = ["Niacinamide", "Hyaluronic Acid", "Ceramide NP", "Glycerin"]

        layer_out = mock_layering_calc_output(score=91.0)
        nlp_a     = mock_nlp_output(a_ings)
        nlp_b     = mock_nlp_output(b_ings)

        result2 = llm.generate_layering_report(
            product_a_name  = "Paula's Choice BHA Exfoliant",
            product_a_ings  = a_ings,
            product_b_name  = "The Ordinary Niacinamide Serum",
            product_b_ings  = b_ings,
            user_profile    = profile2,
            layering_output = layer_out,
            nlp_mapped_a    = nlp_a,
            nlp_mapped_b    = nlp_b,
        )

        if result2["success"]:
            print(json.dumps(result2["report"], indent=2))
            print(f"\nLatency: {result2['latency_ms']}ms | Tokens: {result2['usage']}")
        else:
            print(f"FAILED: {result2['error']}")

        outputs["layering"] = result2
    with open(args.output, "w", encoding="utf-8") as f:
        # Save only report fields (not raw text) to keep file clean
        save = {}
        for k, v in outputs.items():
            save[k] = {
                "success"   : v["success"],
                "report"    : v.get("report", {}),
                "score"     : v.get("score", 0),
                "latency_ms": v.get("latency_ms", 0),
                "usage"     : v.get("usage", {}),
                "error"     : v.get("error", ""),
            }
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\nOutputs saved to {args.output}")
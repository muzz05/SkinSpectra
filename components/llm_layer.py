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

# =============================================================================
# CONFIGURATION
# =============================================================================

GEMINI_MODEL   = "gemini-2.5-flash"
MAX_OUT_TOKENS = 3000
TEMPERATURE    = 0.25   # Low: consistent, clinical tone; not robotic
TOP_P          = 0.90

# =============================================================================
# USER PROFILE DATACLASS
# =============================================================================

@dataclass
class UserProfile:
    """Complete user skin profile passed into the LLM layer."""
    skin_type        : str              # oily / dry / combination / normal / sensitive / mature
    concerns         : List[str]        # e.g. ["acne", "hyperpigmentation"]
    age_group        : str              # teen / adult / mature
    is_pregnant      : bool
    # Optional enrichment fields
    skin_sensitivity : str  = "normal" # low / normal / high
    current_routine  : str  = ""       # free text: "cleanser + SPF only"
    allergies        : str  = ""       # free text: "fragrance, lanolin"
    location_climate : str  = ""       # e.g. "humid tropical" / "dry cold"
    experience_level : str  = "beginner" # beginner / intermediate / advanced

    def to_prompt_str(self) -> str:
        """Compact single-line summary for prompt injection."""
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


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

# --- Feature 1: Individual Product Report ---
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
    "warnings"             : ["str  (critical safety flags only)"],
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

# --- Feature 2: Layering Report ---
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


# =============================================================================
# PROMPT BUILDER
# =============================================================================

class PromptBuilder:
    """
    Builds tight, token-efficient prompts for each feature.
    All prompts follow the same structure:
      [SYSTEM BLOCK] → [USER DATA BLOCK] → [CALC DATA BLOCK] → [OUTPUT INSTRUCTION]
    """

    # ── SYSTEM PROMPT (shared, sent as system role) ────────────────────────
    SYSTEM_PROMPT = textwrap.dedent("""
        You are SkinSpectra, an expert AI dermatology assistant with deep knowledge of cosmetic chemistry, ingredient interactions, and personalised skincare.

        YOUR ROLE:
        - Interpret computational skincare compatibility scores and translate them into clear, actionable, personalised advice
        - Speak like a knowledgeable friend who happens to be a dermatologist — warm, direct, never condescending
        - Prioritise: Safety first → Efficacy → Cosmetic experience
        - Tailor every insight to the specific user profile provided

        STRICT RULES:
        1. Output ONLY valid JSON matching the schema provided. No markdown, no preamble, no explanation outside JSON.
        2. Never diagnose medical conditions. Never claim to replace a dermatologist.
        3. Never invent ingredients not present in the data. Only reference what is given.
        4. When score >= 80: be enthusiastic but accurate. When score < 50: be honest but not alarmist.
        5. Pregnancy warnings must be explicit, clear, and never softened.
        6. Keep all string fields concise. No padding, no filler phrases.
        7. ingredient_highlights: include max 5 most impactful ingredients only.
        8. key_benefits and key_concerns: max 4 each.
    """).strip()

    # ── FEATURE 1: Individual Product Prompt ──────────────────────────────
    @staticmethod
    def build_individual_prompt(
        product_name    : str,
        ingredient_names: List[str],
        user_profile    : UserProfile,
        calc_output     : Dict,
        nlp_mapped      : List[Dict],   # list of {input, inci_name, confidence}
    ) -> str:

        # Compress calc output — only what LLM needs
        score   = calc_output.get("compatibility_score", 0)
        grade   = calc_output.get("grade", "?")
        verdict = calc_output.get("verdict", "")
        pros    = calc_output.get("pros", [])[:5]
        cons    = calc_output.get("cons", [])[:5]
        warns   = calc_output.get("warnings", [])
        details = calc_output.get("ingredient_details", [])

        # Compress ingredient details — top 6 by score impact
        details_sorted = sorted(details, key=lambda x: abs(x.get("score", 50) - 50), reverse=True)[:6]
        ing_summary = []
        for d in details_sorted:
            p_str = "; ".join(d.get("pros", [])[:1])
            c_str = "; ".join(d.get("cons", [])[:1])
            ing_summary.append(
                f"{d['ingredient']}: score={d['score']}"
                + (f", benefit={p_str}" if p_str else "")
                + (f", concern={c_str}"  if c_str else "")
            )

        # NLP confidence flags
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

    # ── FEATURE 2: Layering Report Prompt ─────────────────────────────────
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

        # Compress pair interactions — only show meaningful ones
        pair_lines = []
        for p in pairs[:6]:
            itype = p.get("interaction_type", "unknown")
            ia    = p.get("ingredient_a", "")
            ib    = p.get("ingredient_b", "")
            notes = p.get("notes", "")[:60]
            pair_lines.append(f"{ia} + {ib} = {itype}" + (f" | {notes}" if notes else ""))

        # NLP gaps
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


# =============================================================================
# GEMINI CLIENT
# =============================================================================

class GeminiClient:
    """Thin wrapper around google-genai for SkinSpectra report generation."""

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
        """
        Send prompt to Gemini, parse JSON response.
        Returns parsed dict + metadata.
        """
        t0 = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model    = GEMINI_MODEL,
                contents = user_prompt,
                config   = self._config,
            )
            raw_text = response.text.strip()
            latency  = round((time.perf_counter() - t0) * 1000, 1)

            # Strip markdown fences if model added them despite instructions
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$",          "", raw_text)
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)

            # Usage stats
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


# =============================================================================
# LLM LAYER — MAIN INTERFACE
# =============================================================================

class LLMLayer:
    """
    Top-level interface for the SkinSpectra LLM layer.
    Takes outputs from NLP + Calculation layers and produces
    structured JSON reports via Gemini 2.5 Flash.

    Usage — Feature 1 (Individual Product):
    ----------------------------------------
    llm = LLMLayer(api_key="...")
    result = llm.generate_individual_report(
        product_name     = "The Ordinary Niacinamide 10%",
        ingredient_names = ["Niacinamide", "Zinc PCA", "Glycerin"],
        user_profile     = UserProfile(
            skin_type="oily", concerns=["acne","pores"],
            age_group="adult", is_pregnant=False
        ),
        calc_output  = <output from CompatibilityScorer.score(...)>,
        nlp_mapped   = <output from INCIMapper.batch_map(...)>,
    )

    Usage — Feature 2 (Product Layering):
    ---------------------------------------
    result = llm.generate_layering_report(
        product_a_name  = "Vitamin C Serum",
        product_a_ings  = ["Ascorbic Acid", "Ferulic Acid"],
        product_b_name  = "Retinol Night Cream",
        product_b_ings  = ["Retinol", "Ceramide NP"],
        user_profile    = UserProfile(...),
        layering_output = <output from LayeringScorer.score(...)>,
        nlp_mapped_a    = <output from INCIMapper.batch_map(product_a_ings)>,
        nlp_mapped_b    = <output from INCIMapper.batch_map(product_b_ings)>,
    )
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client  = GeminiClient(api_key=api_key)
        self.builder = PromptBuilder()

    # ------------------------------------------------------------------
    def generate_individual_report(
        self,
        product_name    : str,
        ingredient_names: List[str],
        user_profile    : UserProfile,
        calc_output     : Dict,
        nlp_mapped      : Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Generate a personalised individual product compatibility report.

        Parameters
        ----------
        product_name     : Display name of the product
        ingredient_names : List of INCI (or raw) ingredient names
        user_profile     : UserProfile dataclass
        calc_output      : Output dict from CompatibilityScorer.score()
        nlp_mapped       : Output list from INCIMapper.batch_map() [optional]

        Returns
        -------
        {
          "success"      : bool,
          "report"       : dict (JSON report matching INDIVIDUAL_REPORT_SCHEMA),
          "latency_ms"   : float,
          "usage"        : dict (token counts),
          "feature"      : "individual",
          "product_name" : str,
          "score"        : float,
        }
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

    # ------------------------------------------------------------------
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
        """
        Generate a personalised product layering compatibility report.

        Parameters
        ----------
        product_a_name   : Display name of product A (applied first)
        product_a_ings   : Ingredient list of product A
        product_b_name   : Display name of product B (applied second)
        product_b_ings   : Ingredient list of product B
        user_profile     : UserProfile dataclass
        layering_output  : Output dict from LayeringScorer.score()
        nlp_mapped_a/b   : Output from INCIMapper.batch_map() [optional]

        Returns
        -------
        {
          "success"        : bool,
          "report"         : dict (JSON report matching LAYERING_REPORT_SCHEMA),
          "latency_ms"     : float,
          "usage"          : dict,
          "feature"        : "layering",
          "product_a_name" : str,
          "product_b_name" : str,
          "score"          : float,
        }
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


# =============================================================================
# MOCK CALC OUTPUTS (for standalone testing without trained models)
# =============================================================================

def mock_individual_calc_output(score: float = 82.0) -> Dict:
    """Returns a realistic mock CompatibilityScorer output for testing."""
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
    """Returns a realistic mock LayeringScorer output for testing."""
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
    """Returns mock NLP mapper output."""
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


# =============================================================================
# MAIN — DEMO / QUICK TEST
# =============================================================================

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

    # ── Demo 1: Individual Product ─────────────────────────────────────
    if args.feature in ("individual", "both"):
        print("\n" + "="*60)
        print("  DEMO 1: Individual Product Report")
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

    # ── Demo 2: Layering Report ────────────────────────────────────────
    if args.feature in ("layering", "both"):
        print("\n" + "="*60)
        print("  DEMO 2: Product Layering Report")
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

    # ── Save outputs ───────────────────────────────────────────────────
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
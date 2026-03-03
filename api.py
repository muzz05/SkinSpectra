"""
SkinSpectra FastAPI Application
=================================
Routes
------
  Info / Health
    GET  /                          root welcome + feature map
    GET  /health                    liveness + per-model readiness

  Config  (no auth needed)
    GET  /config/skin-types         valid skin_type values + descriptions
    GET  /config/concerns           valid skin concern values + descriptions
    GET  /config/age-groups         valid age_group values + descriptions
    GET  /config/models             loaded model names + status

  NLP  (ingredient name normalisation)
    POST /nlp/map                   single raw name → INCI standard name
    POST /nlp/map/batch             up to 60 names → INCI (with summary)

  Feature 1 — Individual Product
    POST /analyze/product           skin profile + ingredient list
                                    → NLP → Calc → LLM → JSON report

  Feature 2 — Product Layering
    POST /analyze/layering          skin profile + two ingredient lists
                                    → NLP → Layering → LLM → JSON report

  OCR — Ingredient Label Extraction
    POST /ocr/extract               upload ingredient label image
                                    → pre-process → Tesseract → parse → ingredient list
    GET  /ocr/info                  Tesseract version + supported formats

Run
---
    uvicorn api:app --reload --port 8000
    python api.py               (convenience wrapper)

Env vars
--------
    GEMINI_API_KEY          required for LLM reports
    SS_NLP_MODEL_DIR        default: skinspectra_nlp_model
    SS_CALC_MODEL_DIR       default: skinspectra_calc_model
    SS_LAYERING_MODEL_DIR   default: skinspectra_layering_model
    SS_DATASET2             default: dataset2_ingredient_profiles.csv
    SS_DATASET3             default: dataset3_layering_compatibility.csv
    SS_LLM_ENABLED          default: true
    SS_MAX_INGREDIENTS      default: 60
"""

import os
import sys
import time
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── project layers ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.nlp_layer import INCIMapper
from components.calculation_individual_layer import CompatibilityScorer
from components.calculation_layering_layer import LayeringScorer
from components.llm_layer import LLMLayer, UserProfile
from components.ocr_handler import OCRHandler, TESSERACT_CONFIG
from components.facial_analysis import FacialAnalyzer

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("skinspectra.api")

# =============================================================================
# STATIC CONFIG
# =============================================================================
VALID_SKIN_TYPES    = ["oily", "dry", "combination", "normal", "sensitive", "mature"]
VALID_CONCERNS      = [
    "acne", "dryness", "aging", "hyperpigmentation",
    "redness", "sensitivity", "pores", "texture", "dullness", "barrier",
]
VALID_AGE_GROUPS    = ["teen", "adult", "mature"]
VALID_SENSITIVITIES = ["low", "normal", "high"]
VALID_EXPERIENCE    = ["beginner", "intermediate", "advanced"]
VALID_TOD           = ["AM", "PM", "BOTH"]

API_VERSION     = "1.0.0"
MAX_INGREDIENTS = int(os.getenv("SS_MAX_INGREDIENTS", 60))
LLM_ENABLED     = os.getenv("SS_LLM_ENABLED", "true").lower() == "true"

# Resolve paths relative to this file's directory so they work regardless of cwd
_HERE = Path(__file__).parent

def _resolve(env_val: str, default: str) -> str:
    p = Path(os.getenv(env_val, default))
    return str(p if p.is_absolute() else _HERE / p)

# model artefact paths (override via env)
_NLP_DIR    = _resolve("SS_NLP_MODEL_DIR",       "models/nlp")
_CALC_DIR   = _resolve("SS_CALC_MODEL_DIR",      "models/calculation-individual")
_LAYER_DIR  = _resolve("SS_LAYERING_MODEL_DIR",  "models/calculation-layering")
_FACIAL_DIR = _resolve("SS_FACIAL_MODEL_DIR",    "models/facial_analysis")
_DATASET2   = _resolve("SS_DATASET2",  "data/ingredient_profiles.csv")
_DATASET3   = _resolve("SS_DATASET3",  "data/layering_compatibility.csv")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# =============================================================================
# MODEL STATE  (populated once in lifespan)
# =============================================================================
class _State:
    nlp         : Optional[INCIMapper]          = None
    calc        : Optional[CompatibilityScorer] = None
    layering    : Optional[LayeringScorer]      = None
    llm         : Optional[LLMLayer]            = None
    facial      : Optional[FacialAnalyzer]      = None
    nlp_ok      : bool = False
    calc_ok     : bool = False
    layering_ok : bool = False
    llm_ok      : bool = False
    facial_ok   : bool = False
    boot_secs   : float = 0.0

S = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    log.info("━" * 55)
    log.info("  SkinSpectra API — booting")
    log.info("━" * 55)

    # NLP
    try:
        log.info(f"Loading NLP model from '{_NLP_DIR}'…")
        S.nlp    = INCIMapper.load(_NLP_DIR)
        S.nlp_ok = True
        log.info("NLP model ✓")
    except Exception as e:
        log.error(f"NLP load failed: {e}")

    # Calc
    try:
        log.info(f"Loading Calc model from '{_CALC_DIR}'…")
        S.calc    = CompatibilityScorer.load(_CALC_DIR, data_path=_DATASET2)
        S.calc_ok = True
        log.info("Calc model ✓")
    except Exception as e:
        log.error(f"Calc load failed: {e}")

    # Layering
    try:
        log.info(f"Loading Layering model from '{_LAYER_DIR}'…")
        S.layering    = LayeringScorer.load(_LAYER_DIR, _DATASET2, _DATASET3)
        S.layering_ok = True
        log.info("Layering model ✓")
    except Exception as e:
        log.error(f"Layering load failed: {e}")

    # LLM
    if LLM_ENABLED:
        if not _GEMINI_KEY:
            log.warning("GEMINI_API_KEY not set — LLM layer disabled")
        else:
            try:
                S.llm    = LLMLayer(api_key=_GEMINI_KEY)
                S.llm_ok = True
                log.info("LLM (Gemini 2.5 Flash) ✓")
            except Exception as e:
                log.error(f"LLM init failed: {e}")

    # Facial Analysis
    try:
        log.info(f"Loading Facial Analysis model from '{_FACIAL_DIR}'…")
        S.facial    = FacialAnalyzer.load(model_dir=_FACIAL_DIR)
        S.facial_ok = True
        log.info("Facial Analysis model ✓")
    except Exception as e:
        log.error(f"Facial Analysis load failed: {e}")

    S.boot_secs = round(time.perf_counter() - t0, 2)
    log.info(
        f"Boot complete in {S.boot_secs}s | "
        f"NLP={S.nlp_ok} CALC={S.calc_ok} "
        f"LAYERING={S.layering_ok} LLM={S.llm_ok} FACIAL={S.facial_ok}"
    )
    yield
    log.info("SkinSpectra API — shutdown")


# =============================================================================
# APP
# =============================================================================
app = FastAPI(
    title       = "SkinSpectra API",
    description = (
        "AI-powered skincare analysis — individual product compatibility "
        "and two-product layering analysis with personalised LLM reports."
    ),
    version  = API_VERSION,
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── OCR router ────────────────────────────────────────────────────────────────
_ocr_handler = OCRHandler()
_ocr_router  = APIRouter(prefix="/ocr", tags=["OCR — Ingredient Extraction"])


@_ocr_router.post(
    "/extract",
    summary="Extract ingredient list from an ingredient label image",
    description="""
Upload a photo or scan of a product's ingredient label.

**Supported formats**: JPG, PNG, WEBP, BMP, TIFF

**Best results**:
- Flat, well-lit image of the label
- At least 800px wide
- Text clearly readable (not blurry or heavily shadowed)

**Returns**:
- `ingredients` — parsed list ready for `/analyze/product` or `/analyze/layering`
- `confidence` — high / medium / low / very_low
- `raw_text` — raw Tesseract output (useful for debugging)
- `warnings` — quality or parsing issues detected
""",
)
async def ocr_extract(
    file  : UploadFile = File(..., description="Ingredient label image"),
    debug : bool       = Form(False, description="Include raw OCR text in response"),
):
    allowed_mime = {
        "image/jpeg", "image/png", "image/webp",
        "image/bmp", "image/tiff", "image/tif",
    }
    ct = (file.content_type or "").lower()
    if ct and ct not in allowed_mime:
        raise HTTPException(
            status_code = 415,
            detail      = f"Unsupported media type '{ct}'. "
                          f"Upload a JPG, PNG, WEBP, BMP, or TIFF image.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    result = _ocr_handler.extract_from_bytes(image_bytes, filename=file.filename or "upload")

    if not debug:
        result.pop("raw_text", None)
        result.pop("corrected_text", None)

    status_code = 200 if result["success"] else 422
    return JSONResponse(content=result, status_code=status_code)


@_ocr_router.get(
    "/info",
    summary="OCR engine info — version and supported formats",
)
async def ocr_info():
    return {
        "tesseract_version": _ocr_handler.tesseract_version(),
        "supported_formats": _ocr_handler.supported_formats(),
        "config"           : TESSERACT_CONFIG,
        "description"      : (
            "Tesseract 5 LSTM engine with adaptive binarisation, "
            "auto-deskew, and INCI-aware post-processing"
        ),
    }


app.include_router(_ocr_router)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SkinProfile(BaseModel):
    """User skin profile — shared across both feature endpoints."""
    skin_type        : str       = Field(...,        example="oily",
                                         description=f"One of: {VALID_SKIN_TYPES}")
    concerns         : List[str] = Field([],
                                         description=f"Subset of: {VALID_CONCERNS}")
    age_group        : str       = Field(...,        example="adult",
                                         description=f"One of: {VALID_AGE_GROUPS}")
    is_pregnant      : bool      = Field(False,      description="Is the user pregnant?")
    skin_sensitivity : str       = Field("normal",   example="normal",
                                         description="low / normal / high")
    current_routine  : str       = Field("",         example="gentle cleanser, SPF 50",
                                         description="Free-text: current routine")
    allergies        : str       = Field("",         example="fragrance, lanolin",
                                         description="Known allergens (free text)")
    location_climate : str       = Field("",         example="humid tropical",
                                         description="Climate for personalisation")
    experience_level : str       = Field("beginner", example="intermediate",
                                         description="beginner / intermediate / advanced")

    @field_validator("skin_type")
    @classmethod
    def _v_skin(cls, v):
        if v not in VALID_SKIN_TYPES:
            raise ValueError(f"skin_type must be one of {VALID_SKIN_TYPES}")
        return v

    @field_validator("concerns")
    @classmethod
    def _v_concern(cls, v):
        for item in v:
            if item not in VALID_CONCERNS:
                raise ValueError(f"'{item}' is not a valid concern. Valid: {VALID_CONCERNS}")
        return v

    @field_validator("age_group")
    @classmethod
    def _v_age(cls, v):
        if v not in VALID_AGE_GROUPS:
            raise ValueError(f"age_group must be one of {VALID_AGE_GROUPS}")
        return v

    @field_validator("skin_sensitivity")
    @classmethod
    def _v_sens(cls, v):
        if v not in VALID_SENSITIVITIES:
            raise ValueError(f"skin_sensitivity must be one of {VALID_SENSITIVITIES}")
        return v

    @field_validator("experience_level")
    @classmethod
    def _v_exp(cls, v):
        if v not in VALID_EXPERIENCE:
            raise ValueError(f"experience_level must be one of {VALID_EXPERIENCE}")
        return v


# ── NLP ────────────────────────────────────────────────────────────────────────

class MapSingleRequest(BaseModel):
    ingredient : str = Field(..., example="vitamin c",
                             description="Raw ingredient name to normalise to INCI")


class MapBatchRequest(BaseModel):
    ingredients : List[str] = Field(
        ..., min_length=1,
        example=["niacinamide", "hyaluronic acid", "retinol"],
        description=f"Raw ingredient names to map (max {MAX_INGREDIENTS})",
    )

    @field_validator("ingredients")
    @classmethod
    def _v_batch(cls, v):
        v = [i.strip() for i in v if i.strip()]
        if len(v) > MAX_INGREDIENTS:
            raise ValueError(f"Max {MAX_INGREDIENTS} ingredients per batch request")
        return v


# ── Feature 1 ─────────────────────────────────────────────────────────────────

class ProductAnalysisRequest(BaseModel):
    product_name : str       = Field(
        ..., example="The Ordinary Niacinamide 10% + Zinc 1%",
        description="Display name of the product being analysed",
    )
    ingredients  : List[str] = Field(
        ..., min_length=1,
        example=["Niacinamide", "Zinc PCA", "Glycerin", "Hyaluronic Acid"],
        description=f"Ingredient names — raw or INCI (max {MAX_INGREDIENTS})",
    )
    skin_profile : SkinProfile
    include_llm  : bool = Field(
        True, description="Generate personalised LLM narrative report via Gemini 2.5 Flash",
    )

    @field_validator("ingredients")
    @classmethod
    def _v_ings(cls, v):
        v = [i.strip() for i in v if i.strip()]
        if not v:
            raise ValueError("ingredients list cannot be empty")
        if len(v) > MAX_INGREDIENTS:
            raise ValueError(f"Max {MAX_INGREDIENTS} ingredients allowed")
        return v


# ── Feature 2 ─────────────────────────────────────────────────────────────────

class LayeringAnalysisRequest(BaseModel):
    product_a_name : str       = Field(
        ..., example="Paula's Choice BHA Exfoliant",
        description="Name of product A — applied FIRST",
    )
    product_a_ings : List[str] = Field(
        ..., min_length=1,
        example=["Salicylic Acid", "Zinc PCA", "Aloe Barbadensis Leaf Juice"],
        description=f"Ingredient list for product A (max {MAX_INGREDIENTS})",
    )
    product_b_name : str       = Field(
        ..., example="The Ordinary Niacinamide 10%",
        description="Name of product B — applied SECOND",
    )
    product_b_ings : List[str] = Field(
        ..., min_length=1,
        example=["Niacinamide", "Hyaluronic Acid", "Glycerin"],
        description=f"Ingredient list for product B (max {MAX_INGREDIENTS})",
    )
    skin_profile   : SkinProfile
    time_of_day    : str  = Field(
        "BOTH", example="AM",
        description="AM / PM / BOTH — when this routine is applied",
    )
    include_llm    : bool = Field(
        True, description="Generate personalised LLM narrative report via Gemini 2.5 Flash",
    )

    @field_validator("product_a_ings", "product_b_ings")
    @classmethod
    def _v_ings(cls, v):
        v = [i.strip() for i in v if i.strip()]
        if not v:
            raise ValueError("ingredient list cannot be empty")
        if len(v) > MAX_INGREDIENTS:
            raise ValueError(f"Max {MAX_INGREDIENTS} ingredients per product")
        return v

    @field_validator("time_of_day")
    @classmethod
    def _v_tod(cls, v):
        v = v.upper().strip()
        if v not in VALID_TOD:
            raise ValueError(f"time_of_day must be one of {VALID_TOD}")
        return v


# ── Unified response envelope ──────────────────────────────────────────────────

class APIResponse(BaseModel):
    success    : bool
    version    : str           = API_VERSION
    latency_ms : float
    data       : Optional[Any] = None
    warnings   : List[str]     = []
    error      : Optional[str] = None
    error_code : Optional[str] = None


# =============================================================================
# HELPERS
# =============================================================================

def _need(*flags: str):
    """Raise 503 if any required model has not loaded."""
    labels  = {"nlp": S.nlp_ok, "calc": S.calc_ok,
               "layering": S.layering_ok, "llm": S.llm_ok}
    missing = [f for f in flags if not labels.get(f, False)]
    if missing:
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"Model(s) not loaded: {', '.join(missing)}. "
                "Run the training scripts first and check model directory env vars."
            ),
        )


def _nlp_map(names: List[str]) -> List[Dict]:
    """Map names through the NLP layer, or passthrough if NLP not loaded."""
    if not S.nlp_ok:
        return [
            {"input": n, "inci_name": n, "score": 1.0, "confidence": "high",
             "method": "passthrough", "alternatives": [], "latency_ms": 0.0}
            for n in names
        ]
    return S.nlp.batch_map(names)


def _inci_names(nlp_results: List[Dict]) -> List[str]:
    """Pull INCI names from NLP results — keep all non-empty results."""
    return [r["inci_name"] for r in nlp_results if r.get("inci_name")]


def _user_profile(sp: SkinProfile) -> UserProfile:
    return UserProfile(
        skin_type        = sp.skin_type,
        concerns         = sp.concerns,
        age_group        = sp.age_group,
        is_pregnant      = sp.is_pregnant,
        skin_sensitivity = sp.skin_sensitivity,
        current_routine  = sp.current_routine,
        allergies        = sp.allergies,
        location_climate = sp.location_climate,
        experience_level = sp.experience_level,
    )


def _ok(data: Any, t0: float, warnings: List[str] = None) -> APIResponse:
    return APIResponse(
        success    = True,
        latency_ms = round((time.perf_counter() - t0) * 1000, 2),
        data       = data,
        warnings   = warnings or [],
    )


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def _exc_handler(req: Request, exc: Exception):
    log.error(f"Unhandled on {req.url}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code = 500,
        content     = APIResponse(
            success=False, latency_ms=0.0,
            error=f"Internal server error: {exc}"
        ).dict(),
    )


@app.exception_handler(HTTPException)
async def _http_handler(req: Request, exc: HTTPException):
    return JSONResponse(
        status_code = exc.status_code,
        content     = APIResponse(
            success=False, latency_ms=0.0, error=exc.detail
        ).dict(),
    )


# =============================================================================
# INFO / HEALTH
# =============================================================================

@app.get("/", tags=["Info"], summary="SkinSpectra web UI", include_in_schema=False)
async def root():
    return FileResponse(_HERE / "skinspectra.html", media_type="text/html")


@app.get("/health", tags=["Info"], summary="Liveness and per-model readiness")
async def health():
    ready = S.nlp_ok and S.calc_ok and S.layering_ok
    return {
        "status"        : "healthy" if ready else "degraded",
        "version"       : API_VERSION,
        "boot_time_sec" : S.boot_secs,
        "models"        : {
            "nlp"      : "ready" if S.nlp_ok       else "not_loaded",
            "calc"     : "ready" if S.calc_ok       else "not_loaded",
            "layering" : "ready" if S.layering_ok   else "not_loaded",
            "llm"      : "ready" if S.llm_ok        else "not_loaded",
            "facial"   : "ready" if S.facial_ok     else "not_loaded",
        },
    }


# =============================================================================
# CONFIG
# =============================================================================

@app.get("/config/skin-types", tags=["Config"],
         summary="Valid skin_type values accepted by the API")
async def config_skin_types():
    return {
        "skin_types"   : VALID_SKIN_TYPES,
        "descriptions" : {
            "oily"        : "Excess sebum, shine, enlarged pores",
            "dry"         : "Tight or flaky skin, lacks moisture",
            "combination" : "Oily T-zone (forehead/nose/chin), normal or dry cheeks",
            "normal"      : "Balanced, minimal concerns, small pores",
            "sensitive"   : "Reactive skin, prone to redness and irritation",
            "mature"      : "Reduced elasticity and collagen, fine lines present",
        },
    }


@app.get("/config/concerns", tags=["Config"],
         summary="Valid skin concern values accepted by the API")
async def config_concerns():
    return {
        "concerns"     : VALID_CONCERNS,
        "descriptions" : {
            "acne"             : "Pimples, blackheads, whiteheads, comedones",
            "dryness"          : "Lack of hydration, tight or flaky skin",
            "aging"            : "Fine lines, wrinkles, loss of firmness",
            "hyperpigmentation": "Dark spots, uneven tone, melasma, post-inflammatory marks",
            "redness"          : "Flushing, rosacea, visible inflammation",
            "sensitivity"      : "Reactive skin, frequent irritation",
            "pores"            : "Enlarged or visibly clogged pores",
            "texture"          : "Rough, bumpy, or uneven skin surface",
            "dullness"         : "Lack of radiance or natural glow",
            "barrier"          : "Compromised skin barrier, eczema, excess water loss",
        },
    }


@app.get("/config/age-groups", tags=["Config"],
         summary="Valid age_group values accepted by the API")
async def config_age_groups():
    return {
        "age_groups"   : VALID_AGE_GROUPS,
        "descriptions" : {
            "teen"  : "13–19 years — gentle actives, acne management priority",
            "adult" : "20–49 years — broad active ingredient range",
            "mature": "50+ years — anti-aging focus, richer formulations",
        },
    }


@app.get("/config/models", tags=["Config"],
         summary="Loaded model names and readiness status")
async def config_models():
    return {
        "api_version" : API_VERSION,
        "models"      : {
            "nlp" : {
                "name"     : "sentence-transformers/all-MiniLM-L6-v2  (INCI mapper)",
                "status"   : "ready" if S.nlp_ok else "not_loaded",
                "purpose"  : "Maps raw ingredient names → INCI standard names",
                "model_dir": _NLP_DIR,
            },
            "calc" : {
                "name"     : "XGBoost + Rule Engine  (individual product scorer)",
                "status"   : "ready" if S.calc_ok else "not_loaded",
                "purpose"  : "Scores a single product vs a user skin profile (0–100)",
                "model_dir": _CALC_DIR,
            },
            "layering" : {
                "name"     : "LightGBM + Rule Engine  (layering scorer)",
                "status"   : "ready" if S.layering_ok else "not_loaded",
                "purpose"  : "Scores two-product layering compatibility (0–100)",
                "model_dir": _LAYER_DIR,
            },
            "llm" : {
                "name"    : "Gemini 2.5 Flash  (personalised report generator)",
                "status"  : "ready" if S.llm_ok else "not_loaded",
                "purpose" : "Generates structured JSON reports from calc layer outputs",
                "enabled" : LLM_ENABLED,
            },
            "facial" : {
                "name"     : "EfficientNetV2B2  (facial skin-type classifier)",
                "status"   : "ready" if S.facial_ok else "not_loaded",
                "purpose"  : "Detects face and predicts Dry / Normal / Oily skin type from a photo",
                "model_dir": _FACIAL_DIR,
            },
        },
    }


# =============================================================================
# NLP ENDPOINTS
# =============================================================================

@app.post(
    "/nlp/map",
    tags=["NLP"],
    summary="Map a single raw ingredient name to its INCI standard name",
)
async def nlp_map_single(req: MapSingleRequest):
    t0 = time.perf_counter()
    _need("nlp")
    if not req.ingredient.strip():
        raise HTTPException(422, "ingredient cannot be an empty string")
    result = S.nlp.map(req.ingredient.strip())
    return _ok(result, t0)


@app.post(
    "/nlp/map/batch",
    tags=["NLP"],
    summary=f"Map up to {MAX_INGREDIENTS} raw ingredient names to INCI standard names",
)
async def nlp_map_batch(req: MapBatchRequest):
    t0     = time.perf_counter()
    _need("nlp")
    mapped = S.nlp.batch_map(req.ingredients)
    summary = {
        "high"     : sum(1 for r in mapped if r["confidence"] == "high"),
        "medium"   : sum(1 for r in mapped if r["confidence"] == "medium"),
        "low"      : sum(1 for r in mapped if r["confidence"] == "low"),
        "uncertain": sum(1 for r in mapped if r["confidence"] == "uncertain"),
    }
    return _ok({"count": len(mapped), "mapped": mapped, "summary": summary}, t0)


# =============================================================================
# FEATURE 1 — INDIVIDUAL PRODUCT
# =============================================================================

@app.post(
    "/analyze/product",
    tags=["Feature 1 — Individual Product"],
    summary="Individual product compatibility analysis",
    description="""
Runs the full three-layer pipeline:

1. **NLP** — normalises raw ingredient names to INCI standard names  
2. **Calc** (XGBoost + rule engine) — scores the product 0–100 against the user's skin profile  
3. **LLM** (Gemini 2.5 Flash) — generates a personalised structured JSON report  

**Response includes**: score, grade, verdict, pros/cons, per-ingredient breakdown,  
safety warnings, and a full LLM report with headline, skin-fit rating, usage tips, and routine advice.
""",
)
async def analyze_product(req: ProductAnalysisRequest):
    t0       = time.perf_counter()
    warnings = []
    _need("nlp", "calc")

    # ── Step 1 · NLP mapping ──────────────────────────────────────────
    nlp_results = _nlp_map(req.ingredients)
    inci_names  = _inci_names(nlp_results)
    uncertain   = [r["input"] for r in nlp_results if r["confidence"] == "uncertain"]
    if uncertain:
        warnings.append(
            f"Low-confidence NLP mapping for: {', '.join(uncertain[:5])}. "
            "Accuracy may be reduced for these ingredients."
        )

    # ── Step 2 · Calc layer ───────────────────────────────────────────
    try:
        calc = S.calc.score(
            ingredient_names = inci_names,
            skin_type        = req.skin_profile.skin_type,
            concerns         = req.skin_profile.concerns,
            age_group        = req.skin_profile.age_group,
            is_pregnant      = req.skin_profile.is_pregnant,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    # ── Step 3 · LLM layer ────────────────────────────────────────────
    llm_report = None
    llm_meta   = {"enabled": False, "latency_ms": 0.0, "tokens": {}}

    if req.include_llm:
        if not S.llm_ok:
            warnings.append(
                "LLM report skipped — GEMINI_API_KEY is not configured or LLM failed to load."
            )
        else:
            llm_resp = S.llm.generate_individual_report(
                product_name     = req.product_name,
                ingredient_names = inci_names,
                user_profile     = _user_profile(req.skin_profile),
                calc_output      = calc,
                nlp_mapped       = nlp_results,
            )
            llm_meta = {
                "enabled"   : True,
                "latency_ms": llm_resp.get("latency_ms", 0.0),
                "tokens"    : llm_resp.get("usage", {}),
            }
            if llm_resp["success"]:
                llm_report = llm_resp["report"]
            else:
                warnings.append(f"LLM report failed: {llm_resp.get('error', 'unknown error')}")

    # ── Build response ────────────────────────────────────────────────
    data = {
        "product_name" : req.product_name,
        "score"        : calc["compatibility_score"],
        "grade"        : calc["grade"],
        "verdict"      : calc["verdict"],
        "nlp"          : {
            "ingredients_received" : len(req.ingredients),
            "ingredients_mapped"   : len(inci_names),
            "uncertain"            : uncertain,
            "details"              : nlp_results,
        },
        "calc"         : {
            "score"              : calc["compatibility_score"],
            "grade"              : calc["grade"],
            "verdict"            : calc["verdict"],
            "pros"               : calc["pros"],
            "cons"               : calc["cons"],
            "warnings"           : calc["warnings"],
            "ingredient_details" : calc["ingredient_details"],
            "not_found_in_db"    : calc["not_found"],
            "rule_score"         : calc["rule_score"],
            "ml_score"           : calc["ml_score"],
            "latency_ms"         : calc["latency_ms"],
        },
        "llm_report"   : llm_report,
        "meta"         : {
            "llm"              : llm_meta,
            "total_latency_ms" : round((time.perf_counter() - t0) * 1000, 2),
        },
    }

    log.info(
        f"[product] '{req.product_name}' score={calc['compatibility_score']} "
        f"skin={req.skin_profile.skin_type} "
        f"lat={round((time.perf_counter() - t0) * 1000)}ms"
    )
    return _ok(data, t0, warnings)


# =============================================================================
# FEATURE 2 — PRODUCT LAYERING
# =============================================================================

@app.post(
    "/analyze/layering",
    tags=["Feature 2 — Product Layering"],
    summary="Two-product layering compatibility analysis",
    description="""
Runs the full three-layer pipeline for two products:

1. **NLP** (×2) — normalises ingredient names for both products  
2. **Layering** (LightGBM + rule engine) — scores layering compatibility 0–100  
3. **LLM** (Gemini 2.5 Flash) — generates personalised layering report  

**Response includes**: layering score, grade, compatibility verdict, step-by-step  
application protocol with wait times, ingredient-pair synergy/conflict breakdown,  
pregnancy warnings, and a full LLM report with pro tips and concern coverage.
""",
)
async def analyze_layering(req: LayeringAnalysisRequest):
    t0       = time.perf_counter()
    warnings = []
    _need("nlp", "layering")

    # ── Step 1 · NLP mapping (both products) ─────────────────────────
    nlp_a  = _nlp_map(req.product_a_ings)
    nlp_b  = _nlp_map(req.product_b_ings)
    inci_a = _inci_names(nlp_a)
    inci_b = _inci_names(nlp_b)
    unc_a  = [r["input"] for r in nlp_a if r["confidence"] == "uncertain"]
    unc_b  = [r["input"] for r in nlp_b if r["confidence"] == "uncertain"]
    if unc_a:
        warnings.append(
            f"Low-confidence NLP mapping in '{req.product_a_name}': {', '.join(unc_a[:3])}"
        )
    if unc_b:
        warnings.append(
            f"Low-confidence NLP mapping in '{req.product_b_name}': {', '.join(unc_b[:3])}"
        )

    # ── Step 2 · Layering layer ───────────────────────────────────────
    try:
        layer = S.layering.score(
            product_a_name        = req.product_a_name,
            product_a_ingredients = inci_a,
            product_b_name        = req.product_b_name,
            product_b_ingredients = inci_b,
            skin_type             = req.skin_profile.skin_type,
            concerns              = req.skin_profile.concerns,
            age_group             = req.skin_profile.age_group,
            is_pregnant           = req.skin_profile.is_pregnant,
            time_of_day           = req.time_of_day,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    # ── Step 3 · LLM layer ────────────────────────────────────────────
    llm_report = None
    llm_meta   = {"enabled": False, "latency_ms": 0.0, "tokens": {}}

    if req.include_llm:
        if not S.llm_ok:
            warnings.append(
                "LLM report skipped — GEMINI_API_KEY is not configured or LLM failed to load."
            )
        else:
            llm_resp = S.llm.generate_layering_report(
                product_a_name  = req.product_a_name,
                product_a_ings  = inci_a,
                product_b_name  = req.product_b_name,
                product_b_ings  = inci_b,
                user_profile    = _user_profile(req.skin_profile),
                layering_output = layer,
                nlp_mapped_a    = nlp_a,
                nlp_mapped_b    = nlp_b,
            )
            llm_meta = {
                "enabled"   : True,
                "latency_ms": llm_resp.get("latency_ms", 0.0),
                "tokens"    : llm_resp.get("usage", {}),
            }
            if llm_resp["success"]:
                llm_report = llm_resp["report"]
            else:
                warnings.append(f"LLM report failed: {llm_resp.get('error', 'unknown error')}")

    # ── Build response ────────────────────────────────────────────────
    data = {
        "product_a_name" : req.product_a_name,
        "product_b_name" : req.product_b_name,
        "score"          : layer["layering_score"],
        "grade"          : layer["grade"],
        "verdict"        : layer["verdict"],
        "nlp"            : {
            "product_a" : {
                "received" : len(req.product_a_ings),
                "mapped"   : len(inci_a),
                "uncertain": unc_a,
                "details"  : nlp_a,
            },
            "product_b" : {
                "received" : len(req.product_b_ings),
                "mapped"   : len(inci_b),
                "uncertain": unc_b,
                "details"  : nlp_b,
            },
        },
        "layering"       : {
            "score"               : layer["layering_score"],
            "grade"               : layer["grade"],
            "verdict"             : layer["verdict"],
            "layering_order"      : layer["layering_order"],
            "wait_time_minutes"   : layer["wait_time_minutes"],
            "application_steps"   : layer["application_steps"],
            "pros"                : layer["pros"],
            "cons"                : layer["cons"],
            "warnings"            : layer["warnings"],
            "pair_interactions"   : layer["pair_interactions"],
            "product_a_not_found" : layer["product_a_not_found"],
            "product_b_not_found" : layer["product_b_not_found"],
            "rule_score"          : layer["rule_score"],
            "ml_score"            : layer["ml_score"],
            "latency_ms"          : layer["latency_ms"],
        },
        "llm_report"     : llm_report,
        "meta"           : {
            "time_of_day"      : req.time_of_day,
            "llm"              : llm_meta,
            "total_latency_ms" : round((time.perf_counter() - t0) * 1000, 2),
        },
    }

    log.info(
        f"[layering] '{req.product_a_name}' + '{req.product_b_name}' "
        f"score={layer['layering_score']} skin={req.skin_profile.skin_type} "
        f"lat={round((time.perf_counter() - t0) * 1000)}ms"
    )
    return _ok(data, t0, warnings)


# =============================================================================
# FEATURE 3 — FACIAL SKIN-TYPE DETECTION
# =============================================================================

@app.post(
    "/analyze/skin-type",
    tags=["Feature 3 — Facial Skin-Type Detection"],
    summary="Detect skin type from a face photo",
    description="""
Upload a face photo and receive an AI-predicted skin type.

**Supported formats**: JPG, PNG, WEBP, BMP

**Best results**:
- Clear, front-facing photo in good lighting
- Face takes up most of the frame
- No heavy filters or extreme shadows

**Returns**: `skin_type` (Dry / Normal / Oily), `confidence`, `all_probabilities`, `latency_ms`
""",
)
async def analyze_skin_type(
    file: UploadFile = File(..., description="Face photo (JPG / PNG / WEBP / BMP)"),
):
    t0 = time.perf_counter()

    if not S.facial_ok:
        raise HTTPException(
            status_code=503,
            detail=(
                "Facial analysis model not loaded. "
                "Run training or check SS_FACIAL_MODEL_DIR env var."
            ),
        )

    allowed_mime = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    ct = (file.content_type or "").lower()
    if ct and ct not in allowed_mime:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{ct}'. Upload a JPG, PNG, WEBP, or BMP image.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    import tempfile, cv2
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = S.facial.predict(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Handle face-detection / blur errors ────────────────────────────────────
    if "error" in result:
        error_code = result["error"]
        lat = round((time.perf_counter() - t0) * 1000, 2)
        if error_code == "blurry":
            error_msg = (
                "The uploaded image is too blurry to analyse. "
                "Please upload a sharper, well-lit photo."
            )
        else:  # no_face
            error_msg = (
                "No face detected in the uploaded image. "
                "Please use a clear, front-facing photo."
            )
        log.warning(f"[skin-type] {error_code}: {error_msg}")
        return JSONResponse(
            status_code=422,
            content=APIResponse(
                success    = False,
                latency_ms = lat,
                error      = error_msg,
                error_code = error_code,
            ).dict(),
        )

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        f"[skin-type] skin={result['skin_type']} "
        f"conf={result['confidence']} lat={result['latency_ms']}ms"
    )
    return _ok(result, t0)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SkinSpectra API Server")
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Hot-reload (dev only)")
    a = p.parse_args()
    uvicorn.run("api:app", host=a.host, port=a.port, reload=a.reload, log_level="info")
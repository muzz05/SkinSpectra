"""
SkinSpectra API — Mock Test Script
====================================
Tests all 9 endpoints with realistic payloads and edge cases.
No Gemini key required by default (include_llm=False).
Pass --llm to enable LLM tests (server must have GEMINI_API_KEY set).

Usage
-----
    # Start server first:
    python api.py --port 8000

    # Run all tests (no LLM):
    python test_api.py

    # Run with LLM:
    python test_api.py --llm

    # Custom server URL:
    python test_api.py --url http://localhost:8000

    # Run specific tests only:
    python test_api.py --only T12 T23 T24

    # Skip specific tests:
    python test_api.py --skip T10 T11

Test Map
--------
  Config & Health (6)
    T01  GET  /
    T02  GET  /health
    T03  GET  /config/skin-types
    T04  GET  /config/concerns
    T05  GET  /config/age-groups
    T06  GET  /config/models

  NLP (6)
    T07  POST /nlp/map               — known ingredient
    T08  POST /nlp/map               — unknown ingredient (graceful)
    T09  POST /nlp/map               — empty string (422)
    T10  POST /nlp/map/batch         — batch of 10
    T11  POST /nlp/map/batch         — >60 items (422)
    T12  POST /nlp/map/batch         — confidence summary keys

  Feature 1 — Individual Product (12)
    T13  POST /analyze/product       — ideal product, oily + acne
    T14  POST /analyze/product       — bad product, sensitive skin
    T15  POST /analyze/product       — pregnant user + unsafe ingredients
    T16  POST /analyze/product       — pregnant user + safe ingredients
    T17  POST /analyze/product       — teen + retinol (age penalty)
    T18  POST /analyze/product       — mature dry skin, anti-aging
    T19  POST /analyze/product       — no concerns (empty list)
    T20  POST /analyze/product       — all 6 skin types round-trip
    T21  POST /analyze/product       — invalid skin_type → 422
    T22  POST /analyze/product       — invalid concern → 422
    T23  POST /analyze/product       — empty ingredients → 422
    T24  POST /analyze/product       — response schema validation

  Feature 2 — Layering (11)
    T25  POST /analyze/layering      — ideal pair (HA + ceramide)
    T26  POST /analyze/layering      — classic conflict (VitC + retinol same PM)
    T27  POST /analyze/layering      — unsafe pregnancy layering
    T28  POST /analyze/layering      — safe pregnancy layering
    T29  POST /analyze/layering      — AM routine (VitC + SPF)
    T30  POST /analyze/layering      — PM routine (retinol + ceramide)
    T31  POST /analyze/layering      — teen oily acne routine
    T32  POST /analyze/layering      — wrong order penalty check
    T33  POST /analyze/layering      — invalid time_of_day → 422
    T34  POST /analyze/layering      — missing product_b_ings → 422
    T35  POST /analyze/layering      — response schema validation
"""

import sys
import json
import time
import argparse
import statistics
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed — run: pip install requests")
    sys.exit(1)

# ── terminal colours ──────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[1m";  X = "\033[0m"

def _ok(msg):  print(f"  {G}PASS{X}  {msg}")
def _fl(msg):  print(f"  {R}FAIL{X}  {msg}")
def _wn(msg):  print(f"  {Y}WARN{X}  {msg}")
def _in(msg):  print(f"  {C}INFO{X}  {msg}")

def _hdr(tid: str, method: str, path: str, desc: str):
    print(f"\n{B}{C}{'─'*70}{X}")
    print(f"{B}  {tid}   {method:5}  {path}{X}")
    print(f"       {desc}")
    print(f"{'─'*70}")

# globals set by CLI
BASE = "http://localhost:8000"
LLM  = False


# =============================================================================
# RUNNER UTIL
# =============================================================================

class R_:
    """Tiny per-test result tracker."""
    def __init__(self, tid: str):
        self.tid = tid; self.p = 0; self.f = 0; self.w = 0; self.lat = 0.0
    @property
    def pct(self): return 100*self.p/(self.p+self.f+self.w) if (self.p+self.f+self.w) else 0

ALL: List[R_] = []


def chk(cond: bool, desc: str, r: R_):
    if cond: _ok(desc);  r.p += 1
    else:    _fl(desc);  r.f += 1


def call(method: str, path: str, **kw) -> Tuple[Optional[requests.Response], float]:
    url = BASE + path
    t0  = time.perf_counter()
    try:
        resp = getattr(requests, method)(url, timeout=90, **kw)
        return resp, time.perf_counter() - t0
    except requests.exceptions.ConnectionError:
        print(f"\n{R}  CONNECTION REFUSED — is the server running at {BASE}?{X}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{R}  Request error: {e}{X}")
        return None, 0.0


def envelope_ok(body: dict, r: R_):
    """Validate standard response envelope fields."""
    chk("success"    in body and body["success"] is True, "success=True",    r)
    chk("version"    in body,                             "envelope.version",  r)
    chk("latency_ms" in body,                             "envelope.latency_ms",r)
    chk("data"       in body,                             "envelope.data",     r)


def score_ok(data: dict, key: str, r: R_) -> float:
    s = data.get(key, -1)
    chk(isinstance(s, (int, float)), f"{key} is numeric", r)
    chk(0 <= s <= 100,               f"{key}={s} in [0,100]", r)
    return float(s)


def show_score(label: str, score: float, grade: str):
    col = G if score >= 75 else (Y if score >= 50 else R)
    _in(f"{label}: {col}{score}/100  Grade: {grade}{X}")


# =============================================================================
# SHARED SKIN PROFILES
# =============================================================================

OILY_ACNE = {
    "skin_type": "oily", "concerns": ["acne","pores","hyperpigmentation"],
    "age_group": "adult", "is_pregnant": False,
    "skin_sensitivity": "normal", "current_routine": "CeraVe cleanser + SPF 50",
    "experience_level": "intermediate", "location_climate": "humid tropical",
}
DRY_MATURE = {
    "skin_type": "dry", "concerns": ["aging","dryness","texture"],
    "age_group": "mature", "is_pregnant": False,
    "skin_sensitivity": "low", "current_routine": "gentle cleanser + rich cream",
    "experience_level": "advanced",
}
PREGNANT = {
    "skin_type": "normal", "concerns": ["hyperpigmentation","dryness"],
    "age_group": "adult",  "is_pregnant": True,
    "skin_sensitivity": "high", "experience_level": "beginner",
}
TEEN = {
    "skin_type": "oily", "concerns": ["acne","pores"],
    "age_group": "teen", "is_pregnant": False,
    "skin_sensitivity": "normal", "experience_level": "beginner",
}
SENSITIVE = {
    "skin_type": "sensitive", "concerns": ["redness","sensitivity","barrier"],
    "age_group": "adult", "is_pregnant": False,
    "skin_sensitivity": "high", "experience_level": "beginner",
}
COMBO = {
    "skin_type": "combination", "concerns": ["acne","dullness","pores"],
    "age_group": "adult", "is_pregnant": False,
    "skin_sensitivity": "normal", "experience_level": "intermediate",
}


# =============================================================================
# CONFIG & HEALTH  (T01 – T06)
# =============================================================================

def t01():
    r = R_("T01"); _hdr("T01","GET","/","Root — feature map")
    resp, lat = call("get","/"); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json()
    for k in ["name","version","endpoints"]:
        chk(k in body, f"body.{k} present", r)
    _in(f"name={body.get('name')}  ver={body.get('version')}")
    ALL.append(r); return r

def t02():
    r = R_("T02"); _hdr("T02","GET","/health","Liveness + model readiness")
    resp, lat = call("get","/health"); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json()
    chk("status" in body, "body.status present", r)
    chk("models" in body, "body.models present", r)
    models = body.get("models",{})
    for m in ["nlp","calc","layering","llm"]:
        s = models.get(m,"?")
        if s == "ready": _ok(f"model '{m}' = ready");   r.p += 1
        else:            _wn(f"model '{m}' = {s}");     r.w += 1
    _in(f"overall status = {body.get('status')}")
    ALL.append(r); return r

def t03():
    r = R_("T03"); _hdr("T03","GET","/config/skin-types","Valid skin type values")
    resp, lat = call("get","/config/skin-types"); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json()
    chk("skin_types" in body, "body.skin_types present", r)
    sts = body.get("skin_types",[])
    chk(len(sts) == 6, f"6 skin types returned (got {len(sts)})", r)
    for s in ["oily","dry","combination","normal","sensitive","mature"]:
        chk(s in sts, f"'{s}' in list", r)
    ALL.append(r); return r

def t04():
    r = R_("T04"); _hdr("T04","GET","/config/concerns","Valid concern values")
    resp, lat = call("get","/config/concerns"); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    concerns = resp.json().get("concerns",[])
    chk(len(concerns) >= 10, f">= 10 concerns returned (got {len(concerns)})", r)
    for c in ["acne","aging","dryness","hyperpigmentation","redness","barrier"]:
        chk(c in concerns, f"'{c}' in list", r)
    ALL.append(r); return r

def t05():
    r = R_("T05"); _hdr("T05","GET","/config/age-groups","Valid age group values")
    resp, lat = call("get","/config/age-groups"); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    groups = resp.json().get("age_groups",[])
    chk(set(groups) == {"teen","adult","mature"}, f"groups={groups}", r)
    ALL.append(r); return r

def t06():
    r = R_("T06"); _hdr("T06","GET","/config/models","Model status info")
    resp, lat = call("get","/config/models"); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json()
    chk("models" in body, "body.models present", r)
    for m in ["nlp","calc","layering","llm"]:
        chk(m in body.get("models",{}), f"model '{m}' listed", r)
    ALL.append(r); return r


# =============================================================================
# NLP  (T07 – T12)
# =============================================================================

def t07():
    r = R_("T07"); _hdr("T07","POST","/nlp/map","Single known ingredient")
    resp, lat = call("post","/nlp/map", json={"ingredient":"niacinamide"}); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json(); envelope_ok(body, r)
    data = body.get("data",{})
    for k in ["inci_name","confidence","score","method"]:
        chk(k in data, f"data.{k} present", r)
    _in(f"'niacinamide' → '{data.get('inci_name')}'  conf={data.get('confidence')}")
    ALL.append(r); return r

def t08():
    r = R_("T08"); _hdr("T08","POST","/nlp/map","Unknown ingredient — graceful")
    resp, lat = call("post","/nlp/map", json={"ingredient":"ZZUNKNOWNXYZ999"}); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200 (not 500)", r)
    data = resp.json().get("data",{})
    conf = data.get("confidence","")
    chk(conf in ("low","uncertain","medium"), f"low/uncertain confidence: {conf}", r)
    _in(f"unknown → '{data.get('inci_name')}'  conf={conf}")
    ALL.append(r); return r

def t09():
    r = R_("T09"); _hdr("T09","POST","/nlp/map","Empty string → 422")
    resp, lat = call("post","/nlp/map", json={"ingredient":""}); r.lat=lat
    chk(resp.status_code in (422,400), f"HTTP {resp.status_code} is 400/422", r)
    ALL.append(r); return r

def t10():
    r = R_("T10"); _hdr("T10","POST","/nlp/map/batch","Batch of 10 ingredients")
    ings = ["niacinamide","hyaluronic acid","retinol","vitamin c","salicylic acid",
            "ceramide","glycerin","zinc oxide","bakuchiol","azelaic acid"]
    resp, lat = call("post","/nlp/map/batch", json={"ingredients":ings}); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json(); envelope_ok(body, r)
    data = body.get("data",{})
    chk("count"   in data, "data.count present", r)
    chk("mapped"  in data, "data.mapped present", r)
    chk("summary" in data, "data.summary present", r)
    chk(data.get("count") == len(ings), f"count={data.get('count')} == {len(ings)}", r)
    s = data.get("summary",{})
    total = sum(s.get(k,0) for k in ["high","medium","low","uncertain"])
    chk(total == len(ings), f"summary totals = {total}", r)
    _in(f"high={s.get('high')} med={s.get('medium')} low={s.get('low')} unc={s.get('uncertain')}")
    ALL.append(r); return r

def t11():
    r = R_("T11"); _hdr("T11","POST","/nlp/map/batch",">60 items → 422")
    big = [f"ingredient_{i}" for i in range(65)]
    resp, lat = call("post","/nlp/map/batch", json={"ingredients":big}); r.lat=lat
    chk(resp.status_code in (400,422), f"HTTP {resp.status_code} is 400/422", r)
    ALL.append(r); return r

def t12():
    r = R_("T12"); _hdr("T12","POST","/nlp/map/batch","Summary dict has all 4 keys")
    resp, lat = call("post","/nlp/map/batch", json={"ingredients":["glycerin","retinol"]}); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    s = resp.json().get("data",{}).get("summary",{})
    for k in ["high","medium","low","uncertain"]:
        chk(k in s, f"summary.{k} present", r)
    ALL.append(r); return r


# =============================================================================
# FEATURE 1 — INDIVIDUAL PRODUCT  (T13 – T24)
# =============================================================================

def _prod(name, ings, profile, llm=False):
    return call("post","/analyze/product", json={
        "product_name": name, "ingredients": ings,
        "skin_profile": profile, "include_llm": llm,
    })

def t13():
    r = R_("T13"); _hdr("T13","POST","/analyze/product","Ideal product — oily acne skin")
    resp, lat = _prod("The Ordinary Niacinamide 10% + Zinc",
                      ["Niacinamide","Zinc PCA","Glycerin","Hyaluronic Acid"],
                      OILY_ACNE, llm=LLM); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json(); envelope_ok(body, r)
    data = body.get("data",{})
    s = score_ok(data, "score", r)
    chk(s >= 70, f"ideal score {s} >= 70", r)
    chk("nlp"  in data, "data.nlp present", r)
    chk("calc" in data, "data.calc present", r)
    chk("llm_report" in data, "data.llm_report key present", r)
    calc = data.get("calc",{})
    chk(isinstance(calc.get("pros",[]),list), "calc.pros is list", r)
    chk(isinstance(calc.get("cons",[]),list), "calc.cons is list", r)
    show_score("Ideal product", s, data.get("grade","?"))
    if LLM and data.get("llm_report"):
        _in(f"LLM headline: {data['llm_report'].get('headline','')}")
    ALL.append(r); return r

def t14():
    r = R_("T14"); _hdr("T14","POST","/analyze/product","Bad product — coconut oil + sensitive skin")
    resp, lat = _prod("Heavy Coconut Balm",
                      ["Coconut Oil","Isopropyl Myristate","Mineral Oil"],
                      SENSITIVE); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s <= 60, f"bad product score {s} <= 60", r)
    chk(len(data.get("calc",{}).get("cons",[])) >= 1, "has cons", r)
    show_score("Bad product (sensitive)", s, data.get("grade","?"))
    ALL.append(r); return r

def t15():
    r = R_("T15"); _hdr("T15","POST","/analyze/product","Pregnant + unsafe ingredients")
    resp, lat = _prod("Retinol Night Cream",
                      ["Retinol","Glycolic Acid","Benzoyl Peroxide"],
                      PREGNANT); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s <= 35, f"unsafe pregnancy score {s} <= 35", r)
    warns = data.get("calc",{}).get("warnings",[])
    has_w = any("pregnan" in w.lower() or "PREGNANCY" in w for w in warns)
    chk(has_w, "pregnancy warning in calc.warnings", r)
    show_score("Pregnant + unsafe", s, data.get("grade","?"))
    ALL.append(r); return r

def t16():
    r = R_("T16"); _hdr("T16","POST","/analyze/product","Pregnant + safe ingredients")
    resp, lat = _prod("Gentle Brightening Serum",
                      ["Niacinamide","Azelaic Acid","Hyaluronic Acid","Glycerin","Allantoin"],
                      PREGNANT); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s >= 65, f"safe pregnancy score {s} >= 65", r)
    show_score("Pregnant + safe", s, data.get("grade","?"))
    ALL.append(r); return r

def t17():
    r = R_("T17"); _hdr("T17","POST","/analyze/product","Teen + retinol — age penalty")
    resp, lat = _prod("Retinol Serum", ["Retinol","Glycolic Acid"], TEEN); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s <= 65, f"teen+retinol score {s} <= 65 (age penalty applied)", r)
    show_score("Teen + retinol", s, data.get("grade","?"))
    ALL.append(r); return r

def t18():
    r = R_("T18"); _hdr("T18","POST","/analyze/product","Mature dry skin — anti-aging")
    resp, lat = _prod("Advanced Anti-Aging Serum",
                      ["Retinol","Hyaluronic Acid","Ceramide NP","Squalane","Panthenol"],
                      DRY_MATURE); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s >= 65, f"mature anti-aging score {s} >= 65", r)
    show_score("Mature anti-aging", s, data.get("grade","?"))
    ALL.append(r); return r

def t19():
    r = R_("T19"); _hdr("T19","POST","/analyze/product","Empty concerns list — no crash")
    payload = {**OILY_ACNE, "concerns": []}
    resp, lat = _prod("Basic Moisturiser", ["Glycerin","Hyaluronic Acid"], payload); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200 with empty concerns", r)
    data = resp.json().get("data",{})
    chk(0 <= data.get("score",0) <= 100, "score in [0,100]", r)
    ALL.append(r); return r

def t20():
    r = R_("T20"); _hdr("T20","POST","/analyze/product","All 6 skin types — valid scores")
    for st in ["oily","dry","combination","normal","sensitive","mature"]:
        profile = {"skin_type":st,"concerns":["dryness"],"age_group":"adult","is_pregnant":False}
        resp, _ = call("post","/analyze/product", json={
            "product_name": f"Test ({st})",
            "ingredients": ["Niacinamide","Hyaluronic Acid","Glycerin"],
            "skin_profile": profile, "include_llm": False,
        })
        s = resp.json().get("data",{}).get("score",-1)
        chk(resp.status_code == 200 and 0 <= s <= 100, f"skin_type='{st}' score={s}", r)
    ALL.append(r); return r

def t21():
    r = R_("T21"); _hdr("T21","POST","/analyze/product","Invalid skin_type → 422")
    resp, lat = _prod("X", ["Niacinamide"],
                      {**OILY_ACNE,"skin_type":"cosmic"}); r.lat=lat
    chk(resp.status_code == 422, f"HTTP {resp.status_code} == 422", r)
    ALL.append(r); return r

def t22():
    r = R_("T22"); _hdr("T22","POST","/analyze/product","Invalid concern → 422")
    resp, lat = _prod("X", ["Niacinamide"],
                      {**OILY_ACNE,"concerns":["acne","telekinesis"]}); r.lat=lat
    chk(resp.status_code == 422, f"HTTP {resp.status_code} == 422", r)
    ALL.append(r); return r

def t23():
    r = R_("T23"); _hdr("T23","POST","/analyze/product","Empty ingredients → 422")
    resp, lat = _prod("X", [], OILY_ACNE); r.lat=lat
    chk(resp.status_code == 422, f"HTTP {resp.status_code} == 422", r)
    ALL.append(r); return r

def t24():
    r = R_("T24"); _hdr("T24","POST","/analyze/product","Response schema validation")
    resp, lat = _prod("Schema Test Serum",
                      ["Niacinamide","Hyaluronic Acid","Ceramide NP","Glycerin"],
                      OILY_ACNE); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json()
    envelope_ok(body, r)
    data = body.get("data",{})
    for k in ["product_name","score","grade","verdict","nlp","calc","llm_report","meta"]:
        chk(k in data, f"data.{k} present", r)
    calc = data.get("calc",{})
    for k in ["score","grade","verdict","pros","cons","warnings",
              "ingredient_details","not_found_in_db","rule_score","ml_score"]:
        chk(k in calc, f"calc.{k} present", r)
    nlp = data.get("nlp",{})
    for k in ["ingredients_received","ingredients_mapped","uncertain","details"]:
        chk(k in nlp, f"nlp.{k} present", r)
    meta = data.get("meta",{})
    chk("total_latency_ms" in meta, "meta.total_latency_ms present", r)
    chk("llm" in meta,              "meta.llm present", r)
    ALL.append(r); return r


# =============================================================================
# FEATURE 2 — LAYERING  (T25 – T35)
# =============================================================================

def _layer(an, ai, bn, bi, profile, tod="BOTH", llm=False):
    return call("post","/analyze/layering", json={
        "product_a_name": an, "product_a_ings": ai,
        "product_b_name": bn, "product_b_ings": bi,
        "skin_profile": profile, "time_of_day": tod,
        "include_llm": llm,
    })

def t25():
    r = R_("T25"); _hdr("T25","POST","/analyze/layering","Ideal pair — HA serum + ceramide cream")
    resp, lat = _layer(
        "HA Serum",       ["Hyaluronic Acid","Glycerin","Sodium PCA"],
        "Ceramide Cream", ["Ceramide NP","Ceramide AP","Cholesterol","Squalane"],
        DRY_MATURE, "BOTH", LLM); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json(); envelope_ok(body, r)
    data = body.get("data",{})
    s = score_ok(data, "score", r)
    chk(s >= 70, f"ideal layering score {s} >= 70", r)
    layer = data.get("layering",{})
    for k in ["layering_order","wait_time_minutes","application_steps",
              "pros","cons","warnings","pair_interactions"]:
        chk(k in layer, f"layering.{k} present", r)
    chk(isinstance(layer.get("application_steps",[]),list), "steps is list", r)
    show_score("Ideal layering", s, data.get("grade","?"))
    if LLM and data.get("llm_report"):
        _in(f"LLM headline: {data['llm_report'].get('headline','')}")
    ALL.append(r); return r

def t26():
    r = R_("T26"); _hdr("T26","POST","/analyze/layering","Classic conflict — VitC + retinol same PM")
    resp, lat = _layer(
        "Vitamin C Serum", ["Ascorbic Acid","Ferulic Acid","Glycerin"],
        "Retinol Night Cream", ["Retinol","Squalane","Ceramide NP"],
        DRY_MATURE, "PM", LLM); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s <= 55, f"VitC+retinol conflict score {s} <= 55", r)
    chk(len(data.get("layering",{}).get("cons",[])) >= 1, "cons list non-empty", r)
    show_score("VitC + retinol (conflict)", s, data.get("grade","?"))
    ALL.append(r); return r

def t27():
    r = R_("T27"); _hdr("T27","POST","/analyze/layering","Unsafe pregnancy layering")
    resp, lat = _layer(
        "Retinol Serum",  ["Retinol","Ceramide NP"],
        "Glycolic Toner", ["Glycolic Acid","Hyaluronic Acid"],
        PREGNANT, "PM", LLM); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s <= 35, f"unsafe pregnancy layering score {s} <= 35", r)
    warns = data.get("layering",{}).get("warnings",[])
    has_w = any("PREGNANCY" in w or "pregnan" in w.lower() for w in warns)
    chk(has_w, "pregnancy warning in layering.warnings", r)
    show_score("Pregnant + unsafe layering", s, data.get("grade","?"))
    ALL.append(r); return r

def t28():
    r = R_("T28"); _hdr("T28","POST","/analyze/layering","Safe pregnancy layering")
    resp, lat = _layer(
        "Niacinamide Serum",   ["Niacinamide","Tranexamic Acid","Hyaluronic Acid"],
        "Azelaic Moisturiser", ["Azelaic Acid","Glycerin","Allantoin","Ceramide NP"],
        PREGNANT, "BOTH", LLM); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s >= 60, f"safe pregnancy layering score {s} >= 60", r)
    show_score("Pregnant + safe layering", s, data.get("grade","?"))
    ALL.append(r); return r

def t29():
    r = R_("T29"); _hdr("T29","POST","/analyze/layering","AM routine — VitC serum + SPF")
    resp, lat = _layer(
        "Vitamin C Serum",       ["Ascorbic Acid","Ferulic Acid","Glycerin"],
        "Mineral SPF Moisturiser",["Zinc Oxide","Titanium Dioxide","Dimethicone"],
        OILY_ACNE, "AM", LLM); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s >= 60, f"VitC+SPF AM score {s} >= 60", r)
    chk(data.get("meta",{}).get("time_of_day") == "AM", "meta.time_of_day == AM", r)
    show_score("VitC + SPF (AM)", s, data.get("grade","?"))
    ALL.append(r); return r

def t30():
    r = R_("T30"); _hdr("T30","POST","/analyze/layering","PM routine — retinol + ceramide")
    resp, lat = _layer(
        "Retinol Serum",     ["Retinol","Squalane"],
        "Ceramide Night Cream",["Ceramide NP","Hyaluronic Acid","Panthenol","Allantoin"],
        DRY_MATURE, "PM", LLM); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s >= 65, f"retinol+ceramide PM score {s} >= 65", r)
    show_score("Retinol + Ceramide (PM)", s, data.get("grade","?"))
    ALL.append(r); return r

def t31():
    r = R_("T31"); _hdr("T31","POST","/analyze/layering","Teen oily acne — BHA + niacinamide")
    resp, lat = _layer(
        "BHA Toner",          ["Salicylic Acid","Zinc PCA","Aloe Barbadensis Leaf Juice"],
        "Niacinamide Moisturiser",["Niacinamide","Ceramide NP","Hyaluronic Acid","Glycerin"],
        TEEN, "AM", LLM); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s >= 65, f"teen BHA+niacinamide score {s} >= 65", r)
    show_score("Teen BHA + Niacinamide", s, data.get("grade","?"))
    ALL.append(r); return r

def t32():
    r = R_("T32"); _hdr("T32","POST","/analyze/layering","Wrong order should score lower than correct")
    resp_wrong, _ = _layer(
        "Petrolatum Balm (WRONG)", ["Petrolatum","Mineral Oil"],
        "HA Serum",                ["Hyaluronic Acid","Glycerin"],
        DRY_MATURE, "PM")
    resp_right, lat = _layer(
        "HA Serum",            ["Hyaluronic Acid","Glycerin"],
        "Petrolatum Balm",     ["Petrolatum","Mineral Oil"],
        DRY_MATURE, "PM"); r.lat=lat
    chk(resp_wrong.status_code == 200, "wrong order: HTTP 200", r)
    chk(resp_right.status_code == 200, "correct order: HTTP 200", r)
    sw = resp_wrong.json().get("data",{}).get("score",50)
    sc_ = resp_right.json().get("data",{}).get("score",50)
    chk(sc_ >= sw, f"correct ({sc_}) >= wrong ({sw})", r)
    _in(f"wrong={sw}  correct={sc_}")
    ALL.append(r); return r

def t33():
    r = R_("T33"); _hdr("T33","POST","/analyze/layering","Invalid time_of_day → 422")
    resp, lat = _layer("P1",["Niacinamide"],"P2",["Glycerin"],
                       OILY_ACNE, "MIDNIGHT"); r.lat=lat
    chk(resp.status_code == 422, f"HTTP {resp.status_code} == 422", r)
    ALL.append(r); return r

def t34():
    r = R_("T34"); _hdr("T34","POST","/analyze/layering","Missing product_b_ings → 422")
    resp, lat = call("post","/analyze/layering", json={
        "product_a_name":"P1","product_a_ings":["Niacinamide"],
        "product_b_name":"P2",
        "skin_profile": OILY_ACNE, "include_llm": False,
    }); r.lat=lat
    chk(resp.status_code == 422, f"HTTP {resp.status_code} == 422", r)
    ALL.append(r); return r

def t35():
    r = R_("T35"); _hdr("T35","POST","/analyze/layering","Response schema validation")
    resp, lat = _layer(
        "BHA Toner",     ["Salicylic Acid","Zinc PCA"],
        "Niacinamide S", ["Niacinamide","Hyaluronic Acid","Glycerin"],
        OILY_ACNE, "AM"); r.lat=lat
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json(); envelope_ok(body, r)
    data = body.get("data",{})
    for k in ["product_a_name","product_b_name","score","grade","verdict",
              "nlp","layering","llm_report","meta"]:
        chk(k in data, f"data.{k} present", r)
    layer = data.get("layering",{})
    for k in ["score","grade","verdict","layering_order","wait_time_minutes",
              "application_steps","pros","cons","warnings","pair_interactions",
              "rule_score","ml_score"]:
        chk(k in layer, f"layering.{k} present", r)
    nlp = data.get("nlp",{})
    for k in ["product_a","product_b"]:
        chk(k in nlp, f"nlp.{k} present", r)
        for kk in ["received","mapped","uncertain","details"]:
            chk(kk in nlp.get(k,{}), f"nlp.{k}.{kk} present", r)
    ALL.append(r); return r


# =============================================================================
# CEP TASK 6 SCENARIOS  (T36 – T44)
# =============================================================================

def t36():
    r = R_("T36"); _hdr("T36","POST","/analyze/product","CEP: oily acne + 5 ingredients score in [0,100]")
    resp, lat = _prod(
        "CEP Oily Acne Test",
        ["Niacinamide","Zinc PCA","Glycerin","Hyaluronic Acid","Panthenol"],
        OILY_ACNE,
        llm=False,
    ); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(0 <= s <= 100, f"score {s} in [0,100]", r)
    ALL.append(r); return r


def t37():
    r = R_("T37"); _hdr("T37","POST","/analyze/product","CEP: sensitive skin 15 ingredients no LLM < 500ms")
    ings = [
        "Aqua","Glycerin","Panthenol","Allantoin","Niacinamide","Ceramide NP","Ceramide AP",
        "Ceramide EOP","Cholesterol","Sodium Hyaluronate","Squalane","Centella Asiatica Extract",
        "Madecassoside","Betaine","Urea",
    ]
    resp, lat = _prod("CEP Sensitive 15", ings, SENSITIVE, llm=False); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    chk((lat * 1000) < 500, f"response time {(lat*1000):.1f}ms < 500ms", r)
    ALL.append(r); return r


def t38():
    r = R_("T38"); _hdr("T38","POST","/analyze/layering","CEP: Retinol + AHA low score and conflict")
    resp, lat = _layer(
        "Retinol Serum", ["Retinol","Squalane","Tocopherol"],
        "AHA Exfoliant", ["Glycolic Acid","Lactic Acid","Aqua"],
        SENSITIVE, "PM", False,
    ); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    data = resp.json().get("data",{})
    s = score_ok(data, "score", r)
    chk(s < 50, f"layering score {s} < 50", r)
    interactions = data.get("layering",{}).get("pair_interactions",[])
    has_conflict = any(i.get("interaction_type","").lower() in ("caution","conflicting","avoid") for i in interactions)
    chk(has_conflict, "conflict/caution detected", r)
    ALL.append(r); return r


def t39():
    r = R_("T39"); _hdr("T39","POST","/analyze/layering","CEP: Niacinamide + HA high score > 75")
    resp, lat = _layer(
        "Niacinamide Serum", ["Niacinamide","Panthenol","Zinc PCA"],
        "HA Serum", ["Hyaluronic Acid","Sodium Hyaluronate","Glycerin"],
        OILY_ACNE, "AM", False,
    ); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    s = score_ok(resp.json().get("data",{}), "score", r)
    chk(s > 75, f"layering score {s} > 75", r)
    ALL.append(r); return r


def t40():
    r = R_("T40"); _hdr("T40","POST","/ocr/extract","CEP: OCR clear label >= 3 ingredients")
    try:
        with open("testing/dry_moisturizer.jpg", "rb") as fh:
            t0 = time.perf_counter()
            resp = requests.post(f"{BASE}/ocr/extract", files={"file":("dry_moisturizer.jpg", fh, "image/jpeg")}, timeout=120)
            r.lat = time.perf_counter() - t0
    except FileNotFoundError:
        _fl("test image missing: testing/dry_moisturizer.jpg"); r.f += 1; ALL.append(r); return r
    chk(resp.status_code == 200, "HTTP 200", r)
    body = resp.json()
    ings = body.get("ingredients") or body.get("data",{}).get("ingredients",[])
    chk(isinstance(ings, list) and len(ings) >= 3, f"extracted >= 3 ingredients (got {len(ings) if isinstance(ings,list) else 0})", r)
    ALL.append(r); return r


def t41():
    r = R_("T41"); _hdr("T41","POST","/analyze/skin-type","CEP: oily face photo predicted class valid")
    try:
        with open("testing/oily-face.webp", "rb") as fh:
            t0 = time.perf_counter()
            resp = requests.post(f"{BASE}/analyze/skin-type", files={"file":("oily-face.webp", fh, "image/webp")}, timeout=120)
            r.lat = time.perf_counter() - t0
    except FileNotFoundError:
        _fl("test image missing: testing/oily-face.webp"); r.f += 1; ALL.append(r); return r
    chk(resp.status_code == 200, "HTTP 200", r)
    skin = resp.json().get("data",{}).get("skin_type","").lower()
    chk(skin in ["oily","dry","normal","combination","sensitive","mature"], f"predicted skin_type valid ({skin})", r)
    ALL.append(r); return r


def t42():
    r = R_("T42"); _hdr("T42","POST","/nlp/map/batch","CEP: 10 names resolution >= 80%")
    names = [
        "niacinamide","hyaluronic acid","retinol","vitamin c","salicylic acid",
        "ceramide","glycerin","azelaic acid","panthenol","allantoin",
    ]
    resp, lat = call("post","/nlp/map/batch", json={"ingredients": names}); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    mapped = resp.json().get("data",{}).get("mapped",[])
    resolved = sum(1 for m in mapped if m.get("confidence") in ("high","medium","low"))
    rate = (resolved / max(len(names),1)) * 100.0
    chk(rate >= 80.0, f"resolution rate {rate:.1f}% >= 80%", r)
    ALL.append(r); return r


def t43():
    r = R_("T43"); _hdr("T43","GET","/health","CEP: all model statuses are loaded")
    resp, lat = call("get","/health"); r.lat = lat
    chk(resp.status_code == 200, "HTTP 200", r)
    models = resp.json().get("models",{})
    all_ready = all(v == "ready" for v in models.values()) if models else False
    chk(all_ready, "all models == ready", r)
    ALL.append(r); return r


def t44():
    r = R_("T44"); _hdr("T44","POST","/analyze/product","CEP: stress 50 sequential requests, no failures, p95")
    payload = {
        "product_name": "CEP Stress Product",
        "ingredients": ["Niacinamide","Glycerin","Panthenol","Sodium Hyaluronate","Zinc PCA"],
        "skin_profile": OILY_ACNE,
        "include_llm": False,
    }
    lats_ms, failures = [], 0
    for _ in range(50):
        resp, lat = call("post","/analyze/product", json=payload)
        lats_ms.append(lat * 1000)
        if resp.status_code != 200:
            failures += 1
    r.lat = max(lats_ms)/1000 if lats_ms else 0
    lats_ms.sort()
    p95 = lats_ms[int(0.95 * len(lats_ms))] if lats_ms else 0
    chk(failures == 0, f"no failures in 50 requests (failures={failures})", r)
    _in(f"stress latency: avg={statistics.mean(lats_ms):.1f}ms  p95={p95:.1f}ms")
    ALL.append(r); return r


# =============================================================================
# MAIN
# =============================================================================

def main():
    global BASE, LLM
    parser = argparse.ArgumentParser(description="SkinSpectra API Mock Test Suite")
    parser.add_argument("--url",  default="http://localhost:8000")
    parser.add_argument("--llm",  action="store_true",
                        help="Enable LLM calls (requires GEMINI_API_KEY on server)")
    parser.add_argument("--only", nargs="*", metavar="TXX",
                        help="Run only these test IDs e.g. --only T13 T25")
    parser.add_argument("--skip", nargs="*", metavar="TXX",
                        help="Skip these test IDs e.g. --skip T11")
    parser.add_argument("--save", default="",
                        help="Save JSON summary to file")
    args   = parser.parse_args()
    BASE   = args.url.rstrip("/")
    LLM    = args.llm

    print(f"\n{B}{C}{'='*70}{X}")
    print(f"{B}{C}  SkinSpectra API — Mock Test Suite{X}")
    print(f"{B}{C}{'='*70}{X}")
    print(f"  Target : {BASE}")
    print(f"  LLM    : {'ON — Gemini 2.5 Flash' if LLM else 'OFF  (pass --llm to enable)'}")
    print(f"  Tests  : 44 across 9 endpoints\n")

    tests = [
        ("T01",t01),("T02",t02),("T03",t03),("T04",t04),("T05",t05),("T06",t06),
        ("T07",t07),("T08",t08),("T09",t09),("T10",t10),("T11",t11),("T12",t12),
        ("T13",t13),("T14",t14),("T15",t15),("T16",t16),("T17",t17),("T18",t18),
        ("T19",t19),("T20",t20),("T21",t21),("T22",t22),("T23",t23),("T24",t24),
        ("T25",t25),("T26",t26),("T27",t27),("T28",t28),("T29",t29),("T30",t30),
        ("T31",t31),("T32",t32),("T33",t33),("T34",t34),("T35",t35),
        ("T36",t36),("T37",t37),("T38",t38),("T39",t39),("T40",t40),
        ("T41",t41),("T42",t42),("T43",t43),("T44",t44),
    ]

    skip_set = set(t.upper() for t in (args.skip or []))
    only_set = set(t.upper() for t in (args.only or []))
    lats     = []

    for tid, fn in tests:
        if tid in skip_set:              _wn(f"Skipping {tid}"); continue
        if only_set and tid not in only_set: continue
        tr = fn()
        if tr.lat > 0: lats.append(tr.lat * 1000)

    # ── summary ───────────────────────────────────────────────────────
    tp = sum(t.p for t in ALL); tw = sum(t.w for t in ALL); tf = sum(t.f for t in ALL)
    grand = tp + tw + tf
    pct   = 100 * tp / grand if grand else 0
    oc    = G if pct >= 80 else (Y if pct >= 65 else R)

    print(f"\n{B}{C}{'─'*70}{X}")
    print(f"{B}  FINAL SUMMARY{X}")
    print(f"{'─'*70}")
    print(f"\n  {'Test':<6} {'Pass':>5} {'Warn':>5} {'Fail':>5}  {'%':>5}")
    print(f"  {'─'*35}")
    for t in ALL:
        col = G if t.pct >= 80 else (Y if t.pct >= 60 else R)
        print(f"  {t.tid:<6} {t.p:>5} {t.w:>5} {t.f:>5}  {col}{t.pct:>4.0f}%{X}")
    print(f"  {'─'*35}")
    print(f"  {'TOTAL':<6} {tp:>5} {tw:>5} {tf:>5}  {oc}{pct:>4.0f}%{X}\n")

    if lats:
        lats.sort()
        p50 = statistics.median(lats)
        p95 = lats[int(0.95*len(lats))]
        _in(f"API latency  p50={p50:.0f}ms  p95={p95:.0f}ms  (n={len(lats)}, excl. LLM)")

    print()
    if pct >= 80:   print(f"  {G}{B}API is production-ready.{X}\n")
    elif pct >= 65: print(f"  {Y}{B}API needs minor fixes — check server logs.{X}\n")
    else:           print(f"  {R}{B}API has significant issues — check server logs.{X}\n")

    if args.save:
        out = {
            "summary": {"passed":tp,"warned":tw,"failed":tf,"pct":round(pct,1)},
            "tests"  : [{"tid":t.tid,"p":t.p,"w":t.w,"f":t.f,"pct":round(t.pct,1)}
                        for t in ALL],
        }
        with open(args.save,"w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  Results saved → {args.save}")


if __name__ == "__main__":
    main()
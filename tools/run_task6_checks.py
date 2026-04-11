import json
import statistics
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "testing" / "test_results_summary.json"


def _post(path: str, payload: dict):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}{path}", json=payload, timeout=120)
    dt_ms = (time.perf_counter() - t0) * 1000
    return r, dt_ms


def _get(path: str):
    t0 = time.perf_counter()
    r = requests.get(f"{BASE}{path}", timeout=120)
    dt_ms = (time.perf_counter() - t0) * 1000
    return r, dt_ms


def main() -> None:
    results = []

    profile_oily = {
        "skin_type": "oily",
        "concerns": ["acne", "pores"],
        "age_group": "adult",
        "is_pregnant": False,
        "skin_sensitivity": "normal",
        "current_routine": "",
        "allergies": "",
        "location_climate": "humid tropical",
        "experience_level": "intermediate",
    }

    profile_sensitive = {
        "skin_type": "sensitive",
        "concerns": ["redness", "sensitivity", "barrier"],
        "age_group": "adult",
        "is_pregnant": False,
        "skin_sensitivity": "high",
        "current_routine": "",
        "allergies": "",
        "location_climate": "dry",
        "experience_level": "beginner",
    }

    # 1) Single product analysis — oily skin, acne concerns, 5 ingredients
    p1 = {
        "product_name": "Niacinamide Acne Serum",
        "ingredients": ["Niacinamide", "Zinc PCA", "Glycerin", "Hyaluronic Acid", "Panthenol"],
        "skin_profile": profile_oily,
        "include_llm": False,
    }
    r, dt = _post("/analyze/product", p1)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    score = body.get("data", {}).get("score")
    results.append({
        "scenario": "single_product_oily_acne_5_ingredients",
        "status_code": r.status_code,
        "passed": r.status_code == 200 and isinstance(score, (int, float)) and 0 <= score <= 100,
        "response_time_ms": round(dt, 2),
        "score": score,
    })

    # 2) Single product analysis — sensitive skin, 15 ingredients, no LLM, <500ms
    ing15 = [
        "Aqua", "Glycerin", "Panthenol", "Allantoin", "Niacinamide", "Ceramide NP", "Ceramide AP",
        "Ceramide EOP", "Cholesterol", "Sodium Hyaluronate", "Squalane", "Centella Asiatica Extract",
        "Madecassoside", "Betaine", "Urea",
    ]
    p2 = {
        "product_name": "Sensitive Barrier Repair Cream",
        "ingredients": ing15,
        "skin_profile": profile_sensitive,
        "include_llm": False,
    }
    r, dt = _post("/analyze/product", p2)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    results.append({
        "scenario": "single_product_sensitive_15_no_llm_under_500ms",
        "status_code": r.status_code,
        "passed": r.status_code == 200 and dt < 500,
        "response_time_ms": round(dt, 2),
        "score": body.get("data", {}).get("score"),
    })

    # 3) Layering analysis — Retinol + AHA low score and conflict
    l1 = {
        "product_a_name": "Retinol Serum",
        "product_b_name": "AHA Exfoliant",
        "product_a_ings": ["Retinol", "Squalane", "Tocopherol"],
        "product_b_ings": ["Glycolic Acid", "Lactic Acid", "Aqua"],
        "skin_profile": profile_sensitive,
        "time_of_day": "PM",
        "include_llm": False,
    }
    r, dt = _post("/analyze/layering", l1)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    lscore = body.get("data", {}).get("score")
    interactions = body.get("data", {}).get("layering", {}).get("pair_interactions", [])
    has_conflict = any((x.get("interaction_type", "").lower() in {"conflicting", "avoid", "caution"}) for x in interactions)
    results.append({
        "scenario": "layering_retinol_aha_conflict",
        "status_code": r.status_code,
        "passed": r.status_code == 200 and isinstance(lscore, (int, float)) and lscore < 50 and has_conflict,
        "response_time_ms": round(dt, 2),
        "score": lscore,
        "conflict_detected": has_conflict,
    })

    # 4) Layering analysis — Niacinamide + Hyaluronic Acid high score
    l2 = {
        "product_a_name": "Niacinamide Serum",
        "product_b_name": "HA Serum",
        "product_a_ings": ["Niacinamide", "Panthenol", "Zinc PCA"],
        "product_b_ings": ["Hyaluronic Acid", "Sodium Hyaluronate", "Glycerin"],
        "skin_profile": profile_oily,
        "time_of_day": "AM",
        "include_llm": False,
    }
    r, dt = _post("/analyze/layering", l2)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    lscore2 = body.get("data", {}).get("score")
    results.append({
        "scenario": "layering_niacinamide_hyaluronic_high",
        "status_code": r.status_code,
        "passed": r.status_code == 200 and isinstance(lscore2, (int, float)) and lscore2 > 75,
        "response_time_ms": round(dt, 2),
        "score": lscore2,
    })

    # 5) OCR clear label image -> >=3 ingredients
    img_path = ROOT / "testing" / "dry_moisturizer.jpg"
    with img_path.open("rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/ocr/extract", files=files, timeout=120)
        dt = (time.perf_counter() - t0) * 1000
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    ings = body.get("ingredients") or body.get("data", {}).get("ingredients", [])
    results.append({
        "scenario": "ocr_clear_label_extract_ge_3",
        "status_code": r.status_code,
        "passed": r.status_code == 200 and isinstance(ings, list) and len(ings) >= 3,
        "response_time_ms": round(dt, 2),
        "extracted_count": len(ings) if isinstance(ings, list) else 0,
    })

    # 6) Facial analysis — oily face photo -> valid class
    face_path = ROOT / "testing" / "oily-face.webp"
    with face_path.open("rb") as f:
        files = {"file": (face_path.name, f, "image/webp")}
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/analyze/skin-type", files=files, timeout=120)
        dt = (time.perf_counter() - t0) * 1000
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    pred = body.get("data", {}).get("skin_type") or body.get("skin_type")
    valid = {"oily", "dry", "normal", "combination", "sensitive", "mature"}
    results.append({
        "scenario": "facial_oily_photo_valid_class",
        "status_code": r.status_code,
        "passed": r.status_code == 200 and isinstance(pred, str) and pred.lower() in valid,
        "response_time_ms": round(dt, 2),
        "predicted_skin_type": pred,
    })

    # 7) NLP batch mapping — 10 common names -> >=80% resolved
    names = [
        "niacinamide", "hyaluronic acid", "retinol", "vitamin c", "salicylic acid",
        "ceramide", "glycerin", "azelaic acid", "panthenol", "allantoin",
    ]
    r, dt = _post("/nlp/map/batch", {"ingredients": names})
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    mapped = body.get("data", {}).get("mapped", [])
    resolved = 0
    for item in mapped:
        if item.get("confidence") in {"high", "medium", "low"}:
            resolved += 1
    rate = (resolved / 10.0) * 100.0
    results.append({
        "scenario": "nlp_batch_mapping_resolution_ge_80pct",
        "status_code": r.status_code,
        "passed": r.status_code == 200 and rate >= 80.0,
        "response_time_ms": round(dt, 2),
        "resolution_rate_percent": round(rate, 2),
    })

    # 8) Health endpoint models all loaded
    r, dt = _get("/health")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    models = body.get("models", {})
    loaded = all(v == "ready" for v in models.values()) if models else False
    results.append({
        "scenario": "health_models_all_loaded",
        "status_code": r.status_code,
        "passed": r.status_code == 200 and loaded,
        "response_time_ms": round(dt, 2),
        "models": models,
    })

    # 9) Stress test — 50 sequential single product requests; no failures; p95 latency
    stress_lat = []
    stress_fail = 0
    stress_payload = {
        "product_name": "Stress Test Product",
        "ingredients": ["Niacinamide", "Glycerin", "Panthenol", "Sodium Hyaluronate", "Zinc PCA"],
        "skin_profile": profile_oily,
        "include_llm": False,
    }
    for _ in range(50):
        r, dt = _post("/analyze/product", stress_payload)
        if r.status_code != 200:
            stress_fail += 1
        stress_lat.append(dt)
    p95 = statistics.quantiles(stress_lat, n=100)[94] if len(stress_lat) >= 2 else stress_lat[0]
    results.append({
        "scenario": "stress_50_sequential_product_requests",
        "status_code": 200 if stress_fail == 0 else 500,
        "passed": stress_fail == 0,
        "requests": 50,
        "failures": stress_fail,
        "p95_latency_ms": round(p95, 2),
        "avg_latency_ms": round(sum(stress_lat) / len(stress_lat), 2),
    })

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE,
        "overall_passed": all(item.get("passed", False) for item in results),
        "passed_count": sum(1 for item in results if item.get("passed", False)),
        "failed_count": sum(1 for item in results if not item.get("passed", False)),
        "scenarios": results,
    }

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "passed": summary["passed_count"], "failed": summary["failed_count"]}, indent=2))


if __name__ == "__main__":
    main()

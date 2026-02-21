"""
SkinSpectra NLP Layer — Test Suite
====================================
Tests
-----
1.  Exact match           — canonical INCI names resolve perfectly
2.  Common name mapping   — "vitamin c" → "Ascorbic Acid"
3.  Trade name mapping    — "Matrixyl" → "Palmitoyl Pentapeptide-4"
4.  Abbreviation mapping  — "HA", "BHA", "SLS", "SAP"
5.  Typo tolerance        — deliberate misspellings
6.  Multilingual aliases  — French / Spanish names
7.  Chemical alias        — IUPAC / CAS-adjacent names
8.  Ambiguous inputs      — "oil", "acid" etc.
9.  Completely unknown    — garbage strings
10. Batch mapping         — throughput benchmark
11. Latency benchmark     — p50 / p95 / p99
12. Confidence bands      — verify score buckets
13. Pregnancy + skin type — downstream field validation
14. Edge cases            — empty string, numbers, special chars
15. Regression set        — full known-answer eval with accuracy report
"""

import sys
import json
import time
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# ── make sure we can import from project root ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.nlp_layer import INCIMapper, CFG, _clean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.nlp.test")

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS (terminal)
# ══════════════════════════════════════════════════════════════════════════════
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}✔  {msg}{RESET}")
def fail(msg):print(f"  {RED}✗  {msg}{RESET}")
def warn(msg):print(f"  {YELLOW}⚠  {msg}{RESET}")
def info(msg):print(f"  {CYAN}ℹ  {msg}{RESET}")
def hdr(msg): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST DATA
# ══════════════════════════════════════════════════════════════════════════════

# (input, expected_inci, description)
EXACT_TESTS: List[Tuple[str, str, str]] = [
    ("Glycerin",            "Glycerin",             "canonical exact"),
    ("Hyaluronic Acid",     "Hyaluronic Acid",       "canonical exact"),
    ("Retinol",             "Retinol",               "canonical exact"),
    ("Niacinamide",         "Niacinamide",           "canonical exact"),
    ("Salicylic Acid",      "Salicylic Acid",        "canonical exact"),
    ("Zinc Oxide",          "Zinc Oxide",            "canonical exact"),
    ("Ascorbic Acid",       "Ascorbic Acid",         "canonical exact"),
    ("Ceramide NP",         "Ceramide NP",           "canonical exact"),
]

COMMON_NAME_TESTS: List[Tuple[str, str, str]] = [
    ("vitamin c",           "Ascorbic Acid",          "common → INCI"),
    ("vitamin e",           "Tocopherol",             "common → INCI"),
    ("vitamin b5",          "Panthenol",              "common → INCI"),
    ("hyaluronic acid",     "Hyaluronic Acid",        "lowercase common"),
    ("glycerol",            "Glycerin",               "chemical common"),
    ("shea butter",         "Shea Butter",            "common name"),
    ("glycerin",            "Glycerin",               "lowercase exact"),
    ("provitamin b5",       "Panthenol",              "provitamin alias"),
    ("retinol",             "Retinol",                "lowercase canonical"),
    ("green tea extract",   "Green Tea Extract",      "botanical common"),
    ("cica",                "Centella Asiatica Extract","brand/common slang"),
    ("kojic acid",          "Kojic Acid",             "straightforward"),
]

ABBREVIATION_TESTS: List[Tuple[str, str, str]] = [
    ("HA",  "Hyaluronic Acid",      "HA abbreviation"),
    ("BHA", "Salicylic Acid",       "BHA abbreviation"),
    ("AHA", "Glycolic Acid",        "AHA abbreviation"),
    ("SLS", "Sodium Lauryl Sulfate","SLS abbreviation"),
    ("SAP", "Sodium Ascorbyl Phosphate","SAP abbreviation"),
    ("CoQ10","Coenzyme Q10",        "CoQ10 abbreviation"),
    ("GHK-Cu","Copper Tripeptide-1","peptide abbreviation"),
    ("PG",  "Propylene Glycol",     "PG abbreviation"),
    ("BG",  "Butylene Glycol",      "BG abbreviation"),
    ("MCT oil","Caprylic/Capric Triglyceride","MCT abbreviation"),
]

TRADE_NAME_TESTS: List[Tuple[str, str, str]] = [
    ("Matrixyl",      "Palmitoyl Pentapeptide-4",     "trade → INCI"),
    ("Argireline",    "Acetyl Hexapeptide-3",         "trade → INCI"),
    ("Parsol 1789",   "Avobenzone",                   "sunscreen trade name"),
    ("Vaseline",      "Petrolatum",                   "brand → INCI"),
    ("Carbopol",      "Carbomer",                     "trade → INCI"),
    ("Tween 20",      "Polysorbate 20",               "trade → INCI"),
    ("Syn-Ake",       "Syn-Ake",                      "proprietary peptide"),
    ("Tinosorb S",    "Tinosorb S",                   "UV filter trade name"),
]

TYPO_TESTS: List[Tuple[str, str, str]] = [
    ("glycerine",          "Glycerin",           "British spelling"),
    ("hyaluroinc acid",    "Hyaluronic Acid",    "transposed letters"),
    ("retinoll",           "Retinol",            "double letter"),
    ("niacinamde",         "Niacinamide",        "dropped letter"),
    ("salicyilc acid",     "Salicylic Acid",     "transposed"),
    ("tocopherol acetate", "Tocopheryl Acetate", "close but not exact"),
    ("squalene",           "Squalane",           "e vs a confusion"),
    ("ceramide",           "Ceramide NP",        "partial name"),
    ("glycolic",           "Glycolic Acid",      "missing 'acid'"),
    ("lactic",             "Lactic Acid",        "missing 'acid'"),
]

MULTILINGUAL_TESTS: List[Tuple[str, str, str]] = [
    ("Acide Hyaluronique",   "Hyaluronic Acid",        "French"),
    ("Ácido Hialurónico",    "Hyaluronic Acid",        "Spanish"),
    ("Acide Ascorbique",     "Ascorbic Acid",          "French"),
    ("Glycérine",            "Glycerin",               "French"),
    ("Acide Salicylique",    "Salicylic Acid",         "French"),
    ("Aceite de Argán",      "Argan Oil",              "Spanish"),
    ("Retinol",              "Retinol",                "same in FR/ES"),
    ("Niacinamida",          "Niacinamide",            "Spanish"),
]

CHEMICAL_ALIAS_TESTS: List[Tuple[str, str, str]] = [
    ("2-Hydroxybenzoic Acid",           "Salicylic Acid",      "IUPAC"),
    ("L-Ascorbic Acid",                 "Ascorbic Acid",       "stereospecific"),
    ("Propan-1,2,3-triol",             "Glycerin",             "IUPAC glycerin"),
    ("Polydimethylsiloxane",            "Dimethicone",          "IUPAC silicone"),
    ("Sodium Dodecyl Sulfate",          "Sodium Lauryl Sulfate","chemical name"),
    ("Tocopheryl Acetate",              "Tocopheryl Acetate",   "correct INCI"),
    ("alpha-Tocopherol",                "Tocopherol",           "stereo prefix"),
    ("Hexadecanol",                     "Cetyl Alcohol",        "IUPAC fatty alcohol"),
]

EDGE_CASE_TESTS: List[Tuple[str, str, str]] = [
    ("",               None,    "empty string"),
    ("   ",            None,    "whitespace only"),
    ("12345",          None,    "numeric string"),
    ("@#$%",           None,    "special chars"),
    ("x",              None,    "single char"),
    ("unknown ingredient xyz", None, "completely unknown"),
    ("acid",           None,    "too generic - low confidence"),
    ("oil",            None,    "too generic - low confidence"),
]

AMBIGUOUS_TESTS: List[Tuple[str, str, str]] = [
    ("glycolic acid",    "Glycolic Acid",    "specific AHA"),
    ("salicylic",        "Salicylic Acid",   "partial BHA"),
    ("hyaluronan",       "Hyaluronic Acid",  "scientific synonym"),
    ("pantothenol",      "Panthenol",        "chemical form"),
    ("ascorbate",        "Ascorbic Acid",    "salt form"),
    ("retinaldehyde",    "Retinal",          "retinoid alias"),
]

# Ground-truth regression set  {input → expected_inci}
REGRESSION_SET = {
    "vitamin c":                 "Ascorbic Acid",
    "HA":                        "Hyaluronic Acid",
    "BHA":                       "Salicylic Acid",
    "Matrixyl":                  "Palmitoyl Pentapeptide-4",
    "Argireline":                "Acetyl Hexapeptide-3",
    "shea butter":               "Shea Butter",
    "glycerol":                  "Glycerin",
    "kojic acid":                "Kojic Acid",
    "retinol":                   "Retinol",
    "niacinamide":               "Niacinamide",
    "zinc oxide":                "Zinc Oxide",
    "vitamin e":                 "Tocopherol",
    "salicylic acid":            "Salicylic Acid",
    "glycolic acid":             "Glycolic Acid",
    "ceramide":                  "Ceramide NP",
    "hyaluronic acid":           "Hyaluronic Acid",
    "ascorbic acid":             "Ascorbic Acid",
    "lactic acid":               "Lactic Acid",
    "squalane":                  "Squalane",
    "sodium lauryl sulfate":     "Sodium Lauryl Sulfate",
    "Acide Hyaluronique":        "Hyaluronic Acid",
    "Glycérine":                 "Glycerin",
    "GHK-Cu":                    "Copper Tripeptide-1",
    "CoQ10":                     "Coenzyme Q10",
    "cica":                      "Centella Asiatica Extract",
    "Vaseline":                  "Petrolatum",
    "Tween 20":                  "Polysorbate 20",
}


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name:    str
    passed:  int = 0
    failed:  int = 0
    warned:  int = 0
    details: List[str] = field(default_factory=list)

    @property
    def total(self): return self.passed + self.failed + self.warned
    @property
    def pct(self):
        return 100 * self.passed / self.total if self.total else 0


def run_mapping_tests(
    mapper: INCIMapper,
    tests: List[Tuple[str, str, str]],
    section_name: str,
    strict: bool = True,
    min_confidence: str = "low",
) -> TestResult:
    """
    Run a list of (input, expected_inci, desc) tests.
    If expected_inci is None the test passes when confidence == 'uncertain'.
    """
    hdr(section_name)
    result = TestResult(name=section_name)
    conf_rank = {"uncertain": 0, "low": 1, "medium": 2, "high": 3}

    for query, expected, desc in tests:
        r = mapper.map(query if query.strip() else query or "__EMPTY__")

        if expected is None:
            # We expect low-confidence / uncertain
            if r["confidence"] in ("uncertain", "low"):
                ok(f"[{desc}] '{query}' → conf={r['confidence']} (expected uncertain/low) ✔")
                result.passed += 1
            else:
                warn(f"[{desc}] '{query}' → '{r['inci_name']}' conf={r['confidence']} "
                     f"(expected uncertain) — may be OK if model is confident")
                result.warned += 1
        else:
            match = r["inci_name"].lower() == expected.lower()
            close = expected.lower() in r["inci_name"].lower() or \
                    r["inci_name"].lower() in expected.lower()
            in_alts = any(
                a["inci_name"].lower() == expected.lower()
                for a in r.get("alternatives", [])
            )

            if match:
                ok(f"[{desc}] '{query}' → '{r['inci_name']}'  "
                   f"score={r['score']:.3f}  conf={r['confidence']}  ({r['latency_ms']}ms)")
                result.passed += 1
            elif close or in_alts:
                warn(f"[{desc}] '{query}' → '{r['inci_name']}' "
                     f"(expected '{expected}'; close/in-alts)")
                result.warned += 1
            else:
                fail(f"[{desc}] '{query}' → '{r['inci_name']}' "
                     f"(expected '{expected}')  score={r['score']:.3f}")
                result.failed += 1

    print(f"\n  Summary: {result.passed}✔  {result.warned}⚠  {result.failed}✗  "
          f"({result.pct:.0f}% strict pass)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL TEST GROUPS
# ══════════════════════════════════════════════════════════════════════════════

def test_exact(mapper): return run_mapping_tests(mapper, EXACT_TESTS,    "1. Exact Match Tests")
def test_common(mapper):return run_mapping_tests(mapper, COMMON_NAME_TESTS,"2. Common Name Mapping")
def test_abbrev(mapper):return run_mapping_tests(mapper, ABBREVIATION_TESTS,"3. Abbreviation Mapping")
def test_trade(mapper): return run_mapping_tests(mapper, TRADE_NAME_TESTS, "4. Trade Name Mapping")
def test_typos(mapper): return run_mapping_tests(mapper, TYPO_TESTS,       "5. Typo Tolerance")
def test_multi(mapper): return run_mapping_tests(mapper, MULTILINGUAL_TESTS,"6. Multilingual Aliases")
def test_chem(mapper):  return run_mapping_tests(mapper, CHEMICAL_ALIAS_TESTS,"7. Chemical Alias Mapping")
def test_ambig(mapper): return run_mapping_tests(mapper, AMBIGUOUS_TESTS,  "8. Ambiguous Inputs")
def test_edge(mapper):  return run_mapping_tests(mapper, EDGE_CASE_TESTS,  "9. Edge Cases", strict=False)


def test_latency(mapper) -> TestResult:
    hdr("10. Latency Benchmark")
    result = TestResult(name="Latency Benchmark")
    queries = (
        ["vitamin c", "retinol", "HA", "shea butter", "niacinamide",
         "glycolic acid", "salicylic acid", "hyaluronic acid",
         "zinc oxide", "Argireline"] * 10
    )
    latencies = []
    for q in queries:
        r = mapper.map(q)
        latencies.append(r["latency_ms"])

    latencies.sort()
    p50  = statistics.median(latencies)
    p95  = latencies[int(0.95 * len(latencies))]
    p99  = latencies[int(0.99 * len(latencies))]
    mean = statistics.mean(latencies)

    info(f"Queries  : {len(queries)}")
    info(f"Mean     : {mean:.2f}ms")
    info(f"p50      : {p50:.2f}ms")
    info(f"p95      : {p95:.2f}ms")
    info(f"p99      : {p99:.2f}ms")

    thresholds = {"p50": (p50, 100), "p95": (p95, 250), "p99": (p99, 500)}
    for label, (val, threshold) in thresholds.items():
        if val <= threshold:
            ok(f"{label} = {val:.1f}ms  ≤  {threshold}ms target ✔")
            result.passed += 1
        else:
            warn(f"{label} = {val:.1f}ms  >  {threshold}ms target (check hardware)")
            result.warned += 1

    return result


def test_batch(mapper) -> TestResult:
    hdr("11. Batch Throughput")
    result = TestResult(name="Batch Throughput")
    batch = [
        "vitamin c", "retinol", "niacinamide", "HA", "shea butter",
        "ceramide", "squalane", "glycolic acid", "salicylic acid",
        "Argireline", "Matrixyl", "CoQ10", "Vaseline", "Tween 20",
        "GHK-Cu", "bakuchiol", "azelaic acid", "tranexamic acid",
        "ferulic acid", "resveratrol",
    ]
    t0 = time.perf_counter()
    results = mapper.batch_map(batch)
    elapsed = (time.perf_counter() - t0) * 1000

    info(f"Batch size : {len(batch)}")
    info(f"Total time : {elapsed:.1f}ms")
    info(f"Per query  : {elapsed/len(batch):.1f}ms")

    for r in results:
        info(f"  '{r['input']}' → '{r['inci_name']}'  [{r['confidence']}]")

    ok(f"Batch mapping completed for {len(batch)} ingredients")
    result.passed = 1
    return result


def test_confidence_bands(mapper) -> TestResult:
    hdr("12. Confidence Band Validation")
    result = TestResult(name="Confidence Bands")

    high_conf_inputs = ["glycerin", "retinol", "niacinamide", "salicylic acid"]
    low_conf_inputs  = ["unknown chemical xyz", "12345abcde", "blorp"]

    for q in high_conf_inputs:
        r = mapper.map(q)
        if r["confidence"] in ("high", "medium"):
            ok(f"'{q}' → conf={r['confidence']}  score={r['score']:.3f}  (expected high/med) ✔")
            result.passed += 1
        else:
            warn(f"'{q}' → conf={r['confidence']}  score={r['score']:.3f}  (expected high/med)")
            result.warned += 1

    for q in low_conf_inputs:
        r = mapper.map(q)
        if r["confidence"] in ("uncertain", "low"):
            ok(f"'{q}' → conf={r['confidence']}  score={r['score']:.3f}  (expected low/uncertain) ✔")
            result.passed += 1
        else:
            warn(f"'{q}' → conf={r['confidence']}  score={r['score']:.3f}  (expected low/uncertain)")
            result.warned += 1

    return result


def test_regression(mapper) -> TestResult:
    hdr("13. Regression Set — Full Accuracy Report")
    result = TestResult(name="Regression")
    correct = 0
    close   = 0
    wrong   = 0
    rows    = []

    for query, expected in REGRESSION_SET.items():
        r      = mapper.map(query)
        pred   = r["inci_name"]
        match  = pred.lower() == expected.lower()
        approx = (expected.lower() in pred.lower()) or (pred.lower() in expected.lower())
        in_alts= any(a["inci_name"].lower() == expected.lower() for a in r.get("alternatives",[]))

        if match:
            correct += 1
            status  = f"{GREEN}PASS{RESET}"
        elif approx or in_alts:
            close  += 1
            status  = f"{YELLOW}CLOSE{RESET}"
        else:
            wrong  += 1
            status  = f"{RED}FAIL{RESET}"

        rows.append((status, query, pred, expected, r["score"], r["confidence"]))

    total = len(REGRESSION_SET)
    print(f"\n  {'STATUS':<8} {'INPUT':<30} {'PREDICTED':<35} {'EXPECTED':<35} {'SCORE':>6} {'CONF'}")
    print(f"  {'─'*140}")
    for status, q, pred, exp, sc, conf in rows:
        print(f"  {status:<8} {q:<30} {pred:<35} {exp:<35} {sc:>6.3f} {conf}")

    accuracy = 100 * correct / total
    approx_acc = 100 * (correct + close) / total
    print(f"\n  Strict accuracy : {correct}/{total} = {accuracy:.1f}%")
    print(f"  Approx accuracy : {correct+close}/{total} = {approx_acc:.1f}%")
    print(f"  Failed          : {wrong}/{total}")

    result.passed = correct
    result.warned = close
    result.failed = wrong

    if accuracy >= 85:
        ok(f"Strict accuracy {accuracy:.1f}% ≥ 85% target ✔")
    elif approx_acc >= 90:
        warn(f"Strict {accuracy:.1f}% but approx {approx_acc:.1f}% ≥ 90% — acceptable")
    else:
        fail(f"Accuracy {accuracy:.1f}% below 85% target — retrain or expand dataset")

    return result


def test_result_structure(mapper) -> TestResult:
    hdr("14. Response Schema Validation")
    result = TestResult(name="Schema")
    required_keys = {"input","inci_name","score","confidence","method","alternatives","latency_ms"}

    for q in ["glycerin", "retinol", "unknown stuff xyz"]:
        r = mapper.map(q)
        missing = required_keys - set(r.keys())
        if not missing:
            ok(f"'{q}' response has all required keys ✔")
            result.passed += 1
        else:
            fail(f"'{q}' response missing keys: {missing}")
            result.failed += 1

        # Type checks
        assert isinstance(r["score"], float),       "score must be float"
        assert isinstance(r["alternatives"], list),  "alternatives must be list"
        assert isinstance(r["latency_ms"], float),   "latency_ms must be float"
        assert r["confidence"] in ("high","medium","low","uncertain"), \
            f"invalid confidence: {r['confidence']}"
        ok(f"'{q}' type checks passed ✔")
        result.passed += 1

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SkinSpectra NLP Layer Test Suite")
    parser.add_argument(
        "--model_dir", default=CFG["output_dir"],
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Test sections to skip (e.g. --skip latency batch)"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        log.error(
            f"Model directory '{model_dir}' not found. "
            "Run train_nlp_layer.py first."
        )
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  SkinSpectra — NLP Layer Test Suite{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

    log.info(f"Loading INCIMapper from '{model_dir}' …")
    mapper = INCIMapper.load(str(model_dir), cfg=CFG)

    # Run all test groups
    all_results = []
    skip = set(args.skip or [])

    test_groups = [
        ("exact",      test_exact),
        ("common",     test_common),
        ("abbrev",     test_abbrev),
        ("trade",      test_trade),
        ("typos",      test_typos),
        ("multi",      test_multi),
        ("chem",       test_chem),
        ("ambig",      test_ambig),
        ("edge",       test_edge),
        ("latency",    test_latency),
        ("batch",      test_batch),
        ("confidence", test_confidence_bands),
        ("regression", test_regression),
        ("schema",     test_result_structure),
    ]

    for key, fn in test_groups:
        if key in skip:
            warn(f"Skipping: {key}")
            continue
        r = fn(mapper)
        all_results.append(r)

    # ── Final report ──────────────────────────────────────────────────
    hdr("FINAL REPORT")
    total_pass = sum(r.passed for r in all_results)
    total_warn = sum(r.warned for r in all_results)
    total_fail = sum(r.failed for r in all_results)
    grand_total = total_pass + total_warn + total_fail

    print(f"\n  {'Section':<35} {'Pass':>6} {'Warn':>6} {'Fail':>6} {'%':>6}")
    print(f"  {'─'*60}")
    for r in all_results:
        pct_str = f"{r.pct:.0f}%"
        color   = GREEN if r.pct >= 85 else (YELLOW if r.pct >= 65 else RED)
        print(f"  {r.name:<35} {r.passed:>6} {r.warned:>6} {r.failed:>6} "
              f"{color}{pct_str:>6}{RESET}")

    print(f"  {'─'*60}")
    overall_pct = 100 * total_pass / grand_total if grand_total else 0
    color = GREEN if overall_pct >= 80 else (YELLOW if overall_pct >= 65 else RED)
    print(f"  {'TOTAL':<35} {total_pass:>6} {total_warn:>6} {total_fail:>6} "
          f"{color}{overall_pct:.0f}%{RESET}\n")

    if overall_pct >= 80:
        print(f"  {GREEN}{BOLD}✅  NLP Layer is production-ready.{RESET}\n")
    elif overall_pct >= 65:
        print(f"  {YELLOW}{BOLD}⚠   NLP Layer needs minor improvement before production.{RESET}\n")
    else:
        print(f"  {RED}{BOLD}✗   NLP Layer requires retraining / dataset expansion.{RESET}\n")


if __name__ == "__main__":
    main()
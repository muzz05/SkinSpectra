import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.facial_analysis import FacialAnalyzer, CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.facial.test")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔  {msg}{RESET}")
def fail(msg): print(f"  {RED}✗  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ  {msg}{RESET}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{'─'*60}")

_ROOT = Path(__file__).resolve().parent.parent
_candidates = [
    Path(__file__).resolve().parent / "oily-face.webp",
    _ROOT / "oily-face.webp",
]
TEST_IMAGE = next((p for p in _candidates if p.exists()), _candidates[0])
VALID_SKIN_TYPES = {"Dry", "Normal", "Oily"}
EXPECTED_SKIN_TYPE = "Oily"

PASS = 0
FAIL = 0

def record(passed: bool, msg: str) -> None:
    global PASS, FAIL
    if passed:
        PASS += 1
        ok(msg)
    else:
        FAIL += 1
        fail(msg)


def load_analyzer() -> FacialAnalyzer:
    hdr("Loading FacialAnalyzer")
    t0       = time.perf_counter()
    analyzer = FacialAnalyzer.load()
    elapsed  = (time.perf_counter() - t0) * 1000
    info(f"Model loaded in {elapsed:.1f} ms")
    return analyzer


def test_image_exists() -> None:
    hdr("Test 1 — Image file exists")
    record(TEST_IMAGE.exists(), f"oily-face.webp found at {TEST_IMAGE}")


def test_predict_returns_result(analyzer: FacialAnalyzer) -> None:
    hdr("Test 2 — Prediction returns a result (not None)")
    result = analyzer.predict(str(TEST_IMAGE))
    record(result is not None, "predict() returned a non-None result")
    return result


def test_result_structure(result: dict) -> None:
    hdr("Test 3 — Result has required keys")
    required = {"skin_type", "confidence", "all_probabilities", "latency_ms"}
    for key in required:
        record(key in result, f"key '{key}' present in result")


def test_skin_type_valid(result: dict) -> None:
    hdr("Test 4 — skin_type is one of Dry / Normal / Oily")
    record(
        result["skin_type"] in VALID_SKIN_TYPES,
        f"skin_type='{result['skin_type']}' is valid",
    )


def test_confidence_range(result: dict) -> None:
    hdr("Test 5 — Confidence is in [0, 1]")
    c = result["confidence"]
    record(0.0 <= c <= 1.0, f"confidence={c:.4f} within [0, 1]")


def test_probabilities_sum(result: dict) -> None:
    hdr("Test 6 — all_probabilities sum to ~1.0")
    total = sum(result["all_probabilities"].values())
    record(abs(total - 1.0) < 0.01, f"probabilities sum = {total:.4f}")


def test_all_classes_present(result: dict) -> None:
    hdr("Test 7 — all_probabilities contains all 3 classes")
    probs = result["all_probabilities"]
    for skin_type in VALID_SKIN_TYPES:
        record(skin_type in probs, f"class '{skin_type}' present in probabilities")


def test_expected_skin_type(result: dict) -> None:
    hdr(f"Test 8 — Expected skin type is '{EXPECTED_SKIN_TYPE}'")
    predicted = result["skin_type"]
    record(
        predicted == EXPECTED_SKIN_TYPE,
        f"predicted='{predicted}' matches expected='{EXPECTED_SKIN_TYPE}'",
    )
    if predicted != EXPECTED_SKIN_TYPE:
        warn(f"Confidence breakdown: {result['all_probabilities']}")


def test_latency(result: dict) -> None:
    hdr("Test 9 — Prediction latency under 10 000 ms")
    ms = result["latency_ms"]
    record(ms < 10_000, f"latency={ms:.1f} ms")
    if ms > 3_000:
        warn("Latency above 3 s — consider GPU acceleration")


def test_nonexistent_image(analyzer: FacialAnalyzer) -> None:
    hdr("Test 10 — Non-existent image returns None")
    result = analyzer.predict("does_not_exist.png")
    record(result is None, "predict() returned None for missing file")


def test_display_result(result: dict) -> None:
    hdr("Test 11 — display_result runs without error")
    try:
        FacialAnalyzer.display_result(result)
        record(True, "display_result() completed without exception")
    except Exception as e:
        record(False, f"display_result() raised: {e}")


def test_display_result_none() -> None:
    hdr("Test 12 — display_result(None) runs without error")
    try:
        FacialAnalyzer.display_result(None)
        record(True, "display_result(None) completed without exception")
    except Exception as e:
        record(False, f"display_result(None) raised: {e}")


def main() -> None:
    print(f"\n{BOLD}{'═'*60}")
    print("  SkinSpectra — Facial Analysis Test Suite")
    print(f"{'═'*60}{RESET}")
    info(f"Test image : {TEST_IMAGE}")
    info(f"Model dir  : {CFG['model_dir']}")

    test_image_exists()

    analyzer = load_analyzer()

    result = test_predict_returns_result(analyzer)
    if result is None:
        fail("Cannot continue — prediction returned None (face not detected?)")
        print(f"\n{BOLD}{'═'*60}")
        print(f"  {RED}FAILED{RESET}{BOLD}  |  passed={PASS}  failed={FAIL}")
        print(f"{'═'*60}{RESET}\n")
        return

    test_result_structure(result)
    test_skin_type_valid(result)
    test_confidence_range(result)
    test_probabilities_sum(result)
    test_all_classes_present(result)
    test_expected_skin_type(result)
    test_latency(result)
    test_nonexistent_image(analyzer)
    test_display_result(result)
    test_display_result_none()

    colour = GREEN if FAIL == 0 else RED
    status = "PASSED" if FAIL == 0 else "FAILED"
    print(f"\n{BOLD}{'═'*60}")
    print(f"  {colour}{status}{RESET}{BOLD}  |  passed={PASS}  failed={FAIL}")
    print(f"{'═'*60}{RESET}\n")


if __name__ == "__main__":
    main()

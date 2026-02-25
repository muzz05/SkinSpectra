"""
SkinSpectra OCR Handler
========================
Converts ingredient label images into clean, parsed ingredient lists
ready to pass directly into /analyze/product or /analyze/layering.

Pipeline
--------
  Image input (JPG / PNG / WEBP / BMP / TIFF)
       ↓
  Pre-processing  (auto-deskew, denoise, binarise, upscale, contrast)
       ↓
  Tesseract OCR   (PSM 6 — uniform block of text, OEM 3 — LSTM engine)
       ↓
  Post-processing (OCR error correction, INCI character fixes)
       ↓
  Ingredient segmentation  (split on commas, semicolons, newlines)
       ↓
  Noise filtering  (remove headers, percentages, non-ingredient tokens)
       ↓
  Deduplication + normalisation
       ↓
  Output: { ingredients: [...], raw_text, confidence, warnings }

Usage
-----
    from ocr_handler import OCRHandler

    ocr = OCRHandler()
    result = ocr.extract_from_path("label.jpg")
    result = ocr.extract_from_bytes(image_bytes)
    result = ocr.extract_from_pil(pil_image)

    # result["ingredients"] → ["Niacinamide", "Zinc PCA", ...]
"""

import io
import re
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

log = logging.getLogger("skinspectra.ocr")

# =============================================================================
# CONFIGURATION
# =============================================================================

TESSERACT_CONFIG = "--psm 6 --oem 3"   # uniform block text + LSTM engine
MIN_INGREDIENT_LEN = 3                  # ignore tokens shorter than this
MAX_INGREDIENT_LEN = 80                 # ignore tokens longer than this
MAX_IMAGE_DIM      = 4096              # downscale giant images to this
MIN_IMAGE_DIM      = 800               # upscale tiny images to at least this

SUPPORTED_FORMATS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Common OCR character confusions specific to INCI ingredient names
OCR_CORRECTIONS = {
    # letter/digit confusions
    r"\b0il\b"           : "Oil",
    r"\bAcid\b"          : "Acid",         # already correct but catch case
    r"(?<=[A-Za-z])0"    : "o",            # letter-O read as zero mid-word
    r"(?<=[A-Za-z])1"    : "l",            # letter-l read as one mid-word
    r"\bSod1um\b"        : "Sodium",
    r"\bGlycenn\b"       : "Glycerin",
    r"\bNiacinam1de\b"   : "Niacinamide",
    r"\bCholestero1\b"   : "Cholesterol",
    r"\bSod[il1]um\b"    : "Sodium",
    r"\bPanthen[o0]l\b"  : "Panthenol",
    r"\bAllant[o0][il1]n\b": "Allantoin",
    r"\bCarb[o0]mer\b"   : "Carbomer",
    r"\bDimeth[il1]c[o0]ne\b": "Dimethicone",
    r"\bEDTA\b"          : "EDTA",         # often split
    r"\bPOA\b"           : "PCA",          # C→O confusion
    r"\bNlacin\b"        : "Niacin",
    r"\bGIycerin\b"      : "Glycerin",
    r"\bGiycer\b"        : "Glycer",       # partial match
    r"\bAqua\b"          : "Aqua",
    r"\bVite\b"          : "Vitamin E",
    r"\bVitB\b"          : "Vitamin B",
    r"\bVit\. E\b"       : "Vitamin E",
    r"\bVit\. C\b"       : "Vitamin C",
    # spacing issues
    r"([a-z])([A-Z][a-z])" : r"\1 \2",     # camelCase split e.g. CaprylylGlycol
}

# Patterns that mark lines/tokens as NON-ingredient noise to discard
NOISE_PATTERNS = [
    r"^INGREDIENTS?[\s:/]*$",          # header row
    r"^INGREDIENTS?\s*/\s*INCI.*",     # "INGREDIENTS / INCI" header
    r"^INCI[\s:/]*$",
    r"^Ingredients\s*/\s*Inci.*",      # title-cased variant
    r"^COMPOSITION[\s:/]*$",
    r"^FULL\s+INGREDIENTS?.*",
    r"^CONTENTS?[\s:/]*$",
    r"May\s+Contain",
    r"^\+\/-",
    r"^[\(\[\{]?\+\/\-[\)\]\}]?",
    r"CI\s*\d{5,}",                    # CI colour codes (CI 77891)
    r"^\d+[\.,]\d*\s*%",               # concentrations: 10.0%
    r"^[\d\s\.]+%$",
    r"^[\*†‡§¶•◦▪▸\-–—]+$",           # stray bullet/dash lines
    r"^\s*[\(\[\{][\)\]\}]\s*$",       # empty brackets
    r"^(www\.|http|@)",                # URLs / handles
    r"Lot\s*No",
    r"Batch\s*No",
    r"EXP\.?\s*DATE",
    r"Net\s+Wt",
    r"Mfd\.\s*By",
    r"Dist\.\s*By",
    r"Made\s+in",
    r"^\d{4,}$",                       # bare numbers (batch codes)
    r"^[A-Z]{1,3}\d{4,}",             # product codes like AB12345
    r"^\s*$",
]

# Words that are definitely not ingredients
NON_INGREDIENT_WORDS = {
    "ingredients", "inci", "contains", "composition", "contents",
    "full", "list", "formula", "formulation", "include", "including",
    "disclaimer", "warning", "caution", "note", "notes",
    "contact", "company", "manufactured", "distributed", "imported",
    "www", "com", "net", "org",
}

# =============================================================================
# IMAGE PRE-PROCESSOR
# =============================================================================

class ImagePreprocessor:
    """
    Multi-stage image pre-processing to maximise OCR accuracy on
    real-world ingredient label photos (curved, low-light, blurry, skewed).
    """

    @staticmethod
    def load(source) -> np.ndarray:
        """Accept file path, bytes, or PIL Image → BGR numpy array."""
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise ValueError(f"Could not read image: {source}")
            return img
        if isinstance(source, (bytes, bytearray)):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")
            return img
        if isinstance(source, Image.Image):
            return cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
        raise TypeError(f"Unsupported source type: {type(source)}")

    @staticmethod
    def resize(img: np.ndarray) -> np.ndarray:
        """Scale image into the sweet spot for Tesseract (800–4096px wide)."""
        h, w = img.shape[:2]
        if max(h, w) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_AREA)
        if max(h, w) < MIN_IMAGE_DIM:
            scale = MIN_IMAGE_DIM / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """
        Detect and correct image skew up to ±20 degrees.
        Uses Hough line transform on edge map.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                                 minLineLength=60, maxLineGap=8)
        if lines is None or len(lines) < 3:
            return img

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -20 < angle < 20:        # ignore near-vertical lines
                    angles.append(angle)

        if not angles:
            return img

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:         # negligible skew
            return img

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        log.debug(f"Deskewed by {median_angle:.2f}°")
        return rotated

    @staticmethod
    def denoise(img: np.ndarray) -> np.ndarray:
        """Remove salt-and-pepper noise while preserving text edges."""
        return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

    @staticmethod
    def binarise(gray: np.ndarray) -> np.ndarray:
        """
        Adaptive + Otsu binarisation — handles uneven lighting.
        Returns the better of the two methods by estimating text density.
        """
        # Otsu (global threshold)
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive Gaussian (local threshold — better for shadows/gradients)
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )

        # Pick whichever has more black pixels in the expected text density range
        otsu_density    = np.sum(otsu == 0)   / otsu.size
        adaptive_density= np.sum(adaptive==0) / adaptive.size

        # Typical printed text occupies 5–35% of the image
        def score(d): return abs(d - 0.15)      # ideal ≈ 15% black
        return otsu if score(otsu_density) <= score(adaptive_density) else adaptive

    @staticmethod
    def sharpen(pil_img: Image.Image) -> Image.Image:
        """Sharpen + boost contrast for soft/blurry label photos."""
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.5)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        return pil_img

    def process(self, source) -> Tuple[Image.Image, Dict]:
        """
        Full pre-processing pipeline.
        Returns (pil_image_for_tesseract, meta_dict).
        """
        t0  = time.perf_counter()
        img = self.load(source)

        # 1. Resize
        img = self.resize(img)
        original_size = img.shape[:2]

        # 2. Deskew
        img = self.deskew(img)

        # 3. Denoise (light denoise — skip heavy denoising on sharp images)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        noise_est = float(np.std(lab[:,:,0]))
        if noise_est > 12:
            img = self.denoise(img)

        # 4. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 5. Binarise
        binary = self.binarise(gray)

        # 6. Morphological cleanup — remove tiny specks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 7. To PIL for tesseract
        pil_img = Image.fromarray(binary)
        pil_img = self.sharpen(pil_img)

        latency = round((time.perf_counter() - t0) * 1000, 1)
        meta = {
            "original_size" : original_size,
            "noise_estimate": round(noise_est, 1),
            "preprocess_ms" : latency,
        }
        return pil_img, meta


# =============================================================================
# TEXT POST-PROCESSOR
# =============================================================================

class TextPostprocessor:
    """
    Cleans raw Tesseract output → list of ingredient name strings.
    """

    def __init__(self):
        self._noise_re = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

    # ── OCR error correction ───────────────────────────────────────────────
    @staticmethod
    def correct_ocr_errors(text: str) -> str:
        for pattern, replacement in OCR_CORRECTIONS.items():
            text = re.sub(pattern, replacement, text)
        return text

    # ── Split raw OCR text into candidate ingredient tokens ────────────────
    @staticmethod
    def segment(text: str) -> List[str]:
        """
        Split ingredient text on commas, semicolons, and newlines.
        Preserves parenthetical synonyms: Glycerin (Vegetable) stays together.
        """
        # Normalise whitespace
        text = re.sub(r"[ \t]+", " ", text)

        # Join hyphenated line-breaks (common in narrow labels)
        text = re.sub(r"-\s*\n\s*", "", text)

        # Replace semicolons with commas (some labels use semicolons)
        text = text.replace(";", ",")

        # Split on comma or newline — but not inside parentheses
        tokens = []
        depth  = 0
        buf    = []
        for ch in text:
            if ch in "([":
                depth += 1; buf.append(ch)
            elif ch in ")]":
                depth -= 1; buf.append(ch)
            elif ch in ",\n" and depth == 0:
                t = "".join(buf).strip()
                if t: tokens.append(t)
                buf = []
            else:
                buf.append(ch)
        if buf:
            t = "".join(buf).strip()
            if t: tokens.append(t)
        return tokens

    # ── Filter noise tokens ────────────────────────────────────────────────
    def is_noise(self, token: str) -> bool:
        """Return True if the token is not an ingredient."""
        t = token.strip()
        if len(t) < MIN_INGREDIENT_LEN or len(t) > MAX_INGREDIENT_LEN:
            return True
        if any(r.search(t) for r in self._noise_re):
            return True
        # Pure numbers or punctuation
        if re.match(r"^[\d\s\.\,\-\+\/\*\#]+$", t):
            return True
        # Known non-ingredient words
        if t.lower().rstrip(".,:/;") in NON_INGREDIENT_WORDS:
            return True
        # No letters at all
        if not re.search(r"[A-Za-z]", t):
            return True
        return False

    # ── Normalise a single ingredient name ────────────────────────────────
    @staticmethod
    def normalise(token: str) -> str:
        """
        Strip leading noise characters, normalise spacing and casing.
        Preserves parenthetical INCI synonyms.
        """
        # Strip leading/trailing punctuation (not inside parens)
        t = re.sub(r"^[\s\*†‡•◦▪▸\-–—:;,\.]+", "", token)
        t = re.sub(r"[\s\*†‡•◦▪▸\-–—:;,\.]+$", "", t)
        t = t.strip()

        # Strip leading "Ingredients:" / "INGREDIENTS:" / "INCI:" header prefix
        # that OCR sometimes fuses with the first ingredient on the same line
        t = re.sub(r"^.*?INGREDIENTS?\s*[:/]\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^INCI\s*[:/]\s*", "", t, flags=re.IGNORECASE)

        # Strip leading numbered list markers: "1.", "1)", "10.", etc.
        t = re.sub(r"^\d+[\.\)]\s*", "", t)

        # Strip trailing parentheticals that contain product/lot/batch codes
        # e.g.  "(Code F..L. D213778/1]"  or  "[Lot: AB1234]"
        t = re.sub(
            r"[\(\[]\s*(?:Code|Lot|Batch|Ref|Cat|Item|SKU|EAN|UPC|Reg)[\s\.:/].*?[\)\]\.]*$",
            "", t, flags=re.IGNORECASE,
        ).strip()

        # Strip trailing stray brackets/dots left after above cleanup
        t = re.sub(r"[\s\.\(\[\{]+$", "", t)

        # Title-case only if ALL CAPS (OCR sometimes returns all-caps)
        if t.isupper() and len(t) > 3:
            t = t.title()

        # Normalise internal spaces
        t = re.sub(r"\s{2,}", " ", t)
        return t

    # ── Expand slash-separated dual names into individual ingredients ──────
    def expand_slashes(self, items: List[str]) -> List[str]:
        """
        Split tokens that contain a slash into separate entries.
        e.g. "Aqua/Water"              → ["Aqua", "Water"]
             "Caprylic/Capric Triglyceride" → ["Caprylic", "Capric Triglyceride"]
        Each part is re-normalised and noise-checked before being kept.
        """
        expanded: List[str] = []
        for tok in items:
            if "/" not in tok:
                expanded.append(tok)
                continue
            parts = [self.normalise(p.strip()) for p in tok.split("/")]
            valid = [p for p in parts if p and not self.is_noise(p)]
            if valid:
                expanded.extend(valid)
            else:
                # fall back to original token if all parts are noise
                expanded.append(tok)
        return expanded

    # ── Deduplicate while preserving order ────────────────────────────────
    @staticmethod
    def deduplicate(items: List[str]) -> List[str]:
        seen   = set()
        result = []
        for item in items:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    # ── Main pipeline ─────────────────────────────────────────────────────
    def process(self, raw_text: str) -> Tuple[List[str], List[str]]:
        """
        Returns (ingredient_list, warnings_list).
        """
        warnings: List[str] = []

        # 1. OCR correction
        corrected = self.correct_ocr_errors(raw_text)

        # 2. Segment into tokens
        tokens = self.segment(corrected)

        if not tokens:
            warnings.append("No ingredient tokens found after segmentation.")
            return [], warnings

        # 3. Filter + normalise
        kept     = []
        discarded= []
        for tok in tokens:
            norm = self.normalise(tok)
            if not norm:
                continue
            if self.is_noise(norm):
                discarded.append(norm)
            else:
                kept.append(norm)

        log.debug(f"Segmented {len(tokens)} tokens | kept {len(kept)} | "
                  f"discarded {len(discarded)}")

        # 4. Expand slash-separated names (e.g. "Aqua/Water" → "Aqua", "Water")
        kept = self.expand_slashes(kept)

        # 5. Deduplicate
        ingredients = self.deduplicate(kept)

        # 6. Warnings
        if len(ingredients) == 0:
            warnings.append(
                "No valid ingredients extracted. "
                "Check image quality — ensure the ingredient list is clearly visible."
            )
        elif len(ingredients) < 3:
            warnings.append(
                f"Only {len(ingredients)} ingredient(s) detected. "
                "Image may be partially cropped or too low-resolution."
            )

        if len(discarded) > len(kept):
            warnings.append(
                f"High noise ratio: {len(discarded)} tokens discarded vs {len(kept)} kept. "
                "Consider using a higher-quality or better-lit image."
            )

        return ingredients, warnings


# =============================================================================
# OCR CONFIDENCE ESTIMATOR
# =============================================================================

def estimate_confidence(raw_text: str, ingredients: List[str]) -> str:
    """
    Simple heuristic confidence rating based on:
    - OCR character confidence data
    - Number of recognisable INCI-like tokens
    - Noise level in raw text
    """
    if not raw_text.strip():
        return "very_low"

    word_count  = len(raw_text.split())
    ing_count   = len(ingredients)
    noise_chars = len(re.findall(r"[|\\@#$^&*~`<>{}]", raw_text))
    noise_ratio = noise_chars / max(len(raw_text), 1)

    # Try to get per-character confidence from tesseract data
    try:
        data = pytesseract.image_to_data(
            pytesseract.Output.DICT
            if hasattr(pytesseract.Output, "DICT")
            else None,
        )
    except Exception:
        data = None

    # Simple heuristic scoring
    if ing_count == 0:
        return "very_low"
    if noise_ratio > 0.05:
        return "low"
    if ing_count >= 5 and word_count >= 15 and noise_ratio < 0.01:
        return "high"
    if ing_count >= 3:
        return "medium"
    return "low"


# =============================================================================
# OCR HANDLER — MAIN CLASS
# =============================================================================

class OCRHandler:
    """
    Main interface for SkinSpectra OCR.

    Usage
    -----
        ocr = OCRHandler()

        # From file path
        result = ocr.extract_from_path("label.jpg")

        # From bytes (e.g. uploaded via API)
        result = ocr.extract_from_bytes(image_bytes, filename="label.jpg")

        # From PIL Image
        result = ocr.extract_from_pil(pil_image)

    Result shape
    ------------
    {
      "success"       : bool,
      "ingredients"   : ["Niacinamide", "Zinc PCA", ...],
      "ingredient_count": int,
      "raw_text"      : str,
      "corrected_text": str,
      "confidence"    : "high" | "medium" | "low" | "very_low",
      "warnings"      : [str, ...],
      "meta"          : {
          "preprocess_ms"  : float,
          "ocr_ms"         : float,
          "total_ms"       : float,
          "original_size"  : [h, w],
          "noise_estimate" : float,
          "tesseract_config": str,
      },
      "error"         : str | None,
    }
    """

    # Common Windows install locations for Tesseract
    _WIN_TESSERACT_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\muzam\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]

    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif sys.platform == "win32":
            for path in self._WIN_TESSERACT_PATHS:
                if Path(path).is_file():
                    pytesseract.pytesseract.tesseract_cmd = path
                    log.info(f"Tesseract found at: {path}")
                    break
        self.preprocessor   = ImagePreprocessor()
        self.postprocessor  = TextPostprocessor()
        log.info(
            f"OCRHandler ready | tesseract={pytesseract.get_tesseract_version()}"
        )

    # ── Core extraction ───────────────────────────────────────────────────
    def _extract(self, source, filename: str = "") -> Dict:
        t0       = time.perf_counter()
        warnings = []

        # 1. Pre-process image
        try:
            pil_img, pre_meta = self.preprocessor.process(source)
        except Exception as e:
            log.error(f"Pre-processing failed: {e}")
            return self._error(f"Image pre-processing failed: {e}", t0)

        # 2. Run Tesseract OCR — try original + processed images with multiple PSM modes
        # Load original as PIL for fallback (often better than heavy binarisation)
        try:
            if isinstance(source, (str, Path)):
                pil_orig = Image.open(str(source)).convert("RGB")
            elif isinstance(source, (bytes, bytearray)):
                pil_orig = Image.open(io.BytesIO(source)).convert("RGB")
            elif isinstance(source, Image.Image):
                pil_orig = source.convert("RGB")
            else:
                pil_orig = pil_img
        except Exception:
            pil_orig = pil_img

        raw_text     = ""
        ocr_latency  = 0.0
        best_score   = -1.0
        t_ocr_start  = time.perf_counter()

        def _quality_score(text: str) -> float:
            """
            Score OCR output quality.
            Rewards: long alpha words typical of INCI names, comma-separated structure.
            Penalises: digit/symbol noise, very short garbled tokens.
            """
            if not text.strip():
                return -1.0
            lines   = text.strip().splitlines()
            words   = text.split()
            if not words:
                return -1.0

            # Ratio of words that look like real ingredient words
            # (>=4 chars, mostly letters, starts with letter)
            good_words = sum(
                1 for w in words
                if len(w) >= 4
                and w[0].isalpha()
                and sum(c.isalpha() for c in w) / len(w) >= 0.75
            )
            good_ratio = good_words / max(len(words), 1)

            # Comma count as proxy for ingredient list structure
            comma_score = min(text.count(",") / 15.0, 1.0)

            # Penalise noise symbols heavily
            noise_chars = sum(1 for c in text if c in r"|\@#$^*~`<>{}0123456789_")
            noise_ratio = noise_chars / max(len(text), 1)

            return good_ratio * 0.6 + comma_score * 0.3 - noise_ratio * 0.5

        # Try original image FIRST (almost always better than binarised for real photos)
        # PSM 3 = fully automatic page layout, PSM 6 = assume uniform text block
        for img_candidate, psm in [
            (pil_orig, 3),   # auto-layout on original — most robust for real photos
            (pil_orig, 6),   # uniform block on original
            (pil_img,  6),   # uniform block on binarised (fallback for very noisy images)
            (pil_img,  4),   # single column on binarised
        ]:
            cfg = f"--psm {psm} --oem 3"
            try:
                candidate = pytesseract.image_to_string(img_candidate, lang="eng", config=cfg)
                score     = _quality_score(candidate)
                if score > best_score:
                    best_score = score
                    raw_text   = candidate
            except Exception as e:
                warnings.append(f"OCR PSM {psm} failed: {e}")

        ocr_latency = round((time.perf_counter() - t_ocr_start) * 1000, 1)

        if not raw_text.strip():
            return self._error(
                "Tesseract returned empty output. "
                "Ensure the image is clear, well-lit, and contains readable text.",
                t0, pre_meta,
            )

        # 3. Post-process → ingredient list
        corrected_text             = TextPostprocessor.correct_ocr_errors(raw_text)
        ingredients, post_warnings = self.postprocessor.process(raw_text)
        warnings.extend(post_warnings)

        # 4. Confidence estimate
        confidence = estimate_confidence(raw_text, ingredients)

        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info(
            f"OCR complete | file='{filename}' "
            f"ingredients={len(ingredients)} confidence={confidence} "
            f"total={total_ms}ms"
        )

        return {
            "success"          : True,
            "ingredients"      : ingredients,
            "ingredient_count" : len(ingredients),
            "raw_text"         : raw_text,
            "corrected_text"   : corrected_text,
            "confidence"       : confidence,
            "warnings"         : warnings,
            "meta"             : {
                "preprocess_ms"   : pre_meta.get("preprocess_ms", 0),
                "ocr_ms"          : ocr_latency,
                "total_ms"        : total_ms,
                "original_size"   : list(pre_meta.get("original_size", [])),
                "noise_estimate"  : pre_meta.get("noise_estimate", 0),
                "tesseract_config": TESSERACT_CONFIG,
                "filename"        : filename,
            },
            "error" : None,
        }

    # ── Public methods ────────────────────────────────────────────────────
    def extract_from_path(self, path: str) -> Dict:
        """Extract ingredients from an image file path."""
        p = Path(path)
        if not p.exists():
            return self._error(f"File not found: {path}")
        if p.suffix.lower() not in SUPPORTED_FORMATS:
            return self._error(
                f"Unsupported format '{p.suffix}'. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        return self._extract(path, filename=p.name)

    def extract_from_bytes(self, image_bytes: bytes, filename: str = "upload") -> Dict:
        """Extract ingredients from raw image bytes (e.g. HTTP upload)."""
        if not image_bytes:
            return self._error("Empty image bytes received")
        ext = Path(filename).suffix.lower()
        if ext and ext not in SUPPORTED_FORMATS:
            return self._error(
                f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        return self._extract(bytes(image_bytes), filename=filename)

    def extract_from_pil(self, pil_image: Image.Image, filename: str = "image") -> Dict:
        """Extract ingredients from a PIL Image object."""
        return self._extract(pil_image, filename=filename)

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _error(msg: str, t0: float = None, pre_meta: Dict = None) -> Dict:
        total = round((time.perf_counter() - t0) * 1000, 1) if t0 else 0.0
        return {
            "success"          : False,
            "ingredients"      : [],
            "ingredient_count" : 0,
            "raw_text"         : "",
            "corrected_text"   : "",
            "confidence"       : "very_low",
            "warnings"         : [],
            "meta"             : {
                "preprocess_ms"  : pre_meta.get("preprocess_ms", 0) if pre_meta else 0,
                "ocr_ms"         : 0,
                "total_ms"       : total,
                "original_size"  : list(pre_meta.get("original_size", [])) if pre_meta else [],
                "noise_estimate" : 0,
                "tesseract_config": TESSERACT_CONFIG,
                "filename"       : "",
            },
            "error" : msg,
        }

    @staticmethod
    def supported_formats() -> List[str]:
        return sorted(SUPPORTED_FORMATS)

    @staticmethod
    def tesseract_version() -> str:
        try:
            return str(pytesseract.get_tesseract_version())
        except Exception:
            return "unknown"


# =============================================================================
# FASTAPI ROUTER  (mounted into api.py)
# =============================================================================

def make_ocr_router():
    """
    Returns a FastAPI APIRouter with OCR endpoints.
    Mount in api.py with: app.include_router(make_ocr_router())
    """
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    router  = APIRouter(prefix="/ocr", tags=["OCR — Ingredient Extraction"])
    handler = OCRHandler()

    class OCRResponse:
        """Helper to build consistent response dicts."""
        @staticmethod
        def build(result: Dict, status: int = 200):
            return JSONResponse(content=result, status_code=status)

    @router.post(
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
    async def extract_ingredients(
        file    : UploadFile = File(..., description="Ingredient label image"),
        debug   : bool       = Form(False, description="Include raw OCR text in response"),
    ):
        # Validate content type
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

        result = handler.extract_from_bytes(image_bytes, filename=file.filename or "upload")

        # Strip raw_text if debug=False to keep response clean
        if not debug:
            result.pop("raw_text", None)
            result.pop("corrected_text", None)

        status_code = 200 if result["success"] else 422
        return JSONResponse(content=result, status_code=status_code)

    @router.get(
        "/info",
        summary="OCR engine info — version and supported formats",
    )
    async def ocr_info():
        return {
            "tesseract_version": handler.tesseract_version(),
            "supported_formats": handler.supported_formats(),
            "config"           : TESSERACT_CONFIG,
            "description"      : (
                "Tesseract 5 LSTM engine with adaptive binarisation, "
                "auto-deskew, and INCI-aware post-processing"
            ),
        }

    return router


# =============================================================================
# STANDALONE DEMO
# =============================================================================

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="SkinSpectra OCR Handler")
    parser.add_argument("image", nargs="?", default="../testing/dry_moisturizer.jpg",
                        help="Path to ingredient label image")
    parser.add_argument("--debug", action="store_true",
                        help="Show raw OCR text")
    args = parser.parse_args()

    ocr    = OCRHandler()
    result = ocr.extract_from_path(args.image)

    print(f"\n{'='*55}")
    print(f"  SkinSpectra OCR — Result")
    print(f"{'='*55}")
    print(f"  success     : {result['success']}")
    print(f"  ingredients : {result['ingredient_count']}")
    print(f"  confidence  : {result['confidence']}")
    print(f"  total_ms    : {result['meta']['total_ms']}")
    print(f"  warnings    : {result['warnings']}")
    print(f"\n  Ingredients:")
    for i, ing in enumerate(result["ingredients"], 1):
        print(f"    {i:>2}. {ing}")
    if args.debug:
        print(f"\n  Raw OCR text:\n{result['raw_text']}")
    if result["error"]:
        print(f"\n  ERROR: {result['error']}")
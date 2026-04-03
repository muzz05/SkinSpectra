# SkinSpectra

SkinSpectra is an AI-powered skincare analysis platform that combines:

- NLP-based ingredient name normalization to INCI standards
- ML scoring for single-product compatibility and two-product layering safety
- Optional LLM-generated personalized skincare reports
- OCR extraction of ingredient lists from product label images
- Facial image skin-type detection
- A modern web UI served directly by FastAPI

The application is built around a 3-layer pipeline:

1. NLP mapping
2. ML scoring (individual and layering)
3. LLM report generation (Gemini)

## Features

### 1) Single Product Analysis
- Analyze one product against a user skin profile
- Returns:
	- Compatibility score (0-100)
	- Grade and verdict
	- Pros, cons, warnings
	- Ingredient-level breakdown
	- Optional personalized LLM report

### 2) Product Layering Analysis
- Analyze compatibility between Product A (applied first) and Product B (applied second)
- Returns:
	- Layering score (0-100)
	- Grade and verdict
	- Layering order and wait-time guidance
	- Application steps
	- Ingredient pair interactions (synergy/conflict/neutral)
	- Optional personalized LLM report

### 3) OCR Ingredient Extraction
- Upload product label images (JPG/PNG/WEBP/BMP/TIFF)
- Uses Tesseract OCR with preprocessing/postprocessing
- Returns parsed ingredient list and confidence metadata

### 4) Facial Skin-Type Detection
- Upload a face photo to predict skin type
- Supports error handling for no-face and blurry images
- Returns predicted skin type, confidence, and probabilities

### 5) NLP Ingredient Mapping
- Map single or batch ingredient names to INCI names
- Handles aliases, common naming variants, and uncertain mappings

### 6) Config and Health APIs
- Health endpoint with per-model readiness
- Config endpoints for valid skin types, concerns, age groups, and model status

### 7) Browser UI Included
- Open the root route to access the SkinSpectra interface
- Supports:
	- Single-product and layering workflows
	- Drag-and-drop label image upload
	- Auto OCR extraction into ingredient chips
	- Face-scan assisted skin-type auto-detection
	- Rich result rendering with cards, warnings, and report sections

## Project Structure

Key files:

- `api.py`: FastAPI application and all API routes
- `skinspectra.html`: Frontend interface served by FastAPI root route
- `requirements.txt`: Python dependencies
- `components/`: NLP, scoring, OCR, LLM, and facial-analysis modules
- `models/`: Trained model artifacts
- `data/`: Ingredient profile and layering compatibility datasets
- `testing/`: Unit/API tests

## Requirements

- Windows, macOS, or Linux
- Conda (Miniconda or Anaconda)
- Python 3.11 (recommended)
- Tesseract OCR installed on the OS (required for OCR endpoint)

> Note: `requirements.txt` says Python 3.11+; this project is most predictable on Python 3.11.

## Setup (Conda)

Run these commands from the project root.

### 1) Create and activate environment

```bash
conda create -n skinspectra python=3.11 -y
conda activate skinspectra
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Install Tesseract OCR

OCR depends on the system Tesseract binary, not only `pytesseract`.

- Windows (example using winget):

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

If Tesseract is not found at runtime, add its install directory to `PATH`.

## Environment Variables

Create a `.env` file in the project root (same folder as `api.py`) and set values as needed.

### Common variables

```env
GEMINI_API_KEY=your_gemini_api_key
SS_LLM_ENABLED=true
SS_MAX_INGREDIENTS=60
```

### Model and data paths

If your local folder names differ from defaults, set explicit paths.

```env
SS_NLP_MODEL_DIR=models/nlp
SS_CALC_MODEL_DIR=models/calculation_individual
SS_LAYERING_MODEL_DIR=models/calculation_layering
SS_FACIAL_MODEL_DIR=models/facial_analysis
SS_DATASET2=data/ingredient_profiles.csv
SS_DATASET3=data/layering_compatibility.csv
```

Setting these avoids path mismatch issues across environments.

## Run the App

After setup and activation:

### Option A: Run with Uvicorn

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Option B: Run with Python entrypoint

```bash
python api.py --reload --host 0.0.0.0 --port 8000
```

## Access the Application

- Web UI: `http://127.0.0.1:8000/`
- Interactive API docs (Swagger): `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

## Main API Endpoints

### Info / Health
- `GET /` - serves SkinSpectra web UI
- `GET /health` - API + model readiness status

### Config
- `GET /config/skin-types`
- `GET /config/concerns`
- `GET /config/age-groups`
- `GET /config/models`

### NLP
- `POST /nlp/map` - single ingredient mapping
- `POST /nlp/map/batch` - batch ingredient mapping

### Analysis
- `POST /analyze/product` - single product compatibility analysis
- `POST /analyze/layering` - two-product layering analysis
- `POST /analyze/skin-type` - facial skin-type prediction from photo

### OCR
- `POST /ocr/extract` - extract ingredients from label image
- `GET /ocr/info` - OCR engine details

## Quick API Example (Single Product)

```bash
curl -X POST "http://127.0.0.1:8000/analyze/product" \
	-H "Content-Type: application/json" \
	-d '{
		"product_name": "The Ordinary Niacinamide 10% + Zinc 1%",
		"ingredients": ["Niacinamide", "Zinc PCA", "Glycerin", "Hyaluronic Acid"],
		"skin_profile": {
			"skin_type": "oily",
			"concerns": ["acne", "pores"],
			"age_group": "adult",
			"is_pregnant": false,
			"skin_sensitivity": "normal",
			"current_routine": "",
			"allergies": "",
			"location_climate": "humid tropical",
			"experience_level": "beginner"
		},
		"include_llm": true
	}'
```

## Testing

Run tests from project root:

```bash
pytest -q
```

## Troubleshooting

- `503 Model not loaded`
	- Verify model/data path variables in `.env`
	- Confirm model files exist under `models/`

- OCR fails
	- Ensure Tesseract is installed and available in `PATH`
	- Try clearer, well-lit label images

- LLM report missing
	- Set `GEMINI_API_KEY`
	- Ensure `SS_LLM_ENABLED=true`

## Notes

- This tool provides AI-assisted skincare guidance and is not a medical diagnosis system.
- For persistent skin conditions, consult a licensed dermatologist.

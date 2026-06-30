# TruthLens — Fake News Detector

FISAT UnStop · Python Track

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy English model
python -m spacy download en_core_web_sm
```

## Google Fact Check API key (optional but recommended)

```bash
export GOOGLE_API_KEY="your_key_here"
```

Without the key, three hardcoded demo entries still work offline;
all other claims return "GOOGLE_API_KEY not set".

## Running the app

Open **two terminals** in the project root.

### Terminal 1 — FastAPI backend

```bash
uvicorn app:app --reload --port 8000
```

Wait for: `✓ TruthLens API is ready.`

### Terminal 2 — Streamlit frontend

```bash
streamlit run ui.py
```

Open the URL shown (default http://localhost:8501).

## Project structure

```
truthlens/
├── app.py              # FastAPI backend — all ML logic, port 8000
├── ui.py               # Streamlit frontend — zero ML logic
├── utils/
│   ├── __init__.py
│   ├── ingestor.py     # URL scraping, YouTube transcript, raw text pass-through
│   ├── claims.py       # spaCy sentence splitting + TF-IDF claim scoring
│   ├── classifier.py   # BERT tokenise → inference → label + confidence
│   ├── explainer.py    # LIME predictor wrapper → as_list + as_html
│   ├── fact_check.py   # Google Fact Check API + FACT_CACHE
│   └── baseline.py     # TF-IDF + XGBoost train and predict
├── requirements.txt
└── README.md
```

## Demo inputs

| Type | Example |
|------|---------|
| Text | *"The COVID-19 vaccine contains microchips for tracking."* |
| URL  | Any publicly accessible news article URL |
| YouTube | `https://www.youtube.com/watch?v=<video_id>` |

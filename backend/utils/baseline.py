import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from utils.logging_config import get_logger

log = get_logger("baseline")

# ── Paths ──────────────────────────────────────────────────────────────────
_DATA_DIR = Path("data")
_FAKE_CSV = _DATA_DIR / "fake.csv"
_REAL_CSV = _DATA_DIR / "real.csv"

_CACHE_DIR = Path("model_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PIPELINE_CACHE_PATH = _CACHE_DIR / "xgb_pipeline.joblib"
_CACHE_META_PATH = _CACHE_DIR / "xgb_pipeline_meta.json"


def _file_signature(path: Path) -> Optional[Dict]:
    """Return a (size, mtime) signature for a file — used to detect dataset changes."""
    if not path.exists():
        return None
    stat = path.stat()
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def _dataset_signature() -> Optional[Dict]:
    """Combined signature of both CSVs. None if either file is missing."""
    fake_sig = _file_signature(_FAKE_CSV)
    real_sig = _file_signature(_REAL_CSV)
    if fake_sig is None or real_sig is None:
        return None
    return {"fake": fake_sig, "real": real_sig}


def _load_csv_texts(path: Path) -> List[str]:
    """Load a fake/real news CSV and return combined title+text strings, one per row."""
    log.info(f"Loading dataset file: {path}")
    df = pd.read_csv(path)
    log.debug(f"{path.name}: columns={list(df.columns)}, rows={len(df)}")

    # Flexible column detection — common public datasets vary in naming
    title_col = next((c for c in df.columns if c.lower() == "title"), None)
    text_col = next(
        (c for c in df.columns if c.lower() in ("text", "content", "body", "article")), None
    )

    if text_col is None:
        raise ValueError(
            f"{path.name}: no text/content column found among {list(df.columns)}. "
            f"Expected a column named 'text', 'content', 'body', or 'article'."
        )

    if title_col:
        combined = (df[title_col].fillna("") + ". " + df[text_col].fillna("")).str.strip()
    else:
        combined = df[text_col].fillna("").astype(str).str.strip()

    texts = [t for t in combined.tolist() if t]
    log.info(f"Loaded {len(texts)} usable rows from {path.name} (out of {len(df)} total rows)")
    return texts


def _load_dataset() -> Tuple[List[str], List[int]]:
    """Load and label the fake/real CSV dataset. 0 = Genuine, 1 = Misleading."""
    real_texts = _load_csv_texts(_REAL_CSV)
    fake_texts = _load_csv_texts(_FAKE_CSV)

    X = real_texts + fake_texts
    y = [0] * len(real_texts) + [1] * len(fake_texts)

    log.info(f"Combined dataset: {len(real_texts)} genuine + {len(fake_texts)} misleading = {len(X)} total")
    return X, y


def train_xgb_pipeline(force_retrain: bool = False) -> Pipeline:
    """
    Train (or load a cached) TF-IDF + XGBoost pipeline on data/fake.csv + data/real.csv.

    Caching: if the dataset files haven't changed since the last successful
    training run (same size + mtime), the cached model is loaded from disk
    instead of retraining — this avoids retraining a 20k-row pipeline on
    every server restart.

    Falls back to a small inline placeholder dataset if the CSV files aren't
    present, so the app can still start and be demoed without the full dataset.
    """
    current_sig = _dataset_signature()

    # ── Try cache first ───────────────────────────────────────────────────────
    if not force_retrain and current_sig and _PIPELINE_CACHE_PATH.exists() and _CACHE_META_PATH.exists():
        try:
            cached_meta = json.loads(_CACHE_META_PATH.read_text())
            if cached_meta.get("signature") == current_sig:
                log.info(f"Dataset unchanged — loading cached pipeline from {_PIPELINE_CACHE_PATH}")
                pipeline = joblib.load(_PIPELINE_CACHE_PATH)
                log.info(
                    f"Cached pipeline loaded | trained on {cached_meta.get('n_samples')} samples "
                    f"| test accuracy {cached_meta.get('test_accuracy', 'n/a')} "
                    f"| trained at {cached_meta.get('trained_at', 'unknown')}"
                )
                return pipeline
            log.info("Dataset files changed since last cache (size/mtime differ) — retraining")
        except Exception as e:
            log.warning(f"Could not load cached pipeline ({e}) — retraining from scratch")
    elif force_retrain:
        log.info("force_retrain=True — skipping cache, retraining from scratch")

    # ── No usable CSVs — fall back to inline placeholder ───────────────────────
    if current_sig is None:
        log.warning(
            f"Dataset files not found at {_FAKE_CSV} / {_REAL_CSV} — "
            f"training on a small inline placeholder (80 examples) instead. "
            f"Place your 20k-row fake.csv and real.csv inside the 'data/' folder "
            f"for full-quality results."
        )
        return _train_on_inline_placeholder()

    # ── Real training run ────────────────────────────────────────────────────
    log.info("Training new XGBoost pipeline on the full CSV dataset...")
    t0 = time.time()
    X, y = _load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    log.debug(f"Train/test split: {len(X_train)} train, {len(X_test)} test")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))),
            ("xgb", XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=200, max_depth=6)),
        ]
    )

    log.info(f"Fitting pipeline on {len(X_train)} training examples (this may take a minute)...")
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    log.info(f"Training complete in {train_time:.1f}s")

    y_pred = pipeline.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))
    log.info(f"Held-out test accuracy: {test_accuracy:.4f} ({len(X_test)} samples)")
    log.debug(
        "Classification report:\n"
        + classification_report(y_test, y_pred, target_names=["Genuine", "Misleading"])
    )

    # ── Cache the trained pipeline to disk ──────────────────────────────────
    try:
        joblib.dump(pipeline, _PIPELINE_CACHE_PATH)
        meta = {
            "signature": current_sig,
            "n_samples": len(X),
            "test_accuracy": test_accuracy,
            "train_time_seconds": round(train_time, 1),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _CACHE_META_PATH.write_text(json.dumps(meta, indent=2))
        log.info(f"Pipeline cached to {_PIPELINE_CACHE_PATH} — future startups will skip retraining")
    except Exception as e:
        log.warning(f"Could not write pipeline cache ({e}) — will retrain on next startup")

    return pipeline


def _train_on_inline_placeholder() -> Pipeline:
    """Fallback: train on a tiny built-in dataset when the real CSVs aren't present."""
    genuine_texts = [
        "The capital of France is Paris.", "Water freezes at zero degrees Celsius.",
        "The sun rises in the east.", "A year has 365 days normally.",
        "Gravity pulls objects towards the earth.", "Photosynthesis is how plants make food.",
        "The human body has 206 bones.", "Earth revolves around the sun.",
        "Mount Everest is the highest mountain.", "Oxygen is essential for human life.",
        "Dogs are mammals.", "The Pacific is the largest ocean.",
        "An apple a day keeps the doctor away.", "Water is made of hydrogen and oxygen.",
        "Light travels faster than sound.", "Humans need to drink water to survive.",
        "The moon orbits the earth.", "Rome is the capital of Italy.",
        "Ice is the solid form of water.", "Birds have feathers.",
        "Sharks live in the ocean.", "Bees produce honey.",
        "The Sahara is a large desert.", "Trees produce oxygen.",
        "Fish breathe through gills.", "A triangle has three sides.",
        "A minute has sixty seconds.", "A day has twenty four hours.",
        "The Great Wall of China is very long.", "Jupiter is the largest planet.",
        "Neil Armstrong walked on the moon.", "Shakespeare wrote Hamlet.",
        "Iron is a metal.", "Cats are popular pets.",
        "The boiling point of water is 100 degrees Celsius.", "Diamonds are made of carbon.",
        "The human heart pumps blood.", "Python is a programming language.",
        "The earth is round.", "Fire is hot.",
    ]
    misleading_texts = [
        "The earth is completely flat.", "Vaccines cause widespread autism.",
        "The moon landing was filmed on a soundstage.", "Climate change is a hoax.",
        "Drinking bleach cures all viruses.", "5G towers spread diseases.",
        "Lizards secretly control the government.", "Aliens built the pyramids.",
        "Birds aren't real, they are government drones.", "Chemtrails are mind control chemicals.",
        "The sun revolves around the earth.", "Eating rocks is healthy.",
        "You can charge your phone in the microwave.", "The government hides the cure for aging.",
        "Mermaids have been found in the Atlantic.", "Unicorns exist in North Korea.",
        "Pigs can actually fly if trained.", "Gravity is an illusion.",
        "The earth is hollow inside.", "Elvis is still alive and hiding.",
        "Chocolate milk comes from brown cows.", "Water is dry.",
        "The sky is permanently neon green.", "Humans don't need oxygen.",
        "Trees talk to each other in English.", "Dinosaurs are still alive in Africa.",
        "You can walk to the moon.", "Fire is completely cold.",
        "Ice sinks in water.", "Rocks are soft until you touch them.",
        "Reading in the dark makes you blind.", "Swallowing gum stays in your stomach for 7 years.",
        "Lightning never strikes the same place twice.", "Goldfish have a 3-second memory.",
        "Bulls hate the color red.", "We only use 10 percent of our brains.",
        "Shaving makes hair grow back thicker.", "Cracking your knuckles causes arthritis.",
        "Toads give you warts.", "Bats are completely blind.",
    ]
    X = genuine_texts + misleading_texts
    y = [0] * len(genuine_texts) + [1] * len(misleading_texts)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("xgb", XGBClassifier(eval_metric="logloss", random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    log.info(f"Placeholder pipeline trained on {len(X)} inline examples")
    return pipeline


def xgb_pipeline_predict(text: str, pipeline: Pipeline) -> Dict[str, object]:
    """Run the trained XGBoost pipeline and return its label and winning probability."""
    log.debug(f"xgb_pipeline_predict() called | text length={len(text)}")

    if not text.strip():
        log.warning("xgb_pipeline_predict() received empty text — returning default Genuine/0.0")
        return {"label": "Genuine", "probability": 0.0}

    proba = pipeline.predict_proba([text])[0]
    prob_genuine, prob_misleading = float(proba[0]), float(proba[1])
    log.debug(f"XGBoost probs: genuine={prob_genuine:.4f}, misleading={prob_misleading:.4f}")

    if prob_misleading > prob_genuine:
        log.info(f"xgb_pipeline_predict() -> Misleading (probability={prob_misleading:.3f})")
        return {"label": "Misleading", "probability": prob_misleading}

    log.info(f"xgb_pipeline_predict() -> Genuine (probability={prob_genuine:.3f})")
    return {"label": "Genuine", "probability": prob_genuine}

import time
import warnings
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
load_dotenv()  # loads GOOGLE_API_KEY (and any other vars) from .env into os.environ

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
from lime.lime_text import LimeTextExplainer

from utils.logging_config import get_logger
from utils.ingestor import ingest
from utils.claims import extract_claims
from utils.classifier import bert_classify
from utils.explainer import lime_explain, lime_explain_html
from utils.fact_check import fact_check
from utils.baseline import train_xgb_pipeline, xgb_pipeline_predict
from utils.web_verify import web_verify_claims
from utils.entity_check import check_entity_order_for_claims
from utils.aggregator import compute_final_verdict

warnings.filterwarnings("ignore")

log = get_logger("app")

# ── Module-level singletons ──────────────────────────────────
tokenizer = None
bert_model = None
nlp = None
explainer = None
xgb_pipeline = None

_PRIMARY_MODEL = "mrm8488/bert-tiny-finetuned-fake-news-detection"
_FALLBACK_MODEL = "bert-base-uncased"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, bert_model, nlp, explainer, xgb_pipeline

    startup_start = time.time()
    log.info("=== TruthLens startup sequence begin ===")

    # ── BERT model + tokenizer ───────────────────────────────────────────────
    t0 = time.time()
    log.info(f"[1/4] Loading BERT tokenizer + model: {_PRIMARY_MODEL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(_PRIMARY_MODEL)
        bert_model = AutoModelForSequenceClassification.from_pretrained(_PRIMARY_MODEL)
        log.info(f"[1/4] Primary model loaded OK in {time.time() - t0:.2f}s")
    except Exception as e:
        log.warning(f"[1/4] Primary model failed: {e}")
        log.warning(f"[1/4] Falling back to: {_FALLBACK_MODEL}")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(_FALLBACK_MODEL)
        bert_model = AutoModelForSequenceClassification.from_pretrained(_FALLBACK_MODEL)
        log.info(f"[1/4] Fallback model loaded in {time.time() - t0:.2f}s")

    bert_model.eval()
    log.debug("[1/4] bert_model.eval() set — inference mode active")

    # ── spaCy ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[2/4] Loading spaCy model: en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    log.info(f"[2/4] spaCy loaded in {time.time() - t0:.2f}s")

    # ── LIME ──────────────────────────────────────────────────────────────────
    log.info("[3/4] Initialising LIME TextExplainer (class_names=[Genuine, Misleading])")
    explainer = LimeTextExplainer(class_names=["Genuine", "Misleading"])
    log.debug("[3/4] LIME explainer ready")

    # ── XGBoost baseline (trained on data/fake.csv + data/real.csv, cached) ──
    t0 = time.time()
    log.info("[4/4] Loading/training XGBoost pipeline (data/fake.csv + data/real.csv, cached)")
    xgb_pipeline = train_xgb_pipeline()
    log.info(f"[4/4] XGBoost pipeline ready in {time.time() - t0:.2f}s")

    log.info(f"=== Startup complete in {time.time() - startup_start:.2f}s — API ready ===")
    yield
    log.info("=== TruthLens shutting down ===")


app = FastAPI(title="TruthLens API", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str
    source_type: Literal["text", "youtube"]


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Run the full TruthLens pipeline (model + live web double-check) and return all results."""
    request_start = time.time()
    preview = req.text[:80].replace("\n", " ")
    log.info(f"--> /analyze called | source_type={req.source_type!r} | input preview: '{preview}...'")

    try:
        # ── Step 1 — ingest ──────────────────────────────────────────────────
        t0 = time.time()
        log.debug("[Step 1/9] Ingesting input...")
        text = ingest(req.text, req.source_type)
        if not text or text.startswith("Error during ingestion"):
            log.error(f"[Step 1/9] Ingestion FAILED: {text}")
            raise ValueError(f"Ingestion failed: {text}")
        log.info(f"[Step 1/9] Ingestion OK ({len(text)} chars) in {time.time() - t0:.2f}s")
        log.debug(f"[Step 1/9] Normalised text preview: {text[:120]!r}")

        # ── Step 2 — claim extraction ────────────────────────────────────────
        t0 = time.time()
        log.debug("[Step 2/9] Extracting claims (spaCy + TF-IDF)...")
        claims = extract_claims(text, nlp)
        log.info(f"[Step 2/9] Extracted {len(claims)} claim(s) in {time.time() - t0:.2f}s")
        for i, c in enumerate(claims, 1):
            log.debug(f"[Step 2/9]   claim {i}: {c[:100]!r}")

        # ── Step 3 — BERT classification ─────────────────────────────────────
        t0 = time.time()
        log.debug("[Step 3/9] Running BERT classification...")
        bert_result = bert_classify(text, tokenizer, bert_model)
        log.info(
            f"[Step 3/9] BERT result: {bert_result['label']} "
            f"(confidence={bert_result['confidence']:.3f}) in {time.time() - t0:.2f}s"
        )

        # ── Step 4 — LIME explanation ────────────────────────────────────────
        t0 = time.time()
        log.debug("[Step 4/9] Generating LIME word-weight explanation (num_samples=300)...")
        lime_words = lime_explain(text, tokenizer, bert_model, explainer)
        log.info(f"[Step 4/9] LIME produced {len(lime_words)} word weights in {time.time() - t0:.2f}s")

        # ── Step 5 — Google Fact-Check (highest-weighted signal) ──────────────
        t0 = time.time()
        log.debug(f"[Step 5/9] Fact-checking {len(claims)} claim(s) against Google Fact Check API...")
        fact_results = []
        for i, c in enumerate(claims, 1):
            fc = fact_check(c)
            log.debug(f"[Step 5/9]   claim {i} -> source={fc.get('source')!r}, verdict={fc.get('verdict')!r}")
            fact_results.append(fc)
        log.info(f"[Step 5/9] Fact-check complete in {time.time() - t0:.2f}s")

        # ── Step 6 — Live web double-check ───────────────────────────────────
        t0 = time.time()
        log.debug(f"[Step 6/9] Searching the live web to corroborate {len(claims)} claim(s)...")
        web_results = web_verify_claims(claims)
        for i, w in enumerate(web_results, 1):
            log.debug(f"[Step 6/9]   claim {i} -> {w['corroboration']} ({w['match_count']} result(s))")
        log.info(f"[Step 6/9] Web double-check complete in {time.time() - t0:.2f}s")

        # ── Step 7 — Entity-order consistency check ──────────────────────────
        # Catches contradictions keyword-overlap corroboration misses entirely,
        # e.g. a claim with two named entities swapped relative to real evidence
        # (the classic "Team A beat Team B" vs "Team B beat Team A" case).
        t0 = time.time()
        log.debug(f"[Step 7/9] Checking entity order for {len(claims)} claim(s) against web evidence...")
        entity_results = check_entity_order_for_claims(nlp, claims, web_results)
        mismatches = sum(1 for e in entity_results if e.get("order_mismatch"))
        for i, e in enumerate(entity_results, 1):
            log.debug(f"[Step 7/9]   claim {i} -> order_mismatch={e['order_mismatch']} | {e['detail']}")
        log.info(
            f"[Step 7/9] Entity-order check complete in {time.time() - t0:.2f}s "
            f"({mismatches}/{len(claims)} mismatch(es) found)"
        )

        # ── Step 8 — XGBoost baseline ────────────────────────────────────────
        t0 = time.time()
        log.debug("[Step 8/9] Running XGBoost baseline prediction...")
        xgb_result = xgb_pipeline_predict(text, xgb_pipeline)
        log.info(
            f"[Step 8/9] XGBoost result: {xgb_result['label']} "
            f"(probability={xgb_result['probability']:.3f}) in {time.time() - t0:.2f}s"
        )

        # ── Step 9 — Aggregate final verdict ─────────────────────────────────
        t0 = time.time()
        log.debug("[Step 9/9] Aggregating all signals into final verdict...")
        final_verdict = compute_final_verdict(
            bert_result, xgb_result, fact_results, web_results, entity_results
        )
        log.info(
            f"[Step 9/9] FINAL VERDICT: {final_verdict['label']} "
            f"(confidence={final_verdict['confidence']:.3f}) in {time.time() - t0:.2f}s"
        )

        total_time = time.time() - request_start
        log.info(f"<-- /analyze SUCCESS | total time {total_time:.2f}s")

        return {
            "final_verdict": final_verdict,
            "bert": bert_result,
            "xgb": xgb_result,
            "lime_words": lime_words,
            "fact_checks": fact_results,
            "web_checks": web_results,
            "entity_checks": entity_results,
            "claims_used": claims,
        }
    except Exception as e:
        log.error(f"<-- /analyze FAILED after {time.time() - request_start:.2f}s | {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lime_html")
async def lime_html(req: AnalyzeRequest):
    """Return LIME's HTML explanation for embedding in Streamlit."""
    request_start = time.time()
    log.info(f"--> /lime_html called | source_type={req.source_type!r}")

    try:
        t0 = time.time()
        log.debug("[lime_html] Ingesting input...")
        text = ingest(req.text, req.source_type)
        if not text or text.startswith("Error during ingestion"):
            log.error(f"[lime_html] Ingestion FAILED: {text}")
            raise ValueError(f"Ingestion failed: {text}")
        log.info(f"[lime_html] Ingestion OK ({len(text)} chars) in {time.time() - t0:.2f}s")

        t0 = time.time()
        log.debug("[lime_html] Generating LIME HTML explanation...")
        html = lime_explain_html(text, tokenizer, bert_model, explainer)
        log.info(f"[lime_html] HTML generated ({len(html)} chars) in {time.time() - t0:.2f}s")

        log.info(f"<-- /lime_html SUCCESS | total time {time.time() - request_start:.2f}s")
        return {"html": html}
    except Exception as e:
        log.error(f"<-- /lime_html FAILED after {time.time() - request_start:.2f}s | {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

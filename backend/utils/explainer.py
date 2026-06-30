from typing import List, Tuple

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from utils.logging_config import get_logger

log = get_logger("explainer")

_NUM_FEATURES = 10
_NUM_SAMPLES = 300


def _make_bert_predict_fn(tokenizer, model):
    """
    Return a LIME-compatible prediction function that batches texts through BERT.

    LIME expects a function  f(list[str]) -> np.ndarray of shape (N, num_classes).
    class_names = ["Genuine", "Misleading"]  →  col-0 = Genuine, col-1 = Misleading.

    The mrm8488 model outputs:  logit-0 = FAKE (Misleading), logit-1 = TRUE (Genuine).
    We swap columns so the array aligns with class_names order.
    """
    call_count = {"n": 0}

    def bert_predict(texts: List[str]) -> np.ndarray:
        call_count["n"] += 1
        log.debug(f"bert_predict() batch #{call_count['n']} | batch size={len(texts)}")

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        swapped = np.empty_like(probs)
        swapped[:, 0] = probs[:, 1]   # Genuine
        swapped[:, 1] = probs[:, 0]   # Misleading
        return swapped

    return bert_predict


def lime_explain(
    text: str,
    tokenizer,
    model,
    explainer: LimeTextExplainer,
) -> List[Tuple[str, float]]:
    """Generate a LIME explanation and return it as a list of (word, weight) tuples."""
    log.debug(f"lime_explain() called | text length={len(text)} | num_samples={_NUM_SAMPLES}")
    if not text.strip():
        log.warning("lime_explain() received empty text — returning []")
        return []

    predict_fn = _make_bert_predict_fn(tokenizer, model)
    exp = explainer.explain_instance(
        text, predict_fn, num_features=_NUM_FEATURES, num_samples=_NUM_SAMPLES
    )
    result = exp.as_list()
    log.info(f"lime_explain() produced {len(result)} word-weight pairs")
    log.debug(f"Top weights: {result[:5]}")
    return result


def lime_explain_html(
    text: str,
    tokenizer,
    model,
    explainer: LimeTextExplainer,
) -> str:
    """Generate a LIME explanation and return the raw HTML string for Streamlit rendering."""
    log.debug(f"lime_explain_html() called | text length={len(text)} | num_samples={_NUM_SAMPLES}")
    if not text.strip():
        log.warning("lime_explain_html() received empty text — returning placeholder")
        return "<p>No text provided for explanation.</p>"

    predict_fn = _make_bert_predict_fn(tokenizer, model)
    exp = explainer.explain_instance(
        text, predict_fn, num_features=_NUM_FEATURES, num_samples=_NUM_SAMPLES
    )
    html = exp.as_html()
    log.info(f"lime_explain_html() produced HTML string of length {len(html)}")
    return html

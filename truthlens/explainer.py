from typing import List, Tuple

import numpy as np
import torch
from truthlens.logging_config import get_logger

log = get_logger("explainer")

_NUM_FEATURES = 10
_NUM_SAMPLES = 300

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    log.warning("lime not installed — LIME explanations will be disabled")


def _make_bert_predict_fn(tokenizer, model):
    def bert_predict(texts: List[str]) -> np.ndarray:
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


def get_explainer():
    """Return a LimeTextExplainer instance or None if LIME is not installed."""
    if not LIME_AVAILABLE:
        return None
    return LimeTextExplainer(class_names=["Genuine", "Misleading"])


def lime_explain(text: str, tokenizer, model, explainer) -> List[Tuple[str, float]]:
    """Generate a LIME explanation and return it as a list of (word, weight) tuples."""
    if not LIME_AVAILABLE or explainer is None or not text.strip():
        return []

    predict_fn = _make_bert_predict_fn(tokenizer, model)
    exp = explainer.explain_instance(
        text, predict_fn, num_features=_NUM_FEATURES, num_samples=_NUM_SAMPLES
    )
    return exp.as_list()


def lime_explain_html(text: str, tokenizer, model, explainer) -> str:
    """Generate a LIME explanation and return raw HTML."""
    if not LIME_AVAILABLE or explainer is None or not text.strip():
        return "<p>LIME explanation not available.</p>"

    predict_fn = _make_bert_predict_fn(tokenizer, model)
    exp = explainer.explain_instance(
        text, predict_fn, num_features=_NUM_FEATURES, num_samples=_NUM_SAMPLES
    )
    return exp.as_html()

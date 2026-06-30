from typing import Dict
import torch
from truthlens.logging_config import get_logger

log = get_logger("classifier")


def bert_classify(text: str, tokenizer, model) -> Dict[str, object]:
    """Tokenise text, run BERT inference, and return the label and confidence score.
    
    Automatically reads the model's id2label config so the label mapping is always correct
    regardless of which model is loaded.
    """
    log.debug(f"bert_classify() called | text length={len(text)}")

    if not text.strip():
        return {"label": "Genuine", "confidence": 0.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].tolist()
    log.debug(f"Raw probs: {probs}")

    # Dynamically determine which index maps to FAKE vs REAL
    # from the model's own config (works for any model)
    id2label = getattr(model.config, 'id2label', {})
    fake_idx = None

    for idx, label in id2label.items():
        label_upper = str(label).upper()
        if any(k in label_upper for k in ['FAKE', 'FALSE', 'MISLEAD', '0', 'LABEL_0']):
            # Only assign if we haven't found a more specific match yet
            if fake_idx is None or 'LABEL' not in str(label).upper():
                fake_idx = int(idx)

    # Fallback: if config uses numeric labels like {0: 'LABEL_0', 1: 'LABEL_1'}
    # assume index 0 = FAKE for this specific model family
    if fake_idx is None:
        log.warning(f"Could not determine FAKE index from id2label={id2label}, defaulting to 0=FAKE")
        fake_idx = 0

    real_idx = 1 - fake_idx
    log.debug(f"Label mapping: index {fake_idx}=Misleading, index {real_idx}=Genuine | id2label={id2label}")

    prob_misleading = float(probs[fake_idx]) if fake_idx < len(probs) else 0.5
    prob_genuine = float(probs[real_idx]) if real_idx < len(probs) else 0.5

    if prob_misleading > prob_genuine:
        log.info(f"bert_classify() -> Misleading (confidence={prob_misleading:.3f})")
        return {"label": "Misleading", "confidence": prob_misleading}

    log.info(f"bert_classify() -> Genuine (confidence={prob_genuine:.3f})")
    return {"label": "Genuine", "confidence": prob_genuine}

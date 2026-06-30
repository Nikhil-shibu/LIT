from typing import Dict

import torch

from utils.logging_config import get_logger

log = get_logger("classifier")


def bert_classify(text: str, tokenizer, model) -> Dict[str, object]:
    """Tokenise text, run BERT inference, and return the label and confidence score."""
    log.debug(f"bert_classify() called | text length={len(text)}")

    if not text.strip():
        log.warning("bert_classify() received empty text — returning default Genuine/0.0")
        return {"label": "Genuine", "confidence": 0.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    log.debug(f"Tokenised input shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        logits = model(**inputs).logits
    log.debug(f"Raw logits: {logits.tolist()}")

    probs = torch.softmax(logits, dim=-1)[0].tolist()
    log.debug(f"Softmax probs: index0(fake)={probs[0]:.4f}, index1(true)={probs[1]:.4f}")

    # mrm8488/bert-tiny-finetuned-fake-news-detection label mapping:
    #   index 0 → FAKE  (Misleading)
    #   index 1 → TRUE  (Genuine)
    prob_misleading = probs[0]
    prob_genuine = probs[1]

    if prob_misleading > prob_genuine:
        log.info(f"bert_classify() -> Misleading (confidence={prob_misleading:.3f})")
        return {"label": "Misleading", "confidence": float(prob_misleading)}

    log.info(f"bert_classify() -> Genuine (confidence={prob_genuine:.3f})")
    return {"label": "Genuine", "confidence": float(prob_genuine)}

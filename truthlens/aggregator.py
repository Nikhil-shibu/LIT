from typing import Dict, List
from truthlens.logging_config import get_logger

log = get_logger("aggregator")

_FALSE_KEYWORDS = {
    "false", "pants on fire", "mostly false", "fake",
    "incorrect", "misleading", "fabricated", "unproven", "no evidence",
    "misinformation", "disinformation", "debunked",
}

# Signal weights — BERT is by far the most reliable signal
# Web results are intentionally excluded from scoring because fake news
# is widely shared online, making high match counts unreliable as evidence
# of genuineness. Web results are shown to the user but not scored.
_W_BERT = 0.65
_W_XGB  = 0.35

# Threshold: scores above this → Misleading.
# Lowered from 0.5 to 0.38 so the system leans toward flagging suspicious
# content rather than silently passing it as genuine.
_MISLEADING_THRESHOLD = 0.38


def compute_final_verdict(
    bert_result: Dict,
    xgb_result: Dict,
    fact_results: List[Dict],
    web_results: List[Dict],
) -> Dict:
    """
    Combine BERT and XGBoost signals into a final verdict.

    Web corroboration is intentionally NOT included in the score because
    fake news stories are often widely shared, meaning high web match counts
    are NOT a reliable indicator of truthfulness. Web results are passed
    through for transparency display only.

    Fact-check API results ARE included when available (requires GOOGLE_API_KEY).
    """
    log.debug("compute_final_verdict() called")
    weighted_score = 0.0
    weight_used = 0.0
    reasoning: List[str] = []

    # ── Signal 1: BERT (most reliable — 65% weight) ───────────────────────────
    bert_label = bert_result.get("label", "Genuine")
    bert_conf  = bert_result.get("confidence", 0.0)
    bert_score = bert_conf if bert_label == "Misleading" else (1.0 - bert_conf)
    weighted_score += _W_BERT * bert_score
    weight_used    += _W_BERT
    reasoning.append(
        f"BERT classifier: {bert_label} ({bert_conf * 100:.0f}% confidence)"
    )
    log.debug(f"BERT signal: label={bert_label}, score={bert_score:.3f}")

    # ── Signal 2: XGBoost baseline (35% weight) ───────────────────────────────
    if xgb_result:
        xgb_label = xgb_result.get("label", "Genuine")
        xgb_prob  = xgb_result.get("probability", 0.0)
        xgb_score = xgb_prob if xgb_label == "Misleading" else (1.0 - xgb_prob)
        weighted_score += _W_XGB * xgb_score
        weight_used    += _W_XGB
        reasoning.append(
            f"Baseline (XGBoost): {xgb_label} ({xgb_prob * 100:.0f}% confidence)"
        )
        log.debug(f"XGBoost signal: label={xgb_label}, score={xgb_score:.3f}")

    # ── Signal 3: Google Fact-Check (bonus signal — only when API key set) ────
    # We treat confirmed fact-check hits as a hard override: if a claim is
    # explicitly rated false by a credible source, bump the score significantly.
    fc_hits = [
        fc for fc in fact_results
        if fc.get("source") and fc.get("source") not in ("System", None)
        and fc.get("verdict") not in ("GOOGLE_API_KEY not set", "No match found", None)
    ]
    if fc_hits:
        false_count = sum(
            1 for fc in fc_hits
            if any(k in str(fc.get("verdict", "")).lower() for k in _FALSE_KEYWORDS)
        )
        if false_count > 0:
            # Hard boost: confirmed false claim → push score toward Misleading
            fact_boost = min(0.30, false_count * 0.15)
            weighted_score += fact_boost
            weight_used    += fact_boost
            reasoning.append(
                f"Fact-check databases: {false_count}/{len(fc_hits)} claim(s) "
                f"explicitly rated false by credible sources ⚠️"
            )
            log.debug(f"Fact-check BOOST applied: +{fact_boost:.2f}")
        else:
            reasoning.append(
                f"Fact-check databases: {len(fc_hits)} record(s) found — "
                f"none explicitly rated false"
            )
    else:
        reasoning.append(
            "Fact-check databases: no records found "
            "(set GOOGLE_API_KEY for live fact-checking)"
        )

    # ── Web results: displayed but NOT scored ────────────────────────────────
    # Fake news spreads widely — high match counts ≠ genuine content.
    if web_results:
        total_sources = sum(w.get("match_count", 0) for w in web_results)
        reasoning.append(
            f"Live web search: {total_sources} source(s) found discussing "
            f"these claims (shown below for reference — not used in scoring)"
        )
    else:
        reasoning.append("Live web search: no results retrieved")

    # ── Final weighted average ────────────────────────────────────────────────
    final_score = weighted_score / weight_used if weight_used else 0.5
    final_label = "Misleading" if final_score >= _MISLEADING_THRESHOLD else "Genuine"
    final_confidence = final_score if final_label == "Misleading" else (1.0 - final_score)

    log.info(
        f"Final verdict: {final_label} "
        f"(raw_score={final_score:.3f}, threshold={_MISLEADING_THRESHOLD}, "
        f"confidence={final_confidence:.3f})"
    )

    return {
        "label": final_label,
        "confidence": float(final_confidence),
        "reasoning": reasoning,
        "raw_score": float(final_score),
    }

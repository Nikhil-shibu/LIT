from typing import Dict, List

from utils.logging_config import get_logger

log = get_logger("aggregator")

_FALSE_KEYWORDS = {
    "false", "pants on fire", "mostly false", "fake",
    "incorrect", "misleading", "fabricated", "unproven", "no evidence",
}

# Signal weights — must sum to 1.0
# Fact-check weight increased (was 0.25) per user request: when a curated
# fact-check DB has a direct hit, it's the most reliable signal available.
_W_BERT = 0.20
_W_XGB = 0.10
_W_FACT_CHECK = 0.35
_W_WEB = 0.15
_W_ENTITY = 0.20


def compute_final_verdict(
    bert_result: Dict,
    xgb_result: Dict,
    fact_results: List[Dict],
    web_results: List[Dict],
    entity_results: List[Dict] = None,
) -> Dict:
    """
    Combine BERT, XGBoost, Google Fact-Check, live web-corroboration, and
    entity-order-consistency signals into one weighted final verdict with a
    human-readable reasoning trail.

    Each signal contributes a 0.0 (genuine) – 1.0 (misleading) score, weighted
    and averaged over only the signals that actually produced data.
    """
    log.debug("compute_final_verdict() called")
    weighted_score = 0.0
    weight_used = 0.0
    reasoning: List[str] = []

    # ── Signal 1: BERT ────────────────────────────────────────────────────────
    bert_label = bert_result.get("label", "Genuine")
    bert_conf = bert_result.get("confidence", 0.0)
    bert_score = bert_conf if bert_label == "Misleading" else (1 - bert_conf)
    weighted_score += _W_BERT * bert_score
    weight_used += _W_BERT
    reasoning.append(f"BERT classifier: {bert_label} ({bert_conf * 100:.0f}% confidence)")
    log.debug(f"BERT signal: label={bert_label}, score={bert_score:.3f}")

    # ── Signal 2: XGBoost baseline ───────────────────────────────────────────
    xgb_label = xgb_result.get("label", "Genuine")
    xgb_prob = xgb_result.get("probability", 0.0)
    xgb_score = xgb_prob if xgb_label == "Misleading" else (1 - xgb_prob)
    weighted_score += _W_XGB * xgb_score
    weight_used += _W_XGB
    reasoning.append(f"Baseline (XGBoost): {xgb_label} ({xgb_prob * 100:.0f}% confidence)")
    log.debug(f"XGBoost signal: label={xgb_label}, score={xgb_score:.3f}")

    # ── Signal 3: Google Fact-Check (highest-weighted signal) ───────────────
    fc_hits = [fc for fc in fact_results if fc.get("source") and fc.get("source") != "System"]
    if fc_hits:
        false_count = sum(
            1 for fc in fc_hits
            if any(k in str(fc.get("verdict", "")).lower() for k in _FALSE_KEYWORDS)
        )
        fc_score = false_count / len(fc_hits)
        weighted_score += _W_FACT_CHECK * fc_score
        weight_used += _W_FACT_CHECK
        reasoning.append(
            f"Fact-check databases: {false_count}/{len(fc_hits)} matched claim(s) rated false"
        )
        log.debug(f"Fact-check signal: {false_count}/{len(fc_hits)} false, score={fc_score:.3f}")
    else:
        reasoning.append("Fact-check databases: no matching records found for any claim")
        log.debug("Fact-check signal: no hits, skipped from weighting")

    # ── Signal 4: Live web corroboration (graduated, not binary) ────────────
    # A single loosely-related search hit is much weaker evidence than 3+
    # independent articles. Treating "1 match" the same as "5 matches" let one
    # weak hit fully cancel out a confident model verdict, so each claim now
    # gets a sliding score: 0 matches = strong suspicion (1.0), 3+ matches =
    # strong corroboration (0.0), with linear interpolation in between.
    if web_results:
        per_claim_scores = []
        for w in web_results:
            match_count = w.get("match_count", 0)
            score = max(0.0, (3 - min(match_count, 3)) / 3)
            per_claim_scores.append(score)
        web_score = sum(per_claim_scores) / len(per_claim_scores)
        weighted_score += _W_WEB * web_score
        weight_used += _W_WEB
        corroborated = sum(1 for w in web_results if w.get("match_count", 0) > 0)
        reasoning.append(
            f"Live web search: {corroborated}/{len(web_results)} claim(s) "
            f"have current online corroboration (strength-weighted by match count)"
        )
        log.debug(f"Web signal: per-claim scores={per_claim_scores}, avg score={web_score:.3f}")
    else:
        reasoning.append("Live web search: no claims were searched")
        log.debug("Web signal: no results, skipped from weighting")

    # ── Signal 5: Entity-order consistency ──────────────────────────────────
    # Catches cases keyword-overlap corroboration misses entirely — e.g. a
    # claim with two named entities swapped relative to the real evidence
    # (classic example: sports results with the winning team reversed).
    #
    # CRITICAL: only claims that were ACTUALLY comparable (checked=True) count
    # towards this signal. A claim with too few named entities to compare is a
    # genuine ABSENCE of signal, not evidence of truthfulness — counting it as
    # "0 mismatches found" would silently bias every verdict towards Genuine
    # whenever this check can't run, which is exactly what happened before.
    checked_entities = [e for e in (entity_results or []) if e.get("checked")]
    if checked_entities:
        mismatches = sum(1 for e in checked_entities if e.get("order_mismatch"))
        entity_score = mismatches / len(checked_entities)
        weighted_score += _W_ENTITY * entity_score
        weight_used += _W_ENTITY
        if mismatches:
            reasoning.append(
                f"⚠️ Entity-order check: {mismatches}/{len(checked_entities)} comparable claim(s) "
                f"have named entities in a different order than the matching web evidence — possible swap"
            )
        else:
            reasoning.append(
                f"Entity-order check: {len(checked_entities)} claim(s) compared, "
                f"no swapped-entity contradictions detected"
            )
        log.debug(f"Entity signal: {mismatches}/{len(checked_entities)} mismatches, score={entity_score:.3f}")
    else:
        reasoning.append("Entity-order check: not enough named entities to evaluate — no signal")
        log.debug("Entity signal: no comparable claims, skipped from weighting entirely")

    # ── Final weighted average ───────────────────────────────────────────────
    final_score = weighted_score / weight_used if weight_used else 0.5
    final_label = "Misleading" if final_score > 0.5 else "Genuine"
    final_confidence = final_score if final_label == "Misleading" else (1 - final_score)

    log.info(
        f"Final verdict: {final_label} "
        f"(weighted_score={final_score:.3f}, confidence={final_confidence:.3f})"
    )

    return {
        "label": final_label,
        "confidence": float(final_confidence),
        "reasoning": reasoning,
    }

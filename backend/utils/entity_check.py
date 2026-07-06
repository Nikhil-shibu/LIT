from typing import Dict, List

import spacy

from utils.logging_config import get_logger

log = get_logger("entity_check")

# Entity types we treat as "subjects" worth checking the order of —
# teams, organisations, people, and places are the most common nouns
# whose RELATIVE ORDER carries factual meaning (e.g. "X beat Y" vs "Y beat X").
_RELEVANT_LABELS = {"ORG", "PERSON", "GPE", "NORP"}


def _entity_sequence(doc: spacy.tokens.Doc) -> List[str]:
    """Return relevant named entities in the order they appear in the text."""
    return [ent.text.strip() for ent in doc.ents if ent.label_ in _RELEVANT_LABELS]


def check_entity_order(nlp: spacy.Language, claim: str, evidence_snippets: List[str]) -> Dict:
    """
    Heuristic contradiction check: if a claim and a piece of web evidence both
    mention the SAME two named entities but in REVERSED order, flag a likely
    swap (e.g. claim says "Team A beat Team B" but evidence says "Team B beat
    Team A" — both contain the same two team names, so naive keyword-overlap
    corroboration would wrongly mark the claim as 'verified' either way).

    This does NOT do full fact verification — it only catches the specific,
    common failure mode of swapped subject/object entities in short claims
    like sports results, election outcomes, or "X did Y to Z" statements.

    The returned "checked" flag tells the caller whether a real comparison was
    actually performed. This matters: "not enough entities to compare" must
    NEVER be treated the same as "compared and found no mismatch" — the first
    is an absence of signal, the second is mild positive evidence. Conflating
    the two would silently bias every verdict towards "Genuine" whenever this
    check can't run, regardless of what the other signals say.
    """
    log.debug(f"check_entity_order() called | claim preview: {claim[:80]!r}")

    claim_doc = nlp(claim)
    claim_entities = _entity_sequence(claim_doc)

    if len(claim_entities) < 2:
        log.debug("Fewer than 2 relevant entities in claim — skipping order check")
        return {
            "checked": False,
            "order_mismatch": False,
            "detail": "Not enough named entities in the claim to compare order",
            "claim_entities": claim_entities,
        }

    compared_at_least_once = False

    for snippet in evidence_snippets:
        if not snippet or not snippet.strip():
            continue

        snippet_doc = nlp(snippet)
        snippet_entities = _entity_sequence(snippet_doc)

        # Only compare when the snippet mentions at least the same two entities
        common = [e for e in claim_entities if e in snippet_entities]
        if len(common) < 2:
            continue

        compared_at_least_once = True
        claim_order = [e for e in claim_entities if e in common][:2]
        snippet_order = [e for e in snippet_entities if e in common][:2]

        if claim_order != snippet_order:
            log.warning(
                f"Entity ORDER MISMATCH detected | claim order={claim_order} | "
                f"evidence order={snippet_order} | snippet preview: {snippet[:100]!r}"
            )
            return {
                "checked": True,
                "order_mismatch": True,
                "detail": (
                    f"Claim mentions {claim_order[0]!r} before {claim_order[1]!r}, "
                    f"but web evidence mentions them in the opposite order — "
                    f"possible factual swap"
                ),
                "claim_entities": claim_entities,
            }

    if not compared_at_least_once:
        log.debug("No evidence snippet shared 2+ entities with the claim — check could not run")
        return {
            "checked": False,
            "order_mismatch": False,
            "detail": "No web evidence shared enough named entities with the claim to compare",
            "claim_entities": claim_entities,
        }

    log.debug("No entity-order mismatch found against any evidence snippet")
    return {
        "checked": True,
        "order_mismatch": False,
        "detail": "No entity-order mismatch found",
        "claim_entities": claim_entities,
    }


def check_entity_order_for_claims(
    nlp: spacy.Language,
    claims: List[str],
    web_results: List[Dict],
) -> List[Dict]:
    """
    Run check_entity_order() for each (claim, web evidence) pair.
    web_results must be the same length/order as claims (as produced by web_verify_claims).
    """
    log.debug(f"check_entity_order_for_claims() called | {len(claims)} claim(s)")
    results = []
    for claim, web_result in zip(claims, web_results):
        snippets = [ev.get("snippet", "") for ev in web_result.get("evidence", [])]
        snippets += [ev.get("title", "") for ev in web_result.get("evidence", [])]
        result = check_entity_order(nlp, claim, snippets)
        result["claim"] = claim
        results.append(result)

    checked = [r for r in results if r["checked"]]
    mismatches = sum(1 for r in checked if r["order_mismatch"])
    log.info(
        f"check_entity_order_for_claims() completed | "
        f"{len(checked)}/{len(claims)} claim(s) actually comparable | "
        f"{mismatches} mismatch(es) found"
    )

    return results

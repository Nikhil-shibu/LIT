from typing import Dict, List

from utils.logging_config import get_logger

log = get_logger("web_verify")

_MAX_RESULTS = 5


def _get_ddgs():
    """Import DDGS lazily so a missing/renamed package gives a clear error message."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
            return DDGS
        except ImportError as e:
            log.error(f"Neither 'ddgs' nor 'duckduckgo_search' is installed: {e}")
            raise


def web_verify_claim(claim: str, max_results: int = _MAX_RESULTS) -> Dict:
    """
    Search the live web for current news/articles that corroborate a claim.

    This is what lets TruthLens reason about TODAY's news, not just a static
    fact-check cache — it queries DuckDuckGo's news index in real time.
    """
    claim = claim.strip()
    log.debug(f"web_verify_claim() called | claim preview: {claim[:80]!r}")

    if not claim:
        return {"claim": claim, "evidence": [], "corroboration": "Empty claim", "match_count": 0}

    DDGS = _get_ddgs()
    results = []

    # ── Try recent news search first (best for "is today's news real") ──────
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(claim, max_results=max_results))
        log.debug(f"DDG news search returned {len(results)} result(s)")
    except Exception as e:
        log.warning(f"DDG news search failed ({e}) — will try general text search")

    # ── Fall back to general web search if no news hits ─────────────────────
    if not results:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(claim, max_results=max_results))
            log.debug(f"DDG text search returned {len(results)} result(s)")
        except Exception as e:
            log.error(f"DDG text search also failed: {e}")
            return {
                "claim": claim,
                "evidence": [],
                "corroboration": "Web search unavailable (network/rate-limit issue)",
                "match_count": 0,
            }

    evidence = []
    for r in results:
        evidence.append(
            {
                "title": r.get("title", ""),
                "snippet": r.get("body") or r.get("excerpt") or r.get("description", ""),
                "url": r.get("url") or r.get("href", ""),
                "source": r.get("source", ""),
                "date": r.get("date", ""),
            }
        )

    match_count = len(evidence)
    if match_count == 0:
        corroboration = "No corroborating sources found online"
        log.info(f"No web corroboration for claim: {claim[:60]!r}")
    elif match_count >= 3:
        corroboration = f"{match_count} independent sources discuss this"
        log.info(f"Strong web corroboration ({match_count}) for claim: {claim[:60]!r}")
    else:
        corroboration = f"Limited corroboration ({match_count} source(s)) found"
        log.info(f"Weak web corroboration ({match_count}) for claim: {claim[:60]!r}")

    return {
        "claim": claim,
        "evidence": evidence,
        "corroboration": corroboration,
        "match_count": match_count,
    }


def web_verify_claims(claims: List[str]) -> List[Dict]:
    """Run web_verify_claim() over a list of claims and return all results."""
    log.debug(f"web_verify_claims() called | {len(claims)} claim(s)")
    results = [web_verify_claim(c) for c in claims]
    log.info(f"web_verify_claims() completed for {len(claims)} claim(s)")
    return results

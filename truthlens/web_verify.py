from typing import Dict, List
from truthlens.logging_config import get_logger

log = get_logger("web_verify")

_MAX_RESULTS = 5


def _get_ddgs():
    """Import DDGS lazily so a missing/renamed package gives a clear error message."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            return DDGS
        except ImportError:
            return None


def web_verify_claim(claim: str, max_results: int = _MAX_RESULTS) -> Dict:
    """Search the live web for current news/articles that corroborate a claim."""
    claim = claim.strip()
    if not claim:
        return {"claim": claim, "evidence": [], "corroboration": "Empty claim", "match_count": 0}

    DDGS = _get_ddgs()
    if DDGS is None:
        return {
            "claim": claim,
            "evidence": [],
            "corroboration": "Web search unavailable (ddgs not installed)",
            "match_count": 0,
        }

    results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(claim, max_results=max_results))
    except Exception as e:
        log.warning(f"DDG news search failed ({e}) — trying general text search")

    if not results:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(claim, max_results=max_results))
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
        evidence.append({
            "title": r.get("title", ""),
            "snippet": r.get("body") or r.get("excerpt") or r.get("description", ""),
            "url": r.get("url") or r.get("href", ""),
            "source": r.get("source", ""),
            "date": r.get("date", ""),
        })

    match_count = len(evidence)
    if match_count == 0:
        corroboration = "No corroborating sources found online"
    elif match_count >= 3:
        corroboration = f"{match_count} independent sources discuss this"
    else:
        corroboration = f"Limited corroboration ({match_count} source(s)) found"

    return {
        "claim": claim,
        "evidence": evidence,
        "corroboration": corroboration,
        "match_count": match_count,
    }


def web_verify_claims(claims: List[str]) -> List[Dict]:
    """Run web_verify_claim() over a list of claims and return all results."""
    return [web_verify_claim(c) for c in claims]

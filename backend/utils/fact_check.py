import os
import urllib.parse
from typing import Dict, Optional

import requests

from utils.logging_config import get_logger

log = get_logger("fact_check")

# ── Module-level cache (survives for the lifetime of the process) ────────────
FACT_CACHE: Dict[str, Dict] = {
    "The COVID-19 vaccine contains microchips for tracking.": {
        "source": "PolitiFact",
        "verdict": "Pants on Fire",
        "url": "https://www.politifact.com/factchecks/2021/jan/05/"
               "facebook-posts/no-covid-19-vaccines-dont-contain-microchip/",
    },
    "Climate change is a hoax invented by the government.": {
        "source": "FactCheck.org",
        "verdict": "False",
        "url": "https://www.factcheck.org/2020/01/"
               "is-climate-change-a-hoax/",
    },
    "The moon landing was faked in a Hollywood studio.": {
        "source": "Snopes",
        "verdict": "False",
        "url": "https://www.snopes.com/fact-check/moon-landing-faked/",
    },
}

_FACT_CHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def fact_check(claim: str) -> Dict[str, Optional[str]]:
    """Query the Google Fact Check API for a claim; return cached result when available."""
    claim = claim.strip()
    log.debug(f"fact_check() called | claim preview: {claim[:80]!r}")

    if not claim:
        log.warning("fact_check() received an empty claim")
        return {"source": None, "verdict": "Empty claim", "url": None}

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if claim in FACT_CACHE:
        log.info(f"Cache HIT for claim: {claim[:60]!r}")
        return FACT_CACHE[claim]

    log.debug(f"Cache MISS for claim: {claim[:60]!r} — querying live API")

    # ── Live API call ─────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        log.warning("GOOGLE_API_KEY not set in environment — skipping live API call")
        result: Dict[str, Optional[str]] = {
            "source": "System",
            "verdict": "GOOGLE_API_KEY not set",
            "url": None,
        }
        FACT_CACHE[claim] = result
        return result

    params = {"query": claim, "key": api_key}
    url = f"{_FACT_CHECK_API}?{urllib.parse.urlencode(params)}"
    log.debug(f"Calling Google Fact Check API: {url.split('&key=')[0]}&key=***")

    try:
        resp = requests.get(url, timeout=5)
        log.debug(f"Fact Check API responded with status {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()

        claims_list = data.get("claims", [])
        log.debug(f"API returned {len(claims_list)} claim match(es)")

        if claims_list:
            reviews = claims_list[0].get("claimReview", [])
            if reviews:
                review = reviews[0]
                result = {
                    "source": review.get("publisher", {}).get("name", "Unknown"),
                    "verdict": review.get("textualRating", "Unknown"),
                    "url": review.get("url", "#"),
                }
                log.info(f"Fact-check found: source={result['source']!r}, verdict={result['verdict']!r}")
                FACT_CACHE[claim] = result
                return result

    except requests.exceptions.RequestException as e:
        log.error(f"Fact Check API request FAILED for claim '{claim[:60]}…': {e}")

    # ── No result found ───────────────────────────────────────────────────────
    log.info(f"No fact-check match found for claim: {claim[:60]!r}")
    result = {"source": None, "verdict": "No match found", "url": None}
    FACT_CACHE[claim] = result
    return result

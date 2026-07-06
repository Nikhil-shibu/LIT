import os
import urllib.parse
from typing import Dict, Optional

import requests

from truthlens.logging_config import get_logger

log = get_logger("fact_check")

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
        "url": "https://www.factcheck.org/2020/01/is-climate-change-a-hoax/",
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
    if not claim:
        return {"source": None, "verdict": "Empty claim", "url": None}

    if claim in FACT_CACHE:
        return FACT_CACHE[claim]

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        result: Dict[str, Optional[str]] = {
            "source": "System",
            "verdict": "GOOGLE_API_KEY not set",
            "url": None,
        }
        FACT_CACHE[claim] = result
        return result

    params = {"query": claim, "key": api_key}
    url = f"{_FACT_CHECK_API}?{urllib.parse.urlencode(params)}"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        claims_list = data.get("claims", [])
        if claims_list:
            reviews = claims_list[0].get("claimReview", [])
            if reviews:
                review = reviews[0]
                result = {
                    "source": review.get("publisher", {}).get("name", "Unknown"),
                    "verdict": review.get("textualRating", "Unknown"),
                    "url": review.get("url", "#"),
                }
                FACT_CACHE[claim] = result
                return result
    except requests.exceptions.RequestException as e:
        log.error(f"Fact Check API request FAILED: {e}")

    result = {"source": None, "verdict": "No match found", "url": None}
    FACT_CACHE[claim] = result
    return result

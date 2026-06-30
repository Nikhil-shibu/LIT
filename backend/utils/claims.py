from typing import List

import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logging_config import get_logger

log = get_logger("claims")

# ── NLTK punkt setup (resilient — never crash the app over this) ────────────
# We deliberately do NOT use nltk.data.find() to check both 'punkt' and
# 'punkt_tab' in the same process: NLTK has a known bug where checking both
# resources back-to-back corrupts its internal path-resolution cache (it
# treats "punkt_tab" as a suffix of the already-resolved "punkt/PY3" path
# and looks for the nonsensical "punkt/PY3_tab"). The files can be 100%
# correctly installed on disk and this lookup will still fail.
#
# Instead we just attempt the real download (idempotent — no-op if already
# present) and then functionally test sent_tokenize() directly, which uses
# its own internal resolution and is unaffected by the find() cache bug.

_NLTK_FALLBACK_AVAILABLE = False

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception as e:
    log.warning(f"NLTK download attempt raised an error (continuing anyway): {e}")

try:
    _test_result = nltk.tokenize.sent_tokenize("This is a test. This is another test sentence.")
    if len(_test_result) == 2:
        _NLTK_FALLBACK_AVAILABLE = True
        log.info("NLTK sent_tokenize() verified working — fallback enabled.")
    else:
        log.warning(
            f"NLTK sent_tokenize() returned unexpected output ({_test_result!r}) — "
            f"fallback disabled, relying on spaCy only."
        )
except Exception as e:
    log.warning(
        f"NLTK sent_tokenize() functional test failed ({e}). "
        f"The NLTK sentence-splitting fallback will be disabled; "
        f"spaCy will be used exclusively."
    )

if not _NLTK_FALLBACK_AVAILABLE:
    log.warning("NLTK fallback unavailable — relying solely on spaCy for sentence splitting.")


def extract_claims(text: str, nlp: spacy.Language) -> List[str]:
    """Extract the top-3 claim sentences from text using spaCy + TF-IDF scoring."""
    log.debug(f"extract_claims() called | text length={len(text)}")
    if not text.strip():
        log.warning("extract_claims() received empty text — returning []")
        return []

    # ── Step A: sentence splitting ────────────────────────────────────────────
    doc = nlp(text)
    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.split()) > 4
    ]
    log.debug(f"spaCy found {len(sentences)} candidate sentences (>4 words)")

    if len(sentences) < 3 and _NLTK_FALLBACK_AVAILABLE:
        log.debug("Fewer than 3 spaCy sentences — falling back to NLTK sent_tokenize")
        try:
            sentences = [
                s for s in nltk.tokenize.sent_tokenize(text) if len(s.split()) > 4
            ]
            log.debug(f"NLTK fallback found {len(sentences)} candidate sentences")
        except Exception as e:
            log.warning(f"NLTK sent_tokenize failed at runtime ({e}) — keeping spaCy result")
    elif len(sentences) < 3:
        log.debug("Fewer than 3 spaCy sentences and NLTK fallback unavailable — proceeding with spaCy result only")

    if not sentences:
        log.warning("No usable sentences found — returning truncated raw text")
        return [text[:500]]
    if len(sentences) <= 3:
        log.debug(f"3 or fewer sentences available — returning all {len(sentences)}")
        return sentences

    # ── Step B & C: TF-IDF sentence scoring ──────────────────────────────────
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError as e:
        log.warning(f"TF-IDF vectorisation failed ({e}) — returning first 3 sentences")
        return sentences[:3]

    scores = tfidf_matrix.mean(axis=1).A1
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    top_claims = [sent for sent, _ in ranked[:3]]

    log.info(f"extract_claims() selected top {len(top_claims)} claim(s) from {len(sentences)} sentences")
    for i, (sent, score) in enumerate(ranked[:3], 1):
        log.debug(f"  claim {i} (score={score:.4f}): {sent[:100]!r}")

    return top_claims

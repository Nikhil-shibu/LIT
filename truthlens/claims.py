from typing import List

import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from truthlens.logging_config import get_logger

log = get_logger("claims")

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
except Exception:
    pass


def extract_claims(text: str, nlp: spacy.Language) -> List[str]:
    """Extract the top-3 claim sentences from text using spaCy + TF-IDF scoring."""
    if not text.strip():
        return []

    doc = nlp(text)
    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.split()) > 4
    ]

    if len(sentences) < 3 and _NLTK_FALLBACK_AVAILABLE:
        try:
            sentences = [
                s for s in nltk.tokenize.sent_tokenize(text) if len(s.split()) > 4
            ]
        except Exception:
            pass

    if not sentences:
        return [text[:500]]
    if len(sentences) <= 3:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return sentences[:3]

    scores = tfidf_matrix.mean(axis=1).A1
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [sent for sent, _ in ranked[:3]]

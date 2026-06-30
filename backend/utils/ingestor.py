import re

from youtube_transcript_api import YouTubeTranscriptApi

from utils.logging_config import get_logger

log = get_logger("ingestor")


def ingest(text: str, source_type: str) -> str:
    """
    Ingest and normalise content from either raw text or a YouTube link/ID.

    URL scraping was intentionally removed: arbitrary web pages are unreliable
    to scrape cleanly and their source credibility can't be verified at parse
    time, so they were producing noisy/garbage input into the ML pipeline.
    """
    log.debug(f"ingest() called | source_type={source_type!r} | raw input len={len(text)}")

    try:
        if source_type == "youtube":
            result = _extract_youtube(text.strip())
        elif source_type == "text":
            result = _normalize_text(text)
        else:
            raise ValueError(
                f"Unsupported source_type {source_type!r} — only 'text' and 'youtube' are supported"
            )
        log.debug(f"ingest() succeeded | output len={len(result)}")
        return result
    except Exception as e:
        log.error(f"ingest() FAILED | source_type={source_type!r} | {type(e).__name__}: {e}")
        return f"Error during ingestion: {e}"


def _extract_youtube(video_input: str) -> str:
    """Extract and join the transcript of a YouTube video."""
    log.info(f"Extracting YouTube transcript from input: {video_input}")
    video_id = _parse_youtube_id(video_input)
    log.debug(f"Parsed video_id={video_id!r}")
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    log.debug(f"Transcript has {len(transcript)} segments")
    text = " ".join(entry["text"] for entry in transcript)
    log.info(f"Transcript joined: {len(text)} characters")
    return _normalize_text(text)


def _parse_youtube_id(raw: str) -> str:
    """Parse a YouTube video ID from a full URL or a bare ID string."""
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", raw)
    if match:
        log.debug(f"Matched standard watch URL pattern -> {match.group(1)}")
        return match.group(1)
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", raw)
    if match:
        log.debug(f"Matched short youtu.be URL pattern -> {match.group(1)}")
        return match.group(1)
    match = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]{11})", raw)
    if match:
        log.debug(f"Matched embed URL pattern -> {match.group(1)}")
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        log.debug(f"Input is already a bare video ID -> {raw}")
        return raw
    log.error(f"Could not parse a YouTube video ID from: {raw!r}")
    raise ValueError(f"Could not extract a YouTube video ID from: {raw!r}")


def _normalize_text(text: str) -> str:
    """Re-encode text as UTF-8 and strip leading/trailing whitespace."""
    return text.encode("utf-8", "ignore").decode("utf-8").strip()

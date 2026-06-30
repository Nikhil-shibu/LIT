import re

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

from truthlens.logging_config import get_logger

log = get_logger("ingestor")


def ingest(text: str, source_type: str) -> str:
    """
    Ingest and normalise content from either raw text or a YouTube link/ID.
    """
    log.debug(f"ingest() called | source_type={source_type!r} | raw input len={len(text)}")

    try:
        if source_type == "youtube":
            result = _extract_youtube(text.strip())
        elif source_type in ("text", "url"):
            result = _normalize_text(text)
        else:
            raise ValueError(
                f"Unsupported source_type {source_type!r}"
            )
        log.debug(f"ingest() succeeded | output len={len(result)}")
        return result
    except Exception as e:
        log.error(f"ingest() FAILED | source_type={source_type!r} | {type(e).__name__}: {e}")
        return f"Error during ingestion: {e}"


def _extract_youtube(video_input: str) -> str:
    """Extract and join the transcript of a YouTube video."""
    if not YOUTUBE_AVAILABLE:
        return "YouTube transcript extraction not available (youtube_transcript_api not installed)."
    log.info(f"Extracting YouTube transcript from input: {video_input}")
    video_id = _parse_youtube_id(video_input)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join(entry["text"] for entry in transcript)
    return _normalize_text(text)


def _parse_youtube_id(raw: str) -> str:
    """Parse a YouTube video ID from a full URL or a bare ID string."""
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", raw)
    if match:
        return match.group(1)
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", raw)
    if match:
        return match.group(1)
    match = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]{11})", raw)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw
    raise ValueError(f"Could not extract a YouTube video ID from: {raw!r}")


def _normalize_text(text: str) -> str:
    """Re-encode text as UTF-8 and strip leading/trailing whitespace."""
    return text.encode("utf-8", "ignore").decode("utf-8").strip()

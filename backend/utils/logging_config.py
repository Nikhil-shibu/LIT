import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that prints timestamped, leveled messages to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on reload
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger

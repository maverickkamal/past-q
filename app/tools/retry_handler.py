"""Retry handler with exponential backoff and jitter for Vertex AI 429 RESOURCE_EXHAUSTED errors."""

import asyncio
import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


def is_resource_exhausted_error(exc: Exception) -> bool:
    """Detects whether an exception represents a 429 RESOURCE_EXHAUSTED error."""
    err_str = str(exc).upper()
    if "429" in err_str:
        return True
    if "RESOURCE_EXHAUSTED" in err_str or "RESOURCE EXHAUSTED" in err_str:
        return True
    if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
        return True
    return False


def calculate_backoff(attempt: int, base: float = 10.0, max_delay: float = 120.0) -> float:
    """Calculates exponential backoff with jitter: min(max_delay, base * 2^(attempt-1)) + jitter."""
    calculated = min(max_delay, base * (2 ** (attempt - 1)))
    jitter = random.uniform(1.0, 4.0)
    return calculated + jitter

"""Retry handler with exponential backoff and jitter for Vertex AI 429 contention and network/SSL drops."""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
from typing import Any

logger = logging.getLogger(__name__)


def is_transient_error(exc: Exception) -> bool:
    """Detects whether an exception represents a retryable transient error:
    - 429 RESOURCE_EXHAUSTED / temporary capacity contention
    - 503 UNAVAILABLE / service overload
    - 504 DEADLINE_EXCEEDED / gateway timeout
    - OAuth2 transport errors & SSL/TLS socket drops (SSLEOFError, ConnectionResetError)
    - Intermittent network drops and connection timeouts
    """
    if exc is None:
        return False

    # Check exception type directly
    exc_type_name = type(exc).__name__
    if isinstance(exc, (ssl.SSLError, ConnectionError, TimeoutError, OSError)):
        return True
    if exc_type_name in ("TransportError", "ClientError", "ServerError", "APIError"):
        # Check status codes if present
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (429, 500, 502, 503, 504):
            return True

    err_str = str(exc).upper()

    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "RESOURCE EXHAUSTED" in err_str:
        return True

    if "503" in err_str or "UNAVAILABLE" in err_str or "SERVICE UNAVAILABLE" in err_str:
        return True
    if "504" in err_str or "DEADLINE_EXCEEDED" in err_str or "GATEWAY TIMEOUT" in err_str:
        return True
    if "500" in err_str or "INTERNAL SERVER ERROR" in err_str:
        return True

    ssl_keywords = [
        "UNEXPECTED_EOF",
        "EOF OCCURRED",
        "SSL",
        "SSLEOFERROR",
        "SSLERROR",
        "CONNECTION RESET",
        "CONNECTION CLOSED",
        "CONNECTION REFUSED",
        "REMOTE DISCONNECTED",
        "BROKEN PIPE",
        "MAX RETRIES EXCEEDED",
        "TRANSPORT ERROR",
        "TRANSPORTERROR",
        "HANDSHAKE",
        "TIMED OUT",
        "TIMEOUT",
    ]
    if any(kw in err_str for kw in ssl_keywords):
        return True

    if "402" in err_str or "PREPAYMENT CREDITS" in err_str:
        return False

    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 402:
        return False
    if code in (429, 502, 503, 504):
        return True

    return False


is_resource_exhausted_error = is_transient_error


def get_retry_reason(exc: Exception) -> str:
    """Returns a concise, human-readable summary of the transient failure."""
    err_str = str(exc).upper()
    if "402" in err_str or "PREPAYMENT CREDITS" in err_str:
        return "[402 Credits Depleted] Google AI Studio prepayment credits depleted. Manage billing at https://ai.studio/projects"
    if "404" in err_str or "NOT_FOUND" in err_str:
        return "[404 Not Found] The specified Gemini model is not found or not available to this API key"
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "RESOURCE EXHAUSTED" in err_str:
        return "[429 Resource Exhausted] Temporary capacity contention on Google APIs"
    if any(k in err_str for k in ["EOF", "SSL", "TRANSPORTERROR", "CONNECTION", "DISCONNECTED", "SOCKET", "GETADDRINFO"]):
        return "[Network/SSL Drop] Transient connection error to Google APIs (socket/DNS drop)"
    if "503" in err_str or "UNAVAILABLE" in err_str:
        return "[503 Service Unavailable] Google API service temporarily overloaded"
    if "504" in err_str or "DEADLINE_EXCEEDED" in err_str or "TIMED OUT" in err_str:
        return "[Gateway Timeout] Request deadline exceeded"
    return f"[API Error] Temporary failure ({type(exc).__name__})"


def calculate_backoff(attempt: int, base: float = 10.0, max_delay: float = 120.0) -> float:
    """Calculates exponential backoff with jitter: min(max_delay, base * 2^(attempt-1)) + jitter."""
    calculated = min(max_delay, base * (2 ** (attempt - 1)))
    jitter = random.uniform(1.0, 4.0)
    return calculated + jitter

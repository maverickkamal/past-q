"""Retry handler with exponential backoff and jitter for Vertex AI 429 contention and network/SSL drops."""

from __future__ import annotations

import asyncio
import logging
import random
import socket
import ssl
from typing import Any

try:
    import httpx
    HTTPX_NETWORK_ERRORS: tuple[type[Exception], ...] = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
        httpx.TransportError,
    )
except ImportError:
    HTTPX_NETWORK_ERRORS = ()

try:
    import httpcore
    HTTPCORE_NETWORK_ERRORS: tuple[type[Exception], ...] = (
        httpcore.ConnectError,
        httpcore.ConnectTimeout,
        httpcore.ReadTimeout,
        httpcore.WriteTimeout,
        httpcore.PoolTimeout,
        httpcore.NetworkError,
        httpcore.RemoteProtocolError,
    )
except ImportError:
    HTTPCORE_NETWORK_ERRORS = ()

logger = logging.getLogger(__name__)


def configure_clean_logging() -> None:
    """Silences noisy third-party loggers (Google ADK, GenAI, HTTPX, HTTPCore) so that
    handled transient errors do not dump multi-page tracebacks to console.
    """
    for name in ("google_genai", "google_adk", "google.adk", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def is_transient_error(exc: Exception | None) -> bool:
    """Detects whether an exception represents a retryable transient error:
    - 429 RESOURCE_EXHAUSTED / temporary capacity contention
    - 503 UNAVAILABLE / service overload
    - 504 DEADLINE_EXCEEDED / gateway timeout
    - 500 INTERNAL SERVER ERROR / 502 BAD GATEWAY
    - HTTPX / HTTPCore network and connection errors (ConnectError, getaddrinfo, timeouts)
    - Socket and DNS resolution drops (socket.gaierror, socket.timeout)
    - OAuth2 transport errors & SSL/TLS drops (SSLEOFError, ConnectionResetError)
    """
    if exc is None:
        return False

    err_str = str(exc).upper()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

    # 402 Prepayment depleted is NEVER retryable
    if code == 402 or "402" in err_str or "PREPAYMENT CREDITS" in err_str:
        return False

    direct_types = (
        ssl.SSLError,
        ConnectionError,
        TimeoutError,
        OSError,
        socket.gaierror,
        socket.timeout,
        socket.error,
    ) + HTTPX_NETWORK_ERRORS + HTTPCORE_NETWORK_ERRORS

    if isinstance(exc, direct_types):
        return True

    exc_type_name = type(exc).__name__
    transient_type_names = (
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "NetworkError",
        "TransportError",
        "RemoteProtocolError",
        "ProtocolError",
        "SocketError",
        "gaierror",
    )
    if any(t in exc_type_name for t in transient_type_names):
        return True

    if code in (429, 500, 502, 503, 504):
        return True

    full_diagnostic = f"{exc_type_name}: {err_str}"
    transient_keywords = [
        # Resource Exhaustion / Capacity
        "429",
        "RESOURCE_EXHAUSTED",
        "RESOURCE EXHAUSTED",
        "QUOTA",
        "CAPACITY",
        # Server errors
        "500",
        "INTERNAL SERVER ERROR",
        "502",
        "BAD GATEWAY",
        "503",
        "UNAVAILABLE",
        "SERVICE UNAVAILABLE",
        "504",
        "DEADLINE_EXCEEDED",
        "GATEWAY TIMEOUT",
        # Network & DNS drops
        "GETADDRINFO",
        "11001",
        "GAIERROR",
        "NAMERESOLUTIONERROR",
        "NAME RESOLUTION",
        "DNS",
        "SOCKET",
        "NETWORK",
        "CONNECTION",
        "CONNECTERROR",
        "CONNECT ERROR",
        "CONNECTION REFUSED",
        "CONNECTION RESET",
        "CONNECTION CLOSED",
        "REMOTE DISCONNECTED",
        "BROKEN PIPE",
        "RESET BY PEER",
        "TIMEOUT",
        "TIMED OUT",
        "HANDSHAKE",
        "MAX RETRIES EXCEEDED",
        # SSL / TLS
        "SSL",
        "SSLEOFERROR",
        "SSLERROR",
        "UNEXPECTED_EOF",
        "EOF OCCURRED",
        "EOF",
        # HTTP library markers
        "HTTPX",
        "HTTPCORE",
        "TRANSPORTERROR",
        "TRANSPORT ERROR",
    ]
    if any(kw in full_diagnostic for kw in transient_keywords):
        return True

    if getattr(exc, "__cause__", None) and is_transient_error(exc.__cause__):
        return True
    if getattr(exc, "__context__", None) and is_transient_error(exc.__context__):
        return True

    return False


is_resource_exhausted_error = is_transient_error


def get_retry_reason(exc: Exception | None) -> str:
    """Returns a concise, human-readable summary of the failure."""
    if exc is None:
        return "[Unknown Error]"
    err_str = f"{type(exc).__name__}: {str(exc)}".upper()
    cause = getattr(exc, "__cause__", None)
    if cause:
        err_str += f" | CAUSE: {type(cause).__name__}: {str(cause)}".upper()

    if "402" in err_str or "PREPAYMENT CREDITS" in err_str:
        return "[402 Credits Depleted] Google AI Studio prepayment credits depleted. Manage billing at https://ai.studio/projects"
    if "404" in err_str or "NOT_FOUND" in err_str:
        return "[404 Not Found] The specified Gemini model is not found or not available to this API key"
    if any(k in err_str for k in ["GETADDRINFO", "11001", "DNS", "NAMERESOLUTION"]):
        return "[DNS Resolution Drop] Temporary network/DNS failure ([Errno 11001] getaddrinfo failed)"
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "RESOURCE EXHAUSTED" in err_str:
        return "[429 Resource Exhausted] Temporary capacity contention on Google APIs"
    if any(k in err_str for k in ["EOF", "SSL", "TRANSPORTERROR", "CONNECTION", "CONNECT", "DISCONNECTED", "SOCKET", "RESET BY PEER", "BROKEN PIPE"]):
        return "[Network/SSL Drop] Transient connection error to Google APIs (socket/network drop)"
    if "503" in err_str or "UNAVAILABLE" in err_str:
        return "[503 Service Unavailable] Google API service temporarily overloaded"
    if "504" in err_str or "DEADLINE_EXCEEDED" in err_str or "TIMED OUT" in err_str or "TIMEOUT" in err_str:
        return "[Gateway Timeout] Request deadline exceeded"
    if "500" in err_str or "INTERNAL SERVER ERROR" in err_str:
        return "[500 Internal Server Error] Google API internal server error"
    return f"[API Error] {type(exc).__name__}: {str(exc)[:120]}"


def calculate_backoff(attempt: int, base: float = 6.0, max_delay: float = 120.0) -> float:
    """Calculates exponential backoff with jitter: min(max_delay, base * 2^(attempt-1)) + jitter.
    Attempt 1: ~7s
    Attempt 2: ~13s
    Attempt 3: ~26s
    Attempt 4: ~50s
    Attempt 5: ~98s
    Attempt 6: ~120s
    """
    calculated = min(max_delay, base * (2 ** (attempt - 1)))
    jitter = random.uniform(1.0, 3.0)
    return calculated + jitter

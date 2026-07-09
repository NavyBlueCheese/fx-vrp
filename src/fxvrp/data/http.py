"""Minimal HTTP seam so network code is testable with stub transports."""

from __future__ import annotations

import time
from typing import Protocol

from fxvrp.log import get_logger

logger = get_logger("data.http")


class Response(Protocol):
    """The subset of ``requests.Response`` the pipeline relies on."""

    status_code: int

    @property
    def content(self) -> bytes: ...

    @property
    def text(self) -> str: ...

    def raise_for_status(self) -> None: ...


class Transport(Protocol):
    """The subset of ``requests.Session`` the pipeline relies on."""

    def get(self, url: str, timeout: float) -> Response: ...


def get_with_retries(
    transport: Transport,
    url: str,
    *,
    timeout: float,
    max_retries: int,
    backoff_s: float,
) -> bytes:
    """GET with exponential backoff; raises the last error after ``max_retries``."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = transport.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                wait = backoff_s * 2**attempt
                logger.warning(
                    "GET %s failed (%s); retry %d in %.1fs", url, error, attempt + 1, wait
                )
                time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {max_retries + 1} attempts") from last_error

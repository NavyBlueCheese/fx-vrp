from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fxvrp.data.http import get_with_retries


@dataclass
class _FlakyResponse:
    body: bytes
    fail: bool

    status_code: int = 200

    @property
    def content(self) -> bytes:
        return self.body

    @property
    def text(self) -> str:
        return self.body.decode()

    def raise_for_status(self) -> None:
        if self.fail:
            raise RuntimeError("HTTP 503")


@dataclass
class _FlakyTransport:
    failures_before_success: int
    attempts: int = 0
    urls: list[str] = field(default_factory=list)

    def get(self, url: str, timeout: float) -> _FlakyResponse:
        self.urls.append(url)
        self.attempts += 1
        return _FlakyResponse(b"payload", fail=self.attempts <= self.failures_before_success)


def test_retries_then_succeeds() -> None:
    transport = _FlakyTransport(failures_before_success=2)
    body = get_with_retries(transport, "https://x/y", timeout=1.0, max_retries=3, backoff_s=0.0)
    assert body == b"payload"
    assert transport.attempts == 3


def test_exhausted_retries_raise_with_cause() -> None:
    transport = _FlakyTransport(failures_before_success=10)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        get_with_retries(transport, "https://x/y", timeout=1.0, max_retries=2, backoff_s=0.0)
    assert transport.attempts == 3

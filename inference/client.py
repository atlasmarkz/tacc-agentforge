"""Cancellable, retrying chat client for the TACC-hosted vLLM service (standard library only).

Design points (see docs/ in the reliability plan):
- One total time budget per call, shared by all attempts and back-off sleeps.
- The service address is re-read before every attempt, so a restarted service on a new node is
  picked up without restarting the caller (address file written by the service job).
- Only transient failures are retried: connection errors, 429, 5xx. Other 4xx fail at once.
- A CancelToken closes the open connection, which makes vLLM abort the request and free its slot.
- Errors are typed; callers never have to parse strings.
"""
from __future__ import annotations

import http.client
import json
import random
import select
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit


class ModelClientError(Exception):
    code = "model_error"
    retryable = False


class ServiceUnavailable(ModelClientError):
    """Could not reach a working service within the budget (connection refused, 5xx, 429)."""
    code = "model_unavailable"
    retryable = True


class ServiceTimeout(ModelClientError):
    """The total time budget ran out (slow generation, or the service never came up)."""
    code = "model_timeout"
    retryable = True


class ServiceRejected(ModelClientError):
    """HTTP 4xx other than 429: the request itself is wrong; retrying cannot help."""
    code = "model_request_rejected"


class BadResponse(ModelClientError):
    """The service answered 200 but the body is not a usable chat completion."""
    code = "model_bad_response"


class Cancelled(ModelClientError):
    code = "cancelled"


class CancelToken:
    """Thread-safe cancellation: closes the connection currently in use and wakes any back-off sleep."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._conn: http.client.HTTPConnection | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            conn = self._conn
        if conn is not None:
            _close_quietly(conn)

    def attach(self, conn: http.client.HTTPConnection | None) -> None:
        with self._lock:
            self._conn = conn
        if conn is not None and self._event.is_set():
            _close_quietly(conn)

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds`; return True if cancelled meanwhile."""
        return self._event.wait(max(0.0, seconds))


def _close_quietly(conn: http.client.HTTPConnection) -> None:
    try:
        sock = conn.sock
        if sock is not None:
            try:
                import socket
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        conn.close()
    except Exception:
        pass


@dataclass
class ChatResult:
    content: str
    reasoning: str | None = None
    finish_reason: str = "stop"          # "stop" | "length" | "error"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_s: float = 0.0
    attempts: int = 1
    served_by: str = ""                  # base URL that produced the answer
    raw_finish_reason: str | None = None


_RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class ModelClient:
    """`base_url` is used when no address file is given; `address_file` holds one URL (e.g. http://host:8000/v1)."""
    base_url: str | None = None
    address_file: Path | None = None
    model: str = "qwen3.8-27b"
    max_attempts: int = 4
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    connect_timeout_s: float = 5.0
    rng: random.Random = field(default_factory=random.Random)
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if not self.base_url and not self.address_file:
            raise ValueError("give base_url or address_file")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def _resolve_url(self) -> str | None:
        if self.address_file is not None:
            try:
                text = Path(self.address_file).read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text:
                return text.rstrip("/")
        return self.base_url.rstrip("/") if self.base_url else None

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        json_schema: dict | None = None,
        enable_thinking: bool | None = None,
        extra_body: dict | None = None,
        cancel: CancelToken | None = None,
    ) -> ChatResult:
        token = cancel or CancelToken()
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": temperature, "stream": False}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "action", "schema": json_schema, "strict": True}}
        if enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if extra_body:
            body.update(extra_body)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

        start = self.clock()
        deadline = start + timeout_s
        last_error = "no attempt made"
        attempt = 0
        while True:
            if token.cancelled:
                raise Cancelled("cancelled by caller")
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise ServiceTimeout(f"time budget of {timeout_s:.1f}s exhausted ({last_error})")
            attempt += 1
            url = self._resolve_url()
            try:
                if url is None:
                    raise _Retry("no service address yet")
                result = self._once(url, payload, remaining, token)
                result.attempts = attempt
                result.duration_s = self.clock() - start
                return result
            except _Retry as exc:
                last_error = str(exc)
            if token.cancelled:
                raise Cancelled("cancelled by caller")
            if attempt >= self.max_attempts:
                raise ServiceUnavailable(f"gave up after {attempt} attempts ({last_error})")
            delay = self.rng.uniform(0, min(self.backoff_max_s, self.backoff_base_s * 2 ** (attempt - 1)))
            delay = min(delay, max(0.0, deadline - self.clock()))
            if token.wait(delay):
                raise Cancelled("cancelled by caller")

    def _once(self, url: str, payload: bytes, remaining: float, token: CancelToken) -> ChatResult:
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise ServiceRejected(f"unsupported service address {url!r}")
        started = self.clock()
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80,
                                          timeout=max(0.05, min(self.connect_timeout_s, remaining)))
        token.attach(conn)
        try:
            try:
                try:
                    conn.connect()
                except TimeoutError as exc:      # unreachable host: transient, try again
                    raise _Retry("connect timed out") from exc
                left = remaining - (self.clock() - started)
                if left <= 0:
                    raise TimeoutError
                sock = conn.sock
                sock.settimeout(left)
                conn.request("POST", parts.path.rstrip("/") + "/chat/completions", body=payload,
                             headers={"Content-Type": "application/json"})
                # Wait in short slices so a cancel (or the deadline) takes effect promptly on every
                # platform; closing a socket from another thread does not reliably wake a blocked read.
                while True:
                    if token.cancelled:
                        raise Cancelled("cancelled by caller")
                    left = remaining - (self.clock() - started)
                    if left <= 0:
                        raise TimeoutError
                    readable, _, _ = select.select([sock], [], [], min(0.2, left))
                    if readable:
                        break
                if token.cancelled:
                    raise Cancelled("cancelled by caller")
                conn.sock.settimeout(max(0.05, remaining - (self.clock() - started)))
                resp = conn.getresponse()
                data = resp.read()
            except _Retry:
                raise
            except TimeoutError as exc:          # socket timeout == budget gone
                if token.cancelled:
                    raise Cancelled("cancelled by caller") from exc
                raise ServiceTimeout("no answer within the time budget") from exc
            except (OSError, http.client.HTTPException, ValueError, AttributeError) as exc:
                if token.cancelled:
                    raise Cancelled("cancelled by caller") from exc
                raise _Retry(f"{type(exc).__name__}: {exc}") from exc
            if resp.status in _RETRY_STATUS:
                raise _Retry(f"HTTP {resp.status}")
            if resp.status != 200:
                raise ServiceRejected(f"HTTP {resp.status}: {data[:200].decode('utf-8', 'replace')}")
            return _parse(data, url)
        finally:
            token.attach(None)
            _close_quietly(conn)


class _Retry(Exception):
    """Internal: a transient failure of one attempt."""


def _parse(data: bytes, url: str) -> ChatResult:
    try:
        obj = json.loads(data)
        choice = obj["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise BadResponse("response is not a chat completion") from exc
    raw = choice.get("finish_reason")
    usage = obj.get("usage") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return ChatResult(
        content=message.get("content") or "",
        reasoning=reasoning or None,
        finish_reason={"stop": "stop", "length": "length"}.get(raw, "error"),
        raw_finish_reason=raw,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        served_by=url,
    )

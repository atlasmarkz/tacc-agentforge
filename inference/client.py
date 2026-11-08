"""Cancellable, retrying chat client for the TACC-hosted vLLM service (standard library only).

Design points (see docs/ in the reliability plan):
- One total time budget per call, shared by all attempts and back-off sleeps.
- The service address is re-read before every attempt, so a restarted service on a new node is
  picked up without restarting the caller (address file written by the service job).
- Only transient failures are retried: connection errors, 429, 5xx. Other 4xx fail at once.
- A CancelToken closes the open connection, which makes vLLM abort the request and free its slot.
- Errors are typed; callers never have to parse strings.
- Streaming: pass `on_delta(kind, text)` to `chat` and the answer arrives token by token over SSE
  (`kind` is "content" or "reasoning"). A failed attempt is retried only while nothing has been
  delivered; after the first delta a broken stream raises StreamInterrupted (retrying would repeat text).
"""
from __future__ import annotations

import codecs
import http.client
import json
import logging
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


class StreamInterrupted(ModelClientError):
    """A stream broke after text was already delivered; not retried here because that would repeat it."""
    code = "model_unavailable"
    retryable = True

logger = logging.getLogger(__name__)


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
    first_token_s: float | None = None   # streaming only: call start -> first delta (includes retries)


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
        on_delta: Callable[[str, str], None] | None = None,
    ) -> ChatResult:
        token = cancel or CancelToken()
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": temperature, "stream": on_delta is not None}
        if on_delta is not None:
            body["stream_options"] = {"include_usage": True}
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
                if on_delta is None:
                    result = self._once(url, payload, remaining, token)
                else:
                    result = self._once_stream(url, payload, remaining, token, on_delta, start)
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

    def _once_stream(self, url, payload, remaining, token, on_delta, call_start):
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise ServiceRejected(f"unsupported service address {url!r}")
        started = self.clock()
        deadline = started + remaining
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80,
                                          timeout=max(0.05, min(self.connect_timeout_s, remaining)))
        token.attach(conn)
        state = _StreamState()
        try:
            try:
                try:
                    conn.connect()
                except TimeoutError as exc:
                    raise _Retry("connect timed out") from exc
                sock = conn.sock
                head = (f"POST {parts.path.rstrip('/')}/chat/completions HTTP/1.1\r\n"
                        f"Host: {parts.hostname}:{parts.port or 80}\r\nContent-Type: application/json\r\n"
                        f"Accept: text/event-stream\r\nConnection: close\r\n"
                        f"Content-Length: {len(payload)}\r\n\r\n").encode("ascii")
                sock.settimeout(max(0.05, deadline - self.clock()))
                sock.sendall(head + payload)
                return self._read_stream(sock, url, deadline, token, on_delta, call_start, state)
            except (_Retry, ModelClientError):
                raise
            except TimeoutError as exc:
                if token.cancelled:
                    raise Cancelled("cancelled by caller") from exc
                raise ServiceTimeout("no complete answer within the time budget") from exc
            except (OSError, http.client.HTTPException, ValueError, AttributeError) as exc:
                if token.cancelled:
                    raise Cancelled("cancelled by caller") from exc
                _fail(state, f"{type(exc).__name__}: {exc}")
        finally:
            token.attach(None)
            _close_quietly(conn)


    def _recv(self, sock, deadline, token):
        """Next bytes from the socket, or None at EOF. Waits in short slices so cancel and deadline are prompt."""
        while True:
            if token.cancelled:
                raise Cancelled("cancelled by caller")
            left = deadline - self.clock()
            if left <= 0:
                raise TimeoutError
            readable, _, _ = select.select([sock], [], [], min(0.2, left))
            if readable:
                data = sock.recv(65536)
                return data or None


    def _emit(self, on_delta, kind, text, state, call_start):
        if not state.delivered:
            state.delivered = True
            state.first_token_s = self.clock() - call_start
        try:
            on_delta(kind, text)
        except Exception:                      # a broken consumer must not kill the generation
            logger.warning("on_delta callback failed", exc_info=True)


    def _read_stream(self, sock, url, deadline, token, on_delta, call_start, state):
        head_buf = b""
        status = None
        dechunk = None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        error_body = b""
        content, reasoning = [], []
        finish, usage, finished = None, {}, False
        while not finished:
            data = self._recv(sock, deadline, token)
            if data is None:
                break
            if status is None:
                head_buf += data
                end = head_buf.find(b"\r\n\r\n")
                if end < 0:
                    if len(head_buf) > 65536:
                        raise _Retry("oversized response header")
                    continue
                lines = head_buf[:end].decode("latin-1").split("\r\n")
                data = head_buf[end + 4:]
                try:
                    status = int(lines[0].split()[1])
                except (IndexError, ValueError):
                    raise _Retry("bad status line")
                headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
                if status in _RETRY_STATUS:
                    raise _Retry(f"HTTP {status}")
                if "chunked" in headers.get("transfer-encoding", "").lower():
                    dechunk = _Dechunker()
            body = dechunk.feed(data) if dechunk else data
            if status != 200:
                error_body = (error_body + body)[:4096]
                if dechunk and dechunk.done:
                    break
                continue
            pending = (pending + decoder.decode(body)).replace("\r\n", "\n")
            while "\n\n" in pending:
                event, pending = pending.split("\n\n", 1)
                lines = [ln[5:].lstrip() for ln in event.split("\n") if ln.startswith("data:")]
                if not lines:
                    continue
                payload = "\n".join(lines)
                if payload == "[DONE]":
                    finished = True
                    break
                try:
                    obj = json.loads(payload)
                except ValueError as exc:
                    raise BadResponse("stream event is not JSON") from exc
                if isinstance(obj, dict) and obj.get("error") and not obj.get("choices"):
                    _fail(state, f"engine error event: {str(obj['error'])[:120]}")
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    thought = delta.get("reasoning_content") or delta.get("reasoning")
                    text = delta.get("content")
                    if thought:
                        reasoning.append(thought)
                        self._emit(on_delta, "reasoning", thought, state, call_start)
                    if text:
                        content.append(text)
                        self._emit(on_delta, "content", text, state, call_start)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                usage = obj.get("usage") or usage
            if dechunk and dechunk.done:
                break
        if status is None:
            raise _Retry("connection closed before a response")
        if status != 200:
            raise ServiceRejected(f"HTTP {status}: {error_body[:200].decode('utf-8', 'replace')}")
        if not finished and finish is None:
            _fail(state, "stream ended before the completion finished")
        return ChatResult(
            content="".join(content), reasoning="".join(reasoning) or None,
            finish_reason={"stop": "stop", "length": "length"}.get(finish, "error"), raw_finish_reason=finish,
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
            served_by=url, first_token_s=state.first_token_s,
        )


class _Retry(Exception):
    """Internal: a transient failure of one attempt."""


class _StreamState:
    def __init__(self) -> None:
        self.delivered = False
        self.first_token_s: float | None = None


class _Dechunker:
    """Incremental HTTP chunked-transfer decoder (whole chunks only; SSE servers send one event per chunk)."""

    def __init__(self) -> None:
        self.buf = b""
        self.done = False

    def feed(self, data: bytes) -> bytes:
        self.buf += data
        out = []
        while not self.done:
            end = self.buf.find(b"\r\n")
            if end < 0:
                break
            size = int(self.buf[:end].split(b";")[0].strip() or b"0", 16)
            if size == 0:
                self.done = True
                break
            if len(self.buf) < end + 2 + size + 2:
                break
            out.append(self.buf[end + 2:end + 2 + size])
            self.buf = self.buf[end + 2 + size + 2:]
        return b"".join(out)


def _fail(state: _StreamState, message: str):
    if state.delivered:
        raise StreamInterrupted(f"stream broke after output was delivered ({message})")
    raise _Retry(message)


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

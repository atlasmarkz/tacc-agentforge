"""ModelPort adapter: lets the agent harness use the vLLM service through `inference.client`.

Contract: a schema requests a JSON action; action_schema=None requests free text.
Per-call delta hooks and the optional legacy monitoring hook run on the event loop.

Cancellation: the harness cancels the awaiting task on timeout or stop; we then close the HTTP connection so
vLLM aborts the generation and frees its slot.
"""
from __future__ import annotations

import asyncio
import json
import logging

from agent.contracts import CallContext, CapabilityError, ModelReply, ModelRequest

from typing import Callable

from .client import (
    BadResponse, Cancelled, CancelToken, ModelClient, ModelClientError, ServiceRejected,
    ServiceTimeout, ServiceUnavailable, StreamInterrupted,
)

# (operation_id, kind, text) with kind "content" or "reasoning"
DeltaSink = Callable[[str, str, str], None]
logger = logging.getLogger(__name__)

_PUBLIC_MESSAGES = {
    StreamInterrupted: ("model_unavailable", "The model stream was interrupted.", True),
    ServiceUnavailable: ("model_unavailable", "The model service is unavailable.", True),
    ServiceTimeout: ("model_timeout", "The model service did not answer in time.", True),
    ServiceRejected: ("model_request_rejected", "The model service rejected the request.", False),
    BadResponse: ("model_bad_response", "The model service returned an unusable response.", False),
}


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _anchor(pattern: str) -> str:
    """JSON Schema `pattern` is an unanchored search ("contains a match"), but constrained decoders
    (xgrammar/outlines) implement it as a full match. A schema pattern like `\\S` ("has a non-blank
    character") would then allow exactly one character: ask_user questions came out as "您" or "请".
    Making the implicit `.*` explicit gives both readings the same meaning."""
    if not pattern.startswith("^"):
        pattern = ".*" + pattern
    if not pattern.endswith("$"):
        pattern = pattern + ".*"
    return pattern


def decoder_schema(schema):
    """Copy of a JSON schema with every `pattern` anchored, for use as the decoding constraint only.
    The harness still validates the reply against its own, unmodified schema."""
    if isinstance(schema, dict):
        return {k: (_anchor(v) if k == "pattern" and isinstance(v, str) else decoder_schema(v))
                for k, v in schema.items()}
    if isinstance(schema, list):
        return [decoder_schema(v) for v in schema]
    return schema


def to_chat_messages(request: ModelRequest) -> list[dict]:
    """One leading system message (chat templates require it first) that also carries the tool list and
    the action schema. The text is deterministic, so it stays a stable prefix for prefix caching."""
    messages = [m.model_dump() for m in request.messages]
    if request.action_schema is None:
        return messages
    extra = ("\n\nRegistered tools (JSON):\n" + _compact([t.model_dump() for t in request.tools])
             + "\n\nAction JSON Schema:\n" + _compact(request.action_schema))
    if messages and messages[0]["role"] == "system":
        messages[0] = {"role": "system", "content": messages[0]["content"] + extra}
    else:
        messages.insert(0, {"role": "system", "content": extra.lstrip()})
    return messages


class VllmModelPort:
    def __init__(
        self,
        client: ModelClient,
        *,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
        constrain_output: bool = True,
        timeout_margin_s: float = 0.5,
        on_delta: DeltaSink | None = None,
    ) -> None:
        """`on_delta`, when given, makes every call stream: it is invoked on the event-loop thread with
        (operation_id, kind, text) as tokens arrive. operation_id is "<run_id>:model:<n>", so an
        application can route deltas to the right browser session. generate() still returns the
        complete, validated-by-the-harness reply at the end."""
        self.client = client
        self.on_delta = on_delta
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.constrain_output = constrain_output
        self.timeout_margin_s = timeout_margin_s

    async def generate(self, request: ModelRequest, context: CallContext) -> ModelReply:
        token = CancelToken()
        # Finish slightly before the harness's own timer so our typed error wins over its generic one.
        budget = max(0.5, context.timeout_s - self.timeout_margin_s)
        loop = asyncio.get_running_loop()
        active = True
        forward = None
        legacy, scoped = self.on_delta, context.on_delta
        if legacy is not None or scoped is not None:
            def deliver(kind: str, text: str):
                if not active or context.cancellation.is_set():
                    return
                for sink, arguments in ((scoped, (kind, text)),
                                        (legacy, (context.operation_id, kind, text))):
                    if sink is not None:
                        try:
                            sink(*arguments)
                        except Exception:
                            logger.warning("Model delta callback failed", exc_info=True)

            def forward(kind: str, text: str) -> None:          # runs in the worker thread
                try:
                    loop.call_soon_threadsafe(deliver, kind, text)
                except RuntimeError:                              # loop already closed: run is over
                    pass
        call = asyncio.create_task(asyncio.to_thread(
            self.client.chat,
            to_chat_messages(request),
            max_tokens=request.max_output_tokens,
            temperature=self.temperature,
            timeout_s=budget,
            json_schema=decoder_schema(request.action_schema) if self.constrain_output and request.action_schema is not None else None,
            enable_thinking=False if request.action_schema is None and self.enable_thinking is None else self.enable_thinking,
            cancel=token,
            on_delta=forward,
        ))
        stop = asyncio.create_task(context.cancellation.wait())
        try:
            done, _ = await asyncio.wait({call, stop}, return_when=asyncio.FIRST_COMPLETED)
            if call not in done:
                token.cancel()
                raise CapabilityError("cancelled", "The request was cancelled.", recoverable=False)
            result = call.result()
            if forward is not None:
                # The client queues all deltas before returning. A loop barrier drains them
                # before generate returns, without callers adding arbitrary sleeps.
                flushed = loop.create_future()

                def flush_barrier():
                    if not flushed.done():
                        flushed.set_result(None)

                loop.call_soon(flush_barrier)
                await flushed
        except asyncio.CancelledError:
            token.cancel()
            raise
        except Cancelled:
            raise CapabilityError("cancelled", "The request was cancelled.", recoverable=False)
        except ModelClientError as exc:
            code, message, recoverable = _PUBLIC_MESSAGES.get(
                type(exc), ("model_error", "The model call failed.", False))
            raise CapabilityError(code, message, recoverable=recoverable) from exc
        finally:
            active = False
            stop.cancel()
            if not call.done():
                token.cancel()
                call.cancel()
            await asyncio.gather(stop, call, return_exceptions=True)
        too_long = len(result.content) > 65536
        return ModelReply(
            content=result.content[:65536],
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason="length" if too_long else result.finish_reason,
        )

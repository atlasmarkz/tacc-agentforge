"""ModelPort adapter: lets the agent harness use the vLLM service through `inference.client`.

Contract (agent/contracts.py): `generate(request, context)` returns a `ModelReply` whose `content` is one JSON
action matching `request.action_schema`. The harness passes tools and the schema as structured data, not as
chat text, so this adapter is responsible for showing them to the model and for constraining the output.

Cancellation: the harness cancels the awaiting task on timeout or stop; we then close the HTTP connection so
vLLM aborts the generation and frees its slot.
"""
from __future__ import annotations

import asyncio
import json

from agent.contracts import CallContext, CapabilityError, ModelReply, ModelRequest

from typing import Callable

from .client import (
    BadResponse, Cancelled, CancelToken, ModelClient, ModelClientError, ServiceRejected,
    ServiceTimeout, ServiceUnavailable, StreamInterrupted,
)

# (operation_id, kind, text) with kind "content" or "reasoning"
DeltaSink = Callable[[str, str, str], None]

_PUBLIC_MESSAGES = {
    StreamInterrupted: ("model_unavailable", "The model stream was interrupted.", True),
    ServiceUnavailable: ("model_unavailable", "The model service is unavailable.", True),
    ServiceTimeout: ("model_timeout", "The model service did not answer in time.", True),
    ServiceRejected: ("model_request_rejected", "The model service rejected the request.", False),
    BadResponse: ("model_bad_response", "The model service returned an unusable response.", False),
}


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def to_chat_messages(request: ModelRequest) -> list[dict]:
    """One leading system message (chat templates require it first) that also carries the tool list and
    the action schema. The text is deterministic, so it stays a stable prefix for prefix caching."""
    messages = [m.model_dump() for m in request.messages]
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
        forward = None
        if self.on_delta is not None:
            loop = asyncio.get_running_loop()
            sink, operation_id = self.on_delta, context.operation_id

            def forward(kind: str, text: str) -> None:          # runs in the worker thread
                try:
                    loop.call_soon_threadsafe(sink, operation_id, kind, text)
                except RuntimeError:                              # loop already closed: run is over
                    pass
        call = asyncio.create_task(asyncio.to_thread(
            self.client.chat,
            to_chat_messages(request),
            max_tokens=request.max_output_tokens,
            temperature=self.temperature,
            timeout_s=budget,
            json_schema=request.action_schema if self.constrain_output else None,
            enable_thinking=self.enable_thinking,
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
            stop.cancel()
            if not call.done():
                token.cancel()
        too_long = len(result.content) > 65536
        return ModelReply(
            content=result.content[:65536],
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason="length" if too_long else result.finish_reason,
        )

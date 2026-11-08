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

from .client import (
    BadResponse, Cancelled, CancelToken, ModelClient, ModelClientError, ServiceRejected,
    ServiceTimeout, ServiceUnavailable,
)

_PUBLIC_MESSAGES = {
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
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.constrain_output = constrain_output
        self.timeout_margin_s = timeout_margin_s

    async def generate(self, request: ModelRequest, context: CallContext) -> ModelReply:
        token = CancelToken()
        # Finish slightly before the harness's own timer so our typed error wins over its generic one.
        budget = max(0.5, context.timeout_s - self.timeout_margin_s)
        call = asyncio.create_task(asyncio.to_thread(
            self.client.chat,
            to_chat_messages(request),
            max_tokens=request.max_output_tokens,
            temperature=self.temperature,
            timeout_s=budget,
            json_schema=request.action_schema if self.constrain_output else None,
            enable_thinking=self.enable_thinking,
            cancel=token,
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

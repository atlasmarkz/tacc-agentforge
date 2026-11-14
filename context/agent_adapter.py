"""Optional conversions. Only to_model_request imports Agent/Pydantic dependencies."""
from __future__ import annotations

import json

from .models import ContextRequest, HistoryBlock, identifier


def request_from_agent(request, *, system_prompt, tools, action_schema,
                       model_window_tokens, max_output_tokens, **options):
    """Accept an AgentRequest or JSON dictionary. Persist constraints separately via update_memory."""
    raw = request.model_dump(mode="json") if hasattr(request, "model_dump") else request
    descriptions = [t.model_dump(mode="json") if hasattr(t, "model_dump") else t for t in tools]
    return ContextRequest(conversation_id=raw["conversation_id"], run_id=raw["run_id"],
                          snapshot_id=raw["snapshot_id"], message=raw["message"],
                          system_prompt=system_prompt, tools=descriptions, action_schema=action_schema,
                          model_window_tokens=model_window_tokens,
                          max_output_tokens=max_output_tokens, **options)


def history_block_from_agent(messages, *, block_id, run_id, snapshot_id,
                             constraints_revision):
    """Convert one caller-grouped exchange; reject unfinished or unrelated tool pairs.

    This deliberately does not split arbitrary AgentState.history or infer boundaries.
    """
    messages = [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]
    first = None
    if messages and messages[0]["role"] == "assistant":
        try:
            first = json.loads(messages[0]["content"])
        except ValueError:
            pass
    if isinstance(first, dict) and first.get("type") == "call_tool":
        if len(messages) != 2 or messages[1]["role"] != "user":
            raise ValueError("Tool blocks require one action and its observation.")
        try:
            observation = json.loads(messages[1]["content"])["observation"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("Expected an Agent tool observation.") from exc
        if not isinstance(observation, dict):
            raise ValueError("Expected an Agent tool observation object.")
        if observation.get("status") == "ok":
            if not {"tool", "snapshot_id", "operation_id", "data"} <= observation.keys():
                raise ValueError("Successful tool observations require identity and data fields.")
            identifier(observation["operation_id"])
            if not isinstance(observation["data"], dict):
                raise ValueError("Successful tool observation data must be an object.")
            if observation["tool"] != first["tool_name"]:
                raise ValueError("Tool observation does not match action.")
            if observation["snapshot_id"] != snapshot_id or observation["data"].get("snapshot_id", snapshot_id) != snapshot_id:
                raise ValueError("Tool observation does not match snapshot.")
        elif observation.get("status") == "error":
            if not all(isinstance(observation.get(key), str) and observation[key]
                       for key in ("code", "message")):
                raise ValueError("Tool errors require a code and message.")
        else:
            raise ValueError("Tool observations require an ok or error status.")
    references = set()
    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"offer_id", "evidence_id", "variant_id"} and isinstance(child, str):
                    references.add(child)
                elif key in {"offer_ids", "evidence_ids", "variant_ids"} and isinstance(child, list):
                    references.update(x for x in child if isinstance(x, str))
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    for message in messages:
        try:
            collect(json.loads(message["content"]))
        except ValueError:
            pass
    kind, items = "exchange", []
    if isinstance(first, dict) and first.get("type") == "propose_plan":
        kind, items = "plan", first["plan"]["items"]
    elif isinstance(first, dict) and first.get("type") == "ask_user":
        kind = "question"
    return HistoryBlock(block_id, run_id, snapshot_id, constraints_revision, messages,
                        kind=kind, references=tuple(sorted(references)), plan_items=items)


def to_model_request(bundle):
    """Return the unrendered Agent envelope; its adapter injects the schema exactly once."""
    from agent.contracts import ModelRequest
    return ModelRequest.model_validate(bundle.payload)

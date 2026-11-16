"""Optional conversions. Only to_model_request imports Agent/Pydantic dependencies."""
from __future__ import annotations

import json

from .models import ContextRequest, HistoryBlock, identifier


def request_from_agent(request, *, expected_revision, system_prompt, tools, action_schema,
                       model_window_tokens, max_output_tokens, **options):
    """Bind an Agent request to the stored revision and its original constraints.

    The host must persist the same constraints before building; this helper never
    merges or silently drops AgentRequest.constraints.
    """
    raw = request.model_dump(mode="json") if hasattr(request, "model_dump") else request
    if "expected_constraints" in options:
        raise ValueError("request_from_agent derives expected_constraints from the Agent request.")
    constraints = raw.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("Agent request constraints must be an object.")
    descriptions = [t.model_dump(mode="json") if hasattr(t, "model_dump") else t for t in tools]
    return ContextRequest(conversation_id=raw["conversation_id"], run_id=raw["run_id"],
                          snapshot_id=raw["snapshot_id"], message=raw["message"],
                          system_prompt=system_prompt, tools=descriptions, action_schema=action_schema,
                          model_window_tokens=model_window_tokens,
                          max_output_tokens=max_output_tokens,
                          expected_revision=expected_revision,
                          expected_constraints=constraints, **options)


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
        plan = first.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("propose_plan requires a plan object.")
        items = plan.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("propose_plan requires a nonempty plan items list.")
        if any(not isinstance(item, dict) or not isinstance(item.get("evidence_ids"), list)
               or not item["evidence_ids"] for item in items):
            raise ValueError("propose_plan items require nonempty evidence_ids.")
        kind = "plan"
    elif isinstance(first, dict) and first.get("type") == "ask_user":
        kind = "question"
    return HistoryBlock(block_id, run_id, snapshot_id, constraints_revision, messages,
                        kind=kind, references=tuple(sorted(references)), plan_items=items)


def _as_json_object(value, name):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def validated_plan_block_from_agent(result, request, *, block_id, constraints_revision):
    """Convert a completed Agent result into a trusted, passed plan block.

    Only a trusted host should call this after receiving the actual completed
    result. Ordinary text, tool observations, or the word "PASSED" never establish
    validation. The block remains append-only; earlier blocks are not rewritten.
    """
    result = _as_json_object(result, "Agent result")
    request = _as_json_object(request, "Agent request")
    if result.get("status") != "completed":
        raise ValueError("Only a completed Agent result can provide a validated plan.")
    identity = ("run_id", "conversation_id", "snapshot_id")
    for field in identity:
        identifier(result.get(field))
        if result[field] != request.get(field):
            raise ValueError(f"Agent result and request have different {field}.")
    verified = result.get("verified_plan")
    if not isinstance(verified, dict):
        raise ValueError("Completed Agent results require a verified plan.")
    if verified.get("snapshot_id") != result["snapshot_id"]:
        raise ValueError("Verified plan snapshot does not match the completed result.")
    raw_items = verified.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Verified plans require at least one item.")
    items = []
    for item in raw_items:
        if not isinstance(item, dict) or not {"offer_id", "quantity", "evidence_ids"} <= set(item):
            raise ValueError("Verified plan items must contain offer_id, quantity and evidence_ids.")
        if (not isinstance(item["evidence_ids"], list) or not item["evidence_ids"]):
            raise ValueError("Verified plan items require nonempty evidence_ids.")
        items.append({key: item[key] for key in ("offer_id", "quantity", "evidence_ids")})
    references = tuple(sorted({ref for item in items
                                for ref in [item["offer_id"], *item["evidence_ids"]]}))
    action = {"type": "propose_plan", "plan": {"items": items}}
    return HistoryBlock(block_id, result["run_id"], result["snapshot_id"], constraints_revision,
                        [{"role": "assistant", "content": json.dumps(action, ensure_ascii=False,
                                                                     separators=(",", ":"))}],
                        kind="plan", references=references, plan_items=items,
                        plan_validation="passed")


def to_model_request(bundle):
    """Return the unrendered Agent envelope; its adapter injects the schema exactly once."""
    from agent.contracts import ModelRequest
    return ModelRequest.model_validate(bundle.payload)

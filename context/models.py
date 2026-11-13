"""Standard-library contracts for durable memory and bounded prompt construction."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal


class ContextError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ContextLimitError(ContextError):
    def __init__(self):
        super().__init__("context_limit", "Required context exceeds the input budget.")


class StorageError(ContextError):
    def __init__(self, message="Context storage operation failed."):
        super().__init__("storage_error", message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def clone(value):
    """Validate JSON compatibility and detach nested mutable values."""
    def check(obj):
        if obj is None or type(obj) in (str, bool, int, float):
            return
        if isinstance(obj, list):
            for child in obj:
                check(child)
            return
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
            for child in obj.values():
                check(child)
            return
        raise ValueError("Values must be JSON data with string object keys.")
    check(value)
    return json.loads(canonical(value))


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ValueError("Invalid identifier.")


def nonnegative(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer.")


def sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")


@dataclass(frozen=True)
class ContextPatch:
    set: dict = field(default_factory=dict)
    unset: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.set, dict):
            raise ValueError("set must be an object.")
        sequence(self.unset, "unset")
        keys = list(self.set) + list(self.unset)
        reserved = {"run_id", "conversation_id", "snapshot_id", "schema_version", "message",
                    "constraints", "policy", "usage"}
        if any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", k)
               or k in reserved for k in keys):
            raise ValueError("Patch keys must be top-level constraint fields.")
        if set(self.set).intersection(self.unset) or len(set(self.unset)) != len(self.unset):
            raise ValueError("Patch fields must be unique and disjoint.")
        object.__setattr__(self, "set", clone(self.set))
        object.__setattr__(self, "unset", tuple(self.unset))


@dataclass(frozen=True)
class HistoryBlock:
    block_id: str
    run_id: str
    snapshot_id: str
    constraints_revision: int
    messages: list[dict]
    kind: Literal["exchange", "feedback", "plan", "question"] = "exchange"
    references: tuple[str, ...] = ()
    # Plan items are historical hints only, never a validation receipt.
    plan_items: list[dict] = field(default_factory=list)

    def __post_init__(self):
        for value in (self.block_id, self.run_id, self.snapshot_id):
            identifier(value)
        nonnegative(self.constraints_revision)
        sequence(self.references, "references")
        if self.kind not in {"exchange", "feedback", "plan", "question"}:
            raise ValueError("Unknown history kind.")
        if not isinstance(self.messages, list) or not self.messages or any(
                not isinstance(m, dict) or set(m) != {"role", "content"}
                or m["role"] not in {"user", "assistant"} or not isinstance(m["content"], str)
                for m in self.messages):
            raise ValueError("History must contain user/assistant text messages only.")
        for value in self.references:
            identifier(value)
        if not isinstance(self.plan_items, list):
            raise ValueError("plan_items must be a list.")
        if self.plan_items and self.kind != "plan":
            raise ValueError("Only plan blocks may carry plan_items.")
        for item in self.plan_items:
            if not isinstance(item, dict) or set(item) != {"offer_id", "quantity", "evidence_ids"}:
                raise ValueError("Historical plans contain IDs and quantities only.")
            identifier(item["offer_id"])
            if type(item["quantity"]) is not int or item["quantity"] <= 0:
                raise ValueError("Invalid quantity.")
            if not isinstance(item["evidence_ids"], list):
                raise ValueError("Invalid evidence IDs.")
            for value in item["evidence_ids"]:
                identifier(value)
        object.__setattr__(self, "messages", clone(self.messages))
        object.__setattr__(self, "plan_items", clone(self.plan_items))
        object.__setattr__(self, "references", tuple(self.references))


@dataclass(frozen=True)
class ContextRequest:
    conversation_id: str
    run_id: str
    snapshot_id: str
    message: str
    system_prompt: str
    tools: list[dict]
    action_schema: dict
    model_window_tokens: int
    max_output_tokens: int
    mode: Literal["filtered", "unfiltered"] = "filtered"
    safety_tokens: int = 256
    references: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (self.conversation_id, self.run_id, self.snapshot_id):
            identifier(value)
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be nonempty text.")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must be nonempty text.")
        for value in (self.model_window_tokens, self.max_output_tokens):
            nonnegative(value)
            if not value:
                raise ValueError("Token limits must be positive.")
        nonnegative(self.safety_tokens)
        if self.mode not in {"filtered", "unfiltered"}:
            raise ValueError("Unknown selection mode.")
        sequence(self.references, "references")
        for value in self.references:
            identifier(value)
        if (not isinstance(self.tools, list) or any(not isinstance(t, dict) for t in self.tools)
                or not isinstance(self.action_schema, dict)):
            raise ValueError("Invalid tools or action schema.")
        object.__setattr__(self, "tools", clone(self.tools))
        object.__setattr__(self, "action_schema", clone(self.action_schema))
        object.__setattr__(self, "references", tuple(self.references))


@dataclass(frozen=True)
class TokenCount:
    value: int
    mode: Literal["exact", "estimated"]
    counter_id: str

    def __post_init__(self):
        nonnegative(self.value)
        if self.mode not in {"exact", "estimated"} or not self.counter_id:
            raise ValueError("Counter must identify its method and precision.")


@dataclass(frozen=True)
class ContextBundle:
    payload: dict
    rendered_messages: list[dict]
    constraints: dict
    revision: int
    constraints_revision: int
    selected_block_ids: list[str]
    omitted: dict[str, str]
    token_count: TokenCount
    mode: str
    selection_reasons: dict[str, str]

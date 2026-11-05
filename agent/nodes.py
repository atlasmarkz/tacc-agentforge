"""Async graph nodes. External capabilities are injected; no network clients live here."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import (
    ACTION_ADAPTER, AgentDependencies, AgentEvent, AgentRequest, AgentResult,
    CallContext, CallTool, CapabilityError, Issue, ModelReply, PendingOperation,
    PlanValidationRequest, ProposePlan, Usage, ValidationResult,
)
from .policies import RunPolicy
from .prompts import ContextLimitError, build_model_request
from .state import AgentState

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RunStopped(Exception):
    def __init__(self, reason: str, status: str = "incomplete"):
        self.reason, self.status = reason, status


@dataclass
class RunScope:
    request: AgentRequest
    dependencies: AgentDependencies
    policy: RunPolicy
    cancellation: asyncio.Event
    queue: asyncio.Queue
    started: float = field(default_factory=time.monotonic)
    sequence: int = 0

    def check(self):
        if self.cancellation.is_set():
            raise RunStopped("cancelled", "cancelled")
        if self.remaining <= 0:
            raise RunStopped("deadline_exceeded")

    @property
    def remaining(self) -> float:
        return self.policy.deadline_seconds - (time.monotonic() - self.started)

    def emit(self, node: str, event_type: str, *, result=None, **data):
        self.sequence += 1
        event = AgentEvent(
            run_id=self.request.run_id, sequence=self.sequence, node=node,
            type=event_type, data=data, result=result,
        )
        # A run has bounded decisions/calls and emits only small operational events.
        # An unbounded queue avoids deadlock when a stream consumer closes early.
        self.queue.put_nowait(event)
        if self.dependencies.event_sink is not None:
            try:
                self.dependencies.event_sink(event.model_copy(deep=True))
            except Exception:
                logger.warning("Agent event sink failed (event_type=%s)", event_type)

    async def call(
        self, operation_id: str, invoke: Callable[[CallContext], Awaitable[T]],
        timeout_s: float | None = None,
    ) -> T:
        self.check()
        timeout = min(self.remaining, self.policy.call_timeout_s,
                      timeout_s if timeout_s is not None else self.policy.call_timeout_s)
        context = CallContext(
            run_id=self.request.run_id, snapshot_id=self.request.snapshot_id,
            operation_id=operation_id, timeout_s=timeout, cancellation=self.cancellation,
        )
        work = asyncio.create_task(invoke(context))
        cancel = asyncio.create_task(self.cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                [work, cancel], timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
            )
            self.check()
            if work not in done:
                raise CapabilityError("call_timeout", "External call timed out.", recoverable=True)
            result = await work
            self.check()
            return result
        finally:
            for task in (work, cancel):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, cancel, return_exceptions=True)


def stop(state, reason, *, status="incomplete", issue: Issue | None = None):
    return {
        "status": status, "stop_reason": reason, "next_node": "finalize",
        "verified_plan": None, "clarification": None,
        "issues": (state["issues"] + ([issue.model_dump()] if issue else []))[-20:],
    }


def observation(state, data, *, issue: Issue | None = None):
    return {
        "next_node": "guard", "action": None,
        "history": state["history"] + [{
            "role": "user", "content": json.dumps({"observation": data}, ensure_ascii=False),
        }],
        "issues": (state["issues"] + ([issue.model_dump()] if issue else []))[-20:],
    }


class Nodes:
    def __init__(self, scope: RunScope):
        self.scope = scope
        self.deps, self.policy = scope.dependencies, scope.policy

    async def prepare(self, state: AgentState):
        self.scope.emit("prepare", "run_started", snapshot_id=self.scope.request.snapshot_id)
        return {"next_node": "guard"}

    async def guard(self, state: AgentState):
        try:
            self.scope.check()
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        if state["usage"]["model_calls"] >= self.policy.max_model_calls:
            return stop(state, "model_call_limit")
        return {"next_node": "decide"}

    async def decide(self, state: AgentState):
        try:
            self.scope.check()
            request = build_model_request(state, self.deps.tools, self.policy)
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        except ContextLimitError:
            return stop(state, "context_limit")
        usage = dict(state["usage"])
        usage["model_calls"] += 1
        operation_id = f"{self.scope.request.run_id}:model:{usage['model_calls']}"
        self.scope.emit("decide", "decision_started", operation_id=operation_id)
        try:
            reply = await self.scope.call(
                operation_id, lambda context: self.deps.model.generate(request, context),
            )
        except RunStopped as exc:
            return {**stop(state, exc.reason, status=exc.status), "usage": usage}
        except CapabilityError as exc:
            return {**stop(state, exc.issue.code, status="failed", issue=exc.issue), "usage": usage}
        if isinstance(reply, PendingOperation):
            return {**stop(state, "unsupported_capability"), "usage": usage}
        if not isinstance(reply, ModelReply):
            raise TypeError("ModelPort must return ModelReply or PendingOperation")
        usage["prompt_tokens"] += reply.prompt_tokens or 0
        usage["completion_tokens"] += reply.completion_tokens or 0
        usage["unreported_token_calls"] += int(
            reply.prompt_tokens is None or reply.completion_tokens is None)
        try:
            if reply.finish_reason != "stop":
                raise ValueError("incomplete generation")
            action = ACTION_ADAPTER.validate_json(reply.content)
        except (ValidationError, ValueError):
            errors = state["consecutive_action_errors"] + 1
            issue = Issue(code="invalid_action", message="Return one complete action matching the schema.")
            usage["action_repairs"] += 1
            self.scope.emit("decide", "action_rejected", code=issue.code)
            if errors > self.policy.max_action_repairs:
                update = stop(state, "invalid_action", issue=issue)
            else:
                update = observation(state, {"status": "error", **issue.model_dump()}, issue=issue)
            return {**update, "usage": usage, "consecutive_action_errors": errors}
        self.scope.emit("decide", "action_selected", action_type=action.type)
        route = {"call_tool": "execute_tool", "propose_plan": "validate_plan", "ask_user": "suspend"}
        return {
            "action": action.model_dump(mode="json"), "next_node": route[action.type],
            "usage": usage, "consecutive_action_errors": 0, "verified_plan": None,
            "history": state["history"] + [{"role": "assistant", "content": action.model_dump_json()}],
        }

    async def execute_tool(self, state: AgentState):
        action = CallTool.model_validate(state["action"])
        tool = self.deps.tools.get(action.tool_name)
        issue = None
        if tool is None:
            issue = Issue(code="unknown_tool", message="Choose a registered tool.")
        else:
            try:
                arguments = tool.arguments_model.model_validate(action.arguments, strict=True)
            except ValidationError:
                issue = Issue(code="invalid_tool_arguments", message="Arguments must match the tool schema.")
        if issue:
            self.scope.emit("execute_tool", "tool_rejected", tool=action.tool_name, code=issue.code)
            return observation(state, {"status": "error", **issue.model_dump()}, issue=issue)
        try:
            self.scope.check()
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        if state["usage"]["tool_calls"] >= self.policy.max_tool_calls:
            return stop(state, "tool_call_limit")
        usage = dict(state["usage"])
        usage["tool_calls"] += 1
        operation_id = f"{self.scope.request.run_id}:tool:{usage['tool_calls']}"
        self.scope.emit("execute_tool", "tool_started", tool=tool.name, operation_id=operation_id)
        try:
            raw = await self.scope.call(
                operation_id, lambda context: tool.executor.execute(arguments, context), tool.timeout_s,
            )
        except RunStopped as exc:
            return {**stop(state, exc.reason, status=exc.status), "usage": usage}
        except CapabilityError as exc:
            self.scope.emit("execute_tool", "tool_finished", tool=tool.name, status="error", code=exc.issue.code)
            if exc.recoverable:
                update = observation(state, {"status": "error", **exc.issue.model_dump()}, issue=exc.issue)
            else:
                update = stop(state, exc.issue.code, status="failed", issue=exc.issue)
            return {**update, "usage": usage}
        if isinstance(raw, PendingOperation):
            return {**stop(state, "unsupported_capability"), "usage": usage}
        try:
            if isinstance(raw, BaseModel):
                raw = raw.model_dump(mode="json")
            result = tool.result_model.model_validate(raw, strict=True)
            data = result.model_dump(mode="json")
            encoded = result.model_dump_json().encode("utf-8")
        except ValidationError:
            return {**stop(state, "invalid_tool_result", status="failed"), "usage": usage}
        if len(encoded) > tool.max_result_bytes:
            return {**stop(state, "tool_result_too_large", status="failed"), "usage": usage}
        self.scope.emit("execute_tool", "tool_finished", tool=tool.name, status="ok", operation_id=operation_id)
        return {
            **observation(state, {
                "status": "ok", "tool": tool.name, "operation_id": operation_id,
                "snapshot_id": self.scope.request.snapshot_id, "data": data,
            }),
            "usage": usage,
        }

    async def validate_plan(self, state: AgentState):
        action = ProposePlan.model_validate(state["action"])
        if self.deps.validator is None:
            return stop(state, "validation_unavailable")
        canonical = json.dumps({"request": state["request"], "plan": action.plan.model_dump()},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        request = PlanValidationRequest(
            request=AgentRequest.model_validate(state["request"]),
            # Frozen models still contain mutable collections. Give the adapter
            # its own copy so receipt checks compare against the original proposal.
            plan=action.plan.model_copy(deep=True),
            proposal_fingerprint=fingerprint,
        )
        operation_id = f"{self.scope.request.run_id}:validate:{state['rejected_plans'] + 1}"
        try:
            result = await self.scope.call(
                operation_id, lambda context: self.deps.validator.validate(request, context),
            )
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        except CapabilityError as exc:
            return stop(state, "validation_unavailable", issue=exc.issue)
        if not isinstance(result, ValidationResult):
            raise TypeError("PlanValidator must return ValidationResult")
        self.scope.emit("validate_plan", "validation_finished", status=result.status)
        if result.proposal_fingerprint != fingerprint:
            return stop(state, "invalid_validation_receipt", status="failed")
        if result.status == "passed":
            verified = result.verified_plan
            if verified.snapshot_id != self.scope.request.snapshot_id or verified.items != action.plan.items:
                return stop(state, "invalid_validation_receipt", status="failed")
            return {"verified_plan": verified.model_dump(mode="json"),
                    "status": "completed", "next_node": "finalize", "stop_reason": None}
        issues = [item.model_dump() for item in result.issues]
        if result.status == "unavailable":
            return {**stop(state, "validation_unavailable"), "issues": (state["issues"] + issues)[-20:]}
        rejected = state["rejected_plans"] + 1
        usage = {**state["usage"], "plan_repairs": rejected}
        if rejected > self.policy.max_plan_repairs:
            update = stop(state, "plan_repair_limit")
        else:
            update = observation(state, {"status": "validation_failed", "issues": issues})
        return {**update, "usage": usage, "rejected_plans": rejected,
                "issues": (state["issues"] + issues)[-20:]}

    async def suspend(self, state: AgentState):
        return {"status": "waiting_user", "clarification": state["action"],
                "verified_plan": None, "next_node": "finalize", "stop_reason": None}

    async def finalize(self, state: AgentState):
        try:
            self.scope.check()
        except RunStopped as exc:
            state = {**state, **stop(state, exc.reason, status=exc.status)}
        usage = Usage.model_validate({**state["usage"], "elapsed_s": round(time.monotonic() - self.scope.started, 6)})
        result = AgentResult(
            run_id=self.scope.request.run_id, conversation_id=self.scope.request.conversation_id,
            snapshot_id=self.scope.request.snapshot_id, status=state["status"],
            stop_reason=state["stop_reason"], verified_plan=state["verified_plan"],
            clarification=state["clarification"], issues=state["issues"], usage=usage,
        )
        event_type = {"completed": "run_completed", "waiting_user": "run_waiting"}.get(result.status, "run_stopped")
        self.scope.emit("finalize", event_type, result=result, status=result.status, stop_reason=result.stop_reason)
        return {"result": result.model_dump(mode="json")}

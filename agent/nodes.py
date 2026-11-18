"""实现异步图节点；外部能力通过依赖注入接入，节点本身不创建网络客户端。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar
from uuid import uuid4

from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from .answers import validate_answer
from .budget_fields import USD_ALIASES, normalize_legacy_answer
from .contracts import (
    ACTION_ADAPTER, AgentDependencies, AgentEvent, AgentRequest, AgentResult,
    AskUser, CallContext, CallTool, CapabilityError, Issue, ModelReply, PendingOperation,
    PendingQuestion, PlanValidationRequest, ProposePlan, ResumeRequest, Usage, ValidationResult,
    EvidenceFact, VerifiedPlan,
)
from .explanation import (
    ExplanationRejected, ExplanationStream, build_explanation_request, check_explanation,
    render_fallback, validate_facts,
)
from .policies import RunPolicy
from .prompts import ContextLimitError, build_model_request
from .state import AgentState
from .progress import decision_control, fingerprint, record_result, bounded
from .shopping_context import remember, reusable

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RunStopped(Exception):
    """携带受控停止原因和状态，由节点转换为统一收尾路径。"""

    def __init__(self, reason: str, status: str = "incomplete"):
        self.reason, self.status = reason, status


@dataclass
class RunScope:
    """持有一次调用的依赖、事件队列和计时环境，恢复时接上原任务的累计信息。"""

    request: AgentRequest
    dependencies: AgentDependencies
    policy: RunPolicy
    cancellation: asyncio.Event
    queue: asyncio.Queue
    started: float = field(default_factory=time.monotonic)
    sequence: int = 0
    elapsed_before: float = 0.0
    # 查询句柄可能属于当前服务进程，因此缓存只在本执行段复用，不跨 resume/restart。
    tool_cache: dict = field(default_factory=dict)

    def check(self):
        """在执行边界检查显式取消信号与剩余时间。"""
        if self.cancellation.is_set():
            raise RunStopped("cancelled", "cancelled")
        if self.remaining <= 0:
            raise RunStopped("deadline_exceeded")

    @property
    def elapsed(self) -> float:
        """历史已执行时间加本段执行时间，不包含两次调用之间的用户等待时间。"""
        return self.elapsed_before + time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        """返回整个任务还可使用的实际执行时间。"""
        return self.policy.deadline_seconds - self.elapsed

    def emit(self, node: str, event_type: str, *, result=None, **data):
        """按运行内顺序发布事件，日志接收器失败不应触发业务操作重试。"""
        self.sequence += 1
        event = AgentEvent(
            run_id=self.request.run_id, sequence=self.sequence, node=node,
            type=event_type, data=data, result=result,
        )
        # 决策和调用次数有上限，过程事件体积也较小；使用无界队列，
        # 避免消费者提前关闭时，生产任务因等待队列空间而无法结束。
        self.queue.put_nowait(event)
        if self.dependencies.event_sink is not None:
            try:
                self.dependencies.event_sink(event.model_copy(deep=True))
            except Exception:
                logger.warning("Agent event sink failed (event_type=%s)", event_type)

    async def call(
        self, operation_id: str, invoke: Callable[[CallContext], Awaitable[T]],
        timeout_s: float | None = None,
        *, on_delta: Callable[[str, str], None] | None = None,
    ) -> T:
        """同时等待外部调用、取消或超时，并在退出时回收协作式异步任务。"""
        self.check()
        # 单次调用不能超过任务剩余时间、全局调用上限或工具自身的超时限制。
        timeout = min(self.remaining, self.policy.call_timeout_s,
                      timeout_s if timeout_s is not None else self.policy.call_timeout_s)
        context = CallContext(
            run_id=self.request.run_id, snapshot_id=self.request.snapshot_id,
            operation_id=operation_id, timeout_s=timeout, cancellation=self.cancellation,
            on_delta=on_delta,
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
            # 正常返回、受控错误和调用方取消都需要回收两个子任务。
            for task in (work, cancel):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, cancel, return_exceptions=True)


STOP_MESSAGES = {
    "no_feasible_plan": "No supported plan in the selected catalog meets these requirements.",
    "evidence_insufficient": "There is not enough evidence to support a plan.",
    "requirements_unverified": "The requirements could not be verified from the available evidence.",
}
# 模型的无解/证据不足说明面向用户；其余原因的 issue 文本只面向模型。
USER_FACING_ISSUES = set(STOP_MESSAGES)


def result_message(state):
    """仅为模型弃答提供说明；其他停止原因由调用页面根据 stop_reason 展示文案。"""
    reason = state["stop_reason"]
    if not reason:
        return None
    if reason in USER_FACING_ISSUES:
        for item in reversed(state["issues"]):
            if item.get("code") == reason and item.get("message"):
                return item["message"][:500]
    return STOP_MESSAGES.get(reason)


def stop(state, reason, *, status="incomplete", issue: Issue | None = None):
    """生成停止状态更新，并清除不能继续对外使用的方案和问题信息。"""
    return {
        "status": status, "stop_reason": reason, "next_node": "finalize",
        "verified_plan": None, "explanation": None, "clarification": None,
        "pending_question": None, "resume_answer": None,
        "issues": (state["issues"] + ([issue.model_dump()] if issue else []))[-20:],
    }


def consume_resume_budget(state):
    """回答后的证据额度只供紧接着的一次模型决策使用，之后显式归零。"""
    progress = dict(state.get("progress", {}))
    if progress.get("resume_evidence_budget"):
        progress["resume_evidence_budget"] = max(0, progress["resume_evidence_budget"] - 1)
    return progress


def observation(state, data, *, issue: Issue | None = None):
    """以统一消息记录工具数据或纠错反馈，再交由 guard 决定是否继续。"""
    return {
        "next_node": "guard", "action": None,
        "history": state["history"] + [{
            "role": "user", "content": json.dumps({"observation": data}, ensure_ascii=False),
        }],
        "issues": (state["issues"] + ([issue.model_dump()] if issue else []))[-20:],
    }


class Nodes:
    """组织节点的业务逻辑；具体执行顺序和分支由 graph.py 定义。"""

    def __init__(self, scope: RunScope):
        self.scope = scope
        self.deps, self.policy = scope.dependencies, scope.policy

    async def prepare(self, state: AgentState):
        """记录任务启动；输入校验和初始状态构造已由 Runtime 完成。"""
        self.scope.emit("prepare", "run_started", snapshot_id=self.scope.request.snapshot_id)
        return {"next_node": "guard"}

    async def guard(self, state: AgentState):
        """决策前检查取消、执行时间和模型次数；工具次数由执行节点另行检查。"""
        try:
            self.scope.check()
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        if state["usage"]["model_calls"] >= self.policy.max_model_calls:
            return stop(state, "model_call_limit")
        # 预检是可选的协议能力；包装器（如评测的记录器）只需转发同名方法即可生效。
        preflight = getattr(self.deps.validator, "preflight", None)
        if preflight is not None:
            question = preflight(state["request"]["constraints"])
            if question is not None:
                if self.policy.max_model_calls - state["usage"]["model_calls"] < 2:
                    return stop(state, "clarification_budget_exhausted", issue=Issue(
                        code="clarification_budget_exhausted", message="Not enough decisions remain to use another answer."))
                return {"action": question.model_dump(mode="json"), "next_node": "prepare_question"}
        return {"next_node": "decide"}

    async def decide(self, state: AgentState):
        """执行一次模型决策，校验动作并设置后续路由或有界修复反馈。"""
        try:
            self.scope.check()
            policy_for = getattr(self.deps.validator, "decision_policy", None)
            validation_policy = policy_for(state["request"]["constraints"]) if policy_for is not None else {}
            request = build_model_request(state, self.deps.tools, self.policy, validation_policy)
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        except ContextLimitError:
            return stop(state, "context_limit")
        usage = dict(state["usage"])
        # 外部调用开始前计数，失败尝试也消耗预算，不能通过报错绕开上限。
        usage["model_calls"] += 1
        operation_id = f"{self.scope.request.run_id}:model:{usage['model_calls']}"
        control = decision_control(state, self.policy)
        self.scope.emit("decide", "decision_started", operation_id=operation_id, **control)
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
        # 无论动作是否有效，这次决策已经用掉了回答后的补证据额度。
        progress = consume_resume_budget(state)
        try:
            # 截断或错误结束的回复不能仅因 JSON 可解析就被当成完整动作。
            if reply.finish_reason != "stop":
                raise ValueError("incomplete generation")
            action = ACTION_ADAPTER.validate_json(reply.content)
        except (ValidationError, ValueError) as exc:
            errors = state["consecutive_action_errors"] + 1
            issue = Issue(code="invalid_action", message="Return one complete action matching the schema.")
            usage["action_repairs"] += 1
            self.scope.emit("decide", "action_rejected", code=issue.code)
            if errors > self.policy.max_action_repairs:
                update = stop(state, "invalid_action", issue=issue)
            else:
                # 通用提示不足以定位缺括号或字段错误；把有界的原回复和具体原因交回模型。
                # 不自动补括号、剥离代码围栏，也不把校验库携带的完整输入写入反馈。
                details = (
                    exc.errors(include_input=False, include_url=False, include_context=False)
                    if isinstance(exc, ValidationError)
                    else [{"type": "incomplete_generation", "loc": [], "msg": str(exc)}]
                )
                feedback = {
                    "status": "error", **issue.model_dump(),
                    "instruction": "Return a corrected, complete JSON action without Markdown fences.",
                    "rejected_output": reply.content[:2048],
                    "rejected_output_truncated": len(reply.content) > 2048,
                    "validation_errors": [{
                        "type": detail["type"],
                        "loc": [str(part)[:128] for part in detail["loc"][:8]],
                        "msg": detail["msg"][:256],
                    } for detail in details[:5]],
                }
                update = observation(state, feedback, issue=issue)
            return {**update, "usage": usage, "consecutive_action_errors": errors, "progress": progress}
        self.scope.emit("decide", "action_selected", action_type=action.type,
                        action=bounded(action.model_dump(mode="json")))
        if action.type not in control["allowed_actions"]:
            if action.type == "call_tool" and usage["tool_calls"] >= self.policy.max_tool_calls:
                return {**stop(state, "tool_call_limit"), "usage": usage,
                        "_trace": {"attempted_action": bounded(action.model_dump(mode="json"))}}
            issue = Issue(code="convergence_required", message=(
                "Exploration is closed. Submit a supported plan using existing evidence. "
                "Questions require two remaining decisions after the question."))
            if usage["model_calls"] >= self.policy.max_model_calls:
                return {**stop(state, "clarification_budget_exhausted" if action.type == "ask_user"
                               else "convergence_failed", issue=issue), "usage": usage,
                        "_trace": {"attempted_action": bounded(action.model_dump(mode="json"))}}
            return {**observation(state, {"status": "error", **issue.model_dump(),
                                          "allowed_actions": control["allowed_actions"]}, issue=issue),
                    "usage": usage, "progress": {**progress, "force_finalize": True},
                    "_trace": {"attempted_action": bounded(action.model_dump(mode="json"))}}
        if action.type == "report_incomplete":
            return {**stop(state, action.reason, issue=Issue(code=action.reason, message=action.message)),
                    "usage": usage, "action": action.model_dump(mode="json")}
        route = {"call_tool": "execute_tool", "propose_plan": "validate_plan", "ask_user": "prepare_question"}
        return {
            "action": action.model_dump(mode="json"), "next_node": route[action.type],
            "usage": usage, "consecutive_action_errors": 0, "verified_plan": None, "explanation": None,
            "progress": progress,
            "history": state["history"] + [{"role": "assistant", "content": action.model_dump_json()}],
        }

    async def execute_tool(self, state: AgentState):
        """校验注册工具及参数，执行一次调用，再验证和限制结果大小。"""
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
            # 参数或名称被拒绝时没有调用执行器，因此不增加工具调用次数。
            self.scope.emit("execute_tool", "tool_rejected", tool=action.tool_name, code=issue.code)
            return {**observation(state, {"status": "error", **issue.model_dump()}, issue=issue),
                    "progress": record_result(state, action.tool_name, issue.model_dump(), error=True),
                    "_trace": {"tool": action.tool_name, "arguments": bounded(action.arguments), "error": issue.model_dump()}}
        try:
            self.scope.check()
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        cache_key = fingerprint({"snapshot_id": state["request"]["snapshot_id"],
                                 "constraints": state["request"]["constraints"],
                                 "tool": tool.name, "arguments": arguments.model_dump(mode="json")})
        if reusable(tool) and cache_key in self.scope.tool_cache:
            cached = self.scope.tool_cache[cache_key]
            data = cached["data"]
            return {**observation(state, {"status": "ok", "tool": tool.name, "cached": True,
                                          "no_new_information": True,
                                          "previous_operation_id": cached["operation_id"],
                                          "guidance": "Identical successful query; use the earlier observation or shopping_context. Choose a different evidence read only if essential, otherwise finish.",
                                          "snapshot_id": self.scope.request.snapshot_id, "data": data}),
                    "progress": record_result(state, tool.name, data, arguments=arguments.model_dump(mode="json")),
                    "_trace": {"cached": True, "tool": tool.name, "arguments": bounded(action.arguments),
                               "previous_operation_id": cached["operation_id"]}}
        if state["usage"]["tool_calls"] >= self.policy.max_tool_calls:
            return stop(state, "tool_call_limit")
        usage = dict(state["usage"])
        usage["tool_calls"] += 1
        operation_id = f"{self.scope.request.run_id}:tool:{usage['tool_calls']}"
        self.scope.emit("execute_tool", "tool_started", tool=tool.name, operation_id=operation_id,
                        arguments=bounded(action.arguments))
        try:
            raw = await self.scope.call(
                operation_id, lambda context: tool.executor.execute(arguments, context), tool.timeout_s,
            )
        except RunStopped as exc:
            return {**stop(state, exc.reason, status=exc.status), "usage": usage}
        except CapabilityError as exc:
            self.scope.emit("execute_tool", "tool_finished", tool=tool.name, status="error", code=exc.issue.code)
            # 可恢复错误反馈给下一轮模型决策，不在此处自动重发外部请求。
            if exc.recoverable:
                update = observation(state, {"status": "error", **exc.issue.model_dump()}, issue=exc.issue)
            else:
                update = stop(state, exc.issue.code, status="failed", issue=exc.issue)
            return {**update, "usage": usage,
                    "progress": record_result(state, tool.name, exc.issue.model_dump(), error=True),
                    "_trace": {"tool": tool.name, "arguments": bounded(action.arguments),
                               "operation_id": operation_id, "error": exc.issue.model_dump()}}
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
        if reusable(tool):
            self.scope.tool_cache[cache_key] = {"data": data, "operation_id": operation_id}
        self.scope.emit("execute_tool", "tool_finished", tool=tool.name, status="ok", operation_id=operation_id,
                        result_summary=bounded(data))
        return {
            **observation(state, {
                "status": "ok", "tool": tool.name, "operation_id": operation_id,
                "snapshot_id": self.scope.request.snapshot_id, "data": data,
            }),
            "usage": usage,
            "progress": record_result(state, tool.name, data, arguments=arguments.model_dump(mode="json")),
            "shopping_context": remember(state.get("shopping_context"), tool.name, action.arguments, data),
            "_trace": {"tool": tool.name, "arguments": bounded(action.arguments),
                       "operation_id": operation_id, "result_summary": bounded(data)},
        }

    async def validate_plan(self, state: AgentState):
        """调用业务校验器，确认回执对应当前请求、候选方案和快照。"""
        action = ProposePlan.model_validate(state["action"])
        if self.deps.validator is None:
            return stop(state, "validation_unavailable")
        canonical = json.dumps({"request": state["request"], "plan": action.plan.model_dump()},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        # 对稳定序列化内容计算指纹，用来拒绝其他方案或旧请求的校验回执。
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        request = PlanValidationRequest(
            request=AgentRequest.model_validate(state["request"]),
            # 冻结模型仍可能包含可变容器；给适配器独立副本，
            # 保证回执比较始终以原候选方案为准。
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
        self.scope.emit("validate_plan", "validation_finished", status=result.status,
                        issues=[item.model_dump() for item in result.issues])
        if result.proposal_fingerprint != fingerprint:
            return stop(state, "invalid_validation_receipt", status="failed")
        if result.status == "passed":
            verified = result.verified_plan
            if verified.snapshot_id != self.scope.request.snapshot_id or verified.items != action.plan.items:
                return stop(state, "invalid_validation_receipt", status="failed")
            return {"verified_plan": verified.model_dump(mode="json"),
                    "status": "completed", "next_node": "explain" if self.policy.explain else "finalize",
                    "explanation": None, "stop_reason": None}
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

    async def explain(self, state: AgentState):
        """生成有界解释；可预期解释失败使用模板，全局取消/期限仍走原停止契约。"""
        try:
            self.scope.check()
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        plan = VerifiedPlan.model_validate(state["verified_plan"])
        usage = Usage.model_validate(state["usage"]).model_dump()
        operation_id = f"{self.scope.request.run_id}:explain:1"
        self.scope.emit("explain", "explain_started", operation_id=operation_id)
        deadline = time.monotonic() + self.policy.explain_timeout_s
        stream = ExplanationStream(
            lambda text: self.scope.emit("explain", "explain_delta", kind="content", text=text),
            self.policy.max_explain_chars,
        )

        def remaining():
            duration = deadline - time.monotonic()
            if duration <= 0:
                raise ExplanationRejected("explanation_timeout")
            return duration

        def on_delta(kind, text):
            if not self.scope.cancellation.is_set() and self.scope.remaining > 0:
                stream.feed(kind, text)

        reason = None
        try:
            facts = []
            if self.deps.evidence_resolver is not None:
                raw = await self.scope.call(
                    f"{self.scope.request.run_id}:explain_evidence:1",
                    lambda context: self.deps.evidence_resolver.resolve(plan.model_copy(deep=True), context),
                    remaining(),
                )
                if not isinstance(raw, list) or len(raw) > 400:
                    raise ExplanationRejected("invalid_evidence")
                try:
                    facts = [EvidenceFact.model_validate(fact).model_copy(deep=True) for fact in raw]
                except ValidationError as exc:
                    raise ExplanationRejected("invalid_evidence") from exc
                validate_facts(plan, facts)
            request = build_explanation_request(AgentRequest.model_validate(state["request"]), plan, facts, self.policy)
            timeout = remaining()
            usage["explain_model_calls"] += 1
            usage["unreported_token_calls"] += 1
            reply = await self.scope.call(operation_id, lambda context: self.deps.model.generate(request, context),
                                          timeout, on_delta=on_delta)
            if isinstance(reply, PendingOperation):
                raise ExplanationRejected("unsupported_capability")
            if not isinstance(reply, ModelReply):
                raise TypeError("ModelPort must return ModelReply or PendingOperation")
            usage["prompt_tokens"] += reply.prompt_tokens or 0
            usage["completion_tokens"] += reply.completion_tokens or 0
            usage["unreported_token_calls"] -= int(reply.prompt_tokens is not None and reply.completion_tokens is not None)
            if reply.finish_reason != "stop":
                raise ExplanationRejected("incomplete_explanation")
            if stream.overflow:
                raise ExplanationRejected("explanation_too_long")
            if not stream.parts:
                stream.feed("content", reply.content)
            if stream.overflow:
                raise ExplanationRejected("explanation_too_long")
            if "".join(stream.parts) != reply.content:
                raise ExplanationRejected("explanation_stream_mismatch")
            explanation = check_explanation(reply.content, plan, facts, self.policy.max_explain_chars)
            stream.close(flush=True)
        except RunStopped as exc:
            return {**stop(state, exc.reason, status=exc.status), "usage": usage}
        except (CapabilityError, ExplanationRejected) as exc:
            # CapabilityError 是适配器声明的失败；程序错误和 asyncio 取消继续传播。
            reason = exc.issue.code if isinstance(exc, CapabilityError) else exc.reason
            explanation = render_fallback(plan)
        finally:
            stream.close()
        try:
            self.scope.check()
        except RunStopped as exc:
            return {**stop(state, exc.reason, status=exc.status), "usage": usage}
        self.scope.emit("explain", "explain_done", verified=explanation.verified,
                        fallback=explanation.fallback, reason=reason, chars=len(explanation.text))
        return {"explanation": explanation.model_dump(mode="json"), "usage": usage, "next_node": "finalize"}

    async def prepare_question(self, state: AgentState):
        """在中断前固定问题身份、字段和过期时间，保存可供恢复关联的状态。"""
        try:
            self.scope.check()
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        if state["question_count"] >= self.policy.max_questions:
            return stop(state, "question_limit")
        if self.policy.max_model_calls - state["usage"]["model_calls"] < 2:
            return stop(state, "clarification_budget_exhausted", issue=Issue(
                code="clarification_budget_exhausted", message="A question must leave two decisions for evidence and a proposal."))
        action = AskUser.model_validate(state["action"])
        question = PendingQuestion(
            question_id=uuid4().hex, question=action.question, required_fields=action.required_fields,
            expires_at=time.time() + self.policy.question_timeout_s,
        )
        self.scope.emit("prepare_question", "question_prepared", question_id=question.question_id)
        return {
            "pending_question": question.model_dump(mode="json"),
            "clarification": action.model_dump(mode="json"), "status": "waiting_user",
            "verified_plan": None, "explanation": None, "result": None, "resume_answer": None,
            "stop_reason": None, "question_count": state["question_count"] + 1,
            "next_node": "suspend",
            # 先在检查点中预留 run_waiting 的序号；运行时确认中断已保存后，
            # 再发出该事件，使恢复后的事件序号可以继续衔接。
            "event_sequence": self.scope.sequence + 1,
        }

    async def suspend(self, state: AgentState):
        """挂起当前问题；恢复后把用户回答交给 apply_answer 处理。"""
        # LangGraph 恢复时会重跑本节点，因此 interrupt 前不能执行外部调用、
        # 生成随机 ID 或发布事件；稳定问题已由前一节点保存。
        answer = interrupt(state["pending_question"])
        return {"resume_answer": answer, "next_node": "apply_answer"}

    async def apply_answer(self, state: AgentState):
        """校验问题和回答，只合并有效约束；无效回答受限重问，过期则终止。"""
        try:
            self.scope.check()
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        question = PendingQuestion.model_validate(state["pending_question"])
        if time.time() >= question.expires_at:
            return stop(state, "question_expired")
        reply = ResumeRequest.model_validate(state["resume_answer"])
        if reply.run_id != self.scope.request.run_id or reply.question_id != question.question_id:
            return stop(state, "invalid_resume", status="failed")
        issue = validate_answer(question, reply, state["request"]["constraints"], self.policy)
        if issue is not None:
            errors = state["answer_errors"] + 1
            self.scope.emit("apply_answer", "answer_rejected", question_id=question.question_id, code=issue.code)
            if errors > self.policy.max_answer_repairs:
                update = stop(state, "answer_repair_limit", issue=issue)
            else:
                reask = {}
                if USD_ALIASES.intersection(question.required_fields):
                    fields = [name for name in question.required_fields if name not in USD_ALIASES]
                    fields = list(dict.fromkeys(fields + ["purchase_cap", "currency"]))
                    reask["action"] = AskUser(type="ask_user", required_fields=fields,
                        question="The legacy budget fields conflict; please confirm purchase_cap and currency. "
                                 "Amounts must be non-negative decimal strings.").model_dump(mode="json")
                update = {
                    "next_node": "prepare_question", "resume_answer": None,
                    "issues": (state["issues"] + [issue.model_dump()])[-20:],
                    **reask,
                }
            return {**update, "answer_errors": errors}
        # 仅在全部校验通过后合并约束；运行身份、快照和执行策略不由回答覆盖。
        answer = normalize_legacy_answer(reply.answer, state["request"]["constraints"])
        request = {**state["request"], "constraints": {**state["request"]["constraints"], **answer}}
        request = AgentRequest.model_validate(request)
        self.scope.emit("apply_answer", "answer_accepted", question_id=question.question_id)
        memory = dict(state.get("shopping_context", {}))
        memory["candidates"] = [{k: v for k, v in candidate.items() if k not in ("hard_constraint_status", "candidate_set_id")}
                                for candidate in memory.get("candidates", [])]
        memory["notes"] = (memory.get("notes", []) + [
            "User constraints changed. Re-evaluate selection; old candidate-set handles may have expired. "
            "Use stable evidence IDs or refresh the relevant search if needed."])[-4:]
        return {
            "request": request.model_dump(mode="json"),
            "history": state["history"] + [{"role": "user", "content": json.dumps({
                "clarification_answer": {"question": question.question, "answer": answer},
            }, ensure_ascii=False)}],
            # 约束已改变，旧方案的校验结论不能继续作为有效结果。
            "verified_plan": None, "explanation": None, "pending_question": None, "resume_answer": None,
            "clarification": None, "action": None, "answer_errors": 0,
            "status": None, "stop_reason": None, "result": None, "next_node": "guard",
            # 保留已见结果，避免回答前查过的查询重复时又被算作进展；约束变化后重新允许探索。
            "progress": {**state.get("progress", {}), "no_progress_calls": 0,
                         "force_finalize": False, "resume_evidence_budget": 1},
            "shopping_context": memory,
            "_trace": {"answer": answer, "legacy_budget_normalized": answer != reply.answer},
        }

    async def finalize(self, state: AgentState):
        """统一整理受控结束结果；用户等待由 Runtime 处理中断，不走本节点。"""
        try:
            self.scope.check()
        except RunStopped as exc:
            state = {**state, **stop(state, exc.reason, status=exc.status)}
        usage = Usage.model_validate({**state["usage"], "elapsed_s": round(self.scope.elapsed, 6)})
        result = AgentResult(
            run_id=self.scope.request.run_id, conversation_id=self.scope.request.conversation_id,
            snapshot_id=self.scope.request.snapshot_id, status=state["status"],
            stop_reason=state["stop_reason"], message=result_message(state), verified_plan=state["verified_plan"],
            clarification=state["clarification"], issues=state["issues"], usage=usage,
            explanation=state.get("explanation"),
        )
        event_type = {"completed": "run_completed", "waiting_user": "run_waiting"}.get(result.status, "run_stopped")
        self.scope.emit("finalize", event_type, result=result, status=result.status, stop_reason=result.stop_reason)
        return {"result": result.model_dump(mode="json")}

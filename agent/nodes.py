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
from .contracts import (
    ACTION_ADAPTER, AgentDependencies, AgentEvent, AgentRequest, AgentResult,
    AskUser, CallContext, CallTool, CapabilityError, Issue, ModelReply, PendingOperation,
    PendingQuestion, PlanValidationRequest, ProposePlan, ResumeRequest, Usage, ValidationResult,
)
from .policies import RunPolicy
from .prompts import ContextLimitError, build_model_request
from .state import AgentState

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
    ) -> T:
        """同时等待外部调用、取消或超时，并在退出时回收协作式异步任务。"""
        self.check()
        # 单次调用不能超过任务剩余时间、全局调用上限或工具自身的超时限制。
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
            # 正常返回、受控错误和调用方取消都需要回收两个子任务。
            for task in (work, cancel):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, cancel, return_exceptions=True)


def stop(state, reason, *, status="incomplete", issue: Issue | None = None):
    """生成停止状态更新，并清除不能继续对外使用的方案和问题信息。"""
    return {
        "status": status, "stop_reason": reason, "next_node": "finalize",
        "verified_plan": None, "clarification": None,
        "pending_question": None, "resume_answer": None,
        "issues": (state["issues"] + ([issue.model_dump()] if issue else []))[-20:],
    }


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
    """组织九个节点的业务逻辑；具体执行顺序和分支由 graph.py 定义。"""

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
        return {"next_node": "decide"}

    async def decide(self, state: AgentState):
        """执行一次模型决策，校验动作并设置后续路由或有界修复反馈。"""
        try:
            self.scope.check()
            request = build_model_request(state, self.deps.tools, self.policy)
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        except ContextLimitError:
            return stop(state, "context_limit")
        usage = dict(state["usage"])
        # 外部调用开始前计数，失败尝试也消耗预算，不能通过报错绕开上限。
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
            # 截断或错误结束的回复不能仅因 JSON 可解析就被当成完整动作。
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
        route = {"call_tool": "execute_tool", "propose_plan": "validate_plan", "ask_user": "prepare_question"}
        return {
            "action": action.model_dump(mode="json"), "next_node": route[action.type],
            "usage": usage, "consecutive_action_errors": 0, "verified_plan": None,
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
            # 可恢复错误反馈给下一轮模型决策，不在此处自动重发外部请求。
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

    async def prepare_question(self, state: AgentState):
        """在中断前固定问题身份、字段和过期时间，保存可供恢复关联的状态。"""
        try:
            self.scope.check()
        except RunStopped as exc:
            return stop(state, exc.reason, status=exc.status)
        if state["question_count"] >= self.policy.max_questions:
            return stop(state, "question_limit")
        action = AskUser.model_validate(state["action"])
        question = PendingQuestion(
            question_id=uuid4().hex, question=action.question, required_fields=action.required_fields,
            expires_at=time.time() + self.policy.question_timeout_s,
        )
        self.scope.emit("prepare_question", "question_prepared", question_id=question.question_id)
        return {
            "pending_question": question.model_dump(mode="json"),
            "clarification": action.model_dump(mode="json"), "status": "waiting_user",
            "verified_plan": None, "result": None, "resume_answer": None,
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
                update = {
                    "next_node": "prepare_question", "resume_answer": None,
                    "issues": (state["issues"] + [issue.model_dump()])[-20:],
                }
            return {**update, "answer_errors": errors}
        # 仅在全部校验通过后合并约束；运行身份、快照和执行策略不由回答覆盖。
        request = {**state["request"], "constraints": {**state["request"]["constraints"], **reply.answer}}
        request = AgentRequest.model_validate(request)
        self.scope.emit("apply_answer", "answer_accepted", question_id=question.question_id)
        return {
            "request": request.model_dump(mode="json"),
            "history": state["history"] + [{"role": "user", "content": json.dumps({
                "clarification_answer": {"question": question.question, "answer": reply.answer},
            }, ensure_ascii=False)}],
            # 约束已改变，旧方案的校验结论不能继续作为有效结果。
            "verified_plan": None, "pending_question": None, "resume_answer": None,
            "clarification": None, "action": None, "answer_errors": 0,
            "status": None, "stop_reason": None, "result": None, "next_node": "guard",
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
            stop_reason=state["stop_reason"], verified_plan=state["verified_plan"],
            clarification=state["clarification"], issues=state["issues"], usage=usage,
        )
        event_type = {"completed": "run_completed", "waiting_user": "run_waiting"}.get(result.status, "run_stopped")
        self.scope.emit("finalize", event_type, result=result, status=result.status, stop_reason=result.stop_reason)
        return {"result": result.model_dump(mode="json")}

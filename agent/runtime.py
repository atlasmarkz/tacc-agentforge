"""提供异步运行及问答恢复入口；状态由检查点保存，事件队列按调用隔离。"""

import asyncio
from contextlib import aclosing
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .contracts import (
    AgentDependencies, AgentEvent, AgentRequest, AgentResult, PendingQuestion,
    ResumeRequest, RunStateError, UnsupportedCapabilityError, Usage,
)
from .graph import build_graph
from .nodes import RunScope
from .policies import RunPolicy
from .state import initial_state

_END = object()


class AgentRuntime:
    """负责创建并运行智能体流程，同时提供结果和事件流接口。"""

    def __init__(self, dependencies: AgentDependencies, policy: RunPolicy | None = None,
                 *, checkpointer=None):
        """初始化运行时及其依赖和执行策略。"""
        if checkpointer is not None and not isinstance(checkpointer, BaseCheckpointSaver):
            raise UnsupportedCapabilityError("checkpointer must implement BaseCheckpointSaver's async API.")
        self.dependencies, self.policy = dependencies, policy or RunPolicy()
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        # 只保护同一 Runtime 内的并发。共享持久化存储的多个进程需要宿主串行调度。
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    async def _result(events) -> AgentResult:
        """消费本次调用的事件并提取结果，退出时同步关闭底层异步生成器。"""
        result = None
        async with aclosing(events):
            async for event in events:
                if event.result is not None:
                    result = event.result
        if result is None:
            raise RuntimeError("Agent graph finished without a result or a question")
        return result

    async def run(self, request: AgentRequest | dict, *, cancellation: asyncio.Event | None = None) -> AgentResult:
        """启动新任务，返回本次调用的完成结果或可恢复等待结果。"""
        return await self._result(self.astream(request, cancellation=cancellation))

    async def astream(
        self, request: AgentRequest | dict, *, cancellation: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """启动新任务并产出事件；遇到问题时返回可恢复的等待结果。"""
        if isinstance(request, AgentRequest):
            # 统一转换为 JSON 兼容字典，随后通过模型校验输入。
            request = request.model_dump(mode="json")
        request = AgentRequest.model_validate(request)
        async with aclosing(self._stream(request=request, cancellation=cancellation)) as events:
            async for event in events:
                yield event

    async def resume(self, run_id: str, question_id: str, answer: dict,
                     *, cancellation: asyncio.Event | None = None) -> AgentResult:
        """匹配等待中的问题并恢复原任务；调用次数和执行时间预算不会重置。"""
        return await self._result(self.astream_resume(run_id, question_id, answer, cancellation=cancellation))

    async def astream_resume(self, run_id: str, question_id: str, answer: dict,
                             *, cancellation: asyncio.Event | None = None) -> AsyncIterator[AgentEvent]:
        """校验恢复请求并输出后续事件，保留原任务的事件顺序。"""
        reply = ResumeRequest(run_id=run_id, question_id=question_id, answer=answer)
        async with aclosing(self._stream(reply=reply, cancellation=cancellation)) as events:
            async for event in events:
                yield event

    async def _stream(self, *, request=None, reply=None, cancellation=None):
        """对同一任务的调用串行保护；新建和恢复共用执行及清理路径。"""
        run_id = request.run_id if request is not None else reply.run_id
        lock = self._locks.setdefault(run_id, asyncio.Lock())
        if lock.locked():
            raise RunStateError("run_busy", "Another invocation is already processing this run.")
        async with lock:
            async with aclosing(self._execute_stream(request, reply, cancellation)) as events:
                async for event in events:
                    yield event

    async def _execute_stream(self, request, reply, cancellation):
        """读取或创建任务状态，驱动状态图，并将真实中断转换为等待结果。"""
        run_id = request.run_id if request is not None else reply.run_id
        # 检查点按任务定位；conversation_id 仅用于会话归组，不用于恢复任务。
        config = {"configurable": {"thread_id": f"agent:{run_id}"}}
        saved = await self.checkpointer.aget_tuple(config)
        if request is not None:
            if saved is not None:
                raise RunStateError("run_exists", "Use resume for a waiting run, or choose a new run_id.")
            policy, state = self.policy, initial_state(request, self.policy)
            graph_input = state
        else:
            if saved is None:
                raise RunStateError("run_not_found", "No checkpoint exists for this run.")
            # 先读取已保存字段以重建运行作用域，随后再确认图确实停在可恢复中断。
            state = saved.checkpoint["channel_values"]
            if state.get("state_version") != "2":
                raise RunStateError("incompatible_checkpoint", "Checkpoint schema version is not supported.")
            request = AgentRequest.model_validate(state["request"])
            policy = RunPolicy.model_validate(state["policy"])
            if state.get("status") != "waiting_user" or state.get("pending_question") is None:
                raise RunStateError("not_waiting", "This run is not waiting for a user answer.")
            question = PendingQuestion.model_validate(state["pending_question"])
            if question.question_id != reply.question_id:
                raise RunStateError("question_mismatch", "The answer does not match the pending question.")
            graph_input = Command(resume=reply.model_dump(mode="json"))
        queue = asyncio.Queue()
        # 每次运行都创建独立作用域，避免状态、取消信号和事件相互污染。
        scope = RunScope(request, self.dependencies, self.policy,
                         cancellation if cancellation is not None else asyncio.Event(), queue)
        # 使用检查点中的原策略、序号和已消耗时间，不能因恢复而重新获得执行预算。
        scope.policy = policy
        scope.sequence = state["event_sequence"]
        scope.elapsed_before = state["active_elapsed_s"]
        graph = build_graph(scope, self.checkpointer)
        config["recursion_limit"] = policy.recursion_limit
        if reply is not None:
            snapshot = await graph.aget_state(config)
            if not any(task.interrupts for task in snapshot.tasks):
                raise RunStateError("not_interrupted", "No resumable user interrupt exists.")

        async def execute():
            """在后台执行智能体图，并确保向事件队列发送结束标记。"""
            try:
                if reply is not None:
                    scope.emit("runtime", "run_resumed", question_id=reply.question_id)
                output = await graph.ainvoke(graph_input, config=config)
                if output.get("__interrupt__"):
                    question = PendingQuestion.model_validate(output["pending_question"])
                    if scope.cancellation.is_set() or scope.remaining <= 0:
                        # 取消/执行预算耗尽不能被误报为等待用户。让中断继续到检查节点收尾。
                        await graph.ainvoke(Command(resume=ResumeRequest(
                            run_id=run_id, question_id=question.question_id, answer={},
                        ).model_dump(mode="json")), config=config)
                        return
                    # 此处只返回等待结果，不执行 finalize，也不把任务推进到 END。
                    result = AgentResult(
                        run_id=request.run_id, conversation_id=request.conversation_id,
                        snapshot_id=request.snapshot_id, status="waiting_user",
                        clarification=output["clarification"], pending_question=question,
                        issues=output["issues"], usage=Usage.model_validate({
                            **output["usage"], "elapsed_s": round(output["active_elapsed_s"], 6),
                        }),
                    )
                    # 使用 prepare_question 已预留的序号，避免恢复后重复或跳号。
                    scope.sequence = output["event_sequence"] - 1
                    scope.emit("suspend", "run_waiting", result=result, question_id=question.question_id)
            finally:
                queue.put_nowait(_END)

        task = asyncio.create_task(execute())
        try:
            while True:
                event = await queue.get()
                if event is _END:
                    # 等待任务以显式暴露图执行错误或调用方取消异常。
                    await task
                    break
                yield event
        finally:
            if not task.done():
                # 调用方提前停止消费时，及时取消后台图任务。
                task.cancel()
            # 始终回收任务异常，同时避免覆盖前面已经抛出的异常。
            await asyncio.gather(task, return_exceptions=True)

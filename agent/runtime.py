"""Public async facade. Each invocation owns its state, graph scope and event queue."""

import asyncio
from typing import AsyncIterator

from .contracts import (
    AgentDependencies, AgentEvent, AgentRequest, AgentResult, UnsupportedCapabilityError,
)
from .graph import build_graph
from .nodes import RunScope
from .policies import RunPolicy
from .state import initial_state

_END = object()


class AgentRuntime:
    def __init__(self, dependencies: AgentDependencies, policy: RunPolicy | None = None,
                 *, checkpointer=None):
        if checkpointer is not None:
            raise UnsupportedCapabilityError("Persistent checkpoints are not supported in v1.")
        self.dependencies, self.policy = dependencies, policy or RunPolicy()

    async def run(self, request: AgentRequest | dict, *, cancellation: asyncio.Event | None = None) -> AgentResult:
        result = None
        async for event in self.astream(request, cancellation=cancellation):
            if event.result is not None:
                result = event.result
        if result is None:
            raise RuntimeError("Agent graph finished without a result")
        return result

    async def astream(
        self, request: AgentRequest | dict, *, cancellation: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]:
        if isinstance(request, AgentRequest):
            request = request.model_dump(mode="json")
        request = AgentRequest.model_validate(request)
        queue = asyncio.Queue()
        scope = RunScope(request, self.dependencies, self.policy,
                         cancellation if cancellation is not None else asyncio.Event(), queue)
        graph = build_graph(scope)

        async def execute():
            try:
                await graph.ainvoke(initial_state(request), config={"recursion_limit": self.policy.recursion_limit})
            finally:
                queue.put_nowait(_END)

        task = asyncio.create_task(execute())
        try:
            while True:
                event = await queue.get()
                if event is _END:
                    await task  # Surface programming errors and caller cancellation.
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
            # Always retrieve the task exception; do not mask an exception already raised above.
            await asyncio.gather(task, return_exceptions=True)

    async def resume(self, *args, **kwargs) -> AgentResult:
        raise UnsupportedCapabilityError("Resume is not supported in v1; submit a distinct new run.")

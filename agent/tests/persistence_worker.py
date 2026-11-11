"""重启验收的独立进程：真实图和 SQLite，显式模拟模型与工具。"""

import argparse
import asyncio
import json
import os

from agent import AgentDependencies, AgentRuntime, ModelReply, RunPolicy
from agent.checkpoints import open_checkpointer
from agent.examples.mock_run import MockModel, MockPlanValidator, MockSearch, demo_request, demo_tools


class RecordedModel(MockModel):
    def __init__(self):
        self.calls = 0
        self.messages = []

    async def generate(self, request, context):
        self.calls += 1
        if request.action_schema is None:
            return await super().generate(request, context)
        self.messages = [message.model_dump() for message in request.messages]
        original = json.loads(request.messages[1].content)
        observed = any(m.content.startswith('{"observation":') for m in request.messages)
        if "purchase_cap" not in original["constraints"] and not observed:
            return ModelReply(content=json.dumps({"type": "call_tool", "tool_name": "search_products",
                                                  "arguments": {"query": "earbuds"}}))
        return await super().generate(request, context)


class RecordedSearch(MockSearch):
    def __init__(self):
        self.calls = 0

    async def execute(self, arguments, context):
        self.calls += 1
        return await super().execute(arguments, context)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "resume"))
    parser.add_argument("--db", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--question-id")
    parser.add_argument("--max-model-calls", type=int, default=8)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    model, search = RecordedModel(), RecordedSearch()
    async with open_checkpointer(args.db, create=args.phase == "start") as saver:
        runtime = AgentRuntime(AgentDependencies(model, demo_tools(search), MockPlanValidator()),
                               RunPolicy(max_model_calls=args.max_model_calls, explain=args.explain), checkpointer=saver)
        if args.phase == "start":
            request = demo_request(args.run_id).model_copy(update={"constraints": {"currency": "USD"}})
            stream = runtime.astream(request)
        else:
            stream = runtime.astream_resume(args.run_id, args.question_id, {"purchase_cap": "100.00"})
        events = [event async for event in stream]
        saved = await saver.aget_tuple({"configurable": {"thread_id": f"agent:{args.run_id}"}})
        print(json.dumps({"pid": os.getpid(), "events": [e.model_dump(mode="json") for e in events],
                          "local_model_calls": model.calls, "local_tool_calls": search.calls,
                          "messages": model.messages, "policy": saved.checkpoint["channel_values"]["policy"]}),
              flush=True)
        if args.phase == "start":
            # 持有连接且不退出上下文，父进程必须 SIGKILL，不能依赖关闭连接落盘。
            await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())

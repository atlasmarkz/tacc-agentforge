"""用显式模拟能力演示两个进程（或作业）之间的问题恢复，不访问模型服务。"""

import argparse
import asyncio
import json
from contextlib import aclosing

from agent import AgentDependencies, AgentRuntime, RunPolicy
from agent.checkpoint_cleanup import cleanup_checkpoints
from agent.checkpoints import open_checkpointer
from agent.examples.mock_run import MockModel, MockPlanValidator, demo_request, demo_tools
from agent.settings import load_runtime_settings


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--profile")
    parser.add_argument("--db", required=True)
    parser.add_argument("--events", action="store_true", help="emit AgentEvent JSON lines")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--run-id", required=True)
    start.add_argument("--question-timeout-s", type=float)
    start.add_argument("--hold-after-waiting", action="store_true",
                       help="keep the process open so it can be killed without closing the saver")
    resume = commands.add_parser("resume")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--question-id", required=True)
    resume.add_argument("--answer", required=True, help="JSON object of requested constraint fields")
    args = parser.parse_args()
    settings = load_runtime_settings(args.config, args.profile)
    policy = settings.policy
    if args.command == "start" and args.question_timeout_s is not None:
        policy = RunPolicy.model_validate({**policy.model_dump(), "question_timeout_s": args.question_timeout_s})
    answer = json.loads(args.answer) if args.command == "resume" else None
    async with open_checkpointer(args.db, create=args.command == "start") as saver:
        # 尚未接收任务；同一存储的其他宿主和手工清理由独占租约拒绝。
        if settings.checkpoint.cleanup_on_start:
            await cleanup_checkpoints(saver, settings.checkpoint.retention())
        runtime = AgentRuntime(AgentDependencies(MockModel(), demo_tools(), MockPlanValidator()),
                               policy, checkpointer=saver)
        if args.command == "start":
            request = demo_request(args.run_id).model_copy(update={"constraints": {"currency": "USD"}})
            stream = runtime.astream(request)
        else:
            stream = runtime.astream_resume(args.run_id, args.question_id, answer)
        result = None
        async with aclosing(stream):
            async for event in stream:
                if args.events:
                    print(event.model_dump_json(), flush=True)
                if event.result is not None:
                    result = event.result
        if not args.events:
            print(result.model_dump_json(), flush=True)
        if args.command == "start" and args.hold_after_waiting and result.status == "waiting_user":
            await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())

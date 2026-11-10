"""用本地 HTTP 模型驱动真实 Agent，验证工具、方案校验和问答恢复。

读取项目 .env，模型使用真实推理；商品工具和预算校验使用显式模拟数据。
先启动本地模型服务器，再运行 python -m agent.examples.local_run --help 查看参数。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import time
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from agent import AgentDependencies, AgentRequest, AgentRuntime, RunPolicy
from agent.examples.mock_run import MockPlanValidator, demo_tools
from inference.client import ModelClient
from inference.model_port import VllmModelPort

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path.home() / "models" / "qwen" / "qwen3.5-4b"


@dataclass(frozen=True)
class LocalSettings:
    """只保存与本地推理有关的配置，不携带 .env 中的密钥。"""

    base_url: str
    model_path: Path
    model_id: str
    env_file: Path
    env_loaded: bool


def load_settings(args) -> LocalSettings:
    """显式参数优先于环境配置；本地测试只允许无凭证的回环 HTTP 地址。"""
    from dotenv import dotenv_values

    env_file = Path(args.env_file).expanduser().resolve()
    values = dotenv_values(env_file) if env_file.is_file() else {}

    def configured(name):
        return os.environ.get(name) or values.get(name)

    base_url = (args.base_url or configured("LOCAL_MODEL_BASE_URL")
                or configured("QWEN_BASE_URL") or "http://127.0.0.1:18080/v1")
    parts = urlsplit(base_url)
    if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path.rstrip("/") not in {"", "/v1"}):
        raise ValueError("本地测试只允许回环 HTTP 地址，请通过 --base-url 指定本地服务。")
    base_url = base_url.rstrip("/")
    if not parts.path.rstrip("/"):
        base_url += "/v1"
    model_path = Path(args.model_path or configured("LOCAL_MODEL_PATH") or DEFAULT_MODEL).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise ValueError("本地模型目录缺少 config.json，请检查 --model-path。")
    return LocalSettings(
        base_url=base_url, model_path=model_path,
        model_id=args.model_id or configured("LOCAL_MODEL_ID") or str(model_path),
        env_file=env_file, env_loaded=env_file.is_file(),
    )


class RecordedModelPort:
    """记录真实模型回复和耗时；不替换动作、不为模型伪造成功回复。"""

    def __init__(self, delegate, output: Path):
        self.delegate, self.output = delegate, output
        self.calls = []

    async def generate(self, request, context):
        started = time.perf_counter()
        record = {"operation_id": context.operation_id}
        try:
            reply = await self.delegate.generate(request, context)
            record.update(reply.model_dump(mode="json"))
            return reply
        except Exception as exc:
            # 异常只记录类型，避免把连接配置或潜在凭证写入测试产物。
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["duration_s"] = round(time.perf_counter() - started, 3)
            self.calls.append(record)
            self.output.write_text(json.dumps(self.calls, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_scenario(name, result, events, calls, resumed):
    """必须实际经过模型、工具及校验路径，不能仅以 HTTP 成功判断全流程通过。"""
    checks = {
        "completed": result is not None and result.status == "completed",
        "real_model_replied": any("content" in call for call in calls),
        "tool_executed": any(e.type == "tool_finished" and e.data.get("status") == "ok" for e in events),
        "plan_validated": any(e.type == "validation_finished" and e.data.get("status") == "passed" for e in events),
        "expected_resume_behavior": resumed > 0 if name == "clarify" else resumed == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


async def run_scenario(name, settings: LocalSettings, args, output: Path):
    """每个场景使用独立任务和检查点，恢复时只提交事先给定的模拟用户答案。"""
    client = ModelClient(base_url=settings.base_url, model=settings.model_id, max_attempts=1)
    # MLX 服务不实现 response_format 强制解码；保留提示中的 schema 和 Harness 校验。
    port = RecordedModelPort(VllmModelPort(
        client, temperature=args.temperature, enable_thinking=args.thinking, constrain_output=False,
    ), output / f"{name}-model-calls.json")
    runtime = AgentRuntime(AgentDependencies(port, demo_tools(), MockPlanValidator()), RunPolicy(
        call_timeout_s=args.timeout, deadline_seconds=args.deadline, max_model_calls=8,
        max_tool_calls=4, max_questions=3, max_output_tokens=args.max_tokens,
    ))
    message = "请调用搜索工具，从当前知识快照中挑选一件适合通勤的无线耳机，取得报价和证据后提交候选方案。"
    constraints = {"currency": "USD"}
    if name == "direct":
        constraints["purchase_cap"] = "100.00"
        message += "购买上限 purchase_cap 已明确为 100.00 USD。"
    else:
        message += "必要的购买上限 purchase_cap 尚未提供，请先询问此字段。"
    run_id = f"local-{name}-{uuid4().hex[:12]}"
    request = AgentRequest(run_id=run_id, conversation_id=run_id,
                           snapshot_id="demo-snapshot-v1", message=message, constraints=constraints)
    events, result, resumed = [], None, 0
    answer_pool = {"purchase_cap": "100.00", "currency": "USD"}
    trace_path = output / f"{name}-events.jsonl"
    started = time.perf_counter()

    async def consume(stream):
        nonlocal result
        async with aclosing(stream):
            async for event in stream:
                events.append(event)
                with trace_path.open("a", encoding="utf-8") as trace:
                    trace.write(event.model_dump_json() + "\n")
                print(f"[{name}] {event.sequence:02d} {event.node}: {event.type}", flush=True)
                if event.result is not None:
                    result = event.result

    await consume(runtime.astream(request))
    unanswered = []
    while result is not None and result.status == "waiting_user" and resumed < 3:
        question = result.pending_question
        unanswered = [field for field in question.required_fields if field not in answer_pool]
        if unanswered:
            break  # 不为模型提出的其他问题编造答案。
        resumed += 1
        await consume(runtime.astream_resume(run_id, question.question_id,
                      {field: answer_pool[field] for field in question.required_fields}))
    return {
        "scenario": name, **audit_scenario(name, result, events, port.calls, resumed),
        "resumes": resumed, "unanswered_fields": unanswered,
        "wall_s": round(time.perf_counter() - started, 3),
        "result": result.model_dump(mode="json") if result is not None else None,
        "model_calls_file": port.output.name, "events_file": trace_path.name,
    }


def parser():
    """本地测试参数；不会写回 .env 或更改生产模型服务配置。"""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--env-file", default=str(ROOT / ".env"))
    result.add_argument("--base-url")
    result.add_argument("--model-path")
    result.add_argument("--model-id")
    result.add_argument("--scenario", choices=["all", "direct", "clarify"], default="all")
    result.add_argument("--timeout", type=float, default=180)
    result.add_argument("--deadline", type=float, default=600)
    result.add_argument("--max-tokens", type=int, default=2048)
    result.add_argument("--temperature", type=float, default=0.6)
    result.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--output", type=Path)
    return result


async def main():
    args = parser().parse_args()
    settings = load_settings(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "agent" / "results" / "local-qwen35" / stamp
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((settings.model_path / "config.json").read_text(encoding="utf-8"))
    report = {
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(settings.model_path), "model_type": config.get("model_type"),
        "base_url": settings.base_url, "env_file": str(settings.env_file), "env_loaded": settings.env_loaded,
        "python": platform.python_version(), "architecture": platform.machine(),
        "versions": {name: importlib.metadata.version(name) for name in ["langgraph", "mlx-lm", "mlx", "transformers"]},
        "real_model": True, "synthetic_catalog_and_validator": True,
        "constrained_decoding": False, "scenarios": [], "passed": False,
        "generation": {"temperature": args.temperature, "enable_thinking": args.thinking,
                       "max_output_tokens": args.max_tokens},
    }
    path = output / "report.json"
    try:
        for name in (["direct", "clarify"] if args.scenario == "all" else [args.scenario]):
            report["scenarios"].append(await run_scenario(name, settings, args, output))
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["passed"] = all(case["passed"] for case in report["scenarios"])
    finally:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"测试报告：{path}", flush=True)
    print(json.dumps({"passed": report["passed"], "scenarios": [
        {"name": case["scenario"], "passed": case["passed"], "wall_s": case["wall_s"]}
        for case in report["scenarios"]]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

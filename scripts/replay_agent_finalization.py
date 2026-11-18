"""Replay issue #21 and selected dev cases with real tools/validator and a fast profile.

Run this same file in separate before/after source trees. Only the model transport
is overridden; the eight-decision policy, sampling and output budget stay intact.
No held-out cases are loaded or run.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone

from agent import AgentDependencies, AgentRequest, AgentRuntime
from agent.budget import BudgetPlanValidator
from agent.settings import load_runtime_settings
from evaluations.catalog import Catalog, ground_truth
from evaluations.run import ID_PATTERN, USER_FACING_ISSUES
from evaluations.score import score_record
from inference.client import ModelClient
from inference.model_port import VllmModelPort
from knowledge.agent_tools import build_knowledge_tools
from knowledge.budget_source import KnowledgeOfferCostSource
from knowledge.service import KnowledgeService

ROOT = Path(__file__).resolve().parents[1]


class RecordedModel:
    def __init__(self, model):
        self.model, self.calls = model, []

    async def generate(self, request, context):
        entry = {"request": request.model_dump(mode="json"), "operation_id": context.operation_id}
        self.calls.append(entry)
        reply = await self.model.generate(request, context)
        entry["reply"] = reply.model_dump(mode="json")
        return reply


async def replay(args):
    settings = load_runtime_settings(ROOT / "configs/agent.yaml", args.profile)
    assert settings.policy.max_model_calls == 8
    catalog = Catalog(args.snapshots_root / args.snapshot_id / "catalog.sqlite")
    dev = {case["case_id"]: case for case in map(json.loads, (ROOT / "evaluations/cases.jsonl").read_text().splitlines())
           if case["split"] == "dev"}
    try:
        truths = {name: ground_truth(dev[name], catalog) for name in args.dev_cases}
    finally:
        catalog.close()
    if truths.get("dev-07", {}).get("expected", "no_feasible_plan") != "no_feasible_plan":
        raise ValueError("dev-07 is feasible in this snapshot; use the frozen development snapshot")
    if truths.get("dev-04", {}).get("expected", "plan") != "plan":
        raise ValueError("dev-04 must have a feasible plan in the selected development snapshot")
    cases = [{"case_id": "issue-21-charger", "message": "I need a USB-C charger for my laptop and phone.",
              "constraints": {"purchase_cap": "60.00", "currency": "USD"}},
             *[dev[name] for name in args.dev_cases]]
    report = {"label": args.label, "tested_at": datetime.now(timezone.utc).isoformat(),
              "profile": settings.config.profile, "snapshot_id": args.snapshot_id, "real_model": True,
              "policy": settings.policy.model_dump(), "model": settings.model.port_options(),
              "model_name": settings.model.name, "ground_truth": truths,
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((ROOT / "agent").glob("*.py")) +
                                [ROOT / "configs/agent.yaml", ROOT / "inference/model_port.py",
                                 ROOT / "evaluations/run.py", ROOT / "evaluations/score.py",
                                 ROOT / "scripts/replay_agent_finalization.py"]}, "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rep in range(1, args.repeats + 1):
        for case in cases:
            kb = KnowledgeService(args.snapshots_root)
            tools, costs = build_knowledge_tools(kb), KnowledgeOfferCostSource(kb)
            client = ModelClient(base_url=args.base_url, address_file=args.address_file,
                                 model=settings.model.name, **settings.model.client_options())
            model = RecordedModel(VllmModelPort(client, **settings.model.port_options()))
            runtime = AgentRuntime(AgentDependencies(model, tools, BudgetPlanValidator(costs)), settings.policy)
            run_id = f"replay-{case['case_id']}-{rep}"
            request = AgentRequest(run_id=run_id, conversation_id=run_id, snapshot_id=args.snapshot_id,
                                   message=case["message"], constraints=case["constraints"])
            started = time.monotonic()
            try:
                result = await runtime.run(request)
                trace = await runtime.export_trace(run_id)
                assert result.usage.model_calls == len(model.calls) <= 8
                entry = {"case_id": case["case_id"], "rep": rep, "request": request.model_dump(mode="json"),
                         "status": result.status, "stop_reason": result.stop_reason,
                         "model_calls": result.usage.model_calls, "tool_calls": result.usage.tool_calls,
                         "result": result.model_dump(mode="json"), "trace": trace,
                         "calls": model.calls, "wall_s": round(time.monotonic() - started, 3)}
                if case["case_id"] in truths:
                    if result.verified_plan is not None:
                        outcome = {"kind": "plan", "items": [i.model_dump(mode="json") for i in result.verified_plan.items]}
                    elif result.clarification is not None:
                        outcome = {"kind": "ask_user", **result.clarification.model_dump(mode="json")}
                    elif result.stop_reason in USER_FACING_ISSUES:
                        outcome = {"kind": "abstain", "reason": result.stop_reason, "message": result.message}
                    else:
                        outcome = {"kind": "error", "error": result.stop_reason or result.status}
                    entry["outcome"] = outcome
                    observed_ids = ID_PATTERN.findall(json.dumps(model.calls[-1]["request"]["messages"])) if model.calls else []
                    catalog = Catalog(args.snapshots_root / args.snapshot_id / "catalog.sqlite")
                    try:
                        entry["score"] = score_record({**entry, "config": "B3v", "context_ids": observed_ids},
                                                       case, truths[case["case_id"]], catalog)
                    finally:
                        catalog.close()
                report["runs"].append(entry)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps({k: entry[k] for k in ("case_id", "rep", "model_calls", "tool_calls",
                                                       "status", "stop_reason", "wall_s")}), flush=True)
            finally:
                tools.get("search_products").executor.close()
                costs.close()


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="tacc-fast", help="Agent profile (main: tacc-fast; Cloud branch: cloud-fast)")
    parser.add_argument("--snapshots-root", required=True, type=Path)
    parser.add_argument("--snapshot-id", default="snapshot_5bcd00723fb0b7589cbe50c8")
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--base-url")
    transport.add_argument("--address-file", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--dev-cases", nargs="+", choices=["dev-04", "dev-07"], default=["dev-07"])
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main():
    asyncio.run(replay(parser().parse_args()))


if __name__ == "__main__":
    main()

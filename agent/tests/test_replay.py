"""Main-branch replay entry point with a scripted model and the real frozen catalog."""

import asyncio
import json

from agent import ModelReply
from knowledge.bootstrap import install_snapshot
from knowledge.settings import load_settings
from scripts import replay_agent_finalization as replay


def test_replay_defaults_to_a_main_profile_and_runs_without_cloud_configuration(tmp_path, monkeypatch):
    settings = load_settings(data_root=tmp_path / "knowledge")
    install_snapshot(settings)
    output = tmp_path / "report.json"
    args = replay.parser().parse_args([
        "--snapshots-root", str(settings.snapshots_root),
        "--base-url", "http://model.invalid/v1",
        "--label", "main-entrypoint-test", "--output", str(output),
    ])
    assert args.profile == "tacc-fast"

    class ScriptedPort:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        async def generate(self, request, context):
            self.calls += 1
            if "issue-21-charger" in context.run_id:
                action = {"type": "ask_user", "question": "Required laptop wattage?",
                          "required_fields": ["laptop_wattage"]}
            elif self.calls == 1:
                action = {"type": "call_tool", "tool_name": "search_products", "arguments": {"filters": {
                    "category": "keyboards", "currency": "USD", "max_price_minor": 1000,
                    "include_unknown": False}}}
            else:
                action = {"type": "report_incomplete", "reason": "no_feasible_plan",
                          "message": "No verifiable keyboard under 10 USD in this catalog."}
            return ModelReply(content=json.dumps(action))

    monkeypatch.setattr(replay, "VllmModelPort", ScriptedPort)
    asyncio.run(replay.replay(args))
    report = json.loads(output.read_text())
    assert report["profile"] == "tacc-fast"
    assert report["policy"]["max_model_calls"] == 8
    assert report["model"]["enable_thinking"] is False
    assert report["policy"]["max_output_tokens"] == 512
    charger, keyboard = report["runs"]
    assert charger["status"] == "waiting_user" and charger["model_calls"] == 1
    assert keyboard["stop_reason"] == "no_feasible_plan" and keyboard["model_calls"] == 2
    assert keyboard["score"]["success"]

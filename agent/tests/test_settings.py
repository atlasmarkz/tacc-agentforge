"""Agent 配置控制请求预算、思考开关和路径，不接管其他任务的配置。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
import yaml

from agent.config import DEFAULT_CONFIG, load_config
from agent import RunPolicy
from agent.contracts import CallContext
from agent.examples.mock_run import demo_request, demo_tools
from agent.prompts import build_model_request
from agent.settings import load_runtime_settings
from agent.state import initial_state
from inference.model_port import VllmModelPort


def write_config(tmp_path, change):
    value = yaml.safe_load(DEFAULT_CONFIG.read_text())
    change(value)
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(value, allow_unicode=True))
    return path


@pytest.mark.parametrize("profile,tokens,thinking", [("tacc", 2048, None), ("tacc-thinking", 2048, True), ("tacc-fast", 512, False)])
def test_real_tool_profiles_control_model_request_and_port(profile, tokens, thinking):
    settings = load_runtime_settings(profile=profile)
    assert settings.policy.max_context_chars == 36000
    assert settings.policy.max_history_messages >= 16
    request = build_model_request(initial_state(demo_request(), settings.policy), demo_tools(), settings.policy)
    captured = {}
    class Client:
        def chat(self, messages, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content="{}", prompt_tokens=1, completion_tokens=1, finish_reason="stop")
    async def scenario():
        context = CallContext(run_id="config", snapshot_id="snapshot", operation_id="config:model:1",
                              timeout_s=5, cancellation=asyncio.Event())
        await VllmModelPort(Client(), **settings.model.port_options()).generate(request, context)
    asyncio.run(scenario())
    assert captured["max_tokens"] == tokens
    assert captured["enable_thinking"] is thinking


def test_real_tool_context_keeps_three_large_observations():
    settings = load_runtime_settings(profile="tacc-thinking")
    state = initial_state(demo_request(), settings.policy)
    for index in range(3):
        state["history"] += [{"role": "assistant", "content": json.dumps({"type": "call_tool", "tool_name": "search_products"})},
                             {"role": "user", "content": json.dumps({"observation": {"marker": f"search-{index}", "text": "x" * 7600}})}]
    current = build_model_request(state, demo_tools(), settings.policy)
    old = build_model_request(state, demo_tools(), RunPolicy())
    assert all(f"search-{index}" in current.model_dump_json() for index in range(3))
    assert "search-0" not in old.model_dump_json()
    assert len(current.model_dump_json()) <= settings.policy.max_context_chars


def test_loader_is_cwd_independent_and_resolves_profile_inheritance(tmp_path, monkeypatch):
    path = write_config(tmp_path, lambda value: value["profiles"].update({"custom": {"extends": "tacc-fast", "agent": {"max_tool_calls": 3}}}))
    monkeypatch.chdir(tmp_path.parent)
    settings = load_runtime_settings(path, "custom")
    assert settings.policy.max_tool_calls == 3 and settings.policy.max_output_tokens == 512
    assert settings.model.enable_thinking is False
    monkeypatch.setenv("TACC_USERDIR", str(tmp_path / "user"))
    assert settings.config.resolve_path(settings.checkpoint.path) == tmp_path / "user/tacc-agentforge/checkpoints/agent.sqlite"
    monkeypatch.delenv("TACC_USERDIR")
    with pytest.raises(ValueError, match="undefined environment"):
        settings.config.resolve_path(settings.checkpoint.path)


def test_invalid_configuration_fails_without_silent_defaults(tmp_path):
    for change in (
        lambda value: value["defaults"]["agent"].update({"max_context_char": 36000}),
        lambda value: value["defaults"]["agent"].update({"max_output_tokens": -1}),
        lambda value: value["defaults"]["model"].update({"context_window_tokens": 1024}),
        lambda value: value["profiles"].update({"tacc": {"extends": "tacc"}}),
    ):
        path = write_config(tmp_path, change)
        with pytest.raises(ValueError):
            load_runtime_settings(path, "tacc")
    path.write_text("version: 1\nversion: 1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)
    with pytest.raises(ValueError, match="Unknown"):
        load_config(profile="misspelled")


def test_local_explicit_overrides_win_over_yaml(tmp_path):
    pytest.importorskip("dotenv")
    from agent.examples.local_run import load_settings, parser
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    path = write_config(tmp_path, lambda value: value["profiles"]["local"]["agent"].update({"max_output_tokens": 3072}))
    args = parser().parse_args(["--config", str(path), "--model-path", str(model), "--env-file", str(tmp_path / "empty.env")])
    assert load_settings(args).runtime.policy.max_output_tokens == 3072
    overridden = parser().parse_args(["--config", str(path), "--model-path", str(model), "--env-file", str(tmp_path / "empty.env"),
                                      "--max-tokens", "1024", "--no-thinking", "--temperature", "0"])
    settings = load_settings(overridden)
    assert settings.runtime.policy.max_output_tokens == 1024
    assert settings.runtime.model.enable_thinking is False and settings.runtime.model.temperature == 0


@pytest.mark.parametrize("section", ["service", "knowledge", "e2e"])
def test_agent_configuration_rejects_other_task_sections(tmp_path, section):
    path = write_config(tmp_path, lambda value: value["defaults"].update({section: {}}))
    with pytest.raises(ValueError, match="Extra inputs"):
        load_runtime_settings(path, "tacc")

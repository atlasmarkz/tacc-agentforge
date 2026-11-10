"""验证本地模型测试的配置隔离与验收条件，不加载权重或调用真实服务。"""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from agent.examples.local_run import audit_scenario, load_settings, parser


def test_env_file_is_read_without_exporting_or_reporting_secrets(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    for key in ["QWEN_BASE_URL", "LOCAL_MODEL_BASE_URL", "LOCAL_MODEL_PATH", "LOCAL_MODEL_ID", "SERPAPI_KEY"]:
        monkeypatch.delenv(key, raising=False)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    env = tmp_path / ".env"
    env.write_text(f"SERPAPI_KEY=private-test-secret\nQWEN_BASE_URL=http://cluster-node:8000/v1\nLOCAL_MODEL_PATH={model}\n")
    args = parser().parse_args(["--env-file", str(env), "--base-url", "http://127.0.0.1:18080/v1"])
    settings = load_settings(args)
    assert settings.env_loaded and settings.model_path == model
    assert settings.base_url == "http://127.0.0.1:18080/v1"
    assert "private-test-secret" not in str(asdict(settings))
    import os
    assert "SERPAPI_KEY" not in os.environ


@pytest.mark.parametrize("url", ["https://remote.example/v1", "http://remote.example/v1",
                                "http://key:secret@localhost:18080/v1", "http://localhost:18080/v1?api_key=secret"])
def test_local_run_rejects_remote_or_credential_bearing_endpoints(url):
    pytest.importorskip("dotenv")
    args = parser().parse_args(["--base-url", url])
    with pytest.raises(ValueError, match="回环 HTTP"):
        load_settings(args)


@pytest.mark.parametrize("missing", ["tool", "validation", "model", "resume"])
def test_completed_status_alone_cannot_pass_the_live_scenario_audit(missing):
    events = [
        SimpleNamespace(type="tool_finished", data={"status": "ok"}),
        SimpleNamespace(type="validation_finished", data={"status": "passed"}),
    ]
    calls, resumed = [{"content": "{}"}], 1
    if missing == "tool":
        events = events[1:]
    elif missing == "validation":
        events = events[:1]
    elif missing == "model":
        calls = []
    else:
        resumed = 0
    result = audit_scenario("clarify", SimpleNamespace(status="completed"), events, calls, resumed)
    assert not result["passed"]

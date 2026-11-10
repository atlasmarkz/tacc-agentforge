"""Load generator (deploy/tacc/service/load-test/loadgen.py) against the fake streaming vLLM server."""
import importlib
import importlib.util
import random
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.fake_vllm import FakeVllm, Stream, sse_events

LOADGEN = Path(__file__).resolve().parents[1] / "deploy/tacc/service/load-test/loadgen.py"
WORKLOAD = {"source": "test", "system": "You are a shopping advisor." * 20, "pairs": [[600, 20], [1500, 40]]}
CAL = {"base_tokens": 150, "chars_per_token": 3.0}


def load():
    sys.modules.setdefault("client", importlib.import_module("inference.client"))   # staged name on TACC
    spec = importlib.util.spec_from_file_location("loadgen_under_test", LOADGEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server():
    srv = FakeVllm([(200, Stream(sse_events(["a"] * 5), interval_s=0.01), 0)])
    yield srv
    srv.close()


def test_messages_are_padded_towards_the_target_length_and_unique():
    lg = load()
    rng = random.Random(1)
    a = lg.make_messages("S" * 300, 2000, rng, 100, 3.0)
    b = lg.make_messages("S" * 300, 2000, rng, 100, 3.0)
    filler = len(a[-1]["content"])
    assert [m["role"] for m in a] == ["system", "user", "assistant", "user"]
    assert 5700 <= filler <= 6400                        # (2000 - 100) tokens * 3 chars, plus one item
    assert a[-1]["content"] != b[-1]["content"]          # no artificial prefix-cache hits beyond the system prompt
    assert len(lg.make_messages("S", 50, rng, 100, 3.0)) == 2


def test_open_loop_phase_sends_at_the_offered_rate_and_summarises(server):
    lg = load()
    client = lg.ModelClient(base_url=server.url, max_attempts=1)
    summary, recs = lg.load_phase(client, WORKLOAD, CAL, rate=10, duration=1.5, timeout=5, rng=random.Random(2))
    assert 5 <= summary["sent"] <= 30 and summary["ok"] == summary["sent"] and summary["errors"] == {}
    assert summary["latency_s"]["p50"] is not None and summary["ttft_s"]["p50"] is not None
    body = server.requests[-1]
    assert body["stream"] is True and body["ignore_eos"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_fault_phase_measures_withdrawal_and_never_hangs(server, tmp_path):
    lg = load()
    address_file = tmp_path / "service-url.txt"
    address_file.write_text(server.url + "\n")
    trigger = tmp_path / "fault-request"

    def fake_service():                                   # what serve.py does in validate mode
        while not trigger.exists():
            time.sleep(0.05)
        assert trigger.read_text().strip() == server.url
        trigger.unlink()
        address_file.unlink()
        server.close()

    threading.Thread(target=fake_service, daemon=True).start()
    summary, recs = lg.fault_phase(address_file, server.url, WORKLOAD, CAL, rate=8, duration=3, kill_at=1,
                                   timeout=2, rng=random.Random(3), events=[])
    assert summary["trigger_s"] is not None and summary["address_gone_after_trigger_s"] is not None
    assert summary["address_gone_after_trigger_s"] < 1.0
    assert summary["hung_past_budget"] == 0
    assert any(k != "ok" for k in summary["outcomes_after_trigger"])     # callers saw a typed failure

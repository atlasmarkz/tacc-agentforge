import json
import socket
import threading
import time

import pytest

from tests.fake_vllm import OK_BODY, FakeVllm

from inference.client import (
    BadResponse, Cancelled, CancelToken, ModelClient, ServiceRejected, ServiceTimeout, ServiceUnavailable,
)

@pytest.fixture
def make_server():
    servers = []

    def make(script):
        s = FakeVllm(script)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s.close()


def client_for(url=None, **kw):
    kw.setdefault("backoff_base_s", 0.01)
    kw.setdefault("backoff_max_s", 0.05)
    return ModelClient(base_url=url, **kw)


MSG = [{"role": "user", "content": "hi"}]


def test_success_parses_content_usage_reasoning(make_server):
    srv = make_server([(200, OK_BODY, 0)])
    r = client_for(srv.url).chat(MSG)
    assert r.content == '{"type":"ask_user"}' and r.reasoning == "thought"
    assert (r.prompt_tokens, r.completion_tokens, r.finish_reason, r.attempts) == (11, 7, "stop", 1)
    assert r.served_by == srv.url


def test_request_body_options(make_server):
    srv = make_server([(200, OK_BODY, 0)])
    schema = {"type": "object"}
    client_for(srv.url).chat(MSG, max_tokens=99, temperature=0.0, json_schema=schema, enable_thinking=False)
    body = srv.requests[0]
    assert body["max_tokens"] == 99 and body["stream"] is False
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_finish_reasons_are_normalised(make_server):
    def body(reason):
        return {"choices": [{"message": {"content": "x"}, "finish_reason": reason}]}
    srv = make_server([(200, body("length"), 0), (200, body("tool_calls"), 0)])
    c = client_for(srv.url)
    assert c.chat(MSG).finish_reason == "length"
    r = c.chat(MSG)
    assert r.finish_reason == "error" and r.raw_finish_reason == "tool_calls"


def test_transient_errors_are_retried(make_server):
    srv = make_server([(503, {"error": "starting"}, 0), (500, {"error": "hiccup"}, 0), (200, OK_BODY, 0)])
    r = client_for(srv.url).chat(MSG)
    assert r.attempts == 3 and len(srv.requests) == 3


def test_gives_up_after_max_attempts(make_server):
    srv = make_server([(503, {"error": "down"}, 0)])
    with pytest.raises(ServiceUnavailable):
        client_for(srv.url, max_attempts=3).chat(MSG)
    assert len(srv.requests) == 3


def test_client_errors_are_not_retried(make_server):
    srv = make_server([(400, {"error": "bad"}, 0)])
    with pytest.raises(ServiceRejected):
        client_for(srv.url).chat(MSG)
    assert len(srv.requests) == 1


def test_malformed_success_body(make_server):
    srv = make_server([(200, b"not json", 0)])
    with pytest.raises(BadResponse):
        client_for(srv.url).chat(MSG)


def test_slow_generation_hits_time_budget(make_server):
    srv = make_server([(200, OK_BODY, 3.0)])
    t0 = time.monotonic()
    with pytest.raises(ServiceTimeout):
        client_for(srv.url).chat(MSG, timeout_s=0.4)
    assert time.monotonic() - t0 < 1.5


def _closed_port_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}/v1"


def test_picks_up_new_address_while_retrying(make_server, tmp_path):
    srv = make_server([(200, OK_BODY, 0)])
    addr = tmp_path / "service-url.txt"
    addr.write_text(_closed_port_url())
    threading.Timer(0.3, lambda: addr.write_text(srv.url + "\n")).start()
    c = ModelClient(address_file=addr, backoff_base_s=0.1, backoff_max_s=0.2, max_attempts=50)
    r = c.chat(MSG, timeout_s=5)
    assert r.served_by == srv.url and r.attempts > 1


def test_no_address_yet_is_unavailable(tmp_path):
    c = ModelClient(address_file=tmp_path / "missing.txt", backoff_base_s=0.01, backoff_max_s=0.02, max_attempts=2)
    with pytest.raises(ServiceUnavailable):
        c.chat(MSG, timeout_s=2)


def test_budget_covers_retries(tmp_path):
    c = ModelClient(base_url=_closed_port_url(), backoff_base_s=0.2, backoff_max_s=0.2, max_attempts=100)
    t0 = time.monotonic()
    with pytest.raises((ServiceTimeout, ServiceUnavailable)):
        c.chat(MSG, timeout_s=0.5)
    assert time.monotonic() - t0 < 1.5


def test_cancel_interrupts_request_in_flight(make_server):
    srv = make_server([(200, OK_BODY, 3.0)])
    token = CancelToken()
    threading.Timer(0.2, token.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        client_for(srv.url).chat(MSG, timeout_s=10, cancel=token)
    assert time.monotonic() - t0 < 1.5


def test_cancel_interrupts_backoff():
    token = CancelToken()
    threading.Timer(0.2, token.cancel).start()
    c = ModelClient(base_url=_closed_port_url(), backoff_base_s=20, backoff_max_s=20, max_attempts=10)
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        c.chat(MSG, timeout_s=60, cancel=token)
    # Windows needs about 2 s to report a refused connection; the 20 s back-off must not be waited out.
    assert time.monotonic() - t0 < 6


def test_cancelled_before_start():
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        ModelClient(base_url=_closed_port_url()).chat(MSG, cancel=token)

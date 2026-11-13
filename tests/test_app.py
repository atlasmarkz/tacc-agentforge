"""Task 5: harness service and local backend (app/). Uses the seed snapshot, a scripted model and real HTTP."""
import asyncio
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("pydantic", minversion="2")

ROOT = Path(__file__).resolve().parents[1]


def test_service_selftest_end_to_end():
    from app.selftest import run_selftest
    assert asyncio.run(run_selftest(ROOT / "configs/knowledge.json")) == 0


class FakeHarness:
    """Stands in for HarnessClient: records calls, streams a short run."""

    def __init__(self):
        self.calls = []

    def get(self, path):
        return {"status": "ok", "job_id": "j1", "snapshot_id": "snap"}

    def post(self, path, payload):
        self.calls.append(("post", path, payload))
        return {"cancelled": True}

    def stream(self, path, payload):
        self.calls.append(("stream", path, payload))
        yield "agent", {"run_id": payload.get("run_id", "x"), "sequence": 1, "node": "prepare",
                        "type": "run_completed", "data": {},
                        "result": {"status": "completed", "snapshot_id": "snap"}}
        yield "end", {"run_id": "x", "wall_s": 0.1, "job_id": "j1"}


@pytest.fixture()
def backend(tmp_path):
    from app.backend import Backend, make_handler
    fake = FakeHarness()
    b = Backend(fake, tmp_path / "requests.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(b))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], fake, b, tmp_path
    server.shutdown()


def call(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read().decode("utf-8")
    conn.close()
    return r.status, data


def test_backend_creates_run_ids_and_keeps_sessions_apart(backend):
    port, fake, b, tmp = backend
    alice, bob = "aaaaaaaa-1111", "bbbbbbbb-2222"
    status, text = call(port, "/api/ask", {"session_id": alice, "message": "earbuds", "constraints": {}})
    assert status == 200 and "event: meta" in text and "event: end" in text
    run_id = json.loads(text.split("event: meta\ndata: ")[1].split("\n")[0])["run_id"]
    assert run_id.startswith("r-aaaaaaaa-")
    sent = fake.calls[-1][2]
    assert sent["run_id"] == run_id and sent["conversation_id"] == f"s-{alice}"   # the browser cannot choose IDs
    # another session can neither answer, cancel, inspect nor revise this run
    for path, body in (("/api/answer", {"run_id": run_id, "question_id": "q", "answer": {}}),
                       ("/api/cancel", {"run_id": run_id}),
                       ("/api/lookup", {"run_id": run_id, "offer_ids": ["o"]}),
                       ("/api/ask", {"message": "cheaper", "revises": run_id})):
        assert call(port, path, {"session_id": bob, **body})[0] == 403
    # the owner can revise; the revision is a new run that points at its parent
    status, text = call(port, "/api/ask", {"session_id": alice, "message": "cheaper", "revises": run_id})
    assert status == 200
    rows = [json.loads(l) for l in (tmp / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["request", "revision"]
    assert rows[1]["parent"] == run_id and rows[0]["status"] == "completed" and rows[0]["harness_job"] == "j1"


def test_backend_rejects_bad_sessions_and_reports_an_unreachable_harness(backend, tmp_path):
    port, fake, b, tmp = backend
    assert call(port, "/api/ask", {"session_id": "x", "message": "hi"})[0] == 400
    from app.backend import HarnessError

    def down(*a, **k):
        raise HarnessError("not running")
        yield                                                    # pragma: no cover
    fake.stream = down
    status, text = call(port, "/api/ask", {"session_id": "cccccccc-3333", "message": "hi"})
    assert status == 503 and "harness_unavailable" in text
    rows = [json.loads(l) for l in (tmp / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["status"] == "unreachable"

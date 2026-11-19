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


def test_service_publishes_and_stops_on_its_stop_file(tmp_path):
    """The vLLM job stops the harness with a stop file: it must exit and withdraw its own address."""
    import os
    import subprocess
    import sys
    import time
    env = dict(os.environ, TACC_USERDIR=str(tmp_path), PYTHONUTF8="1")
    stop = tmp_path / "harness-stop"
    proc = subprocess.Popen([sys.executable, "-m", "app.harness_service", "--port", "0", "--stop-file", str(stop)],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = tmp_path / "tacc-agentforge-service/harness-url.txt"
    try:
        for _ in range(240):
            if url.exists() or proc.poll() is not None:
                break
            time.sleep(0.25)
        assert url.exists(), proc.stdout.read() if proc.poll() is not None else "not published"
        assert (tmp_path / "tacc-agentforge-service/harness-token.txt").exists()
        stop.touch()
        out, _ = proc.communicate(timeout=30)
        assert proc.returncode == 0, out
        assert not url.exists() and "stop file" in out
        assert "0 running (they finish here), 0 waiting for an answer" in out     # what a handover drops
    finally:
        if proc.poll() is None:
            proc.kill()


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


def test_backend_rereads_the_address_once_when_the_service_moved(backend):
    """After a handover the old tunnel still accepts locally; the backend re-reads the address and retries."""
    port, fake, b, tmp = backend
    from app.backend import HarnessError
    refreshed, real_stream = [], fake.stream

    def moved_once(path, payload):
        if not refreshed:
            raise HarnessError("connection refused")
        yield from real_stream(path, payload)
    fake.stream = moved_once
    b.refresh = lambda: refreshed.append(1)
    status, text = call(port, "/api/ask", {"session_id": "dddddddd-4444", "message": "earbuds"})
    assert status == 200 and "event: end" in text and refreshed == [1]


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


# ---- review on #20: harness answers are passed on, /runs retries as a new run, waiting runs are reported ----

def _harness_answers(status, code):
    from app.backend import HarnessError

    def answer(*a, **k):
        raise HarnessError(f"harness HTTP {status}: {code}", status=status, code=code)
        yield                                                    # pragma: no cover
    return answer


@pytest.mark.parametrize("status,code", [(409, "run_busy"), (400, "invalid_request"), (404, "run_not_found")])
def test_backend_passes_the_harness_answer_on_instead_of_503(backend, status, code):
    port, fake, b, tmp = backend
    refreshed = []
    b.refresh = lambda: refreshed.append(1)
    fake.stream = _harness_answers(status, code)
    got, text = call(port, "/api/ask", {"session_id": "eeeeeeee-5555", "message": "earbuds"})
    assert got == status and json.loads(text)["error"]["code"] == code
    assert refreshed == []                                       # the service answered: no re-read, no retry
    rows = [json.loads(l) for l in (tmp / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["status"] == "rejected"


@pytest.mark.parametrize("status", [500, 503, 401])
def test_backend_reports_503_only_when_the_service_itself_failed(backend, status):
    port, fake, b, tmp = backend
    b.refresh = lambda: None
    fake.stream = _harness_answers(status, "whatever")
    got, text = call(port, "/api/ask", {"session_id": "ffffffff-6666", "message": "earbuds"})
    assert got == 503 and json.loads(text)["error"]["code"] == "harness_unavailable"


def test_backend_maps_errors_of_cancel_and_lookup_too(backend):
    port, fake, b, tmp = backend
    status, text = call(port, "/api/ask", {"session_id": "abababab-7777", "message": "earbuds"})
    run_id = json.loads(text.split("event: meta\ndata: ")[1].split("\n")[0])["run_id"]
    from app.backend import HarnessError

    def busy(path, payload):
        raise HarnessError("harness HTTP 409: busy", status=409, code="run_busy")
    fake.post = busy
    assert call(port, "/api/cancel", {"session_id": "abababab-7777", "run_id": run_id})[0] == 409

    def refused(path, payload):
        raise ConnectionRefusedError("tunnel down")
    fake.post = refused
    assert call(port, "/api/lookup", {"session_id": "abababab-7777", "run_id": run_id, "offer_ids": []})[0] == 503


def test_backend_retries_a_new_run_under_a_new_run_id(backend):
    """The first POST may have started the run before the stream broke; resending its run_id gets run_exists."""
    port, fake, b, tmp = backend
    from app.backend import HarnessError
    sent, real_stream = [], fake.stream

    def broke_once(path, payload):
        sent.append(payload["run_id"])
        if len(sent) == 1:
            raise HarnessError("stream ended before the first event")
        yield from real_stream(path, payload)
    fake.stream = broke_once
    b.refresh = lambda: None
    status, text = call(port, "/api/ask", {"session_id": "cdcdcdcd-8888", "message": "earbuds"})
    assert status == 200 and len(sent) == 2 and sent[0] != sent[1]
    meta = json.loads(text.split("event: meta\ndata: ")[1].split("\n")[0])
    assert meta["run_id"] == sent[1] and b.owned("cdcdcdcd-8888", sent[1])
    rows = [json.loads(l) for l in (tmp / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["run_id"], r["status"]) for r in rows] == [(sent[0], "retried"), (sent[1], "completed")]


def test_service_tracks_runs_that_wait_for_an_answer():
    from types import SimpleNamespace
    from app.harness_service import HarnessService

    class Writer:
        def __init__(self):
            self.data = b""

        def write(self, chunk):
            self.data += chunk

        async def drain(self):
            pass

    def events(*statuses):
        async def make(cancel):
            for status in statuses:
                yield SimpleNamespace(result=SimpleNamespace(status=status) if status else None,
                                      model_dump=lambda mode=None: {})
        return make

    service = HarnessService.__new__(HarnessService)
    service.cancels, service.waiting, service.info, service.counts = {}, {}, {}, {"errors": 0}

    async def scenario():
        await service.stream(Writer(), "r1", events(None, "waiting_user"))
        await service.stream(Writer(), "r2", events("waiting_user"))
        assert set(service.waiting) == {"r1", "r2"} and "2 waiting for an answer" in service.pending()
        await service.stream(Writer(), "r1", events(None, "completed"))      # answered and finished
        await service.stream(Writer(), "r2", events())                       # a failed resume: still waiting
        assert set(service.waiting) == {"r2"} and service.pending().endswith(": r2")
    asyncio.run(scenario())


def test_lookup_adds_a_product_photo_and_the_shop_link_for_the_page(tmp_path):
    """The page shows the Google Shopping thumbnail that a variant's own evidence points at, and the offer's shop."""
    import sqlite3
    import types
    from app.harness_service import HarnessService
    folder = tmp_path / "snapshots" / "snap"
    (folder / "raw").mkdir(parents=True)
    (folder / "raw" / "r1.json").write_text(json.dumps({"shopping_results": [
        {"title": "A", "thumbnail": "https://img.example/a.webp", "extensions": ["x"]},
        {"title": "B", "thumbnail": "javascript:alert(1)"}]}), encoding="utf-8")
    con = sqlite3.connect(folder / "catalog.sqlite")
    con.executescript("""
        CREATE TABLE raw_records (raw_id TEXT, archive_path TEXT);
        CREATE TABLE evidence (id INTEGER PRIMARY KEY, variant_id TEXT, raw_id TEXT, json_pointer TEXT);
        CREATE TABLE offers (offer_id TEXT, merchant TEXT, url TEXT);
        INSERT INTO raw_records VALUES ('r1', 'raw/r1.json'), ('r2', '../outside.json');
        INSERT INTO evidence (variant_id, raw_id, json_pointer) VALUES
            ('v-a', 'r1', '/shopping_results/0/extensions/0'), ('v-b', 'r1', '/shopping_results/1'),
            ('v-c', 'r2', '/x'), ('v-d', 'r1', '/missing/5');
        INSERT INTO offers VALUES ('o-a', 'Walmart', 'https://walmart.example/a'), ('o-b', 'Shop', 'ftp://bad');
    """)
    con.commit()
    con.close()
    service = types.SimpleNamespace(kb=types.SimpleNamespace(snapshots_root=tmp_path / "snapshots"), snapshot_id="snap", photos={})
    photos, shops = HarnessService.presentation(service, {"v-a", "v-b", "v-c", "v-d", "v-none"}, {"o-a", "o-b"})
    # Found from a pointer inside the result item; non-web URLs, paths outside the snapshot and bad pointers give none.
    assert photos == {"v-a": "https://img.example/a.webp", "v-b": None, "v-c": None, "v-d": None, "v-none": None}
    assert shops == {"o-a": {"merchant": "Walmart", "url": "https://walmart.example/a"}, "o-b": {"merchant": "Shop", "url": None}}

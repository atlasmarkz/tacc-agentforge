"""Issue #25: cloud host pieces. Model tunnel that follows the vLLM address, snapshot download with the pinned
hash, run ownership that survives a backend restart, limits, Cloudflare Access tokens, and a waiting run that
resumes after the harness service restarts (SQLite checkpoints, issue #8)."""
import asyncio
import hashlib
import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "assets/knowledge/pilot-snapshot.zip"


# ---- model tunnel ---------------------------------------------------------------------------------------------

class FakeProc:
    def __init__(self, port):
        self.port, self.alive = port, True

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.alive = False


class FakeCluster:
    """Published address (or None, or an Exception) and the local ports a fake ssh has opened."""

    def __init__(self):
        self.address, self.procs, self.now = "http://10.0.3.10:8000/v1", [], 0.0

    def read(self):
        if isinstance(self.address, Exception):
            raise self.address
        return self.address

    def open(self, port, host, remote_port):
        proc = FakeProc(port)
        proc.target = (host, remote_port)
        self.procs.append(proc)
        return proc

    def is_open(self, port):
        return any(p.alive and p.port == port for p in self.procs)


@pytest.fixture()
def tunnel(tmp_path):
    from app.model_tunnel import ModelTunnel
    cluster = FakeCluster()

    def sleep(seconds):                                        # waiting moves the fake clock
        cluster.now += seconds

    t = ModelTunnel(tmp_path / "service-url.txt", cluster.read, cluster.open, is_open=cluster.is_open,
                    grace_s=60, clock=lambda: cluster.now, sleep=sleep)
    return t, cluster, tmp_path / "service-url.txt"


def test_tunnel_publishes_a_local_address_and_keeps_it(tunnel):
    t, cluster, local = tunnel
    assert t.step() == "opened"
    assert cluster.procs[0].target == ("10.0.3.10", 8000)
    assert local.read_text().strip() == f"http://127.0.0.1:{t.local_port}/v1"
    assert t.step() == "kept" and len(cluster.procs) == 1


def test_tunnel_follows_a_handover_and_lets_the_old_tunnel_drain(tunnel):
    t, cluster, local = tunnel
    t.step()
    old = cluster.procs[0]
    cluster.address = "http://10.0.3.11:8003/v1"              # successor on another node and port
    assert t.step() == "opened"
    assert cluster.procs[1].target == ("10.0.3.11", 8003)
    assert local.read_text().strip() == f"http://127.0.0.1:{cluster.procs[1].port}/v1"
    assert old.alive                                           # streams on the old vLLM may still be running
    cluster.now = 61
    t.step()
    assert not old.alive and cluster.procs[1].alive


def test_tunnel_reopens_a_dead_tunnel_to_the_same_address(tunnel):
    t, cluster, local = tunnel
    t.step()
    cluster.procs[0].alive = False
    assert t.step() == "opened" and len(cluster.procs) == 2
    assert local.read_text().strip().endswith(f":{cluster.procs[1].port}/v1")


def test_tunnel_withdraws_when_nothing_is_published_but_not_when_the_gateway_is_down(tunnel):
    from app.model_tunnel import Unreachable
    t, cluster, local = tunnel
    t.step()
    cluster.address = Unreachable("ssh exit 255")
    assert t.step() == "unreachable" and local.exists() and cluster.procs[0].alive
    cluster.address = None
    assert t.step() == "no_service" and not local.exists()
    cluster.address = "http://10.0.3.10:8000/v1"
    assert t.step() == "opened" and local.exists()


def test_tunnel_withdraws_when_the_forward_cannot_open(tunnel):
    t, cluster, local = tunnel
    t.step()
    cluster.address = "http://10.0.4.10:8000/v1"
    cluster.open = lambda port, host, rp: FakeProc(-1)         # never listens on `port`
    t.open_forward = cluster.open
    assert t.step() == "failed" and not local.exists()


def test_tunnel_close_withdraws_and_stops_every_ssh(tunnel):
    t, cluster, local = tunnel
    t.step()
    cluster.address = "http://10.0.3.11:8000/v1"
    t.step()
    t.close()
    assert not local.exists() and not any(p.alive for p in cluster.procs)


# ---- snapshot download -----------------------------------------------------------------------------------------

def test_fetch_archive_checks_the_pinned_hash(tmp_path):
    from knowledge.remote import fetch_archive
    src = tmp_path / "a.zip"
    src.write_bytes(b"not really a zip")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    got = fetch_archive(src.as_uri(), tmp_path / "dl", digest)
    assert got.read_bytes() == src.read_bytes()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetch_archive(src.as_uri(), tmp_path / "dl2", "0" * 64)
    assert not any((tmp_path / "dl2").iterdir())               # nothing usable left behind
    with pytest.raises(ValueError, match="gs://"):
        fetch_archive("ftp://host/a.zip", tmp_path / "dl3", digest)


def test_bootstrap_installs_from_the_seed_url_when_the_repo_has_no_archive(tmp_path, monkeypatch):
    from knowledge.bootstrap import install_snapshot
    from knowledge.settings import load_settings
    config = json.loads((ROOT / "configs/knowledge.json").read_text())
    config["seed"]["archive"] = "assets/knowledge/missing.zip"
    config["seed"]["url"] = SEED.as_uri()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/knowledge.json").write_text(json.dumps(config))
    settings = load_settings(tmp_path / "configs/knowledge.json", data_root=tmp_path / "data")
    assert settings.seed_url == SEED.as_uri()
    result = install_snapshot(settings)
    assert result["status"] == "installed" and result["verification"]["valid"]
    assert result["snapshot_id"] == config["snapshot_id"]
    monkeypatch.setenv("KNOWLEDGE_SEED_URL", "http://insecure/a.zip")
    with pytest.raises(ValueError, match="seed.url"):
        load_settings(tmp_path / "configs/knowledge.json")


# ---- backend: ownership, limits, Access -------------------------------------------------------------------------

class SlowHarness:
    """Streams one run; holds the stream open until `release` is set, so tests can see concurrent slots."""

    def __init__(self):
        self.release = threading.Event()
        self.release.set()

    def get(self, path):
        return {"status": "ok", "job_id": "cloud", "snapshot_id": "snap"}

    def post(self, path, payload):
        return {"cancelled": True}

    def stream(self, path, payload):
        yield "agent", {"run_id": payload.get("run_id", "x"), "sequence": 1, "type": "run_waiting", "data": {},
                        "result": {"status": "waiting_user", "snapshot_id": "snap"}}
        self.release.wait(10)
        yield "end", {"run_id": "x", "wall_s": 0.1, "job_id": "cloud"}


def start_backend(backend, auth=None):
    from app.backend import make_handler
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(backend, auth))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def call(port, path, body=None, headers=None, method="POST"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    raw = resp.read().decode()
    conn.close()
    if resp.getheader("Content-Type", "").startswith("text/event-stream"):
        events = []
        for chunk in raw.split("\n\n"):
            lines = dict(line.split(": ", 1) for line in chunk.splitlines() if ": " in line)
            if "event" in lines:
                events.append((lines["event"], json.loads(lines["data"])))
        return resp.status, events
    return resp.status, json.loads(raw)


SESSION = "abcdabcd-1111"


def test_run_ownership_survives_a_backend_restart(tmp_path):
    from app.backend import Backend
    store = tmp_path / "runs.sqlite"
    first = Backend(SlowHarness(), tmp_path / "log.jsonl", store_path=store)
    run_id = first.new_run("ann@example.com", "request", session=SESSION)
    first.db.close()
    second = Backend(SlowHarness(), tmp_path / "log.jsonl", store_path=store)
    assert second.owned("ann@example.com", run_id)
    assert not second.owned("bob@example.com", run_id) and not second.owned("ann@example.com", None)


def test_limits_per_owner_and_in_total(tmp_path):
    from app.backend import Backend, Limits
    b = Backend(SlowHarness(), tmp_path / "log.jsonl", limits=Limits(per_owner_active=1, total_active=2,
                                                                       per_owner_per_minute=2))
    assert b.acquire("a", True) is None
    assert b.acquire("a", False) == "rate_limited"             # one stream at a time for this owner
    assert b.acquire("b", True) is None
    assert b.acquire("c", True) == "busy"                      # total reached
    b.release("a")
    b.release("b")
    assert b.acquire("a", True) is None and (b.release("a") or True)
    assert b.acquire("a", True) == "rate_limited"              # third new run within a minute
    assert b.acquire("a", False) is None                       # answering is not a new run


def test_backend_answers_429_when_the_owner_is_already_streaming(tmp_path):
    from app.backend import Backend, Limits
    harness = SlowHarness()
    harness.release.clear()
    b = Backend(harness, tmp_path / "log.jsonl", limits=Limits(per_owner_active=1))
    server = start_backend(b)
    port = server.server_address[1]
    try:
        first = threading.Thread(target=call, args=(port, "/api/ask", {"session_id": SESSION, "message": "hi"}))
        first.start()
        for _ in range(100):
            if b.active:
                break
            time.sleep(0.02)
        status, body = call(port, "/api/ask", {"session_id": SESSION, "message": "again"})
        assert status == 429 and body["error"]["code"] == "rate_limited"
        harness.release.set()
        first.join(10)
        for _ in range(100):                                    # the handler releases after the response closes
            if not b.active:
                break
            time.sleep(0.02)
        assert b.active == {}
        status, events = call(port, "/api/ask", {"session_id": SESSION, "message": "now"})
        assert status == 200 and events[0][0] == "meta"
    finally:
        harness.release.set()
        server.shutdown()


@pytest.fixture()
def access():
    jwt = pytest.importorskip("jwt")
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    from app.access import AccessAuth
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = "https://team.cloudflareaccess.com"
    auth = AccessAuth("team.cloudflareaccess.com", "user-aud", api_aud="api-aud", key_for_token=lambda t: key.public_key())

    def token(aud, email="Ann@Example.com", signer=key, exp=300, iss=issuer):
        now = int(time.time())
        claims = {"aud": [aud], "iss": iss, "iat": now, "exp": now + exp, "email": email}
        return jwt.encode(claims, signer, algorithm="RS256")

    auth.free = AccessAuth("team.cloudflareaccess.com", "user-aud", proxy_secret="s" * 40,
                           key_for_token=lambda t: key.public_key())
    return auth, token, other


def test_free_setup_needs_the_proxy_secret_and_a_user_token(access):
    from app.access import AccessAuth, AuthError, EDGE_HEADER, SECRET_HEADER, USER_HEADER
    auth, token, _ = access
    free = auth.free
    assert free.identify({SECRET_HEADER: "s" * 40, USER_HEADER: token("user-aud")}) == "ann@example.com"
    for headers in ({USER_HEADER: token("user-aud")},                                    # not from the Function
                    {SECRET_HEADER: "s" * 39 + "x", USER_HEADER: token("user-aud")},       # wrong secret
                    {EDGE_HEADER: token("api-aud"), USER_HEADER: token("user-aud")},       # edge token is not it
                    {SECRET_HEADER: "s" * 40}):                                            # no signed-in user
        with pytest.raises(AuthError):
            free.identify(headers)
    with pytest.raises(ValueError):
        AccessAuth("t.cloudflareaccess.com", "u", proxy_secret="short")
    with pytest.raises(ValueError):
        AccessAuth("t.cloudflareaccess.com", "u", api_aud="a", proxy_secret="s" * 40)


def test_access_needs_both_tokens(access):
    from app.access import AuthError, EDGE_HEADER, USER_HEADER
    auth, token, other = access
    good = {EDGE_HEADER: token("api-aud"), USER_HEADER: token("user-aud")}
    assert auth.identify(good) == "ann@example.com"
    bad = [
        {USER_HEADER: token("user-aud")},                                    # did not come through Access
        {EDGE_HEADER: token("api-aud")},                                     # no signed-in user
        {EDGE_HEADER: token("api-aud"), USER_HEADER: token("api-aud")},      # user token for the wrong app
        {EDGE_HEADER: token("user-aud"), USER_HEADER: token("user-aud")},    # edge token for the wrong app
        {EDGE_HEADER: token("api-aud"), USER_HEADER: token("user-aud", exp=-10)},           # expired
        {EDGE_HEADER: token("api-aud"), USER_HEADER: token("user-aud", signer=other)},      # forged
        {EDGE_HEADER: token("api-aud"), USER_HEADER: token("user-aud", iss="https://x.cloudflareaccess.com")},
        {EDGE_HEADER: token("api-aud"), USER_HEADER: token("user-aud", email=None)},
    ]
    for headers in bad:
        with pytest.raises(AuthError):
            auth.identify(headers)


def test_backend_with_access_keys_runs_by_user_not_by_session(access, tmp_path):
    from app.access import EDGE_HEADER, USER_HEADER
    from app.backend import Backend
    auth, token, _ = access
    b = Backend(SlowHarness(), tmp_path / "log.jsonl")
    server = start_backend(b, auth)
    port = server.server_address[1]
    ann = {EDGE_HEADER: token("api-aud"), USER_HEADER: token("user-aud", email="ann@example.com")}
    bob = {EDGE_HEADER: token("api-aud"), USER_HEADER: token("user-aud", email="bob@example.com")}
    try:
        status, body = call(port, "/api/health", headers={}, method="GET")
        assert status == 401 and body["error"]["code"] == "unauthorized"
        assert call(port, "/api/ask", {"session_id": SESSION, "message": "hi"})[0] == 401
        assert call(port, "/api/health", headers=ann, method="GET")[0] == 200
        status, events = call(port, "/api/ask", {"session_id": SESSION, "message": "hi"}, ann)
        run_id = events[0][1]["run_id"]
        # Ann in another tab (another session ID) still owns her run; Bob with Ann's session ID does not.
        assert call(port, "/api/cancel", {"session_id": "ffffffff-2222", "run_id": run_id}, ann)[0] == 200
        status, body = call(port, "/api/cancel", {"session_id": SESSION, "run_id": run_id}, bob)
        assert status == 403 and body["error"]["code"] == "not_your_run"
    finally:
        server.shutdown()


def test_backend_refuses_a_public_bind_without_access(monkeypatch):
    import sys
    from app import backend
    monkeypatch.setattr(sys, "argv", ["backend", "--bind", "0.0.0.0", "--harness-url", "http://127.0.0.1:1"])
    monkeypatch.delenv("CF_ACCESS_TEAM_DOMAIN", raising=False)
    with pytest.raises(SystemExit, match="refusing"):
        backend.main()


# ---- harness: a waiting run resumes after a restart (issue #8) ---------------------------------------------------

def test_a_waiting_run_resumes_after_the_harness_service_restarts(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from agent import RunPolicy
    from agent.checkpoints import open_checkpointer
    from app.backend import HarnessClient
    from app.harness_service import HarnessService, prepare_kb, serve
    from app.selftest import ScriptedPort

    snapshot_id, _, kb, _ = prepare_kb(ROOT / "configs/knowledge.json", tmp_path / "user", tmp_path)
    db = tmp_path / "checkpoints/agent.sqlite"

    async def lifetime(work):
        """One harness process: open the store, serve, run `work(client)` in a thread, stop."""
        async with open_checkpointer(db) as saver:
            service = HarnessService(model=ScriptedPort(), kb=kb, snapshot_id=snapshot_id,
                                     policy=RunPolicy(max_context_chars=36000), token="t", info={"job_id": "cloud"},
                                     explain_default=False, saver=saver)
            stop, ready = asyncio.Event(), asyncio.get_running_loop().create_future()
            server = asyncio.create_task(serve(service, "127.0.0.1", 0, ready=ready.set_result, stop=stop))
            port = await ready
            try:
                return await asyncio.to_thread(work, HarnessClient(lambda: (f"http://127.0.0.1:{port}", "t")))
            finally:
                stop.set()
                await server
                service.close()

    def ask(client):
        assert client.get("/health")["checkpoints"] == "sqlite"
        events = list(client.stream("/runs", {"run_id": "cloud-1", "conversation_id": "c1",
                                               "message": "推荐通勤用的无线耳机", "constraints": {}}))
        results = [e[1]["result"] for e in events if e[0] == "agent" and e[1].get("result")]
        assert results[-1]["status"] == "waiting_user"
        return results[-1]["pending_question"]["question_id"]

    def answer(question_id):
        def work(client):
            events = list(client.stream("/runs/cloud-1/resume", {
                "question_id": question_id, "answer": {"purchase_cap": "100.00", "currency": "USD"}}))
            return [e[1]["result"] for e in events if e[0] == "agent" and e[1].get("result")]
        return work

    question_id = asyncio.run(lifetime(ask))                   # first process stops while the run waits
    results = asyncio.run(lifetime(answer(question_id)))       # a new process resumes it from disk
    assert results and results[-1]["status"] == "completed" and results[-1]["verified_plan"]

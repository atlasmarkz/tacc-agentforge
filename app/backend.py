"""Local backend (Task 5): serves the browser page and forwards its requests to the harness service on TACC.

  browser --HTTP--> this backend (127.0.0.1) --ssh -L via the TACC gateway--> harness service (compute node)
                                                                        --> vLLM (inside the cluster)

- The browser only ever talks to this process. The service token, the SSH key and the cluster addresses stay
  here; no API key is involved at query time (SerpApi is only used when the snapshot is collected).
- The harness publishes its address and token in USERDIR (app/harness_service.py). The backend reads both
  over SSH and opens a tunnel; when the service moves (new job, handover), the next request re-reads them.
- Request IDs: the browser keeps a random session ID; this backend creates every run ID (run = one request or
  revision) and only lets a session answer, cancel or inspect its own runs. Each run is logged to
  app/logs/requests.jsonl with session, run, snapshot, harness job and final status.

  python -m app.backend                      # then open http://127.0.0.1:8765
  python -m app.backend --harness-url http://127.0.0.1:8100 --token ...   # no SSH (local service)

Environment: TACC_SSH (markzhang@gw.turing.ust.hk), TACC_SSH_PORT (30041), TACC_SSH_KEY (~/.ssh/id_rsa_tacc),
TACC_REMOTE_USERDIR (/mnt/home/markzhang/USERDIR).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import secrets
import socket
import sqlite3
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
SESSION_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
MAX_BODY = 64 * 1024


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class HarnessError(Exception):
    """`status` and `code` are what the harness answered; both are None when it could not be reached."""

    def __init__(self, message, status=None, code=None):
        super().__init__(message)
        self.status, self.code = status, code

    @property
    def unavailable(self) -> bool:
        # Not reached, a server error, or a token that no longer matches (the service moved): the service is the
        # problem. Any other answer (400 invalid_request, 404, 409 run_busy, ...) is about this request.
        return self.status is None or self.status >= 500 or self.status == 401


class HarnessClient:
    """Minimal HTTP client for the harness service. `resolve()` returns (base_url, token)."""

    def __init__(self, resolve, timeout: float = 15):
        self.resolve, self.timeout = resolve, timeout

    def _open(self, method, path, payload=None, timeout=None):
        base, token = self.resolve()
        u = urlsplit(base)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout or self.timeout)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Authorization": f"Bearer {token}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            raw = resp.read(4000)
            conn.close()
            code, message = None, raw[:500].decode("utf-8", "replace")
            try:                                     # the harness answers {"error": {"code", "message"}}
                error = json.loads(raw)["error"]
                code, message = error.get("code"), error.get("message") or message
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
            raise HarnessError(f"harness HTTP {resp.status}: {message}", status=resp.status, code=code)
        return conn, resp

    def get(self, path):
        conn, resp = self._open("GET", path)
        try:
            return json.loads(resp.read())
        finally:
            conn.close()

    def post(self, path, payload):
        conn, resp = self._open("POST", path, payload)
        try:
            return json.loads(resp.read())
        finally:
            conn.close()

    def stream(self, path, payload, idle_timeout: float = 60):
        """Yields (event, data) from the service's SSE stream; ping comments are skipped. The service pings
        every 10 s, so `idle_timeout` without any byte means the tunnel or the service is gone."""
        conn, resp = self._open("POST", path, payload, timeout=idle_timeout)
        try:
            event, data = None, []
            while True:
                line = resp.readline()
                if not line:
                    return
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif line == "" and event:
                    yield event, json.loads("\n".join(data) or "null")
                    if event == "end":
                        return
                    event, data = None, []
        finally:
            conn.close()


class Tunnel:
    """Reads the harness address and token from USERDIR over SSH and keeps an `ssh -L` tunnel to it."""

    def __init__(self, ssh_target, ssh_port, key, remote_userdir):
        self.ssh = ["ssh", "-p", str(ssh_port), "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
        self.target, self.remote = ssh_target, remote_userdir.rstrip("/") + "/tacc-agentforge-service"
        self.local_port, self.proc, self.remote_addr, self.token = None, None, None, None
        self.lock = threading.Lock()

    @staticmethod
    def _free_port():
        # A fresh port for every tunnel. With a fixed port, an ssh left over from an earlier backend (killed
        # without cleanup) kept the port forwarded to a stopped harness; the new tunnel failed to bind and
        # exited, but the port still answered, so every request went to the dead forward (job 28963 soak).
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _read_remote(self):
        cmd = self.ssh + [self.target, f"cat {self.remote}/harness-url.txt {self.remote}/harness-token.txt"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        if out.returncode != 0 or len(lines) != 2:
            raise HarnessError("The harness service is not running (no published address). Start the "
                               "service job first (qwen-service staged with: stage.sh serve --harness).")
        return lines[0], lines[1]

    def _alive(self):
        if self.proc is None or self.proc.poll() is not None:
            return False
        try:
            socket.create_connection(("127.0.0.1", self.local_port), timeout=2).close()
            return True
        except OSError:
            return False

    def resolve(self, refresh=False):
        with self.lock:
            if refresh or not self._alive() or self.remote_addr is None:
                addr, token = self._read_remote()
                if addr != self.remote_addr or not self._alive():
                    self.close()
                    u = urlsplit(addr)
                    self.local_port = self._free_port()
                    self.proc = subprocess.Popen(
                        self.ssh + ["-N", "-o", "ExitOnForwardFailure=yes",
                                    "-L", f"127.0.0.1:{self.local_port}:{u.hostname}:{u.port}", self.target],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for _ in range(60):
                        if self._alive():
                            break
                        if self.proc.poll() is not None:
                            raise HarnessError("Could not open the SSH tunnel to the harness service.")
                        time.sleep(0.25)
                    else:
                        raise HarnessError("Timed out opening the SSH tunnel.")
                    log(f"tunnel 127.0.0.1:{self.local_port} -> {addr}")
                self.remote_addr, self.token = addr, token
            return f"http://127.0.0.1:{self.local_port}", self.token

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


@dataclass(frozen=True)
class Limits:
    """Per-owner and total limits on streaming calls (asks and answers). The vLLM service serves at most 8
    sequences and about 6 full recommendations a minute comfortably (RUNBOOK), so a public page needs these."""
    per_owner_active: int = 2
    total_active: int = 8
    per_owner_per_minute: int = 6


class Backend:
    """Run ownership lives in SQLite (`store_path`), so after a restart a user can still answer a question their
    run is waiting on (the cloud harness keeps the checkpoint, issue #8). The owner is the signed-in user when
    Cloudflare Access is on, otherwise the page's session ID (local, single user)."""

    def __init__(self, client: HarnessClient, log_path: Path, refresh=None, *, store_path=":memory:",
                 limits: Limits | None = None):
        self.client, self.refresh = client, refresh or (lambda: None)
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if store_path != ":memory:":
            Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(store_path), check_same_thread=False, isolation_level=None)
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                        "conversation TEXT NOT NULL, kind TEXT NOT NULL, parent TEXT, created REAL NOT NULL)")
        self.lock = threading.Lock()
        self.limits = limits or Limits()
        self.active: dict[str, int] = {}
        self.starts: dict[str, deque] = {}

    def new_run(self, owner: str, kind: str, parent: str | None = None, *, session: str | None = None) -> str:
        session = session or owner
        run_id = f"r-{session[:8]}-{int(time.time())}-{secrets.token_hex(3)}"
        with self.lock:
            self.db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                            (run_id, owner, f"s-{session}", kind, parent, time.time()))
        return run_id

    def owned(self, owner: str, run_id) -> bool:
        if not isinstance(run_id, str):
            return False
        with self.lock:
            row = self.db.execute("SELECT owner FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row is not None and row[0] == owner

    def acquire(self, owner: str, new_run: bool) -> str | None:
        """Reserve a streaming slot; returns an error code when a limit is hit. `release` frees it."""
        now = time.monotonic()
        with self.lock:
            if sum(self.active.values()) >= self.limits.total_active:
                return "busy"
            if self.active.get(owner, 0) >= self.limits.per_owner_active:
                return "rate_limited"
            starts = self.starts.setdefault(owner, deque())
            while starts and now - starts[0] > 60:
                starts.popleft()
            if new_run:
                if len(starts) >= self.limits.per_owner_per_minute:
                    return "rate_limited"
                starts.append(now)
            self.active[owner] = self.active.get(owner, 0) + 1
        return None

    def release(self, owner: str) -> None:
        with self.lock:
            left = self.active.get(owner, 0) - 1
            if left > 0:
                self.active[owner] = left
            else:
                self.active.pop(owner, None)

    def record(self, **row):
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def health(self):
        try:
            return {"backend": "ok", "harness": self.client.get("/health")}
        except Exception as exc:
            return {"backend": "ok", "harness": None, "error": str(exc)}


def make_handler(backend: Backend, auth=None):
    """`auth`: an app.access.AccessAuth for the public deployment (every /api call needs a signed-in user), or
    None locally."""
    page = (HERE / "static" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def identify(self):
            """The signed-in user (Cloudflare Access) or None locally; sends 401 and returns False when rejected."""
            if auth is None:
                return None
            try:
                return auth.identify(self.headers)
            except Exception as exc:                          # AuthError, or the key fetch failed
                log(f"rejected {self.path}: {exc}")
                self._json(401, {"error": {"code": "unauthorized", "message": "Please sign in again."}})
                return False

        def do_GET(self):
            if self.path == "/api/health":
                if self.identify() is False:
                    return
                return self._json(200, backend.health())
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
            else:
                self._json(404, {"error": {"code": "not_found", "message": "No such page."}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json(413, {"error": {"code": "too_large", "message": "Request too large."}})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                session = body.get("session_id", "")
                if not isinstance(session, str) or not SESSION_RE.match(session):
                    raise ValueError("session_id")
            except ValueError:
                return self._json(400, {"error": {"code": "bad_request", "message": "Invalid request."}})
            user = self.identify()
            if user is False:
                return
            # Runs belong to the signed-in user when Access is on; otherwise to the page's session. Limits count
            # per visitor IP in public mode (forwarded by the Pages Function), else per owner.
            owner = user or session
            limit = getattr(auth, "limit_key", lambda headers: None)(self.headers) or owner
            try:
                if self.path == "/api/ask":
                    return self.ask(session, owner, body, limit)
                if self.path == "/api/answer":
                    return self.answer(session, owner, body, limit)
                if self.path == "/api/cancel":
                    return self.cancel(owner, body)
                if self.path == "/api/lookup":
                    return self.lookup(owner, body)
            except (HarnessError, OSError) as exc:
                return self.harness_error(exc)
            self._json(404, {"error": {"code": "not_found", "message": "No such endpoint."}})

        def harness_error(self, exc):
            """503 only when the service is unreachable or failed; otherwise pass the harness's own answer on
            (409 run_busy, 400 invalid_request, 404 run_not_found, ...) so the page shows what really happened."""
            if isinstance(exc, HarnessError) and not exc.unavailable:
                return self._json(exc.status, {"error": {"code": exc.code or "harness_error",
                                                         "message": str(exc)[:300]}})
            return self._json(503, {"error": {"code": "harness_unavailable", "message": str(exc)[:300]}})

        # ---- endpoints ----

        def ask(self, session, owner, body, limit):
            message, constraints = body.get("message"), body.get("constraints") or {}
            if not isinstance(message, str) or not message.strip() or not isinstance(constraints, dict):
                return self._json(400, {"error": {"code": "bad_request", "message": "message is required."}})
            parent = body.get("revises")
            if parent is not None and not backend.owned(owner, parent):
                return self._json(403, {"error": {"code": "not_your_run", "message": "Unknown run."}})
            if not self.reserve(limit, new_run=True):
                return
            try:
                kind = "revision" if parent else "request"
                run_id = backend.new_run(owner, kind, parent, session=session)
                payload = {"run_id": run_id, "conversation_id": f"s-{session}", "message": message,
                           "constraints": constraints, "explain": body.get("explain", True)}
                self.relay(session, owner, run_id, "/runs", payload, kind=kind, parent=parent)
            finally:
                backend.release(limit)

        def answer(self, session, owner, body, limit):
            run_id = body.get("run_id")
            if not backend.owned(owner, run_id):
                return self._json(403, {"error": {"code": "not_your_run", "message": "Unknown run."}})
            if not self.reserve(limit, new_run=False):
                return
            try:
                self.relay(session, owner, run_id, f"/runs/{run_id}/resume",
                           {"question_id": body.get("question_id"), "answer": body.get("answer") or {}},
                           kind="answer")
            finally:
                backend.release(limit)

        def reserve(self, owner, new_run):
            code = backend.acquire(owner, new_run)
            if code is None:
                return True
            message = ("The service is busy with other requests. Please try again in a minute." if code == "busy"
                       else "Too many requests at once. Please wait for the current one or try again in a minute.")
            self._json(429, {"error": {"code": code, "message": message}})
            return False

        def cancel(self, owner, body):
            run_id = body.get("run_id")
            if not backend.owned(owner, run_id):
                return self._json(403, {"error": {"code": "not_your_run", "message": "Unknown run."}})
            self._json(200, backend.client.post(f"/runs/{run_id}/cancel", {}))

        def lookup(self, owner, body):
            run_id = body.get("run_id")
            if not backend.owned(owner, run_id):
                return self._json(403, {"error": {"code": "not_your_run", "message": "Unknown run."}})
            self._json(200, backend.client.post("/lookup", {"offer_ids": body.get("offer_ids") or [],
                                                            "evidence_ids": body.get("evidence_ids") or []}))

        def relay(self, session, owner, run_id, path, payload, kind, parent=None):
            """Pass the harness's SSE stream through, one event per chunk, and log the outcome."""
            t0 = time.monotonic()
            exc = None
            for attempt in range(2):
                try:
                    events = backend.client.stream(path, payload)
                    first = next(events)                     # fail before sending 200 if the harness is down
                    exc = None
                    break
                except (HarnessError, OSError, StopIteration) as e:
                    exc = e
                    if isinstance(e, HarnessError) and not e.unavailable:
                        break                                # the harness answered; a retry gets the same answer
                    # The service may have moved (new job or handover) while the old tunnel still accepts
                    # locally: re-read the published address and try once more before giving up.
                    backend.refresh()
                    if path == "/runs" and attempt == 0:
                        # The first POST may already have started the run before the stream broke; sending the
                        # same run_id again would only get run_exists. Retry as a new run; the abandoned one ends
                        # at its own deadline.
                        backend.record(session=session, run_id=run_id, kind=kind, parent=parent, status="retried",
                                       error=str(e)[:300])
                        run_id = backend.new_run(owner, kind, parent, session=session)
                        payload = {**payload, "run_id": run_id}
            if exc is not None:
                unreachable = not isinstance(exc, HarnessError) or exc.unavailable
                backend.record(session=session, run_id=run_id, kind=kind, parent=parent,
                               status="unreachable" if unreachable else "rejected", error=str(exc)[:300])
                return self.harness_error(exc)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            status, snapshot, error, first_event_s = None, None, None, round(time.monotonic() - t0, 2)
            harness_job = None
            meta = {"run_id": run_id, "kind": kind}
            try:
                self.wfile.write(f"event: meta\ndata: {json.dumps(meta)}\n\n".encode())
                for event, data in _chain(first, events):
                    if event == "agent" and data.get("result"):
                        status, snapshot = data["result"]["status"], data["result"]["snapshot_id"]
                    if event == "error":
                        error = data.get("code")
                    if event == "end":
                        harness_job = data.get("job_id")
                    self.wfile.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (ConnectionError, OSError) as exc:
                error = error or f"stream_broken: {type(exc).__name__}"
                try:                                         # the browser left: stop the run on the cluster
                    backend.client.post(f"/runs/{run_id}/cancel", {})
                except Exception:
                    pass
            backend.record(session=session, run_id=run_id, kind=kind, parent=parent, status=status,
                           snapshot_id=snapshot, harness_job=harness_job, error=error, first_event_s=first_event_s,
                           wall_s=round(time.monotonic() - t0, 2))

    return Handler


def _chain(first, rest):
    yield first
    yield from rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--harness-url", help="talk to this harness service directly instead of over SSH")
    ap.add_argument("--token", help="with --harness-url")
    ap.add_argument("--token-file", type=Path, help="with --harness-url: read the token from this file (cloud host)")
    ap.add_argument("--bind", default="127.0.0.1", help="listen address; keep 127.0.0.1 behind cloudflared")
    ap.add_argument("--log", type=Path, default=HERE / "logs" / "requests.jsonl")
    ap.add_argument("--runs-db", default=":memory:", help="SQLite file for run ownership (cloud host: keep it so "
                                                           "waiting runs can be answered after a restart)")
    ap.add_argument("--access-team-domain", default=os.environ.get("CF_ACCESS_TEAM_DOMAIN"),
                    help="Cloudflare Access team domain (<team>.cloudflareaccess.com); turns on sign-in checks")
    ap.add_argument("--access-user-aud", default=os.environ.get("CF_ACCESS_USER_AUD"),
                    help="audience tag of the Access application that protects the page")
    ap.add_argument("--access-api-aud", default=os.environ.get("CF_ACCESS_API_AUD") or None,
                    help="own domain: audience tag of the Access application on the API hostname")
    ap.add_argument("--proxy-secret-env", default="AGENTFORGE_PROXY_SECRET",
                    help="free setup: environment variable holding the secret the Pages Function sends")
    a = ap.parse_args()
    auth = None
    secret = os.environ.get(a.proxy_secret_env) or None
    if not a.access_team_domain and secret:
        # Public mode: anyone may use the page; only requests through the Pages Function are accepted.
        from app.access import ProxyAuth
        auth = ProxyAuth(secret)
    elif a.access_team_domain:
        if not a.access_user_aud or bool(a.access_api_aud) == bool(secret):
            raise SystemExit("Access sign-in needs CF_ACCESS_USER_AUD and exactly one of CF_ACCESS_API_AUD "
                             f"(own domain) or {a.proxy_secret_env} (free setup)")
        from app.access import AccessAuth
        auth = AccessAuth(a.access_team_domain, a.access_user_aud, api_aud=a.access_api_aud, proxy_secret=secret)
    elif a.bind not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing to listen beyond localhost without a proxy secret or Access sign-in checks")
    tunnel = None
    if a.harness_url:
        token = a.token_file.read_text(encoding="utf-8").strip() if a.token_file else (a.token or "")
        client = HarnessClient(lambda: (a.harness_url, token))
        refresh = None
    else:
        tunnel = Tunnel(os.environ.get("TACC_SSH", "markzhang@gw.turing.ust.hk"),
                        int(os.environ.get("TACC_SSH_PORT", "30041")),
                        Path(os.environ.get("TACC_SSH_KEY", Path.home() / ".ssh" / "id_rsa_tacc")),
                        os.environ.get("TACC_REMOTE_USERDIR", "/mnt/home/markzhang/USERDIR"))
        client = HarnessClient(tunnel.resolve)
        refresh = lambda: tunnel.resolve(refresh=True) if tunnel.remote_addr else None  # noqa: E731
    backend = Backend(client, a.log, refresh=lambda: _quiet(refresh), store_path=a.runs_db)
    server = ThreadingHTTPServer((a.bind, a.port), make_handler(backend, auth))
    mode = {"ProxyAuth": " (public: requests through the Pages proxy only)",
            "AccessAuth": " (Cloudflare Access sign-in required)"}.get(type(auth).__name__, "")
    log(f"open http://{a.bind}:{a.port}{mode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel:
            tunnel.close()


def _quiet(fn):
    try:
        if fn:
            fn()
    except Exception:
        pass


if __name__ == "__main__":
    main()

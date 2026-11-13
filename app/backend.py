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
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
SESSION_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
MAX_BODY = 64 * 1024


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class HarnessError(Exception):
    pass


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
            text = resp.read()[:500].decode("utf-8", "replace")
            conn.close()
            raise HarnessError(f"harness HTTP {resp.status}: {text}")
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

    def __init__(self, ssh_target, ssh_port, key, remote_userdir, local_port=18100):
        self.ssh = ["ssh", "-p", str(ssh_port), "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
        self.target, self.remote = ssh_target, remote_userdir.rstrip("/") + "/tacc-agentforge-service"
        self.local_port, self.proc, self.remote_addr, self.token = local_port, None, None, None
        self.lock = threading.Lock()

    def _read_remote(self):
        cmd = self.ssh + [self.target, f"cat {self.remote}/harness-url.txt {self.remote}/harness-token.txt"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        if out.returncode != 0 or len(lines) != 2:
            raise HarnessError("The harness service is not running (no published address). Start the "
                               "deploy/tacc/service/app-service job first.")
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


class Backend:
    def __init__(self, client: HarnessClient, log_path: Path, refresh=None):
        self.client, self.refresh = client, refresh or (lambda: None)
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, dict] = {}                     # run_id -> {session, conversation, kind, ...}
        self.lock = threading.Lock()

    def new_run(self, session: str, kind: str, parent: str | None = None) -> str:
        run_id = f"r-{session[:8]}-{int(time.time())}-{secrets.token_hex(3)}"
        with self.lock:
            self.runs[run_id] = {"session": session, "conversation": f"s-{session}", "kind": kind, "parent": parent,
                                 "created": time.time()}
        return run_id

    def owned(self, session: str, run_id: str) -> bool:
        with self.lock:
            return self.runs.get(run_id, {}).get("session") == session

    def record(self, **row):
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def health(self):
        try:
            return {"backend": "ok", "harness": self.client.get("/health")}
        except Exception as exc:
            return {"backend": "ok", "harness": None, "error": str(exc)}


def make_handler(backend: Backend):
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

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/api/health":
                self._json(200, backend.health())
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
            try:
                if self.path == "/api/ask":
                    return self.ask(session, body)
                if self.path == "/api/answer":
                    return self.answer(session, body)
                if self.path == "/api/cancel":
                    return self.cancel(session, body)
                if self.path == "/api/lookup":
                    return self.lookup(session, body)
            except HarnessError as exc:
                return self._json(503, {"error": {"code": "harness_unavailable", "message": str(exc)}})
            self._json(404, {"error": {"code": "not_found", "message": "No such endpoint."}})

        # ---- endpoints ----

        def ask(self, session, body):
            message, constraints = body.get("message"), body.get("constraints") or {}
            if not isinstance(message, str) or not message.strip() or not isinstance(constraints, dict):
                return self._json(400, {"error": {"code": "bad_request", "message": "message is required."}})
            parent = body.get("revises")
            if parent is not None and not backend.owned(session, parent):
                return self._json(403, {"error": {"code": "not_your_run", "message": "Unknown run."}})
            run_id = backend.new_run(session, "revision" if parent else "request", parent)
            payload = {"run_id": run_id, "conversation_id": f"s-{session}", "message": message,
                       "constraints": constraints, "explain": body.get("explain", True)}
            self.relay(session, run_id, "/runs", payload, kind="revision" if parent else "request", parent=parent)

        def answer(self, session, body):
            run_id = body.get("run_id")
            if not backend.owned(session, run_id):
                return self._json(403, {"error": {"code": "not_your_run", "message": "Unknown run."}})
            self.relay(session, run_id, f"/runs/{run_id}/resume",
                       {"question_id": body.get("question_id"), "answer": body.get("answer") or {}}, kind="answer")

        def cancel(self, session, body):
            run_id = body.get("run_id")
            if not backend.owned(session, run_id):
                return self._json(403, {"error": {"code": "not_your_run", "message": "Unknown run."}})
            self._json(200, backend.client.post(f"/runs/{run_id}/cancel", {}))

        def lookup(self, session, body):
            run_id = body.get("run_id")
            if not backend.owned(session, run_id):
                return self._json(403, {"error": {"code": "not_your_run", "message": "Unknown run."}})
            self._json(200, backend.client.post("/lookup", {"offer_ids": body.get("offer_ids") or [],
                                                            "evidence_ids": body.get("evidence_ids") or []}))

        def relay(self, session, run_id, path, payload, kind, parent=None):
            """Pass the harness's SSE stream through, one event per chunk, and log the outcome."""
            t0 = time.monotonic()
            try:
                events = backend.client.stream(path, payload)
                first = next(events)                         # fail before sending 200 if the harness is down
            except (HarnessError, OSError, StopIteration) as exc:
                backend.refresh()
                backend.record(session=session, run_id=run_id, kind=kind, parent=parent, status="unreachable",
                               error=str(exc)[:300])
                return self._json(503, {"error": {"code": "harness_unavailable", "message": str(exc)[:300]}})
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
    ap.add_argument("--log", type=Path, default=HERE / "logs" / "requests.jsonl")
    a = ap.parse_args()
    tunnel = None
    if a.harness_url:
        client = HarnessClient(lambda: (a.harness_url, a.token or ""))
        refresh = None
    else:
        tunnel = Tunnel(os.environ.get("TACC_SSH", "markzhang@gw.turing.ust.hk"),
                        int(os.environ.get("TACC_SSH_PORT", "30041")),
                        Path(os.environ.get("TACC_SSH_KEY", Path.home() / ".ssh" / "id_rsa_tacc")),
                        os.environ.get("TACC_REMOTE_USERDIR", "/mnt/home/markzhang/USERDIR"))
        client = HarnessClient(tunnel.resolve)
        refresh = lambda: tunnel.resolve(refresh=True) if tunnel.remote_addr else None  # noqa: E731
    backend = Backend(client, a.log, refresh=lambda: _quiet(refresh))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(backend))
    log(f"open http://127.0.0.1:{a.port}")
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

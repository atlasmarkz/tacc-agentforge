"""Harness service (Task 5, cluster side): the agent harness as a long-running HTTP service on a TACC node.

The cluster has no public HTTP entry, but the SSH gateway reaches compute nodes by IP, so the local backend
(app/backend.py) opens `ssh -L` to this service and the browser never talks to the cluster directly. Model calls
stay inside the cluster (this process -> vLLM address file); SerpApi is not used at query time.

Protocol (every request needs `Authorization: Bearer <token>`; the token is written next to the published
address with mode 600, so only this account can read it):
  GET  /health                         service, snapshot and model address
  POST /runs                           {run_id, conversation_id, message, constraints, explain?} -> SSE events
  POST /runs/<run_id>/resume           {question_id, answer} -> SSE events (answer to a pending question)
  POST /runs/<run_id>/cancel           stop a running run
  POST /lookup                         {offer_ids, evidence_ids} -> offer facts and evidence (source links)
SSE: `event: agent` carries one AgentEvent (its `result` is set on the final or waiting event), `event: error`
a {code, message} the UI can show, `event: end` closes the stream, and `: ping` comments keep the tunnel busy.
A client that disconnects cancels its run.

The harness (agent/) is used as is: AgentRuntime, the real knowledge tools, the budget validator and the
evidence resolver for the streamed explanation. Policy and model settings come from configs/agent.yaml
(profile tacc-fast by default: thinking off). Checkpoints are in memory: a question left unanswered when this
job ends cannot be resumed by the next job (the SQLite store of PR #13 is not verified on TACC yet).

  python -m app.harness_service --selftest          # local: scripted model, real seed snapshot, real HTTP
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from agent import AgentDependencies, AgentRequest, AgentRuntime, RunStateError
from agent.budget import BudgetPlanValidator
from knowledge.agent_tools import build_knowledge_tools
from knowledge.budget_source import KnowledgeOfferCostSource
from knowledge.explanation import KnowledgeEvidenceResolver

ROOT = Path(__file__).resolve().parents[1]
PORT = 8100
MAX_HEADER = 16 * 1024
MAX_BODY = 64 * 1024
PING_S = 10.0


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class HttpError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed",
           409: "Conflict", 413: "Payload Too Large", 500: "Internal Server Error", 503: "Service Unavailable"}


async def read_request(reader: asyncio.StreamReader):
    head = await reader.readuntil(b"\r\n\r\n")
    if len(head) > MAX_HEADER:
        raise HttpError(413, "headers_too_large", "Request headers are too large.")
    lines = head.decode("latin-1").split("\r\n")
    try:
        method, target, _ = lines[0].split(" ", 2)
    except ValueError:
        raise HttpError(400, "bad_request", "Malformed request line.") from None
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length") or 0)
    if length > MAX_BODY:
        raise HttpError(413, "body_too_large", "Request body is too large.")
    body = await reader.readexactly(length) if length else b""
    return method.upper(), urlsplit(target).path, headers, body


def json_response(writer, status: int, payload) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    writer.write(f"HTTP/1.1 {status} {REASONS.get(status, 'OK')}\r\nContent-Type: application/json; charset=utf-8\r\n"
                 f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode("latin-1") + data)


def sse(event: str, payload) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _all_pages(fetch, ids, items_key, pending_key):
    """Every page of a knowledge answer (results are size-limited and paginated)."""
    items, unknown, cursor = [], [], None
    for _ in range(50):
        page = fetch(ids, cursor)
        items += page[items_key]
        unknown += page.get("unknown_ids", [])
        cursor = page.get(pending_key)
        if not cursor:
            return items, unknown
        if pending_key == "remaining_offer_ids":
            ids, cursor = cursor, None
    raise RuntimeError("knowledge answer did not finish within 50 pages")


class HarnessService:
    def __init__(self, *, model, kb, snapshot_id: str, policy, token: str, info: dict, explain_default: bool):
        tools = build_knowledge_tools(kb)
        self.tools, self.kb, self.snapshot_id, self.token, self.info = tools, kb, snapshot_id, token, info
        self.explain_default = explain_default
        deps = AgentDependencies(model, tools, BudgetPlanValidator(KnowledgeOfferCostSource(kb)),
                                 evidence_resolver=KnowledgeEvidenceResolver(kb))
        from langgraph.checkpoint.memory import InMemorySaver
        saver = InMemorySaver()                               # shared: either runtime can resume any run
        self.runtimes = {flag: AgentRuntime(deps, policy.model_copy(update={"explain": flag}), checkpointer=saver)
                         for flag in (False, True)}
        self.cancels: dict[str, asyncio.Event] = {}
        self.started = time.time()
        self.counts = {"runs": 0, "resumes": 0, "errors": 0}

    def close(self) -> None:
        self.tools.get("search_products").executor.close()

    # ---- HTTP plumbing -------------------------------------------------------------------------------------

    async def handle(self, reader, writer):
        try:
            try:
                method, path, headers, body = await asyncio.wait_for(read_request(reader), 30)
                auth = headers.get("authorization", "")
                if not secrets.compare_digest(auth.encode(), f"Bearer {self.token}".encode()):
                    raise HttpError(401, "unauthorized", "Missing or wrong service token.")
                await self.route(method, path, body, writer)
            except HttpError as exc:
                json_response(writer, exc.status, {"error": {"code": exc.code, "message": exc.message}})
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
                return
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def route(self, method, path, body, writer):
        parts = [p for p in path.split("/") if p]
        if method == "GET" and parts == ["health"]:
            return json_response(writer, 200, self.health())
        if method != "POST":
            raise HttpError(405, "method_not_allowed", "Use POST.")
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise HttpError(400, "bad_json", "The body is not JSON.") from None
        if not isinstance(payload, dict):
            raise HttpError(400, "bad_json", "The body must be a JSON object.")
        if parts == ["runs"]:
            return await self.start_run(payload, writer)
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "resume":
            return await self.resume_run(parts[1], payload, writer)
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "cancel":
            event = self.cancels.get(parts[1])
            if event is not None:
                event.set()
            return json_response(writer, 200, {"cancelled": event is not None})
        if parts == ["lookup"]:
            return json_response(writer, 200, await self.lookup(payload))
        raise HttpError(404, "not_found", "No such endpoint.")

    def health(self) -> dict:
        address_file = self.info.get("model_address_file")
        model_address = None
        if address_file and Path(address_file).exists():
            model_address = Path(address_file).read_text(encoding="utf-8").strip() or None
        return {"status": "ok", "snapshot_id": self.snapshot_id, "uptime_s": round(time.time() - self.started, 1),
                "model_address": model_address, "active_runs": len(self.cancels), "counts": self.counts,
                **{k: v for k, v in self.info.items() if k != "model_address_file"}}

    # ---- runs ----------------------------------------------------------------------------------------------

    async def start_run(self, payload, writer):
        try:
            request = AgentRequest.model_validate({
                "run_id": payload.get("run_id"), "conversation_id": payload.get("conversation_id"),
                "snapshot_id": self.snapshot_id, "message": payload.get("message"),
                "constraints": payload.get("constraints") or {}})
        except Exception as exc:
            raise HttpError(400, "invalid_request", str(exc)[:500]) from None
        explain = payload.get("explain")
        runtime = self.runtimes[self.explain_default if explain is None else bool(explain)]
        self.counts["runs"] += 1
        await self.stream(writer, request.run_id,
                          lambda cancel: runtime.astream(request, cancellation=cancel))

    async def resume_run(self, run_id, payload, writer):
        question_id, answer = payload.get("question_id"), payload.get("answer")
        if not isinstance(question_id, str) or not isinstance(answer, dict):
            raise HttpError(400, "invalid_request", "question_id (string) and answer (object) are required.")
        self.counts["resumes"] += 1
        runtime = self.runtimes[False]           # a resumed run keeps the policy saved in its checkpoint
        await self.stream(writer, run_id,
                          lambda cancel: runtime.astream_resume(run_id, question_id, answer, cancellation=cancel))

    async def stream(self, writer, run_id, make_events):
        if run_id in self.cancels:
            raise HttpError(409, "run_busy", "This run is already being processed.")
        cancel = asyncio.Event()
        self.cancels[run_id] = cancel
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
                     b"Cache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n")
        queue: asyncio.Queue = asyncio.Queue()

        async def produce():
            try:
                async for ev in make_events(cancel):
                    await queue.put(sse("agent", ev.model_dump(mode="json")))
            except RunStateError as exc:
                await queue.put(sse("error", {"code": exc.code, "message": str(exc)}))
            except Exception as exc:                          # the UI shows a code, the log keeps the details
                self.counts["errors"] += 1
                log(f"run {run_id} failed: {type(exc).__name__}: {exc}")
                await queue.put(sse("error", {"code": "internal_error", "message": "The run failed unexpectedly."}))
            finally:
                await queue.put(None)

        task = asyncio.create_task(produce())
        t0 = time.monotonic()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), PING_S)
                except asyncio.TimeoutError:
                    chunk = b": ping\n\n"
                if chunk is None:
                    writer.write(sse("end", {"run_id": run_id, "wall_s": round(time.monotonic() - t0, 2),
                                             "job_id": self.info.get("job_id")}))
                    await writer.drain()
                    break
                writer.write(chunk)
                await writer.drain()                          # raises when the client went away
        except (ConnectionError, OSError):
            log(f"client left run {run_id}: cancelling")
            cancel.set()
        finally:
            await asyncio.gather(task, return_exceptions=True)
            self.cancels.pop(run_id, None)

    # ---- evidence lookup for the UI ------------------------------------------------------------------------

    async def lookup(self, payload) -> dict:
        offer_ids = payload.get("offer_ids") or []
        evidence_ids = payload.get("evidence_ids") or []
        if not all(isinstance(x, list) and len(x) <= 50 and all(isinstance(i, str) for i in x)
                   for x in (offer_ids, evidence_ids)):
            raise HttpError(400, "invalid_request", "offer_ids and evidence_ids must be lists of at most 50 IDs.")

        def work():
            offers, unknown_offers = _all_pages(lambda ids, cur: self.kb.get_offer_facts(self.snapshot_id, ids),
                                                offer_ids, "offers", "remaining_offer_ids") if offer_ids else ([], [])
            evidence, unknown_evidence = _all_pages(
                lambda ids, cur: self.kb.resolve_evidence(self.snapshot_id, ids, cur),
                evidence_ids, "evidence", "next_cursor") if evidence_ids else ([], [])
            return {"snapshot_id": self.snapshot_id, "offers": offers, "evidence": evidence,
                    "unknown_offer_ids": sorted(set(unknown_offers)), "unknown_evidence_ids": sorted(set(unknown_evidence))}
        try:
            return await asyncio.to_thread(work)
        except Exception as exc:
            raise HttpError(400, getattr(exc, "code", "lookup_failed"), str(exc)[:300]) from None


# ---- setup -----------------------------------------------------------------------------------------------

def node_ip() -> str:
    """The node's cluster address (10.x); other nodes and the gateway cannot resolve compute node names."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))                      # no packet is sent
        return s.getsockname()[0]


def atomic_write(path: Path, text: str, mode: int = 0o600) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def prepare_kb(config: Path, userdir: Path, scratch_parent):
    from knowledge.bootstrap import install_snapshot
    from knowledge.service import KnowledgeService
    from knowledge.settings import load_settings
    from knowledge.store import verify_snapshot
    import shutil
    settings = load_settings(config, data_root=userdir / "tacc-agentforge-data",
                             snapshots_root=userdir / "tacc-agentforge-data/data/knowledge/snapshots")
    installed = install_snapshot(settings)
    scratch = Path(tempfile.mkdtemp(prefix="harness-kb-", dir=scratch_parent))
    query_root = scratch / "snapshots"
    shutil.copytree(settings.snapshots_root / settings.snapshot_id, query_root / settings.snapshot_id)
    if not verify_snapshot(query_root / settings.snapshot_id)["valid"]:
        raise SystemExit("query snapshot failed verification")
    return settings.snapshot_id, installed, KnowledgeService(query_root, **settings.service), scratch


def real_model(model_settings):
    from inference.client import ModelClient
    from inference.model_port import VllmModelPort
    m = model_settings
    client = ModelClient(address_file=Path(os.path.expandvars(m.address_file)), model=m.name, max_attempts=m.max_attempts,
                         backoff_base_s=m.backoff_base_s, backoff_max_s=m.backoff_max_s)
    return VllmModelPort(client, temperature=m.temperature, enable_thinking=m.enable_thinking,
                         constrain_output=m.constrain_output, timeout_margin_s=m.timeout_margin_s)


async def serve(service: HarnessService, host: str, port: int, ready=None, stop: asyncio.Event | None = None):
    try:
        server = await asyncio.start_server(service.handle, host, port, limit=MAX_HEADER + MAX_BODY)
    except OSError:                                 # nodes are shared: another process may hold the port
        log(f"port {port} is taken; using a free one")
        server = await asyncio.start_server(service.handle, host, 0, limit=MAX_HEADER + MAX_BODY)
    bound = server.sockets[0].getsockname()[1]
    if ready is not None:
        ready(bound)
    async with server:
        if stop is None:
            await server.serve_forever()
        else:
            await stop.wait()


def run_demo(a) -> int:
    from agent import RunPolicy
    from app.selftest import ScriptedPort
    userdir = Path(tempfile.mkdtemp(prefix="harness-demo-"))
    snapshot_id, _, kb, _ = prepare_kb(a.config, userdir, userdir)
    service = HarnessService(model=ScriptedPort(), kb=kb, snapshot_id=snapshot_id,
                             policy=RunPolicy(max_context_chars=36000), token=a.token,
                             info={"job_id": "demo", "node": "local"}, explain_default=True)
    log(f"demo harness on http://127.0.0.1:{a.port} (token {a.token}); scripted model, not a real one")
    try:
        asyncio.run(serve(service, "127.0.0.1", a.port))
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="scripted model, seed snapshot, real HTTP (no TACC)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/knowledge.json")
    ap.add_argument("--agent-config", type=Path, default=ROOT / "configs/agent.yaml")
    ap.add_argument("--profile", default="tacc-fast")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--explain", action="store_true", help="explain verified plans by default (per request: explain)")
    ap.add_argument("--deadline-s", type=float, default=0, help="stop accepting and exit after this many seconds")
    ap.add_argument("--demo", action="store_true", help="offline UI development: scripted model, seed snapshot, "
                                                        "listens on 127.0.0.1 with --token")
    ap.add_argument("--token", default="demo")
    a = ap.parse_args()
    if a.selftest:
        from app.selftest import run_selftest
        return asyncio.run(run_selftest(a.config))
    if a.demo:
        return run_demo(a)

    from agent.settings import load_runtime_settings
    settings = load_runtime_settings(a.agent_config, a.profile)
    userdir = Path(os.environ["TACC_USERDIR"])
    base = userdir / "tacc-agentforge-service"
    base.mkdir(parents=True, exist_ok=True)
    snapshot_id, installed, kb, scratch = prepare_kb(a.config, userdir, os.environ.get("SLURM_TMPDIR"))
    log(f"snapshot {snapshot_id}: {installed['status']}")
    token = secrets.token_urlsafe(32)
    ip = node_ip()
    info = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
            "profile": a.profile, "model_address_file": os.path.expandvars(settings.model.address_file), "explain_default": a.explain}
    service = HarnessService(model=real_model(settings.model), kb=kb, snapshot_id=snapshot_id,
                             policy=settings.policy, token=token, info=info, explain_default=a.explain)
    address_file, token_file = base / "harness-url.txt", base / "harness-token.txt"

    address = None

    def publish(port):
        nonlocal address
        address = info["address"] = f"http://{ip}:{port}"
        atomic_write(token_file, token + "\n")                # token first: the address means "ready"
        atomic_write(address_file, address + "\n")
        log(f"READY harness service {address} (job {info['job_id']}) -> {address_file}")

    async def run():
        stop = asyncio.Event()
        if a.deadline_s > 0:
            asyncio.get_running_loop().call_later(a.deadline_s, stop.set)
        try:
            await serve(service, "0.0.0.0", a.port, ready=publish, stop=stop)
        finally:
            if address and address_file.exists() and address_file.read_text(encoding="utf-8").strip() == address:
                address_file.unlink(missing_ok=True)          # never remove a successor's newer address
            log(f"stopped: {service.counts}")
    try:
        asyncio.run(run())
    finally:
        service.close()
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

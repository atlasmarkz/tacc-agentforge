"""One command for the artifact evaluation: run every stage of the README that needs no GPU and say, for each,
SUCCESS or FAILED with the reason.

    .venv/bin/python scripts/check_artifact.py            # all stages, about 4 minutes
    .venv/bin/python scripts/check_artifact.py --skip-tests

Stages (README section in brackets):
  1. environment        Python >= 3.12 and the locked dependencies import        [Getting Started, Setup]
  2. catalog snapshot   knowledge bootstrap + doctor: ready, verified, pinned ID [Getting Started, Setup]
  3. automated tests    pytest tests agent/tests: no failures                    [Example 1]
  4. harness self-test  app.harness_service --selftest: SELFTEST OK              [Example 2]
  5. end-to-end request demo harness + backend over HTTP, like the browser      [Example 2]
  6. main results       scripts/reproduce_results.py: ALL MATCH (C1-C7)         [Detailed Instructions, Step 1]
  7. v1 benchmark       fixture data hashes and preflight: READY                 [Detailed Instructions, v1]

The last line is `ALL 7 STAGES SUCCEEDED`, or names the stages that failed; the exit code is 0 only if all
succeeded. Nothing here needs a GPU, a TACC account or the network (the snapshot is in the repository).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SNAPSHOT = "snapshot_5bcd00723fb0b7589cbe50c8"
# Example 2's expected verified plan (scripted model, frozen snapshot, real budget validator).
EXPECTED_PLAN = {"offer_id": "offer_640a303bc366430cfd53139e", "total_cost": "16.87", "remaining_funds": "83.13"}


class StageFailed(Exception):
    pass


def run(args, timeout, env=None):
    done = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "PYTHONUTF8": "1", **(env or {})})
    return done.returncode, done.stdout + done.stderr


def tail(text, lines=15):
    return "\n".join(text.strip().splitlines()[-lines:])


# ---- stages ----------------------------------------------------------------------------------------------------

def stage_environment():
    if sys.version_info < (3, 12):
        raise StageFailed(f"Python {sys.version.split()[0]}; this artifact needs 3.12 (see Prerequisites)")
    missing = []
    for module in ("langgraph", "pydantic", "aiosqlite", "yaml", "pytest", "dotenv"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise StageFailed(f"missing packages {missing}: run the Setup install command with this interpreter ({PY})")
    return f"Python {sys.version.split()[0]}, dependencies import"


def stage_snapshot():
    code, out = run([PY, "-m", "knowledge", "bootstrap"], 600)
    if code != 0:
        raise StageFailed("knowledge bootstrap failed:\n" + tail(out))
    code, out = run([PY, "-m", "knowledge", "doctor"], 300)
    try:
        report = json.loads(out[out.index("{"):])
    except ValueError:
        raise StageFailed("knowledge doctor did not print a JSON report:\n" + tail(out)) from None
    verification = report.get("verification") or {}
    if code != 0 or not report.get("ready") or not verification.get("valid") or verification.get("snapshot_id") != SNAPSHOT:
        raise StageFailed(f"doctor: ready={report.get('ready')} valid={verification.get('valid')} "
                          f"snapshot={verification.get('snapshot_id')} errors={verification.get('errors')}")
    return f"{SNAPSHOT} installed and verified (FTS5: {report.get('fts5')})"


def stage_tests():
    code, out = run([PY, "-m", "pytest", "tests", "agent/tests", "-q", "-p", "no:cacheprovider"], 1800)
    summary = next((line for line in reversed(out.splitlines()) if re.search(r"\d+ (passed|failed)", line)), "")
    if code != 0 or "failed" in summary or "error" in summary:
        raise StageFailed("pytest reported failures:\n" + tail(out, 25))
    return summary.strip("= ")


def stage_selftest():
    code, out = run([PY, "-m", "app.harness_service", "--selftest"], 600)
    if code != 0 or "SELFTEST OK" not in out:
        raise StageFailed("self-test did not print SELFTEST OK:\n" + tail(out))
    checks = re.findall(r'"(\w+)": (true|false)', out)
    return f"SELFTEST OK ({sum(v == 'true' for _, v in checks)}/{len(checks)} checks true)"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(port, path, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", path)
            if conn.getresponse().status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def ask(port, constraints):
    """POST /api/ask like the page does; returns the final agent result from the server-sent events."""
    body = json.dumps({"session_id": "artifact-check-1234", "message": "wireless earbuds for commuting, under 100 USD",
                       "constraints": constraints}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)       # direct: no proxy for localhost
    conn.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    text = response.read().decode("utf-8")
    if response.status != 200:
        raise StageFailed(f"/api/ask answered HTTP {response.status}: {text[:300]}")
    result, events = None, []
    for chunk in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in chunk.splitlines() if ": " in line)
        if fields.get("event") == "agent":
            data = json.loads(fields["data"])
            events.append(data["type"])
            result = data.get("result") or result
        elif fields.get("event") == "error":
            raise StageFailed(f"the run reported an error event: {fields.get('data')}")
    if result is None:
        raise StageFailed(f"no final result in the event stream (events: {events})")
    return result, events


def stage_end_to_end():
    harness_port, backend_port, token = free_port(), free_port(), "artifact-check"
    procs = [subprocess.Popen([PY, "-m", "app.harness_service", "--demo", "--port", str(harness_port), "--token", token],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT),
             subprocess.Popen([PY, "-m", "app.backend", "--port", str(backend_port),
                               "--harness-url", f"http://127.0.0.1:{harness_port}", "--token", token],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)]
    try:
        if not wait_http(backend_port, "/api/health", 120):
            raise StageFailed("the backend did not answer /api/health within 120 s")
        end = time.monotonic() + 120                     # the demo harness copies the snapshot before serving
        while time.monotonic() < end:
            conn = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=10)
            conn.request("GET", "/api/health")
            if (json.loads(conn.getresponse().read()).get("harness") or {}).get("status") == "ok":
                break
            time.sleep(1)
        else:
            raise StageFailed("the demo harness did not become healthy within 120 s")
        result, events = ask(backend_port, {"currency": "USD", "purchase_cap": "100", "budget_basis": "pre_tax"})
        plan = result.get("verified_plan") or {}
        item = (plan.get("items") or [{}])[0]
        got = {"offer_id": item.get("offer_id"), "total_cost": plan.get("total_cost"),
               "remaining_funds": plan.get("remaining_funds")}
        if result.get("status") != "completed" or got != EXPECTED_PLAN or plan.get("snapshot_id") != SNAPSHOT:
            raise StageFailed(f"expected a completed run with {EXPECTED_PLAN}, got status={result.get('status')} {got}")
        for needed in ("run_started", "tool_started", "validation_finished", "run_completed"):
            if needed not in events:
                raise StageFailed(f"event {needed} missing from the stream: {events}")
        waiting, _ = ask(backend_port, {"currency": "USD"})
        if waiting.get("status") != "waiting_user":
            raise StageFailed(f"without a budget the run should wait for the user, got {waiting.get('status')}")
        return (f"verified plan {got['offer_id']}: total {got['total_cost']} USD, {got['remaining_funds']} left; "
                f"without a budget the run asks for one")
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()


def stage_reproduce():
    code, out = run([PY, "scripts/reproduce_results.py"], 900)
    matches = out.count("\nMATCH")
    if code != 0 or not out.rstrip().endswith("ALL MATCH"):
        differs = [line for line in out.splitlines() if line.startswith("DIFFERS")]
        raise StageFailed("reproduce_results.py did not end with ALL MATCH:\n" + ("\n".join(differs) or tail(out)))
    return f"ALL MATCH ({matches} comparisons: C1-C7 recomputed from the committed raw data)"


def stage_v1():
    code, out = run([PY, "-m", "evaluations.v1.suite", "verify"], 120)
    if code != 0:
        raise StageFailed("v1 fixture data hashes do not match its manifest:\n" + tail(out))
    code, out = run([PY, "-m", "evaluations.bench", "check"], 300)
    if code != 0 or "RESULT: READY" not in out:
        raise StageFailed("bench check is not READY:\n" + tail(out))
    return "fixture data verified; preflight READY (a model endpoint is only needed for new runs)"


STAGES = [("environment", stage_environment), ("catalog snapshot", stage_snapshot),
          ("automated tests", stage_tests), ("harness self-test", stage_selftest),
          ("end-to-end request", stage_end_to_end), ("main results", stage_reproduce),
          ("v1 benchmark", stage_v1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tests", action="store_true", help="skip stage 3 (the 1.5-minute test suite)")
    a = ap.parse_args()
    failed, total = [], len(STAGES)
    print(f"Artifact check: {ROOT}\n")
    for number, (name, stage) in enumerate(STAGES, 1):
        if a.skip_tests and stage is stage_tests:
            print(f"[{number}/{total}] {name:<20} SKIPPED (--skip-tests)", flush=True)
            continue
        print(f"[{number}/{total}] {name:<20} running ...", flush=True)
        started = time.monotonic()
        try:
            detail = stage()
            print(f"[{number}/{total}] {name:<20} SUCCESS ({time.monotonic() - started:.0f} s): {detail}", flush=True)
        except StageFailed as exc:
            failed.append(name)
            print(f"[{number}/{total}] {name:<20} FAILED ({time.monotonic() - started:.0f} s): {exc}", flush=True)
        except Exception as exc:                          # report, keep going with the other stages
            failed.append(name)
            print(f"[{number}/{total}] {name:<20} FAILED: {type(exc).__name__}: {exc}", flush=True)
    print()
    if failed:
        print(f"FAILED STAGES: {', '.join(failed)}")
        return 1
    ran = total - (1 if a.skip_tests else 0)
    print(f"ALL {ran} STAGES SUCCEEDED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

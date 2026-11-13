"""Platform check for Task 5: drives the running local backend exactly as the browser does.

  browser API (this script) -> app/backend.py -> ssh -L via gateway -> harness service job -> vLLM job

Scenarios: a request with a budget; a request without one (question, answer, resume); a revision of the first
result; a cancelled run; three sessions at once; another session trying to use someone else's run. Prints a JSON
report with statuses, timings and the harness job that served each run.

  python -m app.backend &                     # with the two TACC jobs running
  python -m app.cluster_check --backend http://127.0.0.1:8765 --out report.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import http.client
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit


class Browser:
    def __init__(self, base: str):
        u = urlsplit(base)
        self.host, self.port = u.hostname, u.port
        self.session = str(uuid.uuid4())

    def _post(self, path, body, timeout=300):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        conn.request("POST", path, json.dumps({"session_id": self.session, **body}), {"Content-Type": "application/json"})
        return conn, conn.getresponse()

    def post(self, path, body):
        conn, r = self._post(path, body)
        try:
            return r.status, json.loads(r.read() or b"null")
        finally:
            conn.close()

    def stream(self, path, body, on_event=None):
        """Returns (http status, summary). Timing is measured as the browser would see it."""
        t0 = time.monotonic()
        conn, r = self._post(path, body)
        out = {"http": r.status, "events": [], "first_event_s": None, "first_delta_s": None}
        if r.status != 200:
            out["error"] = json.loads(r.read() or b"null")
            conn.close()
            return out
        event, data = None, []
        try:
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif line == "" and event:
                    payload = json.loads("".join(data) or "null")
                    now = round(time.monotonic() - t0, 2)
                    if out["first_event_s"] is None and event == "agent":
                        out["first_event_s"] = now
                    if event == "meta":
                        out["run_id"] = payload["run_id"]
                    elif event == "agent":
                        out["events"].append([now, payload["type"]])
                        if payload["type"] == "explain_delta" and out["first_delta_s"] is None:
                            out["first_delta_s"] = now
                        if payload.get("result"):
                            out["result"] = payload["result"]
                    elif event == "error":
                        out["error"] = payload
                    elif event == "end":
                        out["harness_job"] = payload.get("job_id")
                    if on_event:
                        on_event(event, payload, out)
                    event, data = None, []
        finally:
            conn.close()
        out["wall_s"] = round(time.monotonic() - t0, 2)
        return out


def brief(out):
    r = out.get("result") or {}
    plan = r.get("verified_plan") or {}
    return {"http": out["http"], "run_id": out.get("run_id"), "status": r.get("status"),
            "stop_reason": r.get("stop_reason"), "error": out.get("error"), "wall_s": out.get("wall_s"),
            "first_event_s": out.get("first_event_s"), "first_explain_delta_s": out.get("first_delta_s"),
            "harness_job": out.get("harness_job"), "snapshot_id": r.get("snapshot_id"),
            "total": plan.get("total_cost"), "remaining": plan.get("remaining_funds"),
            "offers": [i["offer_id"] for i in plan.get("items", [])],
            "explanation": (r.get("explanation") or {}).get("text", "")[:300] or None,
            "explanation_fallback": (r.get("explanation") or {}).get("fallback"),
            "question": (r.get("pending_question") or {}).get("question") or (r.get("clarification") or {}).get("question"),
            "usage": r.get("usage"), "steps": [t for _, t in out["events"] if t not in ("explain_delta",)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8765")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    report = {}

    def run(name, fn):
        t0 = time.monotonic()
        try:
            report[name] = fn()
        except Exception as exc:
            report[name] = {"exception": f"{type(exc).__name__}: {exc}"}
        print(f"{name}: {json.dumps(report[name], ensure_ascii=False)[:400]} ({time.monotonic() - t0:.1f}s)", flush=True)

    b = Browser(a.backend)
    conn = http.client.HTTPConnection(b.host, b.port, timeout=120)
    conn.request("GET", "/api/health")
    report["health"] = json.loads(conn.getresponse().read())
    conn.close()
    print("health:", json.dumps(report["health"], ensure_ascii=False))

    first = {}

    def with_budget():
        out = b.stream("/api/ask", {"message": "Recommend wireless earbuds for my daily commute.",
                                    "constraints": {"purchase_cap": "100.00", "currency": "USD"}, "explain": True})
        first.update(out)
        res = brief(out)
        if out.get("result", {}).get("verified_plan"):
            p = out["result"]["verified_plan"]
            st, looked = b.post("/api/lookup", {"run_id": out["run_id"], "offer_ids": [i["offer_id"] for i in p["items"]],
                                                "evidence_ids": [e for i in p["items"] for e in i["evidence_ids"]]})
            res["lookup"] = {"http": st, "evidence_with_links": sum(1 for e in looked.get("evidence", []) if e.get("source_url")),
                             "evidence": len(looked.get("evidence", []))}
        return res
    run("1_request_with_budget", with_budget)

    def ask_then_answer():
        c = Browser(a.backend)
        out = c.stream("/api/ask", {"message": "我想买一个通勤用的双肩包。", "constraints": {}, "explain": True})
        res = {"ask": brief(out)}
        q = (out.get("result") or {}).get("pending_question")
        if q:
            answer = {f: ("USD" if f == "currency" else "80.00") for f in q["required_fields"]}
            res["answer_sent"] = answer
            res["resume"] = brief(c.stream("/api/answer", {"run_id": out["run_id"], "question_id": q["question_id"],
                                                           "answer": answer}))
        return res
    run("2_question_and_resume", ask_then_answer)

    def revision():
        if not first.get("run_id"):
            return {"skipped": "no first run"}
        out = b.stream("/api/ask", {"message": "Recommend wireless earbuds for my daily commute.\n修改要求:便宜一点,30美元以内",
                                    "constraints": {"purchase_cap": "30.00", "currency": "USD"},
                                    "revises": first["run_id"], "explain": True})
        return brief(out)
    run("3_revision", revision)

    def cancel():
        c = Browser(a.backend)
        sent = {}

        def on_event(event, payload, out):
            if event == "agent" and payload["type"] == "decision_started" and not sent:
                sent["at"] = out["events"][-1][0]
                sent["reply"] = c.post("/api/cancel", {"run_id": out["run_id"]})
        out = c.stream("/api/ask", {"message": "Which mechanical keyboard is good for programming? Budget 120 USD.",
                                    "constraints": {"purchase_cap": "120.00", "currency": "USD"}}, on_event)
        return {**brief(out), "cancel_sent_at_s": sent.get("at"), "cancel_reply": sent.get("reply")}
    run("4_cancel", cancel)

    def isolation():
        other = Browser(a.backend)
        return {"answer_other_run": other.post("/api/answer", {"run_id": first.get("run_id"), "question_id": "q", "answer": {}})[0],
                "lookup_other_run": other.post("/api/lookup", {"run_id": first.get("run_id"), "offer_ids": []})[0]}
    run("5_session_isolation", isolation)

    def concurrent():
        msgs = [("I need a USB-C charger for my laptop and phone.", "60.00"),
                ("Recommend a backpack for commuting.", "80.00"),
                ("Recommend wireless earbuds.", "50.00")]
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(3) as pool:
            outs = list(pool.map(lambda m: Browser(a.backend).stream(
                "/api/ask", {"message": m[0], "constraints": {"purchase_cap": m[1], "currency": "USD"}, "explain": True}), msgs))
        runs = [brief(o) for o in outs]
        return {"wall_s": round(time.monotonic() - t0, 2), "distinct_run_ids": len({r["run_id"] for r in runs}),
                "runs": runs}
    run("6_three_sessions_at_once", concurrent)

    a.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("report:", a.out)


if __name__ == "__main__":
    main()

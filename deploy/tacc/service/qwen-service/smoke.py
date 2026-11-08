"""Smoke test against the live service with prompts built by the real agent harness.

Questions it answers (results in smoke.json):
  1. Does the model return a valid agent action when output is constrained by the action JSON schema?
  2. What does thinking mode do to output length and to finish_reason (max_output_tokens is only 512)?
  3. Does it comply without constrained decoding? (tells how much the constraint is worth)
  4. Latency per call, alone and with 4 concurrent callers; is temperature 0 repeatable?
All requests are synthetic (see scripts/make_service_fixture.py).
"""
import json
import statistics
import threading
import time
from pathlib import Path

from client import ModelClientError

VARIANTS = {
    "constrained_think_default": dict(constrain=True, thinking=None),
    "constrained_no_think": dict(constrain=True, thinking=False),
    "free_no_think": dict(constrain=False, thinking=False),
}
REPEATS = 2


def check_action(text: str) -> tuple[bool, str]:
    try:
        obj = json.loads(text)
    except ValueError:
        return False, "not json"
    if not isinstance(obj, dict):
        return False, "not an object"
    kind = obj.get("type")
    if kind == "call_tool":
        ok = isinstance(obj.get("tool_name"), str) and isinstance(obj.get("arguments"), dict)
        return ok, "" if ok else "bad call_tool"
    if kind == "propose_plan":
        items = (obj.get("plan") or {}).get("items")
        ok = bool(items) and all(isinstance(i.get("quantity"), int) and i["quantity"] > 0 and i.get("offer_id")
                                 and i.get("evidence_ids") for i in items)
        return ok, "" if ok else "bad propose_plan"
    if kind == "ask_user":
        ok = bool(obj.get("question")) and bool(obj.get("required_fields"))
        return ok, "" if ok else "bad ask_user"
    return False, f"unknown type {kind!r}"


def one(client, case, schema, variant, timeout_s=60):
    t0 = time.monotonic()
    rec = {"case": case["name"], "variant": variant}
    try:
        r = client.chat(case["messages"], max_tokens=case["max_tokens"], temperature=0, timeout_s=timeout_s,
                        json_schema=schema if VARIANTS[variant]["constrain"] else None,
                        enable_thinking=VARIANTS[variant]["thinking"])
    except ModelClientError as exc:
        rec.update(error=f"{type(exc).__name__}: {exc}", valid=False, duration_s=round(time.monotonic() - t0, 2))
        return rec
    valid, why = check_action(r.content)
    rec.update(valid=valid and r.finish_reason == "stop", why=why, finish_reason=r.finish_reason,
               prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
               reasoning_chars=len(r.reasoning or ""), duration_s=round(r.duration_s, 2),
               content=r.content[:300])
    return rec


def stream_one(client, case, schema, variant, timeout_s=60):
    """One streamed call; also fetches the blocking answer (temperature 0) to compare the text."""
    stamps, kinds = [], {"content": 0, "reasoning": 0}

    def on_delta(kind, text):
        stamps.append(time.monotonic())
        kinds[kind] += 1

    rec = {"case": case["name"], "variant": variant}
    t0 = time.monotonic()
    try:
        kw = dict(max_tokens=case["max_tokens"], temperature=0, timeout_s=timeout_s, json_schema=schema,
                  enable_thinking=VARIANTS[variant]["thinking"])
        streamed = client.chat(case["messages"], on_delta=on_delta, **kw)
        blocking = client.chat(case["messages"], **kw)
    except ModelClientError as exc:
        rec.update(error=f"{type(exc).__name__}: {exc}", duration_s=round(time.monotonic() - t0, 2))
        return rec
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    valid, _ = check_action(streamed.content)
    rec.update(first_token_s=round(streamed.first_token_s, 3) if streamed.first_token_s is not None else None,
               duration_s=round(streamed.duration_s, 2), deltas=kinds, max_gap_s=round(max(gaps), 3) if gaps else 0,
               completion_tokens=streamed.completion_tokens, finish_reason=streamed.finish_reason,
               valid=valid and streamed.finish_reason == "stop",
               same_as_blocking=streamed.content == blocking.content,
               blocking_duration_s=round(blocking.duration_s, 2))
    return rec


def summarise(rows):
    ok = [r for r in rows if "error" not in r]
    lat = sorted(r["duration_s"] for r in rows)
    finish = {}
    for r in ok:
        finish[r["finish_reason"]] = finish.get(r["finish_reason"], 0) + 1
    return {
        "calls": len(rows), "valid": sum(r["valid"] for r in rows), "errors": len(rows) - len(ok),
        "valid_rate": round(sum(r["valid"] for r in rows) / len(rows), 3) if rows else None,
        "finish_reasons": finish,
        "latency_s": {"median": statistics.median(lat), "max": lat[-1]} if lat else None,
        "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None), 1)
        if any(r.get("completion_tokens") is not None for r in ok) else None,
        "with_reasoning": sum(1 for r in ok if r["reasoning_chars"] > 0),
    }


def run(client, out_dir: Path, cfg: dict) -> None:
    fixture = json.loads((Path(__file__).parent / "smoke_requests.json").read_text(encoding="utf-8"))
    schema, cases = fixture["action_schema"], fixture["cases"]
    report = {"variants": {}, "rows": []}
    for variant in VARIANTS:
        rows = [one(client, c, schema, variant) for _ in range(REPEATS) for c in cases]
        report["rows"] += rows
        report["variants"][variant] = summarise(rows)
        print(variant, json.dumps(report["variants"][variant]), flush=True)

    # Repeatability at temperature 0 (constrained, no thinking): same case twice -> same text?
    same = []
    for c in cases:
        a = one(client, c, schema, "constrained_no_think")
        b = one(client, c, schema, "constrained_no_think")
        same.append({"case": c["name"], "identical": a.get("content") == b.get("content")})
    report["repeatability"] = same

    # Streaming: time to first token, pacing, and whether streamed text equals the blocking answer.
    stream_rows = []
    for variant in ("constrained_no_think", "constrained_think_default"):
        for c in cases:
            stream_rows.append(stream_one(client, c, schema, variant))
    ok = [r for r in stream_rows if "error" not in r]
    report["streaming"] = {
        "calls": len(stream_rows), "errors": len(stream_rows) - len(ok),
        "same_text_as_blocking": sum(r["same_as_blocking"] for r in ok),
        "by_variant": {v: {
            "first_token_s_median": statistics.median([r["first_token_s"] for r in ok if r["variant"] == v and r["first_token_s"] is not None] or [None]),
            "total_s_median": statistics.median([r["duration_s"] for r in ok if r["variant"] == v] or [None]),
            "max_gap_s_median": statistics.median([r["max_gap_s"] for r in ok if r["variant"] == v] or [None]),
        } for v in ("constrained_no_think", "constrained_think_default")},
        "rows": stream_rows,
    }
    print("streaming", json.dumps({k: v for k, v in report["streaming"].items() if k != "rows"}), flush=True)

    # Four callers at once (constrained, no thinking): latency under light concurrency.
    conc_rows, lock = [], threading.Lock()

    def worker(idx):
        for c in cases:
            row = one(client, c, schema, "constrained_no_think")
            with lock:
                conc_rows.append(row)

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    report["concurrent4"] = {**summarise(conc_rows), "wall_s": round(time.monotonic() - t0, 1)}
    print("concurrent4", json.dumps(report["concurrent4"]), flush=True)
    (out_dir / "smoke.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

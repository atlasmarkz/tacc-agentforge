"""Serve an SFT adapter next to the base model in one vLLM and compare their first action on the dev prompts.

Questions answered (results in check.json):
  1. Does vLLM load and apply the adapter saved by training/sft.py (peft layout, rank 8)?
  2. Is the adapter's output valid under the harness's constrained decoding, and does it differ from the base?
  3. On the first step of each dev case, without the test-only category hint, does each model choose the
     right category (or ask when the budget is missing)? Cases are marked train / validation / unseen.

Same request as the deployed harness: thinking off, temperature 0, 512 output tokens, JSON-schema constrained
output with the anchored patterns (inference.model_port.decoder_schema, repeated here so that this script needs
neither the harness nor pydantic). This is a first-action check on ten prompts, not an evaluation of the system.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 8000
BASE = "qwen3.8-27b"


def anchor(pattern: str) -> str:
    """Same as inference.model_port._anchor (a test asserts they agree)."""
    if not pattern.startswith("^"):
        pattern = ".*" + pattern
    if not pattern.endswith("$"):
        pattern = pattern + ".*"
    return pattern


def decoder_schema(schema):
    if isinstance(schema, dict):
        return {k: (anchor(v) if k == "pattern" and isinstance(v, str) else decoder_schema(v)) for k, v in schema.items()}
    if isinstance(schema, list):
        return [decoder_schema(v) for v in schema]
    return schema


def post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ask(model: str, prompt: dict, schema: dict) -> dict:
    t0 = time.monotonic()
    r = post("/v1/chat/completions", {
        "model": model, "messages": prompt["messages"], "max_tokens": 512, "temperature": 0, "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {"name": "action", "schema": schema, "strict": True}},
        "chat_template_kwargs": {"enable_thinking": False}})
    choice = r["choices"][0]
    text = choice["message"].get("content") or ""
    out = {"text": text, "finish": choice["finish_reason"], "seconds": round(time.monotonic() - t0, 2),
           "completion_tokens": r["usage"]["completion_tokens"]}
    try:
        action = json.loads(text)
        out["valid_json"] = isinstance(action, dict) and out["finish"] == "stop"
        out["type"] = action.get("type")
        out["tool"] = action.get("tool_name")
        out["category"] = ((action.get("arguments") or {}).get("filters") or {}).get("category")
        out["question"] = action.get("question")
    except ValueError:
        out["valid_json"] = False
    return out


def correct_first_action(expected: dict, got: dict) -> bool:
    if not got.get("valid_json"):
        return False
    if expected["type"] == "ask_user":
        return got["type"] == "ask_user"
    return got["type"] == "call_tool" and got["tool"] == "search_products" and got["category"] == expected["category"]


def summarise(rows: list[dict]) -> dict:
    out = {}
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        by_role = {}
        for role in ("train", "validation", "unseen"):
            sub = [r for r in mine if r["role"] == role]
            if sub:
                by_role[role] = {"correct": sum(r["correct"] for r in sub), "of": len(sub)}
        out[model] = {"valid_json": {"n": sum(r["valid_json"] for r in mine), "of": len(mine)},
                      "correct_first_action": {"n": sum(r["correct"] for r in mine), "of": len(mine)},
                      "by_role": by_role}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True, help="directory with adapter_config.json")
    ap.add_argument("--prompts", type=Path, default=Path("check_prompts.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout-s", type=int, default=1500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((a.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    data = json.loads(a.prompts.read_text(encoding="utf-8"))
    schema = decoder_schema(data["schema"])
    result = {"adapter": str(a.adapter), "adapter_config": cfg}
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", os.environ["QWEN_MODEL_PATH"],
           "--served-model-name", BASE, "--host", "127.0.0.1", "--port", str(PORT), "--tensor-parallel-size", "2",
           "--safetensors-load-strategy", "prefetch", "--language-model-only", "--dtype", "bfloat16",
           "--gpu-memory-utilization", "0.85", "--max-model-len", "16384", "--max-num-seqs", "8",
           "--max-num-batched-tokens", "4096", "--reasoning-parser", "qwen3", "--enable-prefix-caching",
           "--disable-custom-all-reduce", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", str(cfg["r"]), "--max-loras", "1", "--lora-modules", f"sft={a.adapter}"]
    env = dict(os.environ, NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
    log = open(a.out / "server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    try:
        while True:
            if proc.poll() is not None:
                result.update(status="server_exited", exit_code=proc.returncode)
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                result["startup_s"] = round(time.time() - t0, 1)
                break
            except Exception:
                if time.time() - t0 > a.timeout_s:
                    result["status"] = "startup_timeout"
                    break
                time.sleep(5)
        if "startup_s" in result:
            result["served_models"] = [m["id"] for m in json.load(
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10))["data"]]
            rows = []
            for p in data["prompts"]:
                for model in (BASE, "sft"):
                    try:
                        got = ask(model, p, schema)
                    except Exception as exc:               # an error is a result, not a reason to stop
                        got = {"valid_json": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
                    got.update(model=model, case_id=p["case_id"], role=p["role"], expected=p["expected"])
                    got["correct"] = correct_first_action(p["expected"], got)
                    rows.append(got)
                    print(f"{p['case_id']} {p['role']:10s} {model:12s} ok={got['correct']} "
                          f"{got.get('type')} {got.get('category')} {got.get('seconds')}s", flush=True)
            result.update(status="ok", rows=rows, summary=summarise(rows),
                          differs_from_base=sum(1 for p in data["prompts"]
                                                if next(r for r in rows if r["case_id"] == p["case_id"] and r["model"] == "sft").get("text")
                                                != next(r for r in rows if r["case_id"] == p["case_id"] and r["model"] == BASE).get("text")))
            print(json.dumps(result["summary"], indent=1), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        (a.out / "check.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

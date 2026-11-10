"""Build deploy/tacc/service/load-test/workload.json from real harness data.

- `system`: the exact system message the harness sends through VllmModelPort today, i.e. the harness system
  prompt plus the four knowledge tools and the action schema (built from the current code, not copied text).
- `pairs`: (prompt_tokens, completion_tokens) of every model call in the think-off runs of TACC job 28561
  (real harness + knowledge tools + model), so the load test reproduces the real length distribution.
- `runs`: the same calls grouped per harness run, in order, for replaying whole runs (growing prompts).

Run from the repository root with the agent dependencies installed:
    PYTHONPATH=. python scripts/make_load_workload.py
"""
import json
from pathlib import Path

from agent.contracts import ACTION_ADAPTER, Message, ModelRequest
from agent.prompts import SYSTEM_PROMPT
from inference.model_port import to_chat_messages
from knowledge.agent_tools import build_knowledge_tools

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inference/results/agent-e2e-kb-28561/e2e-kb.json"
OUT = ROOT / "deploy/tacc/service/load-test/workload.json"


class _ToolSizeOnly:
    """build_knowledge_tools only needs the result size limit to describe the tools."""
    max_result_bytes = 8192


def main():
    tools = build_knowledge_tools(_ToolSizeOnly())
    try:
        request = ModelRequest(messages=[Message(role="system", content=SYSTEM_PROMPT),
                                         Message(role="user", content="{}")],
                               tools=tools.descriptions(), action_schema=ACTION_ADAPTER.json_schema(),
                               max_output_tokens=512)
    finally:
        tools.get("search_products").executor.close()
    system = to_chat_messages(request)[0]["content"]

    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    pairs = [[c["prompt_tokens"], c["completion_tokens"]]
             for run in report["runs"] if run["variant"] == "think-off"
             for c in run["model_calls"]
             if c.get("prompt_tokens") and c.get("completion_tokens")]
    # Whole runs, in call order: within a run each prompt extends the previous one (prefix-cache reuse).
    runs = [[[c["prompt_tokens"], c["completion_tokens"]] for c in run["model_calls"]
             if c.get("prompt_tokens") and c.get("completion_tokens")]
            for run in report["runs"] if run["variant"] == "think-off"]
    runs = [r for r in runs if r]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "think-off model calls of TACC job 28561 (harness + knowledge tools + Qwen3.8-27B)",
        "system_chars": len(system), "pairs": pairs, "runs": runs, "system": system,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    prompts = sorted(p for p, _ in pairs)
    print(f"{len(runs)} runs, {len(pairs)} calls, prompt tokens min/median/max {prompts[0]}/{prompts[len(prompts)//2]}/{prompts[-1]}, "
          f"system {len(system)} chars -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Build smoke-test requests with the real harness code (prompt, tools, action schema).

Run with the agent environment from the repository root:
    agent/.venv/Scripts/python.exe scripts/make_service_fixture.py
Output: deploy/tacc/service/qwen-service/smoke_requests.json (all products and evidence are synthetic).
"""
import json
from pathlib import Path

from agent import AgentRequest
from agent.contracts import ACTION_ADAPTER
from agent.examples.mock_run import demo_tools
from agent.policies import RunPolicy
from agent.prompts import build_model_request
from agent.state import initial_state
from inference.model_port import to_chat_messages

OUT = Path("deploy/tacc/service/qwen-service/smoke_requests.json")
POLICY = RunPolicy()
TOOLS = demo_tools()

REQUESTS = [
    ("zh-commute-earbuds", "推荐预算内适合通勤的无线耳机。", {"currency": "USD", "purchase_cap": "100.00"}),
    ("en-kettle", "I need an electric kettle under 40 dollars with auto shut-off.", {"currency": "USD", "purchase_cap": "40.00"}),
    ("zh-missing-budget", "帮我买一台笔记本电脑。", {}),
    ("en-two-items", "Buy two USB-C cables and one 65W charger, total under 60 USD.", {"currency": "USD", "purchase_cap": "60.00"}),
    ("zh-backpack", "给孩子买一个轻便的书包，预算 200 元人民币。", {"currency": "CNY", "purchase_cap": "200.00"}),
]


def build(name, message, constraints, history):
    req = AgentRequest(run_id=f"smoke-{name}", conversation_id=f"smoke-{name}", snapshot_id="demo-snapshot-v1",
                       message=message, constraints=constraints)
    state = initial_state(req)
    state["history"] = history
    model_request = build_model_request(state, TOOLS, POLICY)
    return {"name": name, "messages": to_chat_messages(model_request),
            "max_tokens": model_request.max_output_tokens}


def evidence_blob(n):
    return " ".join(f"Review {i}: battery lasts about {8 + i % 5} hours, comfortable fit, price stable." for i in range(n))


def main():
    cases = []
    for name, message, constraints in REQUESTS:
        cases.append(build(name, message, constraints, []))
    # second step: after a tool observation
    obs = {"observation": {"status": "ok", "data": {"offer_id": "demo-offer-001", "title": "Synthetic commuting earbuds",
           "price": "49.99", "currency": "USD", "evidence_id": "demo-evidence-001", "synthetic": True}}}
    first = {"type": "call_tool", "tool_name": "search_products", "arguments": {"query": "wireless commuting earbuds"}}
    hist = [{"role": "assistant", "content": json.dumps(first)},
            {"role": "user", "content": json.dumps(obs, ensure_ascii=False)}]
    cases.append(build("zh-after-observation", REQUESTS[0][1], REQUESTS[0][2], hist))
    # long observation (several thousand tokens) to exercise prefill
    long_obs = {"observation": {"status": "ok", "data": {"offer_id": "demo-offer-001", "evidence_id": "demo-evidence-001",
                "excerpts": evidence_blob(60)}}}
    hist2 = [{"role": "assistant", "content": json.dumps(first)},
             {"role": "user", "content": json.dumps(long_obs, ensure_ascii=False)}]
    cases.append(build("en-long-observation", REQUESTS[1][1], REQUESTS[1][2], hist2))
    fixture = {"action_schema": ACTION_ADAPTER.json_schema(), "cases": cases}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    sizes = [len(json.dumps(c["messages"], ensure_ascii=False)) for c in cases]
    print(f"wrote {OUT}: {len(cases)} cases, prompt chars {min(sizes)}..{max(sizes)}")


if __name__ == "__main__":
    main()

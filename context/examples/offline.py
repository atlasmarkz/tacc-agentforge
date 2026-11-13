"""Run with python -m context.examples.offline; no model, credentials or network needed."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from context import ContextManager, ContextPatch, ContextRequest, HistoryBlock, InMemoryStore, SQLiteStore
from context.agent_adapter import history_block_from_agent


def demonstrate(store):
    manager = ContextManager(store)
    state = manager.update_memory("demo", ContextPatch(set={"purchase_cap": "1000.00",
        "currency": "HKD", "requirements": ["通勤", "降噪"]}), snapshot_id="snapshot-1",
        operation_id="create", expected_revision=0)
    pair = [{"role": "assistant", "content": json.dumps({"type": "call_tool",
        "tool_name": "get_offer_facts", "arguments": {"offer_ids": ["offer-1"]}})},
        {"role": "user", "content": json.dumps({"observation": {"status": "ok",
        "tool": "get_offer_facts", "snapshot_id": "snapshot-1", "operation_id": "run-1:tool:1", "data": {"offers": [{
            "offer_id": "offer-1", "currency": "HKD", "price_minor": 90000,
            "shipping": {"status": "unknown"}, "evidence_ids": ["ev-1"]}]}}})}]
    block = history_block_from_agent(pair, block_id="facts-1", run_id="run-1",
        snapshot_id="snapshot-1", constraints_revision=state["constraints_revision"])
    state = manager.append_history("demo", block, operation_id="facts", expected_revision=state["revision"])
    # Synthetic proposal reference, not a claim that fees or affordability were verified.
    state = manager.append_history("demo", HistoryBlock("plan-1", "run-1", "snapshot-1",
        state["constraints_revision"], [{"role": "assistant", "content": "Historical proposal"}],
        kind="plan", plan_items=[{"offer_id": "offer-1", "quantity": 1, "evidence_ids": ["ev-1"]}]),
        operation_id="plan", expected_revision=state["revision"])
    state = manager.update_memory("demo", ContextPatch(set={"purchase_cap": "800.00"}),
        snapshot_id="snapshot-1", operation_id="revise", expected_revision=state["revision"])
    request = ContextRequest("demo", "run-2", "snapshot-1", "预算降到800，其他不变", "Shopping advisor",
                             [], {}, 16384, 512)
    before = manager.build(request)
    # Read another offer from the same immutable snapshot; do not change archived prices.
    pair[0]["content"] = pair[0]["content"].replace("offer-1", "offer-2")
    pair[1]["content"] = (pair[1]["content"].replace("offer-1", "offer-2")
        .replace("ev-1", "ev-2").replace("90000", "75000").replace("run-1:tool:1", "run-2:tool:1"))
    fresh = history_block_from_agent(pair, block_id="facts-2", run_id="run-2",
        snapshot_id="snapshot-1", constraints_revision=state["constraints_revision"])
    manager.append_history("demo", fresh, operation_id="refresh", expected_revision=state["revision"])
    results = {mode: manager.build(replace(request, mode=mode)) for mode in ("filtered", "unfiltered")}
    print(json.dumps({"constraints": state["constraints"], "old_history": before.omitted,
        "modes": {mode: {"selected": b.selected_block_ids, "tokens": b.token_count.value,
                          "count_mode": b.token_count.mode} for mode, b in results.items()}}, ensure_ascii=False))


def main():
    demonstrate(InMemoryStore())
    with TemporaryDirectory() as directory:
        demonstrate(SQLiteStore(Path(directory) / "context.sqlite"))


if __name__ == "__main__":
    main()

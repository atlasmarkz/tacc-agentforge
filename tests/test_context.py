"""Shared storage contracts and semantic prompt-selection scenarios."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from context import (ContextError, ContextLimitError, ContextManager, ContextPatch, ContextRequest,
                     HistoryBlock, InMemoryStore, SQLiteStore, StorageError, TokenCount)
from context.agent_adapter import history_block_from_agent, request_from_agent, to_model_request


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    return InMemoryStore() if request.param == "memory" else SQLiteStore(tmp_path / "memory.sqlite")


def create(manager, conversation="c", **constraints):
    return manager.update_memory(conversation, ContextPatch(set=constraints), snapshot_id="s",
                                 operation_id="init", expected_revision=0)


def add(manager, name, content, *, run="r", kind="feedback", plan_items=None):
    state = manager.get("c")
    return manager.append_history("c", HistoryBlock(name, run, state["snapshot_id"],
        state["constraints_revision"], [{"role": "user", "content": content}], kind=kind,
        plan_items=plan_items or []), operation_id=name, expected_revision=state["revision"])


def request(**kwargs):
    return replace(ContextRequest("c", "r", "s", "通勤耳机", "Advisor", [], {}, 16384, 512), **kwargs)


def update(manager, patch, *, snapshot="s", op="update"):
    return manager.update_memory("c", patch, snapshot_id=snapshot, operation_id=op,
                                 expected_revision=manager.get("c")["revision"])


def test_patch_preserves_unchanged_fields_and_detaches_data(store):
    manager = ContextManager(store)
    original = {"purchase_cap": "1000", "currency": "HKD", "requirements": ["通勤", "降噪"]}
    create(manager, **original)
    original["requirements"].clear()
    state = update(manager, ContextPatch(set={"purchase_cap": "800", "optional": None}, unset=("currency",)))
    assert state["constraints"] == {"purchase_cap": "800", "requirements": ["通勤", "降噪"], "optional": None}
    state["constraints"]["requirements"].clear()
    assert manager.get("c")["constraints"]["requirements"] == ["通勤", "降噪"]
    revision = manager.get("c")["constraints_revision"]
    state = update(manager, ContextPatch(set={"purchase_cap": "800"}, unset=("absent",)), op="noop")
    assert state["constraints_revision"] == revision


@pytest.mark.parametrize("patch", [lambda: ContextPatch(set={"a": 1}, unset=("a",)),
    lambda: ContextPatch(set={"run_id": "evil"}), lambda: ContextPatch(unset=("a", "a")),
    lambda: ContextPatch(set={"x": float("nan")}), lambda: ContextPatch(set={"x": {1: "bad"}})])
def test_invalid_patch(patch):
    with pytest.raises(ValueError):
        patch()


def test_idempotency_conflicts_and_isolation(store):
    manager = ContextManager(store)
    state = create(manager, currency="HKD")
    update(manager, ContextPatch(set={"purchase_cap": "800"}))
    assert create(manager, currency="HKD") == state  # replay precedes revision check
    with pytest.raises(ContextError, match="different content"):
        create(manager, currency="USD")
    with pytest.raises(ContextError) as error:
        manager.update_memory("c", ContextPatch(), snapshot_id="s", operation_id="stale", expected_revision=1)
    assert error.value.code == "revision_conflict"
    create(manager, "other", currency="USD")
    assert manager.get("c")["constraints"]["currency"] == "HKD"
    assert manager.get("other")["constraints"]["currency"] == "USD"


def test_append_retry_delete_and_block_identity(store):
    manager = ContextManager(store)
    state = create(manager)
    block = HistoryBlock("b", "r", "s", 0, [{"role": "user", "content": "hello"}])
    first = manager.append_history("c", block, operation_id="b", expected_revision=state["revision"])
    assert manager.append_history("c", block, operation_id="b", expected_revision=1) == first
    with pytest.raises(ContextError) as error:
        manager.append_history("c", block, operation_id="another", expected_revision=2)
    assert error.value.code == "block_exists"
    manager.delete("c", operation_id="delete", expected_revision=2)
    assert manager.get("c") is None
    manager.delete("c", operation_id="delete", expected_revision=2)
    assert create(manager)["revision"] == 1


def test_sqlite_restart_and_transaction_rollback(tmp_path):
    path = tmp_path / "memory.sqlite"
    manager = ContextManager(SQLiteStore(path))
    state = create(manager, currency="HKD")
    assert ContextManager(SQLiteStore(path)).get("c") == state
    # Fail after the session write, before its receipt: the entire transaction must roll back.
    with sqlite3.connect(path) as db:
        db.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON receipts BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(StorageError):
        update(manager, ContextPatch(set={"purchase_cap": "800"}))
    assert manager.get("c") == state
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER reject_receipt")
    assert update(manager, ContextPatch(set={"purchase_cap": "800"}))["revision"] == 2


def test_sqlite_concurrent_instances_do_not_lose_updates(tmp_path):
    path = tmp_path / "memory.sqlite"
    managers = [ContextManager(SQLiteStore(path)) for _ in range(2)]
    create(managers[0])
    def write(index):
        try:
            managers[index].update_memory("c", ContextPatch(set={"winner": index}), snapshot_id="s",
                                          operation_id=f"write-{index}", expected_revision=1)
            return "ok"
        except ContextError as exc:
            return exc.code
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(write, range(2))) == ["ok", "revision_conflict"]
    assert managers[0].get("c")["revision"] == 2


def test_storage_errors_are_explicit(tmp_path):
    with pytest.raises(StorageError):
        SQLiteStore(tmp_path / "missing" / "memory.sqlite")
    path = tmp_path / "bad.sqlite"
    path.write_bytes(b"not a database")
    with pytest.raises(StorageError):
        SQLiteStore(path)


def test_unknown_database_version_and_corrupt_json_are_rejected(tmp_path):
    path = tmp_path / "version.sqlite"
    manager = ContextManager(SQLiteStore(path))
    create(manager)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE sessions SET state='broken-json'")
    with pytest.raises(StorageError):
        manager.get("c")
    with pytest.raises(StorageError):
        manager.update_memory("c", ContextPatch(), snapshot_id="s", operation_id="broken", expected_revision=1)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE context_meta SET version=99")
    with pytest.raises(StorageError, match="version"):
        SQLiteStore(path)


def test_stale_append_rejected_without_consuming_revision(store):
    manager = ContextManager(store)
    create(manager)
    update(manager, ContextPatch(set={"purchase_cap": "800"}))
    for snapshot, revision in [("s", 0), ("wrong", 1)]:
        block = HistoryBlock("stale", "r", snapshot, revision, [{"role": "user", "content": "old"}])
        with pytest.raises(ContextError) as error:
            manager.append_history("c", block, operation_id="stale", expected_revision=2)
        assert error.value.code == "stale_history"
        assert manager.get("c")["revision"] == 2


def test_custom_renderer_and_counter_errors_are_not_hidden(store):
    from context import render_messages
    def render(payload):
        result = render_messages(payload)
        result[0]["content"] += "\ncustom-template-marker"
        return result
    def counter(messages):
        assert messages[0]["content"].endswith("custom-template-marker")
        return TokenCount(1, "exact", "custom")
    manager = ContextManager(store, renderer=render, token_counter=counter)
    create(manager)
    assert manager.build(request()).token_count.counter_id == "custom"
    def fail(messages):
        raise RuntimeError("tokenizer missing")
    manager.token_counter = fail
    with pytest.raises(RuntimeError, match="tokenizer missing"):
        manager.build(request())


def test_invalidation_and_snapshot_roundtrip(store):
    manager = ContextManager(store)
    create(manager, purchase_cap="1000")
    add(manager, "old", "old facts and passed validation")
    update(manager, ContextPatch(set={"purchase_cap": "800"}))
    bundle = manager.build(request())
    assert bundle.omitted == {"old": "requirements_changed"}
    assert "passed validation" not in json.dumps(bundle.payload)
    add(manager, "current", "new facts")
    update(manager, ContextPatch(), snapshot="s2", op="switch")
    update(manager, ContextPatch(), snapshot="s", op="switch-back")
    bundle = manager.build(request())
    assert not bundle.selected_block_ids  # returning to a snapshot does not revive old state
    assert set(bundle.omitted.values()) == {"snapshot_changed"}
    with pytest.raises(ContextError, match="Switch"):
        manager.build(request(snapshot_id="other"))


def test_new_run_gets_only_plan_ids_and_no_pending_question(store):
    manager = ContextManager(store)
    create(manager)
    add(manager, "facts", '{"candidate_set_id":"cand","price":10}')
    add(manager, "plan", "PASSED old total 100", kind="plan",
        plan_items=[{"offer_id": "offer", "quantity": 2, "evidence_ids": ["ev"]}])
    add(manager, "question", '{"question_id":"old-question","required_fields":["budget"]}', kind="question")
    bundle = manager.build(request(run_id="new-run"))
    text = json.dumps(bundle.payload)
    assert "historical_plan_reference" in text and "offer" in text
    assert "PASSED" not in text and "old-question" not in text and "candidate_set_id" not in text
    assert bundle.selected_block_ids == ["plan"]
    assert bundle.selection_reasons["plan"] == "historical_plan_reference"
    assert "plan" not in bundle.omitted


def test_restarted_manager_excludes_candidate_handles(store):
    manager = ContextManager(store)
    create(manager)
    add(manager, "handle", '{"candidate_set_id":"cand","filters":{"category":"earbuds"}}')
    assert manager.build(request()).selected_block_ids == ["handle"]
    restarted = ContextManager(store).build(request())
    assert restarted.omitted == {"handle": "needs_search"}
    assert "cand" not in json.dumps(restarted.payload)
    assert "filters" in manager.get("c")["history"][0]["messages"][0]["content"]


def test_atomic_observation_and_unknown_facts_preserved(store):
    manager = ContextManager(store)
    create(manager)
    messages = [{"role": "assistant", "content": '{"type":"call_tool","tool_name":"get_offer_facts","arguments":{"offer_ids":["offer"]}}'},
                {"role": "user", "content": '{"observation":{"status":"ok","tool":"get_offer_facts","snapshot_id":"s","operation_id":"r:tool:1","data":{"offer_id":"offer","shipping":"unknown","conflict":true,"truncated":true,"observed_at":"2026-01-01","evidence_id":"ev"}}}'}]
    block = history_block_from_agent(messages, block_id="pair", run_id="r", snapshot_id="s", constraints_revision=0)
    manager.append_history("c", block, operation_id="pair", expected_revision=1)
    bundle = manager.build(request())
    assert bundle.payload["messages"][-2:] == messages
    assert set(block.references) == {"offer", "ev"}
    with pytest.raises(ContextLimitError):
        manager.build(request(model_window_tokens=100))
    assert len(manager.get("c")["history"]) == 1  # build has no writes
    with pytest.raises(ValueError, match="require"):
        history_block_from_agent(messages[:1], block_id="bad", run_id="r", snapshot_id="s", constraints_revision=0)


def test_filtered_and_unfiltered_share_budget_and_skip_large_blocks(store):
    # A deterministic test counter measures only history sizes plus fixed required context.
    def counter(messages):
        return TokenCount(100 + sum(len(m["content"]) for m in messages[2:]), "exact", "fixture-v1")
    manager = ContextManager(store, token_counter=counter)
    create(manager)
    add(manager, "relevant", "通勤耳机")
    add(manager, "huge", "不相关" * 100)
    add(manager, "recent", "桌面键盘")
    add(manager, "latest", "反馈")
    req = request(model_window_tokens=107, max_output_tokens=1, safety_tokens=0)
    filtered = manager.build(req)
    recent = manager.build(replace(req, mode="unfiltered"))
    assert filtered.selected_block_ids == ["relevant", "latest"]
    assert recent.selected_block_ids == ["recent", "latest"]
    assert filtered.omitted["huge"] == recent.omitted["huge"] == "budget"
    assert filtered.token_count.value == recent.token_count.value == 106
    assert manager.build(req) == filtered


def test_explicit_reference_priority(store):
    def counter(messages):
        return TokenCount(100 + 10 * (len(messages) - 2), "exact", "fixture")
    manager = ContextManager(store, token_counter=counter)
    create(manager)
    state = manager.get("c")
    block = HistoryBlock("reference", "r", "s", 0, [{"role": "user", "content": "old"}], references=("offer-1",))
    manager.append_history("c", block, operation_id="ref", expected_revision=state["revision"])
    add(manager, "match", "通勤耳机")
    add(manager, "latest", "last")
    bundle = manager.build(request(references=("offer-1",), model_window_tokens=121,
                                   max_output_tokens=1, safety_tokens=0))
    assert bundle.selected_block_ids == ["reference", "latest"]


def test_final_rendering_matches_inference_and_counts_schema(store):
    pytest.importorskip("pydantic")
    from inference.model_port import to_chat_messages
    from agent.contracts import AgentRequest
    manager = ContextManager(store)
    create(manager, currency="HKD")
    req = request_from_agent(AgentRequest(run_id="r", conversation_id="c", snapshot_id="s", message="买耳机"),
        system_prompt="Advisor", tools=[{"name": "search_products", "description": "商品查询",
            "parameters": {"type": "object", "properties": {"filters": {"type": "object"}}}}],
        action_schema={"description": "中文" * 1000},
        model_window_tokens=16384, max_output_tokens=512)
    bundle = manager.build(req)
    assert to_chat_messages(to_model_request(bundle)) == bundle.rendered_messages
    assert bundle.token_count.mode == "estimated"
    assert bundle.token_count.value > 6000
    with pytest.raises(ContextLimitError):
        manager.build(replace(req, model_window_tokens=4096))


def test_exact_counter_receives_final_schema_and_keeps_user_text_out_of_system(store):
    received = []
    def counter(messages):
        received.append(messages)
        assert "Action JSON Schema" in messages[0]["content"]
        return TokenCount(42, "exact", "model-template-revision")
    manager = ContextManager(store, token_counter=counter)
    create(manager, preference="INJECTED USER DATA")
    bundle = manager.build(request(model_window_tokens=43, max_output_tokens=1, safety_tokens=0))
    assert bundle.token_count.value == 42
    assert received and "INJECTED USER DATA" not in bundle.payload["messages"][0]["content"]
    assert "INJECTED USER DATA" in bundle.payload["messages"][1]["content"]


@pytest.mark.parametrize("bad", ["brand", {"brand": True}, 42, None])
def test_string_or_nonsequence_fields_cannot_be_split_into_characters(bad):
    with pytest.raises(ValueError):
        ContextPatch(unset=bad)
    with pytest.raises(ValueError):
        HistoryBlock("b", "r", "s", 0, [{"role": "user", "content": "hello"}], references=bad)
    with pytest.raises(ValueError):
        request(references=bad)


@pytest.mark.parametrize("content", [r'{"candidate_\u0073et_id":"old-candidate"}',
    r'{"observation":{"data":[{"candidate\u005fset_id":"old-candidate"}]}}'])
def test_escaped_json_candidate_keys_expire_on_restart(store, content):
    manager = ContextManager(store)
    create(manager)
    add(manager, "escaped", content)
    assert manager.build(request()).selected_block_ids == ["escaped"]
    bundle = ContextManager(store).build(request())
    assert bundle.omitted == {"escaped": "needs_search"}
    assert not bundle.selected_block_ids


def test_large_historical_plan_is_optional_and_diagnosed(store):
    manager = ContextManager(store)
    create(manager)
    req = request(run_id="new", model_window_tokens=1000, max_output_tokens=100, safety_tokens=0)
    baseline = manager.build(req)
    add(manager, "old-plan", "old proposal", kind="plan", plan_items=[
        {"offer_id": f"offer-{i}", "quantity": 1, "evidence_ids": [f"ev-{i}"]} for i in range(20)])
    bundle = manager.build(req)
    assert bundle.payload == baseline.payload
    assert bundle.omitted == {"old-plan": "budget"}


@pytest.mark.parametrize("observation", [None, [], 42, {}, {"status": "ok"},
    {"status": "error"}, {"status": "ok", "tool": "search_products", "snapshot_id": "s",
        "operation_id": "r:tool:1", "data": {"snapshot_id": "foreign"}}])
def test_incomplete_or_foreign_tool_observations_are_rejected(observation):
    messages = [{"role": "assistant", "content": json.dumps({"type": "call_tool",
        "tool_name": "search_products", "arguments": {}})},
        {"role": "user", "content": json.dumps({"observation": observation})}]
    with pytest.raises(ValueError):
        history_block_from_agent(messages, block_id="b", run_id="r", snapshot_id="s", constraints_revision=0)


def test_recoverable_agent_error_without_tool_identity_is_supported():
    messages = [{"role": "assistant", "content": json.dumps({"type": "call_tool",
        "tool_name": "search_products", "arguments": {}})},
        {"role": "user", "content": json.dumps({"observation": {
            "status": "error", "code": "call_timeout", "message": "Try again."}})}]
    block = history_block_from_agent(messages, block_id="b", run_id="r", snapshot_id="s", constraints_revision=0)
    assert block.messages == messages


@pytest.mark.parametrize("corruption", ["[]", "null", "{}", '{"schema_version":1,"history":{}}'])
def test_valid_json_with_invalid_state_shape_is_a_storage_error(tmp_path, corruption):
    path = tmp_path / "corrupt.sqlite"
    manager = ContextManager(SQLiteStore(path))
    create(manager)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE sessions SET state=?", (corruption,))
    with pytest.raises(StorageError):
        manager.get("c")
    with pytest.raises(StorageError):
        manager.update_memory("c", ContextPatch(), snapshot_id="s", operation_id="update", expected_revision=1)


@pytest.mark.parametrize("receipt", ["{}", "[]", "null", '{"digest":null}'])
def test_corrupt_receipt_shape_is_a_storage_error(tmp_path, receipt):
    path = tmp_path / "receipt.sqlite"
    manager = ContextManager(SQLiteStore(path))
    create(manager)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE receipts SET receipt=?", (receipt,))
    with pytest.raises(StorageError):
        create(manager)


def test_historical_receipt_rebuilds_original_state_after_many_changes(store):
    manager = ContextManager(store)
    initial = create(manager, purchase_cap="1000")
    first = add(manager, "first", "original facts")
    block = HistoryBlock("first", "r", "s", 1, [{"role": "user", "content": "original facts"}], kind="feedback")
    update(manager, ContextPatch(set={"purchase_cap": "800"}))
    update(manager, ContextPatch(), snapshot="s2", op="snapshot")
    add(manager, "second", "other facts")
    assert manager.append_history("c", block, operation_id="first", expected_revision=initial["revision"]) == first
    assert create(manager, purchase_cap="1000") == initial


def test_receipts_do_not_duplicate_cumulative_history(tmp_path):
    path = tmp_path / "bounded.sqlite"
    manager = ContextManager(SQLiteStore(path))
    create(manager)
    for index in range(50):
        add(manager, f"b-{index}", "x" * 8192)
    with sqlite3.connect(path) as db:
        state_bytes = db.execute("SELECT length(state) FROM sessions").fetchone()[0]
        receipt_bytes = db.execute("SELECT sum(length(receipt)) FROM receipts").fetchone()[0]
    assert receipt_bytes < state_bytes


@pytest.mark.parametrize("key,value", [("conversation_id", "foreign"), ("revision", 1000),
    ("constraints_revision", 1000)])
def test_receipt_identity_and_version_corruption_is_rejected(tmp_path, key, value):
    path = tmp_path / "identity.sqlite"
    manager = ContextManager(SQLiteStore(path))
    create(manager)
    with sqlite3.connect(path) as db:
        receipt = json.loads(db.execute("SELECT receipt FROM receipts").fetchone()[0])
        receipt["result"][key] = value
        db.execute("UPDATE receipts SET receipt=?", (json.dumps(receipt),))
    with pytest.raises(StorageError):
        create(manager)


def test_historical_plan_ranking_uses_only_visible_item_references(store):
    def counter(messages):
        return TokenCount(100 + 10 * (len(messages) - 2), "exact", "fixture")
    manager = ContextManager(store, token_counter=counter)
    create(manager)
    add(manager, "plan", "discarded raw plan text", run="old", kind="plan", plan_items=[
        {"offer_id": "offer-target", "quantity": 1, "evidence_ids": ["ev-target"]}])
    add(manager, "match", "通勤耳机")
    add(manager, "latest", "last")
    bundle = manager.build(request(references=("offer-target",), model_window_tokens=121,
                                   max_output_tokens=1, safety_tokens=0))
    assert bundle.selected_block_ids == ["plan", "latest"]
    assert "discarded raw plan text" not in json.dumps(bundle.payload)

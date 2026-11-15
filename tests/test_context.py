"""Shared storage contracts and whole-trace prompt-construction scenarios."""
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from context import (ContextError, ContextLimitError, ContextManager, ContextPatch, ContextRequest,
                     HistoryBlock, InMemoryStore, SQLiteStore, StorageError, TokenCount,
                     conservative_estimate_tokens, estimate_tokens)
from context.agent_adapter import (history_block_from_agent, request_from_agent, to_model_request,
                                   validated_plan_block_from_agent)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    return InMemoryStore() if request.param == "memory" else SQLiteStore(tmp_path / "memory.sqlite")


def create(manager, conversation="c", **constraints):
    return manager.update_memory(conversation, ContextPatch(set=constraints), snapshot_id="s",
                                 operation_id="init", expected_revision=0)


def add(manager, name, content, *, run="r", kind="feedback", plan_items=None,
        plan_validation="unvalidated"):
    state = manager.get("c")
    return manager.append_history("c", HistoryBlock(name, run, state["snapshot_id"],
        state["constraints_revision"], [{"role": "user", "content": content}], kind=kind,
        plan_items=plan_items or [], plan_validation=plan_validation),
        operation_id=name, expected_revision=state["revision"])


def request(**kwargs):
    return replace(ContextRequest("c", "r", "s", "通勤耳机", "Advisor", [], {}, 16384, 512), **kwargs)


def update(manager, patch, *, snapshot="s", op="update"):
    return manager.update_memory("c", patch, snapshot_id=snapshot, operation_id=op,
                                 expected_revision=manager.get("c")["revision"])


def plan_item(offer="offer", quantity=1, evidence=("ev",)):
    return {"offer_id": offer, "quantity": quantity, "evidence_ids": list(evidence)}


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


def test_new_run_gets_only_passed_plan_ids_and_no_pending_question(store):
    manager = ContextManager(store)
    create(manager)
    add(manager, "facts", '{"candidate_set_id":"cand","price":10}')
    add(manager, "plan", "PASSED old total 100", kind="plan", plan_validation="passed",
        plan_items=[{"offer_id": "offer", "quantity": 2, "evidence_ids": ["ev"]}])
    add(manager, "question", '{"question_id":"old-question","required_fields":["budget"]}', kind="question")
    bundle = manager.build(request(run_id="new-run"))
    text = json.dumps(bundle.payload)
    assert "historical_plan_reference" in text and "offer" in text
    assert "PASSED" not in text and "old-question" not in text and "candidate_set_id" not in text
    assert bundle.selected_block_ids == ["plan"]
    assert bundle.selection_reasons["plan"] == "historical_plan_reference"
    assert "plan" not in bundle.omitted


@pytest.mark.parametrize("validation", ["unvalidated", "passed"])
def test_only_passed_plan_blocks_become_cross_run_references(store, validation):
    manager = ContextManager(store)
    create(manager)
    add(manager, "old-plan", "raw plan text", run="old", kind="plan",
        plan_validation=validation, plan_items=[plan_item("offer-old")])
    bundle = manager.build(request(run_id="new", model_window_tokens=1000,
                                   max_output_tokens=1, safety_tokens=0))
    made_reference = "historical_plan_reference" in json.dumps(bundle.payload)
    assert made_reference is (validation == "passed")
    assert "raw plan text" not in json.dumps(bundle.payload)
    if validation == "passed":
        assert bundle.selection_reasons["old-plan"] == "historical_plan_reference"
    else:
        assert bundle.omitted["old-plan"] == "different_run"
        assert "offer-old" not in json.dumps(bundle.payload)


def test_passed_plan_without_items_is_not_a_reference(store):
    manager = ContextManager(store)
    create(manager)
    add(manager, "empty-plan", "passing text", run="old", kind="plan", plan_validation="passed")
    bundle = manager.build(request(run_id="new", model_window_tokens=1000,
                                   max_output_tokens=1, safety_tokens=0))
    assert "historical_plan_reference" not in json.dumps(bundle.payload)
    assert bundle.selected_block_ids == []


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


def content_counter(messages):
    return TokenCount(sum(len(m["content"]) for m in messages), "exact", "content-bytes")


def test_middle_large_block_cannot_create_a_current_trace_gap(store):
    manager = ContextManager(store, token_counter=content_counter)
    create(manager)
    sizes = [("b0", 100), ("b1", 100), ("huge", 3000), ("b3", 100), ("b4", 100)]
    for name, size in sizes:
        add(manager, name, name + "x" * size)
    tight = request(model_window_tokens=1000, max_output_tokens=1, safety_tokens=0)
    # Omitting the middle block would fit, but that would hide a dependency and is forbidden.
    for mode in ("filtered", "unfiltered"):
        with pytest.raises(ContextLimitError):
            manager.build(replace(tight, mode=mode))
    for mode in ("filtered", "unfiltered"):
        bundle = manager.build(replace(tight, mode=mode, model_window_tokens=6000))
        assert bundle.selected_block_ids == [name for name, _ in sizes]
        assert bundle.omitted == {}
        assert bundle.token_count.value == content_counter(bundle.rendered_messages).value


def test_early_search_and_final_observation_are_both_required(store):
    manager = ContextManager(store, token_counter=content_counter)
    create(manager)
    add(manager, "search", '{"candidate_set_id":"candidate","filters":{"category":"earbuds"}}')
    add(manager, "facts", '{"observation":{"data":{"price":10}}}')
    add(manager, "latest", '{"status":"validation_failed","issues":[]}')
    bundle = manager.build(request(model_window_tokens=6000, max_output_tokens=1, safety_tokens=0))
    assert bundle.selected_block_ids == ["search", "facts", "latest"]
    assert "candidate" in json.dumps(bundle.payload)
    assert "validation_failed" in json.dumps(bundle.payload)


def test_filtered_and_unfiltered_are_identical_in_this_version(store):
    manager = ContextManager(store)
    create(manager)
    add(manager, "old-plan", "old", run="old", kind="plan", plan_validation="passed",
        plan_items=[plan_item("offer-old")])
    add(manager, "first", "通勤耳机")
    add(manager, "second", "桌面键盘")
    req = request(run_id="r", model_window_tokens=4096)
    filtered = manager.build(req)
    unfiltered = manager.build(replace(req, mode="unfiltered"))
    assert filtered.selected_block_ids == unfiltered.selected_block_ids
    assert filtered.omitted == unfiltered.omitted
    assert filtered.payload == unfiltered.payload
    assert filtered.token_count == unfiltered.token_count
    assert filtered.selected_block_ids == ["old-plan", "first", "second"]


def test_explicit_references_are_metadata_only(store):
    manager = ContextManager(store)
    create(manager)
    state = manager.get("c")
    manager.append_history("c", HistoryBlock("o1", "r", "s", 0,
        [{"role": "user", "content": "first"}], references=("o1",)),
        operation_id="o1", expected_revision=state["revision"])
    add(manager, "o10", "second")
    plain = manager.build(request(model_window_tokens=4096, message="o10"))
    with_refs = manager.build(request(model_window_tokens=4096, message="o10",
                                      references=("o1",)))
    assert with_refs.selected_block_ids == plain.selected_block_ids == ["o1", "o10"]
    assert with_refs.payload == plain.payload


def test_optional_plan_is_dropped_but_full_required_trace_remains(store):
    def counter(messages):
        return TokenCount(100 + 10 * (len(messages) - 2), "exact", "fixture")
    manager = ContextManager(store, token_counter=counter)
    create(manager)
    add(manager, "early", "search")
    add(manager, "latest", "observation")
    add(manager, "old-plan", "old", run="old", kind="plan", plan_validation="passed",
        plan_items=[plan_item("offer-old")])
    bundle = manager.build(request(model_window_tokens=121, max_output_tokens=1, safety_tokens=0))
    assert bundle.selected_block_ids == ["early", "latest"]
    assert bundle.omitted == {"old-plan": "budget"}
    assert bundle.selection_reasons == {"early": "current_run_required",
                                        "latest": "current_run_required"}


def test_required_trace_over_limit_stops_instead_of_skipping(store):
    def counter(messages):
        return TokenCount(100 + 10 * (len(messages) - 2), "exact", "fixture")
    manager = ContextManager(store, token_counter=counter)
    create(manager)
    add(manager, "early", "search")
    add(manager, "latest", "observation")
    with pytest.raises(ContextLimitError):
        manager.build(request(model_window_tokens=120, max_output_tokens=1, safety_tokens=0))


def test_counter_is_called_once_without_optional_content(store):
    calls = []
    def counter(messages):
        calls.append(messages)
        return TokenCount(100 + 10 * (len(messages) - 2), "exact", "fixture")
    manager = ContextManager(store, token_counter=counter)
    create(manager)
    add(manager, "early", "search")
    add(manager, "latest", "observation")
    bundle = manager.build(request(model_window_tokens=1000))
    assert len(calls) == 1
    assert bundle.token_count.value == 120


def test_counter_is_called_at_most_twice_when_dropping_optional_plan(store):
    calls, renders = [], []
    def counter(messages):
        calls.append(messages)
        return TokenCount(100 + 10 * (len(messages) - 2), "exact", "fixture")
    from context import render_messages
    def render(payload):
        renders.append(payload)
        return render_messages(payload)
    manager = ContextManager(store, renderer=render, token_counter=counter)
    create(manager)
    add(manager, "early", "search")
    add(manager, "latest", "observation")
    add(manager, "old-plan", "old", run="old", kind="plan", plan_validation="passed",
        plan_items=[plan_item("offer-old")])
    # The optional plan fits: one full count.
    bundle = manager.build(request(model_window_tokens=131, max_output_tokens=1, safety_tokens=0))
    assert bundle.selected_block_ids == ["early", "latest", "old-plan"]
    assert len(calls) == len(renders) == 1
    calls.clear()
    renders.clear()
    # The optional plan does not fit: exactly one count with it and one without it.
    bundle = manager.build(request(model_window_tokens=121, max_output_tokens=1, safety_tokens=0))
    assert bundle.selected_block_ids == ["early", "latest"]
    assert len(calls) == len(renders) == 2
    assert bundle.token_count.value == 100 + 10 * (len(calls[-1]) - 2)


def test_whole_prompt_counter_is_not_treated_as_a_message_sum(store):
    def counter(messages):
        return TokenCount(len(json.dumps(messages, ensure_ascii=False)) * 2,
                          "exact", "whole-prompt")
    manager = ContextManager(store, token_counter=counter)
    create(manager)
    add(manager, "early", "search")
    add(manager, "latest", "observation")
    probe = manager.build(request(model_window_tokens=100000))
    expected = counter(probe.rendered_messages).value
    assert probe.token_count.value == expected
    assert manager.build(request(model_window_tokens=expected + 768,
                                 max_output_tokens=512, safety_tokens=256)).token_count.value == expected
    with pytest.raises(ContextLimitError):
        manager.build(request(model_window_tokens=expected + 767,
                              max_output_tokens=512, safety_tokens=256))


def test_final_rendering_matches_inference_and_counts_schema(store):
    pytest.importorskip("pydantic")
    from inference.model_port import to_chat_messages
    from agent.contracts import AgentRequest
    manager = ContextManager(store)
    state = create(manager, currency="HKD")
    req = request_from_agent(AgentRequest(run_id="r", conversation_id="c", snapshot_id="s", message="买耳机",
        constraints={"currency": "HKD"}),
        expected_revision=state["revision"],
        system_prompt="Advisor", tools=[{"name": "search_products", "description": "商品查询",
            "parameters": {"type": "object", "properties": {"filters": {"type": "object"}}}}],
        action_schema={"description": "中文" * 1000},
        model_window_tokens=16384, max_output_tokens=512)
    bundle = manager.build(req)
    assert to_chat_messages(to_model_request(bundle)) == bundle.rendered_messages
    assert bundle.token_count.mode == "estimated"
    assert bundle.token_count.counter_id == "utf8-bytes-div3-plus16-v1"
    assert bundle.token_count.value == estimate_tokens(bundle.rendered_messages).value
    with pytest.raises(ContextLimitError):
        manager.build(replace(req, model_window_tokens=bundle.token_count.value + 512))


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
    req = request(run_id="new", model_window_tokens=700, max_output_tokens=100, safety_tokens=0)
    baseline = manager.build(req)
    add(manager, "old-plan", "old proposal", kind="plan", plan_validation="passed", plan_items=[
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


def test_historical_reference_uses_only_ids_and_quantities(store):
    def counter(messages):
        return TokenCount(100 + 10 * (len(messages) - 2), "exact", "fixture")
    manager = ContextManager(store, token_counter=counter)
    create(manager)
    add(manager, "plan", "discarded raw plan text", run="old", kind="plan",
        plan_validation="passed", plan_items=[
            {"offer_id": "offer-target", "quantity": 1, "evidence_ids": ["ev-target"]}])
    add(manager, "match", "通勤耳机")
    add(manager, "latest", "last")
    bundle = manager.build(request(references=("offer-target",), model_window_tokens=131,
                                   max_output_tokens=1, safety_tokens=0))
    assert bundle.selected_block_ids == ["plan", "match", "latest"]
    text = json.dumps(bundle.payload)
    assert "discarded raw plan text" not in text
    assert "historical_plan_reference" in text and "offer-target" in text and "ev-target" in text
    assert "total_cost" not in text and "remaining_funds" not in text


def test_history_block_from_agent_never_infers_plan_validation():
    action = {"type": "propose_plan", "plan": {"items": [plan_item()]},
              "validation": "PASSED", "note": "validator says PASSED"}
    messages = [{"role": "assistant", "content": json.dumps(action)},
                {"role": "user", "content": json.dumps({"observation": {"status": "ok"}})}]
    block = history_block_from_agent(messages, block_id="plan", run_id="r",
                                     snapshot_id="s", constraints_revision=0)
    assert block.kind == "plan" and block.plan_validation == "unvalidated"
    manager = ContextManager(InMemoryStore())
    create(manager)
    manager.append_history("c", block, operation_id="plan", expected_revision=1)
    bundle = manager.build(request(run_id="new", model_window_tokens=1000,
                                   max_output_tokens=1, safety_tokens=0))
    assert "historical_plan_reference" not in json.dumps(bundle.payload)


def test_validated_plan_block_requires_a_completed_verified_result():
    completed = {"run_id": "r", "conversation_id": "c", "snapshot_id": "s", "status": "completed",
                 "verified_plan": {"snapshot_id": "s", "currency": "HKD", "total_cost": "10.00",
                                   "remaining_funds": "90.00", "items": [plan_item("offer-1",
                                   evidence=("ev-1",))]}}
    agent_request = {"run_id": "r", "conversation_id": "c", "snapshot_id": "s"}
    block = validated_plan_block_from_agent(completed, agent_request, block_id="passed-plan",
                                             constraints_revision=0)
    assert block.kind == "plan" and block.plan_validation == "passed"
    assert block.plan_items == [plan_item("offer-1", evidence=("ev-1",))]
    assert json.loads(block.messages[0]["content"]) == {
        "type": "propose_plan", "plan": {"items": block.plan_items}}
    assert "total_cost" not in json.dumps(block.messages)

    manager = ContextManager(InMemoryStore())
    create(manager)
    manager.append_history("c", block, operation_id="plan", expected_revision=1)
    bundle = manager.build(request(run_id="new", model_window_tokens=1000,
                                   max_output_tokens=1, safety_tokens=0))
    text = json.dumps(bundle.payload)
    assert "historical_plan_reference" in text and "offer-1" in text
    assert "total_cost" not in text and "remaining_funds" not in text


def test_validated_plan_block_accepts_agent_models():
    pytest.importorskip("pydantic")
    from agent.contracts import AgentRequest, AgentResult, PlanItem, Usage, VerifiedPlan
    result = AgentResult(run_id="r", conversation_id="c", snapshot_id="s", status="completed",
                         verified_plan=VerifiedPlan(snapshot_id="s", currency="HKD",
                             total_cost="10.00", remaining_funds="90.00",
                             items=[PlanItem(offer_id="offer-1", quantity=2,
                                             evidence_ids=["ev-1"])]), usage=Usage())
    agent_request = AgentRequest(run_id="r", conversation_id="c", snapshot_id="s", message="buy")
    block = validated_plan_block_from_agent(result, agent_request, block_id="plan",
                                             constraints_revision=0)
    assert block.plan_validation == "passed" and block.plan_items == [
        {"offer_id": "offer-1", "quantity": 2, "evidence_ids": ["ev-1"]}]


@pytest.mark.parametrize("change", [
    lambda result, request: result.update(status="incomplete"),
    lambda result, request: result.pop("verified_plan"),
    lambda result, request: result.update(verified_plan=None),
    lambda result, request: result.update(run_id="other"),
    lambda result, request: result.update(conversation_id="other"),
    lambda result, request: result.update(snapshot_id="other"),
    lambda result, request: result["verified_plan"].update(snapshot_id="other"),
    lambda result, request: result["verified_plan"].update(items=[]),
    lambda result, request: result["verified_plan"].update(items={}),
    lambda result, request: result["verified_plan"].update(items=["bad"]),
    lambda result, request: result["verified_plan"].update(items=[{"offer_id": "o"}]),
    lambda result, request: result["verified_plan"].update(
        items=[{"offer_id": "o", "quantity": 0, "evidence_ids": ["e"]}]),
    lambda result, request: result["verified_plan"].update(
        items=[{"offer_id": "o", "quantity": 1, "evidence_ids": []}]),
    lambda result, request: request.update(run_id="other"),
])
def test_validated_plan_block_rejects_untrusted_or_malformed_results(change):
    completed = {"run_id": "r", "conversation_id": "c", "snapshot_id": "s",
                 "status": "completed", "verified_plan": {"snapshot_id": "s",
                 "items": [plan_item()]}}
    request_data = {"run_id": "r", "conversation_id": "c", "snapshot_id": "s"}
    change(completed, request_data)
    with pytest.raises(ValueError):
        validated_plan_block_from_agent(completed, request_data, block_id="plan",
                                        constraints_revision=0)


def test_old_append_shape_and_passed_validation_idempotency(store):
    manager = ContextManager(store)
    create(manager)
    items = [plan_item()]
    legacy = HistoryBlock("plan", "r", "s", 0, [{"role": "user", "content": "plan"}],
                          kind="plan", plan_items=items)
    first = manager.append_history("c", legacy, operation_id="plan", expected_revision=1)
    persisted = manager.get("c")["history"][0]
    assert "plan_validation" not in persisted
    assert manager.append_history("c", legacy, operation_id="plan", expected_revision=1) == first
    passed = HistoryBlock("plan", "r", "s", 0, [{"role": "user", "content": "plan"}],
                          kind="plan", plan_items=items, plan_validation="passed")
    with pytest.raises(ContextError, match="different content"):
        manager.append_history("c", passed, operation_id="plan", expected_revision=1)
    state = manager.get("c")
    accepted = HistoryBlock("plan-2", "r", "s", state["constraints_revision"],
                            [{"role": "user", "content": "plan"}], kind="plan",
                            plan_items=items, plan_validation="passed")
    manager.append_history("c", accepted, operation_id="plan-2",
                           expected_revision=state["revision"])
    assert manager.get("c")["history"][1]["plan_validation"] == "passed"


def test_record_without_plan_validation_is_read_as_unvalidated(tmp_path):
    path = tmp_path / "legacy.sqlite"
    manager = ContextManager(SQLiteStore(path))
    create(manager)
    add(manager, "plan", "raw", run="old", kind="plan", plan_validation="passed",
        plan_items=[plan_item("offer-old")])
    with sqlite3.connect(path) as db:
        state = json.loads(db.execute("SELECT state FROM sessions WHERE id='c'").fetchone()[0])
        state["history"][0].pop("plan_validation")
        db.execute("UPDATE sessions SET state=? WHERE id='c'", (json.dumps(state),))
    restarted = ContextManager(SQLiteStore(path))
    bundle = restarted.build(request(run_id="new", model_window_tokens=1000,
                                      max_output_tokens=1, safety_tokens=0))
    assert "historical_plan_reference" not in json.dumps(bundle.payload)
    assert "plan_validation" not in restarted.get("c")["history"][0]


@pytest.mark.parametrize("action", [
    {"type": "propose_plan"},
    {"type": "propose_plan", "plan": None},
    {"type": "propose_plan", "plan": {}},
    {"type": "propose_plan", "plan": {"items": None}},
    {"type": "propose_plan", "plan": {"items": {}}},
    {"type": "propose_plan", "plan": {"items": []}},
    {"type": "propose_plan", "plan": {"items": ["not-an-object"]}},
    {"type": "propose_plan", "plan": {"items": [{"offer_id": "offer"}]}},
    {"type": "propose_plan", "plan": {"items": [
        {"offer_id": "offer", "quantity": 1, "evidence_ids": []}]}},
    {"type": "propose_plan", "plan": {"items": [
        {"offer_id": "offer", "quantity": 0, "evidence_ids": ["ev"]}]}},
])
def test_malformed_propose_plan_raises_value_error(action):
    messages = [{"role": "assistant", "content": json.dumps(action)}]
    with pytest.raises(ValueError):
        history_block_from_agent(messages, block_id="b", run_id="r", snapshot_id="s",
                                 constraints_revision=0)


def test_rejected_output_repair_feedback_is_preserved_verbatim():
    rejected = '{"type":"propose_plan", "plan": null, "note":"keep 空格"}'
    feedback = {"observation": {"status": "error", "code": "invalid_action",
                                "instruction": "Return a corrected action.",
                                "rejected_output": rejected,
                                "rejected_output_truncated": False,
                                "validation_errors": [{"type": "json_invalid", "loc": [],
                                                       "msg": "invalid"}]}}
    messages = [{"role": "assistant", "content": "not an action"},
                {"role": "user", "content": json.dumps(feedback, ensure_ascii=False)}]
    block = history_block_from_agent(messages, block_id="repair", run_id="r",
                                     snapshot_id="s", constraints_revision=0)
    assert block.messages == messages
    assert block.kind == "exchange" and block.plan_items == []
    assert json.loads(block.messages[1]["content"])["observation"]["rejected_output"] == rejected


def test_context_request_binding_fields_are_paired_and_detached():
    for options in ({"expected_revision": 1}, {"expected_constraints": {}}):
        with pytest.raises(ValueError):
            request(**options)
    for options in ({"expected_revision": True, "expected_constraints": {}},
                    {"expected_revision": 1, "expected_constraints": []},
                    {"expected_revision": 1, "expected_constraints": {"x": {1: "bad"}}}):
        with pytest.raises(ValueError):
            request(**options)
    constraints = {"requirements": ["通勤"]}
    req = request(expected_revision=0, expected_constraints=constraints)
    constraints["requirements"].append("降噪")
    assert req.expected_constraints == {"requirements": ["通勤"]}


def test_build_rejects_bound_revision_and_constraint_conflicts(store):
    manager = ContextManager(store)
    state = create(manager, currency="HKD")
    req = request(expected_revision=state["revision"], expected_constraints=state["constraints"])
    assert manager.build(req).constraints == {"currency": "HKD"}
    with pytest.raises(ContextError) as error:
        manager.build(replace(req, expected_revision=state["revision"] - 1))
    assert error.value.code == "revision_conflict"
    with pytest.raises(ContextError) as error:
        manager.build(replace(req, expected_constraints={"currency": "USD"}))
    assert error.value.code == "constraints_mismatch"
    update(manager, ContextPatch(set={"currency": "USD"}))
    with pytest.raises(ContextError) as error:
        manager.build(req)
    assert error.value.code == "revision_conflict"  # never silently drop the bound request


def test_request_from_agent_requires_and_copies_the_bound_request(store):
    pytest.importorskip("pydantic")
    from agent.contracts import AgentRequest
    manager = ContextManager(store)
    state = create(manager, currency="HKD")
    agent = AgentRequest(run_id="r", conversation_id="c", snapshot_id="s", message="买耳机",
                         constraints={"currency": "HKD"})
    with pytest.raises(TypeError):
        request_from_agent(agent, system_prompt="Advisor", tools=[], action_schema={},
                           model_window_tokens=1000, max_output_tokens=10)
    req = request_from_agent(agent, expected_revision=state["revision"],
                             system_prompt="Advisor", tools=[], action_schema={},
                             model_window_tokens=1000, max_output_tokens=10)
    assert req.expected_revision == state["revision"]
    assert req.expected_constraints == {"currency": "HKD"}
    agent.constraints["currency"] = "USD"
    assert req.expected_constraints == {"currency": "HKD"}
    assert manager.build(req).constraints == {"currency": "HKD"}


def test_request_from_agent_conflict_with_storage_is_controlled(store):
    pytest.importorskip("pydantic")
    from agent.contracts import AgentRequest
    manager = ContextManager(store)
    state = create(manager, currency="HKD")
    agent = AgentRequest(run_id="r", conversation_id="c", snapshot_id="s", message="买耳机",
                         constraints={"currency": "USD"})
    req = request_from_agent(agent, expected_revision=state["revision"],
                             system_prompt="Advisor", tools=[], action_schema={},
                             model_window_tokens=1000, max_output_tokens=10)
    with pytest.raises(ContextError) as error:
        manager.build(req)
    assert error.value.code == "constraints_mismatch"
    assert manager.get("c")["constraints"] == {"currency": "HKD"}


def test_sqlite_read_preserves_specific_storage_error_messages(tmp_path):
    path = tmp_path / "specific.sqlite"
    manager = ContextManager(SQLiteStore(path))
    create(manager)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE sessions SET state='{}' WHERE id='c'")
    with pytest.raises(StorageError, match="malformed or unsupported"):
        manager.get("c")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE sessions SET state='not-json' WHERE id='c'")
    with pytest.raises(StorageError):
        manager.get("c")


def test_sqlite_identity_mismatch_keeps_its_message(tmp_path):
    path = tmp_path / "identity.sqlite"
    manager = ContextManager(SQLiteStore(path))
    create(manager)
    create(manager, "source", currency="USD")
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM sessions WHERE id='c'")
        db.execute("UPDATE sessions SET id='c' WHERE id='source'")
    with pytest.raises(StorageError, match="different conversation"):
        manager.get("c")
    with pytest.raises(StorageError, match="different conversation"):
        manager.stats("c")


def test_capacity_stats_are_consistent_and_delete_removes_receipts(tmp_path):
    def populate(manager):
        initial = create(manager, currency="HKD")
        add(manager, "first", "one")
        add(manager, "second", "two")
        return initial

    memory = ContextManager(InMemoryStore())
    memory_initial = populate(memory)
    sqlite = ContextManager(SQLiteStore(tmp_path / "capacity.sqlite"))
    sqlite_initial = populate(sqlite)
    memory_stats = memory.stats("c")
    sqlite_stats = sqlite.stats("c")
    assert memory_stats == sqlite_stats
    assert memory_stats["revision"] == memory.get("c")["revision"] == 3
    assert memory_stats["history_blocks"] == 2
    assert memory_stats["receipt_count"] == 3
    assert memory_stats["state_bytes"] > 0 and memory_stats["receipt_bytes"] > 0
    assert memory.stats("absent") is None
    # Stats are read-only and the earliest receipt still replays its original result.
    assert memory_stats == memory.stats("c")
    assert create(memory, currency="HKD") == memory_initial
    assert sqlite.stats("c")["receipt_count"] == 3
    assert create(sqlite, currency="HKD") == sqlite_initial
    assert sqlite.stats("c")["receipt_count"] == 3

    memory.delete("c", operation_id="delete", expected_revision=memory.get("c")["revision"])
    assert memory.stats("c") is None
    path = tmp_path / "capacity.sqlite"
    sqlite.delete("c", operation_id="delete", expected_revision=sqlite.get("c")["revision"])
    assert sqlite.stats("c") is None
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM receipts").fetchone()[0] == 0


def test_token_estimates_use_documented_formulas():
    messages = [{"role": "system", "content": "abc"}, {"role": "user", "content": "中é"}]
    content_bytes = sum(len(message["content"].encode("utf-8")) for message in messages)
    expected = TokenCount(math.ceil(content_bytes / 3) + 16 * len(messages) + 16,
                          "estimated", "utf8-bytes-div3-plus16-v1")
    assert estimate_tokens(messages) == expected
    assert conservative_estimate_tokens(messages) == TokenCount(
        16 + sum(len(message["content"].encode("utf-8")) + 16 for message in messages),
        "estimated", "utf8-bytes-plus16-v1")


def test_legacy_plan_item_with_empty_evidence_remains_readable(store):
    manager = ContextManager(store)
    create(manager)
    add(manager, "legacy-empty", "old", run="old", kind="plan",
        plan_items=[{"offer_id": "offer", "quantity": 1, "evidence_ids": []}])
    assert manager.get("c")["history"][0]["plan_items"][0]["evidence_ids"] == []
    bundle = manager.build(request(run_id="new", model_window_tokens=1000,
                                   max_output_tokens=1, safety_tokens=0))
    assert "historical_plan_reference" not in json.dumps(bundle.payload)

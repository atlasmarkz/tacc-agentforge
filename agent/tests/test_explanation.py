"""可选解释的事件、预算、回退、取消与规则检查。"""

import asyncio
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from agent import AgentDependencies, AgentResult, AgentRuntime, CapabilityError, ModelReply, RunPolicy
from agent.examples.mock_run import (
    DEMO_EVIDENCE, MockEvidenceResolver, MockModel, MockPlanValidator, demo_request, demo_runtime, demo_tools,
)
from agent.explanation import ExplanationRejected, build_explanation_request, check_explanation
from agent.checkpoints import open_checkpointer

GOOD = "Total 49.99 USD, remaining 50.01 USD. [E1]"


class TextModel(MockModel):
    def __init__(self, text=GOOD, *, error=None, delay=0, finish="stop", streaming=True, cancel=False):
        self.text, self.error, self.delay, self.finish = text, error, delay, finish
        self.streaming, self.cancel = streaming, cancel
        self.requests, self.hooks = [], []

    async def generate(self, request, context):
        if request.action_schema is not None:
            return await super().generate(request, context)
        self.requests.append(request)
        self.hooks.append(context.on_delta)
        if self.streaming:
            context.on_delta("reasoning", "private reasoning")
            for offset in range(0, len(self.text), 3):
                context.on_delta("content", self.text[offset:offset + 3])
                await asyncio.sleep(0)
        if self.cancel:
            context.cancellation.set()
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return ModelReply(content=self.text, finish_reason=self.finish, prompt_tokens=11, completion_tokens=7)


def agent(model, *, policy=None, resolver=None, sink=None):
    return AgentRuntime(AgentDependencies(model, demo_tools(), MockPlanValidator(), sink, resolver),
                        policy or RunPolicy(explain=True, max_model_calls=2))


def collect(runtime, **kwargs):
    async def scenario():
        return [event async for event in runtime.astream(demo_request(), **kwargs)]
    return asyncio.run(scenario())


def test_explanation_stream_is_ordered_and_usage_has_a_separate_call_budget():
    model = TextModel()
    events = collect(agent(model, resolver=MockEvidenceResolver()))
    result = events[-1].result
    assert result.status == "completed" and result.explanation.verified and not result.explanation.fallback
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))
    kinds = [e.type for e in events]
    assert kinds.index("validation_finished") < kinds.index("explain_started") < kinds.index("explain_delta")
    assert kinds[-2:] == ["explain_done", "run_completed"]
    deltas = [e for e in events if e.type == "explain_delta"]
    assert all(e.data["kind"] == "content" for e in deltas)
    assert "".join(e.data["text"] for e in deltas) == result.explanation.text == GOOD
    assert result.explanation.cited_evidence_ids == [DEMO_EVIDENCE]
    assert result.usage.model_calls == 2 and result.usage.explain_model_calls == 1
    assert result.usage.prompt_tokens == 11 and result.usage.completion_tokens == 7
    assert result.usage.unreported_token_calls == 2
    assert model.requests[0].tools == [] and model.requests[0].max_output_tokens == 400
    payload = json.loads(model.requests[0].messages[1].content)
    assert payload["source_observations"][0]["evidence_id"] == DEMO_EVIDENCE


@pytest.mark.parametrize("text,reason", [
    ("Total 999 USD. [E1]", "unlisted_number"),
    ("Total 九百九十九 USD. [E1]", "unlisted_number"),
    ("Total ９９９ USD. [E1]", "unlisted_number"),
    ("Total 1e999999999999999999999999999999 USD.", "unlisted_number"),
    ("Total 49.99 USD. [E99]", "unknown_citation"),
    ("Total 49.99 USD. [E1", "unknown_citation"),
    ("   ", "empty_explanation"),
    ('{"type":"ask_user"}', "invalid_explanation_format"),
    ("x" * 101, "explanation_too_long"),
])
def test_rejected_text_uses_only_the_verified_plan_template(text, reason):
    result_events = collect(agent(TextModel(text), policy=RunPolicy(explain=True, max_explain_chars=100)))
    result = result_events[-1].result
    assert result.status == "completed" and result.explanation.fallback and not result.explanation.verified
    assert "49.99 USD" in result.explanation.text and "50.01 USD" in result.explanation.text
    assert result_events[-2].data["reason"] == reason
    assert result.verified_plan == asyncio.run(demo_runtime().run(demo_request())).verified_plan


@pytest.mark.parametrize("options,reason", [
    ({"error": CapabilityError("model_unavailable", "offline")}, "model_unavailable"),
    ({"delay": 0.2}, "call_timeout"),
    ({"finish": "length"}, "incomplete_explanation"),
    ({"finish": "error"}, "incomplete_explanation"),
])
def test_model_failure_or_timeout_keeps_completed_plan(options, reason):
    events = collect(agent(TextModel(**options), policy=RunPolicy(explain=True, explain_timeout_s=0.05)))
    assert events[-1].result.status == "completed" and events[-1].result.explanation.fallback
    assert events[-2].data["reason"] == reason
    assert events[-1].result.usage.explain_model_calls == 1


def test_cancellation_and_global_deadline_clear_the_plan_and_explanation():
    for model, policy, status, reason in [
        (TextModel(cancel=True, delay=1), RunPolicy(explain=True), "cancelled", "cancelled"),
        (TextModel(delay=1), RunPolicy(explain=True, deadline_seconds=0.1), "incomplete", "deadline_exceeded"),
    ]:
        seen = []
        events = collect(agent(model, policy=policy, sink=seen.append))
        result = events[-1].result
        assert result.status == status and result.stop_reason == reason
        assert result.explanation is None and result.verified_plan is None
        assert not any(e.type == "explain_done" for e in events)
        count = len(seen)
        model.hooks[0]("content", "late content")
        assert len(seen) == count


def test_success_closes_delta_hook_and_nonstreaming_port_gets_one_delta():
    seen = []
    model = TextModel(streaming=False)
    events = collect(agent(model, sink=seen.append))
    assert [e.data["text"] for e in events if e.type == "explain_delta"] == [GOOD]
    count = len(seen)
    model.hooks[0]("content", "late content")
    assert len(seen) == count


def test_stream_mismatch_is_replaced_instead_of_appending_a_second_answer():
    class Mismatch(TextModel):
        async def generate(self, request, context):
            reply = await super().generate(request, context)
            return reply.model_copy(update={"content": reply.content + " changed"}) if request.action_schema is None else reply
    events = collect(agent(Mismatch()))
    assert events[-1].result.status == "completed" and events[-1].result.explanation.fallback
    assert events[-2].data["reason"] == "explanation_stream_mismatch"


def test_concurrent_runs_keep_explanation_deltas_separate():
    class Concurrent(MockModel):
        async def generate(self, request, context):
            if request.action_schema is not None:
                return await super().generate(request, context)
            text = context.run_id + ": " + GOOD
            for offset in range(0, len(text), 2):
                context.on_delta("content", text[offset:offset + 2])
                await asyncio.sleep(0)
            return ModelReply(content=text)
    async def scenario():
        runtime = agent(Concurrent())
        async def one(run_id):
            events = [e async for e in runtime.astream(demo_request(run_id))]
            assert events[-1].result.explanation.text == run_id + ": " + GOOD
            assert "".join(e.data["text"] for e in events if e.type == "explain_delta") == run_id + ": " + GOOD
            assert all(e.run_id == run_id for e in events)
        await asyncio.gather(one("alpha"), one("beta"))
    asyncio.run(scenario())


def test_unsupported_evidence_falls_back_without_calling_model():
    class WrongSnapshot(MockEvidenceResolver):
        async def resolve(self, plan, context):
            facts = await super().resolve(plan, context)
            return [facts[0].model_copy(update={"snapshot_id": "other"})]

    model = TextModel()
    events = collect(agent(model, resolver=WrongSnapshot()))
    assert events[-1].result.status == "completed" and events[-1].result.explanation.fallback
    assert events[-2].data["reason"] == "invalid_evidence"
    assert not model.requests and events[-1].result.usage.explain_model_calls == 0


def test_evidence_resolution_shares_the_explanation_timeout():
    class SlowResolver(MockEvidenceResolver):
        async def resolve(self, plan, context):
            await asyncio.sleep(0.2)
            return await super().resolve(plan, context)
    model = TextModel()
    events = collect(agent(model, resolver=SlowResolver(), policy=RunPolicy(explain=True, explain_timeout_s=0.03)))
    assert events[-1].result.status == "completed" and events[-1].result.explanation.fallback
    assert not model.requests


def test_disabled_waiting_and_incomplete_paths_emit_no_explanation_events():
    async def scenario():
        for enabled, missing, limit in [(False, False, 8), (True, True, 8), (True, False, 1)]:
            runtime = agent(TextModel(), policy=RunPolicy(explain=enabled, max_model_calls=limit))
            request = demo_request()
            if missing:
                request = request.model_copy(update={"constraints": {"currency": "USD"}})
            events = [e async for e in runtime.astream(request)]
            assert not any(e.node == "explain" for e in events)
            assert events[-1].result.explanation is None and events[-1].result.usage.explain_model_calls == 0
    asyncio.run(scenario())


def test_unexpected_programming_error_still_propagates():
    with pytest.raises(TypeError, match="programming bug"):
        collect(agent(TextModel(error=TypeError("programming bug"))))


def test_rule_check_normalizes_amounts_and_excludes_identifiers_but_not_user_claims():
    plan = asyncio.run(demo_runtime().run(demo_request())).verified_plan
    text = "demo-offer-001 costs 49.990 USD, remaining 50.010 USD. [E1]"
    assert check_explanation(text, plan, [], 8000).text == text
    with pytest.raises(ExplanationRejected, match="unlisted_number"):
        check_explanation("A discount of 999 USD", plan, [], 8000)
    large = plan.model_copy(update={"total_cost": "100010000"})
    assert check_explanation("合计一亿一万 USD", large, [], 8000).verified
    with pytest.raises(ExplanationRejected, match="unlisted_number"):
        check_explanation("合计一亿一万 USD", plan.model_copy(update={"total_cost": "1000000010000"}), [], 8000)
    with pytest.raises(ExplanationRejected, match="context_limit"):
        build_explanation_request(demo_request().model_copy(update={"message": "x" * 8000}), plan, [],
                                  RunPolicy(max_context_chars=1000))


def test_non_completed_result_cannot_contain_explanation():
    completed = asyncio.run(demo_runtime(explain=True).run(demo_request()))
    with pytest.raises(ValidationError, match="explanation"):
        AgentResult.model_validate({**completed.model_dump(), "status": "failed", "verified_plan": None})


def test_version_two_waiting_checkpoint_without_new_fields_keeps_explanation_disabled(tmp_path):
    async def scenario():
        path = tmp_path / "legacy.sqlite"
        config = {"configurable": {"thread_id": "agent:legacy"}}
        async with open_checkpointer(path) as saver:
            old_runtime = AgentRuntime(AgentDependencies(TextModel(), demo_tools(), MockPlanValidator()),
                                       RunPolicy(), checkpointer=saver)
            waiting = await old_runtime.run(
                demo_request("legacy").model_copy(update={"constraints": {"currency": "USD"}}))
            saved = await saver.aget_tuple(config)
            checkpoint = deepcopy(saved.checkpoint)
            state = checkpoint["channel_values"]
            state.pop("explanation")
            state["usage"].pop("explain_model_calls")
            for field in ("explain", "explain_timeout_s", "max_explain_tokens", "max_explain_chars"):
                state["policy"].pop(field)
            checkpoint["channel_versions"].pop("explanation", None)
            for versions in checkpoint["versions_seen"].values():
                versions.pop("explanation", None)
            await saver.aput(saved.config, checkpoint, saved.metadata, {})
        async with open_checkpointer(path, create=False) as saver:
            # The new host opts in, but resume must honor the old saved policy.
            runtime = AgentRuntime(AgentDependencies(TextModel(), demo_tools(), MockPlanValidator()),
                                   RunPolicy(explain=True), checkpointer=saver)
            events = [e async for e in runtime.astream_resume("legacy", waiting.pending_question.question_id,
                                                              {"purchase_cap": "100.00"})]
            assert events[-1].result.status == "completed" and events[-1].result.explanation is None
            assert not any(e.node == "explain" for e in events)
    asyncio.run(scenario())


def test_fallback_says_when_the_total_excludes_tax():
    from agent.contracts import PlanItem, VerifiedPlan
    from agent.explanation import render_fallback
    base = dict(snapshot_id="s", currency="USD", total_cost="12.00", remaining_funds="88.00",
                items=[PlanItem(offer_id="o1", quantity=1, evidence_ids=["e1"])])
    pre_tax = render_fallback(VerifiedPlan(**base, cost_breakdown={"items": "10.00", "shipping": "2.00"}))
    with_tax = render_fallback(VerifiedPlan(**base, cost_breakdown={"items": "10.00", "shipping": "2.00", "tax": "0.00"}))
    assert "（不含税）" in pre_tax.text and "不含税" not in with_tax.text

"""通过注入模拟能力验证运行场景，不访问模型服务或其他外部接口。"""

import asyncio
import json

import pytest
from pydantic import ValidationError

from agent import (
    AgentDependencies, AgentRuntime, CapabilityError, Issue, ModelReply,
    PendingOperation, RunPolicy, RunStateError, ToolRegistry, ValidationResult,
)
from agent.contracts import UnsupportedCapabilityError
from agent.examples.mock_run import (
    DEMO_EVIDENCE, DEMO_OFFER, MockModel, MockPlanValidator, MockSearch,
    demo_request, demo_runtime, demo_tools,
)

ASK = {"type": "ask_user", "question": "What is your budget?", "required_fields": ["purchase_cap"]}
SEARCH = {"type": "call_tool", "tool_name": "search_products", "arguments": {"query": "earbuds"}}
PLAN = {"type": "propose_plan", "plan": {"items": [
    {"offer_id": DEMO_OFFER, "quantity": 1, "evidence_ids": [DEMO_EVIDENCE]},
]}}


class ScriptedModel:
    """依次返回预设动作，并记录收到的模型请求以供断言。"""

    def __init__(self, *actions):
        self.actions = iter(actions)
        self.requests = []

    async def generate(self, request, context):
        self.requests.append(request)
        action = next(self.actions)
        return ModelReply(content=action if isinstance(action, str) else json.dumps(action))


class CountingSearch(MockSearch):
    """统计搜索执行器的实际调用次数，验证非法请求不会触发工具。"""

    def __init__(self):
        self.calls = 0

    async def execute(self, arguments, context):
        self.calls += 1
        return await super().execute(arguments, context)


def runtime(model, *, validator=None, executor=None, policy=None, sink=None):
    return AgentRuntime(AgentDependencies(model, demo_tools(executor), validator, sink), policy)


def run(agent, request=None):
    return asyncio.run(agent.run(request if request is not None else demo_request()))


def test_demo_completes_with_authoritative_costs_and_citations():
    result = run(demo_runtime())
    assert result.status == "completed"
    assert result.verified_plan.total_cost == "49.99"
    assert result.verified_plan.remaining_funds == "50.01"
    assert result.verified_plan.items[0].evidence_ids == [DEMO_EVIDENCE]
    assert result.usage.model_calls == 2 and result.usage.tool_calls == 1
    assert result.usage.unreported_token_calls == 2


def test_live_events_and_run_have_same_result_contract():
    async def collect():
        return [event async for event in demo_runtime().astream(demo_request())]
    events = asyncio.run(collect())
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))
    assert [e.type for e in events] == [
        "run_started", "decision_started", "action_selected", "tool_started", "tool_finished",
        "decision_started", "action_selected", "validation_finished", "run_completed",
    ]
    assert all(e.run_id == "demo-run-001" for e in events)
    assert events[-1].result.status == "completed"


def test_clarification_is_resumable_and_invalid_checkpointer_is_rejected():
    agent = runtime(ScriptedModel(ASK))
    result = run(agent)
    assert result.status == "waiting_user" and result.clarification.required_fields == ["purchase_cap"]
    assert result.verified_plan is None
    assert result.pending_question.question_id
    with pytest.raises(RunStateError) as exc:
        asyncio.run(agent.resume("some-run", "some-question", {}))
    assert exc.value.code == "run_not_found"
    with pytest.raises(UnsupportedCapabilityError):
        AgentRuntime(agent.dependencies, checkpointer=object())


@pytest.mark.parametrize("action,code", [
    ({**SEARCH, "tool_name": "not_registered"}, "unknown_tool"),
    ({**SEARCH, "arguments": {"query": "earbuds", "unexpected": True}}, "invalid_tool_arguments"),
    ({**SEARCH, "arguments": {"query": 123}}, "invalid_tool_arguments"),
])
def test_invalid_tool_requests_never_execute(action, code):
    executor = CountingSearch()
    result = run(runtime(ScriptedModel(action, ASK), executor=executor))
    assert result.status == "waiting_user"
    assert executor.calls == 0 and result.usage.tool_calls == 0
    assert result.issues[0].code == code


def test_malformed_output_has_bounded_repairs():
    result = run(runtime(ScriptedModel("invalid", "invalid", "invalid")))
    assert result.status == "incomplete" and result.stop_reason == "invalid_action"
    assert result.usage.model_calls == 3


@pytest.mark.parametrize("bad", [json.dumps(PLAN)[:-1],
                               json.dumps({**PLAN, "extra": True}), "x" * 6000])
def test_action_repair_includes_bounded_rejected_output_and_validation_details(bad):
    model = ScriptedModel(bad, ASK)
    result = run(runtime(model))
    assert result.status == "waiting_user" and result.usage.action_repairs == 1
    feedback = json.loads(model.requests[1].messages[-1].content)["observation"]
    assert feedback["rejected_output"] == bad[:2048]
    assert feedback["rejected_output_truncated"] == (len(bad) > 2048)
    assert feedback["validation_errors"]
    assert all("input" not in detail for detail in feedback["validation_errors"])
    assert feedback["instruction"]


def test_valid_action_resets_consecutive_format_errors():
    result = run(runtime(ScriptedModel("invalid", SEARCH, "invalid", ASK),
                         policy=RunPolicy(max_action_repairs=1)))
    assert result.status == "waiting_user" and result.usage.model_calls == 4


def test_model_limit_stops_a_repeating_decision():
    bad = {**SEARCH, "tool_name": "missing"}
    result = run(runtime(ScriptedModel(bad, bad), policy=RunPolicy(max_model_calls=2)))
    assert result.stop_reason == "model_call_limit" and result.usage.model_calls == 2


def test_tool_limit_counts_actual_calls_only():
    executor = CountingSearch()
    result = run(runtime(ScriptedModel(SEARCH, SEARCH), executor=executor,
                         policy=RunPolicy(max_tool_calls=1)))
    assert result.stop_reason == "tool_call_limit"
    assert executor.calls == 1 and result.usage.tool_calls == 1


def test_missing_validator_cannot_complete():
    result = run(runtime(ScriptedModel(PLAN)))
    assert result.status == "incomplete" and result.stop_reason == "validation_unavailable"
    assert result.verified_plan is None


def test_failed_validation_can_be_repaired():
    bad = {"type": "propose_plan", "plan": {"items": [
        {"offer_id": DEMO_OFFER, "quantity": 10, "evidence_ids": [DEMO_EVIDENCE]},
    ]}}
    result = run(runtime(ScriptedModel(bad, PLAN), validator=MockPlanValidator()))
    assert result.status == "completed" and result.usage.plan_repairs == 1
    assert result.verified_plan.total_cost == "49.99"


def test_rejected_plan_stops_after_repair_budget():
    class Reject:
        async def validate(self, request, context):
            return ValidationResult(status="failed", proposal_fingerprint=request.proposal_fingerprint,
                                    issues=[Issue(code="budget_exceeded", message="Over budget.")])
    result = run(runtime(ScriptedModel(PLAN, PLAN), validator=Reject(),
                         policy=RunPolicy(max_plan_repairs=1)))
    assert result.status == "incomplete" and result.stop_reason == "plan_repair_limit"
    assert result.verified_plan is None


@pytest.mark.parametrize("mismatch", ["fingerprint", "snapshot", "items"])
def test_validator_receipt_must_match_current_proposal(mismatch):
    class WrongReceipt(MockPlanValidator):
        async def validate(self, request, context):
            result = await super().validate(request, context)
            if mismatch == "fingerprint":
                return result.model_copy(update={"proposal_fingerprint": "old"})
            plan = result.verified_plan
            if mismatch == "snapshot":
                plan = plan.model_copy(update={"snapshot_id": "other-snapshot"})
            else:
                plan = plan.model_copy(update={"items": [plan.items[0].model_copy(update={"quantity": 2})]})
            return result.model_copy(update={"verified_plan": plan})
    result = run(runtime(ScriptedModel(PLAN), validator=WrongReceipt()))
    assert result.status == "failed" and result.stop_reason == "invalid_validation_receipt"
    assert result.verified_plan is None


def test_untrusted_prices_and_fake_finish_action_are_rejected():
    invented = {"type": "propose_plan", "plan": {**PLAN["plan"], "total_cost": "0.01"}}
    result = run(runtime(ScriptedModel(invented, {"type": "finish", "answer": "approved"}, ASK)))
    assert result.status == "waiting_user" and result.usage.action_repairs == 2


@pytest.mark.parametrize("recoverable", [True, False])
def test_declared_tool_errors_are_normalized(recoverable):
    class Broken:
        async def execute(self, arguments, context):
            raise CapabilityError("tool_unavailable", "Tool is unavailable.", recoverable=recoverable)
    result = run(runtime(ScriptedModel(SEARCH, ASK), executor=Broken()))
    assert result.status == ("waiting_user" if recoverable else "failed")
    assert result.usage.tool_calls == 1


def test_bad_tool_result_fails_without_exposing_payload():
    class Broken:
        async def execute(self, arguments, context):
            return {"password": "must-not-be-returned"}
    result = run(runtime(ScriptedModel(SEARCH), executor=Broken()))
    assert result.stop_reason == "invalid_tool_result"
    assert "must-not-be-returned" not in result.model_dump_json()


def test_pending_model_is_explicitly_unsupported():
    class Pending:
        async def generate(self, request, context):
            return PendingOperation(handle="external-request")
    result = run(runtime(Pending()))
    assert result.status == "incomplete" and result.stop_reason == "unsupported_capability"


def test_context_limit_fails_before_calling_model():
    model = ScriptedModel(ASK)
    result = run(runtime(model, policy=RunPolicy(max_context_chars=1000)))
    assert result.stop_reason == "context_limit" and model.requests == []


def test_deadline_cancels_inflight_work():
    class Slow:
        cancelled = False
        async def generate(self, request, context):
            try:
                await asyncio.sleep(10)
            finally:
                self.cancelled = True
    model = Slow()
    result = run(runtime(model, policy=RunPolicy(deadline_seconds=0.05)))
    assert result.status == "incomplete" and result.stop_reason == "deadline_exceeded"
    assert model.cancelled


def test_explicit_cancellation_produces_cancelled_outcome():
    async def scenario():
        cancellation = asyncio.Event()
        class Cancel:
            async def generate(self, request, context):
                cancellation.set()
                await asyncio.sleep(10)
        return await runtime(Cancel()).run(demo_request(), cancellation=cancellation)
    result = asyncio.run(scenario())
    assert result.status == "cancelled" and result.verified_plan is None


def test_concurrent_runs_do_not_share_history_or_counters():
    async def scenario():
        agent = demo_runtime()
        return await asyncio.gather(agent.run(demo_request("run-a")), agent.run(demo_request("run-b")))
    a, b = asyncio.run(scenario())
    assert (a.run_id, b.run_id) == ("run-a", "run-b")
    assert a.status == b.status == "completed"
    assert a.usage.model_calls == b.usage.model_calls == 2


def test_event_sink_failure_does_not_repeat_external_work(caplog):
    def broken_sink(event):
        raise RuntimeError("telemetry failure")
    executor = CountingSearch()
    result = run(runtime(MockModel(), executor=executor, validator=MockPlanValidator(), sink=broken_sink))
    assert result.status == "completed" and executor.calls == 1
    assert "event sink failed" in caplog.text


def test_programming_errors_are_not_disguised_as_business_failures():
    class Broken:
        async def generate(self, request, context):
            raise RuntimeError("adapter bug")
    with pytest.raises(RuntimeError, match="adapter bug"):
        run(runtime(Broken()))


def test_stream_closure_cleans_up_background_execution():
    async def scenario():
        class Slow:
            cancelled = False
            async def generate(self, request, context):
                try:
                    await asyncio.sleep(10)
                finally:
                    self.cancelled = True
        model = Slow()
        stream = runtime(model).astream(demo_request())
        async for event in stream:
            if event.type == "decision_started":
                await asyncio.sleep(0)
                break
        await stream.aclose()
        return model.cancelled
    assert asyncio.run(scenario())


def test_invalid_request_and_duplicate_tools_fail_before_execution():
    request = {**demo_request().model_dump(), "unexpected": True}
    with pytest.raises(ValidationError):
        run(demo_runtime(), request)
    tool = demo_tools().get("search_products")
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([tool, tool])


def test_validator_cannot_change_the_proposal_it_is_checking():
    class MutatingValidator(MockPlanValidator):
        async def validate(self, request, context):
            # 冻结的 Pydantic 对象仍可包含可变列表；模拟适配器误改列表的情况，
            # 确认它无法连带修改运行时用于比较的原候选方案。
            request.plan.items[0] = request.plan.items[0].model_copy(update={"quantity": 2})
            return await super().validate(request, context)
    result = run(runtime(ScriptedModel(PLAN), validator=MutatingValidator()))
    assert result.status == "failed" and result.stop_reason == "invalid_validation_receipt"
    assert result.verified_plan is None


def test_declared_model_failure_has_a_safe_structured_outcome():
    class Unavailable:
        async def generate(self, request, context):
            raise CapabilityError("model_unavailable", "Model is unavailable.")
    result = run(runtime(Unavailable()))
    assert result.status == "failed" and result.stop_reason == "model_unavailable"
    assert result.usage.model_calls == 1


def test_model_call_timeout_is_distinct_from_the_run_deadline():
    class Slow:
        async def generate(self, request, context):
            await asyncio.sleep(10)
    result = run(runtime(Slow(), policy=RunPolicy(call_timeout_s=0.02, deadline_seconds=2)))
    assert result.status == "failed" and result.stop_reason == "call_timeout"


def test_truncated_generation_cannot_submit_a_valid_looking_plan():
    class Truncated:
        async def generate(self, request, context):
            return ModelReply(content=json.dumps(PLAN), finish_reason="length")
    result = run(runtime(Truncated(), validator=MockPlanValidator(),
                         policy=RunPolicy(max_action_repairs=0)))
    assert result.status == "incomplete" and result.stop_reason == "invalid_action"


def test_oversized_tool_result_stops_without_inserting_it_into_context():
    class Large(MockSearch):
        async def execute(self, arguments, context):
            result = await super().execute(arguments, context)
            return result.model_copy(update={"title": "x" * 9000})
    model = ScriptedModel(SEARCH)
    result = run(runtime(model, executor=Large()))
    assert result.status == "failed" and result.stop_reason == "tool_result_too_large"
    assert len(model.requests) == 1


def test_pending_tool_is_explicitly_unsupported():
    class Pending:
        async def execute(self, arguments, context):
            return PendingOperation(handle="tool-request")
    result = run(runtime(ScriptedModel(SEARCH), executor=Pending()))
    assert result.status == "incomplete" and result.stop_reason == "unsupported_capability"


def test_unavailable_validator_cannot_complete():
    class Unavailable:
        async def validate(self, request, context):
            return ValidationResult(status="unavailable", proposal_fingerprint=request.proposal_fingerprint)
    result = run(runtime(ScriptedModel(PLAN), validator=Unavailable()))
    assert result.status == "incomplete" and result.stop_reason == "validation_unavailable"


def test_cancelling_the_caller_cleans_up_the_model_task():
    async def scenario():
        entered, stopped = asyncio.Event(), asyncio.Event()
        class Slow:
            async def generate(self, request, context):
                entered.set()
                try:
                    await asyncio.sleep(10)
                finally:
                    stopped.set()
        task = asyncio.create_task(runtime(Slow()).run(demo_request()))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
    asyncio.run(scenario())

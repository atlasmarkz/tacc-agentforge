"""使用本地模拟能力验证问题检查点、恢复关联和跨恢复约束。"""

import asyncio
import json
from contextlib import aclosing

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent import AgentDependencies, AgentRuntime, ModelReply, RunPolicy, RunStateError
from agent.examples.mock_run import (
    DEMO_EVIDENCE, DEMO_OFFER, MockModel, MockPlanValidator, MockSearch,
    demo_request, demo_tools,
)

ASK = {"type": "ask_user", "question": "What is your purchase cap?", "required_fields": ["purchase_cap"]}
SEARCH = {"type": "call_tool", "tool_name": "search_products", "arguments": {"query": "earbuds"}}
PLAN = {"type": "propose_plan", "plan": {"items": [
    {"offer_id": DEMO_OFFER, "quantity": 1, "evidence_ids": [DEMO_EVIDENCE]},
]}}


class ScriptedModel:
    """按顺序返回动作并记录调用，可通过延迟检验累计执行时间预算。"""

    def __init__(self, *actions, delay=0):
        self.actions = iter(actions)
        self.calls = 0
        self.delay = delay
        self.contexts = []

    async def generate(self, request, context):
        self.calls += 1
        self.contexts.append(request)
        await asyncio.sleep(self.delay)
        return ModelReply(content=json.dumps(next(self.actions)))


def agent(model=None, policy=None, checkpointer=None, search=None):
    return AgentRuntime(AgentDependencies(model or MockModel(), demo_tools(search), MockPlanValidator()),
                        policy, checkpointer=checkpointer)


def missing_budget(run_id="clarification-run"):
    return demo_request(run_id).model_copy(update={"constraints": {"currency": "USD"}})


def test_pause_and_resume_finishes_the_same_run():
    async def scenario():
        runtime = agent()
        waiting = await runtime.run(missing_budget())
        assert waiting.status == "waiting_user"
        assert waiting.usage.model_calls == 1 and waiting.usage.tool_calls == 0
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"purchase_cap": "100.00"})
        assert result.status == "completed" and result.run_id == waiting.run_id
        assert result.usage.model_calls == 3 and result.usage.tool_calls == 1
        assert result.verified_plan.remaining_funds == "50.01"
        assert result.pending_question is None and result.clarification is None
    asyncio.run(scenario())


def test_completed_search_is_not_replayed_after_answer():
    async def scenario():
        class Search(MockSearch):
            calls = 0
            async def execute(self, arguments, context):
                self.calls += 1
                return await super().execute(arguments, context)
        search = Search()
        model = ScriptedModel(SEARCH, ASK, PLAN)
        runtime = agent(model, search=search)
        waiting = await runtime.run(missing_budget())
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"purchase_cap": "100.00"})
        assert result.status == "completed" and search.calls == 1 and model.calls == 3
        assert any('"observation"' in m.content for m in model.contexts[-1].messages)
        assert any('"clarification_answer"' in m.content for m in model.contexts[-1].messages)
    asyncio.run(scenario())


def test_injected_checkpoint_supports_a_new_runtime_instance():
    async def scenario():
        saver = InMemorySaver()
        waiting = await agent(checkpointer=saver).run(missing_budget())
        result = await agent(checkpointer=saver).resume(waiting.run_id, waiting.pending_question.question_id,
                                                       {"purchase_cap": "100.00"})
        assert result.status == "completed" and result.usage.model_calls == 3
    asyncio.run(scenario())


def test_waiting_result_comes_from_interrupt_without_finalize():
    async def scenario():
        runtime = agent()
        events = [e async for e in runtime.astream(missing_budget())]
        assert events[-1].type == "run_waiting"
        assert "finalize" not in [e.node for e in events]
        config = {"configurable": {"thread_id": f"agent:{events[-1].run_id}"}}
        saved = await runtime.checkpointer.aget_tuple(config)
        assert saved.checkpoint["channel_values"]["pending_question"] == events[-1].result.pending_question.model_dump()
    asyncio.run(scenario())


def test_event_sequence_continues_across_resume():
    async def scenario():
        runtime = agent()
        first = [e async for e in runtime.astream(missing_budget())]
        waiting = first[-1].result
        second = [e async for e in runtime.astream_resume(
            waiting.run_id, waiting.pending_question.question_id, {"purchase_cap": "100.00"})]
        events = first + second
        assert [e.sequence for e in events] == list(range(1, len(events) + 1))
        assert second[0].type == "run_resumed" and second[-1].type == "run_completed"
    asyncio.run(scenario())


@pytest.mark.parametrize("answer,code", [
    ({"purchase_cap": "-1"}, "invalid_budget"),
    ({"purchase_cap": 100}, "invalid_budget"),
    ({"purchase_cap": ""}, "empty_answer"),
    ({"snapshot_id": "other-snapshot"}, "invalid_answer_fields"),
    ({"purchase_cap": "100.00", "run_id": "other-run"}, "invalid_answer_fields"),
    ({}, "invalid_answer_fields"),
])
def test_invalid_answer_reasks_with_a_new_id_without_calling_model(answer, code):
    async def scenario():
        runtime = agent()
        waiting = await runtime.run(missing_budget())
        old_id = waiting.pending_question.question_id
        again = await runtime.resume(waiting.run_id, old_id, answer)
        assert again.status == "waiting_user" and again.usage.model_calls == 1
        assert again.pending_question.question_id != old_id
        assert again.issues[-1].code == code
        with pytest.raises(RunStateError) as exc:
            await runtime.resume(waiting.run_id, old_id, {"purchase_cap": "100.00"})
        assert exc.value.code == "question_mismatch"
        result = await runtime.resume(waiting.run_id, again.pending_question.question_id,
                                      {"purchase_cap": "100.00"})
        assert result.status == "completed" and result.snapshot_id == waiting.snapshot_id
    asyncio.run(scenario())


def test_invalid_answer_has_a_finite_repair_budget():
    async def scenario():
        runtime = agent(policy=RunPolicy(max_answer_repairs=1))
        waiting = await runtime.run(missing_budget())
        again = await runtime.resume(waiting.run_id, waiting.pending_question.question_id, {})
        stopped = await runtime.resume(again.run_id, again.pending_question.question_id, {})
        assert stopped.status == "incomplete" and stopped.stop_reason == "answer_repair_limit"
        assert stopped.usage.model_calls == 1 and stopped.pending_question is None
    asyncio.run(scenario())


def test_question_count_is_bounded_across_valid_answers():
    async def scenario():
        runtime = agent(ScriptedModel(ASK, ASK), RunPolicy(max_questions=1))
        waiting = await runtime.run(missing_budget())
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"purchase_cap": "100.00"})
        assert result.status == "incomplete" and result.stop_reason == "question_limit"
    asyncio.run(scenario())


def test_wrong_run_and_question_do_not_advance_the_checkpoint():
    async def scenario():
        runtime = agent()
        waiting = await runtime.run(missing_budget())
        for run_id, question_id, code in [
            ("unknown-run", waiting.pending_question.question_id, "run_not_found"),
            (waiting.run_id, "other-question", "question_mismatch"),
        ]:
            with pytest.raises(RunStateError) as exc:
                await runtime.resume(run_id, question_id, {"purchase_cap": "100.00"})
            assert exc.value.code == code
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"purchase_cap": "100.00"})
        assert result.status == "completed"
        with pytest.raises(RunStateError) as exc:
            await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                 {"purchase_cap": "100.00"})
        assert exc.value.code == "not_waiting"
    asyncio.run(scenario())


def test_run_id_cannot_be_reused_to_overwrite_a_waiting_checkpoint():
    async def scenario():
        runtime = agent()
        request = missing_budget()
        await runtime.run(request)
        with pytest.raises(RunStateError) as exc:
            await runtime.run(request)
        assert exc.value.code == "run_exists"
    asyncio.run(scenario())


def test_concurrent_answers_are_not_applied_twice():
    async def scenario():
        model = ScriptedModel(ASK, SEARCH, PLAN, delay=0.01)
        runtime = agent(model)
        waiting = await runtime.run(missing_budget())
        results = await asyncio.gather(*[
            runtime.resume(waiting.run_id, waiting.pending_question.question_id, {"purchase_cap": "100.00"})
            for _ in range(2)
        ], return_exceptions=True)
        assert sum(getattr(r, "status", None) == "completed" for r in results) == 1
        errors = [r for r in results if isinstance(r, RunStateError)]
        assert len(errors) == 1 and errors[0].code in {"run_busy", "not_waiting"}
        assert model.calls == 3
    asyncio.run(scenario())


def test_resume_preserves_original_policy_and_model_call_count():
    async def scenario():
        saver = InMemorySaver()
        waiting = await agent(policy=RunPolicy(max_model_calls=1), checkpointer=saver).run(missing_budget())
        result = await agent(policy=RunPolicy(max_model_calls=100), checkpointer=saver).resume(
            waiting.run_id, waiting.pending_question.question_id, {"purchase_cap": "100.00"})
        assert result.stop_reason == "model_call_limit" and result.usage.model_calls == 1
    asyncio.run(scenario())


def test_expired_question_finishes_without_another_model_call():
    async def scenario():
        runtime = agent(policy=RunPolicy(question_timeout_s=0.04))
        waiting = await runtime.run(missing_budget())
        await asyncio.sleep(0.06)
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"purchase_cap": "100.00"})
        assert result.status == "incomplete" and result.stop_reason == "question_expired"
        assert result.usage.model_calls == 1
    asyncio.run(scenario())


def test_wait_time_is_excluded_but_active_execution_time_is_cumulative():
    async def scenario():
        runtime = agent(ScriptedModel(ASK, SEARCH, PLAN, delay=0.15),
                        RunPolicy(deadline_seconds=0.4, question_timeout_s=5))
        waiting = await runtime.run(missing_budget())
        assert waiting.usage.elapsed_s >= 0.15
        await asyncio.sleep(0.45)  # 等待时间超过整个实际执行预算，以验证等待不会被扣除。
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"purchase_cap": "100.00"})
        assert result.stop_reason == "deadline_exceeded"
        # 恢复后仍能通过 guard 并继续两次模型尝试，说明等待时间没有计入预算，
        # 最终超时又说明首次调用已消耗的执行时间没有被清零。
        assert result.usage.model_calls == 3 and result.usage.elapsed_s >= 0.4
    asyncio.run(scenario())


def test_waiting_run_can_be_cancelled_without_merging_an_answer():
    async def scenario():
        runtime = agent()
        waiting = await runtime.run(missing_budget())
        cancelled = asyncio.Event()
        cancelled.set()
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"purchase_cap": "100.00"}, cancellation=cancelled)
        assert result.status == "cancelled" and result.usage.model_calls == 1
    asyncio.run(scenario())


def test_closing_a_waiting_stream_keeps_the_saved_question_resumable():
    async def scenario():
        runtime = agent()
        async with aclosing(runtime.astream(missing_budget())) as events:
            async for event in events:
                if event.type == "run_waiting":
                    waiting = event.result
                    break
        result = await runtime.resume(waiting.run_id, waiting.pending_question.question_id,
                                      {"purchase_cap": "100.00"})
        assert result.status == "completed"
    asyncio.run(scenario())

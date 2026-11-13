"""Tests for the ModelPort adapter. Needs pydantic 2 (the agent environment); skipped otherwise."""
import asyncio
import json
import time

import pytest

pydantic = pytest.importorskip("pydantic")
if int(pydantic.VERSION.split(".")[0]) < 2:
    pytest.skip("needs pydantic 2 (use agent/.venv)", allow_module_level=True)

from agent.contracts import ACTION_ADAPTER, CallContext, CapabilityError, Message, ModelRequest, ToolDescription  # noqa: E402
from inference.client import ModelClient  # noqa: E402
from inference.model_port import VllmModelPort, decoder_schema, to_chat_messages  # noqa: E402
from tests.fake_vllm import OK_BODY, FakeVllm  # noqa: E402

ACTION = {"type": "ask_user", "question": "Which size?", "required_fields": ["size"]}
ACTION_BODY = {"choices": [{"message": {"content": json.dumps(ACTION)}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 120, "completion_tokens": 20}}


def make_request():
    return ModelRequest(
        messages=[Message(role="system", content="You are an advisor."),
                  Message(role="user", content='{"message":"buy a kettle"}')],
        tools=[ToolDescription(name="search_products", description="Search the snapshot.",
                               parameters={"type": "object", "properties": {"q": {"type": "string"}}})],
        action_schema=ACTION_ADAPTER.json_schema(),
        max_output_tokens=256,
    )


def make_context(timeout_s=5.0):
    return CallContext(run_id="r1", snapshot_id="s1", operation_id="r1:model:1",
                       timeout_s=timeout_s, cancellation=asyncio.Event())


@pytest.fixture
def servers():
    made = []

    def make(script):
        s = FakeVllm(script)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


def port_for(srv, **kw):
    client = ModelClient(base_url=srv.url, backoff_base_s=0.01, backoff_max_s=0.05)
    return VllmModelPort(client, **kw)


def test_messages_carry_tools_and_schema_in_one_leading_system_message():
    msgs = to_chat_messages(make_request())
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "search_products" in msgs[0]["content"] and '"call_tool"' in msgs[0]["content"]
    assert msgs[0]["content"].startswith("You are an advisor.")
    assert to_chat_messages(make_request()) == msgs          # deterministic: a stable cache prefix


def test_generate_returns_valid_reply(servers):
    srv = servers([(200, ACTION_BODY, 0)])
    reply = asyncio.run(port_for(srv).generate(make_request(), make_context()))
    assert ACTION_ADAPTER.validate_json(reply.content).type == "ask_user"
    assert (reply.prompt_tokens, reply.completion_tokens, reply.finish_reason) == (120, 20, "stop")
    body = srv.requests[0]
    assert body["max_tokens"] == 256 and body["temperature"] == 0.0
    assert body["response_format"]["json_schema"]["schema"] == decoder_schema(ACTION_ADAPTER.json_schema())


def test_options_can_disable_constraint_and_thinking(servers):
    srv = servers([(200, ACTION_BODY, 0)])
    port = port_for(srv, constrain_output=False, enable_thinking=False)
    asyncio.run(port.generate(make_request(), make_context()))
    body = srv.requests[0]
    assert "response_format" not in body and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_truncated_generation_is_reported_as_length(servers):
    body = {"choices": [{"message": {"content": '{"type":"ask'}, "finish_reason": "length"}]}
    srv = servers([(200, body, 0)])
    reply = asyncio.run(port_for(srv).generate(make_request(), make_context()))
    assert reply.finish_reason == "length" and reply.prompt_tokens is None


def test_timeout_becomes_recoverable_capability_error(servers):
    srv = servers([(200, ACTION_BODY, 3.0)])
    t0 = time.monotonic()
    with pytest.raises(CapabilityError) as err:
        asyncio.run(port_for(srv, timeout_margin_s=0.0).generate(make_request(), make_context(timeout_s=0.6)))
    assert err.value.issue.code == "model_timeout" and err.value.recoverable
    assert time.monotonic() - t0 < 2


def test_unavailable_becomes_recoverable_capability_error(servers):
    srv = servers([(503, {"error": "starting"}, 0)])
    with pytest.raises(CapabilityError) as err:
        asyncio.run(port_for(srv).generate(make_request(), make_context()))
    assert err.value.issue.code == "model_unavailable" and err.value.recoverable


def test_rejected_request_is_not_recoverable(servers):
    srv = servers([(400, {"error": "too long"}, 0)])
    with pytest.raises(CapabilityError) as err:
        asyncio.run(port_for(srv).generate(make_request(), make_context()))
    assert err.value.issue.code == "model_request_rejected" and not err.value.recoverable


def test_harness_cancellation_event_stops_the_call(servers):
    srv = servers([(200, ACTION_BODY, 3.0)])

    async def scenario():
        ctx = make_context(timeout_s=10)
        asyncio.get_running_loop().call_later(0.2, ctx.cancellation.set)
        return await port_for(srv).generate(make_request(), ctx)

    t0 = time.monotonic()
    with pytest.raises(CapabilityError) as err:
        asyncio.run(scenario())
    assert err.value.issue.code == "cancelled" and time.monotonic() - t0 < 1.5


def test_task_cancellation_stops_the_call(servers):
    srv = servers([(200, ACTION_BODY, 3.0)])

    async def scenario():
        task = asyncio.create_task(port_for(srv).generate(make_request(), make_context(timeout_s=10)))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    t0 = time.monotonic()
    asyncio.run(scenario())
    assert time.monotonic() - t0 < 1.5


def test_streaming_port_forwards_deltas_on_the_event_loop_thread(servers):
    import threading
    from tests.fake_vllm import Stream, sse_events
    srv = servers([(200, Stream(sse_events(["{", json.dumps(ACTION)[1:]], reasoning=["hmm"]), interval_s=0.05), 0)])
    seen, threads = [], set()

    def sink(op_id, kind, text):
        seen.append((op_id, kind, text))
        threads.add(threading.get_ident())

    async def scenario():
        main = threading.get_ident()
        reply = await port_for(srv, on_delta=sink).generate(make_request(), make_context(timeout_s=10))
        return main, reply

    main, reply = asyncio.run(scenario())
    assert ACTION_ADAPTER.validate_json(reply.content).type == "ask_user"
    assert reply.finish_reason == "stop" and reply.completion_tokens == 7
    assert [k for _, k, _ in seen] == ["reasoning", "content", "content"]
    assert {op for op, _, _ in seen} == {"r1:model:1"} and threads == {main}
    assert srv.requests[0]["stream"] is True and "response_format" in srv.requests[0]


def test_streaming_port_interruption_is_recoverable(servers):
    from tests.fake_vllm import Stream, sse_events
    srv = servers([(200, Stream(sse_events(["a", "b", "c"]), drop_after=3), 0)])
    with pytest.raises(CapabilityError) as err:
        asyncio.run(port_for(srv, on_delta=lambda *a: None).generate(make_request(), make_context()))
    assert err.value.issue.code == "model_unavailable" and err.value.recoverable


@pytest.mark.parametrize("thinking", [None, True])
def test_free_text_uses_per_call_hook_without_schema_and_drains_before_return(servers, thinking):
    from dataclasses import replace
    import threading
    from tests.fake_vllm import Stream, sse_events
    srv = servers([(200, Stream(sse_events(["Verified ", "plan."], reasoning=["hidden"])), 0)])
    seen, threads = [], set()

    def scoped(kind, text):
        seen.append((kind, text))
        threads.add(threading.get_ident())

    def broken_legacy(*args):
        raise ValueError("monitor failed")

    async def scenario():
        context = replace(make_context(), operation_id="r1:explain:1", on_delta=scoped)
        request = make_request().model_copy(update={"action_schema": None, "tools": []})
        assert to_chat_messages(request) == [message.model_dump() for message in request.messages]
        port = port_for(srv, on_delta=broken_legacy, enable_thinking=thinking)
        reply = await port.generate(request, context)
        assert "".join(text for kind, text in seen if kind == "content") == reply.content == "Verified plan."
        assert threads == {threading.get_ident()}
    asyncio.run(scenario())
    assert "response_format" not in srv.requests[0]
    assert srv.requests[0]["stream"] is True
    assert srv.requests[0]["chat_template_kwargs"]["enable_thinking"] == (thinking if thinking is not None else False)


def test_shared_port_keeps_concurrent_scoped_hooks_separate(servers):
    from dataclasses import replace
    from tests.fake_vllm import Stream, sse_events
    srv = servers([(200, Stream(sse_events(["Alpha ", "answer"]), interval_s=0.005), 0),
                   (200, Stream(sse_events(["Beta ", "answer"]), interval_s=0.005), 0)])

    async def scenario():
        port = port_for(srv)
        request = make_request().model_copy(update={"action_schema": None, "tools": []})
        seen = {"alpha": [], "beta": []}

        async def one(run_id):
            context = replace(make_context(), run_id=run_id, operation_id=f"{run_id}:explain:1",
                              on_delta=lambda kind, text: seen[run_id].append(text) if kind == "content" else None)
            result = await port.generate(request, context)
            assert "".join(seen[run_id]) == result.content
            return result.content
        results = await asyncio.gather(one("alpha"), one("beta"))
        assert set(results) == {"Alpha answer", "Beta answer"}
    asyncio.run(scenario())


def test_port_ignores_late_callbacks_after_success_and_cancellation():
    from dataclasses import replace
    from types import SimpleNamespace

    class Client:
        delta = None
        def chat(self, messages, *, on_delta, cancel, **kwargs):
            self.delta = on_delta
            on_delta("content", "early")
            if self.block:
                while not cancel.cancelled:
                    time.sleep(0.001)
                on_delta("content", "late")
                from inference.client import Cancelled
                raise Cancelled()
            return SimpleNamespace(content="early", prompt_tokens=1, completion_tokens=1, finish_reason="stop")

    async def scenario(block):
        client = Client()
        client.block = block
        seen = []
        ready = asyncio.Event()
        def hook(kind, text):
            seen.append(text)
            ready.set()
        context = replace(make_context(), on_delta=hook)
        request = make_request().model_copy(update={"action_schema": None, "tools": []})
        task = asyncio.create_task(VllmModelPort(client).generate(request, context))
        await asyncio.wait_for(ready.wait(), 2)
        if block:
            context.cancellation.set()
            with pytest.raises(CapabilityError):
                await task
        else:
            await task
        client.delta("content", "late")
        await asyncio.sleep(0)
        assert seen == ["early"]
    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_decoder_schema_anchors_patterns_so_text_fields_are_not_one_character():
    schema = ACTION_ADAPTER.json_schema()
    original = json.dumps(schema, sort_keys=True)
    fixed = decoder_schema(schema)
    question = fixed["$defs"]["AskUser"]["properties"]["question"]
    assert question["pattern"] == r".*\S.*"                       # was `\S`: a one-character full match
    ident = fixed["$defs"]["PlanItem"]["properties"]["offer_id"]["pattern"]
    assert ident.startswith("^") and ident.endswith("$") and not ident.startswith("^.*")   # already anchored
    assert json.dumps(schema, sort_keys=True) == original         # the harness's own schema is untouched

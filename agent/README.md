# agent/

Embeddable, bounded agent loop for the shopping advisor (Task 4).

The first runnable framework uses LangGraph `StateGraph`, Pydantic boundary
validation, and injected model/tool/plan-validation ports. The nine-node graph
supports checkpointed user questions and resuming the same run. Offline examples
cover both direct completion and question → answer → continued execution.

This package owns orchestration, action routing, tool dispatch, execution limits,
and result-validation gates. Model serving, knowledge acquisition/storage,
budget algorithms, SDK transport, web APIs, and infrastructure belong to the
corresponding project modules and integrate through those ports.

## Install and run

Python 3.10 or later is required. Run the following from the repository root.
Create an isolated environment and install the locked development dependencies:

```bash
python -m venv agent/.venv
agent/.venv/bin/python -m pip install -r agent/requirements.lock
agent/.venv/bin/python -m agent.examples.mock_run
```

On Windows, use `agent\.venv\Scripts\python.exe` in place of
`agent/.venv/bin/python`. The example itself needs no API keys, network access,
model weights, TACC job, or database. Installation requires package access.

All example products, prices, and evidence IDs are **synthetic**. The expected
result has `status: completed`, two model calls, one tool call, total cost
`49.99` USD, and remaining funds `50.01` USD. These amounts demonstrate the
interface, not a real shopping recommendation or the complete budget algorithm.

To view live node events as JSON lines:

```bash
agent/.venv/bin/python -m agent.examples.mock_run --events
```

To exercise a missing budget, a real graph interrupt, and a simulated user reply:

```bash
agent/.venv/bin/python -m agent.examples.mock_run --clarify --events
```

This example automatically submits the synthetic answer `purchase_cap=100.00` at
the application boundary. The stream first returns `run_waiting`, then continues
with `run_resumed` and finishes with `run_completed` on the same run ID. There
are three cumulative model calls and one tool call. No real user UI is provided.

The final `run_completed` event contains the structured result in its `result`
field. Events describe graph progress; they are not streamed model tokens.
Unreported model token usage is identified by `usage.unreported_token_calls`;
zero token sums do not mean that a real request consumed no tokens.

`requirements.txt` pins direct runtime dependencies; `requirements-dev.txt`
adds pytest. `requirements.lock` pins the full development dependency graph with
Python/platform markers. It was resolved for Python 3.10+ and exercised on
Python 3.14.7. Other interpreters still need their own runtime validation.
The project-wide manifest and test configuration are unchanged.

## Public API and integration

```python
import asyncio

from agent.examples.mock_run import demo_request, demo_runtime

async def main():
    runtime = demo_runtime()  # Explicit example dependencies; never a runtime default.
    result = await runtime.run(demo_request())
    print(result.model_dump_json(indent=2))

asyncio.run(main())
```

For real integration, construct `AgentRuntime(AgentDependencies(...), RunPolicy(...))`
with the following adapters:

| Capability | Required behavior |
| --- | --- |
| `ModelPort.generate(request, context)` | Async method returning `ModelReply`; normalize output to one schema-valid JSON action |
| `ToolExecutor.execute(arguments, context)` | Async method returning data matching its registered result model |
| `PlanValidator.validate(request, context)` | Return `ValidationResult`, echoing the proposal fingerprint; passed results contain authoritative `VerifiedPlan` data |

`ModelReply.finish_reason` must be normalized to `stop`, `length`, or `error`.
Incomplete generations cannot submit a plan even if their content parses as JSON.
Provider-specific native tool responses can be normalized to the same JSON
action format in the adapter.

Register tools using `ToolRegistry([ToolSpec(...)])`. Each specification supplies
a name, description, Pydantic argument/result models, executor, timeout, and
result byte limit. Both schemas must set `extra="forbid"`; arguments and results
are validated in strict mode. Unknown tools and invalid arguments never invoke
the executor. The registry rejects duplicate names and does not dynamically
import model-selected code.

Every call receives a trusted run ID, snapshot ID, operation ID, timeout and
cooperative cancellation signal through `CallContext`. Adapters should enforce
transport timeouts and wrap expected external failures in `CapabilityError`
with a user-safe message. Unexpected programming exceptions propagate.
Synchronous/blocking I/O must be handled by the adapter; an async declaration
does not make blocking code cancelable.

`run()` returns an `AgentResult` on completion or a checkpointed `waiting_user`
outcome. `astream()` starts a new run and yields `AgentEvent` objects; it does not
subscribe to an existing run. The checkpoint key is `agent:<run_id>`. Reusing a
run ID with `run()` raises `RunStateError(code="run_exists")`; use a distinct ID
for a new task. `conversation_id` groups runs and does not select a checkpoint.

For clarification, the caller shows `clarification.question` and submits a
structured answer matching the returned question ID:

```python
request = demo_request().model_copy(update={"constraints": {"currency": "USD"}})
runtime = demo_runtime()
waiting = await runtime.run(request)
result = await runtime.resume(
    waiting.run_id,
    waiting.pending_question.question_id,
    {"purchase_cap": "100.00"},
)
```

Use `astream_resume(run_id, question_id, answer)` to consume resumed node events.
The run retains its history, snapshot, original policy, call counters, cumulative
active execution time and event sequence. The interrupted node has no side
effects before `interrupt()`, so completed searches/model calls are not replayed
by an ordinary user-answer resume.

Answers must contain exactly the requested **top-level constraint fields**.
Unknown keys cannot change run identity, snapshot or policy. Empty values are
rejected; `purchase_cap` must be a nonnegative decimal string, `currency` an
uppercase three-letter code, and existing other constraints retain their JSON
type. These are boundary checks; the injected business validator still decides
whether a final shopping plan satisfies the full domain rules. Free-text answer
interpretation is not implemented. A bad answer produces a new question ID and
a bounded retry; only a valid answer is merged. Mismatched IDs are rejected
before graph execution. Old/duplicate answers cannot advance a completed run.

`RunStateError.code` distinguishes `run_not_found`, `question_mismatch`,
`not_waiting`, `run_exists`, `run_busy`, `not_interrupted` and
`incompatible_checkpoint`. The host owns authorization to access each run.

Use `contextlib.aclosing` when stopping
event consumption early so the background graph and cooperative calls are
cancelled promptly:

```python
from contextlib import aclosing

async with aclosing(runtime.astream(request)) as events:
    async for event in events:
        # Handle event; breaking here closes the generator on block exit.
        print(event.type)
```

For an explicit cancelled result, pass an `asyncio.Event` as
`cancellation=` to `run` or `astream` and set it when requested. Cancelling the
calling asyncio task instead propagates `CancelledError` after cleanup.

## Behavior and limits

Only `call_tool`, `propose_plan`, and `ask_user` are accepted. Recommendations
must pass the injected validator, and its receipt, snapshot and selections must
match the original proposal. The runtime never accepts model-generated totals
as verified amounts. Missing validation returns `incomplete`.

Defaults allow 8 model calls, 8 tool calls, 2 consecutive malformed-action
repairs, 2 plan repairs, 8 questions, 2 invalid-answer repairs, 120 seconds of
cumulative active execution, and 30 seconds per external call. Questions expire
after 1800 seconds of wall time; answers are bounded to 8192 UTF-8 JSON bytes.
Waiting time does not consume the execution budget. An expired answer finalizes
with `incomplete / question_expired`; expiry is checked on resume, without a
background expiration scheduler. Failed external attempts consume their call budget. Invalid arguments do
not consume a tool call because execution has not started. The repair usage
counters record malformed outputs and rejected proposals, including the final
rejection that exhausts a repair allowance.

Context construction preserves required instructions, request and schemas,
then selects a recent suffix of history within a character budget. Results
also have per-tool byte limits. Character limits are **not token limits**;
the real model adapter must account for tokenizer/chat-template overhead and
enforce its own input/output token limits. The reference service has a 4K context.

Runs have independent state and checkpoints. Injected adapters are shared by the runtime and
must themselves support concurrent calls if the host runs requests concurrently.
The optional event sink is synchronous, best-effort telemetry: keep it quick and
nonblocking. Sink errors are logged and do not retry external operations.

## Tests

```bash
agent/.venv/bin/python -m pytest agent/tests -q
agent/.venv/bin/python -m pytest tests agent/tests -q
```

The root pytest configuration discovers only `tests/` by default, so explicitly
include `agent/tests/`. Agent tests are entirely offline. The existing root tests
start loopback HTTP fixtures and need permission to bind local ports.

## Extension boundary

The runtime defaults to `InMemorySaver`, owned by that runtime instance. It
releases the invocation on interrupt and uses `Command(resume=...)` for the same
question. To resume through a new Runtime in the same process, pass the same
checkpointer object. Process restarts lose the default memory checkpoints.

The host may inject `checkpointer=` implementing `BaseCheckpointSaver`'s async
API. Its connections, lifecycle, authorization, retention, backups and production
acceptance belong to the host; no database adapter is installed by this package.
State version 2 rejects incompatible old checkpoints; migration is not provided.
The run's saved policy is used even if a new Runtime has different defaults.

A per-run lock rejects overlapping calls within one Runtime. Multiple Runtime
instances/processes sharing a saver **must be serialized by the host**, including
new-run creation and resume; this is not a distributed lock or exactly-once
external-operation system. Resuming arbitrary failed/aborted nodes is outside
the current user-question recovery contract.

Real inference, retrieval, SDK file transport, production budget validation,
distributed scheduling, external-operation deduplication and token-aware context
selection remain integrations. Pending model/tool handles still return
`incomplete / unsupported_capability`. No mock is selected automatically when a
real capability is absent.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and deferred capabilities.

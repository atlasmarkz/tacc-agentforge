# agent/

Embeddable, bounded agent loop for the shopping advisor (Task 4).

The first runnable framework uses LangGraph `StateGraph`, Pydantic boundary
validation, and injected model/tool/plan-validation ports. An offline example
completes the entire decision → tool → proposal → validation → result flow.

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

`run()` returns an `AgentResult`. `astream()` executes its own run and yields
`AgentEvent` objects, with per-run sequence numbers and a final result event.
It does not subscribe to an existing run. Use `contextlib.aclosing` when stopping
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
repairs, 2 plan repairs, a 120-second run deadline, and 30 seconds per external
call. Failed external attempts consume their call budget. Invalid arguments do
not consume a tool call because execution has not started. The repair usage
counters record malformed outputs and rejected proposals, including the final
rejection that exhausts a repair allowance.

Context construction preserves required instructions, request and schemas,
then selects a recent suffix of history within a character budget. Results
also have per-tool byte limits. Character limits are **not token limits**;
the real model adapter must account for tokenizer/chat-template overhead and
enforce its own input/output token limits. The reference service has a 4K context.

Runs have independent state. Injected adapters are shared by the runtime and
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

This milestone has immediate async capability calls and in-memory runs.
`ask_user` returns `waiting_user`; the host can collect missing data and start a
distinct new run. `resume()` and non-null `checkpointer=` explicitly raise
`UnsupportedCapabilityError`. Pending model/tool handles return
`incomplete / unsupported_capability`.

Real inference, retrieval, SDK file transport, production budget validation,
checkpoint persistence, durable resume, external-operation deduplication and
token-aware context selection are later integrations. No mock is selected
automatically when a real capability is absent.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and deferred capabilities.

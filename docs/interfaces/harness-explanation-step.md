# Interface proposal: streamed explanation step in the agent harness

Status: proposal for the harness owner. Nothing in `agent/` has been changed. Written 2026-11-08 by the
inference/deployment side (`inference/`), based on measurements from TACC jobs 28462, 28467 and 28469.

## 1. Why

A completed run today returns a structured `VerifiedPlan` (offer IDs, quantities, costs). The model never
writes text for the user: its output is one JSON action. Token streaming of that output is therefore partial
JSON, which a UI cannot show. A short natural-language explanation of the verified plan, generated after
validation, gives streaming something worth showing.

The explanation is a *run* artifact, so its progress belongs in the harness's own `AgentEvent` stream
(same `run_id`, same sequence numbers), not in a second event source the frontend would have to merge.

## 2. What the inference side already provides

| Capability | Where | Status |
| --- | --- | --- |
| SSE streaming with `on_delta(kind, text)`, `kind` is `content` or `reasoning` | `inference/client.py` `ModelClient.chat` | done, tested, verified on the real service |
| Delta callback delivered on the event-loop thread, tagged with `operation_id` | `inference/model_port.py` `VllmModelPort(on_delta=...)` | done, tested |
| Cancellation closes the connection, vLLM frees the slot | `CancelToken` | done |
| Retry only while nothing has been delivered; later breaks raise `StreamInterrupted` (recoverable) | client | done |
| Free-text mode (no schema, no constrained decoding) | `VllmModelPort` | **blocked** until the contract allows it (section 3.1) |

Measured on the real service (Qwen3.8-27B-AWQ-INT4, TP=2, CUDA graphs):

- Time to first streamed token: median 0.12 s (14 calls, with and without thinking), `smoke.json` in `inference/results/service-stream-smoke-28462`.
- Largest gap between two deltas inside one call: median 16 ms (thinking off) and 30 ms (thinking on).
- Streamed text was identical to the blocking answer in 14 of 14 calls (temperature 0).
- Full harness runs with mock tools: about 2 s with thinking off, about 5 s with thinking on (`inference/results/agent-e2e-28469`).
- Not measured yet: the length, latency and delta count of an explanation (no explanation step exists).

## 3. Requested changes in `agent/`

All new fields are optional with defaults, so existing runs, `mock_run` and `agent/tests` behave as before.

### 3.1 `contracts.py`

1. `ModelRequest.action_schema: dict | None = None`. `None` means "free text": the port must not apply
   constrained decoding. `tools` may be empty.
2. `CallContext` gets an optional streaming hook:
   `on_delta: Callable[[str, str], None] | None = None`, called as `on_delta(kind, text)`.
   Contract for the port: invoked on the event-loop thread, never blocks, exceptions are swallowed by the
   port. The port keeps returning the complete `ModelReply` at the end (`finish_reason`, token counts).
3. `AgentEvent.type` gains three values (the field is a free `str` today, so this is documentation plus tests):

   | type | `data` |
   | --- | --- |
   | `explain_started` | `operation_id` |
   | `explain_delta` | `kind` (`content`), `text` |
   | `explain_done` | `verified: bool`, `fallback: bool`, `reason` (code when `verified` is false), `chars: int` |

4. `AgentResult.explanation: Explanation | None = None`, with
   `Explanation(text: str, verified: bool, fallback: bool, cited_evidence_ids: list[Identifier])`.
   Only a `completed` result may carry one (extend `check_outcome`).

### 3.2 `graph.py` / `nodes.py`

New node `explain`, routed from `validate_plan` only when validation passed:
`validate_plan -> explain -> finalize`. Runs that end `waiting_user`, `incomplete`, `failed` or `cancelled`
skip it.

Behaviour of the node:

1. `scope.emit("explain", "explain_started", ...)`.
2. One model call through the existing `ModelPort`, free-text mode, with `context.on_delta` set to a function
   that emits `explain_delta`. Suggested limits in `RunPolicy`: `explain: bool = False` (off by default),
   `explain_timeout_s = 20`, `max_explain_tokens = 400`.
3. The call does **not** count against `max_model_calls` (that budget is for the decision loop).
4. Check the generated text (section 4). If the check fails, or the call fails or times out, use the
   deterministic template built from the verified plan. **An explanation failure never changes the run
   status**: the run stays `completed`, `explanation.fallback = true`.
5. `scope.emit("explain", "explain_done", ...)`, then `finalize` attaches `explanation` to the result.
6. Cancellation and the run deadline apply as for any other call. A cancelled run emits no `explain_done`.

Event volume: a decision action produced 17 to 49 content deltas in the end-to-end runs. The explanation is
expected to be larger (not measured). Optional: coalesce deltas (for example at least 50 ms or 8 characters
apart) before emitting, to keep the event queue and the browser stream small.

### 3.3 Inputs of the explanation (open: who provides them)

The prompt may only contain verified facts: the `VerifiedPlan` (items, quantities, currency, total,
remaining funds, cost breakdown), the user's message, and for each cited evidence ID the product title,
merchant, price, price age, excerpt and source URL. The first group is in the run state. The second needs a
new port, for example
`EvidenceResolver.resolve(evidence_ids, snapshot_id) -> list[EvidenceFact]`, supplied by the knowledge-base
side (Tasks 1 and 2, not built yet). Until it exists, the mock tools can return the facts in their results.

## 4. Faithfulness check (deterministic, harness side)

Run on the final text, using only the inputs from 3.3:

1. Number whitelist: every number and amount in the text must appear among the allowed values (total,
   remaining funds, breakdown values, unit prices, quantities, price ages). Otherwise `reason = "unlisted_number"`.
2. Citation check: every inline marker `[E1]`-style must map to an evidence ID given to the model.
   Otherwise `reason = "unknown_citation"`.
3. Optional: minimum length, and the text must not be only whitespace.

Evidence excerpts are untrusted data: fence them in the prompt and state that instructions inside them are
void. The explanation step has no tools and cannot modify the plan; the UI shows amounts from the structured
plan, not from the prose.

Streaming shows the text before the check completes. Suggested UI contract: label the text as a draft until
`explain_done` arrives; if `fallback` is true, replace it with `Explanation.text` from the result.

## 5. Acceptance tests (harness side)

- Event order: `explain_started`, one or more `explain_delta`, `explain_done`, then `run_completed`;
  concatenated deltas equal `Explanation.text` when `fallback` is false.
- Check failure (inject an unlisted number): fallback used, status still `completed`.
- Model error and timeout during explanation: fallback used, status still `completed`.
- Cancellation during explanation: no `explain_done`, status `cancelled`.
- `RunPolicy(explain=False)` (default): event sequence and result identical to today.
- A run that ends `waiting_user` or `incomplete` emits no explain events.

## 6. Port-side work that follows (inference side)

Once 3.1 is merged, `VllmModelPort` will: skip constrained decoding when `action_schema is None`; default
thinking off for free-text requests (configurable); forward `context.on_delta`; keep today's error mapping.
Tests use the fake streaming server in `tests/fake_vllm.py`; a TACC check repeats the measurements in
section 2 for the explanation call.

## 7. Related findings from the end-to-end runs (for the harness owner)

These are independent of the explanation step.

1. **Schema patterns and constrained decoding.** `ask_user.question` is declared with `pattern=r"\S"`.
   JSON Schema treats a pattern as "contains a match", but the vLLM constrained decoder implements a full
   match, so the question was a single character ("您", "请"). The adapter now sends the decoder a copy of
   the schema with patterns anchored (`inference.model_port.decoder_schema`); the harness schema is
   unchanged and still validates the reply. Any new `pattern` in an action schema is affected the same way.
2. **Missing budget.** With no `purchase_cap` in the constraints the model proposed a plan instead of asking,
   the validator returned `unavailable`, and the run ended `incomplete / validation_unavailable` without ever
   asking the user (in every such run: 4 per job, 8 across jobs 28467 and 28469, both thinking modes). Consider routing `unavailable` with a
   `missing_budget` issue to `ask_user`, or prompting for it explicitly.
3. **Pause and resume.** `interrupt()` / `Command(resume=...)` works with the current state shape
   (experiment, not committed). Pitfalls: resuming an unknown thread raises `KeyError: 'usage'`; the
   `run_waiting` event has to be emitted from the `__interrupt__` result; nothing before `interrupt()` in the
   node may have side effects; the 8-call model budget is shared across resumes; `InMemorySaver` is not
   persistent.
4. **Environment on TACC.** The course vLLM environment has no `pip`. `agent/requirements.txt` was installed
   into `USERDIR/agent-pylib` with a recent pip bootstrapped from the user's Miniconda
   (`deploy/tacc/service/agent-e2e/job.sh`). Decide how the harness's dependencies are managed for delivery.

## 8. Questions for the harness owner

1. Is `action_schema: None` the right way to request free text, or do you prefer an explicit
   `mode: Literal["action", "text"]`?
2. Should the check (section 4) live in the harness or in the plan validator?
3. Fallback text: template built from the verified plan, or no explanation at all?
4. Who implements `EvidenceResolver`: the knowledge-base owner?

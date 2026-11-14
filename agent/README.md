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
field. Decision events describe graph progress; optional explanation events carry
user-facing content deltas.
Unreported model token usage is identified by `usage.unreported_token_calls`;
zero token sums do not mean that a real request consumed no tokens.

`requirements.txt` pins direct runtime dependencies; `requirements-dev.txt`
adds pytest. `requirements.lock` pins the full development dependency graph with
Python/platform markers. It was resolved for Python 3.10+; persistence/restart
tests were exercised on Python 3.12.15 and 3.14.7 on macOS. Other platforms and
interpreters still need their own runtime validation.

## Agent 配置

Agent 专用参数放在新增的 [configs/agent.yaml](../configs/agent.yaml)，每项都有中文注释。
读取和校验在 `agent/config.py`、`agent/settings.py` 内完成，PyYAML 仅列入 Agent 依赖。
原有知识库配置、模型服务配置、训练配置和根项目依赖保持原样；本文件不启动或配置这些服务。

```python
from agent.settings import load_runtime_settings

settings = load_runtime_settings(profile="tacc-thinking")
runtime = AgentRuntime(dependencies, settings.policy, checkpointer=checkpointer)
# 宿主可将 settings.model.port_options() / client_options() 传给现有模型适配器/客户端。
```

| profile | 思考 | 输出 token | 上下文字符 | 用途 |
|---|---|---|---|---|
| `tacc`（默认） | 模型默认 | 2048 | 36000 | 为默认启用思考的 Qwen3 留出 reasoning 额度 |
| `tacc-thinking` | 开启 | 2048 | 36000 | 真实工具与思考模式 |
| `tacc-fast` | 关闭 | 512 | 36000 | 关闭思考时较短的动作输出 |
| `local` | 开启 | 2048 | 36000 | 本地 HTTP 示例 |
| `mock` | 不调用真实模型 | 512 | 16000 | 保持离线示例的轻量配置 |

参数依据用户提供的 TACC 作业 28560/28561 结果。上下文默认保留最多 32 条历史消息，
并受 36000 字符预算约束；任意更长历史仍会裁剪。2048 包含 reasoning 和最终内容。
36000 字符约 12K token 仅是估计，不能保证实际分词后适配外部服务窗口。
`model.context_window_tokens` 是宿主的提示/检查值，不修改 vLLM 服务配置，需由部署方确认。

Agent 示例、持久化示例和清理命令支持 `--config` / `--profile`，也可使用
`AGENT_CONFIG` / `AGENT_PROFILE`。命令行显式参数覆盖 YAML；未指定参数不会覆盖文件设置。
配置里的相对路径以 `configs/` 上一级为基准，自定义文件放其他目录则以其所在目录为基准。
路径中的环境变量在使用时展开，未定义时报错。本地 `.env` 的地址/模型路径覆盖方式保留。
未知 profile、循环继承、重复键、无效策略值以及其他任务的配置节都会报错。

库级 `RunPolicy()` 保持兼容，也不会隐式读 YAML；真实工具宿主应显式传入 `settings.policy`。
恢复任务继续使用检查点中原有策略，不因修改文件而重置额度或过期时间。
本轮仅提供 Agent 侧配置入口，不自动修改其他任务的启动/打包脚本。独立分发 Agent 示例时，
需同时提供 `configs/agent.yaml` 或通过 `--config` 指定文件。

## Optional streamed explanation

Set `RunPolicy(explain=True)` to route passed validation through
`validate_plan -> explain -> finalize`. The default is off: no extra model calls
or events, and all pre-existing result fields retain their meaning. New optional
result/usage fields are serialized with their defaults.

```bash
agent/.venv/bin/python -m agent.examples.mock_run --clarify --explain --events
```

The explanation uses the existing `ModelPort` with `action_schema=None` and
`tools=[]`. The model adapter must disable constrained decoding for this request.
`CallContext.on_delta(kind, text)` is a per-call callback on the event-loop thread;
ports isolate callback errors and return the complete `ModelReply`. The vLLM
adapter drains queued callbacks before returning and drops late callbacks after
completion/cancellation. Its legacy constructor callback remains available for
monitoring. Free-text thinking defaults off unless explicitly configured.

`explain_started` includes the operation ID `<run_id>:explain:1`; `explain_delta`
contains only `kind="content"` and `text`; `explain_done` includes `verified`,
`fallback`, `reason`, and final `chars`. These use the same run ID and sequence
as other agent events. Reasoning is not forwarded into the explanation stream.
Deltas coalesce at 8 characters or 50ms, are bounded by `max_explain_chars=8000`,
and concatenate exactly to a successful `Explanation.text`. A nonstreaming port
gets one synthesized content delta. Consumers must treat deltas as drafts until
done; on fallback replace the draft with the explanation in the final result.

Defaults allow one explanation model call, 400 output tokens, and 20 seconds
for evidence resolution plus generation together. This does not consume the
decision budget `max_model_calls`: it is recorded as `usage.explain_model_calls`.
Returned token usage and unreported attempts count in the existing totals; active
time still counts toward the run deadline. A declared capability error, an
explanation timeout, incomplete generation, invalid text/citation/number or an
unusable evidence response falls back to a template built only from VerifiedPlan.
The run remains completed, with `fallback=True, verified=False`. `reason` records
the cause in the done event. Unexpected programming errors propagate.

Global cancellation/deadline keep their existing semantics: a cancelled run is
cancelled, a run whose overall deadline expires is incomplete/deadline_exceeded,
and both clear the plan/explanation. Cancellation during explanation emits no
`explain_done`. Only completed results may contain an explanation.

`verified=True` means generated text passed the deterministic harness checks,
not that every natural-language assertion was proven. The checks compare numeric
values using Decimal and map inline `[E1]` citations to the plan's evidence IDs;
they cannot detect all invented features or swapped roles of otherwise allowed
amounts. User text and source excerpts do not extend the numeric whitelist.
The UI must display authoritative amounts from the structured plan.

An optional `AgentDependencies.evidence_resolver` supplies bounded `EvidenceFact`
observations. Without one, generation uses only the verified plan and user
message; a configured resolver that fails causes fallback. Facts must cover the
plan's offer/evidence pairs and snapshot. Raw excerpts are untrusted data, not
instructions. See `knowledge.explanation.KnowledgeEvidenceResolver` for the real
snapshot adapter; the offline demo injects `MockEvidenceResolver` explicitly.
Hosts close the real resolver only after active calls finish. Explanation prompts
obey the existing context character limit; overflow falls back without dropping
required verified facts. Actual token/context checks remain the port's job.

Final explanation, usage and event sequence are checkpointed; callbacks and draft
tokens are not persisted. Version-2 checkpoints without new fields retain default
`explain=False` on resume, even if the new host opts in. A waiting run that saved
`explain=True` retains that policy across a SIGKILL/restart. Arbitrary mid-explanation
crash recovery and persisted event replay are outside the user-question resume API.

## 本地 Qwen3.5-4B 全流程测试

Apple Silicon 可复用现有 `agent/.venv`，使用 MLX 加载
`~/models/qwen/qwen3.5-4b` 中的原始权重。该可选测试环境需要 Python 3.11+；
不会改变 Agent 的生产接口、TACC 连接方式或模型文件。

在项目根目录安装可选依赖：

```bash
uv pip install --python agent/.venv/bin/python -r agent/requirements-local.txt
```

在一个终端启动仅监听本机的模型服务，离线选项禁止补下载模型：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
TOKENIZERS_PARALLELISM=false agent/.venv/bin/python -m mlx_lm server \
  --model "$HOME/models/qwen/qwen3.5-4b" \
  --host 127.0.0.1 --port 18080 --max-tokens 2048 --temp 0.6 \
  --chat-template-args '{"enable_thinking":true}' \
  --decode-concurrency 1 --prompt-cache-size 1 --prefill-step-size 512
```

在另一个终端运行两个真实模型场景：

```bash
agent/.venv/bin/python -m agent.examples.local_run \
  --base-url http://127.0.0.1:18080/v1 \
  --model-path "$HOME/models/qwen/qwen3.5-4b"
```

入口默认读取根目录 `.env`，不会导出或打印其中的密钥，也不会写回配置。
配置优先级为显式参数、进程环境变量、`.env`、默认值。
可选配置项为 `LOCAL_MODEL_BASE_URL`、`LOCAL_MODEL_PATH`、`LOCAL_MODEL_ID`；
地址还会回退到 `QWEN_BASE_URL`。当前 `.env` 若指向远端模型，需要像上述命令一样
显式传入本机地址。入口会拒绝远端地址和带凭证的 URL。

默认开启思考模式、温度 `0.6`、每次最多输出 `2048` 个令牌（包含思考令牌），
可通过 `--no-thinking`、`--temperature`、`--max-tokens` 调整。
这里复用 `inference.model_port.VllmModelPort` 的兼容 HTTP 接口；MLX 服务不提供
`response_format` 约束解码，因此只在此示例中设置 `constrain_output=False`。
Harness 仍严格校验 JSON、动作和最终方案；格式错误会携带有界原回复与校验原因
返回模型，超出原有修复上限仍按失败收尾。不会人工补全模型 JSON。

两个场景分别为：

- `direct`：已知预算 → 真实模型决策 → 模拟搜索 → 真实模型提交方案 → 模拟业务校验 → 完成。
- `clarify`：缺少预算 → 真实模型提问 → 检查点挂起 → 示例自动提交
  `purchase_cap=100.00` → 恢复同一任务 → 搜索、方案校验、完成。

可用 `--scenario direct` 或 `--scenario clarify` 单独执行。商品、报价、证据和业务
校验器使用 `mock_run.py` 的模拟实现，模型推理是真实调用。这里验证 Agent 编排链路，
没有调用 SerpAPI、生产知识库、正式预算算法或 TACC 作业。

每次输出独立目录 `agent/results/local-qwen35/<UTC时间>/`，包含 `report.json`、
各场景 `*-model-calls.json` 原始回复与令牌用量、`*-events.jsonl` 节点事件。
只有实际收到模型回复、执行工具、通过方案校验、完成预期问答恢复，且最终状态为
`completed` 才判定通过；全部通过时退出码为 `0`，否则为 `1`。
输出目录已加入 `agent/.gitignore`。测试后可在服务终端按 Ctrl+C 释放模型内存。

2026-11-10 在本机 Python 3.14.7 / MLX-LM 0.32.2 上的实测结论：权重加载、
真实模型调用、搜索工具、提问挂起及回答恢复均已执行；直接场景和问答场景都未通过
最终验收。模型多次返回缺少外层括号、尾随逗号或多余字段的方案，有限修复后以
`incomplete / invalid_action` 收尾。思考与非思考模式均观察到格式问题。
因此本入口可用于复现和检查 Agent 链路，目前不能把这套未提供约束解码的本地
模型组合视为已经完整跑通。要进一步验证稳定的结构化输出，需要模型服务侧实现
真正的 JSON Schema 约束解码，或继续改进模型输出能力；仅开启不被服务实现的
`response_format` 参数不会解决问题。

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
enforce its own input/output token limits. The Agent YAML's 16384-token window is
a host-side hint; the service keeps its separate configuration. The 36000-character
budget is an estimate and cannot prove the tokenized input fits.

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
API. The package includes a SQLite adapter; `agent.checkpoints.open_checkpointer`
owns its connection and an exclusive host lease on Linux/macOS. Keep this context
open for the host's lifetime, including all run/resume calls:

```python
from pathlib import Path
from agent.checkpoints import open_checkpointer
from agent.checkpoint_cleanup import cleanup_checkpoints

async with open_checkpointer(Path("/persistent/agent.sqlite")) as saver:
    # Run once before accepting work, while this host owns the store exclusively.
    await cleanup_checkpoints(saver)
    runtime = AgentRuntime(dependencies, policy, checkpointer=saver)
    # await runtime.run(request), or resume using the original run/question IDs
```

The full checkpoints, interrupt writes, history, original policy, counters and
event sequence survive process restarts. `run_waiting` is emitted only after the
graph invocation finishes saving its interrupt; storage failures propagate.
Opening a store already leased by another host/cleanup process fails. The lock
file stays in place after release. Hosts bypassing this helper must provide their
own serialization. Authorization and backups remain the host's responsibility.
State version 2 rejects incompatible old checkpoints; migration is not provided.
The run's saved policy is used even if a new Runtime has different defaults.

SQLite uses WAL with `synchronous=FULL`. Use a **supported persistent filesystem**;
WAL is not supported on network filesystems. Linux detection rejects known shared
filesystems (including NFS, Lustre and GPFS), but this is not a complete storage
certification. Confirm persistence, locking and sync semantics with the platform.
For unsuitable shared volumes, inject a different durable saver, such as one
backed by a reachable PostgreSQL service. Do not relocate the database to a
node-local temporary directory if runs must survive job/node changes.

Try a deterministic restart example in separate processes:

```bash
agent/.venv/bin/python -m agent.examples.persistent_resume \
  --db /persistent/agent.sqlite start --run-id restart-demo --hold-after-waiting
# Once the waiting_user JSON is printed, kill that process (SIGKILL is supported).
# Copy pending_question.question_id from its output into the next command:
agent/.venv/bin/python -m agent.examples.persistent_resume \
  --db /persistent/agent.sqlite resume --run-id restart-demo \
  --question-id QUESTION_ID --answer '{"purchase_cap":"100.00"}'
```

All model/tool/validator capabilities in this example are explicit synthetic
fixtures; it tests real LangGraph/SQLite recovery without a model service.
`--events` emits ordered event JSON lines. Use a new ID for each new task and
keep the old ID for resume. The example's question TTL is 7200 seconds; existing
Runtime defaults remain 1800 seconds. Neither restart nor cleanup extends TTL.

Cleanup uses the latest root checkpoint per `agent:<run_id>` and deletes the
entire thread and its writes. Defaults retain terminal runs for seven days and
expired questions for 24 hours after their saved expiry. Unexpired questions,
unfinished runs, unrecognized state versions and unrelated thread prefixes are
retained. Startup cleanup runs before accepting work; there is no background
scheduler. Preview or run it manually while the host is stopped:

```bash
agent/.venv/bin/python -m agent.checkpoint_cleanup --db /persistent/agent.sqlite --dry-run
agent/.venv/bin/python -m agent.checkpoint_cleanup --db /persistent/agent.sqlite \
  --terminal-retention-s 604800 --expired-question-grace-s 86400
```

The CLI requires an existing database and takes the same exclusive lease. Errors
are not converted into missing-run responses. A missing/cleaned run returns
`RunStateError(code="run_not_found")`; no tombstones distinguish those cases.
An expired question still retained in storage resumes to
`incomplete / question_expired`. Wrong question IDs and completed runs preserve
the existing `question_mismatch` / `not_waiting` errors.

A per-run lock rejects overlapping calls within one Runtime. Multiple Runtime
instances/processes sharing a saver **must be serialized by the host**, including
new-run creation and resume; this is not a distributed lock or exactly-once
external-operation system. Resuming arbitrary failed/aborted nodes is outside
the current user-question recovery contract.

`agent/tests/test_persistence.py` kills a child with SIGKILL while its SQLite
connection is open, resumes from a fresh child, checks original limits/sequence,
and confirms completed searches are not replayed. It also covers whole-run
cleanup, dry-run, expired questions, duplicate answers, leases and storage errors.
Local tests do not establish TACC filesystem support or cross-job acceptance;
see the deployment guide in `deploy/tacc/service/agent-e2e/README.md`.

Real inference, retrieval, SDK file transport, production budget validation,
distributed scheduling, external-operation deduplication and token-aware context
selection remain integrations. Pending model/tool handles still return
`incomplete / unsupported_capability`. No mock is selected automatically when a
real capability is absent.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and deferred capabilities.

## 预算校验器（Task 3）

`agent/budget.py` 的 `BudgetPlanValidator` 实现 `PlanValidator` 端口：金额只用 `Decimal` 计算，不经过模型。
它不导入知识库，报价事实由 `OfferCostSource` 提供；知识库侧适配器是 `knowledge/budget_source.py`。

```python
from agent.budget import BudgetPlanValidator
from knowledge.budget_source import KnowledgeOfferCostSource

source = KnowledgeOfferCostSource(service)          # 宿主在停机时调用 source.close()
validator = BudgetPlanValidator(source)   # 默认按不含税（pre_tax）比较
```

**状态**：`passed` 附带 `VerifiedPlan`；`failed` 表示规则问题，每个问题点名需要的字段或原因，模型可据此换方案或追问；
`unavailable` 只用于读取报价失败，运行会终止。

**预算输入**（`AgentRequest.constraints`，金额为十进制字符串）：`currency`（必填，不换算）、`purchase_cap`、
`periodic_budget` 减去 `spending` 和 `reserve`（后两项默认 0）。两种限额同时给出时，可用资金取较小者；总价恰好等于可用资金时通过。

**税费口径**（`budget_basis`，可由约束覆盖校验器的 `default_basis`）：项目已决定**按不含税比较**，所以默认是 `pre_tax`。
口径决定税费是否计入，方案里每个条目都按同一口径计算：

- `pre_tax`（默认）：只计标价和已知运费，税费一律不计入（已知的税费也不计），`cost_breakdown` 不含 `tax`。回答里必须注明合计不含税；
  `agent/explanation.py` 的模板回退在 `cost_breakdown` 没有 `tax` 时会自动加上「（不含税）」，模型自己写的解释目前没有这个强制检查；
- `all_in`：每个条目都计税：标价含税计 0，标价不含税且税费已知时按单价同样的方式乘数量；任何条目税费未知，或不知道标价是否含税，
  就不通过（`tax_unknown`），不估算。`cost_breakdown` 总是含 `tax`。当前快照的税费全部未知，所以这个口径下没有方案能通过。

因此 `cost_breakdown` 是否含 `tax` 与口径一一对应。

**其他规则**：运费未知、币种不一致、价格冲突、规格版本冲突、证据不属于所选商品、方案没有引用价格证据，都不通过。
运费按 `shipping_scope` 计算：`item` 乘数量，`order` 每个条目计一次（合并运费时是偏高的上界），范围未知时数量必须为 1。
SKU 身份默认不阻止通过，`require_verified_identity=True` 可收紧。

**尚未包含**：对硬性规格要求（如容量、降噪）的校验、价格新鲜度策略，以及 `VerifiedPlan` 中记录所用口径的字段
（目前由 `cost_breakdown` 是否含 `tax` 判断，两者一一对应；模型写的解释不会被强制注明口径）。
另外，模型看到的提交标准比校验器更严：`get_offer_facts` 的说明写着费用未知不能确认可负担，而默认的 `pre_tax`
放行税费未知、身份未验证，以及数量为 1 时的运费范围未知；`complete_total_minor` 也是含税口径。是否把口径告诉模型
（工具说明、系统提示或校验反馈）与 issue #21 的修复一起决定。

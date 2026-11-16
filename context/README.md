# 上下文模块

独立 Python 3.10+ 库，核心只依赖标准库。管理跨运行的结构化会话需求，构造**完整当前运行轨迹**，
并可选携带上一轮已验证方案的 ID 参考。不自动运行 Agent、调用模型或知识库，也不更改现有 Agent 调用点。

## 快速开始

```python
from context import ContextManager, ContextPatch, ContextRequest, SQLiteStore

# 父目录由宿主创建；不要将个人会话数据库提交到仓库。
manager = ContextManager(SQLiteStore("/tmp/shopping-context.sqlite"))
state = manager.update_memory(
    "conversation-1",
    ContextPatch(set={"purchase_cap": "1000.00", "currency": "HKD",
                      "requirements": ["通勤", "降噪"]}),
    snapshot_id="snapshot-1", operation_id="turn-1", expected_revision=0,
)
state = manager.update_memory(
    "conversation-1", ContextPatch(set={"purchase_cap": "800.00"}),
    snapshot_id="snapshot-1", operation_id="turn-2",
    expected_revision=state["revision"],
)
# 宿主把同一份需求绑定到请求，避免存储与模型/校验器使用不同版本。
bundle = manager.build(ContextRequest(
    conversation_id="conversation-1", run_id="run-2", snapshot_id="snapshot-1",
    message="预算降到800，其他不变", system_prompt="You are a shopping advisor.",
    tools=[], action_schema={}, model_window_tokens=16384, max_output_tokens=512,
    expected_revision=state["revision"], expected_constraints=state["constraints"],
))
assert bundle.constraints["requirements"] == ["通勤", "降噪"]
print(bundle.payload, bundle.token_count)
```

直接使用 core 的调用方可以同时省略 `expected_revision` 与 `expected_constraints`，此时按存储中的
最新需求构造。两者必须成对出现；只提供一个会被 `ValueError` 拒绝。示例模型窗口只是显式配置，
实际窗口必须与服务一致。`build` 只读，失败不会撤销已保存需求。无持久化需求时把 store 换成
`InMemoryStore()`。

## 存储与更新契约

- `get(id)` 返回隔离副本；不存在返回 `None`。新会话的期望版本是 0，首次成功写入后为 1。
- 所有成功的新写操作递增会话 `revision`。仅需求实际变化递增 `constraints_revision`；
  同值更新和删除不存在字段不使需求失效。
- `ContextPatch.set` 整体替换顶层字段，`unset` 删除字段，不递归合并列表或对象；
  未提及字段保留。`None` 是显式值。不能修改身份、策略和运行控制字段。
- 每次写入必须指定稳定的 `operation_id` 和读到的 `expected_revision`。
  同 ID、同请求内容返回原操作的结果（可能早于最新状态）；需最新状态时调用 `get`。
  同 ID 不同内容报 `idempotency_conflict`；版本冲突报 `revision_conflict`，宿主重读后明确重试。
- SQLite 将状态和幂等回执原子保存；使用短连接及 `BEGIN IMMEDIATE`，支持多个实例争用。
  回执仅保存元数据和历史前缀长度，重试从只追加的历史中还原原始结果，避免重复存储累计消息。
  会话和回执 JSON 会在读取时校验形状与身份；形状损坏、会话身份不一致等具体原因会保留在
  `StorageError` 的消息中，而不是被替换成通用错误。
  内存实现用锁遵循相同契约。数据库忙、损坏或路径不可用时报 `StorageError`，不切换到内存。
  数据库格式版本为 1，未知版本拒绝打开。SQLite 文件的目录、备份、权限和保留周期由宿主管理。
- `delete(id, operation_id=..., expected_revision=...)` 删除会话和全部回执；重复删除不存在的会话安全返回。
  删除后开启新的幂等生命周期，宿主应防止旧请求在重新创建会话后重放。

### 容量统计

`ContextManager.stats(conversation_id)`（两种存储返回值一致）报告：

```python
{"conversation_id", "revision", "constraints_revision", "history_blocks",
 "receipt_count", "state_bytes", "receipt_bytes"}
```

会话不存在时返回 `None`。`state_bytes` / `receipt_bytes` 是规范化 JSON 内容的 UTF-8 字节数，
不是 SQLite 文件的实际磁盘占用。SQLite 在一个只读事务中读取状态和回执，内存实现持锁读取，
因此同一次统计的状态版本与回执计数来自同一观察点。

本版**保留全部回执**，不设 TTL、自动淘汰或默认数量上限；任意旧 `operation_id` 仍可按原结果重放。
回执数随操作数增长，长期存储不是有界的。宿主负责监测容量、归档旧会话，并调用 `delete` 删除整个
会话；后续使用新的 `conversation_id`。不要通过删除部分旧回执来换取空间，那会破坏幂等去重。

## 历史、方案验证与完整轨迹

`append_history(id, HistoryBlock(...), operation_id=..., expected_revision=...)` 保存调用方分组好的完整单元。
`HistoryBlock` 的消息只接受 `user` / `assistant`，每个块带稳定 `block_id`、`run_id`、`snapshot_id`、
`constraints_revision`、`kind`、可选商品/证据 `references`；追加顺序定义新旧。
工具动作与观察必须放入同一块。块 ID 在会话中不可重复。

`HistoryBlock.plan_validation` 只能是 `unvalidated`（默认）或 `passed`，且只有 `kind="plan"` 的块
可以声明 `passed`。跨运行历史方案参考必须同时满足：块是 plan、`plan_validation="passed"`、
且带有合法的 `plan_items`。未验证、被拒绝、校验不可用或没有条目的方案都不会导出为参考。
旧记录缺少 `plan_validation` 时按 `unvalidated` 读取，读取路径不会回写旧历史前缀；默认值也不会进入
新的 append 持久化形状或幂等摘要，因此旧 append 请求重试不会变成 `idempotency_conflict`。
显式 `passed` 会进入摘要，同一 operation ID 不能把通过状态改掉。

普通 `history_block_from_agent()` 只做结构转换，**不会**从模型正文、工具观察或 “PASSED” 文本推断
验证通过。可信宿主必须使用实际完成结果调用：

```python
from context.agent_adapter import validated_plan_block_from_agent

block = validated_plan_block_from_agent(
    completed_result, agent_request, block_id="plan-2", constraints_revision=revision,
)
manager.append_history(conversation_id, block, operation_id="plan-2", expected_revision=current)
```

该函数要求 `result.status == "completed"`、存在 `verified_plan`、result 与 request 的
`run_id` / `conversation_id` / `snapshot_id` 一致、verified plan 的 `snapshot_id` 一致，并校验条目。
它只追加新块，不修改既有历史。跨运行参考仍只包含 `offer_id`、`quantity`、`evidence_ids`，
不继承旧金额、可负担性结论或旧运行的 `verified_plan`；新一轮仍须重新读取事实并重新验证。

### 当前运行完整保留

- 通过当前快照、需求版本和候选句柄有效性检查的**本运行全部历史块**，按追加时间顺序作为必需上下文。
- 不按相关性删除当前运行的搜索、工具结果、失败反馈或澄清记录，也不会出现“跳过中间大块、
  保留两端小块”的预算空洞。
- 完整必需上下文仍超限时直接报 `ContextLimitError`，不裁掉早期步骤继续调用。
- 上一轮最近的**已验证**方案引用是唯一可选内容；放不下时整体舍弃并在 `omitted` 标记 `budget`。
- `filtered` / `unfiltered` 参数继续接受，但本版没有实质筛选差异，不能作为有效的筛选消融实验。
  有意义的过滤对照需要未来拆分执行轨迹与证据正文，本版不实现摘要或独立证据压缩。
- `references` 只作为显式 ID 元数据保存；本版没有自然语言子串匹配，因此 `o1` 不会命中 `o10`。
- 候选句柄依赖知识服务的进程内状态。新任务不读取旧任务历史；新 `ContextManager` 实例也不会复用
  旧实例写入的含 `candidate_set_id` 的块，报告 `needs_search`。对应原始消息和搜索 filters 仍保存在历史中，
  宿主可通过 `get` 读取并重新调用 `search_products`。即使 Manager 没有重建，知识服务也可能淘汰句柄；
  宿主须处理 `unknown_candidate_set` 并重新检索，模块不保证句柄存活。
- 显式修改快照通过 `update_memory(..., snapshot_id=新快照)` 完成；即使切回旧快照，也不会恢复旧派生内容。
  追加过期快照或过期需求版本的块报 `stale_history`。

`ContextBundle` 返回选中 ID、舍弃原因、版本、计数方法、模式及选择原因，诊断不附带舍弃正文。

## Token 计数与最终消息渲染

默认 `render_messages(payload)` 与当前 `inference.model_port.to_chat_messages` 的工具/schema 前缀一致，
工具定义须包含 `name`、`description`、`parameters`，按 Agent 字段声明顺序渲染。
只在首条 system 消息附加这些可信定义。用户需求和工具结果不会进入 system 消息。
`bundle.payload` 仍是**未渲染**的请求，避免 Agent 推理适配器重复添加前缀；
`bundle.rendered_messages` 用于检查和计数。

默认估算为：

```text
ceil(最终消息内容的 UTF-8 总字节数 / 3) + 16 × 消息数 + 16
```

返回 `mode="estimated"` 和 `counter_id="utf8-bytes-div3-plus16-v1"`。原先“一字节约一 token”的保守估算
保留为显式可选函数 `conservative_estimate_tokens`（`counter_id="utf8-bytes-plus16-v1"`）：

```python
from context import ContextManager, conservative_estimate_tokens

manager = ContextManager(store, token_counter=conservative_estimate_tokens)
```

估算不保证真实模型窗口；不同 tokenizer 和 chat template 可能产生不同 token 数。生产接入应注入对应
模型及版本的精确计数器；计数器须包含 chat template 和 generation prompt：

```python
from context import TokenCount

def count_qwen(messages):
    # tokenizer 由宿主加载；额外依赖不属于 context 核心。
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    return TokenCount(len(ids), "exact", "checkpoint-and-template-revision")

manager = ContextManager(store, token_counter=count_qwen)
```

注入精确计数器时始终使用精确计数器，不会静默退回估算；计数器异常直接传播。
自定义 `renderer(payload) -> list[dict]` 必须与最终模型端口使用的消息格式一致。
计数针对完整最终渲染消息，不假设 tokenizer/chat template 的 token 数可以按消息或块相加；
工具说明、动作 schema、chat template 和生成前缀都参与实际计数。

预算装入最多执行两次完整计数：先算“完整当前轨迹＋可选历史方案”，放得下直接复用并返回；
放不下则移除可选方案，再对必需轨迹计数一次，仍超限报 `ContextLimitError`。没有可选方案时只计一次。
始终预留 `max_output_tokens + safety_tokens`（安全余量默认 256）。

## 可选 Agent 转换与宿主职责

`context.agent_adapter` 提供 `request_from_agent`、`history_block_from_agent`、
`validated_plan_block_from_agent` 和 `to_model_request`。
只有最后一个函数在调用时导入 Agent/Pydantic。转换工具对实际 Agent 的动作/观察包装检查配对、工具名及快照；
成功观察必须包含身份和 data；错误观察接受 Agent 的 `status=error`、`code`、`message` 形式。
畸形 `propose_plan`（缺失/错误的 plan 容器、缺失/错误/空的 items 列表、非法条目）统一抛 `ValueError`，
不会泄漏 `KeyError`、`AttributeError` 或 `TypeError`。user-role 修复反馈中的 `rejected_output` 原样保留，
不会被重新识别成合法方案。不猜测整份 `AgentState.history` 的分组边界，也不触发 Agent 执行。

`request_from_agent(...)` 现在要求调用方显式传入 `expected_revision`，并把 `AgentRequest.constraints`
复制为 `expected_constraints`。`build()` 会检查：

- 会话版本不一致 -> `revision_conflict`；
- 约束内容不一致 -> `constraints_mismatch`。

它不会静默丢弃请求约束，也不会自动合并或覆盖存储。固定宿主流程：

```text
持久化上下文 patch
→ 用 update_memory 返回的需求构造 AgentRequest
→ 绑定最新 revision 构造 ContextRequest
→ 模型调用与确定性校验器使用同一份请求/需求
```

当前 `AgentRuntime` 仍只使用 `agent/prompts.py` 和自身 LangGraph checkpoint。先在外部调用
`ContextManager.build()` 再调用 `runtime.run()` **不会**自动启用本模块；本轮只解决独立模块的权威契约，
不接入或叠加两套上下文裁剪路径。

宿主仍需负责鉴权、个人数据处理、任务取消/超时、模型调用、重新检索、确定性预算验证，以及
回执容量监测/归档。异步宿主可通过线程执行 SQLite 操作，避免阻塞事件循环。

## 验证

```bash
.venv/bin/python -S -m context.examples.offline
.venv/bin/python -m pytest tests/test_context.py -q
.venv/bin/python -m pytest tests agent/tests -q
```

离线示例在内存和临时 SQLite 中演示需求保留、可信已完成方案转换、历史失效、重新观察、完整轨迹构造、
两种模式的一致结果及容量统计。示例数据为合成数据，不代表已通过真实预算验证的商品。

本版尚未测得 token 节省、延迟或任务成功率改善；估算也不保证真实模型窗口。

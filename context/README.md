# 上下文模块

独立 Python 3.10+ 库，核心只依赖标准库。管理跨轮结构化需求，并按模型预算选择完整历史单元。
不自动运行 Agent、调用模型或知识库，也不更改现有 Agent 调用点。

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
bundle = manager.build(ContextRequest(
    conversation_id="conversation-1", run_id="run-2", snapshot_id="snapshot-1",
    message="预算降到800，其他不变", system_prompt="You are a shopping advisor.",
    tools=[], action_schema={}, model_window_tokens=16384, max_output_tokens=512,
))
assert bundle.constraints["requirements"] == ["通勤", "降噪"]
print(bundle.payload, bundle.token_count)
```

示例的模型窗口只是显式配置示范，实际窗口必须与服务一致。`build` 只读，失败不会撤销已保存需求。
无持久化需求时将 store 换成 `InMemoryStore()`。

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
  会话和回执 JSON 会在读取时校验形状与身份，损坏内容报告存储错误。
  内存实现用锁遵循相同契约。数据库忙、损坏或路径不可用时报 `StorageError`，不切换到内存。
  数据库格式版本为 1，未知版本拒绝打开。SQLite 文件的目录、备份、权限和保留周期由宿主管理。
- `delete(id, operation_id=..., expected_revision=...)` 删除会话和全部回执；重复删除不存在的会话安全返回。
  删除后开启新的幂等生命周期，宿主应防止旧请求在重新创建会话后重放。

## 历史、失效与选择

`append_history(id, HistoryBlock(...), operation_id=..., expected_revision=...)` 保存调用方分组好的完整单元。
`HistoryBlock` 的消息只接受 `user` / `assistant`，每个块带稳定 `block_id`、`run_id`、`snapshot_id`、
`constraints_revision`、`kind`、可选商品/证据 `references`；追加顺序定义新旧。
工具动作与观察必须放入同一块。块 ID 在会话中不可重复。

宿主负责将事实、费用未知、冲突和分页标记保留在原消息中。模块不会摘要、修改消息或截断 JSON。
显式修改快照通过 `update_memory(..., snapshot_id=新快照)` 完成；即使切回旧快照，也不会恢复旧派生内容。
追加过期快照或过期需求版本的块报 `stale_history`。

`build` 包含当前系统提示、全部生效需求、当前消息、工具及动作 schema。
普通历史必须匹配当前任务、快照及需求版本；最新有效块必须保留。
上一轮最近的方案只通过显式 `plan_items` 保留 offer ID、数量和 evidence ID，作为历史参考；
不恢复旧金额、预算验证结论或待回答问题。这些引用也是可选块，可因预算被舍弃；
入选时在 `selection_reasons` 中标为 `historical_plan_reference`，使用原方案块 ID。
需求改变后，方案必须重新验证。

候选句柄依赖知识服务的进程内状态。新任务不读取旧任务历史；新 `ContextManager` 实例也不会复用
旧实例写入的含 `candidate_set_id` 的块，报告 `needs_search`。对应原始消息和搜索 filters 仍保存在历史中，
宿主可通过 `get` 读取并重新调用 `search_products`。即使 Manager 没有重建，知识服务也可能淘汰句柄；
宿主须处理 `unknown_candidate_set` 并重新检索，模块不保证句柄存活。

`filtered` 先比较显式引用命中数，再比较查询及当前约束与历史的词项重合数；中文字符二元组、英文小写分词。
同分优先较新块，再按块 ID。`unfiltered` 仅按新到旧选择。
两者均逐块尝试，放不下则跳过并继续，最终按时间顺序输出；预算和失效规则完全相同。
`ContextBundle` 返回选中 ID、舍弃原因、版本、计数方法及选择模式，诊断不附带舍弃正文。

## Token 计数与最终消息渲染

默认 `render_messages(payload)` 与当前 `inference.model_port.to_chat_messages` 的工具/schema 前缀一致，
工具定义须包含 `name`、`description`、`parameters`，按 Agent 字段声明顺序渲染。
只在首条 system 消息附加这些可信定义。用户需求和工具结果不会进入 system 消息。
`bundle.payload` 仍是**未渲染**的请求，避免 Agent 推理适配器重复添加前缀；
`bundle.rendered_messages` 用于检查和计数。

默认计数是 `16 + sum(UTF8字节数(content) + 16)`，返回 `mode="estimated"`。
这是保守估算，不能保证实际模型窗口；不同 tokenizer 和 chat template 可能产生不同 token 数。
始终预留 `max_output_tokens + safety_tokens`（安全余量默认 256）；必需内容放不下时报 `ContextLimitError`。

生产接入应注入对应模型及版本的精确计数器；计数器须包含 chat template 和 generation prompt：

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

自定义 `renderer(payload) -> list[dict]` 必须与最终模型端口使用的消息格式一致。
`token_counter(messages) -> TokenCount` 必须确定性返回非负计数、精度和方法标识。
模块对每次装入候选后的最终消息重新计数。若计数器/渲染器报错，错误直接传播，不静默改用估算。

## 可选 Agent 转换与宿主职责

`context.agent_adapter` 提供 `request_from_agent`、`history_block_from_agent` 和 `to_model_request`。
只有最后一个函数在调用时导入 Agent/Pydantic。转换工具对实际 Agent 的动作/观察包装检查配对、工具名及快照；
成功观察必须包含身份和 data；错误观察接受 Agent 的 `status=error`、`code`、`message` 形式。
不猜测整份 `AgentState.history` 的分组边界，也不触发 Agent 执行。

接入顺序：宿主从自然语言提取结构化 patch → 持久化需求 → 构造当前模型请求 → 执行工具 →
保存完整动作/观察块 → 构造下一次请求。每次读取当前需求版本用于新块；同会话的流程调度由宿主串行化。
异步宿主可通过线程执行 SQLite 操作，避免阻塞事件循环。
宿主仍负责鉴权、个人数据处理、任务取消/超时、模型调用、重新检索和确定性预算验证。
当前 `AgentRuntime.resume` 继续负责单任务澄清恢复，本模块不替代检查点。

## 验证

```bash
.venv/bin/python -m context.examples.offline
.venv/bin/python -m pytest tests/test_context.py -q
.venv/bin/python -m pytest tests agent/tests -q
```

离线示例在内存和临时 SQLite 中演示需求保留、历史失效、重新观察及两种策略的构造结果。
示例数据为合成数据，不代表已通过预算验证的商品。
筛选报告支持后续固定模型/快照的消融实验；尚未测得 proposal 的 30% token、20% 延迟或成功率目标。

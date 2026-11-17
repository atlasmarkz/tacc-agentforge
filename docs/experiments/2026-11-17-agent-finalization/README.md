# 补充实验：Cloud 分支收尾对照（非冻结基线）

2026-11-17（Asia/Shanghai）。对应 [issue #21](https://github.com/atlasmarkz/tacc-agentforge/issues/21)。

**审阅修订后的主报告见 [REVIEW.md](REVIEW.md)**：以冻结测量和 issue 原始测量作为修复前基线，并补充追问判断标准、评分与页面接线。本文件保留 `41750ea` 与前一轮修复代码 `183e631` 的对照，不能用于衡量整个 #21 修复相对冻结系统的收益。

## 结论

保留 `cloud-fast` 的 **8 次累计决策上限**，前后各重放 3 次，两条请求结果均一致：

| 请求 | Cloud `41750ea` 模型调用 | Cloud `41750ea` 结束状态 | 前一轮修复模型调用 | 前一轮修复结束状态 |
|---|---|---|---|---|
| issue 原始充电器请求，60 USD | 1 / 1 / 1 | `waiting_user` × 3 | 1 / 1 / 1 | `waiting_user` × 3 |
| dev-07：10 USD 以内键盘，不含税 | 3 / 3 / 3 | `incomplete / evidence_insufficient` × 3 | 3 / 3 / 3 | `incomplete / no_feasible_plan` × 3 |

**这次前测没有复现历史上的 8 次 `model_call_limit`。** 对照的 Cloud 分支已经包含上一轮预算提示、查询缓存和预留收尾的基础实现，不能把 issue 中历史上的 8 次当成本次前测，也不能声称本次重放降低了调用次数。

本次补齐的是重复事实识别和明确的目录无解出口。`completed` 仍专指方案通过校验；没有可验证方案时返回 `incomplete / no_feasible_plan` 并携带面向用户的说明，区别于调用耗尽或执行失败。

## 对照条件与原始证据

- Cloud 对照：提交 `41750ea`，已经包含 #23/#24 的改动，未修改其 Agent 实现。
- 前一轮修复：提交 `183e631` 对应的 Agent 代码；逐文件 SHA256 记录在 [after.json](after.json)。本轮审阅修改的原始数据另存为 `review-after.json`。
- 使用相同重放脚本、Python 依赖、模型进程、请求 ID 和冻结快照；前后解析出的整个 `RunPolicy` 相同。
- `cloud-fast`：关闭思考、温度 0、约束 JSON 解码、512 输出 token、36000 字符上下文、8 次模型决策、8 次工具调用；解释关闭。
- 使用真实 `KnowledgeService`、知识库工具和 `BudgetPlanValidator`；每次重放使用独立内存检查点，不自动编造用户答案。
- 模型：课程 Qwen3.8-27B AWQ，TACC 作业 `29394`，节点 `10-0-3-11`。原始运行信息见 [service-info.json](service-info.json)。
- **运行差异**：默认图编译的临时作业 `29390` 卡在冷编译阶段，已取消。成功的前后对照均使用同一个带 `--enforce-eager` 的模型进程，参数见 [service-eager.json](service-eager.json)。这不是线上 Cloud 拓扑或延迟验收，线上服务没有被部署或重启。
- 快照：`snapshot_5bcd00723fb0b7589cbe50c8`（issue 同源冻结种子），不是线上扩充后的快照。没有运行保留集。

[before.json](before.json) 和 [after.json](after.json) 包含每次完整请求、模型输入/原始输出、用量、最终结果和有界 runtime trace；记录的模型调用次数同时与真实 `generate()` 次数核对一致。

### 充电器

原话：`I need a USB-C charger for my laptop and phone.`，约束 `purchase_cap=60.00, currency=USD`。

前后均在第 1 次决策询问 `laptop_model`、`required_wattage`、`simultaneous_charging`，工具调用为 0。没有凭总功率假设设备适配，也没有建立一个回答后无法继续的等待状态。

### 无可行方案的开发用例

`dev-07` 原话：`Is there any keyboard under 10 USD before tax?`。

独立枚举器 `evaluations.catalog.ground_truth()` 直接读取冻结 SQLite：**0 个可行报价，51 个超预算，12 个无法验证**。结论是这个目录没有可验证方案，不是现实中不存在这种商品。

前测先以错误品类 `keyboard` 查询，得到可恢复参数错误，再用 `keyboards` 查询；未知价格候选触发证据不足收尾。

后测同样有一次品类纠正，第 2 次工具请求使用完整类别、USD、1000 分上限及 `include_unknown=false`。结果为 `scope_complete=true, total_count=0, next_cursor=null`，第 3 次决策返回：

> No keyboard under 10 USD before tax was found in the catalog snapshot (complete search, unknowns excluded).

两边每次都执行 2 次工具请求（含 1 次可恢复参数错误）；没有交付伪造方案。这两条实测没有出现相同成功查询的缓存命中，重复查询的变化由下面的专门回归验证。

## 修复与验证

- 查询进展按候选、报价和证据片段分别记录，换游标、句柄、排名、结果顺序或重复读取已有片段不会制造进展；新文本片段和不同条件的空结果仍可算新发现。
- 相同成功的只读查询复用原结果，标记 `no_new_information` 和 `previous_operation_id`；不消耗新的工具额度，模型决策仍计数。
- 参数纠正后的首次新成功结果重新开始无进展序列，保留一次针对性核查机会；参数错误仍占用累计决策额度。
- YAML 显式保留最后 2 次收尾决策。schema 和运行时都禁止收尾阶段继续执行工具；追问必须为回答后保留可用预算。
- 允许在所有阶段报告无解或证据不足，避免为等待收尾阶段而继续查。提示区分完整目录核查、部分分页、未知事实及现实世界的不存在结论。
- 回归验证第 8 次可以修正无效方案并完成；即使模型忽略 schema，第 8 次也不能继续执行工具。既有真实种子、脚本模型的 Belkin 提案、预算校验和跨恢复行为仍通过。

最终全量检查：

```text
agent/.venv/bin/python -m pytest agent/tests tests -q -o faulthandler_timeout=40
648 passed, 5 skipped, 16 subtests passed
```

测试中的本地 HTTP fixture 需要允许绑定 loopback。新增提示曾挤占默认 16000 字符上下文，已压缩文字，并由既有大观察上下文测试验证，未提高上下文预算。

## 重放方法

入口为 [scripts/replay_agent_finalization.py](../../../scripts/replay_agent_finalization.py)。在两个独立源码目录分别运行同一脚本，先安装同一个冻结种子快照，并为模型建立可用连接：

```bash
agent/.venv/bin/python -m scripts.replay_agent_finalization \
  --snapshots-root /tmp/agent-finalization-data/data/knowledge/snapshots \
  --base-url http://127.0.0.1:18082/v1 \
  --label after --repeats 3 --output after.json
```

也可以传 `--address-file` 代替 `--base-url`。只覆盖模型传输地址，不覆盖 `cloud-fast` 决策策略。脚本在开发用例被枚举为可行时拒绝继续，避免把别的快照误报为无解测试。

临时作业 `29390`、`29394` 和本地重放隧道均已释放。

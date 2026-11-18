# 修复 Agent 收尾：冻结基线对照、决策标准与评测接线

分支：`fix/agent-main-work`；目标：`main`。保留整个 run（包括 resume）的 **8 次累计模型调用上限**。本 PR 移植已有 Agent 修复并适配主分支重放配置，不合入 Cloud 部署祖先提交。

## 1. 回复“修复前应与冻结基线比较”

采用 #23/#24 之前的冻结测量与 issue #21 原始测量作为主基线。`41750ea` 已包含部分修复，它与修复后的 1→1、3→3 对照只能作为补充实验，不能代表原始故障的修复前状态。

| 请求 | 冻结或 issue 原始测量 | 当前主分支版本，tacc-fast，3 次真实重放 |
|---|---|---|
| 原始充电器，60 USD | 12/12 耗尽 8 次上限，`model_call_limit` | **1 / 1 / 1 次，`waiting_user` × 3** |
| 无解 dev-07：10 USD 以内键盘 | **8 / 8 / 8 次，`model_call_limit` × 3** | **3 / 3 / 3 次，`incomplete / no_feasible_plan` × 3**；独立评分 3/3 |
| 应给方案 dev-04：至少 30W、50 USD | **5 / 5 / 5 次，`completed` × 3** | **5 / 5 / 5 次，`completed` × 3**；独立评分 3/3 |

这里的修复后数据来自**当前 main-based 代码的新增真实重放**，不是此前 Cloud 实验的复用。原始记录见 [main-real-replay.json](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/docs/experiments/2026-11-17-agent-finalization/main-real-replay.json)。

基线来源为 `docs/experiments/baseline.md`、冻结开发集/保留集 records 和 issue 原始 soak。所选原始记录及文件哈希见 [frozen-baseline-evidence.json](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/docs/experiments/2026-11-17-agent-finalization/frozen-baseline-evidence.json)。

冻结 B3v 保留集的 **48/96 次 `model_call_limit`** 是历史背景，本 PR 未重跑整个保留集，不据少量开发用例宣称新的整体 TSR 或调用上限率。这是冻结系统与累计修复后的比较，包含先前收敛改动，不是本轮每个文件的独立因果消融。

原始充电器 soak 中 12 条均为 `model_call_limit`；该 JSON 没保存 usage，8 次来自 issue 原始记录和当时上限，不能说成从 soak JSON 直接读出了次数。

## 2. 回复“第 1 次就追问是否符合预期”

判断标准已经写入 [agent/prompts.py:27](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/prompts.py#L27)：

- **先追问**：缺少的用户信息会改变适配判断，例如只说给笔记本和手机用，却没有所需功率、是否同时充电。预算明确不等于兼容性要求明确，不能默认任意笔记本都需要 65W，也不能用总功率推断多口分配。
- **直接提交**：功率要求或设备兼容性已明确，已有报价及规格证据足够支撑选择，就提交给校验器，不为了可选型号、品牌等继续追问。`dev-04` 已明确至少 30W、50 USD，应直接给方案。
- **目录证据未知**：读必要证据或说明缺口；不能把目录缺证据一律转为追问用户。

原始充电器不是冻结评分用例；按上述产品规则，追问是一个有依据的结束状态，但不代表已经交付方案。如果某个评测把它定义为应给方案，追问仍须判 `unnecessary_clarification`。既有 [evaluations/score.py:84](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/evaluations/score.py#L84) 保持该规则，没有放宽评分来获得通过。

## 3. 回复“no_feasible_plan 要接到评测和页面”

- [agent/contracts.py:98](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/contracts.py#L98)：在 `ReportIncomplete.reason` 中增加 `no_feasible_plan`。它仍返回 `status=incomplete`，`completed` 继续表示方案通过配置的校验器。
- [agent/nodes.py:129](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/nodes.py#L129)：加入面向用户的说明，并由 `USER_FACING_ISSUES = set(STOP_MESSAGES)` 统一导出原因集合。
- [evaluations/run.py:274](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/evaluations/run.py#L274)：复用从 Agent 导入的集合，结果进入 `outcome.kind=abstain`，不落入 `error`；没有维护第二套重复白名单。
- [app/static/index.html:358](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/app/static/index.html#L358)：STOP 表补充 `No verifiable option in this catalog meets your requirements and budget.`，页面保留 `message` 优先、STOP 后备的顺序。
- [tests/test_evaluation.py:604](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/tests/test_evaluation.py#L604)：走真实 Runtime → harness_call → scorer，确认无解期望下通过，方案期望下仍为 `abstain_without_plan`。

该原因表示**当前目录内没有可验证方案**，不是现实世界不存在商品，也不是预算校验器做了完整的不可满足性证明。完整目录核查、预算范围和 unknown 限定必须写清；部分分页或只限定一个商品的空结果不能支持全目录结论。

## 4. 代码修改位置与内容

| 文件与位置 | 修改内容 | 主要评审点 |
|---|---|---|
| [agent/progress.py:12](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/progress.py#L12)、[result_facts:44](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/progress.py#L44) | 从整页响应指纹改为候选、报价、证据片段的事实指纹，忽略游标、句柄、排名和候选顺序变化 | 同一事实不能伪装成进展；新文本片段及不同条件的空结果仍需识别 |
| [agent/progress.py:57](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/progress.py#L57) | 新的可恢复参数诊断保留修正机会；相同诊断重复仍累计无进展；指纹记录有界为 256 项 | 纠正品类后还能纠正属性；不能利用换诊断绕过 8 次总上限 |
| [agent/nodes.py:328](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/nodes.py#L328) | 成功只读查询的缓存反馈增加 `no_new_information`、`previous_operation_id` | 缓存命中不再消耗工具额度，但已经发生的模型决策仍计数；失败不缓存，缓存不跨 resume |
| [agent/progress.py:76](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/progress.py#L76)、[configs/agent.yaml:9](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/configs/agent.yaml#L9) | 沿用现有预算和动作硬限制，显式配置预留 2 次收尾、连续 2 次无进展收尾，并允许提前报告结论 | 最后一次不能再执行工具；正常阶段通常留两次给提案/修复；回答后可留一次补证据，但全 run 仍最多 8 次 |
| [agent/prompts.py:27](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/prompts.py#L27)、[49](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/prompts.py#L49) | 明确追问、提交、有范围限定的无解/证据不足三种出口 | 不默认笔记本功率、不放松硬要求、不强迫编造 offer/evidence ID |
| [scripts/replay_agent_finalization.py:44](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/scripts/replay_agent_finalization.py#L44)、[119](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/scripts/replay_agent_finalization.py#L119) | 增加 `--profile`，main 默认 `tacc-fast`，报告记录实际配置、调用和源码哈希 | main 不依赖不存在的 Cloud profile；同一冻结快照、每例至少三次，开发结果交独立 scorer |

此前三次正向重放暴露了“品类错误 → 属性错误”被当成两次无进展、拿不到候选就过早关工具的问题。修正针对的是**新的纠错信息**，不把错误当作商品事实；失败尝试仍占硬预算，初次失败轨迹也保留在仓库中。

## 5. 对其他包与接口的影响

- **Agent 调用方**：结果形状不变，但 reason 枚举增加 `no_feasible_plan`。按停止原因枚举处理的调用方需要识别新值；不能把它误当成已交付方案。
- **evaluations**：复用共享原因集合并补链路测试；`score.py`、冻结清单和用例期望不变。没有把追问或弃答一律算成功。
- **app**：只补 STOP 英文后备说明；backend、harness service 与访问控制没有改动。
- **knowledge**：查询接口、结构化映射、采集与快照没有改动。`unknown` 未被改写成满足要求，字段映射缺口本身也没有在此 PR 中修复。
- **inference / deploy**：模型客户端、模型服务参数、部署脚本没有被带入 main；没有新增运行依赖。Eager 只是临时验证作业的执行方式，不是上线配置变更。
- **状态与预算**：没有更改检查点 schema version 或重置跨恢复计数；新增进展键按缺省值读取。成功方案仍走原预算校验器，预算通过不等于产品兼容性已被校验器证明。

## 6. 测试与证据

主分支完整回归：`python -m pytest agent/tests tests -q -o faulthandler_timeout=40`，**622 passed、2 skipped、14 subtests passed**。

重点覆盖：

- [agent/tests/test_convergence.py:59](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/tests/test_convergence.py#L59)：重复只读查询只执行一次，并触发收尾。
- [同文件:128](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/tests/test_convergence.py#L128)：不同纠错诊断可以继续修正，相同诊断重复会收敛。
- [同文件:188](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/tests/test_convergence.py#L188)、[200](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/tests/test_convergence.py#L200)：第 8 次不执行工具，且可以修正无效方案并完成。
- [同文件:220](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/tests/test_convergence.py#L220)：澄清恢复后仍在累计 8 次内补证据并提交。
- [同文件:293](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/tests/test_convergence.py#L293)：真实知识库中的品类、属性纠错链路不会被过早关闭。
- [agent/tests/test_replay.py:12](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/agent/tests/test_replay.py#L12)：脚本模型＋真实冻结知识库，验证 main 默认 profile 的入口和独立评分；不将这类测试当成真实模型测量。
- 页面 JavaScript 经 Node 解析检查，英文后备说明已核对。

本次主分支实测：2026-11-18（Asia/Hong_Kong），运行源版本 `dde58f3`，Qwen3.8-27B AWQ，临时 TACC 作业 `29542`、节点 `10-0-4-11`。使用 `tacc-fast`、温度 0、关闭思考、约束 JSON、512 输出 token、8 次决策上限、同一冻结快照；解释关闭。每个请求独立运行三次，不自动编造用户答案。

本次重放采用 eager 模型执行，冻结服务曾使用图编译；只比较调用次数与结束状态，不比较延迟。源码、知识库实现与 catalog 哈希见 [main-real-source-manifest.json](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/docs/experiments/2026-11-17-agent-finalization/main-real-source-manifest.json)，服务配置见 [main-real-service-info.json](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/docs/experiments/2026-11-17-agent-finalization/main-real-service-info.json)。已逐条核对 `usage.model_calls == generate 次数 <= 8`，测试前后运行时与知识库源码哈希一致。临时模型作业和隧道在结果归档后释放。

## 7. 评审重点与验证边界

1. 基线是否确实来自冻结数据，是否误用了 #23/#24 之后的 Cloud 对照；所有重复都应展示，不只保留成功轮次。
2. 结束是否是证据支撑的方案、必要追问或限定范围的说明，而不是把 8 次耗尽换成更早、无根据的弃答。
3. 新诊断可修正与重复结果应收敛之间的边界是否正确，缓存与 resume 是否仍遵守原调用上限。
4. `no_feasible_plan` 是否贯通模型 schema、运行结果、runner、scorer 和页面，且不改变方案类用例的失败标准。
5. 少量开发用例不能代表 96 次保留集整体表现；临时 eager 重放也不是线上 Cloud 拓扑、扩充目录或延迟的验收。

Related: #21, #27。PR #29 的目标为 main。

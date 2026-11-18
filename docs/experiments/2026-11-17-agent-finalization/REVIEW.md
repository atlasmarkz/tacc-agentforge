# Issue #21：冻结基线对照与审阅修订

日期：2026-11-17（Asia/Shanghai）。代码分支：`fix/cloud-agent-work`。

移植说明：本文及 JSON 记录的是 Cloud 分支的历史实测。2026-11-18 移植到 `main` 的代码和本地验证见 [MAIN_PORT.md](MAIN_PORT.md)，随后补齐的主分支真实模型重放见 [MAIN_REPLAY.md](MAIN_REPLAY.md)。重放历史 Cloud 配置需检出原 Cloud 修复版本；主分支默认使用 `tacc-fast`，不能将历史 Cloud 实测当成主分支的新实测。

## 1. 正确的修复前基线

主对照使用 **#23/#24 之前的冻结测量与 issue #21 原始测量**。原先 `41750ea` 与 `183e631` 的对照保留在 [README.md](README.md)，它已经包含第一轮收敛修复，只能说明增量行为，不能代表 #21 原始故障的修复前状态。

| 请求 / 范围 | 冻结或 issue 原始测量 | 本轮代码，cloud-fast，每例 3 次 |
|---|---|---|
| issue 原始充电器，60 USD | 12/12 停在 `model_call_limit`，8 次上限 | **1 / 1 / 1 次，`waiting_user` × 3** |
| dev-07：10 USD 以内键盘 | **8 / 8 / 8 次，`incomplete / model_call_limit` × 3** | **3 / 3 / 3 次，`incomplete / no_feasible_plan` × 3**；独立评分 3/3 通过 |
| dev-04：至少 30W 充电器，50 USD | **5 / 5 / 5 次，`completed` × 3** | **5 / 5 / 5 次，`completed` × 3**；独立评分 3/3 通过 |
| B3v 保留集整体 | 96 次中 48 次 `model_call_limit` | 本轮未重跑保留集，不报告新的整体 TSR 或调用上限率 |

这是“冻结系统 → 累计修复后代码”的历史对照，包括 #23/#24 与本轮修订的作用，**不是本轮几个文件改动的独立因果消融**。当前 `dev-04` 三次都提交方案，没有通过修改评分规则豁免不必要追问。

基线证据：

- 冻结开发集和保留集实际执行提交：`b98cd7f4c5fb5c626bd261fd831cc45114d4dfaa`，执行清单摘要 `c3f8ee0a`，报告的 `matches=true`。
- 冻结原始记录：[开发集 records.jsonl](../../../evaluations/results/frozen-dev/records.jsonl)、[保留集 records.jsonl](../../../evaluations/results/frozen-heldout/records.jsonl)。原始分析见 [baseline.md](../baseline.md)。
- 充电器：作业 `28943` 的 [soak 记录](../../../app/results/handover-soak-28943.json)，取 issue 所述最初 37 次中的充电器请求，按 `app.cluster_check.soak()` 的顺序为 `i % 3 == 2`，共 12 条，全部 `model_call_limit`。该 soak JSON 没有保存 usage，8 次来自 issue 原始记录及当时上限，不能说成从 JSON 直接读出了次数。
- [frozen-baseline-evidence.json](frozen-baseline-evidence.json) 保存所选冻结记录、48/96 汇总、12 条充电器结果及源文件 SHA256。没有覆盖、重新冻结或重写原始数据。

## 2. 什么时候追问，什么时候提交方案

**先追问**：缺少的用户信息会改变商品是否适合，而且当前证据无法消除这个分歧。例如原始请求只说给笔记本和手机用，没有功率及是否同时充电；“任意 65W 总功率充电器能满足这台笔记本”没有依据。应在还有回答后执行预算时追问必要信息。

本轮原始充电器三次均只询问 `laptop_wattage` 与 `simultaneous_use`，没有要求用户额外提供品牌或型号：

> What wattage does your laptop charger require, and do you need to charge the laptop and phone simultaneously?

**直接提交**：预算、功率等实际硬要求已经明确，或设备需求已能从可信证据确认，且已有候选/报价/规格证据能支持选择。此时不为了可选细节或型号继续追问。例如 `dev-04` 已要求至少 30W、预算 50 USD，没有额外的特定笔记本适配要求；应从证据中选商品提交校验，而不是强制问笔记本型号。

**目录证据未知**：目录缺少价格或规格，不等于用户遗漏需求。应读必要的商品证据、核查可验证匹配，或明确报告证据缺口。不能把“目录里没有证据”一律转换成追问用户。

**评测标准保持不变**：若评测用例期望 `plan`，即使追问在某种产品策略下说得通，`waiting_user` 仍会被记为 `unnecessary_clarification`，本轮不豁免它。issue 原始充电器不是冻结用例，当前按上述明确的产品标准判断追问合理；它不计入开发集方案成功率，也不借此宣称原始请求已交付方案。若将来把它定义为应直接给方案的评测用例，当前追问必须按失败记录。

规则已写入 `agent/prompts.py`；正向用例 `dev-04` 用独立枚举器确认存在可行报价，再按原 scorer 验证实际方案与引用，三次均通过。

## 3. no_feasible_plan 的评测和页面接线

`agent.nodes.STOP_MESSAGES` 已包含 `no_feasible_plan`，`USER_FACING_ISSUES = set(STOP_MESSAGES)`。`evaluations/run.py` 导入同一个集合，所以该原因会进入 `outcome.kind = abstain`，不会落入 `error`。本轮保留这一个共享来源，没有在两个模块维护重复的白名单。

新增测试走完整 `AgentRuntime → evaluations.harness_call → score_record`：

- `no_feasible_plan`、`evidence_insufficient`、`requirements_unverified` 的模型说明均保留为 `abstain`。
- 在期望无解的用例上评分通过。
- 在期望方案的用例上仍失败为 `abstain_without_plan`，不会把所有弃答都算成功。

页面 `STOP` 表补充：

> No verifiable option in this catalog meets your requirements and budget.

页面继续优先展示模型自己的 `message`，缺失时使用上述英文后备说明；没有只依赖通用 `Stopped (no feasible plan)` 文案。

## 4. 三次重复发现并修复的运行时缺口

本轮初次增加 `dev-04` 三次重放时，分别为：5 次 `completed`、5 次 `plan_repair_limit`、4 次 `evidence_insufficient`，独立评分仅 1/3 通过。完整记录保存在 [review-initial.json](review-initial.json)，没有只保留成功的首轮。

失败路径先把 `charger` 改成 `chargers`，随后工具提示将 `wattage` 改为 `power`。这两条**不同的新纠错信息**被当成连续两次无进展，提前关闭工具，模型还没拿到候选就被迫收尾，其中一轮重复提交了示例占位 ID，被校验器拒绝。

修正后，新出现的可恢复诊断会保留一次修正机会，但不被当作商品事实；相同诊断重复仍累计无进展。错误尝试始终占用 8 次硬上限，未增加预算。新增回归用真实知识库验证“品类错误 → 属性错误 → 正确查询 → 提交成功”，并验证重复错误仍会收敛。

修正后的整套九次重放见 [review-after.json](review-after.json)：保留每次模型输入/输出、真实调用次数、工具轨迹、结束状态、开发集 ground truth 与独立评分。三条请求每条都重跑三次，`dev-04` 最终 3/3 通过。

## 5. 设置、验证与复现

- 本轮临时 TACC 作业 `29431`，Qwen3.8-27B AWQ，`cloud-fast`：温度 0、关闭思考、约束 JSON、512 输出 token、36000 字符上下文、最多 8 次决策和 8 次工具调用，解释关闭。
- 固定种子快照 `snapshot_5bcd00723fb0b7589cbe50c8`，真实知识库工具和预算校验器，每次独立内存检查点，不编造用户答案。
- 本轮仍使用 eager 模型执行，配置保存在 [review-service-info.json](review-service-info.json)。历史冻结服务启用图编译；本次不是同一运行环境的配对延迟实验，不比较耗时。也未验证线上扩充快照与 Cloud 服务拓扑。
- 首次临时作业 `29428` 因节点 GPU 被 Slurm 外的进程占用而在启动前退出，随后排除该节点重试。该次没有执行模型重放，不计入 Agent 失败率。临时 GPU 作业与本地隧道均已释放。
- 最终代码文件 SHA256 与 `review-after.json` 一致，9 次计数逐条核对 `usage.model_calls == generate 次数 <= 8`。

```text
agent/.venv/bin/python -m pytest agent/tests tests -q -o faulthandler_timeout=40
653 passed, 5 skipped, 16 subtests passed
```

页面脚本经 Node 解析检查，英文后备说明已核对。重放命令：

```bash
agent/.venv/bin/python -m scripts.replay_agent_finalization \
  --snapshots-root /tmp/agent-finalization-data/data/knowledge/snapshots \
  --base-url http://127.0.0.1:18083/v1 \
  --label review-after --repeats 3 --dev-cases dev-07 dev-04 \
  --output docs/experiments/2026-11-17-agent-finalization/review-after.json
```

先建立可用的模型连接；也可用 `--address-file`。主对照使用已封存历史测量，没有重新运行或调参保留集。`evaluations/freeze.json` 和原评分规则保持原样。

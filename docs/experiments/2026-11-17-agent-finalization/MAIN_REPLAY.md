# 当前 main-based 代码的真实模型重放

2026-11-18（Asia/Hong_Kong），分支 `fix/agent-main-work`，运行源版本 `dde58f3`。本轮只增加评审说明和测试证据，没有修改运行时源码。

## 结果

| 请求 | 历史冻结或 issue 原始测量 | 当前主分支代码，三次真实重放 |
|---|---|---|
| 原始充电器，60 USD | 12/12 `model_call_limit`，8 次上限 | **1 / 1 / 1 次，`waiting_user` × 3** |
| dev-07，无可验证方案 | **8 / 8 / 8 次，`model_call_limit` × 3** | **3 / 3 / 3 次，`no_feasible_plan` × 3**；独立评分 3/3 |
| dev-04，应给方案 | **5 / 5 / 5 次，`completed` × 3** | **5 / 5 / 5 次，`completed` × 3**；独立评分 3/3 |

原始充电器仍只追问笔记本所需功率与是否同时充电，不能把等待回答写成已交付方案。若评测将其定义为应给方案，追问依然应判 `unnecessary_clarification`。dev-04 已明确 30W 和 50 USD，本次没有追问，三次交付方案均由独立 scorer 验证通过。

主分支检索实现不接受 `>=` 别名。三次 dev-04 的实际轨迹均能在工具返回诊断后改用 `gte`，在第 5 次决策提交通过校验的方案；这验证了当前主分支自己的知识库实现，而不是将 Cloud 记录换名复用。

主对照的冻结记录、原始充电器 soak 与源文件哈希见 [frozen-baseline-evidence.json](frozen-baseline-evidence.json)。96 次保留集中的 48 次耗尽仍仅是历史背景，本轮没有重跑保留集，没有报告新的整体 TSR。

## 环境与证据

- 原始九次记录：[main-real-replay.json](main-real-replay.json)，含模型输入/输出、调用次数、工具轨迹、ground truth 和评分。
- 源码与 catalog 哈希：[main-real-source-manifest.json](main-real-source-manifest.json)。运行前后核对一致，源提交为 `dde58f3`；后续文档/数据提交不改变这些运行时源码。
- TACC 临时作业 `29542`，节点 `10-0-4-11`，Qwen3.8-27B AWQ，vLLM eager。实际服务配置见 [main-real-service-info.json](main-real-service-info.json)。
- profile 为 **tacc-fast**，温度 0、关闭思考、约束 JSON、512 输出 token、36000 字符上下文、8 次决策和 8 次工具调用上限，解释关闭。
- 同一冻结种子 `snapshot_5bcd00723fb0b7589cbe50c8`，真实知识库和预算校验器，每个 run 独立检查点，不自动填写用户答案。
- 全部 9 次已核对 `usage.model_calls == 记录的 generate 次数 <= 8`。开发用例均由原独立枚举器/scorer 判定，没有修改评分标准或冻结清单。
- eager 与冻结时期的图编译执行方式不同，本轮只比较调用次数与结束状态，不比较延迟；也不代表线上 Cloud 拓扑、扩充目录或服务持久化的验收。

## 2026-11-21：提示词排列调整后重放

`inference/model_port.py` 的 `to_chat_messages` 调整了消息排列：智能体每次调用都会变化的 `Runtime policy (trusted JSON)`（`model_calls_remaining`、阶段、允许的动作）原本拼在 system 消息末尾，现在改为最后单独一条 user 消息；system 消息里加一句说明它的位置。其余消息原样传递。`agent/` 未改动。

**为什么改**：vLLM 的前缀缓存只复用与之前请求完全相同的开头部分。剩余调用数放在 system 消息里，每次调用都会变，后面的工具列表、动作 Schema、用户请求和全部历史因此每次都要重新预填充。线上一次请求（`r-054ab9dd-1795254744-c77861`）约 12 次模型调用、约 8.4 万输入 token，前缀缓存命中率降到 11%，回答阶段耗时 78 秒，几乎全部是预填充（约 1000 token/秒）。调整后，同一运行内相邻两次调用的前缀在上一条真实消息之前完全相同。

**重放结果**（TACC 服务作业 `30320`，节点 `10-0-3-11`，CUDA graphs，profile `tacc-fast`，其余设置与上文相同；原始记录 [main-prefix-layout-replay.json](main-prefix-layout-replay.json)）：

| 请求 | 修复合入时（`dde58f3`） | 提示词排列调整后，三次真实重放 |
|---|---|---|
| 原始充电器，60 USD | 1 / 1 / 1 次，`waiting_user` × 3 | **1 / 1 / 1 次，`waiting_user` × 3** |
| dev-07，无可验证方案 | 3 / 3 / 3 次，`no_feasible_plan` × 3 | **4 / 4 / 4 次，`no_feasible_plan` × 3**；独立评分 3/3 |
| dev-04，应给方案 | 5 / 5 / 5 次，`completed` × 3 | **5 / 5 / 5 次，`completed` × 3**；独立评分 3/3 |

结束状态和评分不变；模型看到的输入排列变了，dev-07 的决策路径多用了 1 次调用，仍远低于 8 次上限。

**速度**（同一服务，旧排列与新排列交替运行，同样的 3 个请求，每种排列 2 轮）：

| | 旧排列 | 新排列 |
|---|---|---|
| 每次决策调用平均耗时 | 4.52 / 4.03 秒 | **3.04 / 2.85 秒** |
| dev-04（5 次调用）端到端 | 30.5 / 28.4 秒 | **15.6 / 15.8 秒** |
| 3 个请求的模型总耗时 | 40.7 / 36.2 秒（9 次调用） | 30.4 / 28.5 秒（10 次调用） |

速度测量是一次性的对比，不属于 C1–C7 的可复现结果；调用越多、历史越长，收益越大。

**哈希说明**：本次在 Windows 工作区运行，`source_sha256` 的路径键已统一为正斜杠，哈希按 LF 换行内容计算（即仓库中提交的内容）；未改动文件的哈希已与 `HEAD` 逐一核对一致。

本轮命令与下文相同，只改了 `--label main-prefix-layout-replay` 和 `--output .../main-prefix-layout-replay.json`。

## 执行命令

先建立到可用模型的连接，再运行：

```bash
agent/.venv/bin/python -m scripts.replay_agent_finalization \
  --profile tacc-fast \
  --snapshots-root /tmp/agent-finalization-data/data/knowledge/snapshots \
  --base-url http://127.0.0.1:18084/v1 \
  --label main-real-replay --repeats 3 --dev-cases dev-07 dev-04 \
  --output docs/experiments/2026-11-17-agent-finalization/main-real-replay.json
```

独立的完整本地回归结果为 **622 passed、2 skipped、14 subtests passed**；离线编排测试与本次真实模型测量分开记录。完整评审回复和代码定位见 [PR_MAIN_DESCRIPTION.md](PR_MAIN_DESCRIPTION.md)。

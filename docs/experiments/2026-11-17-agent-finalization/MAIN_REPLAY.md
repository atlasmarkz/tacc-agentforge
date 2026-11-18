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

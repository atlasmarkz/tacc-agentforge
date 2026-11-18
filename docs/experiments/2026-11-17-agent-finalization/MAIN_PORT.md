# 移植到 main：fix/agent-main-work

日期：2026-11-18（Asia/Hong_Kong）。

## 基点与范围

本分支直接基于 `origin/main` 的 `14a358019aac5240ee1c7c0baf158555aa75ad83`，移植来源为 `fix/cloud-agent-work` 的两个 Agent 修复提交：

| 来源提交 | main 分支上的移植提交 |
|---|---|
| `183e631` | `2d3bf7c` |
| `1b519a6` | `c7db008` |

两个提交均可干净 cherry-pick；随后增加主分支兼容的重放入口与回归测试。Git 分支按用户指定命名为 `fix/agent-main-work`。

相对 main 的代码范围为 Agent 收敛与结果契约、配置中的收尾额度、页面英文后备说明、评测接线测试和重放脚本，附带原有实验记录。已检查下列路径相对 main 没有变化：

```text
README.md
knowledge/
deploy/
app/backend.py
app/harness_service.py
```

因此，这个 main PR 不包含 Cloudflare/GCP 部署、Cloud 服务管理、知识库扩充或 Cloud 分支 README 改写。原来的 Cloud 修复分支及 PR #27 保留原有用途。

## 主分支适配

`main` 没有 `cloud-fast` profile。重放入口增加 `--profile`，默认选择 main 已有的 `tacc-fast`；输出报告记录实际选中的配置。Cloud checkout 中仍可显式传 `--profile cloud-fast`。

没有向 main 引入 Cloud 部署配置。默认决策额度仍为 8，预留收尾 2 次，关闭思考、输出上限 512 token、上下文预算 36000 字符。

## 本轮验证

在这个 main-based 工作树中运行：

```text
python -m pytest agent/tests tests -q -o faulthandler_timeout=40
622 passed, 2 skipped, 14 subtests passed
```

新增 `agent/tests/test_replay.py` 使用**脚本模型和真实冻结知识库/预算校验器**，从重放 CLI 的默认参数运行到独立评分：

- 默认配置为 `tacc-fast`，不依赖 Cloud profile。
- 充电器脚本模型在 1 次决策后等待必要用户信息。
- 无解 dev-07 经真实目录查询，在 2 次脚本模型决策后返回 `no_feasible_plan`，独立评分通过。

这些调用次数是离线编排回归，不是真实模型测量。页面 JavaScript 也通过 Node 语法检查，并核对了 `no_feasible_plan` 英文后备说明。

本轮没有启动 TACC 作业或新增真实模型重放。`REVIEW.md`、`review-after.json` 等既有原始记录仍明确属于此前 Cloud 分支的实测，不能被当作此 main 版本的新实测或新的保留集结果。

## 后续合并操作

使用这个分支新建 **base 为 `main`、head 为 `fix/agent-main-work`** 的 PR，审查并合入该 PR。旧 PR #27 的 head 仍带有 Cloud 部署提交，不适合直接将它的 base 改为 main。

如需在主分支配置下追加真实模型验证，先准备相同冻结快照和模型连接，再运行：

```bash
python -m scripts.replay_agent_finalization \
  --profile tacc-fast \
  --snapshots-root /path/to/snapshots \
  --address-file /path/to/service-url.txt \
  --label main-real-replay --repeats 3 --dev-cases dev-07 dev-04 \
  --output /path/to/main-real-replay.json
```

评分规则和冻结清单保持原样，方案类用例中的追问仍判为 `unnecessary_clarification`。

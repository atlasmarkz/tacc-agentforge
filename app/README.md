# 浏览器、后端与集群通道(Task 5)

用户在浏览器里提问(页面为英文,电商风格:顶部搜索栏、商品卡片、订单汇总),请求经本机后端和 SSH 隧道到达 TACC 计算节点上的 harness 服务,模型调用留在集群内。

```
浏览器 ──HTTP──▶ 本机后端 app/backend.py (127.0.0.1:8765)
                    │  ssh -L,经网关 gw.turing.ust.hk
                    ▼
            harness 服务 app/harness_service.py (CPU 作业,计算节点 IP:8100)
                    │  地址文件 service-url.txt
                    ▼
            vLLM 服务 (GPU 作业,deploy/tacc/service/qwen-service)
```

## 为什么这样设计

- 集群没有对外开放的 HTTP 入口,但网关能按 IP 连到计算节点的端口,本机 `ssh -L` 经网关可以直达(已实测)。所以浏览器到后端、后端到 harness 都能流式,不需要文件队列。
- 浏览器只和本机后端通信。服务令牌、SSH 密钥、集群地址都留在后端;查询时不用 SerpApi,没有 API 密钥参与。
- harness 服务启动后把自己的地址和一次性令牌写到 `USERDIR/tacc-agentforge-service/harness-url.txt`、`harness-token.txt`(权限 600)。后端经 SSH 读取这两个文件再建隧道;服务换作业后,下一个请求会重新读取。
- harness 本身没有改动:服务直接使用 `AgentRuntime`、真实知识库工具、预算校验器(#18)和流式解释的证据解析器;运行策略来自 `configs/agent.yaml` 的 `tacc-fast`(关闭思考)。

## 请求 ID

- 浏览器生成一个随机会话 ID(存在 `sessionStorage`)。
- 运行 ID 由后端生成(一次提问或一次修改各是一个运行),对话 ID 是 `s-<会话 ID>`。浏览器不能自己指定。
- 后端只允许会话回答、停止、查看或修改**自己的**运行,其他会话一律 403。
- 每个运行记一行到 `app/logs/requests.jsonl`:会话、运行、类型(提问/回答/修改)、上一个运行、快照、处理它的 harness 作业、最终状态、首个事件和总用时。

## 运行

```bash
# 1. 集群:先启动 vLLM 服务,再启动 harness 服务(两个作业正好用满每人 2 个名额)
bash deploy/tacc/service/qwen-service/stage.sh serve
bash deploy/tacc/tools/submit-win.sh deploy/tacc/service/qwen-service
bash deploy/tacc/service/app-service/stage.sh
bash deploy/tacc/tools/submit-win.sh deploy/tacc/service/app-service

# 2. 本机:启动后端,浏览器打开 http://127.0.0.1:8765
python -m app.backend
```

后端需要 harness 的依赖之外的东西只有标准库和 `ssh`。环境变量 `TACC_SSH`、`TACC_SSH_PORT`、`TACC_SSH_KEY`、`TACC_REMOTE_USERDIR` 可覆盖默认的账户、端口、密钥和 USERDIR。

不连集群开发页面时,用脚本模型的演示服务:

```bash
python -m app.harness_service --demo --port 8101 --token demo
python -m app.backend --harness-url http://127.0.0.1:8101 --token demo
```

## 接口

后端(浏览器调用,均为 POST JSON,`session_id` 必填;流式接口返回 SSE):

| 接口 | 作用 |
|---|---|
| `GET /api/health` | 后端和 harness 服务状态 |
| `/api/ask` | `{message, constraints, explain, revises?}` 新提问或修改,流式返回事件 |
| `/api/answer` | `{run_id, question_id, answer}` 回答追问,继续原运行 |
| `/api/cancel` | `{run_id}` 停止运行 |
| `/api/lookup` | `{run_id, offer_ids, evidence_ids}` 查报价详情和来源链接 |

SSE 事件:`meta`(运行 ID)、`agent`(harness 的 `AgentEvent`,最后一个带 `result`)、`error`(`{code, message}`)、`end`(含处理它的作业号)。harness 服务每 10 秒发一次 `: ping`,浏览器断开时运行会被取消。

## 已知限制

- **暂停的运行不能跨作业恢复。** 检查点在内存里:USERDIR 是 NFS(`findmnt` 显示 `nfs4`),PR #13 的 SQLite 检查点会拒绝它。作业结束(最长 1 小时)时还在等回答的运行会丢失,页面提示"找不到这次运行,请重新提交"。
- **还没有自动交接。** harness 作业在 54 分钟时停止接收,需要手动再提交;vLLM 服务的交接方案(RUNBOOK)用到第二个作业名额,而 harness 正好占着它。把 harness 并进 vLLM 作业可以解决,留作下一步。
- 每人最多 2 个作业:vLLM 和 harness 各占一个,运行期间不能再跑评测或训练作业。
- 只有一个用户的后端:后端绑定 127.0.0.1,没有登录,适合演示和个人使用,不是公开服务。
- 方案目前只有一个(`VerifiedPlan`),没有多个方案的排序。

## 平台实测(2026-11-13)

vLLM 作业 28872(节点 10-0-3-10),harness 服务作业 28873(节点 10-0-1-10),本机后端经网关 SSH 隧道连接。
脚本 `python -m app.cluster_check` 按浏览器的方式调用后端,完整结果在 `results/cluster-check-28872-28873.json`。

| 场景 | 结果 | 用时 |
|---|---|---|
| 带预算的请求(耳机,100 USD) | completed,22.99 USD,剩余 77.01;3 条来源都有链接 | 18.3 s,首个事件 0.4 s |
| 不带预算(中文双肩包) | 先追问,回答后继续;模型用了自造字段 `max_price_usd`,校验器不认,于是再问一次 `purchase_cap`/`currency`,第二次回答后 completed | 4.1 s + 13.1 s + 第二轮 |
| 不带预算(英文双肩包) | 追问 `purchase_cap`/`currency`,一次回答后 completed | |
| 修改(同一会话,预算改为 30) | 新运行、指向上一个运行,completed,剩余 7.01 | 15.3 s |
| 停止 | 在第一次模型调用开始时停止,cancelled | 0.7 s |
| 会话隔离 | 另一个会话回答或查看他人的运行:403 | |
| 三个会话同时请求 | 3 个不同运行 ID,全部 completed | 34 至 41 s(单个约 15 至 18 s) |
| 浏览器(本机后端 + 真实模型) | 页面完整显示:商品卡片、价格、运费与税、订单汇总、预算条、说明、来源 | 约 20 s |
| 服务未就绪(vLLM 还没起来) | failed / model_unavailable,页面给出提示,不卡住 | 立即 |

### 平台上发现的问题

1. **节点被作业以外的进程占用。** 10-0-1-10 在 Slurm 里显示空闲,但上面有一个他人的 vLLM(同一课程模型、32K 上下文),占满两块 GPU(23.5/24 GiB)和 8000 端口。我们的 vLLM 绑定端口失败,而就绪探测连到了那个进程,把它当成我们的服务发布出去(作业 28868)。已修:`serve.py` 端口被占时改用空闲端口,并在启动前确认端口没人应答;harness 服务的 8100 端口同样处理。
2. **USERDIR 是 NFS(`nfs4`)。** PR #13 的 SQLite 检查点在 NFS 上会拒绝运行,所以集群上不能用;本服务的检查点放在内存里。
3. **推荐理由全部回退到模板(实测 5/5)。** 模型写的理由内容正确,但会引用商品标题里的数字(如 "30 hours"、"2 mics"、"Bluetooth 5.3"),harness 的检查只接受方案里的数字,判为 `unlisted_number`。回退模板是中文,所以页面在回退时改用英文摘要(由已校验的方案和报价详情生成)。检查规则在 `agent/explanation.py`,需要 harness 负责人决定是否放宽到证据标题里的数字。
4. **追问字段不规范。** 中文请求时模型会自造字段名(`max_price_usd`),多问一轮。需要在 harness 提示里列出允许的约束字段。
5. **部分请求用完 8 次模型调用。** 一次充电器请求反复搜索和查报价,没提出方案就停在 `model_call_limit`;同一请求另一次成功。页面会给出英文提示,建议把需求说具体。
6. **首次安装依赖约 23 分钟。** #13 改了依赖锁文件,pip 往 NFS 写得很慢;之后的作业复用安装结果(28873 启动约 1 分钟)。

## 测试

- `tests/test_app.py`:本地完整流程(脚本模型、种子快照、真实 HTTP 与预算校验器),以及后端的运行 ID 与会话隔离。
- `python -m app.cluster_check --out <report.json>`:在平台上按浏览器的方式跑六个场景,结果见上面的平台实测。

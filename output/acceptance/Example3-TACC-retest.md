# Example 3：TACC 真模型部署补验

> 更新：[2026-11-23 修复后的单作业复验](Example3-TACC-30851.md)通过。下文保留修复前的故障和替代部署记录。

**时间：** 2026-11-22 22:51–23:30（UTC+8）

**版本：** 本地缓存的 `origin/main` 提交 `96e40bee49563209b44b0f4082429dbdcf5fcdf1`，独立干净检出。GitHub 直连检出实际报 443 连接超时；未声称该 SHA 是当日远端最新版本。以下步骤编号来自该提交的[根 README](../../README.md) 第 180–266 行。本次只验证 Example 3，不复跑保留集或其他课程实验。

**结论：部分通过。** 原 README 一键启动未直接通过；但在同一个本人双 GPU 分配内用离线依赖单独启动 harness 后，**本机浏览器 → 本机后端 → SSH 隧道 → TACC harness → vLLM** 全链路通过。未运行保留集复评。

| 检查项 | 结果 | 实际证据 |
|---|---|---|
| 3.1–3.2：本人 SSH 和 SDK | **通过** | 网关 SSH 返回 `ok` 和本人的 `USERDIR`；SDK 0.4.1 用隔离配置，`tcloud init` 和 `setup-sdk-account.sh` 确认课程环境可用。[接入日志](evidence/example3-20261122/01-access.log) |
| 3.3：原样提交单作业 | **部分通过** | 执行 `bash deploy/tacc/service/qwen-service/stage.sh serve --harness` 和 `tcloud submit`，本人作业 **30837** 获得双 GPU。vLLM 加载 5/5 分片，动态改用可用端口，启动后 **286.6 秒**达到 ready；同作业的原 harness 随后因 `ImportError: cannot import name 'RunTree' from 'langsmith'` 退出，本机健康接口仍返回 `harness: null`。[提交](evidence/example3-20261122/03-submit.log) · [失败与就绪日志](evidence/example3-20261122/06-status.log) |
| 打包及依赖排查 | **失败**（原制品） | 缓存 SHA-256 标记匹配，但依赖目录缺少 `langsmith/__init__.py`；现有启动前检查仍报告依赖已安装，实际加载 harness 才失败。干净 `stage.sh` 打包也遗漏源码已有的 `app/formspec.py` 和 `app/narrative.py`，这两个模块分别供问卷和推荐理由使用。[缓存检查](evidence/example3-20261122/06b-preflight.log) · [打包导入检查](evidence/example3-20261122/06d-stage-imports.log) |
| 3.4–3.5：后端及补验 SSH | **通过（调整后）** | 本机 `.venv/bin/python -m app.backend` 仍用 README 的 SSH 通道。独立 CPU 作业 30846 预计开始得太晚，未运行即取消。改在 30837 内用 `srun --jobid=30837 --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=none --immediate=20` 启动 CPU step；旧 Slurm 不支持 `--overlap`，去掉后通过。离线安装锁定的 45 个 wheel 到节点本地 scratch，在临时包补齐上述两个模块，并让 `TACC_REMOTE_USERDIR` 指向本次独立发布目录。浏览器显示 `qwen3.8-27b`、`在线 · 作业 30837`；健康接口同时返回该作业、模型和冻结快照。[CPU step](evidence/example3-20261122/06i-own-allocation-step.log) · [后端隧道日志](evidence/example3-20261122/05b-backend-fallback.log) · [健康 JSON](evidence/example3-20261122/07-health.json) |
| 3.6：真模型浏览器及 API | **通过（调整后）** | README 耳机请求和补答后均给出 Walmart 的 Onn Mini Sticks；商品 **$9.88**、运费 **$6.99**、税前 **$16.87**、$100 预算剩余 **$83.13**。浏览器显示商品卡、费用明细、来源及实时生成的推荐理由；空预算出现追问，补答 100 USD 后同一运行 `r-42e248e9-1795361331-df878d` 从 `waiting_user` 转为 `completed`。独立 API 核对两次各 3 次模型决策、报价及来源与冻结 SQLite 一致；问卷和推荐理由接口也有响应。[独立核对](evidence/example3-20261122/13-verified.json) · [浏览器请求日志](evidence/example3-20261122/08-requests.jsonl) · [问卷](evidence/example3-20261122/10b-form.txt) · [推荐理由 SSE](evidence/example3-20261122/12b-narrative.txt) |
| 3.7：资源清理 | **通过** | 只停止本人验收作业和本机后端。复查 Slurm 已没有 30837 或 30846。[清理日志](evidence/example3-20261122/14-cleanup.log) |

原始 API 输出：[有预算请求](evidence/example3-20261122/09-budgeted.sse) · [缺预算追问](evidence/example3-20261122/10-waiting.sse) · [同次运行补答](evidence/example3-20261122/11-resumed.sse) · [有预算来源](evidence/example3-20261122/12-lookup-budgeted.json) · [补答后来源](evidence/example3-20261122/12-lookup-resumed.json)。商品金额是快照税前价，不表示当前库存或实时售价。本轮补救仅在临时副本执行。公开证据已隐藏账号、本机路径、临时地址和会话标识。

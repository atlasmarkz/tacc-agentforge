# Example 3：启动脚本修复后复验

**结论：通过（修复版，使用随包的锁定离线依赖）。** 本次不需要独立 harness 作业或手动追加 Slurm step 来启动服务。仍按 README 执行 `stage.sh serve --harness`、`tcloud submit`，并启动本机 `app.backend`；打包前额外设置 `TACC_HARNESS_WHEELHOUSE`，以避免把计算节点访问 PyPI 作为前提。在线安装分支未在该次 GPU 作业中验证。

**版本与时间：** 从缓存的 `origin/main` 提交 `96e40bee49563209b44b0f4082429dbdcf5fcdf1` 独立检出，应用工作区的启动及打包修改后提交。2026-11-22 23:42（UTC+8）提交，2026-11-23 00:26:37 开始运行。修复相关的 28 个本地测试通过。

| 检查 | 作业 30851 的实测结果 |
|---|---|
| 单作业启动 | vLLM 记录 `ready_after_s: 322.5`，harness 记录 `harness dependencies installed and validated` 和 `READY harness service … (job 30851)`。新缓存通过真实导入后才发布，本次未重现 `langsmith.RunTree` 报错。冷启动的共享盘安装较慢，整个 harness 约在作业开始后 15 分钟才就绪。 |
| SSH 和真实模型 API | 本机后端经默认发布目录建立 SSH 通道，健康接口报告 `job_id: 30851`、`model: qwen3.8-27b`。API 的原有预算请求用 3 次模型调用完成。商品及运费与冻结 SQLite 和来源记录一致。 |
| 浏览器与预算补答 | 浏览器显示在线作业 30851；未填预算时追问，补答 100 USD 后同一运行 `r-9c113af4-1795365770-7bb5ee` 由 `waiting_user` 转为 `completed`。Walmart 商品 $9.88 + 运费 $6.99 = 税前 **$16.87**，余额 **$83.13**；问卷及推荐理由 API 也正常响应。 |
| 清理 | 00:43:30 完成验收后按 README 停止本次 GPU 作业，本机后端亦已停止。Slurm 的 `CANCELLED` 是主动清理，不是验证失败。 |

证据（已隐藏账号、本机路径、临时地址和会话标识）：[提交](evidence/example3-30851/01-submit.log) · [测试](evidence/example3-30851/02-clean-tests.log) · [harness 日志](evidence/example3-30851/14-harness.log) · [vLLM 事件](evidence/example3-30851/15-service-events.jsonl) · [健康检查](evidence/example3-30851/07-health.json) · [API 独立核对](evidence/example3-30851/13-verified.json) · [浏览器请求日志](evidence/example3-30851/05-requests.jsonl) · [清理](evidence/example3-30851/16-cleanup.log)。

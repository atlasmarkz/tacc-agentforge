# ShoppingBench 恢复评测

状态：`paused_allocation_limit`。原始中断批次另存，不覆盖。

恢复规则仅依据 API 空响应，不依据答案正确性；有完整模型响应的错误答案保持原判。

当前记录数：183；尚无可报告的完整模型分数。

本结果为当前模型＋官方 harness 的外部对照，不代表项目生产 B3、真实价格正确性或浏览器端到端延迟。token usage 未由官方轨迹记录时保持 N/A。

## 文件说明

| 文件 | 大小 | SHA256 | 用途 |
| --- | ---: | --- | --- |
| `rollouts.jsonl.gz` | 5.0 MB | `5faa8fde66d7f750925f62096d1d6765d94f6bdc9819fe4fc9eb3171d0b97bd9` | 归档的恢复轨迹；供审计/继续恢复使用 |
| `supervisor.json` / `status.json` / `submission-status.json` | — | — | 调度状态，不是模型分数 |
| `recovery.json` | — | — | 恢复策略：只重试空 API 响应 |
| `service-28995.json` / `model-info.json` / `preflight.json` | — | — | 服务与上游 provenance |

两个 `rollouts.jsonl.gz` 合计约 7.3 MB，保留它们是为了让评审能检查原始输出；`report`/`publish` 读取解压后的 JSONL，不直接读取 gzip。

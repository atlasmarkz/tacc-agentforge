"""Render a proposal-style Markdown report from a v1 benchmark output directory.

The numbers come from `report.json` (deterministic scoring).  Faithfulness and
preference labels are human judgements and are only filled when a `review.py
score` report is passed with --review; they are never invented here.

Usage:
    python -m evaluations.bench report --input results/my-run
    python -m evaluations.bench report --input results/my-run --review review.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONFIG_ORDER = ['B0', 'B1', 'B2', 'B3', 'A1', 'A2']
CONFIG_LABELS = {
    'B0': ('B0', 'Single call'),
    'B1': ('B1', 'Keyword RAG'),
    'B2': ('B2', 'Structured RAG'),
    'B3': ('B3', 'Full advisor'),
    'A1': ('A1', 'Without loop'),
    'A2': ('A2', 'Without validator'),
}
CONFIG_DESCRIPTIONS = {
    'B0': 'No extra text; no budget tool',
    'B1': 'FTS5 evidence; single call; no budget tool',
    'B2': 'Filtered FTS5; single call; no budget tool',
    'B3': 'B2 + bounded loop + budget tool',
    'A1': 'B3; one-pass plan with the same tools',
    'A2': 'B3; budget tool disabled',
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _rate(value):
    if value is None or value.get('rate') is None:
        return None
    return value['rate']


def _pct(value, digits=1):
    rate = _rate(value)
    return 'N/A' if rate is None else f'{rate * 100:.{digits}f}'


def _fraction(value):
    if not value or not value.get('m'):
        return 'n/m'
    return f"{value['n']}/{value['m']}"


def _fraction_pct(value, digits=1):
    if not value or not value.get('m'):
        return 'N/A'
    return f"{_pct(value, digits)} ({value['n']}/{value['m']})"


def _number(value, digits=1):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def _ci(value):
    if not value or not value.get('ci95'):
        return 'N/A'
    low, high = value['ci95']
    return f'[{low * 100:.1f}, {high * 100:.1f}]'


def _short_hash(value: str | None, length: int = 12) -> str:
    return (value or '')[:length] if value else 'N/A'


def _code_hash(code) -> str:
    if not code:
        return 'N/A'
    payload = json.dumps(code, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:12]


def _review_faithfulness(review: dict | None, config: str):
    """Pool the two reviewers' claim-level faithfulness for one configuration."""
    if not review:
        return None
    metrics = review.get('by_config', {}).get(config, {})
    n = m = 0
    for value in metrics.values():
        claim = value.get('faithfulness_micro') or {}
        if claim.get('m'):
            n += claim.get('n', 0)
            m += claim['m']
    if not m:
        return None
    return {'n': n, 'm': m, 'rate': n / m}


def _review_preference(review: dict | None, config: str):
    if not review:
        return None
    values = [v.get('preference_mean') for v in review.get('by_config', {}).get(config, {}).values()]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _ordered_configs(report: dict) -> list[str]:
    present = set(report.get('summary', {}))
    ordered = [c for c in CONFIG_ORDER if c in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _method_cell(config: str) -> str:
    code, label = CONFIG_LABELS.get(config, (config, config))
    return f'**{code} {label}**' if code == 'B3' else f'{code} {label}'


def _proposal_table(manifest: dict, report: dict, review: dict | None) -> list[str]:
    summary = report.get('summary', {})
    configs = _ordered_configs(report)
    lines = [
        '## 主结果（proposal 表）',
        '',
        '| Method | Configuration | TSR % ↑ | Faithfulness ↑ | BVR % ↓ | p95 s ↓ |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for cfg in configs:
        row = summary.get(cfg, {})
        faith = _review_faithfulness(review, cfg)
        faith_text = f"{faith['rate']:.2f}" if faith else 'N/A'
        lines.append(
            f"| {_method_cell(cfg)} | {CONFIG_DESCRIPTIONS.get(cfg, '')} | "
            f"{_pct(row.get('tsr'))} | {faith_text} | "
            f"{_pct(row.get('bvr'))} | {_number((row.get('wall_s') or {}).get('p95'), 2)} |"
        )
    lines += [
        '',
        '口径：',
        '',
        '- **TSR** = 成功完成次数 / 全部计划尝试次数；超时、崩溃、缺失记录都计入分母。',
        '- **Faithfulness** = 两位评审对可评判事实声明的支持率；没有标注时为 N/A，不能用 0 或 1 代替。',
        '- **BVR** = 已交付且费用可核实的计划中超出真实可用预算的比例；没有已交付计划时为 N/A。',
        '- **p95** 是 adapter 子进程完整往返耗时，不含浏览器、桥接和用户输入等待；不是 proposal 的完整请求 p95。',
    ]
    return lines


def _detail_table(report: dict) -> list[str]:
    summary = report.get('summary', {})
    configs = _ordered_configs(report)
    lines = [
        '## 自动指标与计数',
        '',
        '| Config | runs | plans | TSR n/m (95% CI) | BVR n/m | amount errors | unverifiable plans | '
        'constraint violations | ungrounded IDs | invalid actions | model calls | tokens in/out |',
        '| --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |',
    ]
    for cfg in configs:
        row = summary.get(cfg, {})
        amount = row.get('amount_error_rate')
        tokens = f"{_number(row.get('prompt_tokens_mean'))}/{_number(row.get('completion_tokens_mean'))}"
        lines.append(
            f"| {cfg} | {row.get('runs', 0)} | {row.get('plans_delivered', 0)} | "
            f"{_fraction(row.get('tsr'))} ({_ci(row.get('tsr_by_case'))}) | {_fraction(row.get('bvr'))} | "
            f"{_fraction_pct(amount)} | {row.get('unverifiable_amount_plans', 0)} | "
            f"{_fraction_pct(row.get('constraint_violation'))} | {_fraction_pct(row.get('ungrounded_ids'))} | "
            f"{row.get('invalid_actions', 0)} | {_number(row.get('model_calls_mean'))} | {tokens} |"
        )
    if report.get('missing_attempts'):
        lines += ['', f"> **警告**：roster 中还有 {report['missing_attempts']} 次尝试没有记录，"
                      "上面的分母已按完整 roster 计入；补齐前不要把这些数字当最终结果。"]
    if report.get('malformed_lines_ignored'):
        lines += ['', f"> 记录了 {report['malformed_lines_ignored']} 行无法解析的 records，已忽略。"]
    return lines


def _strata_table(report: dict, field: str, title: str) -> list[str]:
    strata = (report.get('strata') or {}).get(field) or {}
    if not strata:
        return []
    values = sorted(strata)
    configs = []
    for value in values:
        for cfg in (strata[value] or {}):
            if cfg not in configs:
                configs.append(cfg)
    configs = [c for c in CONFIG_ORDER if c in configs] + sorted(set(configs) - set(CONFIG_ORDER))
    lines = [f'## {title}', '', '| Config | ' + ' | '.join(values) + ' |',
             '| --- | ' + ' | '.join('---:' for _ in values) + ' |']
    for cfg in configs:
        cells = []
        for value in values:
            rate = _rate((strata[value].get(cfg) or {}).get('tsr'))
            cells.append('N/A' if rate is None else f'{rate * 100:.1f}')
        lines.append(f'| {cfg} | ' + ' | '.join(cells) + ' |')
    lines += ['', '单元格是 TSR %；完整的分母、区间和 `by_category` 在 `report.json` 的 `strata` 中。']
    return lines


def _paired_table(report: dict) -> list[str]:
    paired = report.get('paired') or []
    if not paired:
        return []
    lines = ['## 配对差值（按 case 的 95% bootstrap 区间）', '',
             '| Compare | Δ TSR (pp) | 95% CI (pp) | cases |', '| --- | ---: | --- | ---: |']
    for item in paired:
        diff, ci, cases = item.get('diff'), item.get('ci95'), item.get('cases')
        if diff is None:
            continue
        ci_text = 'N/A' if not ci else f'[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]'
        lines.append(f"| {item.get('from')} → {item.get('to')} | {diff * 100:+.1f} | {ci_text} | {cases} |")
    lines += ['', '区间包含 0 时，不要声称该配置优于对照。']
    return lines


def _provenance(manifest: dict, report: dict) -> list[str]:
    dataset = manifest.get('dataset') or {}
    adapter = manifest.get('adapter') or {}
    identity = adapter.get('identity') or {}
    service = manifest.get('service_info') or {}
    if not isinstance(service, dict):
        service = {}
    lines = ['## 运行溯源', '']
    lines.append(f"- split: `{manifest.get('split', 'unknown')}`；configs: `{', '.join(manifest.get('configs', []))}`；repeats: {manifest.get('repeats', '?')}")
    lines.append(f"- dataset: status=`{dataset.get('status', '?')}`, dev={dataset.get('dev', '?')}, "
                 f"heldout={dataset.get('heldout', '?')}, pending={dataset.get('pending', '?')}")
    hashes = dataset.get('hashes') or {}
    if hashes:
        lines.append('- dataset hashes: ' + ', '.join(f"`{k}`={_short_hash(v)}" for k, v in sorted(hashes.items())))
    if isinstance(identity, dict):
        parts = [f"{k}=`{v}`" for k, v in identity.items() if k in ('model', 'runtime', 'adapter', 'validator',
                                                                    'temperature', 'thinking', 'max_output_tokens')]
        if parts:
            lines.append('- adapter: ' + ', '.join(parts))
    elif identity:
        lines.append(f'- adapter: `{identity}`')
    model_env = {k: v for k, v in (manifest.get('model_env') or {}).items() if v}
    if model_env:
        lines.append('- model env: ' + ', '.join(f'{k}=`{v}`' for k, v in model_env.items() if v))
    if service:
        bits = [f"{k}=`{service[k]}`" for k in ('job_id', 'node', 'address', 'model') if service.get(k)]
        if bits:
            lines.append('- service: ' + ', '.join(bits))
    resumes = manifest.get('resumes') or []
    if resumes:
        last = resumes[-1] if isinstance(resumes[-1], dict) else {}
        bits = [f"count={len(resumes)}", f"last_remaining={last.get('remaining_at_start', '?')}"]
        lines.append('- resumes: ' + ', '.join(bits))
    git = manifest.get('git') or {}
    if isinstance(git, dict) and git.get('commit'):
        lines.append(f"- git: `{git['commit'][:12]}`; dirty={git.get('dirty')}")
    if manifest.get('only'):
        lines.append(f"- only: `{manifest['only']}`（dev smoke，不是完整集）")
    code = manifest.get('code')
    if code:
        lines.append(f"- code fingerprint: `{_code_hash(code)}` ({len(code)} files)")
    lines.append(f"- timing boundary: `{manifest.get('timing_boundary', 'unknown')}`")
    lines.append('')
    lines.append('> 本报告由 `python -m evaluations.bench report` 从 `report.json` 生成；'
                 '模型、服务参数和原始记录以同目录的 `manifest.json` / `records.jsonl` 为准。')
    return lines


def render(manifest: dict, report: dict, review: dict | None = None) -> str:
    """Return a deterministic Markdown report (no timestamps or invented labels)."""
    configs = _ordered_configs(report)
    lines = [
        f"# Benchmark report: {manifest.get('track', 'project-v1-synthetic-fixtures')}",
        '',
        f"- split: `{manifest.get('split', 'unknown')}`；configs: `{', '.join(manifest.get('configs', configs))}`；"
        f"repeats: {manifest.get('repeats', '?')}",
        f"- attempts recorded: {sum((report.get('summary', {}).get(c, {}) or {}).get('runs', 0) for c in configs)}",
    ]
    status = (manifest.get('dataset') or {}).get('status')
    if status:
        lines.append(f'- dataset status: `{status}`')
    if review:
        lines.append('- human review merged: yes（Faithfulness 来自两位评审的 claim-level 标注）')
    else:
        lines.append('- human review merged: no（Faithfulness 为 N/A，需要 `python -m evaluations.review`）')
    lines += ['']
    lines += _proposal_table(manifest, report, review)
    lines += ['']
    lines += _detail_table(report)
    for field, title in (('task_type', '按任务类型 TSR %'), ('language', '按语言 TSR %'),
                         ('budget_basis', '按预算口径 TSR %')):
        table = _strata_table(report, field, title)
        if table:
            lines += [''] + table
    paired = _paired_table(report)
    if paired:
        lines += [''] + paired
    lines += [''] + _provenance(manifest, report)
    return '\n'.join(lines).rstrip() + '\n'


def load_run(input_dir: Path) -> tuple[dict, dict]:
    report_path = Path(input_dir) / 'report.json'
    manifest_path = Path(input_dir) / 'manifest.json'
    if not report_path.is_file():
        raise ValueError(f'{report_path} not found; run the benchmark or `python -m evaluations.v1.run --rescore DIR` first')
    report = _read_json(report_path)
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    return manifest, report


def write_report(input_dir: Path, out: Path | None = None, review_path: Path | None = None) -> Path:
    input_dir = Path(input_dir)
    manifest, report = load_run(input_dir)
    review = _read_json(Path(review_path)) if review_path else None
    destination = Path(out) if out else input_dir / 'REPORT.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(manifest, report, review), encoding='utf-8')
    return destination

"""Wait for durable recovery, then score on CPU and archive final artifacts.

Does not start GPU jobs or alter task outcomes. Interrupted and recovered ASR are
kept separate. A terminal incomplete recovery yields a status report, not a
fabricated model score. This watcher can outlive an interactive agent turn.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import time

TERMINAL = {'completed', 'completed_with_api_failures', 'incomplete_container_exit',
            'paused_allocation_limit', 'paused_wall_limit', 'stopped_by_request', 'error',
            'publish_blocked_cleanup'}


def archive(root, out, state):
    out.mkdir(parents=True, exist_ok=True)
    raw = root / 'output/voucher'
    for name in ('rollouts.jsonl', 'rollout.log', 'search.log'):
        source = raw / name
        if source.exists():
            with source.open('rb') as src, gzip.open(out / (name + '.gz'), 'wb') as dst:
                shutil.copyfileobj(src, dst)
    for name in ('config.json', 'preflight.json', 'recovery.json', 'pip-freeze.txt'):
        if (raw / name).exists():
            shutil.copy(raw / name, out / name)
    for source in root.glob('service-*.json'):
        shutil.copy(source, out / source.name)
    for name in ('model-info.json', 'supervisor.json', 'container.log', 'supervisor.log'):
        if (root / name).exists():
            shutil.copy(root / name, out / name)
    (out / 'status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_summary(out, state, report=None):
    report = report or {}
    rows = report.get('rows', [])
    successes = sum(r.get('success', False) for r in rows)
    asr = report.get('asr')
    text = '# ShoppingBench 恢复评测\n\n'
    text += f"状态：`{state['status']}`。原始中断批次另存，不覆盖。\n\n"
    text += '恢复规则仅依据 API 空响应，不依据答案正确性；有完整模型响应的错误答案保持原判。\n\n'
    if rows:
        text += f"完整任务清单：{report['tasks']}；成功 {successes}；恢复后 ASR：{asr:.1%}。\n\n"
        text += f"API 不完整任务：{report.get('api_incomplete')}；解释边界：{report.get('quality_interpretation')}。\n\n"
    else:
        text += f"当前记录数：{state.get('completed', 0)}；尚无可报告的完整模型分数。\n\n"
    text += '本结果为当前模型＋官方 harness 的外部对照，不代表项目生产 B3、真实价格正确性或浏览器端到端延迟。token usage 未由官方轨迹记录时保持 N/A。\n'
    (out / 'README.md').write_text(text, encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--project', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--hf-cache', required=True, type=Path)
    p.add_argument('--image', default='agentforge-shoppingbench:204dfba')
    p.add_argument('--model', default='qwen3.8-27b')
    p.add_argument('--max-hours', type=float, default=4.5)
    a = p.parse_args()
    started = time.monotonic()
    state_file = a.root / 'supervisor.json'
    while time.monotonic() - started < a.max_hours * 3600:
        state = json.loads(state_file.read_text())
        if state['status'] in TERMINAL:
            cleanup = [e for e in state.get('events', []) if e.get('kind') == 'cleanup_error']
            if cleanup:
                state = {**state, 'status': 'publish_blocked_cleanup', 'cleanup_error': cleanup}
                archive(a.root, a.out, state)
                (a.out / 'status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                write_summary(a.out, state)
                return
            done = {e.get('job') for e in state.get('events', []) if e.get('kind') == 'allocation_released'}
            if set(state.get('jobs', [])) <= done:
                break
        time.sleep(15)
    else:
        raise SystemExit('publisher wait limit reached; supervisor remains authoritative')
    archive(a.root, a.out, state)
    if state['status'] not in ('completed', 'completed_with_api_failures'):
        write_summary(a.out, state)
        return
    scores = a.root / 'scores'
    scores.mkdir(exist_ok=True)
    scores.chmod(0o777)
    cmd = ['docker', 'run', '--rm', '--memory=5g', '-e', 'JAVA_TOOL_OPTIONS=-Xmx2g',
        '-e', 'PYTHONPATH=/project', '-e', 'HF_HOME=/hf', '-e', 'GIT_CONFIG_COUNT=1',
        '-e', 'GIT_CONFIG_KEY_0=safe.directory', '-e', 'GIT_CONFIG_VALUE_0=/bench',
        '--mount', f'type=bind,source={a.repo.resolve()},target=/bench,readonly',
        '--mount', f'type=bind,source={a.project.resolve()},target=/project,readonly',
        '--mount', f'type=bind,source={(a.root / "output/voucher").resolve()},target=/run,readonly',
        '--mount', f'type=bind,source={scores.resolve()},target=/scores',
        '--mount', f'type=bind,source={a.hf_cache.resolve()},target=/hf', a.image,
        'python', '-u', '-m', 'evaluations.shoppingbench', '--repo', '/bench',
        '--revision', '204dfba767cf849fe33535e42293e8973fb017ea', '--rollouts', '/run/rollouts.jsonl',
        '--out', '/scores/report.json', '--index-dir', '/bench/indexes', '--model', a.model]
    try:
        with (a.out / 'scoring.log').open('w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        shutil.copy(scores / 'report.json', a.out / 'report.json')
        report = json.loads((scores / 'report.json').read_text())
        write_summary(a.out, state, report)
    except Exception as exc:
        state = {**state, 'status': 'scoring_failed', 'scoring_error': str(exc)}
        (a.out / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
        write_summary(a.out, state)
        raise


if __name__ == '__main__':
    main()

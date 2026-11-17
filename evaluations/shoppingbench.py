"""ShoppingBench voucher track audit. Run with the official environment's Python.

Uses upstream eval_voucher unchanged. Counts missing rollouts as failures against
all 250 released tasks and records the exact upstream commit and input hashes.
No model inference, cloning, installs, or paid API calls are triggered here.
"""
import argparse
from collections import defaultdict
from .shoppingbench_recovery import api_complete
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(tasks, rollouts, evaluate):
    queries = [t['query'] for t in tasks]
    if len(queries) != len(set(queries)):
        raise ValueError('duplicate task query')
    by_query, unidentified = {}, []
    for rollout in rollouts:
        try:
            q = rollout[0]['extra_info']['query']
        except (IndexError, KeyError, TypeError, AttributeError):
            unidentified.append(rollout)
            continue
        if q not in queries or q in by_query:
            raise ValueError('unknown or duplicate rollout query')
        by_query[q] = rollout
    rows = []
    for task in tasks:
        q = task['query']
        rollout = by_query.get(q)
        row = {'query': q, 'success': False, 'budget_match': False, 'present': rollout is not None,
               'finished': False, 'tool_calls': 0, 'prompt_tokens': None, 'completion_tokens': None,
               'api_complete': api_complete(rollout) if rollout else False}
        if rollout:
            try:
                result = evaluate(rollout, task)
                if not isinstance(result, dict):
                    raise ValueError('evaluator returned a non-object')
            except Exception as exc:
                row['error'] = type(exc).__name__ + ': ' + str(exc)[:200]
                row['api_complete'] = False
            else:
                row.update(result)
                row['success'] = result.get('rule', 0) >= 1 and result.get('budget', 0) >= 1
                row['budget_match'] = result.get('budget', 0) >= 1
        rows.append(row)
    n = len(rows)
    api_incomplete = sum(not r['api_complete'] for r in rows)
    errors = any(r.get('error') for r in rows)
    flags = []
    if api_incomplete:
        flags.append('interrupted/API-incomplete')
    if errors:
        flags.append('scoring-error')
    return {'track': 'ShoppingBench-official-harness-voucher', 'tasks': n, 'missing': n - len(by_query),
            'unidentified_rollouts': len(unidentified),
            'asr': sum(r['success'] for r in rows) / n if n else None,
            'budget_match_rate': sum(r['budget_match'] for r in rows) / n if n else None,
            'completion_rate': sum(r['finished'] for r in rows) / n if n else None,
            'api_incomplete': api_incomplete,
            'quality_interpretation': ('/'.join(flags) + '; not a clean model comparison'
                                       if flags else 'API-complete'),
            'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--revision', required=True, help='full immutable git commit ID')
    p.add_argument('--rollouts', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--model', required=True)
    p.add_argument('--index-dir', type=Path, help='official Lucene index named indexes')
    a = p.parse_args()
    repo = a.repo.resolve()
    a.rollouts = a.rollouts.resolve()
    a.out = a.out.resolve()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--', 'src/agent', 'src/search_engine', 'data/synthesize_voucher_test.jsonl'], text=True)
    if commit != a.revision or dirty:
        p.error('upstream revision mismatch or tracked source/data modifications')
    tasks_path = repo / 'data/synthesize_voucher_test.jsonl'
    tasks = [json.loads(l) for l in tasks_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    if len(tasks) != 250:
        p.error('expected the complete 250-task voucher test split')
    if a.out.exists():
        p.error('output exists; choose a new path')
    rollouts, malformed_lines = [], 0
    for line in a.rollouts.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rollouts.append(json.loads(line))
        except json.JSONDecodeError:
            malformed_lines += 1
    sys.path.insert(0, str(repo / 'src/agent'))
    # Upstream creates LuceneSearcher("indexes") relative to CWD.
    import os
    index = a.index_dir.resolve() if a.index_dir else repo / 'indexes'
    if not index.is_dir() or index.name != 'indexes':
        p.error('provide --index-dir pointing to the official indexes directory')
    os.chdir(index.parent)
    upstream = importlib.import_module('run_evaluate')

    def evaluate(rollout, task):
        result = defaultdict(float)
        upstream.eval_voucher(result, rollout, task['reward'], task['voucher'])
        messages = [step.get('completion', {}).get('message') or {} for step in rollout]
        calls = [call for m in messages for call in (m.get('tool_call') or [])]
        return {**dict(result), 'finished': any(c.get('name') == 'terminate' for c in calls),
                'tool_calls': len(calls)}

    report = audit(tasks, rollouts, evaluate)
    report['malformed_rollout_lines'] = malformed_lines
    manifest = {'commit': commit, 'model': a.model, 'tasks_sha256': sha(tasks_path),
                'rollouts_sha256': sha(a.rollouts.resolve()),
                'python': sys.version, 'pip_freeze': subprocess.check_output(
                    [sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    index_build = repo / 'index-build.json'
    if index_build.is_file():
        manifest['index_build'] = json.loads(index_build.read_text(encoding='utf-8'))
    preflight = a.rollouts.parent / 'preflight.json'
    if preflight.is_file():
        manifest['preflight'] = json.loads(preflight.read_text(encoding='utf-8'))
    report['manifest'] = manifest
    a.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    a.out.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

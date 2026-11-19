"""Schedule/score a complete roster using an explicit system adapter.

Adapter: module:function(payload) -> dict, invoked in an isolated subprocess per
turn. It must return an outcome and measurements; it never receives oracle data.
No test-only oracle adapter is installed as a model baseline. B3 requires a
validator-capable adapter declaration. Adapters may use the existing ModelClient.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .aggregate import aggregate, paired_difference
from .oracle import score
from .suite import BASE, digest, fingerprint, git_state

CONFIGS = ['B0', 'B1', 'B2', 'B3', 'A1', 'A2']


def schedule(cases, configs, repeats):
    if repeats < 1 or not configs or len(set(configs)) != len(configs) or set(configs) - set(CONFIGS):
        raise ValueError('positive repeats and unique known configurations required')
    # Cyclic order balances first-position cache effects across cases/repeats.
    return [(c, cfg, rep) for rep in range(repeats) for n, c in enumerate(cases)
            for cfg in configs[(n + rep) % len(configs):] + configs[:(n + rep) % len(configs)]]


def invoke(adapter, payload, timeout):
    start = time.monotonic()
    try:
        p = subprocess.run([sys.executable, '-m', 'evaluations.v1.run', '--worker', adapter],
                           input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
        if p.returncode:
            return {'outcome': {'kind': 'error'}, 'error': 'adapter_error', 'wall_s': time.monotonic() - start}
        result = json.loads(p.stdout)
        if not isinstance(result, dict) or not isinstance(result.get('outcome'), dict):
            raise ValueError('adapter must return object with an outcome object')
        result['wall_s'] = time.monotonic() - start
        return result
    except subprocess.TimeoutExpired:
        return {'outcome': {'kind': 'error'}, 'stop_reason': 'timeout', 'wall_s': time.monotonic() - start}
    except (ValueError, OSError):
        return {'outcome': {'kind': 'error'}, 'stop_reason': 'invalid_action', 'wall_s': time.monotonic() - start}


def run_episode(adapter, case, catalog, cfg, timeout):
    history, responses = [], []
    constraints = case['initial_constraints'].copy()
    message = case['message']
    script = list(case['script'])
    started = time.monotonic()
    while True:
        payload = {'schema_version': 1, 'config': cfg, 'case_id': case['case_id'],
                   'message': message, 'constraints': constraints, 'history': history,
                   'catalog': catalog, 'budget_basis': case['budget_basis'],
                   'output_contract': {'kind': 'plan|ask_user|no_feasible_plan|insufficient_evidence',
                       'plan_fields': ['items (offer_id,quantity,evidence_ids)', 'lines (offer_id,quantity,subtotal)',
                                       'shipping', 'total', 'remaining'],
                       'abstention_fields': ['catalog_id', 'reason_codes']}}
        # User text carries requirements; evaluator-only structured requirements are not passed.
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            responses.append({'outcome': {'kind': 'error'}, 'stop_reason': 'timeout'})
            break
        rec = invoke(adapter, payload, remaining)
        if rec.get('stop_reason') == 'invalid_action':
            rec['invalid_actions'] = 1
        responses.append(rec)
        out = rec.get('outcome', {})
        # An error outcome ends the episode as an adapter error even without an error or stop_reason field.
        # Before, the scripted-turn check below failed on it and recorded an invalid action, which counted the
        # adapter's own failure in the invalid-action metric (PR #28 review).
        if isinstance(out, dict) and out.get('kind') == 'error' and not rec.get('stop_reason'):
            rec['stop_reason'] = 'adapter_error'
        if rec.get('error') or rec.get('stop_reason') or not script:
            break
        turn = script[0]
        if 'trigger_fields' in turn:
            if out.get('kind') != 'ask_user' or set(out.get('required_fields', [])) != set(turn['trigger_fields']):
                rec['stop_reason'] = 'invalid_action'
                rec['invalid_actions'] = 1
                break
        else:
            interim = {**case, 'constraints': constraints}
            check = score({**rec, 'config': cfg, 'rep': 0}, interim, catalog)
            if not check['success']:
                rec['stop_reason'] = 'invalid_action'
                rec['invalid_actions'] = 1
                break
        script.pop(0)
        history += [{'role': 'user', 'content': message}, {'role': 'assistant', 'content': json.dumps(out)}]
        constraints.update(turn['answer'])
        message = turn.get('revision', json.dumps(turn['answer'], ensure_ascii=False))
    final = responses[-1]
    return {**final, 'config': cfg, 'case_id': case['case_id'], 'wall_s': time.monotonic() - started,
            'turns': responses, 'prompt_tokens': sum(r.get('prompt_tokens') or 0 for r in responses) if all(r.get('prompt_tokens') is not None for r in responses) else None,
            'completion_tokens': sum(r.get('completion_tokens') or 0 for r in responses) if all(r.get('completion_tokens') is not None for r in responses) else None,
            'model_calls': sum(r.get('model_calls') or 0 for r in responses)}


def pip_freeze():
    try:
        return subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
    except (OSError, subprocess.SubprocessError):
        return None


def service_info():
    path = os.environ.get('BENCHMARK_SERVICE_INFO')
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def strip_traces(record):
    """Remove prompts, replies and intermediate agent state before persisting."""
    out = dict(record)
    for key in ('turns', 'model_trace', 'events', 'parsed_requirements', 'unchecked_proposal'):
        out.pop(key, None)
    return out


def summarize(records, cases, catalogs):
    by_id = {c['case_id']: c for c in cases}
    scores = [score(r, by_id[r['case_id']], catalogs[by_id[r['case_id']]['catalog_id']]) for r in records]
    # Reuse case-clustered inference; no pooled token repeats as independent cases.
    def summarize_subset(subset):
        result = aggregate(subset, records)
        for cfg, report in result.items():
            plans = [s for s in subset if s['config'] == cfg and s['plan_delivered']]
            checked = [s for s in plans if s['amount_error'] is not None]
            report['amount_error_rate'] = {'n': sum(s['amount_error'] is True for s in checked), 'm': len(checked),
                'rate': sum(s['amount_error'] is True for s in checked) / len(checked) if checked else None}
            report['unverifiable_amount_plans'] = len(plans) - len(checked)
            known = [s for s in plans if s['plan_total'] is not None]
            n = sum(s['over_budget'] is True for s in known)
            report['bvr'] = {'n': n, 'm': len(known), 'rate': n / len(known) if known else None}
            report['faithfulness'] = None
        return result
    summary = summarize_subset(scores)
    strata = {}
    for field in ('language', 'budget_basis', 'task_type'):
        strata[field] = {v: summarize_subset([s for s in scores if by_id[s['case_id']][field] == v])
                        for v in sorted({c[field] for c in cases})}
    paired = [paired_difference(scores, a, b) for a, b in [('B0', 'B1'), ('B1', 'B2'), ('A1', 'B3'), ('A2', 'B3')]]
    return scores, {'summary': summary, 'strata': strata, 'paired': [p for p in paired if p]}


def _resume_errors(a, metadata, manifest, configs, roster):
    """Refuse to append to a run made under different data, code or adapter."""
    errors = []
    if metadata.get('dataset', {}).get('hashes') != manifest.get('hashes'):
        errors.append('dataset hashes changed')
    for key, expected in (('split', a.split), ('configs', configs), ('repeats', a.repeats),
                          ('only', a.only), ('timeout', a.timeout), ('record_traces', a.record_traces)):
        if metadata.get(key) != expected:
            errors.append(f'{key} differs from the original run')
    if metadata.get('adapter_sha256') != digest(a.adapter_manifest):
        errors.append('adapter manifest changed')
    if metadata.get('code') != fingerprint():
        errors.append('code changed after the original run (start a new run)')
    expected_roster = [[c['case_id'], cfg, rep] for c, cfg, rep in roster]
    if metadata.get('roster') != expected_roster:
        errors.append('roster differs from the original run')
    return errors


def _load_records(path, roster):
    """Read records.jsonl, tolerating only a truncated final line from a kill."""
    expected = set(roster)
    if not path.exists():
        return {}
    by_key, lines = {}, path.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                continue
            raise ValueError(f'malformed records line {index + 1}') from exc
        if not isinstance(rec, dict):
            if index == len(lines) - 1:
                continue
            raise ValueError(f'records line {index + 1} is not an object')
        key = (rec.get('case_id'), rec.get('config'), rec.get('rep'))
        if key not in expected:
            raise ValueError(f'unexpected attempt {key}')
        if key in by_key:
            raise ValueError(f'duplicate attempt {key}')
        by_key[key] = rec
    return by_key


def _write_records(path, records):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker')
    p.add_argument('--rescore', type=Path, help='rebuild report including interrupted/missing roster entries')
    p.add_argument('--resume', action='store_true', help='continue a run from its existing manifest and records')
    p.add_argument('--adapter', help='module:function; callable receives only model-visible payload')
    p.add_argument('--adapter-manifest', type=Path, help='JSON with identity, validator_configs and cache_policy')
    p.add_argument('--data', type=Path, default=BASE)
    p.add_argument('--split', choices=['dev', 'heldout'], default='dev')
    p.add_argument('--only', default='', help='comma-separated IDs for dev smoke runs only')
    p.add_argument('--unseal-heldout', action='store_true')
    p.add_argument('--configs', default=','.join(CONFIGS))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--out', type=Path)
    p.add_argument('--record-traces', action='store_true',
                   help='dev-only: persist raw prompts/replies (heldout traces are forbidden)')
    a = p.parse_args()
    if a.worker:
        module, name = a.worker.split(':')
        result = getattr(importlib.import_module(module), name)(json.load(sys.stdin))
        print(json.dumps(result))
        return
    if a.rescore:
        metadata = json.loads((a.rescore / 'manifest.json').read_text(encoding='utf-8'))
        catalogs = json.loads((a.data / 'catalogs.json').read_text(encoding='utf-8'))
        if any(digest(a.data / name) != sha for name, sha in metadata['dataset']['hashes'].items()):
            p.error('rescore dataset hash mismatch')
        cases = [json.loads(l) for l in (a.data / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        rows, malformed = [], 0
        for line in (a.rescore / 'records.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1  # a process killed mid-write must not destroy the rescue
                continue
            if not isinstance(value, dict) or any(k not in value for k in ('case_id', 'config', 'rep')):
                malformed += 1
                continue
            rows.append(value)
        roster = [tuple(k) for k in metadata['roster']]
        if all(k in metadata for k in ('split', 'configs', 'repeats')):
            expected_cases = [c for c in cases if c['split'] == metadata['split']]
            only = metadata.get('only', '')
            if only:
                selected = set(only.split(','))
                expected_cases = [c for c in expected_cases if c['case_id'] in selected]
            expected = {(c['case_id'], cfg, rep)
                        for c, cfg, rep in schedule(expected_cases, metadata['configs'], metadata['repeats'])}
            if set(roster) != expected:
                p.error('roster is not the complete scheduled roster')
        by_key = {(r['case_id'], r['config'], r['rep']): r for r in rows}
        if len(by_key) != len(rows) or set(by_key) - set(roster):
            p.error('duplicate or unexpected attempts')
        records = [by_key.get(k, {'case_id': k[0], 'config': k[1], 'rep': k[2],
                   'outcome': {'kind': 'error'}, 'error': 'missing_or_interrupted'}) for k in roster]
        used = {k[0] for k in roster}
        scores, report = summarize(records, [c for c in cases if c['case_id'] in used], catalogs)
        report['missing_attempts'] = len(roster) - len(by_key)
        report['malformed_lines_ignored'] = malformed
        (a.rescore / 'scores.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in scores), encoding='utf-8')
        (a.rescore / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return
    if not a.adapter or not a.adapter_manifest or not a.out:
        p.error('--adapter, --adapter-manifest and --out required')
    manifest = json.loads((a.data / 'manifest.json').read_text(encoding='utf-8'))
    if any(digest(a.data / name) != sha for name, sha in manifest['hashes'].items()):
        p.error('snapshot hash mismatch')
    if a.record_traces and a.split != 'dev':
        p.error('raw prompts/replies are dev-only; heldout trace export is forbidden')
    if a.split == 'heldout':
        if not a.unseal_heldout or not (a.data / 'release.json').exists():
            p.error('heldout sealed: freeze, release fifth category, then --unseal-heldout')
        release = json.loads((a.data / 'release.json').read_text())
        freeze = json.loads((a.data / 'freeze.json').read_text())
        if release['manifest_sha256'] != digest(a.data / 'manifest.json') or freeze['code'] != fingerprint():
            p.error('release or code changed after freeze')
    caps = json.loads(a.adapter_manifest.read_text(encoding='utf-8'))
    configs = a.configs.split(',')
    if not caps.get('identity') or not caps.get('cache_policy'):
        p.error('adapter manifest must identify model/runtime and cache policy')
    if (set(configs) & {'B3', 'A1'}) - set(caps.get('validator_configs', [])):
        p.error('B3/A1 require a real validator declaration; recording stubs are A2 diagnostics')
    cases = [json.loads(l) for l in (a.data / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = [c for c in cases if c['split'] == a.split]
    if a.only:
        if a.split != 'dev':
            p.error('--only is dev-only; formal heldout must use the complete roster')
        selected = set(a.only.split(','))
        if selected - {c['case_id'] for c in cases}:
            p.error('unknown development case ID')
        cases = [c for c in cases if c['case_id'] in selected]
    catalogs = json.loads((a.data / 'catalogs.json').read_text(encoding='utf-8'))
    roster = schedule(cases, configs, a.repeats)
    roster_keys = [(c['case_id'], cfg, rep) for c, cfg, rep in roster]
    current_env = {k: os.environ.get(k) for k in
                   ('BENCHMARK_BASE_URL', 'BENCHMARK_ADDRESS_FILE', 'BENCHMARK_MODEL')}
    if a.timeout <= 0:
        p.error('positive timeout required')
    if a.resume:
        manifest_path = a.out / 'manifest.json'
        if not a.out.is_dir() or not manifest_path.is_file():
            p.error('--resume requires an existing output directory with manifest.json')
        metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
        errors = _resume_errors(a, metadata, manifest, configs, roster)
        if errors:
            p.error('cannot resume: ' + '; '.join(errors))
        try:
            by_key = _load_records(a.out / 'records.jsonl', roster_keys)
        except (OSError, ValueError) as exc:
            p.error(f'cannot resume: {exc}')
        missing = {k for k in roster_keys if k not in by_key}
        _write_records(a.out / 'records.jsonl', [by_key[k] for k in roster_keys if k in by_key])
        metadata.setdefault('resumes', []).append({
            'remaining_at_start': len(missing), 'model_env': current_env, 'service_info': service_info()})
    else:
        if a.out.exists():
            p.error('output directory must be new; use --resume to continue an existing run')
        a.out.mkdir(parents=True)
        metadata = {'track': 'project-v1-synthetic-fixtures', 'timing_boundary': 'adapter subprocess round-trip; excludes browser',
                    'dataset': manifest, 'adapter': caps, 'configs': configs, 'repeats': a.repeats,
                    'split': a.split, 'only': a.only, 'timeout': a.timeout, 'record_traces': a.record_traces,
                    'model_env': current_env, 'service_info': service_info(), 'python': sys.version,
                    'git': git_state(), 'pip_freeze': pip_freeze(), 'resumes': [],
                    'roster': [[c['case_id'], cfg, rep] for c, cfg, rep in roster],
                    'code': fingerprint(), 'adapter_sha256': digest(a.adapter_manifest)}
        by_key, missing = {}, set(roster_keys)
        (a.out / 'records.jsonl').write_text('', encoding='utf-8')
    (a.out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (a.out / 'records.jsonl').open('a', encoding='utf-8') as f:
        for case, cfg, rep in roster:
            if (case['case_id'], cfg, rep) not in missing:
                continue
            rec = run_episode(a.adapter, case, catalogs[case['catalog_id']], cfg, a.timeout)
            rec['rep'] = rep
            if a.split == 'heldout' or not a.record_traces:
                rec = strip_traces(rec)
            by_key[(case['case_id'], cfg, rep)] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
    records = [by_key[k] for k in roster_keys]
    scores, report = summarize(records, cases, catalogs)
    report['missing_attempts'] = 0
    (a.out / 'scores.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in scores), encoding='utf-8')
    (a.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

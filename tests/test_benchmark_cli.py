"""Checks for the one-command benchmark entry point and report renderer."""
import json
from pathlib import Path
import subprocess
import sys

from evaluations.v1.report import render, write_report


def _minimal_run(root: Path) -> Path:
    out = root / 'run'
    out.mkdir()
    rng = {'n': 2, 'm': 2, 'rate': 1.0}
    report = {
        'summary': {
            'B0': {'runs': 2, 'plans_delivered': 2, 'tsr': rng, 'tsr_by_case': {'mean': 1.0, 'ci95': [1.0, 1.0], 'cases': 2},
                   'tsr_by_expected': {}, 'tsr_by_category': {}, 'bvr': {'n': 0, 'm': 2, 'rate': 0.0},
                   'constraint_violation': {'n': 0, 'm': 2, 'rate': 0.0}, 'unverifiable_claim': {'n': 0, 'm': 2, 'rate': 0.0},
                   'ungrounded_ids': {'n': 0, 'm': 2, 'rate': 0.0}, 'evidence_of_another_product': {'n': 0, 'm': 2, 'rate': 0.0},
                   'ids_not_in_context': {'n': 0, 'm': 2, 'rate': 0.0}, 'amount_error_rate': {'n': 0, 'm': 2, 'rate': 0.0},
                   'unverifiable_amount_plans': 0, 'outcome_reasons': {'ok': 2},
                   'wall_s': {'p50': 1.0, 'p95': 1.2, 'max': 1.2}, 'model_calls_mean': 1.0,
                   'prompt_tokens_mean': 100.0, 'completion_tokens_mean': 20.0, 'invalid_actions': 0,
                   'faithfulness': None},
        },
        'strata': {'task_type': {'single': {'B0': {'tsr': rng, 'bvr': {'n': 0, 'm': 2, 'rate': 0.0}}}}},
        'paired': [], 'missing_attempts': 0,
    }
    manifest = {'track': 'project-v1-synthetic-fixtures', 'split': 'dev', 'configs': ['B0'], 'repeats': 1,
                'dataset': {'status': 'awaiting_post_freeze_category', 'dev': 10, 'heldout': 32, 'pending': 8,
                            'hashes': {'cases.jsonl': 'a' * 64}},
                'adapter': {'identity': {'model': 'qwen3.8-27b'}, 'validator_configs': []},
                'timing_boundary': 'adapter subprocess round-trip; excludes browser', 'code': {'x.py': 'hash'}}
    (out / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return out


def test_report_has_proposal_columns_and_na_faithfulness(tmp_path):
    out = _minimal_run(tmp_path)
    text = render(json.loads((out / 'manifest.json').read_text()), json.loads((out / 'report.json').read_text()))
    assert '| Method | Configuration | TSR % ↑ | Faithfulness ↑ | BVR % ↓ | p95 s ↓ |' in text
    assert '| B0 Single call | No extra text; no budget tool | 100.0 | N/A | 0.0 | 1.20 |' in text
    assert 'human review merged: no' in text


def test_write_report_command(tmp_path):
    out = _minimal_run(tmp_path)
    destination = tmp_path / 'REPORT.md'
    result = subprocess.run([sys.executable, '-m', 'evaluations.bench', 'report',
                             '--input', str(out), '--out', str(destination)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert destination.is_file() and 'B0 Single call' in destination.read_text()


def test_check_reports_ready_without_endpoint(tmp_path):
    from evaluations.v1.suite import build
    build(tmp_path / 'data')
    caps = tmp_path / 'adapter.json'
    caps.write_text(json.dumps({'identity': {'model': 'qwen3.8-27b'}, 'validator_configs': [],
                                'cache_policy': 'none'}), encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'evaluations.bench', 'check',
                             '--data', str(tmp_path / 'data'), '--configs', 'B0',
                             '--adapter-manifest', str(caps)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'RESULT: READY' in result.stdout
    assert '没有配置模型端点' in result.stdout


def test_write_report_rejects_missing_report(tmp_path):
    (tmp_path / 'manifest.json').write_text('{}', encoding='utf-8')
    try:
        write_report(tmp_path)
    except ValueError as exc:
        assert 'report.json' in str(exc)
    else:
        raise AssertionError('missing report.json must be rejected')


def test_preflight_rejects_a_missing_model_alias(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from evaluations import bench
    from evaluations.v1.suite import build

    data = tmp_path / 'data'
    build(data)
    caps = tmp_path / 'adapter.json'
    caps.write_text(json.dumps({'identity': {'model': 'qwen3.8-27b'}, 'validator_configs': [],
                                'cache_policy': 'none'}), encoding='utf-8')
    args = SimpleNamespace(data=data, split='dev', configs='B0', repeats=1, only='',
                           adapter='evaluations.v1.model_adapter:respond',
                           adapter_manifest=caps, timeout=240, base_url='http://127.0.0.1:1/v1',
                           address_file=None, model='qwen3.8-27b', service_info=None,
                           out=tmp_path / 'run', resume=False)
    monkeypatch.setattr(bench, '_probe', lambda url: (True, 'ok', ['another-model']))
    blockers, _, _ = bench.preflight(args)
    assert any('没有请求的模型' in item for item in blockers)

    monkeypatch.setattr(bench, '_probe', lambda url: (True, 'ok', ['qwen3.8-27b']))
    blockers, _, _ = bench.preflight(args)
    assert not any('没有请求的模型' in item for item in blockers)

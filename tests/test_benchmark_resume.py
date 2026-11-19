"""Resume must append only missing attempts and refuse a changed run."""
import json
from pathlib import Path
import subprocess
import sys

from evaluations.v1.suite import build


def _caps(path: Path, policy='none'):
    path.write_text(json.dumps({'identity': 'test-only-error-adapter', 'validator_configs': [],
                                'cache_policy': policy}), encoding='utf-8')


def _command(data: Path, caps: Path, out: Path, *extra):
    return [sys.executable, '-m', 'evaluations.v1.run', '--data', str(data),
            '--adapter', 'tests.test_benchmark_v1:always_error', '--adapter-manifest', str(caps),
            '--configs', 'B0', '--repeats', '1', '--out', str(out), *extra]


def test_resume_appends_only_missing_attempts(tmp_path):
    data = tmp_path / 'data'
    build(data)
    caps = tmp_path / 'adapter.json'
    _caps(caps)
    out = tmp_path / 'run'
    subprocess.run(_command(data, caps, out), check=True, capture_output=True)
    lines = (out / 'records.jsonl').read_text().splitlines()
    assert len(lines) == 10
    (out / 'records.jsonl').write_text('\n'.join(lines[:6]) + '\n', encoding='utf-8')
    subprocess.run(_command(data, caps, out, '--resume'), check=True, capture_output=True)
    rows = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    assert len(rows) == 10
    assert len({(r['case_id'], r['config'], r['rep']) for r in rows}) == 10
    report = json.loads((out / 'report.json').read_text())
    assert report['missing_attempts'] == 0
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['resumes'][-1]['remaining_at_start'] == 4


def test_resume_refuses_a_changed_adapter_manifest(tmp_path):
    data = tmp_path / 'data'
    build(data)
    caps = tmp_path / 'adapter.json'
    _caps(caps)
    out = tmp_path / 'run'
    subprocess.run(_command(data, caps, out), check=True, capture_output=True)
    _caps(caps, 'different')
    result = subprocess.run(_command(data, caps, out, '--resume'), capture_output=True, text=True)
    assert result.returncode != 0
    assert 'adapter manifest changed' in result.stderr


def test_load_records_tolerates_a_non_object_final_line(tmp_path):
    from evaluations.v1.run import _load_records
    roster = [('x', 'B0', 0)]
    good = {'case_id': 'x', 'config': 'B0', 'rep': 0, 'outcome': {'kind': 'error'}}
    path = tmp_path / 'records.jsonl'
    path.write_text(json.dumps(good) + '\n[]\n', encoding='utf-8')
    assert _load_records(path, roster) == {('x', 'B0', 0): good}


def test_load_records_rejects_a_non_object_middle_line(tmp_path):
    from evaluations.v1.run import _load_records
    path = tmp_path / 'records.jsonl'
    path.write_text('[]\n{"case_id":"x","config":"B0","rep":0}\n', encoding='utf-8')
    try:
        _load_records(path, [('x', 'B0', 0)])
    except ValueError as exc:
        assert 'not an object' in str(exc)
    else:
        raise AssertionError('non-object middle line must be rejected')


def test_rescore_counts_a_non_object_line_as_malformed(tmp_path):
    data = tmp_path / 'data'
    build(data)
    caps = tmp_path / 'adapter.json'
    _caps(caps)
    out = tmp_path / 'run'
    subprocess.run(_command(data, caps, out), check=True, capture_output=True)
    lines = (out / 'records.jsonl').read_text().splitlines()
    (out / 'records.jsonl').write_text('\n'.join(lines[:-1] + ['[]']) + '\n', encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'evaluations.v1.run', '--data', str(data),
                             '--rescore', str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / 'report.json').read_text())
    assert report['malformed_lines_ignored'] == 1
    assert report['missing_attempts'] == 1

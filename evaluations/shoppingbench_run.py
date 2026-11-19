"""Prepare and run a pinned, unmodified ShoppingBench voucher harness.

Preflight never calls the LLM. Upstream's python_execute tool runs model-generated
Python; run the actual harness inside the provided disposable container, mounting
only public benchmark assets and this run's output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

UPSTREAM_COMMIT = '204dfba767cf849fe33535e42293e8973fb017ea'
CORPUS_SHA256 = 'a0ba9a8618ce72a7b383a6de31acf6e5bf138873da0d13ad8a1a6b0271e8e9d1'
CORPUS_BYTES = 1471795629


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_corpus(path, expected_bytes=CORPUS_BYTES, expected_sha256=CORPUS_SHA256):
    """Return (digest, blocker); the raw corpus hash is always checked."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected_bytes:
        return None, 'missing_full_lfs_corpus'
    digest = file_hash(path)
    if digest != expected_sha256:
        return digest, 'corpus_hash_mismatch'
    return digest, None


def index_digest(directory):
    """Same merged hash as deploy/benchmark/build_shoppingbench_index.py."""
    directory = Path(directory)
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(directory)).encode() + b'\0')
        digest.update(bytes.fromhex(file_hash(path)))
    return {'index_files': len(files), 'index_bytes': sum(p.stat().st_size for p in files),
            'index_sha256': digest.hexdigest()}


def verify_index(directory, manifest):
    errors = []
    actual = index_digest(directory)
    for key in ('index_sha256', 'index_files', 'index_bytes'):
        if manifest.get(key) != actual[key]:
            errors.append('index_' + key.replace('index_', '') + '_mismatch')
    return errors


def preflight(repo, revision=UPSTREAM_COMMIT, check_corpus=True):
    blockers, hashes = [], {}
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != revision:
        blockers.append('upstream_commit_mismatch')
    dirty = subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--', 'src/agent', 'src/search_engine', 'data/synthesize_voucher_test.jsonl'], text=True)
    if dirty:
        blockers.append('modified_or_incomplete_upstream_sources')
    required = ['src/agent/run_rollout.py', 'src/agent/run_evaluate.py', 'src/search_engine/server.py',
                'src/agent/toolkit/python_execute.py', 'requirements.txt', 'data/synthesize_voucher_test.jsonl']
    for relative in required:
        path = repo / relative
        if not path.is_file():
            blockers.append('missing:' + relative)
        else:
            hashes[relative] = file_hash(path)
    task_file = repo / 'data/synthesize_voucher_test.jsonl'
    count = 0
    if task_file.is_file():
        tasks = [json.loads(l) for l in task_file.read_text().splitlines() if l.strip()]
        count = len(tasks)
        if count != 250 or len({t['query'] for t in tasks}) != count:
            blockers.append('voucher_roster_must_have_250_unique_tasks')
    corpus = repo / 'resources/documents.jsonl.gz'
    # The hash is always checked. A size match alone cannot tie results to the
    # declared corpus, and `--check-corpus` is kept only as a CLI compatibility alias.
    corpus_digest, corpus_blocker = verify_corpus(corpus)
    if corpus_blocker:
        blockers.append(corpus_blocker)
    if corpus_digest:
        hashes['resources/documents.jsonl.gz'] = corpus_digest
    if not (repo / 'indexes').is_dir() or not list((repo / 'indexes').glob('segments*')):
        blockers.append('missing_lucene_index')
    index_build = repo / 'index-build.json'
    index_manifest = None
    if not index_build.is_file():
        blockers.append('missing:index-build.json')
    else:
        hashes['index-build.json'] = file_hash(index_build)
        try:
            index_manifest = json.loads(index_build.read_text(encoding='utf-8'))
        except ValueError:
            blockers.append('invalid_index_manifest')
        if index_manifest and (not index_manifest.get('index_sha256') or not index_manifest.get('index_files')):
            blockers.append('incomplete_index_manifest')
        elif index_manifest:
            blockers.extend(verify_index(repo / 'indexes', index_manifest))
    return {'track': 'ShoppingBench-voucher', 'commit': commit, 'tasks': count,
            'hashes': hashes, 'index_build': index_manifest,
            'index_verified': bool(index_manifest and not any(
                b.startswith('index_') or b == 'incomplete_index_manifest' for b in blockers)),
            'blockers': blockers, 'ready': not blockers,
            'corpus_contract': {'sha256': CORPUS_SHA256, 'bytes': CORPUS_BYTES}}


def config(repo, out, model):
    return {'task': 'voucher', 'system_prompt_file': str(repo / 'src/agent/prompt/rollout.md'),
        'synthesize_file': str(repo / 'data/synthesize_voucher_test.jsonl'),
        'rollout_file': str(out / 'rollouts.jsonl'), 'threads': 4,
        'model_config': {'model': model, 'temperature': 0, 'max_tokens': 2048, 'stream': True,
                         'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--revision', default=UPSTREAM_COMMIT)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--model', default='qwen3.8-27b')
    p.add_argument('--check-corpus', action='store_true',
                   help='deprecated compatibility flag; corpus SHA256 is always checked')
    p.add_argument('--resume-from', type=Path, help='preserve API-complete episodes, retry empty API episodes only')
    p.add_argument('--run', action='store_true', help='only inside a disposable isolated environment')
    a = p.parse_args()
    repo, out = a.repo.resolve(), a.out.resolve()
    if out.exists():
        p.error('use a new output directory')
    result = preflight(repo, a.revision, a.check_corpus)
    out.mkdir(parents=True)
    (out / 'preflight.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    cfg = config(repo, out, a.model)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    if a.resume_from:
        from .shoppingbench_recovery import prepare
        tasks = [json.loads(l) for l in (repo / 'data/synthesize_voucher_test.jsonl').read_text().splitlines() if l.strip()]
        recovery = prepare(tasks, a.resume_from.resolve(), out / 'rollouts.jsonl')
        (out / 'recovery.json').write_text(json.dumps(recovery, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not a.run:
        return
    if not result['ready']:
        p.error('preflight blockers; see preflight.json')
    if not os.environ.get('SHOPPINGBENCH_ISOLATED') or not Path('/SHOPPINGBENCH_ISOLATED').is_file():
        p.error('use the disposable container: upstream executes generated Python')
    if not os.environ.get('OPENAI_BASE_URL'):
        p.error('OPENAI_BASE_URL must point to the intended model')
    env = {**os.environ, 'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', 'local-no-key')}
    # Official imports resolve relative to the upstream agent directory.
    env['PYTHONPATH'] = str(repo / 'src/agent') + os.pathsep + env.get('PYTHONPATH', '')
    search = subprocess.Popen([sys.executable, str(repo / 'src/search_engine/server.py')], cwd=repo, env=env,
                              stdout=(out / 'search.log').open('w'), stderr=subprocess.STDOUT)
    try:
        import time
        from urllib.request import urlopen
        for _ in range(120):
            if search.poll() is not None:
                raise RuntimeError('search server exited; see search.log')
            try:
                urlopen('http://127.0.0.1:5631/find_product?q=keyboard&page=1', timeout=2).close()
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('search server readiness timeout')
        (out / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        with (out / 'rollout.log').open('w') as log:
            subprocess.run([sys.executable, str(repo / 'src/agent/run_rollout.py'), str(out / 'config.json')],
                           cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        search.terminate()
        try:
            search.wait(timeout=10)
        except subprocess.TimeoutExpired:
            search.kill()
            search.wait()


if __name__ == '__main__':
    main()

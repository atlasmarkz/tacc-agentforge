"""Build the official Lucene index in a disposable Docker workspace.

Only the uncompressed JSONL is exposed to JsonCollection: exposing both .jsonl
and .jsonl.gz would index the same corpus twice. Existing indexes are never
removed. The source corpus stays read-only; temporary input uses a hard link.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--image', default='agentforge-shoppingbench:204dfba')
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--memory', default='6g')
    p.add_argument('--java-xmx', default='4g')
    a = p.parse_args()
    repo = a.repo.resolve()
    document = repo / 'resources/documents.jsonl'
    if not document.is_file() or document.stat().st_size != 5112146514:
        p.error('full verified corpus must be decompressed first')
    if (repo / 'indexes').exists():
        p.error('index exists; use a new dedicated benchmark checkout')
    work = Path(tempfile.mkdtemp(prefix='shoppingbench-index-', dir=repo.parent))
    inputs, output = work / 'input', work / 'output'
    inputs.mkdir()
    output.mkdir()
    output.chmod(0o777)  # This fresh directory holds only public generated indexes.
    os.link(document, inputs / 'documents.jsonl')
    command = ['docker', 'run', '--rm', f'--memory={a.memory}', '-e', f'JAVA_TOOL_OPTIONS=-Xmx{a.java_xmx}',
        '--mount', f'type=bind,source={repo},target=/upstream,readonly',
        '--mount', f'type=bind,source={output},target=/bench',
        '--mount', f'type=bind,source={inputs},target=/bench/resources,readonly',
        a.image, 'bash', '/upstream/build_index.sh']
    print('INDEX_BUILD_WORKSPACE=' + str(work), flush=True)
    with (work / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout, check=True)
    index = output / 'indexes'
    if not list(index.glob('segments*')):
        raise RuntimeError('no completed Lucene index; see ' + str(work / 'build.log'))
    index.rename(repo / 'indexes')
    (inputs / 'documents.jsonl').unlink()
    index = repo / 'indexes'
    files = sorted(p for p in index.rglob('*') if p.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(index)).encode() + b'\0')
        digest.update(bytes.fromhex(file_sha256(path)))
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', a.image], text=True).strip()
    (repo / 'index-build.json').write_text(json.dumps({
        'image': image, 'corpus_bytes': document.stat().st_size,
        'corpus_sha256': file_sha256(document), 'input_files': ['documents.jsonl'],
        'index_files': len(files), 'index_bytes': sum(p.stat().st_size for p in files),
        'index_sha256': digest.hexdigest(), 'log': str(work / 'build.log')}, indent=2), encoding='utf-8')
    print('INDEX_READY=' + str(repo / 'indexes'))


if __name__ == '__main__':
    main()

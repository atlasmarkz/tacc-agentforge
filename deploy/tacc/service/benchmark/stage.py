"""Stage only public sources and the development fixtures for a CPU-only job."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
package = HERE / 'pkg'
if package.exists():
    shutil.rmtree(package)
package.mkdir()
for name in ('agent', 'budget', 'evaluations', 'inference', 'knowledge'):
    shutil.copytree(ROOT / name, package / name,
        ignore=shutil.ignore_patterns('.venv', '__pycache__', 'tests', 'results', '*.pyc'))
# The worker is a dev-only smoke job. Never copy the sealed heldout split to a
# shared TACC user directory.
data = package / 'evaluations/v1/data'
if data.is_dir():
    cases = [json.loads(line) for line in (ROOT / 'evaluations/v1/data/cases.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    dev = [case for case in cases if case['split'] == 'dev']
    catalogs = json.loads((ROOT / 'evaluations/v1/data/catalogs.json').read_text(encoding='utf-8'))
    dev_catalogs = {case['catalog_id']: catalogs[case['catalog_id']] for case in dev}
    (data / 'cases.jsonl').write_text(
        ''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in dev), encoding='utf-8')
    (data / 'catalogs.json').write_text(json.dumps(dev_catalogs, ensure_ascii=False, indent=2) + '\n',
                                        encoding='utf-8')
    hashes = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
              for name in ('cases.jsonl', 'catalogs.json')}
    (data / 'manifest.json').write_text(json.dumps({
        'version': 1, 'synthetic': True, 'status': 'dev_only',
        'dev': len(dev), 'heldout': 0, 'pending': 0, 'hashes': hashes}, indent=2) + '\n', encoding='utf-8')
# Runtime metadata must distinguish current serving config from controlled-cache claims.
(package / 'configs').mkdir()
shutil.copy(ROOT / 'deploy/tacc/service/qwen-service/service.json', package / 'configs/serving-observed.json')
print(f'Staged {sum(p.is_file() for p in package.rglob("*"))} public files in {package}')

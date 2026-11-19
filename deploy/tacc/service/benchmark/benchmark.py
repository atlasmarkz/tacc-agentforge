"""Real-model development smoke; never unseals heldout data or starts GPUs."""
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlsplit

from evaluations.v1.run import main
from evaluations.run import wait_for_service

root = Path(os.environ['TACC_USERDIR']) / 'tacc-agentforge-service'
out = root / ('benchmark-v1-' + os.environ['SLURM_JOB_ID'])
out.parent.mkdir(parents=True, exist_ok=True)
address_file = Path(os.environ['BENCHMARK_ADDRESS_FILE'])
# Refuse a stale publication on the old shared port while the new job starts.
expected_port = int(os.environ.get('BENCHMARK_EXPECTED_PORT', '18966'))
deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    address = address_file.read_text().strip() if address_file.exists() else ''
    if address and urlsplit(address).port == expected_port:
        break
    time.sleep(5)
else:
    raise SystemExit('expected account-owned service was not published')
wait_for_service(address_file, wait_s=120)
expected_service_job = os.environ.get('BENCHMARK_EXPECTED_SERVICE_JOB', '').strip()
service_info = None
if expected_service_job:
    info_path = root / f'service-{expected_service_job}/service-info.json'
    if not info_path.is_file():
        raise SystemExit('expected service job *' + expected_service_job + ' has no service-info.json')
    service_info = json.loads(info_path.read_text())
    if service_info.get('address') != address:
        raise SystemExit('published address does not match the expected service job')
if service_info is None:
    for info_path in sorted(root.glob('service-*/service-info.json'), reverse=True):
        try:
            candidate = json.loads(info_path.read_text())
        except (OSError, ValueError):
            continue
        if candidate.get('address') == address:
            service_info = candidate
            break
metadata = {'job_id': os.environ['SLURM_JOB_ID'], 'node': socket.gethostname(),
            'service_address': Path(os.environ['BENCHMARK_ADDRESS_FILE']).read_text().strip(),
            'service_job_id': service_info.get('job_id') if service_info else expected_service_job or None,
            'service_started_at': service_info.get('started_at') if service_info else None,
            'serving_config_reference': json.loads(Path('pkg/configs/serving-observed.json').read_text()),
            'cache_controlled': False, 'track': 'synthetic-fixtures-real-model-dev-smoke'}
(root / ('benchmark-v1-' + os.environ['SLURM_JOB_ID'] + '-service.json')).write_text(json.dumps(metadata, indent=2))
# run.py records BENCHMARK_SERVICE_INFO in its manifest; without this the TACC
# worker's run has service_info=null even though this job knows the service.
service_meta = dict(service_info or {})
service_meta.setdefault('job_id', expected_service_job or None)
service_meta.setdefault('address', metadata['service_address'])
service_meta.setdefault('node', socket.gethostname())
service_meta['runner_job_id'] = os.environ['SLURM_JOB_ID']
service_info_path = root / ('benchmark-v1-' + os.environ['SLURM_JOB_ID'] + '-service-info.json')
service_info_path.write_text(json.dumps(service_meta, indent=2, ensure_ascii=False), encoding='utf-8')
os.environ['BENCHMARK_SERVICE_INFO'] = str(service_info_path)
sys.argv = ['benchmark', '--data', 'pkg/evaluations/v1/data', '--split', 'dev',
    '--configs', 'B0,B1,B2,B3,A1,A2', '--only', 'v1-dev-01,v1-dev-03,v1-dev-04', '--repeats', '1',
    '--adapter', 'evaluations.v1.agent_adapter:respond',
    '--adapter-manifest', 'pkg/evaluations/v1/agent-adapter.example.json', '--out', str(out)]
main()
(out / 'service.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('RESULTS_DIR=' + str(out), flush=True)

"""Durable, bounded supervision of a ShoppingBench recovery run.

Run detached with a new output root. Keeps SSH independent of an interactive tool
session, pauses the container for service rotation, and cancels only allocations
it submits. A stop.request file pauses the container and releases owned jobs.
The initial interrupted run remains immutable. No reward-based retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

# Keep the script runnable from the repository root or directly from deploy/.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from evaluations.shoppingbench_recovery import api_complete  # noqa: E402


class Supervisor:
    def __init__(self, args):
        self.a = args
        self.root = args.out.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'supervisor.json'
        self.sock = '/tmp/af-bench-' + hashlib.sha256(str(self.root).encode()).hexdigest()[:12] + '.sock'
        self.name = 'agentforge-recovery-' + hashlib.sha256(str(self.root).encode()).hexdigest()[:10]
        self.state = {'status': 'initializing', 'jobs': [], 'container': self.name, 'events': [],
                      'model': args.model, 'max_allocations': args.max_allocations}
        self.tunnel_target = None
        self.released = set()
        self.stopping = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stopping', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stopping', True))

    def save(self, status=None, **values):
        if status:
            self.state['status'] = status
        self.state.update(values, updated_epoch=time.time())
        temp = self.state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(self.state_path)
        print(json.dumps({k: self.state.get(k) for k in ('status', 'completed', 'current_job', 'updated_epoch')}, ensure_ascii=False), flush=True)

    def event(self, kind, **values):
        self.state['events'].append({'kind': kind, 'epoch': time.time(), **values})
        self.save()

    def ssh_base(self):
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
                '-p', str(self.a.ssh_port), '-i', str(self.a.identity), self.a.target]

    def remote_raw(self, command):
        return subprocess.run(self.ssh_base() + [command], capture_output=True, text=True, timeout=45)

    def remote(self, command, check=True):
        r = self.remote_raw(command)
        if check and r.returncode:
            raise RuntimeError('SSH command failed: ' + r.stderr[-300:])
        return r.stdout.strip()

    def submit(self):
        if len(self.state['jobs']) >= self.a.max_allocations:
            raise RuntimeError('bounded allocation limit reached')
        # Only the user's pre-staged service job is invoked; no user-home mounts enter Docker.
        work = self.a.remote_workdir
        self.remote(f'printf "%s\\n" "--deadline-s 3400 --port {self.a.model_port} --max-restarts 0" > {work}/args.txt')
        job = self.remote(f'sbatch --parsable --nodelist={self.a.node} --time=01:00:00 {work}/configurations/run.slurm')
        if not job.isdigit():
            raise RuntimeError('unexpected sbatch identity: ' + job)
        self.state['jobs'].append(job)
        self.save('waiting_for_allocation', current_job=job)
        self.event('allocation_submitted', job=job)
        return job

    @staticmethod
    def _elapsed(text):
        text = (text or '0:00').strip()
        days = 0
        if '-' in text:
            day_text, text = text.split('-', 1)
            days = int(day_text)
        seconds = 0
        for value in (int(v) for v in text.split(':') if v != ''):
            seconds = seconds * 60 + value
        return days * 86400 + seconds

    def sacct(self, job):
        r = self.remote_raw(f'sacct -j {job} --format=State,Elapsed,User -n -P')
        if r.returncode:
            return None
        terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
                    'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
        states = []
        for line in r.stdout.splitlines():
            parts = line.strip().split('|')
            if len(parts) < 2 or not parts[0].strip():
                continue
            state = parts[0].split()[0].upper()
            owner = parts[2].strip() if len(parts) > 2 else ''
            if owner and owner != self.a.target.split('@')[0]:
                raise RuntimeError('job owner mismatch; refusing control')
            states.append((state, self._elapsed(parts[1])))
        if not states:
            return None
        for state, seconds in states:
            if state in terminal:
                return state, seconds
        for state, seconds in states:
            if state in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'):
                return state, seconds
        return states[0]

    def job(self, job):
        r = self.remote_raw(f'squeue -h -j {job} -o "%T|%M|%u"')
        text = r.stdout.strip() if r.returncode == 0 else ''
        if text:
            state, elapsed, owner = text.splitlines()[0].split('|')
            if owner != self.a.target.split('@')[0]:
                raise RuntimeError('job owner mismatch; refusing control')
            return state, self._elapsed(elapsed)
        # Empty squeue is not proof of termination: an SSH failure and the
        # submit/registration race look identical. Confirm with sacct or wait.
        return self.sacct(job) or ('UNKNOWN', 0)

    def close_tunnel(self):
        subprocess.run(self.ssh_base()[:-1] + ['-S', self.sock, '-O', 'exit', self.a.target],
                       capture_output=True)
        self.tunnel_target = None

    def tunnel(self, address):
        match = re.fullmatch(r'http://(10\.[0-9]+\.[0-9]+\.[0-9]+):([0-9]+)/v1', address)
        if not match or int(match[2]) != self.a.model_port:
            raise RuntimeError('unexpected model endpoint')
        target = (match[1], int(match[2]))
        check = subprocess.run(self.ssh_base()[:-1] + ['-S', self.sock, '-O', 'check', self.a.target],
                               capture_output=True)
        if self.tunnel_target != target or check.returncode:
            # A live control socket can still forward to the previous job's
            # node/port. Never reuse it across a service rotation.
            self.close_tunnel()
            base = self.ssh_base()[:-1]
            subprocess.run(base + ['-M', '-S', self.sock, '-fnNT', '-o', 'ExitOnForwardFailure=yes',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-L',
                f'127.0.0.1:{self.a.local_port}:{match[1]}:{match[2]}', self.a.target], check=True, timeout=30)
            self.tunnel_target = target
        with urlopen(f'http://127.0.0.1:{self.a.local_port}/v1/models', timeout=10) as r:
            models = json.load(r)
        if self.a.model not in {m['id'] for m in models['data']}:
            raise RuntimeError('model alias mismatch')
        return models

    def ready(self, job):
        info_file = f'{self.a.remote_userdir}/tacc-agentforge-service/service-{job}/service-info.json'
        value = self.remote(f'cat {info_file}', check=False)
        if not value:
            return False
        info = json.loads(value)
        if str(info.get('job_id')) != job:
            raise RuntimeError('service identity mismatch')
        models = self.tunnel(info['address'])
        (self.root / f'service-{job}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        (self.root / 'model-info.json').write_text(json.dumps(models, indent=2), encoding='utf-8')
        return True

    def docker_state(self):
        r = subprocess.run(['docker', 'inspect', '--format', '{{.State.Status}}', self.name], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    def pause(self):
        if self.docker_state() == 'running':
            subprocess.run(['docker', 'pause', self.name], check=True, capture_output=True)
            self.event('container_paused')

    def start_or_resume(self):
        state = self.docker_state()
        if state == 'paused':
            subprocess.run(['docker', 'unpause', self.name], check=True, capture_output=True)
            self.event('container_resumed')
        elif state is None:
            output = self.root / 'output'
            output.mkdir(exist_ok=True)
            output.chmod(0o777)
            host_alias = (['--add-host', 'host.docker.internal:host-gateway']
                          if sys.platform.startswith('linux') else [])
            cmd = ['docker', 'run', '-d', '--name', self.name,
                f'--memory={self.a.container_memory}', *host_alias,
                '-e', 'JAVA_TOOL_OPTIONS=-Xmx3g', '-e', 'PYTHONPATH=/project',
                '-e', f'OPENAI_BASE_URL=http://host.docker.internal:{self.a.local_port}/v1',
                '-e', 'OPENAI_API_KEY=local-no-key', '-e', 'GIT_CONFIG_COUNT=1',
                '-e', 'GIT_CONFIG_KEY_0=safe.directory', '-e', 'GIT_CONFIG_VALUE_0=/bench',
                '--mount', f'type=bind,source={self.a.repo.resolve()},target=/bench,readonly',
                '--mount', f'type=bind,source={self.a.project.resolve()},target=/project,readonly',
                '--mount', f'type=bind,source={output},target=/outputs',
                '--mount', f'type=bind,source={self.a.prior.resolve()},target=/prior.jsonl,readonly',
                self.a.image, 'python', '-u', '-m', 'evaluations.shoppingbench_run', '--repo', '/bench',
                '--out', '/outputs/voucher', '--model', self.a.model, '--check-corpus', '--run', '--resume-from', '/prior.jsonl']
            container = subprocess.check_output(cmd, text=True).strip()
            self.event('container_started', container_id=container)

    def counts(self):
        p = self.root / 'output/voucher/rollouts.jsonl'
        queries, api_ok = set(), 0
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                try:
                    row = json.loads(line)
                    query = row[0]['extra_info']['query']
                except (ValueError, KeyError, IndexError, TypeError):
                    continue  # Last line may be in flight; final report performs strict validation.
                queries.add(query)
                if api_complete(row):
                    api_ok += 1
        inherited = 0
        recovery = self.root / 'output/voucher/recovery.json'
        if recovery.exists():
            try:
                inherited = int(json.loads(recovery.read_text(encoding='utf-8')).get('inherited_api_complete', 0))
            except (ValueError, TypeError):
                inherited = 0
        return {'completed': len(queries), 'api_complete': api_ok,
                'inherited_api_complete': inherited,
                'retried_new_api_complete': max(0, api_ok - inherited)}

    def completed(self):
        return self.counts()['completed']

    def cancel(self, job):
        if job not in self.state['jobs']:
            raise RuntimeError('refusing to cancel unowned allocation')
        r = self.remote_raw(f'scancel {job}')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            q = self.remote_raw(f'squeue -h -j {job} -o "%T"')
            if q.returncode == 0 and not q.stdout.strip():
                # Already completed jobs can make scancel return non-zero; an
                # empty, successfully queried queue is the release evidence.
                self.released.add(job)
                self.event('allocation_released', job=job)
                return
            time.sleep(2)
        if r.returncode:
            raise RuntimeError('scancel failed: ' + r.stderr[-300:])
        raise RuntimeError(f'job {job} still visible after scancel')

    def run(self):
        started, job = time.monotonic(), None
        if self.state_path.exists():
            raise RuntimeError('output already supervised; inspect state before restarting')
        self.save()
        try:
            job = self.submit()
            while time.monotonic() - started < self.a.max_hours * 3600:
                if self.stopping or (self.root / 'stop.request').exists():
                    self.pause()
                    self.save('stopped_by_request', **self.counts())
                    return
                current, state = self.counts(), self.docker_state()
                count = current['completed']
                if state == 'exited':
                    logs = subprocess.check_output(['docker', 'logs', self.name], stderr=subprocess.STDOUT)
                    (self.root / 'container.log').write_bytes(logs)
                    current = self.counts()
                    count = current['completed']
                    self.save('completed' if count == current['api_complete'] == 250 else
                              'completed_with_api_failures' if count == 250 else
                              'incomplete_container_exit', **current)
                    return
                status, elapsed = self.job(job)
                if status == 'UNKNOWN':
                    # No trustworthy scheduler state: never cancel or resubmit
                    # on an empty/ambiguous read.
                    self.pause()
                    self.save('waiting_for_allocation', completed=count,
                              last_error='scheduler returned no state; refusing to cancel')
                elif status == 'RUNNING' and elapsed < 3200:
                    try:
                        if self.ready(job):
                            self.start_or_resume()
                            self.save('evaluating', completed=count)
                        else:
                            self.pause()
                            self.save('waiting_for_model', completed=count)
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        self.pause()
                        self.save('waiting_for_model', completed=count, last_error=str(exc)[:300])
                elif status in ('PENDING', 'CONFIGURING'):
                    self.pause()
                    self.save('waiting_for_allocation', completed=count)
                else:
                    self.pause()
                    self.cancel(job)
                    if len(self.state['jobs']) >= self.a.max_allocations:
                        self.save('paused_allocation_limit', **self.counts())
                        return
                    job = self.submit()
                time.sleep(15)
            self.pause()
            self.save('paused_wall_limit', **self.counts())
        except Exception as exc:
            self.pause()
            self.save('error', error=str(exc)[:500], **self.counts())
            raise
        finally:
            cleanup_errors = []
            for own_job in self.state['jobs']:
                if own_job in self.released:
                    continue
                try:
                    self.cancel(own_job)
                except Exception as exc:
                    cleanup_errors.append({'job': own_job, 'error': str(exc)[:200]})
                    self.state['events'].append({'kind': 'cleanup_error', 'job': own_job, 'error': str(exc)[:200]})
            self.close_tunnel()
            self.save(cleanup_errors=cleanup_errors, **self.counts())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--project', required=True, type=Path, help='public-only Python module bundle')
    p.add_argument('--prior', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--target', required=True, help='account@gateway')
    p.add_argument('--identity', required=True, type=Path)
    p.add_argument('--remote-workdir', required=True)
    p.add_argument('--remote-userdir', required=True)
    p.add_argument('--ssh-port', type=int, default=30041)
    p.add_argument('--model-port', type=int, default=18966)
    p.add_argument('--local-port', type=int, default=18978)
    p.add_argument('--node', default='10-0-4-10')
    p.add_argument('--model', default='qwen3.8-27b')
    p.add_argument('--image', default='agentforge-shoppingbench:204dfba')
    p.add_argument('--max-allocations', type=int, default=2,
                   help='number of 1-hour allocations this supervisor may submit (1..10)')
    p.add_argument('--max-hours', type=float, default=4)
    p.add_argument('--container-memory', default='6g',
                   help='Docker memory limit for the official harness container')
    a = p.parse_args()
    if not 1 <= a.max_allocations <= 10:
        p.error('--max-allocations must be 1..10')
    if not re.fullmatch(r'[0-9]+[mg]', a.container_memory):
        p.error('--container-memory must look like 5g or 4096m')
    for value in (a.target, a.node, a.remote_workdir, a.remote_userdir):
        if not re.fullmatch(r'[A-Za-z0-9@_./-]+', value):
            p.error('unsafe remote identifier')
    Supervisor(a).run()


if __name__ == '__main__':
    main()

"""No SSH, Docker or scheduler mutations in these resource-safety tests."""
from types import SimpleNamespace
from pathlib import Path

import pytest
from deploy.benchmark.recover_public_run import Supervisor


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    import deploy.benchmark.recover_public_run as module
    monkeypatch.setattr(module.signal, 'signal', lambda *_: None)
    return Supervisor(SimpleNamespace(out=tmp_path, model='test', max_allocations=2,
        target='mine@gateway', max_hours=1, identity=Path('/unused'), ssh_port=30041))


def test_cannot_cancel_an_allocation_not_submitted_by_this_run(supervisor, monkeypatch):
    monkeypatch.setattr(supervisor, 'remote', lambda *_a, **_k: pytest.fail('must not contact scheduler'))
    with pytest.raises(RuntimeError, match='unowned'):
        supervisor.cancel('999')


def test_job_owner_mismatch_prevents_control(supervisor, monkeypatch):
    monkeypatch.setattr(supervisor, 'remote_raw', lambda *_a, **_k: SimpleNamespace(
        returncode=0, stdout='RUNNING|1:00|someone_else\n', stderr=''))
    with pytest.raises(RuntimeError, match='owner mismatch'):
        supervisor.job('123')


def test_empty_squeue_is_not_treated_as_finished(supervisor, monkeypatch):
    monkeypatch.setattr(supervisor, 'remote_raw', lambda *_a, **_k: SimpleNamespace(
        returncode=0, stdout='', stderr=''))
    assert supervisor.job('123') == ('UNKNOWN', 0)


def test_failed_scancel_is_not_recorded_as_released(supervisor, monkeypatch):
    import deploy.benchmark.recover_public_run as module
    supervisor.state['jobs'] = ['123']
    def raw(command):
        if command.startswith('scancel'):
            return SimpleNamespace(returncode=255, stdout='', stderr='ssh: connect failed')
        return SimpleNamespace(returncode=0, stdout='RUNNING\n', stderr='')
    ticks = iter(range(0, 1000, 5))
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(module.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(supervisor, 'remote_raw', raw)
    with pytest.raises(RuntimeError, match='scancel failed'):
        supervisor.cancel('123')
    assert not [e for e in supervisor.state['events'] if e['kind'] == 'allocation_released']


def test_cancel_confirms_squeue_absence(supervisor, monkeypatch):
    supervisor.state['jobs'] = ['123']
    def raw(command):
        if command.startswith('scancel'):
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(supervisor, 'remote_raw', raw)
    supervisor.cancel('123')
    released = [e for e in supervisor.state['events'] if e['kind'] == 'allocation_released']
    assert len(released) == 1 and released[0]['job'] == '123'


def test_pending_job_can_be_stopped_without_starting_container(supervisor, monkeypatch):
    import deploy.benchmark.recover_public_run as module
    released = []
    def submit():
        supervisor.state['jobs'].append('123')
        return '123'
    def pause():
        supervisor.stopping = True
    monkeypatch.setattr(supervisor, 'submit', submit)
    monkeypatch.setattr(supervisor, 'docker_state', lambda: None)
    monkeypatch.setattr(supervisor, 'job', lambda *_: ('PENDING', 0))
    monkeypatch.setattr(supervisor, 'pause', pause)
    monkeypatch.setattr(supervisor, 'start_or_resume', lambda: pytest.fail('must not start before model readiness'))
    def cancel(job):
        supervisor.released.add(job)
        released.append(job)
    monkeypatch.setattr(supervisor, 'cancel', cancel)
    monkeypatch.setattr(module.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(module.subprocess, 'run', lambda *_a, **_k: None)
    supervisor.run()
    assert supervisor.state['status'] == 'stopped_by_request'
    assert released == ['123']

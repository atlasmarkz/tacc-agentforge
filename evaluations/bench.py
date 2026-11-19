"""One-command entry point for the v1 benchmark.

    python -m evaluations.bench check
    python -m evaluations.bench run --split dev --repeats 1 --out /tmp/v1-dev
    python -m evaluations.bench report --input evaluations/results/v1-replay-smoke-2026-11-15

The runner itself lives in `evaluations.v1.run`; this module adds a small
preflight, endpoint check, resume-friendly output directory and the proposal
style Markdown report.  It uses only the standard library plus the project's
agent dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evaluations.v1.report import write_report
from evaluations.v1.suite import git_state

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evaluations/v1/data'
CONFIGS = ['B0', 'B1', 'B2', 'B3', 'A1', 'A2']
DEFAULT_MANIFEST = ROOT / 'evaluations/v1/agent-adapter.example.json'
DEFAULT_ADAPTER = 'evaluations.v1.agent_adapter:respond'


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _endpoint(args) -> tuple[str | None, str]:
    address = args.address_file or os.environ.get('BENCHMARK_ADDRESS_FILE')
    if address:
        path = Path(address).expanduser()
        try:
            text = path.read_text(encoding='utf-8').strip()
        except OSError as exc:
            return None, f'无法读取 address file {path}: {exc}'
        return (text.rstrip('/') or None), f'address file {path}'
    value = args.base_url or os.environ.get('BENCHMARK_BASE_URL')
    return (value.rstrip('/') if value else None), 'BENCHMARK_BASE_URL'


def _probe(url: str, timeout: float = 5.0) -> tuple[bool, str, list[str]]:
    """GET an OpenAI-compatible /models endpoint; never sends a generation request."""
    candidates = [url + '/models'] if url.rstrip('/').endswith('/v1') else [url + '/v1/models', url + '/models']
    errors = []
    for candidate in candidates:
        try:
            request = Request(candidate, headers={'Accept': 'application/json'})
            with urlopen(request, timeout=timeout) as response:     # noqa: S310 - user-supplied endpoint
                if response.status != 200:
                    errors.append(f'{candidate}: HTTP {response.status}')
                    continue
                data = json.loads(response.read().decode('utf-8'))
            models = [m.get('id') for m in data.get('data', []) if isinstance(m, dict)]
            if not models:
                errors.append(f'{candidate}: /models 返回空列表')
                continue
            return True, f'{candidate} -> {", ".join(str(m) for m in models[:5])}', [str(m) for m in models]
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            errors.append(f'{candidate}: {exc}')
    return False, '; '.join(errors[-2:]), []


def _suite_state(data: Path, split: str) -> tuple[list[str], list[str], dict]:
    blockers, warnings = [], []
    manifest_path = data / 'manifest.json'
    if not manifest_path.is_file():
        return [f'缺少 {manifest_path}'], warnings, {}
    manifest = _read_json(manifest_path)
    for name, expected in (manifest.get('hashes') or {}).items():
        path = data / name
        if not path.is_file():
            blockers.append(f'缺少数据文件 {path}')
        elif _digest(path) != expected:
            blockers.append(f'{name} 的哈希与 manifest 不一致')
    cases_path = data / 'cases.jsonl'
    if not cases_path.is_file():
        return blockers, warnings, manifest
    cases = _read_jsonl(cases_path)
    if split == 'heldout':
        release_path, freeze_path = data / 'release.json', data / 'freeze.json'
        if not release_path.is_file() or not freeze_path.is_file():
            blockers.append('heldout 仍是 sealed 状态：先运行 `python -m evaluations.v1.suite freeze`、'
                            '`release-fifth`、`verify`')
        else:
            release, freeze = _read_json(release_path), _read_json(freeze_path)
            if release.get('manifest_sha256') != _digest(manifest_path):
                blockers.append('release.json 与当前 manifest 不一致（数据在 release 后改过）')
            if freeze.get('code') != _fingerprint():
                blockers.append('代码在 freeze 后改过：release 失效，需要重新冻结')
        if manifest.get('heldout') != 40:
            blockers.append(f"heldout={manifest.get('heldout')}，释放第五类后应为 40")
    else:
        if manifest.get('pending'):
            warnings.append(f"当前 dev 可用；heldout 还有 {manifest['pending']} 题待释放")
    return blockers, warnings, manifest


def _fingerprint() -> dict:
    files = sorted(p for folder in ('agent', 'budget', 'inference', 'knowledge', 'configs', 'evaluations')
                   for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.sql')
                   and not any(x in p.parts for x in ('results', '__pycache__', '.venv')))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _manifest_check(path: Path, configs: list[str]) -> tuple[list[str], dict]:
    blockers = []
    if not path.is_file():
        return [f'缺少 adapter manifest {path}'], {}
    try:
        caps = _read_json(path)
    except (OSError, ValueError) as exc:
        return [f'adapter manifest 无法解析: {exc}'], {}
    if not caps.get('identity'):
        blockers.append('adapter manifest 缺少 identity')
    if not caps.get('cache_policy'):
        blockers.append('adapter manifest 缺少 cache_policy')
    missing = ({'B3', 'A1'} & set(configs)) - set(caps.get('validator_configs', []))
    if missing:
        blockers.append(f"配置 {', '.join(sorted(missing))} 需要真实校验器，adapter manifest 的 validator_configs 未声明")
    return blockers, caps


def _dependency_check() -> list[str]:
    problems = []
    if sys.version_info < (3, 10):
        problems.append(f'需要 Python >= 3.10，当前 {sys.version.split()[0]}')
    for module in ('pydantic', 'langgraph'):
        if importlib.util.find_spec(module) is None:
            problems.append(f'缺少依赖 {module}；按 README 安装 `python -m pip install -r agent/requirements.lock`')
    return problems


def preflight(args, require_endpoint: bool = False):
    blockers, warnings = [], []
    data = Path(args.data)
    blockers, warnings, manifest = _suite_state(data, args.split)
    git = git_state()
    if git.get('dirty'):
        warnings.append('工作区有未提交改动；manifest 会记录 git dirty=True')
    configs = [x.strip() for x in args.configs.split(',') if x.strip()]
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        blockers.append(f'未知配置: {", ".join(sorted(unknown))}')
    if args.repeats < 1:
        blockers.append('--repeats 必须 >= 1')
    if args.only and args.split != 'dev':
        blockers.append('--only 只能用于 dev')
    cases = _read_jsonl(data / 'cases.jsonl') if (data / 'cases.jsonl').is_file() else []
    selected = [c for c in cases if c.get('split') == args.split]
    if args.only:
        ids = {x.strip() for x in args.only.split(',') if x.strip()}
        unknown_ids = ids - {c['case_id'] for c in selected}
        if unknown_ids:
            blockers.append(f'未知 dev case: {", ".join(sorted(unknown_ids))}')
        selected = [c for c in selected if c['case_id'] in ids]
    if not selected:
        blockers.append(f'{args.split} split 没有可用用例')
    manifest_blockers, caps = _manifest_check(Path(args.adapter_manifest), configs)
    blockers += manifest_blockers
    blockers += _dependency_check()
    info = {'cases': len(selected), 'configs': configs, 'attempts': len(selected) * len(configs) * args.repeats,
            'manifest': manifest, 'caps': caps}

    endpoint, source = _endpoint(args)
    info['endpoint'] = endpoint
    requested_model = args.model or os.environ.get('BENCHMARK_MODEL') or 'qwen3.8-27b'
    info['requested_model'] = requested_model
    if endpoint:
        ok, detail, models = _probe(endpoint)
        info['endpoint_detail'] = detail
        info['available_models'] = models
        if not ok:
            message = f'模型端点不可用（{source}）：{detail}'
            (blockers if require_endpoint else warnings).append(message)
        elif requested_model not in models:
            message = (f'端点没有请求的模型 `{requested_model}`（可用: '
                       f'{", ".join(models[:5]) or "无"}）')
            blockers.append(message)
    else:
        message = '没有配置模型端点（BENCHMARK_BASE_URL / BENCHMARK_ADDRESS_FILE）'
        (blockers if require_endpoint else warnings).append(message)

    if getattr(args, 'out', None):
        out = Path(args.out)
        if args.resume:
            if not (out / 'manifest.json').is_file():
                blockers.append(f'--resume 需要已有 manifest: {out}/manifest.json')
        elif out.exists():
            blockers.append(f'输出目录已存在: {out}；换一个目录或加 --resume')
    return blockers, warnings, info


def _print_preflight(args, blockers, warnings, info):
    print('Benchmark preflight')
    if info:
        manifest = info.get('manifest') or {}
        dataset = manifest.get('status', '?')
        print(f"  data: {Path(args.data)}  (status={dataset}, dev={manifest.get('dev', '?')}, "
              f"heldout={manifest.get('heldout', '?')}, pending={manifest.get('pending', '?')})")
        print(f"  split={args.split}  cases={info['cases']}  configs={len(info['configs'])}  "
              f"repeats={args.repeats}  attempts={info['attempts']}")
        caps = info.get('caps') or {}
        identity = caps.get('identity')
        if isinstance(identity, dict):
            print(f"  adapter: {identity.get('adapter', '?')}  model={identity.get('model', '?')}")
        elif identity:
            print(f"  adapter: {identity}")
        print(f"  endpoint: {info.get('endpoint_detail') or ('未配置' if not info.get('endpoint') else info.get('endpoint'))}")
    for item in blockers:
        print(f'  [fail] {item}')
    for item in warnings:
        print(f'  [warn] {item}')
    print('  RESULT: ' + ('BLOCKED' if blockers else 'READY'))
    return 1 if blockers else 0


def _common(parser):
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--split', choices=['dev', 'heldout'], default='dev')
    parser.add_argument('--configs', default=','.join(CONFIGS))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--only', default='')
    parser.add_argument('--adapter', default=DEFAULT_ADAPTER)
    parser.add_argument('--adapter-manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--address-file', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--service-info', default=None)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', action='store_true')


def _run_command(args) -> int:
    if args.split == 'heldout' and not args.unseal_heldout:
        print('heldout 是 sealed 的：release 后仍需显式加 --unseal-heldout')
        return 2
    blockers, warnings, info = preflight(args, require_endpoint=not args.skip_endpoint_check)
    status = _print_preflight(args, blockers, warnings, info)
    if blockers:
        return status
    env = os.environ.copy()
    if args.base_url:
        env['BENCHMARK_BASE_URL'] = args.base_url
    if args.address_file:
        env['BENCHMARK_ADDRESS_FILE'] = str(Path(args.address_file).expanduser())
    if args.model:
        env['BENCHMARK_MODEL'] = args.model
    if args.service_info:
        env['BENCHMARK_SERVICE_INFO'] = str(Path(args.service_info).expanduser())
    command = [sys.executable, '-m', 'evaluations.v1.run', '--data', str(Path(args.data)),
               '--split', args.split, '--configs', args.configs, '--repeats', str(args.repeats),
               '--timeout', str(args.timeout), '--adapter', args.adapter,
               '--adapter-manifest', str(Path(args.adapter_manifest)), '--out', str(Path(args.out))]
    if args.only:
        command += ['--only', args.only]
    if args.unseal_heldout:
        command.append('--unseal-heldout')
    if args.record_traces:
        command.append('--record-traces')
    if args.resume:
        command.append('--resume')
    print('Running: ' + ' '.join(command))
    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f'Benchmark 运行失败（exit {exc.returncode}）。')
        if args.out and (Path(args.out) / 'records.jsonl').is_file():
            print(f'可修复后继续: python -m evaluations.bench run --resume --out {args.out}（其余参数不变）')
        return exc.returncode
    destination = write_report(Path(args.out))
    print(f'Report: {destination}')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m evaluations.bench', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    check = sub.add_parser('check', help='检查数据、冻结状态、adapter、依赖和模型端点')
    _common(check)
    check.add_argument('--unseal-heldout', action='store_true')
    check.add_argument('--strict-endpoint', action='store_true', help='没有配置端点也算 blocker')

    run = sub.add_parser('run', help='运行 benchmark 并自动生成 REPORT.md')
    _common(run)
    run.add_argument('--unseal-heldout', action='store_true')
    run.add_argument('--record-traces', action='store_true', help='仅 dev；原始 prompt/reply 会落盘')
    run.add_argument('--skip-endpoint-check', action='store_true', help='自定义 adapter 不需要模型时使用')

    report = sub.add_parser('report', help='从已有运行目录生成 proposal 风格 REPORT.md')
    report.add_argument('--input', type=Path, required=True)
    report.add_argument('--out', type=Path)
    report.add_argument('--review', type=Path, help='evaluations.review score 生成的 review-report.json')

    args = parser.parse_args(argv)
    if args.command == 'check':
        blockers, warnings, info = preflight(args, require_endpoint=args.strict_endpoint)
        return _print_preflight(args, blockers, warnings, info)
    if args.command == 'run':
        if not args.out:
            parser.error('run 需要 --out')
        return _run_command(args)
    destination = write_report(args.input, args.out, args.review)
    print(f'Report: {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

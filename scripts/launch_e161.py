"""Supervise the frozen E161 producer and analysis within the allocated project."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import psutil


PROJECT = Path('/data/robot/p001')


def checked(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT) or PROJECT.resolve() != PROJECT:
        raise ValueError(f'path is outside the allocated project: {path}')
    return resolved


def write_json(path: Path, value: dict) -> None:
    checked(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def children(pid: int) -> list[int]:
    """Read only the direct-child handles of a process launched by this job."""
    try:
        return [int(value) for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []


def usage(kind: int) -> float:
    value = resource.getrusage(kind)
    return value.ru_utime + value.ru_stime


def verify_tree(root: Path) -> None:
    """Reject links before any downstream project file traversal."""
    checked(root)
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f'project tree contains a symlink: {path}')
            checked(path)


def verify_source(source: Path, config: dict) -> None:
    verify_tree(source)
    manifest = json.loads((source / 'bundle-manifest.json').read_text())
    names = {row['path'] for row in manifest['files']}
    required = {'configs/e161.json', 'configs/c1.json', 'data/foundation.pt',
                'data/pretrain_source.npz', 'scripts/launch_e161.py'}
    if not required.issubset(names):
        raise ValueError('frozen manifest is missing a required execution input')
    for row in manifest['files']:
        path = checked(source / row['path'])
        if not path.is_relative_to(source):
            raise ValueError('source manifest escaped its frozen directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'frozen source changed: {path}')
    # The producer obtains both immutable inputs from the original C1 config.
    base = json.loads((source / 'configs' / 'c1.json').read_text())
    for key in ('foundation', 'dataset'):
        path = checked(source / base[key])
        if not path.is_relative_to(source):
            raise ValueError(f'input escaped frozen source: {key}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', default='configs/e161.json')
    parser.add_argument('--cohort', type=Path, help='read-only existing producer for an analysis repair')
    parser.add_argument('--cpu-seconds', type=float, default=3600)
    parser.add_argument('--wall-seconds', type=float, default=2700)
    args = parser.parse_args()
    source, run = checked(args.source), checked(args.run)
    if os.getuid() == 0:
        raise RuntimeError('run only as the allocated unprivileged account')
    if run.exists():
        raise FileExistsError(f'preserve existing attempt: {run}')
    if args.cpu_seconds > 3600 or args.wall_seconds > 2700:
        raise ValueError('E161 resource limits may not be expanded')
    run.mkdir(parents=True)
    logs = run / 'logs'
    logs.mkdir()
    config_path = checked(source / args.config)
    if config_path != source / 'configs' / 'e161.json':
        raise ValueError('only the manifested frozen E161 config may be executed')
    config = json.loads(config_path.read_text())
    expected = config['expected_main']
    if config.get('seed_status') != 'frozen' or expected['total_steps'] != 2244608:
        raise ValueError('production requires the complete frozen E161 configuration')
    verify_source(source, config)
    (run / 'retention_candidate_audit').mkdir()
    cohort = checked(args.cohort) if args.cohort else run / 'producer'
    artifact_root = cohort.parent
    if args.cohort:
        verify_tree(artifact_root)
        if not (cohort / 'summary.json').is_file():
            raise ValueError('analysis repair requires a completed producer summary')
    started = time.monotonic()
    self_start = usage(resource.RUSAGE_SELF)
    state = {'state': 'RUNNING', 'pid': os.getpid(), 'source': str(source),
             'run': str(run), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'limits': {'cpu_seconds': args.cpu_seconds, 'wall_seconds': args.wall_seconds,
                        'environment_interactions': 2500000}, 'phases': [], 'active_job': None}
    state_path = run / 'state.json'
    commands = [
        ('producer', [sys.executable, 'scripts/run_retention_candidate.py', '--config',
                      str(config_path), '--output', str(run / 'producer')]),
        ('audit', [sys.executable, 'scripts/audit_retention_candidate.py', '--root',
                   str(cohort), '--report',
                   str(artifact_root / 'retention_candidate_audit' /
                       ('main002.json' if args.cohort else 'main.json'))]),
        ('analysis', [sys.executable, 'scripts/report_retention_candidate.py', '--input',
                      str(cohort), '--output', str(artifact_root / 'analysis001')]),
    ]
    if args.cohort:
        commands = commands[1:]
    completed_cpu = 0.0
    for name, command in commands:
        verify_source(source, config)
        verify_tree(run)
        if args.cohort:
            verify_tree(artifact_root)
        if (completed_cpu + usage(resource.RUSAGE_SELF)-self_start >= args.cpu_seconds - 15
                or time.monotonic()-started >= args.wall_seconds - 15):
            state['state'] = 'LIMIT_BEFORE_PHASE'
            break
        known: dict[int, dict] = {}
        stopped = None
        phase_started = time.monotonic()
        phase_child_start = usage(resource.RUSAGE_CHILDREN)
        phase_cpu = 0.0
        with (logs / f'{name}.txt').open('xb') as output:
            process = subprocess.Popen(command, cwd=source, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            phase = {'name': name, 'command': command, 'cwd': str(source),
                     'pid': process.pid, 'process_group': process.pid, 'state': 'RUNNING'}
            state['phases'].append(phase)
            state['active_job'] = phase
            while process.poll() is None:
                pending = [process.pid]
                visited = set()
                while pending:
                    pid = pending.pop()
                    if pid in visited:
                        continue
                    visited.add(pid)
                    try:
                        handle = psutil.Process(pid)
                        identity = handle.create_time()
                        previous = known.get(pid)
                        if previous and previous['create_time'] != identity:
                            raise RuntimeError('a recorded project PID was reused')
                        cpu = handle.cpu_times()
                        known[pid] = {'pid': pid, 'create_time': identity,
                                      'cpu_seconds': max(previous['cpu_seconds'] if previous else 0,
                                                         cpu.user + cpu.system)}
                        pending.extend(children(pid))
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
                phase_cpu = max(phase_cpu, sum(value['cpu_seconds'] for value in known.values()))
                current_cpu = completed_cpu + usage(resource.RUSAGE_SELF) - self_start + phase_cpu
                wall = time.monotonic() - started
                state.update(cpu_seconds_observed=current_cpu, wall_seconds=wall,
                             project_job_handles=list(known.values()))
                if current_cpu >= args.cpu_seconds - 15:
                    stopped = 'CPU_LIMIT'
                if wall >= args.wall_seconds - 15:
                    stopped = 'WALL_LIMIT'
                write_json(state_path, state)
                if stopped:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    break
                time.sleep(1)
        phase.update(returncode=process.returncode, wall_seconds=time.monotonic()-phase_started,
                     state='STOPPED' if stopped else ('COMPLETED' if process.returncode == 0 else 'FAILED'))
        phase_cpu = max(phase_cpu, usage(resource.RUSAGE_CHILDREN) - phase_child_start)
        completed_cpu += phase_cpu
        phase['cpu_seconds'] = phase_cpu
        state.update(active_job=None, cpu_seconds=completed_cpu + usage(resource.RUSAGE_SELF)-self_start,
                     wall_seconds=time.monotonic()-started)
        write_json(state_path, state)
        if stopped or process.returncode:
            state['state'] = stopped or f'{name.upper()}_FAILED'
            break
    else:
        state['state'] = 'COMPLETED'
    state['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    verify_source(source, config)
    verify_tree(run)
    write_json(state_path, state)
    print(json.dumps(state, indent=2))
    return 0 if state['state'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())

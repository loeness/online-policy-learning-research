"""Single-worker local E162 execution with process-handle CPU accounting."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
SOURCE = ROOT / 'research/appendix/H033/delivery_v1/source_snapshot'


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_cpu(process):
    """Windows retains exact kernel/user CPU on our own Popen process handle."""
    values = [wintypes.FILETIME() for _ in range(4)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    call = kernel.GetProcessTimes
    call.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    call.restype = wintypes.BOOL
    if not call(wintypes.HANDLE(int(process._handle)), *(ctypes.byref(v) for v in values)):
        raise ctypes.WinError(ctypes.get_last_error())
    return sum(((v.dwHighDateTime << 32) | v.dwLowDateTime) for v in values[2:]) / 1e7


def bytes_used():
    total = 0
    for directory, dirs, files in os.walk(E, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            if p.is_symlink() or not p.resolve().is_relative_to(E):
                raise ValueError('E162 output tree contains a symlink or escape')
        total += sum((Path(directory)/name).stat().st_size for name in files)
    return total


def verify(admission):
    for row in admission['inputs_and_sources']:
        path = Path(row['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'input path escape: {path}')
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'frozen source or input changed: {path}')
    freeze = json.loads((E/'preparation/execution_freeze.json').read_text())
    for name, digest in freeze['files_sha256'].items():
        if hashlib.sha256((E/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'frozen execution file changed: {name}')


def main():
    if os.name != 'nt':
        raise RuntimeError('this admitted executor uses Windows process handles')
    admission = json.loads((E/'preparation/admission.json').read_text())
    verify(admission)
    budget_path = E/'budget.json'
    budget = json.loads(budget_path.read_text())
    if budget.get('active_job') or budget['spent']['cpu_seconds']:
        raise RuntimeError('first campaign requires an unused reservation')
    limits = admission['limits']
    run = E/'runs/run001'
    if run.exists():
        raise FileExistsError('preserve any previous run attempt')
    run.mkdir(parents=True)
    logs = run/'logs'; logs.mkdir()
    env = dict(os.environ)
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'TEMP': str(E/'tmp'), 'TMP': str(E/'tmp'), 'TMPDIR': str(E/'tmp'),
                'MPLCONFIGDIR': str(E/'cache/mpl'), 'XDG_CACHE_HOME': str(E/'cache/xdg'),
                'TORCH_HOME': str(E/'cache/torch')})
    cli = SOURCE/'scripts/h033_diagnostic.py'
    producer = ROOT/'experiments/exp161/runs/run001/producer'
    old_analysis = ROOT/'experiments/exp161/analyses/analysis001/analysis.json'
    stats = E/'preparation/statistics.py'
    commands = [
        ('features', [sys.executable, '-B', str(cli), 'features', '--producer', str(producer),
                      '--output', str(run/'features.json')]),
        ('outcomes', [sys.executable, '-B', str(cli), 'outcomes', '--analysis', str(old_analysis),
                      '--output', str(run/'outcomes.json')]),
        ('crossfit', [sys.executable, '-B', str(cli), 'crossfit', '--features', str(run/'features.json'),
                      '--outcomes', str(run/'outcomes.json'), '--output', str(run/'crossfit.json')]),
        ('statistics', [sys.executable, '-B', str(stats), '--features', str(run/'features.json'),
                        '--outcomes', str(run/'outcomes.json'), '--crossfit', str(run/'crossfit.json'),
                        '--output', str(E/'analyses/analysis001')]),
    ]
    state = {'state': 'RUNNING', 'experiment_id': 'E162', 'supervisor_pid': os.getpid(),
             'started_utc': utc(), 'phases': [], 'active_job': None,
             'environment_interactions': 0, 'limits': limits,
             'environment_overrides': {k: v for k, v in env.items() if k in
                 {'TEMP','TMP','TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','TORCH_HOME',
                  'PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'}},
             'home_unchanged': True}
    cpu_total = 0.0
    started, self_cpu = time.perf_counter(), time.process_time()
    try:
        for name, command in commands:
            phase_start = time.perf_counter()
            with (logs/f'{name}.txt').open('xb') as log:
                p = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
                phase = {'name': name, 'pid': p.pid, 'command': command, 'cwd': str(SOURCE)}
                state['phases'].append(phase)
                state['active_job'] = {'pid': p.pid, 'phase': name}
                while True:
                    code = p.poll()
                    cpu = cpu_total + process_cpu(p) + time.process_time()-self_cpu
                    wall = time.perf_counter()-started
                    size = bytes_used()
                    state.update(cpu_seconds=cpu, wall_seconds=wall, artifact_cache_temp_bytes=size)
                    write(run/'state.json', state)
                    budget.update(state='RUNNING', active_job=state['active_job'])
                    budget['spent'] = {'environment_interactions': 0, 'cpu_seconds': cpu,
                                       'campaign_wall_seconds': wall, 'artifact_cache_temp_bytes': size,
                                       'measurement_status': 'RUNNING_WINDOWS_PROCESS_HANDLE'}
                    budget['reservations'][0]['state'] = 'RUNNING'
                    write(budget_path, budget)
                    if cpu >= limits['cpu_seconds']-1 or wall >= limits['campaign_wall_seconds']-1 or size >= limits['new_artifact_cache_temp_bytes']:
                        if code is None:
                            p.terminate()
                            p.wait(timeout=5)
                        raise RuntimeError('original E162 resource limit reached')
                    if code is not None:
                        break
                    time.sleep(.05)
                phase.update(cpu_seconds=process_cpu(p), wall_seconds=time.perf_counter()-phase_start,
                             returncode=code)
                cpu_total += phase['cpu_seconds']
                print(f"{name}: exit={code}, CPU={phase['cpu_seconds']:.6f}s", flush=True)
                if code:
                    raise RuntimeError(f'{name} failed; preserve its log and partial output')
            verify(admission)
        state['state'] = 'COMPLETED'
    except BaseException:
        if state['active_job'] and 'cpu_seconds' not in state['phases'][-1]:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
            phase.update(cpu_seconds=process_cpu(p), returncode=p.returncode,
                         wall_seconds=time.perf_counter()-phase_start)
            cpu_total += phase['cpu_seconds']
        state['state'] = 'FAILED'
        state['error'] = traceback.format_exc()
        print(state['error'], file=sys.stderr, flush=True)
    state.update(active_job=None, finished_utc=utc(), cpu_seconds=cpu_total+time.process_time()-self_cpu,
                 wall_seconds=time.perf_counter()-started, artifact_cache_temp_bytes=bytes_used())
    write(run/'state.json', state)
    write(run/'provenance.json', {
        'git_commit': admission['git_commit'], 'method_source_identity': admission['source_identity_sha256'],
        'admission_sha256': hashlib.sha256((E/'preparation/admission.json').read_bytes()).hexdigest(),
        'execution_freeze': '../../preparation/execution_freeze.json', 'execution': state,
        'inputs_and_sources': admission['inputs_and_sources'],
        'data_paths': ['features.json', 'outcomes.json', 'crossfit.json'],
        'analysis_paths': ['../../analyses/analysis001/metrics.json', '../../analyses/analysis001/report.md'],
        'figure_paths': {'state': 'NOT_APPLICABLE', 'reason': 'full seed and fold tables suffice'},
    })
    budget.update(state=state['state'], active_job=None)
    budget['spent'] = {'environment_interactions': 0, 'cpu_seconds': state['cpu_seconds'],
                       'campaign_wall_seconds': state['wall_seconds'],
                       'artifact_cache_temp_bytes': bytes_used(),
                       'measurement_status': 'WINDOWS_KERNEL_USER_PROCESS_HANDLES_PLUS_SUPERVISOR_SELF'}
    budget['remaining'] = {'cpu_seconds': limits['cpu_seconds']-state['cpu_seconds'],
                           'campaign_wall_seconds': limits['campaign_wall_seconds']-state['wall_seconds'],
                           'new_artifact_cache_temp_bytes': limits['new_artifact_cache_temp_bytes']-bytes_used(),
                           'environment_interactions': 0}
    budget['reservations'][0]['state'] = 'SETTLED'
    budget['next_decision'] = 'Return E162 evidence to coordinator; no automatic followup.'
    write(budget_path, budget)
    print(json.dumps({'state': state['state'], 'cpu_seconds': state['cpu_seconds'],
                      'wall_seconds': state['wall_seconds']}), flush=True)
    return 0 if state['state'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())

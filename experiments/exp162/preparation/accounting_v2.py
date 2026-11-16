"""CPU-accounting repair proof; never retroactively measures the E162 run.

Windows venv redirectors can launch another interpreter. An outer runpy wrapper
records that interpreter's full lifetime CPU, including imports, at exit; the
launcher handle and supervising process remain separately measurable.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WRAPPER = r'''
import atexit, json, os, runpy, sys, time
from pathlib import Path
record, target, *arguments = sys.argv[1:]
def capture():
    Path(record).write_text(json.dumps({'pid':os.getpid(), 'full_interpreter_cpu_seconds':time.process_time()})+'\n')
atexit.register(capture)
sys.argv=[target]+arguments
runpy.run_path(target, run_name='__main__')
'''


def launcher_cpu(process):
    values = [wintypes.FILETIME() for _ in range(4)]
    call = ctypes.WinDLL('kernel32', use_last_error=True).GetProcessTimes
    call.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    call.restype = wintypes.BOOL
    if not call(wintypes.HANDLE(int(process._handle)), *(ctypes.byref(v) for v in values)):
        raise ctypes.WinError(ctypes.get_last_error())
    return sum((v.dwHighDateTime<<32)|v.dwLowDateTime for v in values[2:])/1e7


def main():
    e = Path(__file__).resolve().parents[1]
    folder = e/'preparation/accounting_check001'
    folder.mkdir()
    target = folder/'dummy.py'
    target.write_text('import time\ns=time.process_time()\nwhile time.process_time()-s<.15: pass\n')
    record = folder/'interpreter.json'
    environment = dict(os.environ)
    environment.update({'TEMP':str(e/'tmp'), 'TMP':str(e/'tmp'),
                        'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
                        'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    before, wall = time.process_time(), time.perf_counter()
    p = subprocess.Popen([sys.executable,'-B','-c',WRAPPER,str(record),str(target)],
                         creationflags=subprocess.CREATE_NO_WINDOW, env=environment)
    p.wait(timeout=10)
    inner = json.loads(record.read_text())
    shim = launcher_cpu(p)
    distinct = inner['pid'] != p.pid
    total = inner['full_interpreter_cpu_seconds']+(shim if distinct else 0)
    if p.returncode or total < .14:
        raise RuntimeError('synthetic CPU-accounting repair check failed')
    proof = {'kind':'ZERO_MODEL_SOFTWARE_ACCOUNTING_REPAIR',
             'used_on_original_run':False,'real_feature_or_crossfit_reruns':0,
             'launcher_pid':p.pid,'interpreter_pid':inner['pid'],
             'distinct_venv_redirector':distinct,'launcher_cpu_seconds':shim,
             'full_interpreter_cpu_seconds':inner['full_interpreter_cpu_seconds'],
             'complete_child_cpu_seconds':total,
             'supervisor_check_cpu_seconds':time.process_time()-before,
             'wall_seconds':time.perf_counter()-wall,
             'passed':True,'environment_interactions':0,
             'limitation':'Repair verified on synthetic single-interpreter software only; '
                          'no recovered exact original-run CPU is asserted.'}
    (e/'preparation/accounting_repair_check.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()

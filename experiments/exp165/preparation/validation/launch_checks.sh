set -eu
. /data/robot/p001/enter.sh
export OMP_NUM_THREADS='1'
export MKL_NUM_THREADS='1'
export OPENBLAS_NUM_THREADS='1'
export NUMEXPR_NUM_THREADS='1'
export PYTHONDONTWRITEBYTECODE='1'
export CUDA_VISIBLE_DEVICES=''
export TMPDIR='/data/robot/p001/e165/tmp'
export TMP='/data/robot/p001/e165/tmp'
export TEMP='/data/robot/p001/e165/tmp'
export XDG_CACHE_HOME='/data/robot/p001/e165/cache'
export XDG_CONFIG_HOME='/data/robot/p001/e165/config'
export XDG_DATA_HOME='/data/robot/p001/e165/state'
export XDG_STATE_HOME='/data/robot/p001/e165/state'
export MPLCONFIGDIR='/data/robot/p001/e165/cache/mpl'
export TORCH_HOME='/data/robot/p001/e165/cache/torch'
export TORCH_EXTENSIONS_DIR='/data/robot/p001/e165/cache/ext'
cd '/data/robot/p001/e165/s001'
E165_LAUNCH_PARENT_PID=$$ '/data/robot/p001/env/bin/python' '-I' '-B' '/data/robot/p001/e165/o001/supervisor.py' '--stage' 'CHECKS' '--spec' '/data/robot/p001/e165/o001/stage_checks.json' '--authorization' '/data/robot/p001/e165/o001/authorization_checks.json' &
childpid=$!
printf 'launch_parent_pid=%s\nlaunch_child_pid=%s\n' "$$" "$childpid"
if wait "$childpid"; then
    rc=0
else
    rc=$?
fi
printf 'supervisor_exit_code=%s\n' "$rc"
exit "$rc"

# E163 Linux supervisor metering design

Preparation only; this design and its implementation have not been executed.
The supervisor may start probes or a worker only after the coordinator's
authorization JSON binds the exact `run_spec.json` bytes and repeats the fixed
limits with `execution_authorized: true` and a nonempty coordinator directive.
An absent, false, or mismatched authorization exits before creating `run001` or
performing a metering precheck.

## Frozen command and scope

The command accepts only the canonical E163 paths:

```text
--spec /data/robot/p001/e163/o001/run_spec.json
--authorization /data/robot/p001/e163/o001/authorization.json
```

It requires `/data/robot/p001/env/bin/python` to be both the supervisor's
resolved interpreter and the worker interpreter. The Python environment path
may resolve through an internal project symlink; it must resolve within
`/data/robot/p001`. The `o001`, `s001`, and `run001` paths are fixed. The
worker argv is exactly `-B o001/runner.py --spec <spec> --authorization
<authorization>`. The supervisor itself must run as isolated, bytecode-disabled
Python on Linux. It passes a small runtime environment to the worker, omitting
SSH-agent and credential variables, preserving the inherited `HOME`, applying
the fixed one-thread values, and disabling visible GPU devices. Temp and cache
directories are created below `e163`.

The accepted limits are 120 aggregate CPU seconds, 180 campaign wall seconds,
one scientific worker, one numeric thread, zero GPU devices, zero environment
interactions, 32 candidate fits, 268,435,456 apparent bytes under `e163`, and
100,000,000,000 aggregate project bytes. `project_bytes_before` is the measured
non-E163 project baseline supplied by the run spec. Capacity accounting adds
every regular-file byte under `e163`, including source, archives, caches,
temporary files, logs, and outputs. It rejects symlinks, non-regular files, and
hard links in that tree. It does not walk outside `e163`.

## Campaign start and CPU accounting

The first process start timestamp comes from the supervisor's own
`/proc/self/stat` field 22 and the kernel clock tick rate. The wall origin is
reconstructed from `/proc/uptime`, so it includes supervisor interpreter
startup and all subsequent probes, imports, prefix checks, fits, reporting, and
finalization. This boundary starts at the supervisor process; no shell or SSH
launcher accounting is available or claimed.

Aggregate CPU is the supervisor's `getrusage(RUSAGE_SELF)` CPU plus CPU from
completed direct children returned by `wait4`, plus current CPU for recorded
live PIDs. Live PID samples read only `/proc/<recorded-pid>/stat`; each sample
checks the recorded start tick before using `utime`, `stime`, `cutime`, and
`cstime`. A missing or changed identity fails closed. No process list, unrelated
PID, command line, or other user's job is inspected. Every child launch records
its PID, process-group ID, purpose, and start tick. The final child audit
compares the sum of recorded wait4 CPU with `RUSAGE_CHILDREN`.

Before any real fit, the campaign runs two short checks under the same clocks
and budgets:

1. A CPU-bound child forks a CPU-bound grandchild, waits for it, reports both
   usage records, and exits. The supervisor compares that reported total with
   its own `wait4` result and the cumulative `RUSAGE_CHILDREN` value. A mismatch,
   a missing report, or an unexpectedly small CPU sample prevents worker start.
2. A sleeping child runs in its recorded process group. The supervisor verifies
   its start identity, sends `SIGTERM` to that group, and verifies termination
   with `wait4`. If the grace period expires, it sends `SIGKILL` only to that
   recorded group. A failed stop check prevents worker start.

The single worker is created with `fork` and replaced directly by `execve` of
the allocated Python interpreter; no shell or launcher process sits between the
supervisor and the scientific interpreter. Before `execve`, the child receives
an `RLIMIT_CPU` hard and soft ceiling set to the floored CPU remaining before
the 115-second safety stop. The worker's live `/proc` sample and final `wait4`
record cover its interpreter startup through process exit. The scientific
runner is a single process and does not intentionally spawn subprocesses; the
recorded worker process group is also the only signal target for a worker stop.

## Stops, evidence, and storage

The supervisor polls every 25 milliseconds and stops the worker at 115 observed
aggregate CPU seconds or 170 wall seconds, leaving the reserved 5 CPU seconds
and 10 wall seconds for signal handling, wait4 collection, and final evidence.
It also sets an individual-file `RLIMIT_FSIZE` no larger than the remaining E163
byte allowance. It samples aggregate `e163` bytes every 250 milliseconds and
stops on a project-cap or E163-cap breach. A 65,536-byte tail is kept available
for the terminal evidence. The kernel does not provide an aggregate per-tree
quota here, so the aggregate byte guard is sampled rather than atomic; the
per-file limit and frequent monitoring reduce overshoot risk but cannot prove
that a fast series of multiple writes never crosses the aggregate cap between
samples. If that limitation is unacceptable for admission, do not run the
campaign without an allocated project quota or equivalent enforced cap.

All supervisor evidence is created exclusively below `e163/run001`; existing
campaign paths are rejected, never overwritten. `meter.jsonl` records start,
limits, live samples, capacity samples, precheck results, signals, and final
status. `wait4.jsonl` retains raw child usage fields and wait statuses. The
worker retains its own detailed scientific outputs. `state.json` records the
last observed complete CPU and wall totals, all wait4 records, the storage
count, the wait4 audit, the explicit footer reserve, and conservative campaign
upper bounds equal to the pre-footer CPU and wall samples plus the full reserved
tail. A nominally complete result is downgraded if either upper bound exceeds
120 CPU seconds or 180 wall seconds. The footer's own tiny write tail is not
observed after process exit; it is bounded through that reserve because a
process cannot report its final post-exit usage from within itself.

The supervisor never retries the scientific worker. Any precheck failure,
accounting mismatch, identity failure, capacity failure, worker error, or
budget stop leaves the partial evidence in place and returns an incomplete or
invalid state for coordinator review.

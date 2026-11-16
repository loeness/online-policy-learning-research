# E163 run001 — ACCOUNTING_INVALID before scientific worker

The single approved launch stopped at the first CPU-descendant accounting
precheck. It raised `RUSAGE_CHILDREN and recorded wait4 accounting differ`;
the final audit also failed. No scientific worker, prefix deserialization,
candidate fit, encoded score or environment interaction occurred. The stop
precheck did not start. This is an execution failure, not `NO_ADVANCEMENT`.

The three remote files in `remote/run001/` are byte-identical mirrors verified
against the remote size/SHA-256 inventory in `mirror_manifest.json`. They retain
the original status and counters unchanged. `provenance.json` links the frozen
source, orchestration, directive, authorization, handles and launch evidence.
Recorded child handles were reaped; SSH sessions are closed.

Complete campaign CPU and wall cost are **UNKNOWN** under the failed audit.
The known probe wait4 CPU is 0.122070 seconds. Raw recorded CPU subtotal is
0.268236 seconds; raw process-age wall is67.527558 seconds. These cannot be
silently relabeled as exact campaign measurements. The final raw nominal
upper figures include5CPU/10wall reserves; they are preserved rather than
treated as independently validated accounting. See
`../../analyses/a001/cost_accounting.json`.

The frozen command uses shell `exec`: the supervisor PID's kernel start age
already includes roughly67seconds before its first campaign event. Inherited
pre-start shell child accounting is the leading explanation for the audit
mismatch. No corrective probe or retry was run. Both the frozen implementation
and this failed attempt remain intact; the coordinator owns the next decision.

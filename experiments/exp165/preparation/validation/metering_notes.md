# E165 staged metering adapter

Status: implementation and static review only. No supervisor, worker, research
module, audit, runtime probe, SSH command, transfer, or experiment was run.
The CHECKS control schema is frozen, so `OUTPUT_CONTRACT_FROZEN` is `True`.
Each stage still refuses to start without its exact authorized spec, its
stage-specific coordinator directive, required inputs and worker. This flag
does not grant resources or authorize a later phase.

## Stage layout and limits

| Stage | Output root | Spec / authorization | Worker |
| --- | --- | --- | --- |
| CHECKS | `/data/robot/p001/e165/c001` | `o001/stage_checks.json` / `o001/authorization_checks.json` | `checks_entry.py` |
| DIAGNOSTIC | `/data/robot/p001/e165/run001` | `o001/stage_diagnostic.json` / `o001/authorization_diagnostic.json` | `runner.py` |
| AUDIT | `/data/robot/p001/e165/a001` | `o001/stage_audit.json` / `o001/authorization_audit.json` | `audit_entry.py` |

Each stage is a fresh supervisor process. The worker receives `E165_STAGE` for
the exact stage alongside the supervised marker. Its observed lifetime includes Python
startup, its two metering prechecks, its stage worker, stop/reap, and final
accounting. The per-stage limits are CHECKS 20 CPU / 60 wall seconds,
DIAGNOSTIC 80 / 140, and AUDIT 20 / 40. Each stage stops at its hard cap minus
5 CPU / 10 wall seconds and reserves 65,536 bytes for its footer. Stages cannot
borrow one another's allocations. The total upper-bound ledger is checked
against 120 CPU / 240 controlled wall seconds.

The run spec must bind `experiment_id`, `stage`, `project_root`, `e_root`,
`source_cwd`, `orchestration_root`, `run_root`, `python`, exact `limits`, exact
`round_limits`, `project_bytes_before`, `source_identity_sha256`,
`source_files`, `orchestration_files`, `prior_stage_states`, `environment`, and
the exact `worker_argv`. Manifest entries use absolute remote paths and exact
`bytes` / `sha256`; source entries stay under `e165/s001`, orchestration entries
under `e165/o001`. Prior-state descriptors are keyed by the exact predecessor
stage and bind `path`, `bytes`, and `sha256`.

Before sampling `/proc`, `RUSAGE_SELF`, `RUSAGE_CHILDREN`, monotonic time, or
kernel ticks, the supervisor validates the stage-specific spec and
authorization, including its exact spec-byte SHA256, stage, one-attempt
reservation ID, coordinator directive, resource/execution flags, and stage
limits. It then verifies the allowlisted file manifests and predecessor state
descriptors. The predecessor states must be COMPLETE, match the same source
identity, and reconcile their observed and reserved costs without exceeding
their own stage caps.

The supervisor retains the E164 accounting fields used by its independent
auditor: fresh process-birth start source, startup proof with
`rusage_children_post_authorization_pre_fork`, RUSAGE_CHILDREN versus `wait4`
reconciliation, direct-child `wait4` rows, live owned-process CPU snapshots,
stage observed and bounded CPU / wall values, safety stops, footer reserve,
and an empty final child-handle list. The descendant precheck verifies waited
descendant CPU accounting. The owned process-group stop precheck must reap with
exit code `-15` and raw wait status `15`. The worker runs in its own recorded
process group; the supervisor reads only its recorded PID and process-group
identity, never process arguments or unrelated user files.

## Worker completion structure

All workers write `state_worker.json` with format `e165_worker_state_v1`, their
stage, COMPLETE status, frozen source identity, and exact zero values for
environment interactions, GPU devices, real candidate fits, and candidate
recovery calls.

CHECKS compares the nonempty spec `expected_test_nodeids` with the exact worker
`executed_test_nodeids` list and requires `failed`, `error`, and `skipped` to be
zero. DIAGNOSTIC requires all 16 logical prefixes, at most 336 full-training
oracle evaluations and 33 new complete score packages, one confirmed anchor,
a matching completed-score list length, and a nonempty cohort lock identity.
AUDIT requires `audit_result.status == COMPLETE_VALID`, a saved `audit.json`
with the same status, and `remote_statistics_completed == true`. These are
structural checks only; the supervisor does not recompute scientific metrics.

The final AUDIT `state.json` stores the predecessor observed / upper costs,
its own stage costs, and the round ledger. It writes no separate budget file.
All new outputs, logs, caches, and temporary files remain under `e165`; the
stage capacity check includes the 128 MiB new artifact/cache/temp cap and the
100,000,000,000-byte project cap using the non-E165 `project_bytes_before`
baseline.

## Static validation boundary

Static validation may parse supervisor source with AST and compile the source
string without executing it; JSON and text-manifest checks may compare declared
limits, stage paths, and hashes. The Python supervisor, shell launchers, workers,
scientific modules, pytest, synthetic checks, audits, and metering probes must
not run on this workstation. Remote stage execution still requires the
coordinator's separate exact stage authorization.

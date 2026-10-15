# Shared online writes and bounded local reuse

Date: 2026-10-16. Branch: `codex/c1-strong-baselines`.

The extension archives S1a true online shared parameters, S1b family-free
probe grouping under supplied routes, S2 twelve contexts competing for six
local slots, and S3 missing-action/ambiguity and reward-conflict diagnostics.
It includes pilot and formal streams, source snapshots, checkpoints, binary
actor packages, protocols, figures, tests and positive/negative/null results.

The eight interaction-generating cohorts contain 2,846,720 training-stream
steps, 61,440 uniform probe steps and 728,210 evaluation steps: 3,636,370 total.
Offline replay and report construction add zero interactions. Unit-test
environment fixtures are outside this cohort ledger. All deployments use one
shared W0. The work remains controlled C1 diagnostics, with no standard-benchmark
SOTA or completed unknown-boundary knowledge organizer claim.

Formal raw-log audits pass 369 checks; pilots pass another 120. The targeted
implementation tests comprise 80 policy/storage/grouping/runner/cache/conflict/
support tests and one report test. The eight cohorts' independent cost ledger
is `outputs/shared_write_archive_audit/cohort_ledger.json`.

`scripts/audit_shared_write_archive.py` verifies all 12,625 older data/output
files against the manifest at `5979956485113ef1907675baf37137d4cb2b1604` and
reconciles all new cohort and audit totals. No older experimental file changed.
The current whole-archive byte inventory is `archive/manifest.json`.

The research summary is `outputs/shared_write_archive_audit/研究推进总结.md`.
The policy and structural limitations are in
`docs/SHARED_WRITE_STRUCTURE_AND_LIMITS.md`; individual result reports are indexed
in `docs/EXPERIMENT_INDEX.md`. All original confidential third-party extractions
remain excluded under `tmp/` and are not uploaded.

The 256 KiB cap describes W0 plus one agent-state checkpoint, not the archive
or runtime RAM. Periodic checkpoints omit running environment and schedule
state and do not support exact trajectory continuation. Source versions used
at execution remain inside each run, even when later reporting code changed.

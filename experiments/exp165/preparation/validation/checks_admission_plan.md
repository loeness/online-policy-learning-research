# E165 CHECKS admission plan

Status: preparation only. The candidate CHECKS package binds the frozen
scientific source, nine test nodes and five control files. Independent static
review remains pending. Neither remote access nor a stage execution is
authorized by this document.

The CHECKS transfer is limited to the frozen scientific snapshot and its
manifest, plus `supervisor.py`, `phase_common.py`, `checks_entry.py`,
`launch_checks.sh` and the dedicated `pytest.ini`. A machine-readable exact
transfer list must bind each local source to one destination under
`/data/robot/p001/e165`, its byte length and SHA-256. No machine-local
instruction file, credentials, Git data or unrelated project history is
included. DIAGNOSTIC and AUDIT code are not prerequisites for this phase.

The candidate list contains 42 files totaling 1,202,701 bytes: the 36-file
scientific snapshot, its manifest and five CHECKS controls. The scientific
source identity is
`4e96df16ca207ea961bafaed6b92dfbee07d7cdd55dc7cf986b5fb2b48189f0b`.
The candidate transfer manifest SHA-256 is
`1076fadb7ac7503d5fc0dcc21be49c57ee17cce83fb86d989bab677507d58e2b`.
The candidate stage spec remains non-executable: its resource flag is false
and its project baseline has not been measured. The exact dynamic spec and its
hash must be admitted separately after authorized capacity checks.

Before any transfer, the coordinator must review the exact list and issue an
explicit preparation directive. Then only the allocated project path and the
minimum destination filesystem capacity/inode metadata may be checked. The
100,000,000,000-byte aggregate project cap and 134,217,728-byte E165 new
artifact/cache/temp cap both apply. The pre-E165 project byte baseline,
transfer size, temporary copies, test output, retained evidence, later scoring
packages and headroom must be included in the peak calculation. Free space
alone is not the project's allowance. A future stage specification must record
the measured baseline and the identity of the admitted bytes.

For planning only, reserve up to 2 MiB for this CHECKS transfer and metadata,
32 MiB for CHECKS output/temp, 48 MiB for diagnostic inputs/output/cache,
16 MiB for audit output/temp, and 30 MiB as shared headroom. These allowances
sum to 128 MiB; they are not measurements or independent per-stage grants.
The 33 complete score-package ceiling and 262,144-byte package limit alone
allow approximately 8.25 MiB; traces, metadata, logs and temporary writes are
additional. Admission must compare current project use, current filesystem
free bytes/inodes and the full 128 MiB allowance before the first transfer;
the supervisor then monitors the aggregate E165 cap. If it cannot fit, stop
without deleting existing evidence.

CHECKS is one controlled attempt with a 20 CPU-second and 60 wall-second limit,
including a 5 CPU-second and 10 wall-second footer reserve. The supervisor
stops work at 15 CPU seconds or 50 wall seconds. A dedicated coordinator
directive bound to the final stage-spec byte hash is required before launch.
There is no automatic retry. All test work runs remotely as one CPU worker
with one numerical thread, zero GPU devices and no new environment interaction.
Actual synthetic helper/scoring calls and mocks are recorded separately from
the 16-seed scientific-oracle and score-package limits. A missing count is
UNKNOWN, not zero.

Run results and measurement records remain under `e165/c001`; the exact
return list and SHA-256 reconciliation are determined after execution. Passing
CHECKS alone does not authorize DIAGNOSTIC or AUDIT. The algorithm owner must
review the receipt and the coordinator must separately admit each next phase.

# H037 v2 independent static review

Status: BLOCKED FOR REMOTE EXECUTION. This is a static review only. No SSH,
research implementation, pytest collection, synthetic check, or experiment
was run. The user's reported original p002 upload is preserved. The main
coordinator and algorithm owner have not been messaged by this review.

## Reviewed identity and change

- Uploaded draft: `p001.zip`, SHA-256
  `2c35fd2a7fd2ed16f1b3c0d2c430346ffacdf537f8e7f166518e97c085ae9960`.
  The user reported successful unpacking and checksum verification; current
  remote state and capacity have not been independently observed.
- Algorithm owner delivery: `handoff/h037_exploration_v2/h037_exploration_v2.zip`,
  SHA-256 `8fa20263f90f7ba83ae80205ed39ebef0791500c6ecce0a4dcfaea2aaa0993ea`;
  source identity `7fbcfe43b62b17a5009f4e36972e5748c87c4c9cd250b5396d7a85d26dccabfa`.
  Its 60 payload entries match the local delivery manifest by size and hash,
  and the matching current local files match that manifest. This is not a
  remote acceptance, runtime dependency check, or scientific validation.
- Changed relative to the uploaded draft: `scripts/run_h037_initial.py`,
  `src/online_rl/h037_candidate.py`, and `src/online_rl/h037_integration.py`.
  New entries: `configs/h037_exploration_v1.json`,
  `research/appendix/H037/EXPLORATORY_DELIVERY_V1.md`, and
  `tests/test_h037_transactions.py`. No v2 files have been reported uploaded.

## Execution blockers requiring an algorithm-side repair

1. The user requires stopping on an exception. `propose_h037` currently
   catches every ordinary `Exception` and returns `accepted: false`
   (`src/online_rl/h037_candidate.py:673`). `pre_reset` classifies that as an
   ordinary candidate rejection (`src/online_rl/h037_integration.py:964`).
   Thus fitting, policy-forward or codec faults can be indistinguishable from
   an expected gate rejection. Introduce explicit, auditable distinction
   between expected scientific rejection and unexpected execution errors.
2. Even if the controller returns `status: error`, `_prepare_task` logs it
   and then resets and proceeds (`scripts/run_h037_initial.py:193`). The
   test `test_prepare_task_runs_handoff_once_before_exactly_one_reset`
   currently expects this continuation even for `error`
   (`tests/test_h037_transactions.py:456`). Preserve the error and rollback
   evidence, stop the experiment before subsequent environment steps, and
   revise the associated test. Preserve appropriate normal gate-rejection
   semantics.
3. The configured deployment uses int8. The current synthetic fixtures and
   candidate transaction encoding use `float32`
   (`tests/test_h037_initial.py:44`, `tests/test_h037_transactions.py:200`).
   A narrowly scoped real int8 encode/decode test must check the actual
   candidate and no-write comparison used for acceptance; do not infer it
   from the float32 mock or the current-policy teacher test.
4. Add a direct synthetic assertion that the ordered train/holdout split
   excludes later rows from training and that stale-parent-version rows are
   excluded. The current tests demonstrate ring order and version recording,
   but do not directly exercise these selection boundaries.
5. The common-prefix synthetic restore uses an empty H037 evidence cache.
   Add nonempty cache content to the paired restoration test and check both
   arms restore its rows, identity, ordering and version exactly.
6. Distinguish storage/inode admission failure from a candidate learner
   package exceeding its 2 MiB cap. The latter is currently a `skip` and the
   runner continues (`src/online_rl/h037_integration.py:837`,
   `scripts/run_h037_initial.py:193`). Until the user specifies a narrower
   meaning of 'stop on insufficient capacity', apply it conservatively to
   both cases: retain the candidate rollback and capacity event, then stop
   the campaign before more environment interaction.

## Checks already defined, not yet executed

- Victim routes and current synthetic policy probabilities have checks;
  train/holdout and stale-version selection need the direct assertions above.
  The 17 node IDs in the delivery are inferred statically.
- Successful commit, ordinary reject, package rejection and exceptions have
  controller-level tests, using candidate/core substitutes. They are not a
  demonstration of the actual optimized int8 candidate committing.
- Both branches restore the H037 evidence controller and the learner from a
  common prefix. The synthetic fork test uses an empty H037 cache and a
  simulator substitute; actual server-side pre-branch equality checks must
  still pass and be recorded.
- Both branches incur the same cache. The comparison is the effect of
  enabling handoff under that common cache, not a net deployment comparison
  against a cache-free LRU. One seed provides exploratory evidence only.

## Resource and staging status

The delivered proposal fixes seed 27101 and thresholds 0.0001/0.01/0.01,
with a 2 MiB learner package and 3,584 planned environment interactions.
The algorithm owner suggests CHECKS 120 CPU/180 wall seconds, RUN 900/1200,
AUDIT 120/180, and 256 MiB peak new output/cache/temp. These are proposals in
the delivered document, not evidence of accepted coordinator stage budgets.
The last accessible coordinator record predates its review of this delivery;
do not infer acceptance from the existence of the proposal. User-provided
approval information and a recorded exact stage directive must be reconciled.

The runner's internal CPU/time limits begin after imports and do not alone
cover startup, children, output/cache/temp, or full campaign finalization.
A reviewed independent supervisor and exact phase-specific identities/limits
remain required. CHECKS must complete without new environment interactions;
no RUN follows failed checks, capacity failure, errors, unknown accounting,
or unapproved stage scope. No automatic retry or later seed is admitted.

Once repaired and accepted, generate a new immutable transfer allowlist and
package with byte/SHA identities, choose a fresh neutral project subdirectory,
and retain the existing `p002` and its original package unchanged. Measure
aggregate project use, destination free space and inode capacity before
transfer; do not access other accounts or read unrelated model payloads.
The user performs all server commands and relays results manually.

# Frozen score feature benchmark

This checkpoint-only diagnostic compares the preserved score function with a paired-feature evaluator. It did not construct or step an environment, and did not train or modify an agent.
Timing context: clean checkpoint-only run; root confirmed no concurrent RL process

- Cases: 32 storage decision prefixes; scored group rows: 64 (old and new groups per case).
- Repetitions: 25 after one warm-up per method and score case; torch threads: 1.
- CPU timer: GetProcessTimes() (reported resolution 1e-07 s); per-call CPU values may be quantized, so interpret aggregate CPU means with wall timings.
- Equivalence: PASS; max probability error: 0; max gain error: 0; max lower error: 0.
- Mean CPU per score: reference 0.005723 s; paired 0.006152 s; paired saving -0.000430 s.
- Estimated saving per full case using the actual 2 score calls: -0.000859 s (-0.0090% of mean prefix CPU and -0.0011% of mean full case CPU). These are scaled estimates from scorer timing, not an end-to-end cohort speedup measurement.
- Paired feature workspace: 152576 bytes per score (mean); added persistent bytes: 0 bytes.

## Timing setup

- Runner import: 0.551 wall s / 0.531 CPU s.
- Checkpoint/model restore: 2.028 wall s / 1.984 CPU s across 32 cases.
- Scorer construction mean per case: reference 0.000003 wall s; paired 0.000001 wall s.
- Timed scorer totals over 25 rounds: reference 9.156 CPU s; paired 9.844 CPU s.

## Correctness

- Probability, gain/lower, support, stored score, and selector checks passed: True.
- Probability max absolute error: 0; gain max absolute error: 0; lower max absolute error: 0.
- Support-flag mismatches: 0; stored decision-score mismatches: 0; selector mismatches: 0; timed-output mismatches: 0.

## Per-score cases

| Case | Group | Rows | Reference CPU (ms) | Paired CPU (ms) | Saved CPU (ms) | Max probability error |
|---|---:|---:|---:|---:|---:|---:|
| s20101_dynamics_new_major | 0 | 256 | 3.125 | 7.500 | -4.375 | 0 |
| s20101_dynamics_new_major | 1 | 256 | 5.625 | 6.250 | -0.625 | 0 |
| s20101_dynamics_old_major | 0 | 256 | 5.625 | 5.000 | 0.625 | 0 |
| s20101_dynamics_old_major | 1 | 256 | 6.250 | 6.250 | 0.000 | 0 |
| s20101_reward_conflict_new_major | 0 | 256 | 7.500 | 5.000 | 2.500 | 0 |
| s20101_reward_conflict_new_major | 1 | 256 | 5.000 | 5.625 | -0.625 | 0 |
| s20101_reward_conflict_old_major | 0 | 256 | 8.125 | 6.875 | 1.250 | 0 |
| s20101_reward_conflict_old_major | 1 | 256 | 6.250 | 8.125 | -1.875 | 0 |
| s20107_dynamics_new_major | 0 | 256 | 4.375 | 6.250 | -1.875 | 0 |
| s20107_dynamics_new_major | 1 | 256 | 8.750 | 4.375 | 4.375 | 0 |
| s20107_dynamics_old_major | 0 | 256 | 6.875 | 5.000 | 1.875 | 0 |
| s20107_dynamics_old_major | 1 | 256 | 3.750 | 5.625 | -1.875 | 0 |
| s20107_reward_conflict_new_major | 0 | 256 | 4.375 | 6.250 | -1.875 | 0 |
| s20107_reward_conflict_new_major | 1 | 256 | 7.500 | 5.625 | 1.875 | 0 |
| s20107_reward_conflict_old_major | 0 | 256 | 4.375 | 7.500 | -3.125 | 0 |
| s20107_reward_conflict_old_major | 1 | 256 | 9.375 | 5.000 | 4.375 | 0 |
| s20113_dynamics_new_major | 0 | 256 | 7.500 | 5.625 | 1.875 | 0 |
| s20113_dynamics_new_major | 1 | 256 | 4.375 | 5.000 | -0.625 | 0 |
| s20113_dynamics_old_major | 0 | 256 | 7.500 | 6.250 | 1.250 | 0 |
| s20113_dynamics_old_major | 1 | 256 | 5.000 | 6.250 | -1.250 | 0 |
| s20113_reward_conflict_new_major | 0 | 256 | 6.875 | 6.250 | 0.625 | 0 |
| s20113_reward_conflict_new_major | 1 | 256 | 5.625 | 6.250 | -0.625 | 0 |
| s20113_reward_conflict_old_major | 0 | 256 | 6.875 | 5.000 | 1.875 | 0 |
| s20113_reward_conflict_old_major | 1 | 256 | 4.375 | 6.250 | -1.875 | 0 |
| s20117_dynamics_new_major | 0 | 256 | 4.375 | 7.500 | -3.125 | 0 |
| s20117_dynamics_new_major | 1 | 256 | 5.625 | 6.250 | -0.625 | 0 |
| s20117_dynamics_old_major | 0 | 256 | 5.000 | 6.250 | -1.250 | 0 |
| s20117_dynamics_old_major | 1 | 256 | 5.625 | 4.375 | 1.250 | 0 |
| s20117_reward_conflict_new_major | 0 | 256 | 9.375 | 5.000 | 4.375 | 0 |
| s20117_reward_conflict_new_major | 1 | 256 | 6.250 | 6.250 | 0.000 | 0 |
| s20117_reward_conflict_old_major | 0 | 256 | 6.250 | 6.875 | -0.625 | 0 |
| s20117_reward_conflict_old_major | 1 | 256 | 4.375 | 8.750 | -4.375 | 0 |
| s20123_dynamics_new_major | 0 | 256 | 6.250 | 8.125 | -1.875 | 0 |
| s20123_dynamics_new_major | 1 | 256 | 3.750 | 6.250 | -2.500 | 0 |
| s20123_dynamics_old_major | 0 | 256 | 6.250 | 8.125 | -1.875 | 0 |
| s20123_dynamics_old_major | 1 | 256 | 6.875 | 6.250 | 0.625 | 0 |
| s20123_reward_conflict_new_major | 0 | 256 | 6.875 | 6.875 | 0.000 | 0 |
| s20123_reward_conflict_new_major | 1 | 256 | 7.500 | 5.625 | 1.875 | 0 |
| s20123_reward_conflict_old_major | 0 | 256 | 6.250 | 3.125 | 3.125 | 0 |
| s20123_reward_conflict_old_major | 1 | 256 | 7.500 | 4.375 | 3.125 | 0 |
| s20129_dynamics_new_major | 0 | 256 | 6.875 | 6.250 | 0.625 | 0 |
| s20129_dynamics_new_major | 1 | 256 | 6.250 | 6.250 | 0.000 | 0 |
| s20129_dynamics_old_major | 0 | 256 | 3.750 | 6.250 | -2.500 | 0 |
| s20129_dynamics_old_major | 1 | 256 | 5.625 | 6.875 | -1.250 | 0 |
| s20129_reward_conflict_new_major | 0 | 256 | 4.375 | 8.125 | -3.750 | 0 |
| s20129_reward_conflict_new_major | 1 | 256 | 5.000 | 6.250 | -1.250 | 0 |
| s20129_reward_conflict_old_major | 0 | 256 | 4.375 | 6.875 | -2.500 | 0 |
| s20129_reward_conflict_old_major | 1 | 256 | 3.750 | 6.875 | -3.125 | 0 |
| s20143_dynamics_new_major | 0 | 256 | 6.875 | 4.375 | 2.500 | 0 |
| s20143_dynamics_new_major | 1 | 256 | 3.750 | 7.500 | -3.750 | 0 |
| s20143_dynamics_old_major | 0 | 256 | 6.875 | 5.000 | 1.875 | 0 |
| s20143_dynamics_old_major | 1 | 256 | 6.875 | 4.375 | 2.500 | 0 |
| s20143_reward_conflict_new_major | 0 | 256 | 4.375 | 7.500 | -3.125 | 0 |
| s20143_reward_conflict_new_major | 1 | 256 | 2.500 | 5.625 | -3.125 | 0 |
| s20143_reward_conflict_old_major | 0 | 256 | 6.875 | 4.375 | 2.500 | 0 |
| s20143_reward_conflict_old_major | 1 | 256 | 3.750 | 7.500 | -3.750 | 0 |
| s20149_dynamics_new_major | 0 | 256 | 5.000 | 6.875 | -1.875 | 0 |
| s20149_dynamics_new_major | 1 | 256 | 5.625 | 5.000 | 0.625 | 0 |
| s20149_dynamics_old_major | 0 | 256 | 6.250 | 7.500 | -1.250 | 0 |
| s20149_dynamics_old_major | 1 | 256 | 5.625 | 5.625 | 0.000 | 0 |
| s20149_reward_conflict_new_major | 0 | 256 | 6.250 | 6.250 | 0.000 | 0 |
| s20149_reward_conflict_new_major | 1 | 256 | 3.750 | 6.875 | -3.125 | 0 |
| s20149_reward_conflict_old_major | 0 | 256 | 2.500 | 6.875 | -4.375 | 0 |
| s20149_reward_conflict_old_major | 1 | 256 | 6.250 | 5.625 | 0.625 | 0 |

## Memory and scope

- Mean / min / max temporary feature tensor bytes per score: 152576 / 152576 / 152576.
- Added persistent cache state: 0 bytes. No cache survives a score call and no learner/checkpoint state is changed.
- This is common-subexpression reuse within the same candidate/baseline score. It is not a SparseForest local-structure result and does not support a core-contribution claim.
- The comparison holds the frozen checkpoint, evidence arrays, candidate, baseline, and surrogate_gain logic fixed. Timing covers scorer calls only; setup and model restore are reported separately.

## Mismatches

None.

Full hashes, per-case records, timing order, and detailed metrics are in report.json.

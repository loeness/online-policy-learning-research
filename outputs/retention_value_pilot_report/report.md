# Selected-Feedback Retention Stream Report

Calibration status: **PILOT · descriptive only**.

Pilot results are plumbing and accounting observations; they do not support a confirmation conclusion.

## Calibration

No complete-cohort bootstrap estimate is reported. Per-stream observed scores, failures, and costs remain in `analysis.json` and `per_seed.csv`.

Only the selected action's pre-action prediction is scored against `feedback.utility_target`. Unchosen actions have no observed label and are not scored.

Comparison status: **PILOT · descriptive only**.

Protocol audit: root summary and frozen-start protocol hashes do not match; treat this input as incomplete.

Pilot results are not a confirmation conclusion.

## Repeated-retention comparison

Whole-stream return/step includes warmup and all outcome blocks. Postwarmup return/step uses outcome blocks only. The primary contrast is learned_linear − learned_intercept within seed and relation; its four-seed intervals are descriptive. The lower confidence bound is checked separately against zero and the frozen practical threshold.

No paired bootstrap interval is reported because the input is a pilot or does not cover the complete frozen cohort. Failed and missing streams remain listed with their recorded costs.

Learned_linear and learned_intercept share the first 16 uniform warmup decisions and then use the frozen epsilon-greedy schedule. Random stays uniform. Deterministic keep/replace/mean/frequency/lower controls have different exploration costs.

All methods versus every control are included in `analysis.json`; stream and seed-level values are in `per_seed.csv`.

## Cost, retention, and scope

### calibration

| Quantity | Total / maximum |
|---|---:|
| Actual raw step rows | 9216 |
| Reported environment steps | 9216 |
| PPO rollouts | 36 |
| Minibatch updates | 576 |
| Campaign elapsed wall seconds | unknown |
| Campaign total CPU seconds | unknown |
| Campaign child CPU seconds | unknown |
| Campaign orchestrator CPU seconds | unknown |
| Sum of per-stream CPU seconds | 12.516 |
| Sum of per-stream wall seconds (not elapsed) | 12.744 |
| CPU seconds charged to utility | 8.125 |
| Model update CPU seconds | 0.000 |
| Parent writes | 1 |
| Retention operations | 4 |
| Workers actually used | unknown |
| Peak package bytes | 238466 / cap 262144 |
| Maximum model state bytes | 1732 |

### comparison

| Quantity | Total / maximum |
|---|---:|
| Actual raw step rows | 92160 |
| Reported environment steps | 92160 |
| PPO rollouts | 360 |
| Minibatch updates | 5760 |
| Campaign elapsed wall seconds | unknown |
| Campaign total CPU seconds | unknown |
| Campaign child CPU seconds | unknown |
| Campaign orchestrator CPU seconds | unknown |
| Sum of per-stream CPU seconds | 124.531 |
| Sum of per-stream wall seconds (not elapsed) | 127.753 |
| CPU seconds charged to utility | 81.594 |
| Model update CPU seconds | 0.031 |
| Parent writes | 18 |
| Retention operations | 32 |
| Workers actually used | unknown |
| Peak package bytes | 238722 / cap 262144 |
| Maximum model state bytes | 1732 |

Comparison recurrence metrics are factual natural-recurrence first-256 reward rates, including after prior natural eviction. They are reported with bout counts and are not causal recovery loss estimates.

Campaign elapsed and total CPU values are shown only when the root summary logs `cohort_costs`; legacy per-stream wall totals are labeled as sums and are not campaign elapsed time.

This is selected-feedback utility prediction and a four-seed exploratory policy comparison. It does not observe unchosen-action returns, identify per-write causal benefit, or establish novelty of the ridge/discounting component.

Bootstrap settings: 10000 resamples, seed 20261019; intervals are descriptive. Report generation itself runs no RL and adds no environment steps.

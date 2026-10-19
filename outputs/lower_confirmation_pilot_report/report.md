# Surrogate-Lower Confirmation Report

**Status: `incomplete_pilot`**

This is an incomplete pilot with history condition(s) old_major. It is not confirmation evidence; no confirmation interval is calculated.

## Primary contrast

The unit is the seed: within each seed, average the two matched history=future conditions, then average the 16 seeds equally. The bootstrap resamples eight seeds within old-family A and eight within B. Ties remain in the denominator.

No primary confirmation estimate is available for this incomplete input.

## Original practical threshold

The frozen threshold remains **0.002000 reward/step** and is assessed separately from direction. Status: **not_evaluated_incomplete_input**.

## Secondary contrasts

Each contrast is `surrogate_lower` minus the named comparator. `frequency` uses the frozen prefix-only frequency choice. All scored values come from the already-paid physical action branches; branch scores are observer-only and do not update the learner.

| Comparator | Mean reward/step | Stratified 95% CI | Positive / negative / tie seeds |
|---|---:|---:|---:|
| replace | N/A | N/A | N/A |
| mean | N/A | N/A | N/A |
| frequency | N/A | N/A | N/A |
| none | N/A | N/A | N/A |

## Primary contrast by history and old family

These are descriptive paired seed means; the overall confirmation interval is stratified by old family.

| Group | Seeds | Mean lower − keep | Positive / negative / tie |
|---|---:|---:|---:|
| old_major | 1 | 0.000000 | 0 / 0 / 1 of 1 |
| new_major | 0 | N/A | 0 / 0 / 0 of 0 |
| old family A | 0 | N/A | 0 / 0 / 0 of 0 |
| old family B | 0 | N/A | 0 / 0 / 0 of 0 |

## Cost and raw-row accounting

| Quantity | Total |
|---|---:|
| Environment steps from summary | 15360.000 |
| Prefix environment steps | 3072.000 |
| Branch environment steps | 12288.000 |
| Actual persisted step rows | 15360.000 |
| CPU seconds | 22.547 |
| Wall seconds | 25.551 |

Actual raw-row count complete: **True**. Raw rows are summed from each case's saved `prefix.raw_environment_steps` and branch `raw_environment_steps` fields.

## Reproducibility and scoring

Bootstrap: 10000 stratified seed resamples, RNG seed 20261019, percentile interval. Input phase: `pilot`; observed histories: old_major.

The report reads saved summaries only. It runs no RL, environment steps, or new branch evaluations.

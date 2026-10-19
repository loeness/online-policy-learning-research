# Surrogate-Lower Confirmation Report

**Status: `complete_confirmation`**

This report evaluates the frozen, matched-future confirmation conditions.

## Primary contrast

The unit is the seed: within each seed, average the two matched history=future conditions, then average the 16 seeds equally. The bootstrap resamples eight seeds within old-family A and eight within B. Ties remain in the denominator.

Mean `surrogate_lower − keep`: **0.000050 reward/step**; stratified percentile 95% CI [-0.000515, 0.000527].
Directional result: **uncertain**. Positive / negative / tie seeds: 3 / 1 / 12 of 16.

## Original practical threshold

The frozen threshold remains **0.002000 reward/step** and is assessed separately from direction. Status: **does_not_support_practical_threshold**.
Primary mean: 0.000050; 95% CI: [-0.000515, 0.000527].

## Secondary contrasts

Each contrast is `surrogate_lower` minus the named comparator. `frequency` uses the frozen prefix-only frequency choice. All scored values come from the already-paid physical action branches; branch scores are observer-only and do not update the learner.

| Comparator | Mean reward/step | Stratified 95% CI | Positive / negative / tie seeds |
|---|---:|---:|---:|
| replace | 0.001963 | [0.001034, 0.002915] | 12 / 4 / 0 of 16 |
| mean | 0.002189 | [0.001303, 0.003045] | 14 / 2 / 0 of 16 |
| frequency | -0.000180 | [-0.000722, 0.000333] | 5 / 8 / 3 of 16 |
| none | 0.003019 | [0.002104, 0.003941] | 15 / 1 / 0 of 16 |

## Primary contrast by history and old family

These are descriptive paired seed means; the overall confirmation interval is stratified by old family.

| Group | Seeds | Mean lower − keep | Positive / negative / tie |
|---|---:|---:|---:|
| old_major | 16 | -0.000448 | 0 / 1 / 15 of 16 |
| new_major | 16 | 0.000547 | 3 / 0 / 13 of 16 |
| old family A | 8 | 0.000547 | 3 / 0 / 5 of 8 |
| old family B | 8 | -0.000448 | 0 / 1 / 7 of 8 |

## Cost and raw-row accounting

| Quantity | Total |
|---|---:|
| Environment steps from summary | 1966080.000 |
| Prefix environment steps | 393216.000 |
| Branch environment steps | 1572864.000 |
| Actual persisted step rows | 1966080.000 |
| CPU seconds | 2495.281 |
| Wall seconds | 877.576 |

Actual raw-row count complete: **True**. Raw rows are summed from each case's saved `prefix.raw_environment_steps` and branch `raw_environment_steps` fields.

## Reproducibility and scoring

Bootstrap: 10000 stratified seed resamples, RNG seed 20261019, percentile interval. Input phase: `main`; observed histories: new_major, old_major.

The report reads saved summaries only. It runs no RL, environment steps, or new branch evaluations.

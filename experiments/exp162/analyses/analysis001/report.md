# E162 independent statistics report

Status: **NO_ADVANCEMENT**.

This is a read-only audit of the frozen 16-seed, four-fold H033 v1 output. The script independently recomputed primary arm means, the fixed standardized ridge OOF predictions, actions, selected values, and aggregates from the supplied packets, then compared them with the crossfit packet.

## Provenance

Input packet hashes and all frozen H033 delivery snapshot hashes, including this statistics source hash, are recorded in `metrics.json` under `provenance.file_hashes` and `provenance.frozen_delivery_snapshot_files`.

| Input | SHA256 |
| --- | --- |
| features (`D:\MovedFromC\ChatGPT\online\experiments\exp162\runs\run001\features.json`) | `2859c9f9f93c2bf27956b7366e0591f413d92a5cf59d3cf7427090a4bf3e3ba5` |
| outcomes (`D:\MovedFromC\ChatGPT\online\experiments\exp162\runs\run001\outcomes.json`) | `36df20b169a9116646d990757283762182c6dd9ab2712a5cec2d14b61f4ae0e5` |
| crossfit (`D:\MovedFromC\ChatGPT\online\experiments\exp162\runs\run001\crossfit.json`) | `6ea38fed9f0630a823275c41c621f75b8eb02c794d2b720004f90d7d27f42fbb` |

The independent recomputation matched 565 reported fields; maximum absolute numeric difference was 5.55e-17 (tolerance: relative and absolute 1e-10).

## Policy results

Values are reward per step averaged over the 16 seed-level selections. Balanced is secondary and scores the same decisions.

| Policy | Keep | Replace | Primary value | Primary gain vs keep | Balanced value | Balanced gain vs keep |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crossfit | 6 | 10 | -0.00610283 | 0.00203323 | -0.00640864 | 0.00034332 |
| fixed_keep | 16 | 0 | -0.00813606 | 0.00000000 | -0.00675196 | 0.00000000 |
| fixed_replace | 0 | 16 | -0.00683525 | 0.00130081 | -0.00788747 | -0.00113551 |
| intercept_only | 0 | 16 | -0.00683525 | 0.00130081 | -0.00788747 | -0.00113551 |
| frequency_rule | 8 | 8 | -0.00682254 | 0.00131353 | -0.00639720 | 0.00035477 |

### Paired primary value differences

| Control | G_b = mean(ridge − control) |
| --- | ---: |
| fixed_keep | 0.00203323 |
| fixed_replace | 0.00073242 |
| intercept_only | 0.00073242 |
| frequency_rule | 0.00071971 |

G_min = **0.00071971** against `frequency_rule`. The fixed screen requires G_min ≥ 0.002 and at least 3 of 4 fold worst-case means > 0.

| Fold | Test seeds | Fold worst-case min_b mean(d) | Positive |
| ---: | --- | ---: | --- |
| 0 | 26201, 26202, 26204, 26207 | -0.00089518 | False |
| 1 | 26203, 26206, 26208, 26209 | -0.00101217 | False |
| 2 | 26205, 26210, 26211, 26216 | 0.00000000 | False |
| 3 | 26212, 26213, 26215, 26218 | 0.00211843 | True |

## Frozen ridge folds and feature ranges

Feature ranges cover all 16 input rows. Each fold's training-only means, scales, coefficients, intercept, and seed membership are fully listed in `metrics.json` under `fold_models`.

| Feature | Minimum | Maximum |
| --- | ---: | ---: |
| recent_frequency_contrast | -0.5 | 0.5 |
| new_fit_gain | 0.004526024573 | 0.2916400374 |
| old_prior_loss | 0.05923133047 | 0.406774258 |

Primary OOF MSE is 3.5355087e-05; it is diagnostic only and did not enter the advancement gate.

## Conditional bootstrap

The fixed OOF paired values were resampled 10,000 times, drawing 4 seeds with replacement within each family-by-history stratum and using the same sampled seed indices for every contrast and G_min. NumPy PCG64 seed: 162033; quantiles use NumPy's default linear method.

| Quantity | 2.5% | 97.5% |
| --- | ---: | ---: |
| G_b: fixed_keep | 0.00004257 | 0.00395141 |
| G_b: fixed_replace | -0.00065104 | 0.00210635 |
| G_b: intercept_only | -0.00065104 | 0.00210635 |
| G_b: frequency_rule | -0.00097211 | 0.00238991 |
| G_min | -0.00102997 | 0.00178528 |

These are conditional resampling ranges for already fixed held-out values. They do not refit the predictor, include training uncertainty, claim formal 95% coverage, or provide p-values.

All per-seed predictions, features, actions, primary and balanced selected values, gains, and paired differences are in `metrics.json` and `per_seed.csv`.

The result is exploratory reuse of E161 outcomes. A PROMISING_EXPLORATORY status only meets the registered Stage A screen; it does not establish confirmation or authorize another phase.

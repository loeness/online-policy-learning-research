# Experiment index

This index covers the archived A, B, B+, compression, original C1 and C1 supplemental work. The archive preserves exploratory runs, failed validation attempts, intermediate checkpoints, and negative results. C2 and later phases are outside the archive and are not implemented here.

## Read first

| Question | Start here |
|---|---|
| What the code actually runs | [ALGORITHM.md](ALGORITHM.md) |
| How to reproduce checks and experiment cohorts | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| What succeeded, failed, or was invalidated | [SUCCESS_AND_FAILURE_LOG.md](SUCCESS_AND_FAILURE_LOG.md) |
| A/B results | [outputs/validated/结果解读.md](../outputs/validated/结果解读.md) |
| B+ controls | [outputs/ab_plus/结果解读.md](../outputs/ab_plus/结果解读.md) |
| Primary Top-K screen and fixed repair | [outputs/compression_attempt/结果解读.md](../outputs/compression_attempt/结果解读.md), [repair/结果解读.md](../outputs/compression_attempt/repair/结果解读.md) |
| Alternative compression comparison | [outputs/compression_alternatives/综合判断.md](../outputs/compression_alternatives/综合判断.md), [结果解读.md](../outputs/compression_alternatives/结果解读.md) |
| Original C1 results and complete validation record | [outputs/c1/结果解读.md](../outputs/c1/结果解读.md), [validation_ledger.json](../outputs/c1/validation_ledger.json) |

## C1 supplement added after the original snapshot

| Evidence | Entry point |
| --- | --- |
| Three missing comparisons, full raw validation/formal records | [Supplement report](../outputs/c1_supplement/结果解读.md), [protocol](C1_SUPPLEMENT_PROTOCOL.md), [amendments](C1_SUPPLEMENT_AMENDMENTS.md) |
| New compact representations and stronger full actor | [Alternative report](../outputs/c1_supplement/alternative_results.md), [report scope](C1_ALTERNATIVE_RESULTS.md) |
| What the completed evidence supports | [C1_SUPPLEMENT_CONCLUSIONS.md](C1_SUPPLEMENT_CONCLUSIONS.md) |
| Exact grouped/tree equivalence | [C1_GROUPED_EQUIVALENCE.md](C1_GROUPED_EQUIVALENCE.md), [paired PPO evidence](../outputs/c1_supplement/step2_grouping/equivalence/ppo_parity.json) |
| Source-output span and favorable task prior | [C1_SOURCE_BASIS_DERIVATION.md](C1_SOURCE_BASIS_DERIVATION.md) |
| Optimizer counter audit, original metrics preserved | [optimizer_step_audit.json](../outputs/c1_supplement/optimizer_step_audit.json) |
| Source behavior after hidden deployment | [source_retention.json](../outputs/c1_supplement/source_retention.json) |
| Exact source versions used in the original supplemental core | [source_versions](../archive/source_versions/), [cohort source hashes](../outputs/c1_supplement/cohort_source_manifest.json) |

The main supplemental neural cohort has 39 formal and 24 validation runs with
seeds 131/149/167 and 911. The separate alternatives have 18 formal and 15
validation runs with deployment seeds 181/193/211 and validation seeds 919/923.
Do not pool these cohorts as though all arms were measured on all six seeds.
All start from the same single W0 checkpoint. Step 1 has twelve formal and four
validation trajectories plus separate clear-history diagnostics. No C2/C3
runner is executed by this supplement.

## Shared data and model inputs

| Artifact | Purpose and evidence |
|---|---|
| `data/pretrain_source.npz` | 20,000 recorded normal-environment expert transitions, collected with seed 1001. The adjacent [metadata](../data/pretrain_source.json) records the dataset SHA-256 `493c67dc07393a674deb6aa7ab5ba4feea3c1a48eacc16b37046a12606bc56f2`. |
| `data/foundation.pt` | Shared behavior-cloned foundation. [Metadata](../data/foundation.json) records foundation-weight SHA-256 `e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222`, 1,500 behavior-cloning updates, and held-out source / rotation evaluations. |
| `configs/pilot.json`, `configs/selected.json` | Initial and selected A/B training settings. `outputs/validated/config.json` and per-run `config.json` files preserve effective run configurations. |
| `requirements.txt`, `requirements-lock.txt` | Version constraints and the recorded Windows x64 Python 3.12 package resolution. See the platform caveat in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). |

The shared W0 checkpoint is not a per-seed pretraining cohort: A/B deployment seeds 41, 53, and 67, and C1 seeds 71, 83, and 97 all start from the same `data/foundation.pt`.

## Cohort map

| Cohort | Inputs and configuration | Runner / implementation | Results and checkpoints |
|---|---|---|---|
| Autoresearch configuration validation | `configs/pilot.json`, `configs/selected.json`, `configs/ab_plus.json`; validation seed 901 | `scripts/autoresearch.py`, `src/online_rl/experiment.py` | `outputs/autoresearch_validation/` holds the declared trial records and `results.csv`; selected configuration is `configs/selected.json`. This is configuration selection, not an additional A/B seed cohort. |
| A/B pilot, seeds 11, 23, 37 | `configs/pilot.json`; shared dataset and foundation | `scripts/run_experiments.py prepare|one|suite|report`; `src/online_rl/experiment.py`, `policy.py`, `routing.py` | `outputs/pilot/`: 24 runs: A frozen, full, and global residual; B frozen, global residual, history keep/reset, and oracle keep. Each run retains config, summary, episodes, capacity, router, and available agent/audit artifacts. The top-level summary, report, and plots preserve the gate and aggregate view. |
| A/B validated replication, seeds 41, 53, 67 | `configs/selected.json`; shared dataset and foundation | Same A/B runner and implementation | `outputs/validated/`: the same 24 arm × seed combinations, effective `config.json`, top-level summary/report/plots, `analysis.json`, and `provenance.json`. Run directories retain `summary.json`, `episodes.csv`, `capacity.csv`, `router.json`, and checkpoints or rollout audits where applicable. The first pilot's router defect remains in `outputs/pilot`; it is not overwritten by this cohort. |
| Post-hoc A/B diagnostics | Existing validated run files | `outputs/current_analysis/analyze_existing.py`, `plot_diagnostics.py` | `outputs/current_analysis/` contains derived mechanism diagnostics, figures, JSON, and provenance; it does not add environment interactions or change the A/B runs. |
| B+ controls and source checkpoint reconstruction | `configs/ab_plus.json`; three `B_history_keep` terminals under `outputs/validated/` | `scripts/run_ab_controls.py`, `scripts/resume_ab_controls.py`, `src/online_rl/controls.py` | `outputs/ab_plus/`: `no_gradient/` has P+H+, P+H0, and P0H0 branches across three checkpoints and five deployment contexts; `optimizer_isolation/` compares retained/reset Adam per checkpoint; `analytic/` contains A and B history-only controls; `deployment_bundles/` contains reconstructed B checkpoints; root summary, report, `audit.json`, `source_audit.json`, and `provenance.json` capture completion and integrity. The separate matched-initialization A/B PPO reference is `outputs/ab_plus/matched_online/` (six runs) from `scripts/run_matched_online_references.py`. |
| Primary Top-K compression screen | `configs/compression.json`; reconstructed B checkpoints from the B+ source checkpoints | `scripts/run_compression_attempt.py`, `src/online_rl/compression_experiment.py`, `compression.py` | `outputs/compression_attempt/`: `bundles/` stores original, compact dense, and five packed fractions for seeds 41, 53, 67; `validation/` and `test/` retain each fraction × checkpoint × context evaluation; `summary.json`, `结果解读.md`, and `provenance.json` retain the screen, continuation checks, and accounting. The screen uses the existing A/B tasks, not new task families. |
| Fixed 50% compression repair diagnostic | `configs/compression_repair.json`; original primary screen bundles and dense compact references | `scripts/run_compression_repair.py` | `outputs/compression_attempt/repair/`: `before/`, `continuation/`, and `after/` each preserve the three checkpoint pairings, plus config, summary, report, and provenance. The script reads and writes the literal relative path `outputs/compression_attempt`; see the reproducibility limitation. |
| Compression alternatives, nine methods × three checkpoints | `configs/compression_alternatives.json`, `compression_alternatives_protocol.md`; B+ packed/dense source bundles | `scripts/run_compression_alternatives.py`, `scripts/report_compression_alternatives.py`, `src/online_rl/alternative_experiment.py`, `alternative_compression.py` | `outputs/compression_alternatives/`: `bundles/`, `before/`, `continuation/`, and `after/` contain nine predeclared methods × seeds 41, 53, 67, with five contexts per deployment. `summary.json`, `analysis.json`, `spectral_diagnostics.json`, report(s), plots, and `provenance.json` retain the comparison. All nine methods completed; failures remain in the result tables. |
| C1 pre-formal validation and configuration history | `configs/c1.json`, `c1_protocol.md`, common foundation/data | `scripts/run_c1.py --mode precheck`; retained validation runs in `outputs/c1/validation_lr/` and `validation_temperature/` | `outputs/c1/precheck/` is the initial calibration-index-invalid prototype cohort. `precheck_selected/` is the corrected six-context learnability gate. The complete 40-entry accounting record is `outputs/c1/validation_ledger.json`; do not filter it down to selected successes. `representation_diagnostic.json` is an offline known-map representability check with zero environment interactions. |
| C1 formal mechanism cohort, 18 runs | `configs/c1.json`, frozen copy `outputs/c1/frozen_config.json`; selected precheck and preflight budget artifacts | `scripts/run_c1.py --mode formal`; `src/online_rl/c1_experiment.py`, `c1_policy.py`, `c1_routing.py`, `c1_envs.py` | Runs are under `outputs/c1/formal/{low,high}/{flat,tree,dictionary}/s{71,83,97}/`. Each run has phase logs, evaluations, `result.json`, and saved agents. Root `summary.json`, `analysis.json`, `budget_preflight.json`, `representation_diagnostic.json`, report, plots, and validation ledger give the aggregate and audit record. |

## How to navigate a run directory

- `config.json` or the root frozen config defines the run's settings; prefer the per-run effective copy when present.
- `summary.json` / `result.json` is the machine-readable result. `episodes.csv`, `steps.csv`, and `capacity.csv` contain trajectory and resource evidence.
- `router.json` and agent bundles/checkpoints preserve learned state. NPZ audit files preserve rollout evidence when generated.
- Top-level `analysis.json`, reports, plots, `provenance.json`, and `checks.txt` are derived artifacts. They do not replace per-run records.
- `outputs/c1/formal/.../history_eval` is a separate post-history evaluation. Its route is oracle-diagnostic; it measures the changed policy after the preceding hidden-label history continuation, not the hidden-label router's online decisions.


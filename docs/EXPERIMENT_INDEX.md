# Experiment index

## Retention candidate construction and later online state (2026-10-20)

[Review](RETENTION_CANDIDATE_REVIEW.md) · [frozen protocol](RETENTION_CANDIDATE_PROTOCOL.md) · [configuration](../configs/retention_candidate.json) · [reproducibility guide](REPRODUCIBILITY.md) · [completed main report](../outputs/retention_candidate_completed_report/report.md) · [analysis](../outputs/retention_candidate_completed_report/analysis.json) · [independent audit](../outputs/retention_candidate_audit/main_completed.json) · [interrupted inventory](../outputs/retention_candidate_interrupted_inventory.json) · [restart consistency](../outputs/retention_candidate_restart_consistency.json) · [encoding diagnostic](../outputs/retention_candidate_encoding_diagnostics/diagnostics.json) · [diagnostic startup receipt](../outputs/retention_candidate_encoding_diagnostics/startup_attempt_01.json) · [audit records](../outputs/retention_candidate_audit/) · [completed cohort](../outputs/retention_candidate_completed/).

The frozen candidate study has 24 prefixes across six independent deployment seeds, two relations and early/late states; its five fixed arms are keep, replace_mean, replace_behavioral, merge_mean and merge_behavioral. The completed main has 480 continuations and 3,391,488 actual steps; the separate pilot adds 98,304. The interrupted prior cohort preserves 46 complete branch summaries and 355,148 persisted step rows, a lower bound because buffered rows may be missing. A fresh full rerun was used instead of an approximate resume. Hashes for 208 raw ledger/actor files across two completed interrupted prefixes match the corresponding rerun files; these duplicates are not new samples. The independent main audit reports zero errors and warnings.

At 6,144 steps, only dynamics/late shows both positive primary descriptive contrasts: replace_behavioral−replace_mean is +0.003394 reward/step (95% interval [+0.000634,+0.006151]) and replace_behavioral−keep is +0.003254 ([+0.000541,+0.006429]). The other three relation/state conditions do not establish both advantages. In dynamics/late, the future split is unfavorable under old-major and favorable under new-major futures; behavioral merging does not beat keep and is harmful in reward_conflict early and late. The first-256-step allocated-recovery contrasts show no clear replace_behavioral advantage. These six-seed, unadjusted descriptive intervals are candidate-construction evidence, not validation of autonomous knowledge organization. Late jointly changes history, parent knowledge, critic state and source-policy coverage; no selector, tree/DAG, C2, universal or SOTA claim follows. Main campaign CPU is 1,957.625 s (1,957.59375 s child case CPU plus 0.03125 s parent CPU) and wall time is 1,025.3635592 s. The narrower prefix+branch phase counters sum to 1,952.90625 s and omit 4.71875 s of campaign CPU; their residual tasks were not profiled. The 0.625 s fit CPU is already included in the prefix and campaign totals. Complete packages are 232,870–235,246 B under 262,144 B; actors are 89 B.

The zero-interaction [encoding diagnostic](../outputs/retention_candidate_encoding_diagnostics/diagnostics.json) covers 24 prefixes, 120 prefix-arm cells and 480 branches. Pooled and owner training-source feature matrices are rank 3, with some high condition numbers; this is descriptive conditioning evidence and does not establish a causal codec effect. Its [startup receipt](../outputs/retention_candidate_encoding_diagnostics/startup_attempt_01.json) records the first bare-Python attempt failing at NumPy import before output or environment interaction; the repository virtual-environment rerun completed.

## Retention prediction decomposition and action/horizon diagnosis (2026-10-19)

[New hypotheses, verdicts and next gate](RETENTION_MECHANISM_REVIEW.md) · [frozen protocol](RETENTION_MECHANISM_PROTOCOL.md) · [configuration](../configs/retention_mechanism.json) · [complete report](../outputs/retention_mechanism_main_report/report.md) · [machine results](../outputs/retention_mechanism_main_report/analysis.json) · [figure](../outputs/retention_mechanism_main_report/figure.png) · [independent audit](../outputs/retention_mechanism_audit/main.json) · [24 software checks](../outputs/retention_mechanism_software_checks.xml).

Six new mapping-balanced seeds, two relations and two histories give 24 common prefixes. Source matrices each have 1024 fitted rows at storage/proposal time; later task-lifetime exposure is separately recorded. Two futures, two stochastic continuations and four operations give 384 physical branches. Local PPO and natural LRU continue for 12288 steps after a single operation with fixed parent. The 1536/6144/12288 observations are nested, with no duplicate interaction charging. Main cost is **4890624 steps**, 19104 rollouts and 305664 minibatch updates. Independent source/restoration/RNG/codec/continuity/count/capacity audits pass with zero errors and warnings. Packages occupy 231043–232899 B, actors 89 B.

At 12288 steps, cross-replica selection minus keep is −0.00026788 reward/step for dynamics (descriptive seed CI [−0.00061671,+0.00015216]) and +0.00041453 for reward conflict ([−0.00028653,+0.00125843]). Selection minus fixed replace is +0.00021659 and +0.00082906, both intervals spanning zero. Future-prevalence interactions remain positive under matched source-time fits, but longer feedback does not establish a stable selection increment. The future-aware observer is not a deployable selector or a global upper bound. No new organizer or C2 result is claimed.

The successful [pilot v2](../outputs/retention_mechanism_pilot_v2/summary.json) adds 56320 steps and passes its [final audit](../outputs/retention_mechanism_audit/pilot_v2_final.json). The [failed first pilot](../outputs/retention_mechanism_pilot/summary.json) retains 7168 steps from an artifact path error. Preliminary audit and plotting corrections remain archived. This round totals **4954112 physical interactions**. Main campaign wall/CPU are 1446.663/2844.922 seconds with two workers; analysis/report costs are separate.

Zero-interaction additions: [selected-feedback prediction decomposition](../outputs/retention_prediction_diagnostics/report.md), [288-checkpoint coefficient range diagnostic v2](../outputs/retention_content_diagnostics_v2/summary.json), and [verified action-contrast/DR literature notes](literature/20261019_ACTION_CONTRAST_NOTES.md). Prediction improvement is primarily reward-related, while coefficient amplitudes alone do not measure policy quality. The superseded 27-versus-9 coefficient diagnostic is retained with its correction notice.

## Ordered retention validation: confirmation, prediction, and policy comparison (2026-10-19)

[Overall review](ORDERED_RETENTION_VALIDATION_REVIEW.md) · [Stage 1 frozen protocol](SURROGATE_LOWER_CONFIRMATION_PROTOCOL.md) · [Stage 1 report](../outputs/lower_confirmation_report/report.md) · [Stage 1 pilot report](../outputs/lower_confirmation_pilot_report/report.md) · [Stage 1 main and pilot audits](../outputs/lower_confirmation_audit/) · [Stage 2 frozen protocol](RETENTION_VALUE_STREAM_PROTOCOL.md) · [implemented method](RETENTION_VALUE_STREAM_METHOD.md) · [configuration](../configs/retention_value_stream.json) · [Stage 2 calibration report](../outputs/retention_value_calibration_report/report.md) · [Stage 2 pilot report](../outputs/retention_value_pilot_report/report.md) · [calibration main audit](../outputs/retention_value_calibration_main_audit.json) · [pilot audits](../outputs/retention_value_calibration_pilot_audit.json) and [comparison pilot audit](../outputs/retention_value_comparison_pilot_audit.json) · [Stage 3 final report](../outputs/retention_value_main_report/report.md) · [machine results](../outputs/retention_value_main_report/analysis.json) · [comparison audit](../outputs/retention_value_comparison_main_audit.json) · figure exports: [whole stream PNG](../outputs/ordered_retention_figures/s3_comparison_whole_stream.png), [block returns PNG](../outputs/ordered_retention_figures/s3_comparison_block_returns.png), [all formats](../outputs/ordered_retention_figures/).

The independent lower confirmation completes 16 seeds (eight per old-family stratum), 32 prefixes, and 128 observer-scored future branches for 1,966,080 actual steps; its 15,360-step pilot is separate. The primary lower−keep estimate is +0.000049591 reward/step (stratified 95% CI [−0.000514984, +0.000527064]), with 3 positive, 1 negative, and 12 tied seeds. Direction remains uncertain, and the separately frozen +0.002 practical threshold is not supported. All four paid branches are scored by the observer and do not feed back into learning.

Stage 2 calibration completes 12 streams across six seeds and two relations, with 768 selected labels over 1,216,512 steps. On the final 32 of 64 labels per stream, linear−intercept prequential MSE is −3.0619875e−6 (descriptive seed-bootstrap 95% CI [−4.3653208e−6, −1.4695639e−6]). Reward-conflict prediction error improves 11.7% in all six seeds, while dynamics error worsens 3.1% in three of six. This is a selected-feedback prediction result, not policy improvement or an estimate of unchosen-action returns. The independent main audit reports zero errors and warnings.

Stage 3 comparison completes all 80 streams (four new seeds × two relations × ten methods), 48 decisions per stream, and 6,144,000 actual steps with no runtime failures. The primary learned_linear−learned_intercept whole-stream contrasts are +0.00020996094 reward/step for dynamics (descriptive 95% seed-bootstrap CI [−0.00015625, +0.00040852865], 3 positive/1 negative seed) and +0.00004638672 for reward conflict (CI [−0.00034993490, +0.00056233724], 2 positive/2 negative). Neither lower bound exceeds zero or the frozen +0.002 practical threshold. Learned_linear trails replace, lower, and direct shared PPO in both relations; all four seed contrasts are negative for each comparison. Its direct shared-PPO gaps are −0.04211995 in dynamics and −0.01026204 in reward conflict. The Stage 2 selected-feedback prediction gain therefore does not validate a storage-policy gain. Peak package is 240,322 B under the 262,144 B cap; campaign wall is 1,059.754 s and total CPU 5,936 s. The independent ordered protocol audit passes with zero errors and zero warnings, validating all 80 streams, selected-only updates, matched exploration prefixes, actor/checkpoint consistency, and capacity accounting. Across all three stages and pilots, actual cost is 9,443,328 environment steps. See the [final report](../outputs/retention_value_main_report/report.md), [machine results](../outputs/retention_value_main_report/analysis.json), and [ordered review](ORDERED_RETENTION_VALIDATION_REVIEW.md) for full contrasts and scope.

## Source dose and persistence × future PPO (2026-10-18)

[New hypotheses and verdict](SOURCE_FOLLOWUP_REVIEW.md) · [source protocol](SOURCE_DOSE_PROTOCOL.md) · [learning interaction protocol](LEARNING_INTERACTION_PROTOCOL.md) · [combined seed-level report](../outputs/source_followup_main_report/report.md) · [interaction figure](../outputs/source_followup_main_report/learning_interaction_T.png) · [mechanism curves](../outputs/source_followup_main_report/learning_mechanism_curves.png) · [independent final audit](../outputs/source_followup_audit/main_final.json).

Eight new mapping-stratified seeds yield 32 common prefixes. The old-candidate content-only intervention has 512 logical / 384 physical PPO futures; 128 exact post-action hash aliases add zero new interactions. The source main costs 5,111,808 steps. Another 128 fresh no-gradient replace/none futures cost 1,572,864, with no new prefix cost and no reuse of PPO outcomes. Actor/critic optimizer updates are zero only in the latter branches; LRU allocation and persistent-prior inheritance continue.

Source-dose B is +0.001481/+0.000937 reward per step across the two relations; the practical +0.002 criterion is uncertain/not supported, respectively. Primary persistence × future-PPO T is +0.000019 [−0.000957,+0.001020] and −0.000528 [−0.001110,+0.000148], with no clear practical interaction. Existing lower has a small positive signal in one old2048/dynamics/primary cell; this is not a new algorithm or cross-condition success. All means, seed pairs and failures remain available.

Main total is 6,684,672 actual interactions; including the failed source pilot, successful pilots and zero-step ablation startup failure, this round adds 6,776,832. The checkpoint-only feature benchmark adds zero interactions and finds no meaningful time saving. It is not the proposed Sparse Forest structural-maintenance experiment. All new results remain known-context, fixed-layout, single-write C1 mechanism tests.

## Attached online-learning manuscripts: design review only (2026-10-18)

[Synthesis and next discriminating comparisons](literature/20261018_SYNTHESIS.md) · [twelve-source inventory](literature/20261018_sources.json). Four group notes cover all 430 attachment pages, including appendices. The reading itself adds zero environment interactions. The later source-content and future-learning interventions are indexed above. A delayed cost-aware selector was subsequently implemented in the 2026-10-19 ordered study, without establishing a policy increment; affected-state structural maintenance remains unimplemented. This entry is separate from the original storage-decision cohort below.

## One scarce parent position: complete online decision effects (2026-10-18)

[Hypotheses and verdict](STORAGE_DECISION_REVIEW.md) · [frozen protocol](STORAGE_DECISION_PROTOCOL.md) · [seed-level report](../outputs/storage_decision_report_v2/report.md) · [figure](../outputs/storage_decision_report_v2/figure.png) · [independent audit](../outputs/storage_decision_audit/artifact_audit.json) · [read-only diagnostics](../outputs/storage_decision_diagnostics/analysis.json).

Eight new deployment seeds, one W0, two relations and two histories yield 32 common prefixes. Four physical storage actions and two futures produce 256 continuations of 12,288 steps with local PPO, critic updates and natural LRU eviction. The predeclared schedule interaction direction passes in both relations (8/8 and 6/8 positive seeds), but preference reversals occur in only 7/16 and 5/16 nested cases. Frequency has not established superiority over both fixed keep and replace; gain/lower add no consistent value over frequency. The old/new sources have 2,048/1,024 fitting steps each, a limitation and next isolated hypothesis, not a proven explanation. Primary and reversal-stress outcomes stay separate.

Main consumes 3,538,944 interactions; pilot 27,648; total 3,566,592. Main wall is 666.91 s with four workers and full process CPU 2,603.92 s. All actors are 89 B and actual complete packages are 231,683–233,923 B under 262,144 B. Main and pilot audits pass. Reporter v2 fixes a sidecar-field alias without changing the data or package totals; v1 remains archived. Known context IDs, boundaries and group membership are supplied uniformly. This is one controlled allocation decision, not repeated adaptive storage, unknown-context discovery or a SOTA result.

## Frozen score prediction at natural recurrence (2026-10-16)

[New hypothesis and verdict](SCORE_CALIBRATION_REVIEW.md) · [frozen protocol](SCORE_CALIBRATION_PROTOCOL.md) · [seed-level report](../outputs/score_calibration_report/report.md) · [artifact audit](../outputs/score_calibration_audit/artifact_audit.json).

Three new seeds and two budgets produce 47 initial-prefix write tests; all are evaluated later, with no exclusions. The original mean-lower predictor has mean Spearman rho -0.043 (low) and -0.325 (wide), failing the frozen exploratory criterion. Of 35 conditionally MC-beneficial candidates, 27 are not approved; 12 outcomes are ambiguous and none are MC-harmful, limiting risk assessment. The target is cold, frozen-parent conditional 128-step gain, not whole-write or final-policy return. Pilot and main cost 1110016 interactions, including all paid observer forks. No algorithm was tuned from these outcomes.

## Natural eviction and retention writes: Stage 2 (2026-10-16)

[Interpretation](RETENTION_STREAM_STAGE2_REVIEW.md) · [implemented method](RETENTION_STREAM_STAGE2_METHOD.md) · [frozen protocol](RETENTION_STREAM_STAGE2_PROTOCOL.md) · [seed and matched-recovery report](../outputs/retention_stream_stage2_report_v2/report.md) · [artifact/cost audit](../outputs/retention_stream_stage2_audit/artifact_audit.json).

Thirty complete exploratory runs cover five methods, two actor budgets and three new deployment seeds. Simple mean consolidation beats independent relearning in all six full-stream seed/budget pairs; direct shared PPO beats mean in all six. The additional reuse gate harms all six matched full-stream comparisons with ungated mean. The retention write gate has no stable advantage; it approves only seven of 127 actual write tests. The two pilots and main cohort consume 829440 new interactions, with no extra probe/evaluation interactions. Low-budget recovery comparisons match the same task/bout evicted in both methods; the wide budget has no local evictions. These remain controlled C1 mechanism experiments, not C2 or a SOTA comparison.

## Latest write-decision investigation (2026-10-16)

[Full interpretation](WRITE_INVESTIGATION_REVIEW.md) · [algorithm and limits](WRITE_INVESTIGATION_METHOD.md) · [seed-level report](../outputs/write_investigation_report/report.md) · [current stream audits](../outputs/write_current_reports/index.json).

Sixteen completed cohorts contain 100 completed runs, including validation and pilots. The new ledger records at least 5,278,326 interactions including failed and interrupted attempts; this is separate from earlier project totals. The controlled sharing effect does not establish superiority of the benefit gate. Capacity, unknown-boundary, and both shifted-foundation comparisons favor the stronger independent full-use control in mean online reward. Parent recycling validation is negative; its formal/unknown expansion was skipped by the declared rule.

The `write_*` runners, frozen source copies, holdout/gate/route logs, int8 actors, complete checkpoints, failure markers, repair provenance, and figures remain archived. No C2/C3 standard benchmark or SOTA reproduction is implied.

This index covers the archived A, B, B+, compression, original C1, C1 supplemental work, and the subsequent isolated P0 diagnostic. The archive preserves exploratory runs, failed validation attempts, intermediate checkpoints, and negative results. C2 and later phases are outside the archive and are not implemented here.

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


## Research design after C1

[Online knowledge-structure design](ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md) reviews existing tree/modular RL, identifies overlap, and proposes falsifiable shared-write, consolidation and fixed-budget decisions. It is literature and algorithm design, with no new experiments and no C2/C3 results.

### P0: pre-core transfer diagnostic

[Frozen protocol and hypothesis gates](PRE_CORE_VALIDATION_PROTOCOL.md) separates
the definitions required before automatic organization from hypotheses to test.
The protocol was committed as `e8d715a` before formal P0 launch. The separate
[P0 report](../outputs/pre_core_p0/结果解读.md) describes 18 independent local PPO
updates (3 deployment seeds × 6 contexts), each applied unchanged to six targets
with constant route 1 and paired 32-episode evaluation streams. The initial
gradient and final correction vectors share the same 27-dimensional coordinates.
All runs use one W0 and its favorable frozen source-output basis. Contexts are
isolated by the experimenter; no family ID enters training. P0 is not a task-free
learner, dynamic-structure method, capacity-consolidation result, or C2 launch.
Raw episodes, all checkpoints, provenance, and actual interaction accounting
are under `outputs/pre_core_p0`; a reduced software smoke run is separately
retained in `outputs/pre_core_p0_smoke` and excluded from the formal cohort.

## Bounded knowledge organization, three-stage mechanism study

Protocol: `docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md`, committed before formal runs.
Main interpretation: `outputs/knowledge_organization_report/结果与研究判断.md`.

| Cohort | Implementation | Results / scope |
|---|---|---|
| Static consolidation, 3 learned teachers × 5 budgets × 7 original codecs | `compact_knowledge.py`, `run_static_consolidation.py` | `outputs/knowledge_static_formal`; 277,327 interactions; oracle isolation only. Historical source-snapshot limitation recorded in `serialization_audit/source_provenance.json`. |
| Independent global Top-K fairness control | Same codec/runner, explicit `global_control` phase | `outputs/knowledge_static_global_control`; 61,961 interactions; exact binary byte audit. |
| Same-rank policy KL distillation | `scripts/run_knowledge_distillation.py` | `outputs/knowledge_distillation_control`; 32,106 evaluation interactions, no new fit interactions. Post-hoc declared control, 3 teachers × 3 budgets. |
| From-zero task-free identification | `task_free_router.py`, `run_task_free_identification.py` | `outputs/knowledge_identification_formal`; 3 actual streams / 18,644 interactions, 27 same-stream evaluations; no label calibration, no active-exploration comparison. |
| Unknown-boundary online loop | `scripts/run_knowledge_loop.py` | `outputs/knowledge_loop_formal`; 7 arms × 3 seeds × 32,768 steps = 688,128 training interactions. Legacy revisit consumes 105,844 extra steps and is excluded from persistence conclusions. |
| Corrected paired revisit | `scripts/run_knowledge_revisit_corrected.py`; correction protocol in `docs/KNOWLEDGE_REVISIT_CORRECTION.md` | `outputs/knowledge_loop_revisit_corrected`; 1,728 episodes / 102,991 steps; no gradients, slot reuse clears obsolete corrections, episode-specific paired action streams. |
| Pilots and aborted duplicate | Separate `knowledge_*_pilot` directories and `knowledge_static_formal/reproduction_7methods/ABORTED.json` | Completed pilots add 14,753 steps; aborted duplicate records at least 85,781 steps, not a valid formal result. No artifact is silently removed. |

Derived plots and `analysis.json` live in `outputs/knowledge_organization_report`.
New checks cover codec round trips/physical sizes, sparse-support routing,
PPO log probabilities and GAE, held-out timing, generations, and revisit semantics.

## V2 causal diagnosis and optimization (2026-10-15)

For the exact current actor, router, PPO update cycle, codec consolidation and
unimplemented parts of the original research plan, read the
[V2 structure and algorithm description](KNOWLEDGE_V2_STRUCTURE_AND_ALGORITHM.md).

Read [the full V2 report](../outputs/knowledge_v2_report/验证与优化结论.md),
[frozen protocol](KNOWLEDGE_V2_VALIDATION.md),
[implementation review](KNOWLEDGE_V2_IMPLEMENTATION_NOTES.md), and
[endpoint evaluation design](KNOWLEDGE_V2_REVISIT.md).

| Cohort | Implementation | Artifacts and scope |
|---|---|---|
| Common-stream router replay, pilot/formal | `scripts/evaluate_evidence_router.py`, `src/online_rl/evidence_router.py` | `outputs/knowledge_v2_router_{pilot,formal}`; existing recorded streams, zero new interactions; includes short-segment negative result. |
| Six causal controls × 3 new seeds | `scripts/run_knowledge_causal_ablation.py` | `outputs/knowledge_causal_formal`; 589,824 steps. Allocation preservation, oracle routing, dense/int8 and old-gate controls. Oracle is diagnostic and not budget/information matched. Pilot separately contains 49,152 steps. |
| Five V2 conditions × 3 new seeds | `scripts/run_knowledge_v2.py`, `scripts/knowledge_v2_components.py` | `outputs/knowledge_v2_formal`; 491,520 steps. New router, independent int8, old gate, nonblocking consolidation and single head. Same W0; 25% of each online stream reserved from PPO. Pilot separately contains 24,576 steps. |
| Behavior-reference score diagnostic | `scripts/diagnose_knowledge_gate_score.py` | `outputs/knowledge_v2_score_diagnostic`; 8,192 steps, actions/rewards identical to original gate pilot, 4/10 score-sign disagreements. It returns the old score and is not a corrected-gate performance result. |
| Restored endpoint persistence | `scripts/revisit_knowledge_v2.py`, `scripts/restore_knowledge_v2.py` | `outputs/knowledge_v2_revisit`; 15 endpoints, 1,440 paired episodes, 80,319 interactions, no gradients. Clear recent history; slot reuse clears obsolete corrections. |
| Reporting and artifact audit | `scripts/report_knowledge_v2.py`, `scripts/audit_knowledge_v2.py` | `outputs/knowledge_v2_report`; plots, per-seed contrasts, means, raw-log/source/budget checks. No additional interactions. |

All new interactions total 1,243,583. Main V2 enforces W0 file plus saved checkpoint
≤256 KiB at each persistence operation. This is not a whole-process memory bound.
Six true contexts fit six slots; capacity-saturated replacement has not been tested.
Current V2 runners add output-directory options after these cohorts; exact execution
versions remain in each run's source snapshot. See reproducibility instructions.

## Active shared writes and local-capacity pressure (2026-10-16)

These new cohorts preserve all V2 outputs. Read the
[current shared-write structure](SHARED_WRITE_STRUCTURE_AND_LIMITS.md),
[S1a report](../outputs/knowledge_shared_write_report/阶段S1a结果.md),
[S1b report](../outputs/knowledge_shared_write_auto_report/阶段S1b结果.md), and
[S2 report](../outputs/knowledge_shared_write_capacity_report/阶段S2结果.md).

| Cohort | Frozen protocol / implementation | Results and scope |
|---|---|---|
| S1a, true online shared coefficients | `SHARED_WRITE_PROTOCOL.md`, `scripts/run_shared_write.py` | `outputs/knowledge_shared_write_formal`, 20 runs / 5 seeds; 1,010,652 total interactions. Known routes and families, float32 actor cap 1,024 B. Pilot separately 71,578 interactions. |
| S1b, family-free grouping | `SHARED_WRITE_AUTO_PROTOCOL.md`, `scripts/run_shared_write_auto.py` | `outputs/knowledge_shared_write_auto_formal`, 12 runs / 3 seeds; 638,995 interactions including 36,864 probes. Routes and switch times still known. Pilot separately 85,783 interactions, including preserved postprocessing failure. |
| S2, 12 contexts / 6 local slots | `SHARED_WRITE_CAPACITY_PROTOCOL.md`, `scripts/run_shared_write_capacity.py` | `outputs/knowledge_shared_write_capacity_formal`, 12 runs / 3 seeds; 1,442,340 interactions. Given families/IDs, LRU, actual int8 actor 204 B independent / 230 B shared under common 230 B cap. Pilot separately 154,631 interactions. |
| Audits and figures | `scripts/report_shared_write*.py` | Corresponding `knowledge_shared_write*_report` folders. All calculations use saved raw logs; no additional interactions. |

The six S1a/S1b/S2 formal and pilot cohorts total **3,403,979** new interactions.
The same W0 is used throughout; S1's float32 and S2's int8 cohorts are separate.
These tests support actual shared online learning and bounded local reuse, not
automatic full-capacity organization or standard-benchmark SOTA.

S2 checkpoint scope and the independent review are in
[SHARED_WRITE_CAPACITY_REVIEW.md](SHARED_WRITE_CAPACITY_REVIEW.md).

## S3: support, ambiguity and reward conflict (2026-10-16)

- `outputs/knowledge_shared_write_support_diagnostic`: zero new interactions;
  reuses 9,216 saved probe rows for prefix/censoring analysis. Family grouping
  can be correct while A0/A1/A2 remain indistinguishable on actions 0/1.
  Three synthetic tied-reference queries abstain. Protocol:
  `docs/SHARED_WRITE_SUPPORT_DIAGNOSTIC.md`.
- `outputs/knowledge_shared_write_conflict_formal`: nine runs / three seeds,
  147,456 PPO-stream steps + 9,216 probes + 51,176 evaluation steps = 207,848
  interactions. Pilot separately contains 24,543. Same A0 dynamics, opposite
  signed reward, supplied route slots. It is a C1 conflict diagnostic, not a
  standard benchmark or unknown-context algorithm. Protocol:
  `docs/SHARED_WRITE_CONFLICT_PROTOCOL.md`.
- Shared transfer lowers the untrained negative-objective return in all
  three seeds (mean -0.5477). Later old-objective return changes are small
  and mixed; no catastrophic-forgetting claim is made.
- [S3 result and limits](../outputs/knowledge_shared_write_conflict_report/阶段S3结果.md),
  `report.json`, raw audit checks and paired comparisons are in the report folder;
  pilot 42/42 and formal 120/120 checks pass.

All eight S1a/S1b/S2/S3 formal and pilot cohorts total **3,636,370** interactions.
Offline diagnostic reads and report calculations are not additional interactions.

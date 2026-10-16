# Experiment index

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

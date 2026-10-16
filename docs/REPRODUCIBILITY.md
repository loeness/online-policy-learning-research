# Reproducibility

## Latest ordered write investigation

Use the current `.venv` and new, empty output directories. The original evidence and frozen snapshots remain under `outputs/write_*`. `configs/write_followup.json` and `configs/write_detour_budget_supplement.json` declare the matrix; `docs/WRITE_*PROTOCOL.md` and `WRITE_BUDGET_BASELINE.md` state stage-specific assumptions and costs. Maximum concurrent learners is two, with one torch thread per learner.

The completed queue is recorded in `outputs/write_execution_logs/followups_status.json`; shifted-generator full-use controls are recorded in `outputs/write_detour_budget_logs/coordinator_status.json`. These queues skip completed work and preserve reports; use individual runner CLIs and new paths for a fresh reproduction. Do not overwrite historical directories to rerun.

Regenerate derived aggregate/figures without new interactions:

```powershell
.venv\Scripts\python.exe scripts/summarize_write_investigation.py --output outputs/write_investigation_report_reproduction
.venv\Scripts\python.exe scripts/plot_write_investigation_focus.py --analysis outputs/write_investigation_report_reproduction/analysis.json --output outputs/write_investigation_figures_reproduction
.venv\Scripts\python.exe scripts/rebuild_write_stream_reports.py --output outputs/write_current_reports_reproduction
```

The metadata repair script is a one-time audited repair and intentionally refuses a second repair. Restored checkpoints contain learner state, not a saved live environment trajectory; this is not exact mid-episode trajectory resumption. See [current findings](WRITE_INVESTIGATION_REVIEW.md).

The archive contains both the original experiment records and commands for fresh reruns. Fresh runs should write to a new output directory; several historical runners default to paths inside `outputs/` and would otherwise overwrite archive records. The original snapshot software check contained 82 tests; C1 supplement checks add to that suite. The current recorded total is in `outputs/c1_supplement/software_checks.txt`. Full A/B+ and C1 runs execute many hundreds of thousands to over a million simulated transitions and are not quick checks.

## Environment

The recorded environment is Python 3.12, CPU PyTorch 2.8.0, Stable-Baselines3 2.8.0, Gymnasium 1.2.2, NumPy 2.5.4, Matplotlib 3.11.2, pytest 9.1.1, and psutil 7.2.2. `requirements-lock.txt` says it is resolved from a Windows x64 Python 3.12 environment; it is not a platform-neutral lock. Use it for the recorded Windows environment. On Linux, use the version constraints in `requirements.txt` and resolve a Linux CPU environment; exact dependency resolution can vary by platform.

### PowerShell (recorded Windows setup)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -r requirements-lock.txt
& .\.venv\Scripts\python.exe -m pytest tests -q
```

### Linux (CPU, version constraints)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -r requirements.txt
.venv/bin/python -m pytest tests -q
```

The original snapshot reported `82 passed`. The expanded suite includes real short PPO integration checks and does not generate the archived formal experimental cohorts. No tests generate experiment results. For provenance-sensitive reruns, verify the shared files before starting:

```powershell
Get-FileHash data\pretrain_source.npz -Algorithm SHA256
Get-FileHash data\foundation.pt -Algorithm SHA256
```

```bash
sha256sum data/pretrain_source.npz data/foundation.pt
```

Expected hashes are in `data/pretrain_source.json` and `data/foundation.json`. Do not regenerate these two inputs in place if the goal is to reproduce the archived cohorts.

## A/B runs

The first archived software check is the 82-test suite. A fresh A/B suite with the checked-in W0 checkpoint can be run as follows.

### PowerShell

```powershell
& .\.venv\Scripts\python.exe scripts/run_experiments.py suite --config configs/selected.json --output outputs/reproduction/ab --seeds 41,53,67 --dataset data/pretrain_source.npz --checkpoint data/foundation.pt
```

### Linux

```bash
.venv/bin/python scripts/run_experiments.py suite --config configs/selected.json --output outputs/reproduction/ab --seeds 41,53,67 --dataset data/pretrain_source.npz --checkpoint data/foundation.pt
```

The full A/B suite trains the A and B arms across three deployment seeds. `prepare` regenerates the dataset and checkpoint, so use it only with new `--dataset` and `--checkpoint` paths when reproducing pretraining. The original source transitions and W0 checkpoint are already archived.

## B+ and compression reruns

These commands preserve the checked-in results by using new roots wherever the runner exposes an output option.

```powershell
& .\.venv\Scripts\python.exe scripts/run_ab_controls.py --config configs/ab_plus.json --output outputs/reproduction/ab_plus
& .\.venv\Scripts\python.exe scripts/run_matched_online_references.py --config configs/ab_plus.json --output outputs/reproduction/ab_plus/matched_online
& .\.venv\Scripts\python.exe scripts/run_compression_attempt.py --config configs/compression.json --output outputs/reproduction/compression_attempt
& .\.venv\Scripts\python.exe scripts/run_compression_alternatives.py --config configs/compression_alternatives.json --output outputs/reproduction/compression_alternatives
```

```bash
.venv/bin/python scripts/run_ab_controls.py --config configs/ab_plus.json --output outputs/reproduction/ab_plus
.venv/bin/python scripts/run_matched_online_references.py --config configs/ab_plus.json --output outputs/reproduction/ab_plus/matched_online
.venv/bin/python scripts/run_compression_attempt.py --config configs/compression.json --output outputs/reproduction/compression_attempt
.venv/bin/python scripts/run_compression_alternatives.py --config configs/compression_alternatives.json --output outputs/reproduction/compression_alternatives
```

`run_compression_repair.py` has no CLI options and reads/writes `outputs/compression_attempt` directly. Run it from an isolated copy or worktree that contains the completed primary screen at that relative path; it writes `outputs/compression_attempt/repair`. It performs fixed 50% repair, not a new proportion search. `report_supplements.py` and `report_compression_alternatives.py --root <output>` generate derived reports from their corresponding saved runs.

## C1: quick gate and formal run

The checked-in C1 archive has one common foundation, a six-context selected precheck, a full 40-entry validation ledger, a frozen 18-run config, and an offline representation diagnostic. The formal cohort is 18 runs and adds 1,142,784 training plus 276,480 fixed-policy evaluation interactions; the selected precheck alone adds 55,296 interactions. The 304,641 validation-ledger total includes pre-formal development and the selected precheck, so do not add it to the selected-precheck count as if the counts were disjoint.

Run the selected six-context precheck in a fresh directory first:

```powershell
& .\.venv\Scripts\python.exe scripts/run_c1.py --mode precheck --config configs/c1.json --output outputs/c1_reproduction
```

```bash
.venv/bin/python scripts/run_c1.py --mode precheck --config configs/c1.json --output outputs/c1_reproduction
```

This creates the configured `precheck_selected/` results and is only the learnability gate; it does **not** create the historical validation ledger or the representation diagnostic. `--mode formal` reads the selected precheck, runs the 18 formal arms, and then reads `validation_ledger.json` to write the summary's validation interaction count. The reporter additionally requires a nonempty validation ledger and `representation_diagnostic.json`. The original full ledger and diagnostic are retained under `outputs/c1/`.

To run the formal cohort while retaining the original development record, copy those two archival static records into the fresh output before formal/reporting. This reuses the archive's validation accounting; it does not claim to regenerate the 40 historical validation entries.

### PowerShell

```powershell
Copy-Item outputs\c1\validation_ledger.json outputs\c1_reproduction\validation_ledger.json
Copy-Item outputs\c1\representation_diagnostic.json outputs\c1_reproduction\representation_diagnostic.json
& .\.venv\Scripts\python.exe scripts/run_c1.py --mode formal --config configs/c1.json --output outputs/c1_reproduction
& .\.venv\Scripts\python.exe scripts/report_c1.py --output outputs/c1_reproduction
```

### Linux

```bash
cp outputs/c1/validation_ledger.json outputs/c1_reproduction/validation_ledger.json
cp outputs/c1/representation_diagnostic.json outputs/c1_reproduction/representation_diagnostic.json
.venv/bin/python scripts/run_c1.py --mode formal --config configs/c1.json --output outputs/c1_reproduction
.venv/bin/python scripts/report_c1.py --output outputs/c1_reproduction
```

`run_formal` computes `budget_preflight.json` from the selected precheck logs if that file is absent. It is possible to rerun all 18 formal runs from the checked-in code and shared W0; a fresh output root is not self-contained for the full historical validation narrative because no checked-in command regenerates the 40-entry ledger or the diagnostic JSON. Keep the copied archival provenance explicit when presenting a fresh formal rerun.

## Archive evidence versus fresh execution

- `outputs/c1/summary.json` records `complete=true` and 18 runs; all original formal phase logs and agent checkpoints remain under `outputs/c1/formal/`.
- The original C1 report is generated from saved results by `scripts/report_c1.py --output outputs/c1`; report generation does not rerun training.
- Original compression alternative outputs are already complete (`complete=true`, no runner errors, 774,144 new interactions). The primary Top-K screen and its fixed 50% repair have separate summaries and reports.
- A/B+ terminal archives do not include the entire historical RNG state for the original training run. Reconstructed deployment bundles restore the stored policy, optimizer, and router state and use the explicitly recorded new RNG arrangement; they are not exact interrupted-run continuations.
- Resource fields called serialized state or bundle bytes exclude process runtime and are distinct from separately sampled RSS. RSS is a sampled process measure, not a continuously measured hard peak.


## Ordered C1 supplements

The archived results are already present. Runners resume from existing complete
records. For an independent reproduction, use an isolated repository copy with
an empty supplemental output directory; retain the original C1 checkpoints
needed by Step 1's frozen-tree diagnostic. Do not erase or replace the original
A/B/C1 records in the research checkout. The current report/probe commands read
the literal `outputs/c1_supplement` layout.

Run sequentially from the repository root:

```powershell
.venv\Scripts\python.exe scripts/run_c1_models.py --mode all
$env:C1_RUN_GROUPED_PPO_PARITY = '1'
.venv\Scripts\python.exe -m pytest tests/test_c1_grouped_equivalence_online.py -q
.venv\Scripts\python.exe scripts/run_c1_supplement.py --stage step2
.venv\Scripts\python.exe scripts/run_c1_supplement.py --stage step3
.venv\Scripts\python.exe scripts/run_c1_tensor_alternatives.py
.venv\Scripts\python.exe scripts/run_c1_context_heads.py
.venv\Scripts\python.exe scripts/audit_c1_optimizer_steps.py
.venv\Scripts\python.exe scripts/check_c1_source_retention.py
.venv\Scripts\python.exe scripts/report_c1_supplement.py
.venv\Scripts\python.exe scripts/report_c1_alternatives.py
```

The Linux equivalents use `.venv/bin/python` and
`C1_RUN_GROUPED_PPO_PARITY=1 .venv/bin/python -m pytest ...` for the opt-in paired
PPO check. The source-retention probe requires all 57 formal neural records
before evaluating. Hyperparameters are chosen on validation streams before the
corresponding formal runs; failures and all candidates are retained.

`archive/source_versions/c1_supplement_experiment_core.py` preserves the exact
core-run source whose PyTorch nested step hook could double-count LeafAdam
calls. Checkpoint `optimizer.step_count` increments once per real update. The
sidecar audit corrects reporting, leaving raw JSON and checkpoint bytes
unchanged. Current code prevents double-counting for future cohorts without
changing PPO update behavior. The original Step 2 full-control module before
the visible-band amendment is also archived and hash-verified; Step 3 uses the
amended full-control module. These version differences are recorded, not
retroactively substituted into historical hashes.

The default large comparisons are CPU experiments with one Torch thread.
Measured process CPU includes trace and capacity instrumentation. Byte caps
apply to declared serialized agent plus numeric rollout state, not the whole
Python/Torch process; RSS and live numeric estimates are separate. Full actor
controls use a larger declared complete cap and remain outside the original
adaptive actor budgets. They are not memory-matched victories or faithful
SOTA reproductions.
# 2026-10-15 bounded knowledge organization additions

Use new output directories when reproducing these experiments; preserve the
archived cohorts. The runner snapshots in each result directory record the
exact code used, including pre-CLI amendments. All formal deployment seeds share
one W0. The study report records the historical static-source limitation and
the separate corrected revisit.

```powershell
python scripts/run_static_consolidation.py --phase formal --output outputs/knowledge_static_replication
python scripts/run_static_consolidation.py --phase global_control --output outputs/knowledge_global_replication
python scripts/run_knowledge_distillation.py --output outputs/knowledge_distillation_replication
python scripts/run_task_free_identification.py --phase formal --output outputs/knowledge_identification_replication
python scripts/run_knowledge_loop.py --phase formal --output-dir outputs/knowledge_loop_replication
python scripts/run_knowledge_revisit_corrected.py --output outputs/knowledge_revisit_replication
python -m pytest -q tests/test_compact_knowledge.py tests/test_task_free_router.py tests/test_knowledge_loop.py tests/test_knowledge_revisit.py
```

The distillation control reads the archived static reference CSV. The corrected
revisit command reads the archived original online-loop checkpoints; using the
newly replicated training endpoints requires changing that explicit input root.
They are separate controls and do not overwrite their inputs. A restored loop
checkpoint supports evaluation; it does not contain the environment/rollout
state needed to resume training at an arbitrary interrupted step.

## V2 causal validation and optimized loop

The V2 protocol was frozen at `36962ea` before formal execution. Exact executed
sources are inside each run's `source_snapshot`; current runner versions add
output-directory options after execution without changing the archived cohorts.
All runners refuse to overwrite nonempty run directories. Use new directories:

```powershell
python scripts/evaluate_evidence_router.py --phase formal --output outputs/knowledge_router_replication
python scripts/run_knowledge_causal_ablation.py --phase formal --formal-reviewed --output-dir outputs/knowledge_causal_replication
python scripts/run_knowledge_v2.py --phase formal --output outputs/knowledge_v2_replication
python scripts/revisit_knowledge_v2.py --input outputs/knowledge_v2_replication --output outputs/knowledge_v2_revisit_replication
python -m pytest -q tests/test_evidence_router.py tests/test_knowledge_causal_ablation.py tests/test_knowledge_v2_components.py tests/test_knowledge_v2_harness.py tests/test_restore_knowledge_v2.py
```

The score diagnostic has a fixed archival output root; it has already run and
refuses overwrite. Inspect its saved source and existing outputs. Reporting and
audit commands operate on the original completed V2 cohort, without new interactions:

```powershell
python scripts/report_knowledge_v2.py
python scripts/audit_knowledge_v2.py
```

The audit compares older `data/` and `outputs/` against the manifest in `b7cd1e1`,
then checks the 33 formal raw streams, source snapshots, main-V2 checkpoint byte
caps, episode pairing and interaction ledger. The report summarizes three new
deployment seeds with one shared W0; it does not estimate independent pretraining
variability or certify policy improvement. Restoring a V2 checkpoint requires
`restore_v2` and its saved condition, rather than the V1 default router factory.
The main recovery-package cap includes W0 plus serialized checkpoint, while
transient workspaces and measured process RSS are separate quantities. Six
contexts and six slots do not test saturated persistent capacity.

## Actual online shared writes and local-capacity pressure (2026-10-16)

Use new output directories; archived results are immutable inputs. All these
experiments share `data/foundation.pt`, rather than independently pretraining
one foundation per deployment seed. Two CPU workers, each with one Torch
thread, were used. CUDA was unavailable in the installed CPU PyTorch runtime;
no GPU timing or CPU/GPU speed comparison is claimed.

```powershell
python scripts/run_shared_write.py --phase pilot --output outputs/s1a_pilot_replication --workers 2
python scripts/run_shared_write.py --phase formal --output outputs/s1a_replication --workers 2
python scripts/report_shared_write.py --input outputs/s1a_replication --output outputs/s1a_report_replication
python scripts/run_shared_write_auto.py --phase pilot --output outputs/s1b_pilot_replication --workers 2
python scripts/run_shared_write_auto.py --phase formal --output outputs/s1b_replication --workers 2
python scripts/report_shared_write_auto.py --input outputs/s1b_replication --output outputs/s1b_report_replication
python scripts/run_shared_write_capacity.py --phase pilot --output outputs/s2_pilot_replication --workers 2
python scripts/run_shared_write_capacity.py --phase formal --output outputs/s2_replication --workers 2
python scripts/report_shared_write_capacity.py --input outputs/s2_replication --output outputs/s2_report_replication
```

The frozen protocols are `SHARED_WRITE_PROTOCOL.md`,
`SHARED_WRITE_AUTO_PROTOCOL.md`, and `SHARED_WRITE_CAPACITY_PROTOCOL.md`.
Each formal run snapshots its executed sources and dependencies. The report
scripts check those snapshots against their own hashes, not current source
hashes. S1a's later restore-map assertion, S1b's pilot dictionary-comparison
repair, and S2's checkpoint wording clarification do not substitute new code
for historical versions.

S1a tests supplied groups, S1b discovers groups from 512 real probe steps per
context with supplied routes, and S2 tests 12 known contexts with 6 local
slots and supplied families. S1b pays probe costs even on held-out targets;
its cold transfer is probe-mediated. S2's actor is quantized and reinstalled
every 1,024 steps, while S1a/S1b retain float32 and Adam. The same-byte-cap
comparisons still have different actual bytes. These cohorts must not be
pooled as one unknown-context or memory-saving result.

Saved state supports agent restoration and evaluation. It does not include
the running environment, partial episode, or schedule position needed for
exact trajectory continuation. W0 plus one checkpoint is measured against
256 KiB; source, archived traces and historical checkpoints are reproducibility
artifacts and not a retrieval store available to the deployed learner.

The S3 support diagnostic reads saved S1b probes and generates no new environment
steps. Its observer local-reference analysis is not an online routing decision.
The reward-conflict experiment is a separate float32 cohort:

```powershell
python scripts/diagnose_shared_write_support.py --input outputs/s1b_replication --output outputs/s3_support_replication
python scripts/run_shared_write_conflict.py --phase pilot --output outputs/s3_conflict_pilot_replication --workers 2
python scripts/run_shared_write_conflict.py --phase formal --output outputs/s3_conflict_replication --workers 2
python scripts/report_shared_write_conflict.py --input outputs/s3_conflict_replication --output outputs/s3_conflict_report_replication
```

S3 uses identical transitions and reversed rewards with two supplied route slots.
Goal reaching remains metadata in the negative-reward task; signed return is its
score. Do not combine this cohort with S2 as an unknown-boundary capacity result.

Archive-only validation, with no new environment interactions:

```powershell
python scripts/audit_shared_write_archive.py
python scripts/verify_archive.py
```


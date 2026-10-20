# Research archive inventory

## Retention candidate construction and later online state (2026-10-20)

The local record includes the frozen [protocol](RETENTION_CANDIDATE_PROTOCOL.md), [review](RETENTION_CANDIDATE_REVIEW.md), configuration and freeze receipt; the completed pilot and main cohort; the preserved interrupted cohort and lower-bound inventory; source snapshots, raw ledgers, actor/checkpoint/simulator state; restart-consistency hashes; software checks and their failed predecessor records; pilot and main reports; and all preliminary/final audit records. The [reproducibility guide](REPRODUCIBILITY.md), [completed main report](../outputs/retention_candidate_completed_report/report.md), [main audit](../outputs/retention_candidate_audit/main_completed.json), [encoding diagnostic](../outputs/retention_candidate_encoding_diagnostics/diagnostics.json), [retrospective startup receipt](../outputs/retention_candidate_encoding_diagnostics/startup_attempt_01.json), and [diagnostic/audit directory](../outputs/retention_candidate_audit/) provide entry points. The completed main covers 24 prefixes and 480 continuations for 3,391,488 steps; the pilot adds 98,304. The interrupted record has 46 complete branch summaries and 355,148 persisted step rows as a lower bound. The 208 matching raw-ledger/actor files across two completed prefixes are duplicate executions with unchanged hashes, not additional independent samples. Preserve the interrupted outputs and the software, auditor, and pilot-report failures alongside the successful rerun.

The encoding diagnostic adds no RL interactions and covers 24 prefixes, 120 prefix-arm cells and 480 branches. Pooled and owner training-source feature matrices have rank 3, with some high condition numbers; these descriptive checks do not establish a causal codec effect. The first bare-Python attempt failed on NumPy import before writing output; the repository virtual-environment rerun completed.

Main campaign CPU is 1,957.625 s (1,957.59375 s child case CPU plus 0.03125 s parent CPU) and wall time is 1,025.3635592 s. The narrower prefix+branch phase counters sum to 1,952.90625 s and omit 4.71875 s of campaign CPU; their residual tasks were not profiled. Fit CPU is 0.625 s and is included in the prefix and campaign totals. All complete packages are 232,870–235,246 B under the 262,144 B cap, with 89 B actor containers. Once the final archive manifest is generated, its exact file and byte counts supersede all earlier inventory totals. This entry describes local archive coverage and does not assert that these files have been pushed.

## Retention action/horizon and prediction-source update (2026-10-19)

The current archive adds the matched-source mechanism protocol/configuration, producer, report/audit tools, four focused test files and 24-check JUnit result; 24 common prefixes, 384 physical PPO continuations, complete raw logs, actor/checkpoint/simulator packages and source snapshots; the failed first pilot and successful v2 pilot; all preliminary/final audits and reports; and PNG/PDF/SVG exports with preserved prior figures and analysis/replot source provenance. Main audit has zero errors and warnings. Including both pilots, this round costs 4954112 actual simulated interactions. The [review](RETENTION_MECHANISM_REVIEW.md) records the weak selector findings, conditional operation effects, complete cost ledger and next gate.

Zero-interaction additions include prediction-error decomposition, input/code hashes, the corrected 288-checkpoint coefficient-range diagnostic and its superseded v1, plus three verified action-contrast/DR references. These analyses do not create new policy-learning outcomes. The regenerated manifest's exact file/byte totals supersede historical tables below; reproducibility archive bytes are separate from the 262144-byte deployment package ceiling.

## Ordered retention validation update (2026-10-19)

The local research record now includes the Stage 1 lower-confirmation protocol, configuration, pilot and main reports, per-seed data, raw branch accounting and independent audits; the Stage 2 selected-feedback protocol, method, configuration, runner, reporters, audit tools, model, tests, frozen source snapshots, pilot outputs and audits, completed calibration main streams/report/audit; and the completed Stage 3 comparison main streams, frozen configuration, final report and analysis. All exported figures are under `outputs/ordered_retention_figures/`. Stage 3 has 80/80 completed streams and zero runtime failures; the separate ordered protocol audit at `outputs/retention_value_comparison_main_audit.json` passes with zero errors and zero warnings. The three main stages and pilots total 9,443,328 actual environment steps. Pilot protocol-snapshot differences, the shared actor-Adam reset caveat, same-layout independent baseline, and scope limits are documented in [the ordered review](ORDERED_RETENTION_VALIDATION_REVIEW.md) and [success/failure log](SUCCESS_AND_FAILURE_LOG.md).

The refreshed `archive/manifest.json` covers this update. Its exact file and byte counts take precedence over historical inventory tables. Run `scripts/verify_archive.py` to verify every recorded file. Original experiment outputs are preserved.

## Literature-driven follow-up update (2026-10-18)

The current manifest adds the twelve-paper notes/source index, source-dose and learning-interaction protocols/configuration/code/tests, 32 common prefixes, 384 physical PPO futures and 128 fresh no-gradient futures, exact-state alias metadata, learner/simulator checkpoints, all raw logs, failed/successful pilots, source snapshots, derived reports/figures and independent audits. Superseded pilot/schema-debug reports stay labeled and preserved. The checkpoint-only clean/concurrent score-feature benchmarks are separate zero-interaction diagnostics. [The review](SOURCE_FOLLOWUP_REVIEW.md) gives the 6,776,832-interaction incremental ledger, narrow positive signals, negative practical findings and remaining method gap.

Previous unsynced retention Stage 2, score-calibration and original storage-decision artifacts remain included. Archive size is reproducibility storage, not the deployed learner budget. Original source-document extracts and machine-local material remain excluded. Rebuild and verify the manifest after all final result/document edits; its exact current file/byte counts take precedence over historical tables below.

## Storage decision update (2026-10-18)

The local manifest covers the complete single-position storage experiment: 32 common prefixes, 256 continued-PPO branches, separate pilot, frozen configuration/protocol, original source snapshots, raw steps and update logs, learner and simulator states, v1/v2 reports, figures, independent audits and negative results. [The review](STORAGE_DECISION_REVIEW.md) separates supported schedule effects from unvalidated adaptive selector value. It also records candidate maturity and codec limits. Manifest inclusion describes local archival coverage, not confirmation that new files have been pushed to GitHub.

## Latest write investigation (2026-10-16)

The regenerated manifest includes all new `write_*` experiment artifacts, independent foundation checkpoints, candidate/control source snapshots, failed/interrupted ledgers, metadata-repair history, current audits, raw and focused figures, references, code and tests. There are 16 complete cohort roots / 100 complete runs plus separately costed failed or interrupted attempts. [Full interpretation](WRITE_INVESTIGATION_REVIEW.md) records negative results and provenance limits. Manifest byte totals describe the Git archive, not the deployed learner memory budget. Confidential local extraction remains excluded.

Initial inventory date: 2026-10-14; expanded on 2026-10-15. The archive includes A/B, B+, compression, C1, P0, and the three-stage bounded knowledge-organization study. The new closed loop uses the controlled C1 environment; it is not a completed general-environment C2 benchmark. Historical size tables below describe their original snapshots; the manifest gives current counts and bytes.

## Coverage

The archive retains the raw traces, checkpoints, reports, figures, provenance records, source data, configuration, protocols, code, and tests. Failed runs and negative results remain part of the record. The manifest generator scans the workspace directly and does not follow `.gitignore`, so research files ignored by Git are still checksummed.

The 16 files under `tmp/pdfs/` (5,925,239 bytes) are extracted pages and text from a local source document. They are not experimental data or reproducibility scripts and are excluded from the Git/cloud archive; keep them local. No runnable or reproducibility scripts were found under `tmp/`. Record the runnable research entry points in `scripts/`, including `autoresearch.py`, `run_experiments.py`, `run_ab_controls.py`, `run_matched_online_references.py`, the compression runners and report scripts, and `run_c1.py` / `report_c1.py`. `outputs/current_analysis/` also includes the analysis and plotting scripts for that diagnostic.

The manifest excludes `.git/`, `.venv/`, `.archive-verification/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, `tmp/`, and `archive/manifest.json` itself. It also skips filenames matching `.env`, `.env.*`, `*.pem`, or `*.key`; the generator checks only path names and never reads those files. The manifest itself is omitted to prevent recursive hashing. All other files are included regardless of extension or Git ignore status. Paths use forward slashes and are sorted lexicographically. Version 1 records each file's exact byte count and SHA-256, with a total count and size and no timestamp so identical inputs produce identical JSON.

Build or refresh it from the repository root with:

```powershell
python scripts/build_archive_manifest.py
```

This writes `archive/manifest.json`. Rerun it after any source file, report, result, or `.gitignore` change.

## Size and packaging facts

The baseline below is the initial research-file scan after excluding `.git/`, `.venv/`, Python caches, and `tmp/`; it includes the payloads that were initially ignored by Git. It predates the archive documentation and generator added during this task. The exact current file count and size are recorded in `archive/manifest.json`.

| Scope | Files | Raw bytes |
|---|---:|---:|
| Whole research workspace | 2,771 | 280,077,831 |
| `outputs/` | 2,709 | 279,226,335 |
| Largest single file | 1 | 3,188,498 |

No included file is larger than 50 MB or 100 MB. The largest files are step-level CSV traces under `outputs/ab_plus/analytic/` and `outputs/c1/`; the CSV collection totals 241,013,393 bytes across 958 files. The largest non-CSV artifacts are mostly model checkpoints and ZIP bundles, each below 1 MB in this snapshot. A Git archive would carry about 280 MB of raw research files, chiefly text traces; there is no single large binary that requires LFS based on file size alone. The root can choose whether to keep all traces in Git or package some elsewhere.

Top-level output cohorts:

| Path | Files | Raw bytes | Coverage |
|---|---:|---:|---|
| `outputs/autoresearch_validation/` | 23 | 939,477 | Three declared configuration trials using validation seed 901. |
| `outputs/pilot/` | 149 | 11,360,433 | Initial A/B runs on seeds 11, 23, and 37, including the repeated-node defect and its negative results. |
| `outputs/validated/` | 171 | 12,860,046 | Corrected A/B reruns on seeds 41, 53, and 67. |
| `outputs/ab_plus/` | 196 | 43,960,237 | B+ history/parameter controls, optimizer reset versus retention, analytical history-only controls, and matched online references. |
| `outputs/compression_attempt/` | 364 | 39,433,508 | Five prespecified sparse Top-K ratios, their full validation results, and the separate fixed 50% repair cohort. The original screening failures and repair outcomes are preserved. |
| `outputs/compression_alternatives/` | 745 | 72,211,399 | Nine-arm comparison: dense, Top-K 50%, centered Top-K 50%, activation Top-K 50%, 32-column pruning, BF16, exact contrast coordinates, rank-2 SVD, and distilled rank-2 SVD; includes calibration, before/after evaluation, bundles, continuation, analysis, and figures. |
| `outputs/c1/` | 1,054 | 98,023,335 | Early precheck failures, corrected selected prechecks, learning-rate and temperature validation, 18 formal runs, the validation ledger, reports, JSON analyses, and figures. |
| `outputs/current_analysis/` | 7 | 437,900 | Offline diagnostics and their analysis/plot scripts. |

The C1 formal cohort is two capacity levels × three methods (flat, tree, dictionary) × three seeds (71, 83, 97). The archive includes the early calibration-index precheck interruption, independent validation screens, and formal runs so the validation history can be audited alongside final results.

The 21 Markdown files in the baseline include the research plan, protocols, design notes, and output interpretation reports. The 41 Python files include run/report scripts, the `src/online_rl/` implementation, tests, and analysis utilities. Seven JSON files under `configs/` record the runnable experiment configurations.

## Git-ignore gap found and resolved

The initial scan found `.gitignore` rules for `data/*.npz`, `data/*.pt`, and `outputs/**/agent.zip`. They would have hidden 59 archive inputs totaling 6,417,939 bytes from ordinary `git add`:

| Ignored path group | Count | Bytes | Current ignore rule |
|---|---:|---:|---|
| `data/pretrain_source.npz` | 1 | 116,878 | `data/*.npz` |
| `data/foundation.pt` | 1 | 55,477 | `data/*.pt` |
| `outputs/pilot/{A_frozen,A_full,A_global_residual,B_frozen,B_global_residual,B_history_keep,B_history_reset,B_oracle_keep}_s{11,23,37}/agent.zip` | 24 | 2,629,831 | `outputs/**/agent.zip` |
| `outputs/validated/{A_frozen,A_full,A_global_residual,B_frozen,B_global_residual,B_history_keep,B_history_reset,B_oracle_keep}_s{41,53,67}/agent.zip` | 24 | 2,556,088 | `outputs/**/agent.zip` |
| `outputs/ab_plus/optimizer_isolation/s{41,53,67}/{reset_optimizer,retained_optimizer}/agent.zip` | 6 | 716,554 | `outputs/**/agent.zip` |
| `outputs/autoresearch_validation/trial_{0,1,2}/agent.zip` | 3 | 343,111 | `outputs/**/agent.zip` |

The archive ignore policy was updated after that scan. The current `.gitignore` intentionally tracks research datasets, weights, raw logs, and figures, and the paths above no longer match ignore rules. The manifest includes all 59 files; rebuild it after any later `.gitignore` or artifact change.

## Provenance and source examples

- `data/pretrain_source.npz` holds the collected source-environment transitions; `data/pretrain_source.json` records the dataset metadata and hash. `data/foundation.pt` and its JSON sidecar form the fixed foundation checkpoint and metadata.
- `outputs/validated/provenance.json`, `outputs/ab_plus/provenance.json`, and the corresponding compression and C1 provenance files record source/configuration/code/log checksums for their runs.
- Individual run folders retain configuration, episode and step CSVs, capacity data, summaries, routers, and restorable `agent.zip` checkpoints. The C1 directory additionally retains `validation_ledger.json`, prechecks, validation screens, formal stages, and the figures used in the report.

The filename-only scan found no names matching credential or private-key patterns. No file contents were read for that scan.

## C1 supplement update (2026-10-15)

The `codex/c1-strong-baselines` branch adds the complete ordered C1 controls,
independent validation/formal alternatives, source probes, trajectory-derived
interaction diagnostics, reports, figures and software checks. See
`archive/C1_SUPPLEMENT_SNAPSHOT.md` for scope and
`outputs/c1_supplement/complete_ledger.json` for disjoint experimental counts.
The original `v0.1.0-ab-c1` tag and original data/result bytes are retained.
The current branch manifest covers the larger archive; historical size figures
above remain labeled as the original snapshot scan. Two exact source modules
and the tensor launch document are preserved under `archive/source_versions`.

## V2 validation update (2026-10-15)

The V2 additions preserve 9,134 existing data/output files byte-for-byte against
the manifest at `b7cd1e1`. New causal controls, evidence-router comparisons,
nonblocking consolidation, scoring diagnostics and restored endpoint evaluations
are indexed in `EXPERIMENT_INDEX.md`. Their raw traces, source snapshots,
checkpoints, plots, negative results and interaction ledger are included.
`outputs/knowledge_v2_report/audit.json` records raw-log checks and 1,243,583 new
interactions; same-stream router replay adds no interactions. The main V2 recovery
package cap and its limited scope are recorded separately from archival file size.
The current complete inventory remains `archive/manifest.json`.

## Shared-write update (2026-10-16)

The S1a/S1b/S2/S3 pilots and formal experiments, offline support diagnostics,
source snapshots, exact actor/checkpoint files, figures, result interpretations
and failed postprocessing record are included. Scope and separate byte/interaction
accounting are in `archive/SHARED_WRITE_SNAPSHOT.md` and
`outputs/shared_write_archive_audit/cohort_ledger.json`. All 12,625 older
data/output files were checked unchanged against `5979956` before this extension.
Use the regenerated manifest for current total file count and bytes; earlier
tables remain historical snapshots.

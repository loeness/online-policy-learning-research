# S3: Offline shared-write support diagnostic

This diagnostic measures how the fixed S1b transition grouper behaves as saved probe evidence grows. It reads only `independent/probe_steps.csv` from formal seeds 677, 691, and 709. It does not create or step an environment, run a policy, estimate policy reward, or alter existing S1/S2 outputs.

Run it from the repository root with:

```powershell
.venv/Scripts/python.exe scripts/diagnose_shared_write_support.py `
  --input outputs/knowledge_shared_write_auto_formal `
  --output outputs/knowledge_shared_write_support_diagnostic
```

The output directory must be absent or empty, or contain the matching report previously written by this script for the same input directory. The script writes `summary.json`, per-context decisions in `assignments.csv`, pair-level audits in `pairwise_audit.csv`, explicit tie results in `tied_prototype_checks.json`, the post-assignment local-reference audit in `local_reference_audit.json` and `local_reference_audit.csv`, and `support_coverage_error.png`. The JSON records SHA-256 hashes for all three input CSVs and `configs/shared_write_auto.json`.

## Evidence reconstruction

For each seed and each prefix length 16, 32, 64, 128, 256, and 512, the script rebuilds one `(3, 4, 4)` uint16 transition profile per context from the first prefix rows in that context's saved CSV order. It then creates a fresh `DynamicsGrouper` with the unchanged S1b thresholds and calls `assign(profile)` once per profile in the fixed order `A0, B0, A2, B2, A1, B1`. Each prefix scenario starts from an empty grouper. The grouper receives only the numeric profile; context and family fields are not passed to it.

Each decision records the full overlap/agreement scores, decision reason, rows replayed into that profile, movement transitions counted, supported-cell count and coverage, assignment coverage through that decision, unresolved count, and the numeric aggregate state size. The fixed grouper stores 192 numeric bytes for its two `(3, 4, 4)` uint16 aggregate profiles; this excludes Python object overhead.

The family audit happens after the decisions. It compares all 15 context pairs, counts true positives, true negatives, false positives, and false negatives among pairs where both contexts received a group, and reports unresolved pairs separately. Pairwise accuracy and error use only resolved pairs; resolved-pair coverage is reported beside them. Assignment coverage is the fraction of six contexts assigned to a nonnegative group. Mean supported-cell coverage is reported separately so the two coverage measures are not conflated.

## Action evidence censoring

For a second offline replay, each context's same 512 saved transitions is scanned, but only rows whose recorded action is 0 or 1 contribute to the profile. This is labeled `offline_action_censoring_0_1`. The output records rows scanned, retained profile rows, and the full elapsed source probe length of 512 per context. The chart places this case at source elapsed 512; it does not imply that retaining fewer rows reduced the original interaction cost.

This condition is a data-censoring sensitivity check. It can show whether this grouper returns an insufficient-support, low-overlap, between-threshold, ambiguous, or matched result under that reduced evidence. It does not represent an intervention or a new probe policy.

## Local-context identifiability audit

The censored group decision is made first using only the numeric action 0/1 profile. After all decisions, the diagnostic compares that same profile with all six full profiles from the same seed. It lists every full reference whose direction agrees on the censored query's supported cells, along with overlap, conflicts, and differences hidden in the query's unsupported cells. These context names are observer-oracle references used only for this audit; they are never supplied to `DynamicsGrouper` and never change an assignment.

On these inputs, the action-censored group assignments have 100% pairwise family accuracy, but the audit does not identify every local context. For each A0, A1, or A2 query, the fully covered no-conflict reference set is all three A contexts. Each query's full profile differs from each of the other two A references on four of the six supported action 2/3 cells, which are entirely absent from that query's action 0/1 evidence. Those differences are hidden by the censoring. By contrast, each B query has only itself as a fully covered no-conflict reference, because the saved action 0/1 profiles distinguish those B contexts. This is a per-stream observation, not a general property of the grouper.

## Explicit tied-prototype pressure test

For each seed, the script derives full offline profiles for A0, A1, and A2. It directly constructs a grouper state with full A0 and A1 profiles installed as two separate reference aggregates. It masks the A2 query to cells where all three profiles have a supported and identical direction, then submits that masked numeric profile. This deliberately creates two equally scoring references from observed cells; it does not claim that ordinary assignment would discover A0 and A1 as separate groups. The required outcome is `group = -1` with reason `ambiguous_match`.

The pressure-test profiles are saved evidence. The paired reference-group state is synthetic and manually assembled, not a learned state, not an observed S1b assignment, and not a training result. It checks that a score tie remains unresolved rather than being broken by numeric group order.

## Accounting and limits

Across three seeds, 9,216 previously recorded probe rows are loaded from disk. Prefix scenarios reprocess prefixes from those rows, and censoring, full-reference, and tie checks also reuse them. `summary.json` distinguishes rows loaded from disk, source rows scanned during each offline replay, rows actually used in profiles after action censoring, reused reference rows, and the 0 new environment interactions. These are diagnostic reads, not new interaction counts.

The chart plots assignment coverage and pairwise family error over resolved pairs against the offline source prefix length. It reports means and standard deviations across the three seeds; unresolved pairs remain explicit in the JSON and CSV. Action-censored points use 512 as their source elapsed length.

On the requested formal inputs, mean assigned-context coverage across seeds is 27.8% at 16 rows, 27.8% at 32, 50.0% at 64, 88.9% at 128, and 100% at 256 and 512. At 128 rows, seed 709 assigns all six contexts while the other two each leave one unresolved. Where pairs are resolved, pairwise family error is zero; at 16 and 32 rows only 0–1 of 15 pairs are resolved per seed, so those error rates have little pair coverage behind them. At 256 and 512 rows all 15 pairs are resolved for all seeds, with zero pairwise errors on these saved profiles.

In the action 0/1 censoring replay, 4,512 of the 9,216 saved transitions are retained for profile counts. All six contexts are assigned in each seed, with mean supported-cell coverage of 50%; each seed has 15 resolved pairs and zero pairwise errors. The source probe still elapsed 512 steps per context. In the three explicit tie checks, both prototypes score overlap 6 and agreement 1.0; each query remains unresolved with `-1 / ambiguous_match`. These outcomes describe the supplied saved streams and synthetic tie construction only.

This is a narrow evidence-support and ambiguity audit on these saved streams. It gives no policy-benefit estimate, does not establish that a match is safe for parameter sharing, and does not establish general family discovery or unknown-context discovery. The S1b route and group conditions are fixed. No thresholds are tuned here.

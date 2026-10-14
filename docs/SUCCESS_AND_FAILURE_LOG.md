# Success and failure log

This log separates implementation or validation failures from real method-performance results. Every cohort's per-run files remain in `outputs/`; this summary is not a substitute for those records. There is no C2 cohort in this archive.

## Shared foundation and A/B

**Input validation, not online performance:** The 20,000-transition normal expert dataset and W0 checkpoint pass their recorded source checks. W0 reached 100% success on its source validation episodes and 0% on the recorded unseen CW and CCW validation episodes, establishing the intended adaptation challenge. Hashes are recorded beside the data artifacts.

**Pilot implementation defect:** `outputs/pilot/` contains seeds 11, 23, and 37. Its route learner could treat a mixed old/new history window as a novel context and create a duplicate prototype even when the stable tail matched an existing route. The resulting route competition and return-to-root behavior made this prototype version invalid evidence about the intended router. The original pilot results were preserved. The stable-tail prototype recheck and CW → CCW → CW regression were added before the independent validated seeds 41, 53, and 67 in `outputs/validated/`; the validation cohort is not presented as though the prototype issue had never happened.

**Real A/B performance results:** In the validated cohort, global residual adaptation reached 100% CW success in all three A seeds, while applying that global CW correction in normal reduced source success to 0%. The full-PPO arm had higher seed variation. In B, persistent context residuals helped on revisits relative to clearing those residuals under the tested schedule; this is evidence for persistence in these runs, not proof of general transfer or reduced interaction requirements. All seeds share one W0; three deployment seeds do not estimate pretraining variation.

## B+ controls

**Real negative baseline result:** The analytic history-only controller outperformed the current PPO arms on both global-rotation A and B schedules in the matched experiment while using no gradient updates. That result means the original global-rotation A/B environment does not establish that neural parameter updates are necessary or superior to all temporary adaptation.

**Real short-term history effect:** On the B+ no-gradient controls, retaining learned residual parameters while clearing recent working history continued to perform well on CW/CCW; retaining terminal recent history could initially hurt when it described a different context. The cleared-history branches retain calibrated routing prototypes, so this is not a total-memory-erasure comparison.

**State-reconstruction limitation:** The original A/B terminal archives lack the complete historical RNG state. B+ reload checks validate the stored policy/optimizer/router reconstruction, but these are fresh declared deployment streams, not exact original-run resumes. B+ also records that the first analytic B budget initialization was wrong: 57 completed records were reused and six B analytic/frozen runs were repaired; no completed record was deleted.

## Primary compression and repair

**Real negative compression screen:** No tested Top-K fraction passed the full screen. At 100%, index overhead increased adaptive numeric state by 16.65% versus compact dense. At 75%, the CCW-slip cumulative-return decline exceeded the 5% criterion. At 50%, positive-return-context declines were roughly 14–16%; 25% and 12.5% had larger success and return losses. All fractions were retained in `outputs/compression_attempt/summary.json` and `test/`.

**Separate repair success with additional cost:** The predeclared fixed 50% support was continued for 8,192 more CW → CCW interactions. Its evaluation recovered most of the direct truncation loss and met the report's five-percent average post-repair tolerance, while adding training cost. The repaired result does not retroactively make the primary direct-compression gate pass, and the minimum recovery budget was not measured.

## Compression alternatives

**All nine arms completed:** The comparison keeps all dense and compressed alternatives, three checkpoints, before/after deployments, and online continuation records. Rank-2 factorization and BF16 had low immediate loss on these tested trajectories; rank-2 used 3,280 bytes of warmed adaptive state, BF16 3,128 bytes, and Top-K50 3,648 bytes. Exact action-contrast coordinates were numerically equivalent within the documented probability check but used 4,688 bytes, over the Top-K50 state target. This is an out-of-budget diagnostic, not an invalid run.

**Real negative results:** Centered and activation-weighted Top-K did not improve immediate-loss screening over the comparison they were designed to isolate. The 32-column method had a large direct loss and retained substantial post-repair loss. Individual seed/context losses and the complete resource table are in `outputs/compression_alternatives/结果解读.md`; the one timing sample per checkpoint and method is not evidence of stable acceleration. The compression-alternatives report also records 0 runner errors and 774,144 interactions, not just the better-performing rows.

## C1 validation and formal results

**Invalid prototype validation, excluded from formal learning:** Early `outputs/c1/precheck/` records failed with a calibration-index error. Their policy/trajectory outputs remain in the archive and the 40-entry `outputs/c1/validation_ledger.json`; the run report marks the original incomplete calibration bank invalid and says it was not used by formal training or calibration. One additional transition occurred before an exception and is explicitly counted in the ledger. These exceptions are implementation failures, not evidence that the policy failed to learn the task.

**Real precheck performance failures:** After the routing-index correction, temperature-1 trials still failed learnability on A0 in several tested learning-rate/budget conditions, including 0.003 and 0.02 at 16,384 steps and 0.01 at 32,768 steps. These are genuine negative policy-learning outcomes under those configurations and remain in the ledger. At temperature 2, with learning rate 0.01, the selected precheck passed all six contexts at 8,192 training steps plus 512-step evaluation. The formal comparison was frozen only after that result.

**C1 formal result and its failures:** The 18 formal runs are complete. The low-tier tree mean final success rate is 100.00%; high-tier tree is 99.60%. Across all methods and tiers, 90 of 108 task instances reached the 80% target and 10 of 18 runs passed the target on all six tasks. This is descriptive evidence for learned sharing under the fixed family/tree/band prior. It does not show automatic hierarchy discovery.

The low-tier tree had two old-task comparisons (2 of 12) that fell more than five percentage points immediately after the held-out adaptation; both returned to the five-point retention band after old-task revisits. Across all methods, 11 of 72 old-task comparisons had such an immediate decline; all 11 recovered by final evaluation, but two other comparisons still failed the final five-point retention gate. The largest transient decline was 70 percentage points for high-tier dictionary, seed 97, A1. Those failures and recoveries are listed task-by-task in `outputs/c1/结果解读.md` and machine-readable in `outputs/c1/analysis.json`.

**Interpretation limits:** C1's final and post-history policy evaluations are oracle-route diagnostics. The preceding history continuation hid labels and switch times from the router but used transition prototypes calibrated with labeled online transitions. Do not describe it as from-zero hidden-task discovery. The fixed task family, hierarchy, x-band features, and ranks were supplied in advance; the known-map representability diagnostic reveals favorable alignment between this prior and the task generator. There was no full-capacity consolidation, no safety proof, no statistically powered result, no SOTA comparison, and no test of generalization to real hardware.

## Reproduction and audit gaps to carry forward

- `requirements-lock.txt` is explicitly a Windows x64/Python 3.12 lock record. `requirements.txt` is a set of constraints that can resolve on Linux but is not a hash-pinned cross-platform lock.
- `scripts/run_c1.py --mode precheck` creates the selected learnability gate, not the 40-entry historical `validation_ledger.json` or the offline `representation_diagnostic.json`. Formal execution reads the validation ledger at completion, and the report refuses to render without both archival records. Those originals are present in this archive; fresh validation-ledger and diagnostic generation is not exposed as a standalone checked-in command.
- `scripts/run_compression_repair.py` hard-codes `outputs/compression_attempt` and has no `--output` option. A rerun must use an isolated copy/worktree containing the required primary bundles, or the runner should be parameterized before attempting a fresh-root reproduction.
- `outputs/ab_plus/summary.json` uses reconstructed source checkpoint state and a declared new RNG arrangement because complete original RNG state was not saved. Label it reconstruction and new-stream evaluation, not exact continuation.


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


## C1 supplemental controls and alternatives (2026-10-15)

**Completed experimental inventory:** 57 formal neural runs and 39 validation
runs completed without runtime or capacity failures. Step 1 additionally keeps
12 formal and 4 validation model trajectories, frozen-tree/cleared-window
matching diagnostics and the paired grouped-equivalence check. The declared
non-overlapping research interaction ledger totals 7,205,888 steps, including
post-training source probes. Original A/B/C1 and software integration-test
interactions are excluded from that count. Current software checks: 148 passed.

**Negative result for broad necessity/topology claims:** Without task IDs,
Bayesian assignment controllers reached 100% actual online success in this
synthetic bandwise command-permutation family, versus 10.10% for W0 without
history. They update statistical beliefs even though neural gradient counts
are zero. Tree and the same additive grouped representation were bitwise equal
in paired PPO. C1 cannot claim the tree container itself is uniquely better or
that neural weight updates are necessary for this particular family.

**Weak methods and candidates retained:** Individually selected input-conditioned
full adaptation reached only 34.80% mean final success, EWC 18.27%, behavior
replay 26.84%. Low-tier tree replay also reduced final success to 94.37% from
99.67%. All their validation candidates remain in the archive. The stronger
full context-head architecture reached 96.12% in the separate matched-seed
cohort, and EWC 80.61%; comparing those values with the first full architecture
is not a paired architecture-only ablation. Full context-head seed 181 still
ended A2 at 50%. EWC did improve source-root retention (64.51% to 96.29%) at an
adaptation cost. Custom EWC/PPO replay is not a published SOTA replication.

**Promising bounded representation:** Tree, trainable source-basis tensor, and
hierarchical frozen-basis mixer each achieved 100% mean final success and
3/3 all-six targets at fresh seeds 181/193/211. Hierarchical mixer kept all
immediate old tasks at 100%, with 2,972 bytes of correction/Adam state plus a
6-byte mask. Its complete state was about 149 KB, not a 3 KB full agent. The
independent frozen-basis mixer was less reliable: seed 193 ended A0 at 22.2%,
six-task mean 87.0%. All candidates and this failure cell are retained.

**Reporting defect fixed transparently:** A PyTorch post-hook can fire at both
Adam.step and LeafAdam.step for one actual update. Nine old tree+behavior-replay
records (three validation and six formal) therefore overcounted optimizer
calls by a factor of two. Policy updates were not doubled. Their raw JSON and
checkpoints remain unchanged. Derived reports use the independent single-step
counter preserved inside checkpoints; `optimizer_step_audit.json` contains
both counts. Future code prevents the accounting error, and exact prior source
is archived and hash-verified. The tensor launch document is also archived
before post-run wording corrections about oracle IDs and the trainable critic.

**Limits retained:** One W0, three deployment seeds per comparison, fixed
families and visible-band structure, oracle IDs in main phases, calibrated
bank in hidden history, larger full-reference budgets, and a favorable source
output span. No dynamic tree, from-zero latent-context discovery, full-capacity
replacement, calibrated policy-improvement gate, standard benchmark SOTA claim
or C2/C3 execution. See `C1_SUPPLEMENT_CONCLUSIONS.md` and both full reports.

## P0 pre-core transfer diagnostic (2026-10-15)

The P0 protocol was frozen in commit `e8d715a` before formal results. All 18
independent 4,096-step local PPO branches and 108 source-target evaluation
batches completed. The formal ledger records 73,728 training and 235,623
evaluation interactions, totaling 309,351; the reduced smoke run has 1,772
separate interactions. Six focused pairing/route tests passed. All runs share
one W0, fixed source-output coordinates and experimenter-isolated contexts;
they do not implement automatic discovery or a C2 agent.

**Positive but limited signal:** All 18 local updates improved their own paired
mean episode return. Same-family off-diagonal transfer averaged +0.621 return
per episode; other-family transfer averaged -0.113. Across 90 dependent
off-diagonal cells, 51 had positive observed gains and 39 negative. This is
descriptive evidence for selective reuse within this generator, not a
statistically powered discovery of a universal sharing relation.

**Incomplete learning and weak early signals retained:** Own-context success
rose from a matched zero-correction average of 9.03% to 34.38%, with trained
run success only 12.5%–75%. None reached the old C1 80% reference; this was not
a new P0 acceptance threshold. First-rollout gradient cosine correlated with
finite-update transfer at 0.582, 0.686 and 0.096 for seeds 271/283/307. There
were 16 negative observed transfers among 58 positive-gradient-cosine cells.
Those are noisy observed outcomes, not calibrated false-accept probabilities.
The 4,096-step correction-cosine associations were stronger (0.811, 0.838,
0.698) but required actually obtaining each local update. No real byte-saving
consolidation, online policy-improvement gate, no-label stream discovery, or
independent pretraining checkpoint comparison has been established.

Full artifacts: `outputs/pre_core_p0/结果解读.md`, `analysis.json`, `summary.json`,
paired episode CSVs, all branch checkpoints, gradients and provenance. Old
experiment outputs remain unchanged. Formal CPU time was 126.52 seconds and
wall time 129.90 seconds excluding imports; no CPU/GPU speed comparison is
claimed.

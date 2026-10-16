# Ordered write-decision investigation

Frozen before new interaction-generating runs. This extends the controlled
C1 mechanism experiments; earlier artifacts are immutable. All cohorts use
the same W0 unless an explicit separate foundation is declared.

## Step 1: controlled routing, real reward and benefit evidence

The route identities and bout boundaries are supplied only in this first
isolation. Shared grouping receives transition and conditional-reward evidence,
never task names, preference, family, or evaluation returns. A/B map, reward
sign and task name are observer labels only outside the controlled router.
The fixed stream includes related positive-reward variants and the same
dynamics with negative reward. Probe and evaluation interactions are paid.

Comparisons: independent local PPO with LR 0.004; dynamics-only shared PPO;
reward-aware shared PPO; proposed trial gate, relaxed trial gate, projected
gate, local promotion; and trial rollback. Local LR is 0.003, shared 0.001.
PPO uses three fresh 256-step fit rollouts and one fresh holdout per 1,024
steps. Probe/evaluation/old evidence never become fresh PPO training data.

Trial gates first perform a joint A/C trial. Rejection restores C to its
cycle start and adds the removed C increment to the current A in every band.
Projection analogously compensates the difference. This retains the current
learned policy while controlling persistent effects on other routes. Local
promotion instead freezes C during PPO, proposes a support-weighted part of
the cycle's local increment, and compensates current A on acceptance.

Current compensated policy is algebraically unchanged, so its holdout cannot
prove transfer. Bootstrap evidence evaluates a cold-parent candidate (zero
current A) against the previous parent using the real current holdout's
behavior probabilities. Other affected resident routes use at most 24 saved
transitions each, with observations, actions, advantages, actual behavior
log probabilities and episode IDs. ESS and behavior drift restrict reuse.
The score is a clipped advantage surrogate with empirical episode-cluster
uncertainty, not a policy-value lower bound or a calibrated safety guarantee.
No reset to an old environment supplies decision evidence; new evidence arrives
only through the normal stream. Slot eviction clears its anchors/generation.

All actor Adam state is reset every 1,024 steps across conditions after commit
or rollback/quantization, to avoid stale moments after parameter transfers.
State storage includes all live profiles, group statistics, evidence anchors,
optimizer, RNG and actor in W0 plus one checkpoint <=256 KiB. Temporary model
copies, runtime RSS and serialization/evidence time are counted separately.
Step 1 float32 actor cap 1,024 B is separate from later int8 budgets.

Validation seeds 859/863 evaluate the eight fixed conditions with 2,048 steps
per bout. The best two proposed rules are chosen by mean online reward per
training step; all three simple baselines remain. Formal seeds 1009/1013/1019
are new and use 4,096 steps per bout. No formal ranking changes a rule or
threshold. Negative and null candidates remain archived. Evaluations use
paired complete episodes; negative tasks are scored by signed reward.

## Step 2: automatic grouping and finite local storage

After Step 1 validation, compare selected decisions with reward-aware sharing
and independent caches. Start with known local route IDs but inferred groups,
then remove route IDs and switch hints. Twelve A/B x band x reward-sign tasks
compete for six shared local slots; seven independent slots are permitted by
the common 236 B actor cap (shared 230 B, independent seven 236 B). LRU,
random and resident-frequency replacement are separate controls. The learner
may save only bounded profiles and current residents, never an unbounded
task-keyed database. Any unknown-route actions, delayed recognition, mistaken
writes and partial rollouts are logged and costed.

## Step 3: budgets, stronger controls and shifted generator

Compare actual actor/checkpoint bytes and performance, without padding an
independent arm to pretend equal useful capacity. Use identical full checkpoint
caps and offer extra independent heads where the codec permits. Additional
detour-wall tasks change transitions, reward shaping and exploration demands;
they are a second generator, not a standard-benchmark SOTA reproduction.
Predeclare new seeds/conditions after Step 1 selection and before these runs.
Do not pool float32/int8, known/unknown routing, or different generators.

## Attribution and limitations

Learned conditional rewards can reject S3-style conflicts; a benefit rule must
be compared against this repair, not merely dynamics-only sharing. Shared
proposal gradients/means do not identify an intrinsic common factor because
A+C admits equivalent decompositions. Old evidence is finite and off-policy;
repeated tests and distribution shift prevent safety/monotonicity claims.
Admission/rejection counts, surrogate scores, support, CPU time, online return,
per-task revisit return and all physical bytes are reported even on failure.

## Related algorithmic controls and independent decision audit

The projected proposal uses bounded-reference half-space constraints inspired
by GEM/A-GEM (Chaudhry et al., ICLR 2019, arXiv:1812.00420) and gradient-conflict
methods such as PCGrad (Yu et al., NeurIPS 2020, arXiv:2001.06782). This is a
small adapted proposal filter, not a reproduction or a published-SOTA claim.
For projected_gate, old episodes are divided into disjoint reference and
gate subsets; evidence used to construct a proposal cannot independently
validate it. Unsupported gate evidence rejects the proposal.

An observer-only post-run counterfactual audit uses the first and last nonzero
accepted and rejected proposals in each formal run, deduplicated (at most four
decisions). Each affected resident old route and the current cold parent get
eight paired complete simulator episodes per candidate/incumbent policy.
These reset interactions are separately paid and never fed back into learning
or method selection. A mean old-task return drop below -0.05 labels an audited
harm; a current cold gain above 0.05 with no audited old harm labels an audited
benefit, only when old targets exist. These are descriptive small-sample
diagnostic labels, not ground-truth safety/error-rate guarantees. Unsampled
decisions remain undetermined.

## Repaired validation revision, frozen before its interactions

The first implementation cohort is retained under
`outputs/write_decision_validation/FAILED_COHORT.json`: 16 attempted cells,
9 complete, and 392,671 raw interactions (267,264 training, 22,016 probing,
103,391 evaluation). Three gated conditions encountered a scalar-generation
metadata collision when real old-route anchor arrays were scored. A regression
test now exercises the actual AnchorBank-to-gate path.

That cohort also revealed false reward mismatches: absence of terminal reward
observations yielded a zero fitted terminal coefficient, which the full-vector
cosine treated as a disagreement. The separate supported revision uses
`configs/write_decision_supported.json` and `scripts/run_write_supported.py`.
It compares only identifiable features supported by both profiles, while
requiring an identifiable movement/intercept subspace. Missing terminal evidence
does not verify terminal reward compatibility. The old mode remains available
and unchanged by default. Validation seeds may be reused for this repair;
formal seeds remain unused. Failed/cohort-revision results are not pooled.

The stream runner uses `actor_codec=int8`: candidate validation and the next
deployed actor both use its decoded bounded representation. All arms roundtrip
at commit, including direct joint updates and local-preserving rejections.
Quantization can perturb the otherwise algebraically preserved current policy;
coefficient error is recorded separately from pre-serialization preservation.

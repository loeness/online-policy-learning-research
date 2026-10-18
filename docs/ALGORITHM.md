# Implemented algorithms

## Source content dose and persistence × future PPO (2026-10-18)

`scripts/run_source_dose.py` instruments the unchanged storage schedule after PPO, int8 encoding and actor-Adam clearing. It observes the two old source residuals at their own 1,024-step points, while the actual prefix still learns each old source for 2,048 steps. Both low- and high-dose past-only selectors freeze before future expansion. Future branches differ only in installed old-candidate content; replace/none outcomes are reused only after exact complete post-action learner/simulator hash equality. Observer candidates are absent from learner checkpoints. This is a controlled content intervention, not a shorter-history deployment or an equal-quality guarantee.

`scripts/run_learning_ablation.py` restores the same prefixes and adds fresh replace/none continuations with actor/critic gradient and optimizer steps disabled. The original source-dose PPO outcomes are referenced rather than rerun or charged twice. LRU allocation and persistent-prior inheritance continue, so this is not a static-policy control. Trained rollouts and fit steps cannot increase; collected rollouts are counted separately. Each branch verifies W0, critic and optimizer invariance and exact post-action matching to its PPO reference.

The model remains two 27-coefficient local slots and one 9-coefficient parent, with an 89 B int8 actor and a 262,144 B complete package cap. Neither runner adds automatic grouping, a tree, repeated allocation decisions, or a new retention selector. [Frozen source protocol](SOURCE_DOSE_PROTOCOL.md), [learning interaction protocol](LEARNING_INTERACTION_PROTOCOL.md), and [interpretation](SOURCE_FOLLOWUP_REVIEW.md) define the estimands and boundaries.

`scripts/benchmark_score_features.py` is a checkpoint-only common-subexpression diagnostic. It shares actor features within one candidate/baseline scoring call, adds no persistent cache, and does not implement event-triggered or local structural maintenance. Its cost result is reported separately from policy-learning effects.

## Latest controlled mechanism: one storage decision with continued learning

[STORAGE_DECISION_PROTOCOL.md](STORAGE_DECISION_PROTOCOL.md) freezes the current one-decision implementation in `src/online_rl/storage_decision.py` and `scripts/run_storage_decision.py`. Two local slots contain 27 coefficients each; one parent contains 9. The common int8 actor container is 89 B. An old parent and one temporary new candidate are learned from online PPO. From the same learner/simulator snapshot, keep, replace, mean and none install different content in the single parent position. Resident effective matrices are compensated before encoding; measured encoding drift is retained. Displaced content is absent from the deployed learner backup.

New local allocations inherit the frozen parent only when the known group matches its binding. Local LRU eviction and PPO/critic learning continue for 12,288 future steps. Actor Adam is cleared after each quantized rollout; critic Adam is retained. All new rollout rows enter training. Complete state restoration includes optimizers, RNGs, LRU/evidence state and the physical simulator. This experiment supplies context IDs, boundaries and old/new membership to every arm; it does not implement latent grouping or a dynamic tree.

Past-only frequency, recency and existing historical gain/lower rules are fixed before any future branch runs. Their realized values use the chosen complete learning branch, not a frozen-policy proxy. Full learner packages are capped at 262,144 B; fixed actor shape is not a claim of globally optimal byte allocation. [The results](STORAGE_DECISION_REVIEW.md) support schedule-dependent action effects but do not validate a superior automatic selector. The later source-dose intervention isolates candidate content as described above; repeated adaptive allocation remains unimplemented.

## Current diagnostic: score prediction

[SCORE_CALIBRATION_PROTOCOL.md](SCORE_CALIBRATION_PROTOCOL.md) defines a diagnostic observer around the unchanged retention learner. It freezes all initial-prefix test scores and matrices, captures real pre-update recurrence states, and pays for paired cold-policy evaluations after learning. The observer has no feedback path to the learner. This adds no new policy-selection rule; [the result](SCORE_CALIBRATION_REVIEW.md) tests the existing score's predictive value.

## Latest Stage 2 stream implementation

[RETENTION_STREAM_STAGE2_METHOD.md](RETENTION_STREAM_STAGE2_METHOD.md) describes the current bounded local/parent model, online PPO, fixed prototype archive, pending average candidate, later-rollout retention test, compensated installation and selective prior reuse. The [result](RETENTION_STREAM_STAGE2_REVIEW.md) does not validate the new gate: direct shared PPO and simple mean remain necessary stronger controls. Local context indices and boundaries are provided for this mechanism study; true family labels are not used for grouping.

This document describes the code and fixed protocols in the archived A/B, B+, compression, and C1 cohorts. It does not describe a future C2 design. The environment is a small mechanism study; results do not establish a general continual-learning method, safety guarantee, or state-of-the-art claim.

## Environment and shared foundation

The common navigation task is a 9 × 9 grid with four commands in fixed up/right/down/left order and a 64-step episode limit. The observation is normalized `[x, y, goal_x, goal_y]`; it contains no task/context identifier. Each task changes command-to-movement response. A/B uses normal, clockwise (CW), counterclockwise (CCW), and 12% local-slip variants. C1 has two action-permutation families, each with a separate local swap in one of three x bands. The reward is `0.25 × (previous Manhattan distance − new distance) − 0.03`, plus 1.0 on reaching the goal.

`data/pretrain_source.npz` contains 20,000 recorded normal-context expert transitions, generated by a shortest-path action rule. Behavior cloning uses a whole-episode validation split (episode ID modulo 10), with 1,500 updates. The result `data/foundation.pt` is W0. Its actor is a two-layer 64-unit tanh MLP from the four navigation coordinates. This one checkpoint is shared by all deployment seeds.

In A/B, the critic receives the four coordinates plus an eight-value causal history summary. The policy observation also carries an internally computed route ID. C1's residual learner receives the four coordinates, eight route-summary features, and route ID, while its actor's frozen base feature path remains W0's 64-unit actor. Critic weights are trainable during PPO; the frozen-base claim refers to the actor parameters.

## A/B online learning and routing

The policy starts from W0. It adds a small path-selected residual to the four base action logits. A residual is a 4 × 65 matrix (64 actor features plus bias); it does not copy the full actor. `LeafAdam` ignores an exactly zero gradient for a residual so that an inactive leaf does not advance its Adam moment state.

- **A:** A 16,384-step CW bout. Arms are frozen foundation, full PPO fine-tuning, and a global residual update. The selected residual configuration uses learning rate 0.01 and entropy coefficient 0.06. The configuration was chosen for the residual method and was not an independently optimized full-PPO comparison.
- **B:** 32,768 interactions per run, scheduled in 4,096-step bouts: CW, CCW, CW, CW-slip, normal, CCW-slip, CW, CCW. Switching follows the external interaction counter and can occur within an episode. The arms are frozen, full/global controls where present, history keep, history reset, and oracle keep.
- **History router:** From observed `(state, action, next state)`, it normalizes movement and compares it with W0's normal action response using dot/cross features in a 16-event window. It skips zero-displacement wall collisions, requires five informative events, tracks prototypes, and selects or creates residual routes using fixed novelty and confidence thresholds. The eight-value critic summary contains the recent dot/cross mean and standard deviation plus normalized action counts. A/B's context labels and true mapping do not enter the history route; `oracle_keep` is a diagnostic arm that does use the hidden context.
- **PPO:** SB3 PPO with 256-step rollouts, batch size 64, four epochs, gamma 0.98, GAE lambda 0.95, clip 0.2, target KL 0.06, plus config-specific learning rate and entropy coefficient. The real optimizer update count can vary because KL stopping may end PPO epochs early.

The B router maintains a root path plus residual nodes and can reuse a path on later visits. The original experiment uses the same W0 for A and B; B is not a continuation from A's trained terminal policy.

## B+ controls

B+ starts from the three archived `B_history_keep` terminals (seeds 41, 53, and 67).

- **No-gradient path/history controls:** P+H+ retains residuals and recent history; P+H0 retains residuals but clears current working history; P0H0 zeros residuals and clears current history. All branches keep the learned routing prototypes, reload from a serialized bundle, and run with gradients off across normal, CW, CCW, and both slip contexts. Thus P0H0 is not a reset of every form of route knowledge.
- **Optimizer isolation:** With residuals kept and current history cleared, compare the stored Adam state with a reset Adam state during a CW → CCW 8,192-step continuation. The no-gradient control isolates policy weights from continued optimization; the optimizer comparison tests a combined learning-state intervention.
- **Analytic history-only controller:** A fixed W0 policy is paired with a controller that estimates a global quarter-turn from recent observed action responses (16-event window, at least five moving observations) and maps the intended action back through that estimate. It uses no hidden label and performs no gradient update. A uses CW; B uses the declared 32,768-step schedule. A separate matched-initialization PPO comparison restarts global-residual A and history-keep B from the common W0 and shared initial environment/action RNG arrangement.
- **Checkpoint reconstruction:** The original terminal saves policy, optimizer, and router state but not the full historical RNG state. The B+ bundles therefore reconstruct those learned states and use a declared new RNG arrangement; they are not exact continuations of original RNG streams.

The analytic controller is a strong task-specific baseline for global rotation. It is not a learned policy update and is not a generic POMDP solution.

## Compression cohorts

The primary screen compresses learned residuals from the same B checkpoints. It compares original eight-slot layout, compact dense active residuals, and indexed Top-K representations retaining 100%, 75%, 50%, 25%, or 12.5% of coordinates. The selection rule requires at least 10% adaptive numeric-state savings, smaller whole bundles, no more than five percentage points success drop per context mean, and no more than 5% cumulative-return drop on positive-return contexts. The primary screen did not select a winner. A separately predeclared fixed 50% arm then received 8,192 additional CW → CCW steps; this repair is not evidence that direct compression passed.

The alternative cohort keeps the three existing B checkpoints and evaluates nine predeclared residual representations: compact dense, original Top-K50, exact three-dimensional action-contrast coordinates, centered Top-K50, activation-weighted Top-K50, 32 retained feature columns plus bias, rank-2 SVD, rank-2 SVD plus fixed 500-batch teacher KL distillation, and BF16 values/Adam moments. Every method uses the same fresh Adam state for the 8,192-step online continuation. Before and after deployments cover five contexts with new test RNG offsets. These methods compare representations of known A/B residuals; they do not test new-context discovery, shared-parent consolidation, or full-capacity knowledge integration.

Persistent tensor/bundle bytes, rollout buffers, temporary construction workspaces, wall time, and sampled process RSS are reported separately. A serialized-bundle or adaptive-state reduction is not evidence of a matching RAM reduction or inference speedup.

## C1 fixed-structure shared residuals

C1 creates six related tasks (`A0`–`A2`, `B0`–`B2`) on the same 9 × 9 navigation interface. The task-specific action map first applies a family permutation and then a local swap in one of the visible x bands. The fixed task tree, family grouping, and x-band feature gate are part of the experimental prior.

### Policy parameterization

W0's actor and action head are copied and frozen. For actor feature vector `h ∈ R^64`, the base feature with bias is `φ = [h; 1] ∈ R^65`. The visible x coordinate selects one of three bands; one-hot gating creates a leaf feature in `R^195`. The four residual action logits are constrained to the fixed orthonormal `4 × 3` action-contrast basis `H`, removing the softmax-invariant common-logit direction. Base logits plus residual correction are divided by temperature 2.0; the resulting categorical policy is mixed with 5% uniform action probability. PPO rollout and loss evaluation use this same distribution.

The three compared layouts are:

| Layout | Implemented residual |
|---|---|
| Flat | Six endpoint leaves; each is a rank-1 (low tier) or rank-2 (high tier) factorization over the common 195-dimensional band-gated feature. |
| Tree | Two learned family parents (IDs 7 and 8), each rank 1 over global `R^65` features, plus six endpoint-local leaves over `R^195`, rank 1 or 2 by tier. The family parent and local leaf corrections add. |
| Dictionary | Six endpoint coefficient matrices over a shared learned dictionary; K = 6 in low tier and K = 12 in high tier, with the same local feature width. This is shared basis storage, not a learned tree. |

The tree leaf factorization is `U @ V`, then projected through `H`. A new tree leaf starts with zero `V` and nonzero `U`, so `V` has a learning signal while the initial output remains zero. `LeafAdam` sets exactly zero residual gradients to `None`, avoiding moment updates for inactive residual factors. Flat held-out leaves initialize from the rank-limited SVD of the two same-family seen siblings' mean; dictionary leaves initialize from the same-family coefficient mean; tree held-out leaves start at zero and inherit the already learned family parent. These initializations and the held-out setup are explicit implementation choices.

### C1 learning, evaluation, and router information

The selected precheck trains a dense residual learner from W0 on each task separately for up to 16,384 interactions (8,192 initial plus an 8,192 extension if required), then tests 512 steps. All six tasks must reach 80% complete-episode success before formal ranking. The selected temperature-2 precheck passes. It is independent of the formal training trajectories.

Each formal run uses one deployment seed and one of the two capacity tiers. It trains the seen sequence `A0, B0, A1, B1, A0, B0, A1, B1` for 4,096 steps per bout; tests A2/B2 cold with foundation and shared/sibling route; initializes the held-out residuals and trains A2/B2 for 8,192 steps each; evaluates all six tasks; revisits A0/B0/A1/B1 for 2,048 steps each; evaluates all six tasks again; then saves/restores the bundle, clears recent router history, and runs a six-context hidden-label continuation for 1,024 steps per context.

During `seen`, `heldout`, and `revisit`, the runner knows the context ID and calls the router's labeled calibration method on real transitions. That populates empirical movement/outcome banks for all six profiles. During the final `history` continuation, context labels and switch times are hidden from the router; it receives only the recent `(state, action, next state)` sequence and classifies against the already labeled calibration bank, with a fixed 32-event window and 0.8 profile/family confidence threshold. Calibration and history evidence are different: this is **hidden-label deployment continuation with labeled calibration**, not from-zero unsupervised task discovery. The family tree and profiles are fixed in advance, and confidence is not a policy-improvement safety gate.

`final_eval` occurs after oracle-routed `revisit` and uses the known task route as an oracle diagnostic. The later `history_eval` occurs after the hidden-label `history` continuation, but it also evaluates each task with its known task route. It reports the post-history policy under oracle routing; it does not score the hidden-label route decisions made during the history phase. Read `history` route accuracy/reward and `history_eval` policy performance as separate outcomes.

### Capacity accounting

Before formal results, the run measures fully warmed policy + Adam + router + fixed rollout state for each method and adds a 16 KiB serialization safety margin before rounding upward to 4 KiB. The strict adaptive residual + Adam caps are 15,952 bytes (low) and 30,700 bytes (high); the complete serialized-agent-plus-rollout caps are 188,416 bytes and 204,800 bytes. The high-tier dictionary's 30,700-byte adaptive cap is the tier maximum, but the flat/tree warm-up values can differ. C1 separately reports actual tensor state, complete bundles, rollout arrays, explicit workspace lower bounds, and sampled process RSS. RSS is not the cap metric and is not a continuous peak measurement.


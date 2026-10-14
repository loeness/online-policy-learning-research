# C1 supplement: frozen protocol

Protocol written before supplemental outcome inspection. Original `outputs/c1` and the v0.1.0 archive remain immutable. Configuration: `configs/c1_supplement.json`.

## Order and questions

1. State-dependent online action models, then a matched short hidden-context deployment diagnostic. Window counts are the simple baseline; Bayesian bijection constraints and surprise detection are predeclared alternatives. These use observed transitions, source data and visible geometry, without task IDs or switch clocks. They do not establish a need for neural gradient updates.
2. Non-tree grouping. A flat grouped additive representation is mathematically equivalent to the existing tree and is checked by exact logits, gradients, Adam states, and a paired real PPO rollout. A grouped dictionary tests whether separating unrelated families improves the existing global dictionary.
3. Individually tuned compressed policies, a full shared online actor, EWC, and bounded behavior replay. Tree + replay is an alternative for the original transient forgetting. EWC follows the diagonal Fisher quadratic penalty adapted to task-free PPO. Replay is CLEAR-inspired PPO + behavior cloning, not a faithful CLEAR reproduction. Full tuning uses a shared actor with a zero-initialized 6 x 64 route bias; it does not save six full networks.

## Data and evaluation

All methods start from the same archived W0 and source dataset. Validation seed 911 is disjoint from original and supplemental formal seeds 131, 149, 167. Formal schedules, evaluation lengths and random seeds are fixed. Known family assignments and local x-band features are favorable priors common to the compressed neural controls. Oracle training and no-gradient oracle diagnostic success are reported separately from the actual hidden-router stream reward and success. Neural history inference retains an oracle-calibrated bank from its own prior interactions; the bank is frozen during hidden deployment. This is not unknown-task discovery.

Each compressed method receives three learning-rate/entropy validation candidates at high capacity. The selected setting is then used at both capacities, a declared limited tuning budget rather than a claim of global optimization. Full tuning receives three candidates. Full + EWC and full + replay use the selected full setting and three coefficient candidates each. Tree + replay uses the selected tree setting and three coefficients. Selection is the weighted success score in the configuration. All candidates, errors, failed capacity checks and negative results remain recorded. Settings are frozen before the corresponding new formal seeds. Any additional alternative must be validated first and use unused formal seeds.

## Capacity and cost

Compressed adaptive parameter + Adam caps and complete serialized-agent + rollout caps remain those of original C1. Behavior replay retains 128 observations and pre-action behavior probability vectors (8,704 array bytes), counted in persistent state. Replay and EWC observe only real collected transitions; source Fisher uses 64 source examples and is offline compute, not free uncounted learning. Full actor adaptive state exceeds both original adaptive caps, so full methods are explicitly outside-budget Pareto references. Their full state, Fisher, optimizer, rollout and replay must fit the separately declared 393,216-byte complete-agent cap; no equal-capacity claim is made for them.

Report actual cumulative rewards, per-context success, cold transfer, immediate retention, recovery, actual hidden-stream performance, interactions, gradient/statistical updates, wall and process CPU time, serialized state, rollout, explicit live arrays/tensors, and sampled process RSS. Evaluations are additional environment interactions and logged separately. Model controllers count statistical updates even though their neural gradient count is zero. Cross-protocol comparisons disclose different information and geometry priors.

## Acceptance and limits

Do not require the tree to win. A functionally equivalent flat representation means existing results support shared factorization, not a unique topology advantage. A model controller outperforming the tree weakens the claim that this benchmark requires persistent policy parameter learning. A replay or EWC failure is retained and cannot be hidden by reporting only final revisit success. Three formal seeds and one pretrained foundation support exploratory mechanism evidence, not benchmark SOTA or publication readiness. No dynamic branching, replacement, C2, or C3 is introduced.

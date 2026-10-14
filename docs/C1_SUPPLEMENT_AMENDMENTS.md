# Supplemental protocol clarifications before Step 3

Steps 1 and 2 are complete before this amendment. No full fine-tuning validation or formal outcome has been inspected. Formal neural controls keep the frozen seed and schedule protocol.

## Full actor input parity

Compressed controls explicitly receive the visible three x-band partition through gated actor features. To avoid withholding that prior from the full actor, its first hidden preactivation now also receives a zero-initialized 3 x 64 band bias, alongside the existing 6 x 64 route bias. These are equivalent to additional one-hot input columns. At initialization all biases are zero, so W0 and its logits are unchanged. All shared actor and critic parameters are trainable; no full network is stored per task. The extra band weights and Adam moments are counted. The full reference remains outside both original adaptive caps and uses the same declared 393,216-byte complete-state cap.

The source Fisher and online Fisher include band bias parameters. Fisher is evaluated on already-collected observation/action pairs at the post-update policy, with its reference anchored at the same parameters. This is an empirical Fisher under collected actions, not a fresh exact expectation under that policy. Source Fisher is initialized once; it and the global consolidation clock survive restoration.

## Evaluation clarification

All neural stages use the same initial environment/reset RNG and action-generator seeds across checkpoints. The deterministic C1 environment draws reset randomness only on reset, so episode-index start/goal sequences are coupled; different policies nevertheless consume different-length prefixes in the fixed interaction budget. This does not imply identical observations at every time step. Cold diagnostics include the pristine W0, the seen-family fallback, and the actual initialized held-out path: six cold streams per run. Evaluation totals explicitly include these additional diagnostics. No evaluation observations are used to train or select a formal policy. A final source-task root-route probe is planned after training, solely to assess retention of pretrained behavior, and is not a tuning endpoint.

## Additional representation alternative, still C1

The small grouped dictionary showed variable performance despite individually chosen settings. A separate predeclared follow-up will test a feature-mode tensor dictionary and a frozen source-output basis. They share a 3 x 65 feature basis across the visible bands, with small context/band coefficients, instead of flattening bands into 195 feature columns. This tests whether a better representation can use fewer retained parameters. The source basis is derived only from W0's centered action head. No simulator transition map is used. Fixed routing/group priors remain, so this does not introduce C2 or discover a tree.

Additional validation seed: 919. Additional formal seeds: 181, 193, 211. Each of three new representations (learned feature basis, frozen feature basis, frozen basis with shared family coefficients) receives the same three learning-rate/entropy candidates. A tree reference uses the previously frozen setting. Original and first supplemental formal outcomes remain intact and are not mixed as paired samples with these new seeds.

## Stronger full-update conditioning alternative

The first input-conditioned full actor and its EWC/replay controls performed poorly in their limited search. A separate architecture control adds six small zero-start 3 x 195 contrast output heads to one fully trainable shared actor and critic, with the same visible-band features. It stores no separate full backbone per context. Ordinary PPO and the same task-free EWC are compared. The source is still exact W0 at initialization. Complete state, all head and backbone Adam state, and EWC importance/anchors are counted under the existing 393,216-byte cap.

Its independent validation seed is 923, with three learning rates (0.001, 0.003, 0.01; entropy 0.06), followed by three EWC coefficients on the selected ordinary setting. Formal seeds are 181, 193, 211, unused before this amendment. This preserves the first full-update failures and avoids treating a weak conditioning architecture as the best possible full-update reference. It remains a custom method-level control, not a faithful reproduction of a benchmark SOTA.

# Selected-feedback retention method

This note records the implemented method for the selected-feedback and repeated-retention study. The frozen protocol contains the full schedule, cohort seeds, decision gates, and reporting rules; this note summarizes the learner and its boundaries for method review. See the [protocol](RETENTION_VALUE_STREAM_PROTOCOL.md), including its [pre-main fidelity clarification](RETENTION_VALUE_STREAM_PROTOCOL.md#pre-main-fidelity-clarification), the [configuration](../configs/retention_value_stream.json), and the [runner](../scripts/run_retention_value_stream.py).

## Learner and stream

Each stream starts from the existing pretrained foundation and continues one PPO learner, one simulator instance, and one two-slot task cache across task switches. The environment resets only at episode boundaries, not when the task changes. The actor has two local slots with 27 coefficients each and one shared parent with 9 coefficients. It uses 256-step PPO rollouts, 512-step bouts, local learning rate 0.003, critic learning rate 0.003, and a common 89-byte int8 actor container. The shared parent is frozen at learning rate 0 for discrete-retention arms; only direct shared PPO unfreezes it at 0.003. The complete persistent learner package is capped at 262144 bytes, counting the foundation, serialized learner state, and actor sidecar. See the [retention state and action implementation](../src/online_rl/retention_stream_value.py), [selected-value model](../src/online_rl/retention_value.py), and the [shared PPO runner path](../scripts/run_storage_decision.py).

Warmup trains group 0 variants 0 and 1 for 1024 steps each. For arms with a parent, their mean effective policy initializes a group-0 parent, installed with residual compensation. Group 1 variants 0 and 1 then receive 512 steps each. This is 3072 real warmup steps. Subsequent candidates are transient means of two *currently resident* generations from the same logical group and different variants; each source policy is evaluated as its effective coefficients, including a bound parent. Each source must contribute at least 512 actually fitted steps in its current generation. No extra candidate-training trajectories or persistent candidate bank are used. The runner checks task, slot, generation, group, and fitted-step provenance before forming each candidate.

After each selected storage operation, the stream runs three consecutive 512-step bouts, with no intervening operation. One visits the minority group and two visit the dominant group. Dominance changes every eight decision blocks; seed parity sets the initial dominant group. The next candidate uses the last two bouts, so fitting is charged to the ordinary stream. The schedule is generated only after selection; selectors receive observed past task/group history and bounded evidence, not the future schedule. Task IDs, logical groups, and bout boundaries are known inputs. Unknown-context routing is outside this experiment.

## Retention operations

The four discrete actions are:

- **Keep:** preserve the current shared parent and its binding.
- **Replace:** install the candidate as the parent for its source group.
- **Mean:** install the equal mean of the incumbent parent and candidate as a wildcard parent covering both groups. If no incumbent is bound, the incumbent matrix is zero.
- **None:** install a zero parent and leave it unbound.

Parent changes compensate resident local residuals so their effective policies are preserved before codec quantization; the runner records post-codec drift. Candidate coefficients are released before the outcome horizon. The generalized lower selector uses the frozen surrogate gate (12 minimum rows, 3 blocks, ESS ratio 0.2, mean KL at most 0.5, clipping ratio 5, and one standard-error multiplier). It compares each group’s currently applicable incumbent against the candidate only in the candidate’s group, weights lower estimates by recent group frequency, and replaces only when every required estimate is finite and the weighted value is positive. It is an extension for repeated operations, not the same rule as the earlier one-write confirmation.

The comparison includes learned linear and learned intercept selectors, uniform random, fixed keep/replace/mean, frequency, generalized lower, direct shared PPO, and independent local learning. Learned methods explore uniformly for their first 16 decisions and then use epsilon 0.1; their exploration RNG streams are matched. Direct shared PPO keeps a wildcard parent and trains it online with learning rate 0.003. Independent learning reserves the same actor container but keeps the parent zero and every local slot unbound; it is a same-layout control, not an optimal reallocation of the reserved bytes.

The direct shared PPO arm uses this study’s common PPO harness. After every 256-row rollout, the harness codec-round-trips the actor and clears local/shared actor Adam moments. Actor parameters and the critic optimizer state continue, and global update counters continue. This limitation applies to every arm, including direct shared PPO; the baseline is not an untouched external PPO or SOTA reproduction. The [pre-main clarification](RETENTION_VALUE_STREAM_PROTOCOL.md#pre-main-fidelity-clarification) records this implementation detail.

## Features and selected feedback

The learned linear selector observes six pre-action features; the intercept model observes only the bias. Each model maintains a separate discounted ridge estimate for each of the four actions, with ridge 1 and discount 0.97. The linear features are:

1. Bias, equal to 1.
2. Candidate-group frequency minus incumbent-covered frequency in the trailing eight bouts. A wildcard covers both groups; an unbound parent covers neither.
3. Candidate-group mean first-256-step reward EWMA minus the mean for the group or groups covered by the incumbent, divided by 0.02 and clipped to [-2, 2]. The EWMA uses alpha 0.25; missing evidence contributes zero.
4. The difference between `log1p` candidate mean fitted steps and `log1p` parent fitted steps, divided by `log(8193)` and clipped to [-1, 1].
5. Decisions since the parent was written, divided by 16 and clipped to [0, 2].
6. Fraction of the trailing eight bouts that caused an eviction.

The selected action is logged before its outcome begins. Its only new target arrives after the full 1536-step, three-bout block: block reward per step minus measured decision-and-block CPU seconds per step (`lambda_cpu = 1`). The measured decision-and-block cost covers feature calculation, inference, selection, installation, and block execution; the target excludes its own later value-model update and package serialization, which are recorded separately. Both shadow models receive the same selected target and update only the selected action. Unchosen actions and branch labels are never used as feedback. There is at most one pending selected decision, included in the in-flight package measurement before its outcome; saved decision-boundary checkpoints have no pending decision. Environment state and raw traces remain external to the learner package, so arbitrary-step environment resumption is not claimed.

The bounded learner state stores six-task visit counts, mean first-256-step reward EWMA and eviction counts, the trailing eight group/eviction indicators, parent source-fit and age metadata, both ridge models, selector RNG state, and any pending decision. It is serialized with the learner under the package cap. Experiment logs may retain coefficient diagnostics as evidence, but those traces are not read by the selector and are not deployment-agent memory.

## Interpretation and status

The learned model predicts the factual outcome of a selected action. It does not estimate each write’s causal benefit: outcomes from unselected branches are missing, and one selected block can change later policy behavior and task recurrence. Recovery after eviction is a factual whole-policy observation, not a counterfactual recovery loss. No safety guarantee, unknown-context claim, or tree/DAG expansion claim follows from this method.

The completed [independent surrogate-lower report](../outputs/lower_confirmation_report/report.md) is linked as prior evidence; this method note does not restate or reinterpret its conclusion. The selected-feedback calibration plumbing pilot completed one four-decision stream with 9216 environment steps and no failures; it validates plumbing and accounting only. Its rewards are not efficacy evidence. The calibration main cohort and all 80 comparison streams are complete. The [ordered evidence review](ORDERED_RETENTION_VALIDATION_REVIEW.md) separates the conditional prediction result from the unsupported primary policy increment, records independent audits, and links all ten comparison methods and complete costs. This method has not established a storage-policy advantage over direct shared PPO.

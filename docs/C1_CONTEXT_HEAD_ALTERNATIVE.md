# C1 shared actor with compact context heads

This alternative adds endpoint-specific output heads to the full shared actor
baseline. It is intended to test whether the weak route-and-band input
conditioning control was limited by its way of expressing endpoint behavior.
It is a distinct architecture from the flat and dictionary residual controls,
and it does not replace or revise their recorded results.

## Policy structure

The policy copies W0 once and keeps one trainable actor and critic. It has the
same zero-initialized 6 x 64 endpoint route bias and 3 x 64 visible-band bias
as the current full control. It adds six zero-initialized context heads, each
with a 3 x 195 matrix. Each head maps the shared 64-value actor latent plus a
constant, gated by the three visible x bands, into the fixed four-action
contrast basis. The total head parameter storage is 14,040 bytes. Head
outputs, route biases, and band biases all start at zero, so every route and
band has the exact W0 action distribution at initialization.

This policy keeps a common actor representation and six small output
functions. It does not retain six full actor networks. Its actor, critic,
endpoint route biases, visible-band biases, and context heads remain
trainable. EWC's actor parameter set includes every context head.

The initial known endpoints are A0, A1, B0, and B1. Before training A2 or B2,
its route bias and context head are initialized from known siblings; the route
is then marked known. Parent routes use the detached mean of all known family
heads. A parent-route sample can update the shared actor and route bias, but
does not update endpoint heads through that fallback. Endpoint routes update
their selected head.

## Validation and formal runs

The runner reuses the frozen C1 schedules, oracle-calibrated history router,
no-gradient oracle evaluation, capacity sampler, and checkpoint lifecycle.
It validates three learning-rate candidates on seed 923 and selects with the
predeclared weighted score. It then validates EWC strengths 10, 100, and
1,000 using that selected learning rate. Only after both selections are saved
does it start formal seeds 181, 193, and 211 for the ordinary policy and the
selected EWC setting.

Each formal arm starts from W0. Ordinary PPO uses no source examples after
initialization. The EWC arm uses 64 existing source observations once for its
initial Fisher estimate, then consolidates per-example score-gradient squares
from actual rollout data every 2,048 global steps. The actual hidden-router
continuation is reported separately from oracle evaluation. The transition
model bank is calibrated only during the declared oracle training schedule
and remains frozen during hidden continuation.

The runner writes under `outputs/c1_supplement/context_head_alternatives/`.
It records each validation result before selection and each formal result
individually, so a capacity failure remains visible. It restores the named
Adam state, route table, endpoint known-mask, EWC/Fisher state, and RNG
metadata from the revisit checkpoint before hidden continuation.

## Results

All six validation runs and all six formal runs completed. The frozen
ordinary-PPO setting was learning rate 0.003 and entropy coefficient 0.06
(validation score 0.884). The EWC setting used the same PPO parameters and
lambda 10 (validation score 0.695). Scores use 0.50 final mean success, 0.25
immediate old-task mean success, and 0.25 immediate heldout mean success.

| Control | Weighted score | Final mean | Immediate old | Immediate heldout | Hidden history | Peak complete agent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Context heads, ordinary PPO | 0.905 ± 0.056 | 0.961 ± 0.046 | 0.794 ± 0.031 | 0.902 ± 0.108 | 0.975 ± 0.025 | 236,589 bytes |
| Context heads, EWC λ = 10 | 0.804 ± 0.099 | 0.806 ± 0.108 | 0.704 ± 0.076 | 0.898 ± 0.121 | 0.908 ± 0.046 | 316,003 bytes |

Values are mean ± population standard deviation over seeds 181, 193, and 211.
The ordinary control scored higher than EWC in this cohort; the result does
not establish that EWC is generally harmful. Each arm used 63,488 training
interactions and 17,408 evaluation interactions per seed. Ordinary PPO used
no offline source observations after W0 initialization. EWC used 64 source
observations once for the initial Fisher estimate and then made 31 task-free
consolidations from rollout observations.

Both controls stayed below the frozen 393,216-byte complete-agent cap, and
serialized restoration was verified for all six formal runs. Their adaptive
parameter-and-optimizer state exceeds the original compressed-policy adaptive
caps, so these remain full-policy Pareto references rather than equal-budget
comparisons. Per-run metrics and capacity ledgers are in
`outputs/c1_supplement/context_head_alternatives/`; the execution log is
`outputs/c1_supplement/context_head_alternatives/execution.log`.

## Capacity and interpretation

The policy is outside the original C1 adaptive parameter and Adam caps. It is
reported as a full shared-actor Pareto reference and must fit the frozen
393,216-byte complete-agent cap, including optimizer, EWC, route data, rollout
storage, and explicit live state. A failed cap check is a failed result; the
cap is not relaxed.

An oracle residual-policy-to-context-head transfer appears only in a unit
test. Training does not receive hidden environment mappings or counterfactual
labels. This is a bounded architecture alternative with one pretrained
foundation and three formal seeds, not a benchmark ranking or a faithful
reproduction of a published method.

# C1 tensor source-basis alternatives

This follow-up compares three compact C1 residual representations under the
amended supplement protocol. All keep the pretrained W0 actor and critic
source common and use fixed C1 family membership and visible x-band gating.
During the seen, held-out, and revisit phases, every neural control receives
the same oracle endpoint IDs. During hidden-history training, every control
receives inputs from the shared calibrated router. These representations do
not read hidden environment labels directly, use transition maps, or perform
C2 discovery.

## Representations

Each policy derives one `3 × 65` source basis from the centered W0 action head:

`V = Bᵀ [W0 | b0]`,

where `B` is the fixed orthonormal four-action contrast basis. The three 65-wide
feature slices are selected by the visible x-band gate. The flat variants give
each endpoint coefficients `A` of shape `3 × 3 × 3`; for band `b`, its
contrast-space residual is `A[b] V`. The three band blocks are concatenated
into the existing 195-wide local feature representation and projected back to
four action logits. All coefficients start at zero, so the initial action
distribution matches W0.

`hierarchical_output_mixer` retains one family coefficient matrix `C` of shape
`3 × 3` for each fixed family, plus the same per-endpoint band coefficients.
For endpoint `i` and band `b`, the reduced residual is `(C_family(i) + A_i[b])
V`. A family route uses only its `C` matrix. Held-out endpoints keep the
family prior and initialize their local `A` to zero.

The source basis spans any action-row permutation of W0 in contrast space.
That is a structural property checked with known permutations in unit tests;
no context permutation or transition map is used to initialize or train a
policy.

| Method | Shared source basis | Trainable residual values | Residual + Adam bytes | Seen mask | Warmed adaptive bytes* |
| --- | --- | ---: | ---: | ---: | ---: |
| `tensor_learned` | Initialized from W0 and trained | 357 floats | 4,312 | 6 bytes | 4,318 |
| `output_mixer` | Initialized from W0 and frozen | 162 floats | 2,748 | 6 bytes | 2,754 |
| `hierarchical_output_mixer` | Initialized from W0 and frozen | 180 floats | 2,972 | 6 bytes | 2,978 |

*Includes all residual values, Adam's two moment tensors and per-parameter step
scalar for each trainable residual tensor, and the six-byte seen mask. It
excludes the shared frozen actor/critic and live PPO/router state, which the
runner accounts for separately. The source basis is counted in residual values
for both frozen-basis methods; neither has optimizer state for that basis.

The 195-float source basis is stored in each method (780 bytes). The flat
output mixer has 162 endpoint coefficient floats; the hierarchical variant
adds 18 family coefficient floats. The seen mask begins with endpoints 1, 2,
4, and 5. The flat variants initialize a held-out endpoint from the mean of
currently seen siblings and mark it seen. The hierarchical variant retains
the shared family coefficient and starts the held-out endpoint's local
coefficient at zero. Flat family-route fallback averages all endpoints marked
seen; hierarchical family-route fallback uses its family coefficient only.

## Frozen run protocol

The separate runner uses the supplement validation and formal schedules and
candidate grid. It substitutes validation seed 919 and formal seeds 181, 193,
and 211. Each of the three tensor methods is validated at the low tier against
the three predeclared learning-rate / entropy candidates (nine validations in
total). The low-tier tree reference uses the selected Step 2 tree parameters already saved in
`outputs/c1_supplement/step2_grouping/tree/selection.json`; the runner does not
retune it. The three tensor methods plus the tree reference produce twelve
formal low-tier runs if validation selects parameters for every alternative.
All arms use the generic supplement runner for PPO, routing, evaluation,
capacity accounting, checkpoint restoration, and raw traces. The oracle IDs
during seen, held-out, and revisit phases are the same across all four formal
arms; only hidden-history training relies on calibrated router inputs.

Run only after the ordered Step 3 job has completed and the owner authorizes
the follow-up:

```powershell
.venv\Scripts\python.exe scripts\run_c1_tensor_alternatives.py
```

Results are written under `outputs/c1_supplement/representation_alternatives/`.
Each arm records its config, phase progress, raw stream/evaluation outputs,
checkpoints, result JSON, and failures. The final frozen settings file adds
hashes for the tensor control, runner, launcher, and protocol amendment.

## Restore and accounting

Tensor checkpoints use the packed residual payload format with an explicit
`c1_tensor_source_basis_v1` layout marker. The source basis trainability flag,
seen mask, all tensors, and named `LeafAdam` state are restored from the
checkpoint. `output_mixer` has no Adam state for its frozen basis. The original
low adaptive cap of 15,952 bytes and low complete-agent cap of 188,416 bytes
remain enforced by the generic supplement capacity callback.

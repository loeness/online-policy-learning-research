# Why the source-output basis can represent C1 action permutations

This is a representability observation about our C1 control. It is not the
supervisor's private proof, an online-learning lower bound, or a performance
guarantee. Known permutations are used only in this derivation and unit tests;
the experimental learner receives no permutation matrix for initialization or
updates.

## Policy representation

Let the frozen source actor produce features `phi(s)` in R^65 (64 hidden
features plus bias). Its four logits are

`l0(s) = W phi(s)`, with `W` of shape 4 x 65.

Let `H` be the fixed 4 x 3 orthonormal action-contrast basis:

`H^T H = I3`, `H H^T = I4 - (1/4) 1 1^T`, `H^T 1 = 0`.

Store `V = H^T W`, a 3 x 65 source-output basis. For a visible band `b`, the
frozen output mixer learns a small 3 x 3 matrix `R_b` and uses

`l(s) = l0(s) + H R_b V phi(s)`.

These are policy coefficients updated by deployment rewards. They persist
in checkpoints even if the recent working-history window is cleared. No
complete actor is saved per context.

## Representing a permutation

For any four-action permutation matrix `P`, both `P 1 = 1` and
`1^T P = 1^T`. Decompose the source logits as

`l0 = H V phi + c(s) 1`, where `c(s) = (1/4) 1^T W phi`.

Choose

`R = H^T P H - I3`.

Then

`l0 + H R V phi = H H^T P H V phi + c(s) 1 = P l0`.

The equality holds because `P H V phi` is centered, so projection by `H H^T`
leaves it unchanged. Consequently, one 3 x 3 coefficient matrix can represent
any command permutation of the W0 output distribution. Temperature scaling
and mixing with a uniform epsilon-exploration distribution also commute with
permutations. Within C1, separate band coefficients can represent different
permutations in the three bands. Learned family coefficients plus leaf
coefficients can represent the same result by their sum.

## What this explains, and what it does not

The construction explains why a very small frozen-basis control is expressive
enough for this particular task family. Expressivity does not imply that PPO
will find those coefficients: exploration, feature scaling, optimization and
sharing interference still determine learning. All poor validation and formal
results must be retained even though a representation exists.

The construction also exposes a favorable task prior. Each correction is a
linear transform of the three source action-contrast logits; a task requiring
new information absent from that source output span need not be expressible.
The learned-basis variant can move the retained feature basis, but still keeps
the frozen actor and a low-dimensional linear correction. Neither variant
establishes general new-skill learning or unlimited capacity.

The original C1 action-contrast controls already remove the redundant common
logit direction. The additional saving here comes from sharing a feature basis
across bands and constraining it to W0's output span, not merely from going
from four logits to three contrasts. Frozen-basis coefficient adapters have
close methodological precedents, including [LoRA-XS](https://arxiv.org/abs/2405.17604).
The current diagnostic therefore provides an implementation and an explicit
scope condition, rather than a novel general adapter theorem.

Finally, a saved small coefficient block per endpoint still grows with the
number of endpoints unless a strict node/byte cap and consolidation policy are
enforced. C1 allocates six endpoints in advance. Discovery, promotion, merging,
replacement and a growing stream under a fixed persistent budget remain
research questions; this derivation does not solve them.

# Fixed C1 tree and flat grouped additive equivalence

The online parity test is complete. It ran 512 seen steps, initialized held-out endpoints from siblings, then ran 512 continuation steps. At both capacity tiers, tree and flat grouped policies had exactly equal observations, sampled actions, logits, action probabilities, final policy tensors and named Adam state. Maximum logits and probability differences were zero. Four actual policy trajectories consumed 4,096 interactions in total; these are a separate implementation/mechanism check, not additional formal seeds.

Evidence: `outputs/c1_supplement/step2_grouping/equivalence/ppo_parity.json`; runnable check: `tests/test_c1_grouped_equivalence_online.py` with `C1_RUN_GROUPED_PPO_PARITY=1`.

For a leaf c in family g, both representations implement the same correction:

\[
\Delta\ell_c(s)=H\left(U_g V_g\phi(s)+U_cV_c\psi(s)\right).
\]

The flat implementation selects g with a lookup table and holds no path tuples. This preserves shared and specialized parameters, their initialization, gradient flow and optimizer semantics. Changing the container alone is not a new statistical algorithm. Complete serialized metadata can differ, although retained parameter and optimizer tensor bytes agree.

The result narrows the original fixed C1 claim: its controlled advantage concerns the shared factorization and its inductive prior. C1 has not demonstrated a unique advantage from tree topology. The long-term research question remains how online experience discovers and consolidates bounded reusable knowledge; an equivalent flat representation does not answer or invalidate that question.

# C1 modular controls

These controls use the same frozen C1 actor, trainable critic, 64 actor features, four-action contrast basis, three observed-x bands, temperature, and epsilon mixture as `C1ResidualPolicy`.

## Flat grouped additive control

`FlatGroupedAdditivePolicy.from_reference(reference, leaf_rank=1 or 2, exploration_epsilon=0.05, temperature=2.0)` has the same family and leaf adapters as `C1ResidualPolicy.from_reference(reference, method="tree", ...)`:

- Family adapters are the rank-one `3 × 65` contrast factors for family IDs 7 and 8.
- Leaf adapters are rank-one or rank-two `3 × 195` factors for endpoint IDs 1 through 6.
- For a leaf route, its family adapter uses the 65-value actor feature vector and its leaf adapter uses the 195-value one-hot band feature vector. A family route uses its family adapter.

It selects adapters with flat `endpoint → family` and `family → leaves` tables. It stores no route paths. `from_tree(treecore)` copies every policy tensor exactly and keeps matching parameter names, so `optimizer_manifest` state transfers by name. This control is algorithmically equivalent to C1 tree traversal; it tests whether the observed result depends on the traversal representation, not whether sharing family adapters helps over an independent algorithm.

## Grouped dictionary control

`GroupedDictionaryPolicy.from_reference(reference, method="grouped_dictionary", leaf_rank=1, dictionary_rank=3 or 6, exploration_epsilon=0.05, temperature=2.0)` replaces the family-plus-leaf decomposition with:

- One learned feature basis per family, each with shape `K × 195`. The `shared_V` property exposes the stacked shape `(2, K, 195)`.
- One leaf coefficient matrix of shape `3 × K` for each endpoint 1 through 6.
- No hierarchical paths or parent adapter parameters.

The two family bases are separate `nn.Parameter` tensors. This lets `LeafAdam` leave the inactive family's value and Adam moments untouched while the other family trains.

IDs 1, 2, 3 belong to family 7; IDs 4, 5, 6 belong to family 8. The persistent seen mask starts with endpoints 1, 2, 4, and 5 available. Family fallback IDs 7 and 8 average the effective residual matrices for the leaves marked seen in that family. `initialize_heldout(3)` or `initialize_heldout(6)` copies the mean seen sibling coefficients and marks that endpoint seen. After adapting a leaf, call `mark_seen(endpoint)` if it should enter subsequent fallback averages.

The learned residual has `2 × K × 195 + 6 × 3 × K` float32 values. Fully warmed residual parameters plus their two Adam moments use 14,688 bytes for `K = 3` and 29,376 bytes for `K = 6`; the persistent seen mask adds 6 policy bytes. Both fit under the corresponding 15,952-byte and 30,700-byte residual budgets. These figures exclude the common frozen base, critic state, optimizer metadata, router, and rollout buffer.

Both controls support the existing `make_optimizer`, `to_spec`, `export_payload`, and `from_payload` workflow. The focused tests are in `tests/test_c1_modular_controls.py`.

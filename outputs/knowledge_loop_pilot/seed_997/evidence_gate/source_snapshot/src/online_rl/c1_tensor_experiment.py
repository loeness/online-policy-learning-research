"""Runner for the predeclared C1 tensor source-basis alternatives.

The runner delegates training, evaluation, routing, checkpointing, capacity
measurement, and phase restoration to ``c1_supplement_experiment``.  It only
provides factories and payload restoration for the two additional layouts.
Importing this module never starts an experiment.
"""
from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import torch

from . import c1_supplement_experiment as generic
from .c1_tensor_controls import TensorSourceBasisPolicy


_TENSOR_METHODS = {"tensor_learned", "output_mixer", "hierarchical_output_mixer"}


def _tensor_make_core(config: Mapping[str, Any], seed: int, method: str,
                      tier: Mapping[str, Any], params: Mapping[str, Any]):
    if method not in _TENSOR_METHODS:
        return _ORIGINAL_MAKE_CORE(config, seed, method, tier, params)
    generic.seed_everything(seed)
    reference = generic._foundation(config, seed)
    try:
        core = TensorSourceBasisPolicy.from_reference(
            reference.policy,
            method=method,
            leaf_rank=int(tier.get("leaf_rank", 1)),
            dictionary_rank=3,
            exploration_epsilon=float(config["exploration_epsilon"]),
            temperature=float(config.get("temperature", 2.0)),
            train_critic=True,
        )
    finally:
        reference.env.close()
    optimizer = core.make_optimizer(
        learning_rate=float(params["learning_rate"]), transfer=False
    )
    return core, optimizer


def _tensor_restore_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("spec", {}).get("supplemental_layout") != "c1_tensor_source_basis_v1":
        return _ORIGINAL_RESTORE_PAYLOAD(path)
    restored = TensorSourceBasisPolicy.from_payload(payload)
    restored["continual_state"] = payload.get("replay") or payload.get("supplemental_continual_state")
    return restored


def _tensor_source_hashes(config_path: str | Path, supplement: Mapping[str, Any],
                          base: Mapping[str, Any]) -> dict[str, str]:
    hashes = _ORIGINAL_SOURCE_HASHES(config_path, supplement, base)
    for path in (
        Path("src/online_rl/c1_tensor_controls.py"),
        Path("src/online_rl/c1_tensor_experiment.py"),
        Path("scripts/run_c1_tensor_alternatives.py"),
        Path("tests/test_c1_tensor_controls.py"),
        Path("docs/C1_TENSOR_ALTERNATIVES.md"),
        Path("docs/C1_SUPPLEMENT_AMENDMENTS.md"),
    ):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


_ORIGINAL_MAKE_CORE = generic._make_core
_ORIGINAL_RESTORE_PAYLOAD = generic._restore_payload
_ORIGINAL_SOURCE_HASHES = generic._source_hashes


@contextmanager
def _installed_factories():
    """Temporarily install tensor factories in this runner process only."""
    old_make, old_restore, old_hashes = (
        generic._make_core, generic._restore_payload, generic._source_hashes
    )
    generic._make_core = _tensor_make_core
    generic._restore_payload = _tensor_restore_payload
    generic._source_hashes = _tensor_source_hashes
    try:
        yield
    finally:
        generic._make_core = old_make
        generic._restore_payload = old_restore
        generic._source_hashes = old_hashes


def run_alternatives(config_path: str | Path = "configs/c1_supplement.json") -> dict[str, Any]:
    """Run new seed-919 validation and seed-181/193/211 low-tier formals.

    The tree comparator uses the already-frozen Step 2 selection.  No new tree
    hyperparameter selection is performed here.
    """
    supplement, base = generic.load_protocol(config_path)
    supplement = copy.deepcopy(supplement)
    supplement["validation_seeds"] = [919]
    supplement["formal_seeds"] = [181, 193, 211]
    output = Path(supplement["output"])
    root = output / "representation_alternatives"
    root.mkdir(parents=True, exist_ok=True)
    low_tier = base["tiers"]["low"]
    selections: dict[str, Any] = {}
    formals: dict[str, Any] = {}

    tree_selection_path = output / "step2_grouping" / "tree" / "selection.json"
    if not tree_selection_path.exists():
        raise FileNotFoundError(
            "the tensor alternatives require the already-frozen Step 2 tree selection at "
            f"{tree_selection_path}"
        )
    tree_selection = generic._load_json(tree_selection_path)
    tree_params = tree_selection.get("selected_params")
    if not isinstance(tree_params, Mapping):
        raise ValueError("the frozen Step 2 tree selection has no selected parameters")

    with _installed_factories():
        for method in ("tensor_learned", "output_mixer", "hierarchical_output_mixer"):
            selection = generic._run_validations(
                root / method,
                supplement,
                base,
                method,
                "low",
                low_tier,
                supplement["compressed_candidates"],
                config_path=config_path,
            )
            selections[method] = selection
            selected = selection.get("selected_params")
            if selected is not None:
                formals[method] = generic._run_formal(
                    root / method,
                    supplement,
                    base,
                    method,
                    "low",
                    low_tier,
                    selected,
                    config_path=config_path,
                )

        formals["tree_reference"] = generic._run_formal(
            root / "tree_reference",
            supplement,
            base,
            "tree",
            "low",
            low_tier,
            tree_params,
            config_path=config_path,
        )

    frozen = {
        "stage": "representation_alternatives",
        "scope": "C1 only; no transition maps or C2 discovery",
        "methods": ["tensor_learned", "output_mixer", "hierarchical_output_mixer"],
        "validation_seeds": supplement["validation_seeds"],
        "formal_seeds": supplement["formal_seeds"],
        "validation_candidates": supplement["compressed_candidates"],
        "selected": selections,
        "tree_reference_params": dict(tree_params),
        "tree_selection_source": str(tree_selection_path),
        "low_tier": dict(low_tier),
        "formal_results": formals,
        "source_sha256": _tensor_source_hashes(config_path, supplement, base),
    }
    generic.write_json(root / "frozen_settings.json", frozen)
    return frozen


__all__ = ["run_alternatives"]

"""Validation-first runner for the full shared actor with compact context heads."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import torch

from . import c1_supplement_experiment as neural
from .c1_context_head_controls import C1ContextHeadFullPolicy


METHOD = "full_context_head"
VALIDATION_SEED = 923
FORMAL_SEEDS = (181, 193, 211)
LR_CANDIDATES = (
    {"learning_rate": 0.001, "ent_coef": 0.06},
    {"learning_rate": 0.003, "ent_coef": 0.06},
    {"learning_rate": 0.01, "ent_coef": 0.06},
)
EWC_COEFFICIENTS = (10.0, 100.0, 1000.0)


def _make_context_head_core(config, seed, method, tier, params):
    if method != METHOD:
        return _ORIGINAL_MAKE_CORE(config, seed, method, tier, params)
    neural.seed_everything(int(seed))
    reference = neural._foundation(config, int(seed))
    try:
        core = C1ContextHeadFullPolicy.from_reference(
            reference.policy,
            route_conditioned=True,
            band_conditioned=True,
            epsilon=float(config["exploration_epsilon"]),
            temperature=float(config.get("temperature", 2.0)),
        )
    finally:
        reference.env.close()
    optimizer = core.make_optimizer(learning_rate=float(params["learning_rate"]))
    return core, optimizer


def _restore_context_head_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("spec", {}).get("architecture") != "C1ContextHeadFullPolicy-v1":
        del payload
        return _ORIGINAL_RESTORE_PAYLOAD(path)
    restored = C1ContextHeadFullPolicy.from_payload(payload)
    restored["continual_state"] = payload.get("replay") or payload.get("ewc")
    del payload
    return restored


@contextmanager
def _context_head_hooks():
    """Dispatch this method through the shared runner without editing it."""
    global _ORIGINAL_MAKE_CORE, _ORIGINAL_RESTORE_PAYLOAD, _ORIGINAL_IS_METHOD_FULL
    _ORIGINAL_MAKE_CORE = neural._make_core
    _ORIGINAL_RESTORE_PAYLOAD = neural._restore_payload
    _ORIGINAL_IS_METHOD_FULL = neural._is_method_full
    neural._make_core = _make_context_head_core
    neural._restore_payload = _restore_context_head_payload
    neural._is_method_full = lambda method: method == METHOD or _ORIGINAL_IS_METHOD_FULL(method)
    try:
        yield
    finally:
        neural._make_core = _ORIGINAL_MAKE_CORE
        neural._restore_payload = _ORIGINAL_RESTORE_PAYLOAD
        neural._is_method_full = _ORIGINAL_IS_METHOD_FULL


def _select(rows: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
    candidates = []
    for index, row in enumerate(rows):
        result = row["result"]
        candidates.append({
            "candidate": index,
            "params": row["params"],
            "coefficient": row.get("coefficient"),
            "status": result.get("status"),
            "score": neural._score_result(result),
            "result_path": row["result_path"],
        })
    complete = [candidate for candidate in candidates
                if candidate["status"] == "complete" and candidate["score"] == candidate["score"]]
    selected = max(complete, key=lambda candidate: (candidate["score"], -candidate["candidate"])) if complete else None
    return {
        "kind": kind,
        "selection_formula": "0.50 final mean success + 0.25 immediate old-task mean success + 0.25 heldout mean success",
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_params": None if selected is None else selected["params"],
        "selected_coefficient": None if selected is None else selected["coefficient"],
        "selected_score": None if selected is None else selected["score"],
        "candidates": candidates,
    }


def _validation_job(root: Path, supplement, base, params, *, index: int,
                    control: str = "none", coefficient: float = 0.0,
                    config_path="configs/c1_supplement.json") -> dict[str, Any]:
    path = root / "validation" / control / f"candidate_{index:02d}" / f"s{VALIDATION_SEED}"
    with _context_head_hooks():
        result = neural.run_arm(
            supplement=supplement,
            base=base,
            output=path,
            seed=VALIDATION_SEED,
            method=METHOD,
            tier_name="high",
            tier=base["tiers"]["high"],
            params=params,
            stage_name="context_head_alternatives",
            control=control,
            coefficient=float(coefficient),
            validation=True,
            config_path=config_path,
        )
    return {"params": dict(params), "coefficient": float(coefficient), "result": result,
            "result_path": str(path / "result.json")}


def _formal_jobs(root: Path, supplement, base, params, *, control="none", coefficient=0.0,
                 config_path="configs/c1_supplement.json") -> list[dict[str, Any]]:
    rows = []
    for seed in FORMAL_SEEDS:
        path = root / "formal" / control / f"s{seed}"
        with _context_head_hooks():
            result = neural.run_arm(
                supplement=supplement,
                base=base,
                output=path,
                seed=seed,
                method=METHOD,
                tier_name="high",
                tier=base["tiers"]["high"],
                params=params,
                stage_name="context_head_alternatives",
                control=control,
                coefficient=float(coefficient),
                validation=False,
                config_path=config_path,
            )
        rows.append({"seed": seed, "result": result, "result_path": str(path / "result.json")})
        neural.write_json(root / "formal_progress.json", rows)
    return rows


def run_context_head_alternatives(config_path="configs/c1_supplement.json") -> dict[str, Any]:
    """Run 6 validations, freeze settings, then run 6 formal jobs.

    Calling this function performs the experiments. Merely importing this
    module has no side effects.
    """
    supplement, base = neural.load_protocol(config_path)
    output = Path(supplement["output"]) / "context_head_alternatives"
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = neural._source_hashes(config_path, supplement, base)
    source_hashes["src/online_rl/c1_context_head_controls.py"] = hashlib.sha256(
        Path("src/online_rl/c1_context_head_controls.py").read_bytes()
    ).hexdigest()
    source_hashes["src/online_rl/c1_context_head_experiment.py"] = hashlib.sha256(
        Path("src/online_rl/c1_context_head_experiment.py").read_bytes()
    ).hexdigest()
    frozen_protocol = {
        "method": METHOD,
        "scope": "one shared full actor/critic, shared route and band biases, six small context heads",
        "validation_seed": VALIDATION_SEED,
        "formal_seeds": list(FORMAL_SEEDS),
        "learning_rate_candidates": list(LR_CANDIDATES),
        "ewc_coefficients": list(EWC_COEFFICIENTS),
        "source_sha256": source_hashes,
        "capacity_bytes": int(supplement["full_reference_agent_budget_bytes"]),
        "budget_classification": "outside original adaptive caps; measured Pareto reference",
        "continual_control": "task-free diagonal empirical-Fisher EWC; source Fisher 64; 2048-step cadence",
    }
    neural.write_json(output / "protocol.json", frozen_protocol)

    ordinary_rows = [
        _validation_job(output, supplement, base, params, index=index, config_path=config_path)
        for index, params in enumerate(LR_CANDIDATES)
    ]
    ordinary_selection = _select(ordinary_rows, kind="shared_actor_context_head")
    neural.write_json(output / "validation" / "selection.json", ordinary_selection)
    if ordinary_selection["selected_params"] is None:
        result = {"status": "no_valid_shared_actor_setting", "protocol": frozen_protocol,
                  "shared_actor_selection": ordinary_selection}
        neural.write_json(output / "summary.json", result)
        return result

    selected_params = ordinary_selection["selected_params"]
    ewc_rows = [
        _validation_job(output, supplement, base, selected_params, index=index, control="ewc",
                        coefficient=coefficient, config_path=config_path)
        for index, coefficient in enumerate(EWC_COEFFICIENTS)
    ]
    ewc_selection = _select(ewc_rows, kind="shared_actor_context_head_ewc")
    neural.write_json(output / "validation" / "ewc_selection.json", ewc_selection)
    if ewc_selection["selected_coefficient"] is None:
        result = {"status": "no_valid_ewc_setting", "protocol": frozen_protocol,
                  "shared_actor_selection": ordinary_selection, "ewc_selection": ewc_selection}
        neural.write_json(output / "summary.json", result)
        return result

    # Freeze both selections to disk before starting any new formal seed.
    selected = {
        "protocol": frozen_protocol,
        "shared_actor_selection": ordinary_selection,
        "ewc_selection": ewc_selection,
        "formal_started": False,
    }
    neural.write_json(output / "frozen_settings.json", selected)
    ordinary_formal = _formal_jobs(output, supplement, base, selected_params,
                                   config_path=config_path)
    ewc_formal = _formal_jobs(output, supplement, base, selected_params, control="ewc",
                              coefficient=ewc_selection["selected_coefficient"],
                              config_path=config_path)
    selected["formal_started"] = True
    selected["formal"] = {"shared_actor": ordinary_formal, "ewc": ewc_formal}
    selected["formal_complete"] = all(
        row["result"].get("status") == "complete" for row in ordinary_formal + ewc_formal
    )
    neural.write_json(output / "summary.json", selected)
    return selected


__all__ = ["run_context_head_alternatives", "METHOD", "LR_CANDIDATES", "EWC_COEFFICIENTS"]

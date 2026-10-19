"""Pure NumPy helpers for estimating and applying retention writes.

These helpers do not access an environment, policy, or persistent history. The
surrogate estimate is a small-sample diagnostic, not a strict off-policy
evaluation guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from online_rl.consolidation_probe import recompose_shared


_ACTION_COUNT = 4
_MATRIX_SHAPE = (3, 3)


def _finite_real_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _slot_indices(source_slots: Sequence[int], slot_count: int, *, allow_empty: bool) -> list[int]:
    if isinstance(source_slots, (str, bytes)):
        raise ValueError("source_slots must be a sequence of integer slot indices")
    try:
        raw_slots = list(source_slots)
    except TypeError as exc:
        raise ValueError("source_slots must be a sequence of integer slot indices") from exc
    slots: list[int] = []
    for value in raw_slots:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("source_slots must contain integer slot indices")
        slot = int(value)
        if slot < 0 or slot >= slot_count:
            raise ValueError("source_slots contains an out-of-range slot")
        if slot not in slots:
            slots.append(slot)
    if not slots and not allow_empty:
        raise ValueError("source_slots must contain at least two unique live slots")
    return slots


def _validate_layout(
    local: Any, shared: Any, groups: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_array = np.asarray(local)
    shared_array = np.asarray(shared)
    groups_array = np.asarray(groups)
    if local_array.ndim != 2 or local_array.shape[1] != 27:
        raise ValueError(f"local must have shape (S, 27), got {local_array.shape}")
    if shared_array.ndim != 2 or shared_array.shape[1] != 9 or shared_array.shape[0] == 0:
        raise ValueError(f"shared must have shape (G, 9) with G > 0, got {shared_array.shape}")
    if groups_array.shape != (local_array.shape[0],):
        raise ValueError(f"groups must have shape ({local_array.shape[0]},), got {groups_array.shape}")
    if local_array.dtype.kind != "f" or shared_array.dtype.kind != "f":
        raise ValueError("local and shared must contain floating point values")
    if groups_array.dtype.kind not in "iu":
        raise ValueError("groups must contain integer group indices")
    local_values = _finite_real_array(local_array, "local")
    shared_values = _finite_real_array(shared_array, "shared")
    if np.any((groups_array < -1) | (groups_array >= shared_array.shape[0])):
        raise ValueError("groups contains an invalid group index")
    return local_values, shared_values, groups_array


def common_mean(
    local: Any, shared: Any, groups: Any, source_slots: Sequence[int]
) -> np.ndarray:
    """Return the uniform mean effective matrix for at least two live slots.

    Each selected slot contributes all three bands equally. An unbound slot
    (group ``-1``) contributes its local matrices without shared parameters.
    Duplicate slot IDs are counted once.
    """
    local_values, shared_values, groups_array = _validate_layout(local, shared, groups)
    slots = _slot_indices(source_slots, local_values.shape[0], allow_empty=False)
    if len(slots) < 2:
        raise ValueError("source_slots must contain at least two unique live slots")

    effective = local_values.reshape(-1, 3, 3, 3)[np.asarray(slots, dtype=np.int64)].copy()
    selected_groups = groups_array[np.asarray(slots, dtype=np.int64)]
    assigned = selected_groups >= 0
    if np.any(assigned):
        effective[assigned] += shared_values[selected_groups[assigned]].reshape(-1, 1, 3, 3)
    return effective.mean(axis=(0, 1))


def compensated_install(
    local: Any,
    shared: Any,
    groups: Any,
    *,
    group: int,
    common: Any,
    join_slots: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Install ``common`` in a group while preserving every affected slot.

    All current residents of the destination and all ``join_slots`` retain
    their old effective matrices. Unassigned slots use only their local
    matrices when computing their pre-install effective values.
    """
    local_values, shared_values, groups_array = _validate_layout(local, shared, groups)
    if isinstance(group, (bool, np.bool_)) or not isinstance(group, (int, np.integer)):
        raise ValueError("group must be an integer group index")
    destination = int(group)
    if destination < 0 or destination >= shared_values.shape[0]:
        raise ValueError("group is outside the shared parameter array")
    common_values = _finite_real_array(common, "common", _MATRIX_SHAPE)
    joining = _slot_indices(join_slots, local_values.shape[0], allow_empty=True)

    # Including existing residents lets the shared recomposition helper
    # compensate them even when the caller supplies no new joiners.
    affected_input = np.flatnonzero(groups_array == destination).tolist()
    affected_input.extend(slot for slot in joining if slot not in affected_input)
    if affected_input:
        return recompose_shared(
            local,
            shared,
            groups,
            source_slots=affected_input,
            group=destination,
            common=common_values,
        )

    # Installing into an empty group with no joiners still updates the shared
    # value but does not bind an unrelated local slot.
    new_local = np.asarray(local).copy()
    new_shared = np.asarray(shared).copy()
    new_groups = groups_array.copy()
    new_shared[destination] = common_values.reshape(9).astype(new_shared.dtype, copy=False)
    return new_local, new_shared, new_groups, {
        "affected_slots": (),
        "max_effective_preservation_error": 0.0,
    }


def _first_evidence_value(evidence: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in evidence:
            return evidence[name]
    return None


def _unknown_gain(reason: str, *, rows: int = 0, blocks: int = 0) -> dict[str, Any]:
    return {
        "status": "unknown",
        "supported": False,
        "gain": None,
        "lower": None,
        "blocks": int(blocks),
        "ess": {"candidate": None, "baseline": None},
        "ess_ratio": {"candidate": None, "baseline": None},
        "kl": {"candidate": None, "baseline": None},
        "coverage": {"candidate": None, "baseline": None},
        "stderr": None,
        "rows": int(rows),
        "reason": reason,
    }


def _probability_rows(value: Any, name: str, rows: int) -> np.ndarray:
    probabilities = _finite_real_array(value, name, (rows, _ACTION_COUNT))
    if np.any(probabilities < 0):
        raise ValueError(f"{name} cannot contain negative values")
    totals = probabilities.sum(axis=1)
    if np.any(totals <= 0) or not np.allclose(totals, 1.0, atol=1e-4, rtol=0):
        raise ValueError(f"each row of {name} must sum to one")
    return probabilities


def _forward_kl_rows(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = np.zeros(source.shape[0], dtype=np.float64)
    for index in range(source.shape[0]):
        positive_source = source[index] > 0
        if np.any(target[index, positive_source] <= 0):
            rows[index] = float("inf")
        else:
            rows[index] = float(
                np.sum(
                    source[index, positive_source]
                    * (
                        np.log(source[index, positive_source])
                        - np.log(target[index, positive_source])
                    )
                )
            )
    return np.maximum(rows, 0.0)


def _ess_from_log_weights(log_weights: np.ndarray) -> float:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        return 0.0
    shifted = np.exp(log_weights[finite] - np.max(log_weights[finite]))
    squared_sum = float(np.dot(shifted, shifted))
    return float(shifted.sum() ** 2 / squared_sum) if squared_sum > 0 else 0.0


def surrogate_gain(
    evidence: dict[str, Any],
    candidate_probs: Any,
    baseline_probs: Any,
    *,
    min_rows: int = 12,
    min_blocks: int = 3,
    min_ess_ratio: float = 0.2,
    max_mean_kl: float = 0.5,
    clip_ratio: float = 5.0,
    stderr_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Estimate a clipped behavior-sample surrogate gain with support checks.

    Importance ratios use the stored behavior action log probability. Forward
    KL uses the stored full ``behavior_probs`` distribution. The row-weighted
    gain and finite-sample cluster standard error are diagnostic estimates,
    not safety bounds or a strict off-policy evaluation guarantee.
    """
    candidate_raw = np.asarray(candidate_probs)
    baseline_raw = np.asarray(baseline_probs)
    if candidate_raw.ndim != 2 or candidate_raw.shape[1:] != (_ACTION_COUNT,):
        raise ValueError(f"candidate_probs must have shape (N, {_ACTION_COUNT})")
    if baseline_raw.shape != candidate_raw.shape:
        raise ValueError("baseline_probs must have the same shape as candidate_probs")
    rows = int(candidate_raw.shape[0])
    if rows < 1:
        return _unknown_gain("empty_evidence")
    candidate = _probability_rows(candidate_probs, "candidate_probs", rows)
    baseline = _probability_rows(baseline_probs, "baseline_probs", rows)
    if not isinstance(evidence, Mapping):
        return _unknown_gain("missing_evidence", rows=rows)

    actions_raw = _first_evidence_value(evidence, "actions", "action")
    behavior_probs_raw = _first_evidence_value(evidence, "behavior_probs", "probabilities")
    behavior_logp_raw = _first_evidence_value(
        evidence, "behavior_log_prob", "behavior_log_probs", "log_prob"
    )
    advantages_raw = _first_evidence_value(evidence, "advantages", "advantage")
    block_ids_raw = _first_evidence_value(evidence, "block_id", "episode_id")
    if any(
        value is None
        for value in (actions_raw, behavior_probs_raw, behavior_logp_raw, advantages_raw, block_ids_raw)
    ):
        return _unknown_gain("missing_required_evidence", rows=rows)

    actions = np.asarray(actions_raw)
    if actions.shape != (rows,) or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("actions must be an integer vector of length N")
    actions = actions.astype(np.int64, copy=False)
    if np.any((actions < 0) | (actions >= _ACTION_COUNT)):
        raise ValueError(f"actions must be in [0, {_ACTION_COUNT - 1}]")
    behavior = _probability_rows(behavior_probs_raw, "behavior_probs", rows)
    behavior_logp = _finite_real_array(behavior_logp_raw, "behavior_log_prob")
    if behavior_logp.shape != (rows,) or np.any(behavior_logp > 1e-6):
        raise ValueError("behavior_log_prob must be a finite nonpositive vector of length N")
    advantages = _finite_real_array(advantages_raw, "advantage")
    if advantages.shape != (rows,):
        raise ValueError("advantage must be a finite vector of length N")
    block_ids = np.asarray(block_ids_raw)
    if block_ids.shape != (rows,) or not np.issubdtype(block_ids.dtype, np.integer):
        raise ValueError("episode_id or block_id must be an integer vector of length N")

    if isinstance(min_rows, (bool, np.bool_)) or int(min_rows) != min_rows or min_rows < 1:
        raise ValueError("min_rows must be a positive integer")
    if isinstance(min_blocks, (bool, np.bool_)) or int(min_blocks) != min_blocks or min_blocks < 1:
        raise ValueError("min_blocks must be a positive integer")
    for value, name in ((min_ess_ratio, "min_ess_ratio"), (max_mean_kl, "max_mean_kl"),
                        (clip_ratio, "clip_ratio"), (stderr_multiplier, "stderr_multiplier")):
        if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if min_ess_ratio < 0 or min_ess_ratio > 1:
        raise ValueError("min_ess_ratio must be in [0, 1]")
    if max_mean_kl < 0:
        raise ValueError("max_mean_kl must be nonnegative")
    if clip_ratio < 1:
        raise ValueError("clip_ratio must be at least 1")
    if stderr_multiplier < 0:
        raise ValueError("stderr_multiplier must be nonnegative")

    behavior_selected = behavior[np.arange(rows), actions]
    if np.any(behavior_selected <= 0):
        return _unknown_gain("sampled_action_outside_behavior_support", rows=rows)

    candidate_selected = candidate[np.arange(rows), actions]
    baseline_selected = baseline[np.arange(rows), actions]

    def ratios_and_ess(selected: np.ndarray) -> tuple[np.ndarray, float]:
        log_ratio = np.full(rows, -np.inf, dtype=np.float64)
        positive = selected > 0
        log_ratio[positive] = np.log(selected[positive]) - behavior_logp[positive]
        return log_ratio, _ess_from_log_weights(log_ratio)

    candidate_log_ratio, candidate_ess = ratios_and_ess(candidate_selected)
    baseline_log_ratio, baseline_ess = ratios_and_ess(baseline_selected)
    candidate_ess_ratio = candidate_ess / rows
    baseline_ess_ratio = baseline_ess / rows

    candidate_kl_rows = _forward_kl_rows(behavior, candidate)
    baseline_kl_rows = _forward_kl_rows(behavior, baseline)
    candidate_kl = float(np.mean(candidate_kl_rows))
    baseline_kl = float(np.mean(baseline_kl_rows))
    candidate_coverage = bool(np.isfinite(candidate_kl_rows).all())
    baseline_coverage = bool(np.isfinite(baseline_kl_rows).all())

    log_clip = math.log(float(clip_ratio))
    candidate_clipped = np.exp(np.minimum(candidate_log_ratio, log_clip))
    baseline_clipped = np.exp(np.minimum(baseline_log_ratio, log_clip))
    with np.errstate(over="ignore", invalid="ignore"):
        row_gains = (candidate_clipped - baseline_clipped) * advantages
    gain = float(np.mean(row_gains))

    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    blocks = int(len(unique_blocks))
    if blocks < 2:
        stderr = float("inf")
    else:
        centered_block_sums = np.bincount(
            inverse, weights=row_gains - gain, minlength=blocks
        )
        variance = (
            (blocks / (blocks - 1.0))
            * float(np.dot(centered_block_sums, centered_block_sums))
            / (rows * rows)
        )
        stderr = float(math.sqrt(max(variance, 0.0)))
    lower = gain - float(stderr_multiplier) * stderr

    reasons: list[str] = []
    if rows < int(min_rows):
        reasons.append("insufficient_rows")
    if blocks < int(min_blocks):
        reasons.append("insufficient_blocks")
    if not candidate_coverage:
        reasons.append("candidate_lacks_behavior_coverage")
    if not baseline_coverage:
        reasons.append("baseline_lacks_behavior_coverage")
    if candidate_ess_ratio < float(min_ess_ratio):
        reasons.append("candidate_low_ess")
    if baseline_ess_ratio < float(min_ess_ratio):
        reasons.append("baseline_low_ess")
    if candidate_kl > float(max_mean_kl):
        reasons.append("candidate_kl_too_high")
    if baseline_kl > float(max_mean_kl):
        reasons.append("baseline_kl_too_high")
    if not math.isfinite(gain) or not math.isfinite(stderr) or not math.isfinite(lower):
        reasons.append("nonfinite_surrogate_estimate")

    supported = not reasons and math.isfinite(lower)
    return {
        "status": "supported" if supported else "unknown",
        "supported": bool(supported),
        "gain": gain if supported else None,
        "lower": lower if supported else None,
        "blocks": blocks,
        "ess": {"candidate": candidate_ess, "baseline": baseline_ess},
        "ess_ratio": {"candidate": candidate_ess_ratio, "baseline": baseline_ess_ratio},
        "kl": {"candidate": candidate_kl, "baseline": baseline_kl},
        "coverage": {"candidate": candidate_coverage, "baseline": baseline_coverage},
        "stderr": stderr if math.isfinite(stderr) else None,
        "rows": rows,
        "reason": "supported" if supported else ";".join(reasons),
    }


def decide_retention_write(
    scores: Sequence[dict[str, Any]],
    *,
    min_supported_profiles: int = 2,
    min_mean_gain: float = 0.0,
    max_profile_loss: float = 0.05,
) -> dict[str, Any]:
    """Accept only when supported profiles clear the mean and loss limits."""
    if isinstance(min_supported_profiles, (bool, np.bool_)) or int(min_supported_profiles) != min_supported_profiles or min_supported_profiles < 1:
        raise ValueError("min_supported_profiles must be a positive integer")
    if isinstance(min_mean_gain, (bool, np.bool_)) or not math.isfinite(float(min_mean_gain)):
        raise ValueError("min_mean_gain must be finite")
    if isinstance(max_profile_loss, (bool, np.bool_)) or not math.isfinite(float(max_profile_loss)) or max_profile_loss < 0:
        raise ValueError("max_profile_loss must be finite and nonnegative")

    if isinstance(scores, (str, bytes)):
        raise ValueError("scores must be a sequence of score mappings")
    valid_lowers: list[float] = []
    for score in scores:
        if not isinstance(score, Mapping) or score.get("supported") is not True:
            continue
        lower = score.get("lower")
        if isinstance(lower, (bool, np.bool_)):
            continue
        try:
            value = float(lower)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            valid_lowers.append(value)

    count = len(valid_lowers)
    mean_lower = float(np.mean(valid_lowers)) if valid_lowers else None
    min_lower = float(min(valid_lowers)) if valid_lowers else None
    if count < int(min_supported_profiles):
        accepted = False
        reason = "insufficient_supported_profiles"
    elif mean_lower is None or mean_lower <= float(min_mean_gain):
        accepted = False
        reason = "mean_lower_not_above_threshold"
    elif min_lower is None or min_lower < -float(max_profile_loss):
        accepted = False
        reason = "supported_profile_loss_too_large"
    else:
        accepted = True
        reason = "accepted"
    return {
        "accepted": accepted,
        "status": "accept" if accepted else "reject",
        "reason": reason,
        "supported_profiles": count,
        "mean_lower": mean_lower,
        "min_lower": min_lower,
    }

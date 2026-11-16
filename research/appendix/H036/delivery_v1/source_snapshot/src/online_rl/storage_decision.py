"""Bounded state and pure interventions for the single storage decision study.

The module deliberately treats task and group IDs as supplied experiment
metadata. It makes no claim about discovering task identity or boundaries.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any, Callable

import numpy as np


GROUPS = (0, 1)
ACTIONS = ("keep", "replace", "mean", "none")
HISTORY_TEMPLATES = {
    "old_major": ((0, 0), (0, 1), (0, 2), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    "new_major": ((0, 0), (1, 0), (1, 1), (1, 2), (0, 1), (1, 2), (1, 0), (1, 1)),
}
HISTORY_REGIME_ORDER = ("old_major", "new_major")
FUTURE_TEMPLATES = {
    "old_major": ((0, 0), (0, 1), (0, 2), (1, 0), (0, 0), (0, 1), (0, 2), (1, 1), (0, 0), (0, 1), (0, 2), (1, 2)),
    "new_major": ((1, 0), (1, 1), (1, 2), (0, 0), (1, 0), (1, 1), (1, 2), (0, 1), (1, 0), (1, 1), (1, 2), (0, 2)),
}
FUTURE_REGIME_ORDER = ("old_major", "new_major", "balanced")


def _ordered_nonempty_subset(values: Sequence[Any], allowed: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty ordered subset of {tuple(allowed)}")
    result = [str(value) for value in values]
    expected_order = [value for value in allowed if value in result]
    if not result or len(result) != len(set(result)) or result != expected_order:
        raise ValueError(f"{name} must be a nonempty ordered subset of {tuple(allowed)}")
    return result


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen config and return a detached JSON-friendly copy."""
    result = json.loads(json.dumps(dict(config)))
    expected = {
        "local_slots": 2,
        "shared_groups": 1,
        "rollout_steps": 256,
        "local_learning_rate": 0.003,
        "critic_learning_rate": 0.003,
        "shared_learning_rate": 0.0,
        "actor_budget_bytes": 89,
        "package_budget_bytes": 262144,
        "evidence_rows_per_group": 256,
        "recent_bouts": 8,
    }
    for key, value in expected.items():
        if key in result and result[key] != value:
            raise ValueError(f"{key} is frozen to {value!r}")
        result[key] = value
    if result.get("format") not in ("storage_decision_v1", "storage_decision_config_v1", "storage_decision_config_v2"):
        raise ValueError("format must be storage_decision_v1")
    for name in ("bootstrap_steps", "bout_steps"):
        value = int(result.get(name, 0))
        if value < 1 or value % result["rollout_steps"]:
            raise ValueError(f"{name} must be a positive multiple of rollout_steps")
        result[name] = value
    result["pair_kinds"] = list(result.get("pair_kinds", ("dynamics", "reward_conflict")))
    result["history_regimes"] = list(result.get("history_regimes", HISTORY_REGIME_ORDER))
    result["future_regimes"] = _ordered_nonempty_subset(
        result.get("future_regimes", HISTORY_REGIME_ORDER), FUTURE_REGIME_ORDER, "future_regimes",
    )
    result["actions"] = list(result.get("actions", ACTIONS))
    if not set(result["pair_kinds"]) or not set(result["pair_kinds"]) <= {"dynamics", "reward_conflict"}:
        raise ValueError("pair_kinds must be a nonempty subset of dynamics and reward_conflict")
    if not set(result["history_regimes"]) or not set(result["history_regimes"]) <= {"old_major", "new_major"}:
        raise ValueError("history_regimes must be a nonempty subset of old_major and new_major")
    if tuple(result["actions"]) != ACTIONS:
        raise ValueError(f"actions must be ordered as {ACTIONS}")
    result.setdefault("seeds", [20101, 20107, 20113, 20117, 20123, 20129, 20143, 20149])
    result.setdefault("pilot_seed", 20089)
    result.setdefault("max_workers", 4)
    result.setdefault("torch_threads", 1)
    result.setdefault("domain", "navigation")
    if result["domain"] != "navigation":
        raise ValueError("storage decision protocol requires navigation domain snapshots")
    return result


def pair_mapping(seed: int, pair_kind: str) -> dict[str, Any]:
    """Return the protocol's deterministic old/new family and preference map."""
    if pair_kind not in ("dynamics", "reward_conflict"):
        raise ValueError("pair_kind must be dynamics or reward_conflict")
    rng = np.random.default_rng(int(seed) + 81001)
    old_family = str(rng.choice(np.asarray(("A", "B"))))
    if pair_kind == "dynamics":
        new_family = "B" if old_family == "A" else "A"
        old_preference = new_preference = 1
    else:
        new_family = old_family
        old_preference = int(rng.choice(np.asarray((-1, 1), dtype=np.int8)))
        new_preference = -old_preference
    return {
        "pair_kind": pair_kind,
        "families": {"old": old_family, "new": new_family},
        "preferences": {"old": old_preference, "new": new_preference},
        "groups": {"old": 0, "new": 1},
    }


def task_name(mapping: Mapping[str, Any], group: int, variant: int) -> str:
    """Convert supplied group and variant metadata to the canonical task ID."""
    if int(group) not in GROUPS or int(variant) not in (0, 1, 2):
        raise ValueError("group and variant must be in their protocol ranges")
    role = "old" if int(group) == 0 else "new"
    family = str(mapping["families"][role])
    preference = int(mapping["preferences"][role])
    return f"{family}{int(variant)}{'+' if preference > 0 else '-'}"


def bootstrap_schedule(mapping: Mapping[str, Any], steps: int) -> list[dict[str, Any]]:
    return [
        {"phase": "bootstrap", "bout": index, "group": 0, "variant": variant,
         "task": task_name(mapping, 0, variant), "steps": int(steps)}
        for index, variant in enumerate((0, 1))
    ]


def template_schedule(
    mapping: Mapping[str, Any], regime: str, *, kind: str, steps: int, start_bout: int = 0,
) -> list[dict[str, Any]]:
    templates = HISTORY_TEMPLATES if kind == "history" else FUTURE_TEMPLATES if kind == "future" else None
    if templates is None or regime not in templates:
        raise ValueError("kind must be history or future and regime must be old_major or new_major")
    return [
        {"phase": kind, "bout": int(start_bout) + index, "group": group, "variant": variant,
         "task": task_name(mapping, group, variant), "steps": int(steps)}
        for index, (group, variant) in enumerate(templates[regime])
    ]


def prefix_schedule(mapping: Mapping[str, Any], config: Mapping[str, Any], regime: str) -> list[dict[str, Any]]:
    boot = bootstrap_schedule(mapping, int(config["bootstrap_steps"]))
    history = template_schedule(mapping, regime, kind="history", steps=int(config["bout_steps"]), start_bout=len(boot))
    return boot + history


def effective_matrices(local: Any, shared: Any, groups: Any) -> np.ndarray:
    """Return each slot's effective (3, 3, 3) coefficient matrix."""
    a = np.asarray(local, dtype=np.float32)
    c = np.asarray(shared, dtype=np.float32)
    g = np.asarray(groups)
    if a.ndim != 2 or a.shape[1] != 27 or c.shape != (1, 9) or g.shape != (a.shape[0],):
        raise ValueError("expected local (S, 27), shared (1, 9), and groups (S,)")
    out = a.reshape(-1, 3, 3, 3).copy()
    for slot, group in enumerate(g.tolist()):
        if int(group) == 0:
            out[slot] += c.reshape(1, 3, 3)
        elif int(group) != -1:
            raise ValueError("physical group tags must be -1 or 0")
    return out


def common_mean_effective(local: Any, shared: Any, groups: Any, slots: Sequence[int]) -> np.ndarray:
    """Mean selected slots and all three bands, returning a (3, 3) parent."""
    effective = effective_matrices(local, shared, groups)
    indices = [int(value) for value in slots]
    if len(indices) < 2 or len(set(indices)) != len(indices) or any(i < 0 or i >= len(effective) for i in indices):
        raise ValueError("slots must contain at least two unique valid slot indices")
    return effective[np.asarray(indices, dtype=np.int64)].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)


def compensated_single_parent_install(
    local: Any,
    shared: Any,
    physical_groups: Any,
    logical_tags: Any,
    *,
    parent: Any,
    parent_group: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Install one shared parent and preserve each resident's effective matrix."""
    a = np.asarray(local, dtype=np.float32)
    c = np.asarray(shared, dtype=np.float32)
    pg = np.asarray(physical_groups, dtype=np.int8)
    tags = np.asarray(logical_tags, dtype=np.int8)
    new_parent = np.asarray(parent, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 27 or c.shape != (1, 9):
        raise ValueError("actor must have two local matrices and one shared parent")
    if pg.shape != (len(a),) or tags.shape != (len(a),) or np.any((tags < -1) | (tags > 1)):
        raise ValueError("physical/logical group arrays do not match local slots")
    if new_parent.shape not in ((3, 3), (9,)) or not np.isfinite(new_parent).all():
        raise ValueError("parent must be finite with shape 3 by 3 or 9")
    tag = int(parent_group)
    if tag not in (-2, -1, 0, 1):
        raise ValueError("parent_group must be -2 (wildcard), -1 (none), 0, or 1")
    before = effective_matrices(a, c, pg)
    assigned = np.asarray([
        tag == -2 or (tag >= 0 and int(group_tag) == tag)
        for group_tag in tags
    ], dtype=bool)
    new_groups = np.where(assigned, 0, -1).astype(np.int8)
    new_shared = new_parent.reshape(1, 9).copy()
    new_local = before.copy()
    if np.any(assigned):
        new_local[assigned] -= new_shared.reshape(1, 1, 3, 3)
    new_local = new_local.reshape(-1, 27).astype(np.float32)
    after = effective_matrices(new_local, new_shared, new_groups)
    per_slot = np.max(np.abs(after - before), axis=(1, 2, 3)).astype(float).tolist()
    return new_local, new_shared, new_groups, {
        "parent_group": tag,
        "bound_slots": np.flatnonzero(assigned).astype(int).tolist(),
        "max_effective_preservation_error": float(max(per_slot, default=0.0)),
        "per_slot_preservation_error": per_slot,
    }


def apply_action(
    local: Any,
    shared: Any,
    physical_groups: Any,
    logical_tags: Any,
    old_parent: Any,
    new_parent: Any,
    action: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one frozen decision while preserving all residents before codec."""
    old = np.asarray(old_parent, dtype=np.float32).reshape(3, 3)
    new = np.asarray(new_parent, dtype=np.float32).reshape(3, 3)
    if action == "keep":
        target, parent_group = old, 0
    elif action == "replace":
        target, parent_group = new, 1
    elif action == "mean":
        target, parent_group = (old + new) / 2.0, -2
    elif action == "none":
        target, parent_group = np.zeros((3, 3), dtype=np.float32), -1
    else:
        raise ValueError(f"unknown action: {action}")
    return compensated_single_parent_install(
        local, shared, physical_groups, logical_tags,
        parent=target, parent_group=parent_group,
    )


class BoundedGroupEvidence:
    """At most `capacity` finalized PPO rows per supplied group."""
    FIELDS = ("obs", "action", "behavior_probs", "behavior_log_prob", "advantage", "returns", "episode_id", "task", "bout")

    def __init__(self, capacity: int = 256) -> None:
        if int(capacity) < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._rows: dict[int, dict[str, list[Any]]] = {
            group: {field: [] for field in self.FIELDS} for group in GROUPS
        }

    def add_rollout(self, rows: Sequence[Any], *, group: int, task: str, bout: int) -> None:
        if int(group) not in GROUPS:
            raise ValueError("group must be zero or one")
        target = self._rows[int(group)]
        for row in rows:
            target["obs"].append(np.asarray(row.obs, dtype=np.float32).copy())
            target["action"].append(int(row.action))
            target["behavior_probs"].append(np.asarray(row.probabilities, dtype=np.float32).copy())
            target["behavior_log_prob"].append(float(row.log_prob))
            target["advantage"].append(float(row.advantage))
            target["returns"].append(float(row.returns))
            target["episode_id"].append(int(getattr(row, "episode_id", -1)))
            target["task"].append(str(task))
            target["bout"].append(int(bout))
        overflow = len(target["action"]) - self.capacity
        if overflow > 0:
            for field in self.FIELDS:
                del target[field][:overflow]

    def arrays(self, group: int) -> dict[str, np.ndarray]:
        if int(group) not in GROUPS:
            raise ValueError("group must be zero or one")
        source = self._rows[int(group)]
        count = len(source["action"])
        return {
            "obs": np.stack(source["obs"]).astype(np.float32) if count else np.empty((0, 13), dtype=np.float32),
            "action": np.asarray(source["action"], dtype=np.int64),
            "behavior_probs": np.stack(source["behavior_probs"]).astype(np.float32) if count else np.empty((0, 4), dtype=np.float32),
            "behavior_log_prob": np.asarray(source["behavior_log_prob"], dtype=np.float32),
            "advantage": np.asarray(source["advantage"], dtype=np.float32),
            "returns": np.asarray(source["returns"], dtype=np.float32),
            "episode_id": np.asarray(source["episode_id"], dtype=np.int64),
            "task": np.asarray(source["task"], dtype="U4"),
            "bout": np.asarray(source["bout"], dtype=np.int32),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "groups": {group: self.arrays(group) for group in GROUPS}}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BoundedGroupEvidence":
        result = cls(int(state["capacity"]))
        for group in GROUPS:
            arrays = state["groups"][group] if group in state["groups"] else state["groups"][str(group)]
            count = len(np.asarray(arrays["action"]))
            if count > result.capacity:
                raise ValueError("evidence exceeds per-group capacity")
            for field in cls.FIELDS:
                result._rows[group][field] = np.asarray(arrays[field]).tolist()
            result._rows[group]["obs"] = [np.asarray(row, dtype=np.float32).copy() for row in arrays["obs"]]
            result._rows[group]["behavior_probs"] = [np.asarray(row, dtype=np.float32).copy() for row in arrays["behavior_probs"]]
        return result


def decide_from_past(
    history: Sequence[int],
    evidence: BoundedGroupEvidence,
    old_parent: Any,
    new_parent: Any,
    score_fn: Callable[[Mapping[str, Any], Any, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute frozen past-only rules; callers must not pass future rows."""
    recent = [int(group) for group in history[-8:]]
    if len(recent) != 8 or any(group not in GROUPS for group in recent):
        raise ValueError("selectors require exactly eight recent group IDs")
    frequency = {group: recent.count(group) / len(recent) for group in GROUPS}
    old = np.asarray(old_parent, dtype=np.float32).reshape(3, 3)
    new = np.asarray(new_parent, dtype=np.float32).reshape(3, 3)
    zero = np.zeros((3, 3), dtype=np.float32)
    old_rows, new_rows = evidence.arrays(0), evidence.arrays(1)
    old_score = dict(score_fn(old_rows, zero, old))
    new_score = dict(score_fn(new_rows, new, zero))

    def weighted(field: str) -> float | None:
        scores = (old_score, new_score)
        values = [score.get(field) for score in scores]
        if any(value is None or not np.isfinite(float(value)) for value in values):
            return None
        return float(sum(float(values[group]) * frequency[group] for group in GROUPS))

    latest_old = max((i for i, group in enumerate(recent) if group == 0), default=-1)
    latest_new = max((i for i, group in enumerate(recent) if group == 1), default=-1)
    gain = weighted("gain")
    lower = weighted("lower")
    selectors = {
        "keep": "keep",
        "replace": "replace",
        "mean": "mean",
        "none": "none",
        "frequency": "replace" if frequency[1] > frequency[0] else "keep",
        "recency": "replace" if latest_new > latest_old else "keep",
        "surrogate_gain": "replace" if gain is not None and gain > 0 else "keep",
        "surrogate_lower": "replace" if lower is not None and lower > 0 else "keep",
    }
    return {
        "recent_group_ids": recent,
        "group_frequency": {str(group): frequency[group] for group in GROUPS},
        "past_only_selectors": selectors,
        "surrogate": {"old_zero_vs_old_parent": old_score, "new_parent_vs_zero": new_score,
                      "frequency_weighted_gain": gain, "frequency_weighted_lower": lower},
        "support_rows": {"old": len(old_rows["action"]), "new": len(new_rows["action"])},
        "decision_source_groups": [0, 1],
    }


def semantic_hash(value: Any) -> str:
    """Stable recursive hash for checkpoint state, including RNG and tensors."""
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"array\0" + str(array.dtype).encode() + repr(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda value: str(value)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple, deque)):
            digest.update(b"sequence\0" + str(len(item)).encode())
            for part in item:
                visit(part)
        elif isinstance(item, bytes):
            digest.update(b"bytes\0" + len(item).to_bytes(8, "little") + item)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode() + b"\0" + repr(item).encode())
        else:
            raise TypeError(f"cannot hash {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


__all__ = [
    "ACTIONS", "BoundedGroupEvidence", "FUTURE_TEMPLATES", "GROUPS", "HISTORY_TEMPLATES",
    "apply_action", "bootstrap_schedule", "common_mean_effective", "compensated_single_parent_install",
    "decide_from_past", "effective_matrices", "pair_mapping", "prefix_schedule", "semantic_hash",
    "task_name", "template_schedule", "validate_config",
]

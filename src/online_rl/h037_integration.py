"""Reusable, fail-closed integration for the H037 pre-reset handoff.

This module owns only the bounded H037 evidence/controller state and the
transaction around a single candidate commit. A runner calls ``record_rollout``
on finalized real rollouts before changing the actor used to sample them, then
calls ``pre_reset`` after its allocation cache has resolved an eviction and
before it clears the victim slot. The latter returns one JSON-friendly event
for immediate external logging; events are deliberately not accumulated in
checkpoint state.

Evidence policy: query the exact slot generation, task, and logical group. Of
those rows, use only rows recorded under the current parent version. This
strict equality is conservative: observations recorded under an older parent
are not treated as anchors for a newer parent. Preserve insertion order within
each resident stream, reserve its trailing time block for holdout, and only
then combine same-group resident streams into the old-prior train/holdout sets.
The old-prior anchors therefore always come from a currently resident
same-group local; no such resident means fail closed. Route repair writes a
one-based slot ID only into copied observations.

The complete learner/package-size check is supplied by the runner. A candidate
cannot commit without a callback that verifies the projected serialized
package, including this controller's evidence and the proposed parent version.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import math
import random
import time
from typing import Any, Callable

import numpy as np
import torch

from .h037_candidate import propose_h037
from .h037_evidence import H037Evidence
from .write_decision_policy import decode_actor


_FORMAT = "h037_controller_v1"
_CANDIDATE_KEYS = frozenset(
    {
        "lambda_old",
        "ridge",
        "max_iter",
        "max_eval",
        "delta_v",
        "epsilon_old",
        "epsilon_res",
        "codec",
        "actor_budget_bytes",
        "compensation_atol",
    }
)
_CONTROLLER_KEYS = frozenset(
    {
        "holdout_fraction",
        "min_victim_train_rows",
        "min_victim_holdout_rows",
        "min_old_train_rows",
        "min_old_holdout_rows",
        "min_resident_holdout_rows",
    }
)


def _strict_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: Any, name: str, *, minimum: float, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= minimum if strict else result < minimum:
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("H037 config must be a mapping")
    allowed = _CANDIDATE_KEYS | _CONTROLLER_KEYS | {"format"}
    missing = sorted((allowed - {"format"}) - set(config))
    unexpected = sorted(set(config) - allowed)
    if missing:
        raise ValueError(f"H037 config is missing required keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"H037 config has unsupported keys: {', '.join(map(str, unexpected))}")
    if config.get("format") != _FORMAT:
        raise ValueError(f"H037 config format must be {_FORMAT!r}")

    result: dict[str, Any] = {"format": _FORMAT}
    for key in ("lambda_old", "ridge", "epsilon_old", "epsilon_res"):
        result[key] = _finite_real(config[key], key, minimum=0.0)
    result["delta_v"] = _finite_real(config["delta_v"], "delta_v", minimum=0.0, strict=True)
    result["compensation_atol"] = _finite_real(
        config["compensation_atol"], "compensation_atol", minimum=0.0, strict=True
    )
    for key in ("max_iter", "max_eval"):
        result[key] = _strict_int(config[key], key, minimum=1)
    result["actor_budget_bytes"] = _strict_int(config["actor_budget_bytes"], "actor_budget_bytes", minimum=0)
    codec = config["codec"]
    if not isinstance(codec, str) or codec not in {"float32", "int8"}:
        raise ValueError("codec must be 'float32' or 'int8'")
    result["codec"] = codec

    result["holdout_fraction"] = _finite_real(
        config["holdout_fraction"], "holdout_fraction", minimum=0.0, strict=True
    )
    if result["holdout_fraction"] >= 1.0:
        raise ValueError("holdout_fraction must be less than 1")
    for key in (
        "min_victim_train_rows",
        "min_victim_holdout_rows",
        "min_old_train_rows",
        "min_old_holdout_rows",
        "min_resident_holdout_rows",
    ):
        result[key] = _strict_int(config[key], key, minimum=1)
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(part) for part in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _integer_vector(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return raw.astype(np.int64, copy=False)


def _observations(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric observations") from exc
    if result.ndim != 2 or result.shape[1] != 13:
        raise ValueError(f"{name} must have shape (N, 13)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result, dtype=np.float32)


def _routed_copy(observations: np.ndarray, slot: int) -> np.ndarray:
    result = np.array(observations, dtype=np.float32, copy=True, order="C")
    result[:, -1] = np.float32(slot + 1)
    return result


def _time_split(
    observations: np.ndarray,
    *,
    holdout_fraction: float,
    min_train: int,
    min_holdout: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    count = int(len(observations))
    if count < min_train + min_holdout:
        return None
    holdout_count = max(min_holdout, int(math.ceil(count * holdout_fraction)))
    holdout_count = min(holdout_count, count - min_train)
    if holdout_count < min_holdout:
        return None
    split = count - holdout_count
    return observations[:split].copy(), observations[split:].copy()


class H037Controller:
    """Manage H037 evidence, one eligible candidate attempt, and atomic commit.

    ``config`` uses format ``h037_controller_v1`` and requires every optimizer,
    acceptance, codec, sample-floor, and split value explicitly. The candidate
    subset is passed to ``propose_h037`` without controller-only keys.

    ``pre_reset`` returns the complete event for that invocation. The runner
    should append that mapping to its JSONL event stream immediately. There is
    no unbounded in-memory event history; ``last_event`` is only a convenience
    for interactive inspection and is not serialized.
    """

    def __init__(self, config: Mapping[str, Any], evidence: H037Evidence) -> None:
        self.config = _validate_config(config)
        if not callable(getattr(evidence, "record", None)) or not callable(getattr(evidence, "get", None)):
            raise TypeError("evidence must implement record/get and H037 state serialization")
        if not callable(getattr(evidence, "state_dict", None)):
            raise TypeError("evidence must implement state_dict and nbytes")
        evidence_nbytes = getattr(evidence, "nbytes", None)
        if isinstance(evidence_nbytes, (bool, np.bool_)) or not isinstance(evidence_nbytes, (int, np.integer)) or int(evidence_nbytes) < 0:
            raise TypeError("evidence.nbytes must be a non-negative integer property")
        self.evidence = evidence
        self.parent_version = 0
        self.parent_group: int | None = None
        self.parent_fingerprint: str | None = None
        self.eligible_attempt_used = False
        self.counters = {
            "pre_reset_calls": 0,
            "skipped": 0,
            "candidate_attempts": 0,
            "rejected": 0,
            "accepted": 0,
            "errors": 0,
            "rollout_rows_recorded": 0,
            "rollout_calls": 0,
        }
        self.last_event: dict[str, Any] | None = None

    @property
    def error_count(self) -> int:
        return int(self.counters["errors"])

    @property
    def candidate_config(self) -> dict[str, Any]:
        return {key: self.config[key] for key in sorted(_CANDIDATE_KEYS)}

    def state_dict(self) -> dict[str, Any]:
        """Return bounded, resumable controller state; event logs stay external."""
        return {
            "format": _FORMAT,
            "config": copy.deepcopy(self.config),
            "parent_version": int(self.parent_version),
            "parent_group": self.parent_group,
            "parent_fingerprint": self.parent_fingerprint,
            "eligible_attempt_used": bool(self.eligible_attempt_used),
            "counters": {key: int(value) for key, value in self.counters.items()},
            "evidence": self.evidence.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
        evidence: H037Evidence | None = None,
    ) -> "H037Controller":
        if not isinstance(state, Mapping) or state.get("format") != _FORMAT:
            raise ValueError(f"unsupported H037 controller state; expected {_FORMAT}")
        stored_config = _validate_config(state["config"])
        selected_config = stored_config if config is None else _validate_config(config)
        if selected_config != stored_config:
            raise ValueError("restored H037 config does not match checkpoint config")
        evidence_state = state.get("evidence")
        if evidence is None:
            evidence = H037Evidence.from_state_dict(evidence_state)
        else:
            if not isinstance(evidence_state, Mapping):
                raise ValueError("checkpoint is missing H037 evidence state")
            if not cls._state_equal(evidence.state_dict(), evidence_state):
                raise ValueError("supplied H037 evidence does not match checkpoint state")
        result = cls(selected_config, evidence)
        result.parent_version = _strict_int(state["parent_version"], "parent_version", minimum=0)
        parent_group = state.get("parent_group")
        if parent_group is not None:
            result.parent_group = _strict_int(parent_group, "parent_group", minimum=-2)
        fingerprint = state.get("parent_fingerprint")
        if fingerprint is not None and (not isinstance(fingerprint, str) or len(fingerprint) != 64):
            raise ValueError("parent_fingerprint must be a 64-character SHA-256 hex string or None")
        result.parent_fingerprint = fingerprint
        result.eligible_attempt_used = bool(state["eligible_attempt_used"])
        counters = state.get("counters")
        if not isinstance(counters, Mapping) or set(counters) != set(result.counters):
            raise ValueError("H037 controller counters have an invalid schema")
        result.counters = {
            key: _strict_int(counters[key], f"counters.{key}", minimum=0)
            for key in result.counters
        }
        result.last_event = None
        return result

    @staticmethod
    def _state_equal(left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and left.device == right.device
                and bool(torch.equal(left, right))
            )
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                return (
                    isinstance(left, np.ndarray)
                    and isinstance(right, np.ndarray)
                    and left.shape == right.shape
                    and left.dtype == right.dtype
                    and bool(np.array_equal(left, right))
                )
            except (TypeError, ValueError):
                return False
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            return (
                isinstance(left, Mapping)
                and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(H037Controller._state_equal(left[key], right[key]) for key in left)
            )
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            return (
                isinstance(left, (list, tuple))
                and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(H037Controller._state_equal(a, b) for a, b in zip(left, right))
            )
        return bool(left == right)

    @staticmethod
    def _state_view(value: Any) -> Any:
        """Make a comparable copy of mutable runner state without running it."""
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.random.Generator):
            return {
                "__type__": "numpy.Generator",
                "bit_generator": type(value.bit_generator).__qualname__,
                "state": H037Controller._state_view(value.bit_generator.state),
            }
        if isinstance(value, Mapping):
            return {key: H037Controller._state_view(part) for key, part in value.items()}
        if isinstance(value, tuple):
            return tuple(H037Controller._state_view(part) for part in value)
        if isinstance(value, list):
            return [H037Controller._state_view(part) for part in value]
        if isinstance(value, (set, frozenset)):
            return ("__set__", tuple(sorted((H037Controller._state_view(part) for part in value), key=repr)))
        if callable(value):
            return {"__callable_id__": id(value)}
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "attributes": H037Controller._state_view(attributes),
            }
        return copy.deepcopy(value)

    @staticmethod
    def _capture_agent_fields(agent: Any) -> dict[str, Any]:
        excluded = {"core", "optimizer", "h037_controller"}
        attributes = vars(agent)
        values = {name: value for name, value in attributes.items() if name not in excluded}
        return {
            "names": set(values),
            "references": dict(values),
            "copies": {name: copy.deepcopy(value) for name, value in values.items()},
            "views": {name: H037Controller._state_view(value) for name, value in values.items()},
            "core": getattr(agent, "core", None),
            "optimizer": getattr(agent, "optimizer", None),
            "controller": getattr(agent, "h037_controller", None),
        }

    @staticmethod
    def _agent_fields_match(agent: Any, snapshot: Mapping[str, Any]) -> bool:
        excluded = {"core", "optimizer", "h037_controller"}
        attributes = vars(agent)
        values = {name: value for name, value in attributes.items() if name not in excluded}
        if set(values) != snapshot["names"]:
            return False
        if getattr(agent, "core", None) is not snapshot["core"]:
            return False
        if getattr(agent, "optimizer", None) is not snapshot["optimizer"]:
            return False
        if getattr(agent, "h037_controller", None) is not snapshot["controller"]:
            return False
        return all(
            H037Controller._state_equal(
                H037Controller._state_view(values[name]), snapshot["views"][name]
            )
            for name in values
        )

    @staticmethod
    def _restore_value(reference: Any, saved: Any) -> Any:
        """Restore common mutable values in place, retaining wrapper identities."""
        if isinstance(reference, np.random.Generator) and isinstance(saved, np.random.Generator):
            reference.bit_generator.state = copy.deepcopy(saved.bit_generator.state)
            return reference
        if isinstance(reference, np.ndarray) and isinstance(saved, np.ndarray):
            if reference.shape == saved.shape and reference.dtype == saved.dtype:
                reference[...] = saved
                return reference
        if isinstance(reference, dict) and isinstance(saved, dict):
            reference.clear()
            reference.update(copy.deepcopy(saved))
            return reference
        if isinstance(reference, list) and isinstance(saved, list):
            reference[:] = copy.deepcopy(saved)
            return reference
        if isinstance(reference, set) and isinstance(saved, set):
            reference.clear()
            reference.update(copy.deepcopy(saved))
            return reference
        if type(reference) is type(saved) and isinstance(getattr(reference, "__dict__", None), dict):
            reference.__dict__.clear()
            reference.__dict__.update(copy.deepcopy(vars(saved)))
            return reference
        return copy.deepcopy(saved)

    @staticmethod
    def _restore_agent_fields(agent: Any, snapshot: Mapping[str, Any]) -> None:
        excluded = {"core", "optimizer", "h037_controller"}
        attributes = vars(agent)
        for name in list(attributes):
            if name not in excluded and name not in snapshot["names"]:
                del attributes[name]
        for name in snapshot["names"]:
            reference = snapshot["references"][name]
            restored = H037Controller._restore_value(reference, snapshot["copies"][name])
            setattr(agent, name, restored)
        if snapshot["core"] is not None:
            agent.core = snapshot["core"]
        if snapshot["optimizer"] is not None:
            agent.optimizer = snapshot["optimizer"]
        if snapshot["controller"] is not None:
            agent.h037_controller = snapshot["controller"]

    def _parent_identity(self, agent: Any) -> tuple[int, str]:
        group = int(agent.parent_group)
        shared = np.asarray(agent.core.read_shared_matrix(), dtype=np.float32)
        if shared.shape != (1, 9) or not np.isfinite(shared).all():
            raise ValueError("H037 requires one finite 3 by 3 shared parent")
        digest = hashlib.sha256()
        digest.update(np.asarray([group], dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(shared[0], dtype="<f4").tobytes())
        return group, digest.hexdigest()

    def _sync_parent_identity(self, agent: Any) -> None:
        group, fingerprint = self._parent_identity(agent)
        if self.parent_fingerprint is None:
            self.parent_group = group
            self.parent_fingerprint = fingerprint
            return
        if group != self.parent_group or fingerprint != self.parent_fingerprint:
            self.parent_version += 1
            self.parent_group = group
            self.parent_fingerprint = fingerprint

    def record_rollout(
        self,
        agent: Any,
        rows: Sequence[Any],
        *,
        slot: int,
        generation: int,
        task: str,
        group: int,
    ) -> dict[str, Any]:
        """Record copied real sample observations for one finalized rollout.

        Call before the PPO update for ``rows``. The parent fingerprint is
        synchronized at that boundary, so the recorded version identifies the
        policy which generated these observations. The input rows and their
        observations are never changed.
        """
        slot_id = _strict_int(slot, "slot", minimum=0)
        generation_id = _strict_int(generation, "generation", minimum=0)
        group_id = _strict_int(group, "group", minimum=0)
        if group_id not in (0, 1):
            raise ValueError("group must be zero or one")
        task_name = str(task)
        if not task_name:
            raise ValueError("task must be a nonempty string")
        self._sync_parent_identity(agent)
        self.counters["rollout_calls"] += 1
        copied: list[np.ndarray] = []
        for index, row in enumerate(rows):
            row_slot = getattr(row, "slot", slot_id)
            row_generation = getattr(row, "generation", generation_id)
            row_task = getattr(row, "task_name", task_name)
            row_group = getattr(row, "group_id", group_id)
            if int(row_slot) != slot_id or int(row_generation) != generation_id:
                raise ValueError(f"rollout row {index} does not match slot/generation")
            if str(row_task) != task_name or int(row_group) != group_id:
                raise ValueError(f"rollout row {index} does not match task/group")
            observation = _observations(np.asarray(row.obs).reshape(1, -1), f"rows[{index}].obs")
            routed = observation[0].copy()
            routed[-1] = np.float32(slot_id + 1)
            copied.append(routed)
        if not copied:
            return {"recorded": 0, "parent_version": int(self.parent_version)}
        matrix = np.stack(copied).astype(np.float32, copy=False)
        self.evidence.record(
            slot_id,
            generation_id,
            task_name,
            group_id,
            int(self.parent_version),
            matrix,
        )
        count = int(len(matrix))
        self.counters["rollout_rows_recorded"] += count
        return {"recorded": count, "parent_version": int(self.parent_version)}

    def _read_current_evidence(
        self,
        *,
        slot: int,
        generation: int,
        task: str,
        group: int,
        route_slot: int,
    ) -> tuple[np.ndarray, dict[str, int]]:
        batch = self.evidence.get(slot, generation, task, group)
        if not isinstance(batch, Mapping):
            raise ValueError("H037 evidence.get must return a mapping")
        observations = _observations(batch.get("observations"), "evidence observations")
        versions = _integer_vector(batch.get("parent_versions"), "evidence parent_versions")
        if len(versions) != len(observations):
            raise ValueError("H037 observations and parent_versions have different row counts")
        current = observations[versions == int(self.parent_version)]
        return _routed_copy(current, route_slot), {
            "total_rows": int(len(observations)),
            "current_parent_rows": int(len(current)),
            "excluded_parent_rows": int(len(observations) - len(current)),
        }

    def _build_candidate_evidence(
        self,
        agent: Any,
        allocation: Mapping[str, Any],
        victim_slot: int,
        parent_group: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any], str | None, tuple[int, ...]]:
        generation = int(agent.slot_generation[victim_slot])
        task = str(allocation["evicted_task"])
        victim_rows, victim_stats = self._read_current_evidence(
            slot=victim_slot,
            generation=generation,
            task=task,
            group=parent_group,
            route_slot=victim_slot,
        )
        victim_split = _time_split(
            victim_rows,
            holdout_fraction=self.config["holdout_fraction"],
            min_train=self.config["min_victim_train_rows"],
            min_holdout=self.config["min_victim_holdout_rows"],
        )
        stats: dict[str, Any] = {"victim": victim_stats}
        if victim_split is None:
            return None, stats, "insufficient_victim_evidence", ()
        victim_train, victim_holdout = victim_split

        tags = np.asarray(agent.slot_group_tags)
        if tags.shape != (int(agent.slots),):
            return None, stats, "invalid_logical_group_tags", ()
        if not hasattr(agent, "cache") or agent.cache is None:
            return None, stats, "task_cache_unavailable", ()
        resident_slots: list[int] = []
        resident_train: list[np.ndarray] = []
        resident_holdout: dict[int, np.ndarray] = {}
        resident_stats: dict[str, Any] = {}
        for resident_slot in range(int(agent.slots)):
            if resident_slot == victim_slot:
                continue
            physical = int(agent.core.group_for_slot(resident_slot))
            logical = int(tags[resident_slot])
            if physical == 0 and logical != parent_group:
                return None, stats, "resident_group_binding_mismatch", ()
            if physical != 0 or logical != parent_group:
                continue
            resident_task = agent.cache.slot_to_task[resident_slot]
            if resident_task is None:
                return None, stats, "bound_resident_missing_task", ()
            resident_generation = int(agent.slot_generation[resident_slot])
            rows, row_stats = self._read_current_evidence(
                slot=resident_slot,
                generation=resident_generation,
                task=str(resident_task),
                group=parent_group,
                route_slot=resident_slot,
            )
            split = _time_split(
                rows,
                holdout_fraction=self.config["holdout_fraction"],
                min_train=1,
                min_holdout=self.config["min_resident_holdout_rows"],
            )
            resident_stats[str(resident_slot)] = row_stats
            if split is None:
                return None, stats, f"insufficient_resident_evidence_slot_{resident_slot}", ()
            train_rows, holdout_rows = split
            resident_slots.append(resident_slot)
            resident_train.append(train_rows)
            resident_holdout[resident_slot] = holdout_rows

        stats["residents"] = resident_stats
        if not resident_slots:
            return None, stats, "no_same_group_resident_anchor", ()
        old_train_all = np.concatenate(resident_train, axis=0)
        old_holdout_all = np.concatenate([resident_holdout[index] for index in resident_slots], axis=0)
        if len(old_train_all) < self.config["min_old_train_rows"]:
            return None, stats, "insufficient_old_train_anchors", tuple(resident_slots)
        if len(old_holdout_all) < self.config["min_old_holdout_rows"]:
            return None, stats, "insufficient_old_holdout_anchors", tuple(resident_slots)
        evidence = {
            "victim_train": victim_train.copy(),
            "victim_holdout": victim_holdout.copy(),
            "old_train": old_train_all.copy(),
            "old_holdout": old_holdout_all.copy(),
            "resident_holdout": {
                slot: resident_holdout[slot].copy() for slot in resident_slots
            },
        }
        stats["selected"] = {
            "victim_train_rows": int(len(victim_train)),
            "victim_holdout_rows": int(len(victim_holdout)),
            "old_train_rows": int(len(old_train_all)),
            "old_holdout_rows": int(len(old_holdout_all)),
            "resident_holdout_rows": {str(slot): int(len(value)) for slot, value in resident_holdout.items()},
            "resident_slots": list(resident_slots),
            "parent_version": int(self.parent_version),
        }
        return evidence, stats, None, tuple(resident_slots)

    @staticmethod
    def _has_unfinished_rollout(agent: Any) -> bool:
        for name in ("rollout", "holdout_rows", "pending_rollout"):
            value = getattr(agent, name, None)
            if value is not None and len(value) > 0:
                return True
        if getattr(agent, "rollout_phase", None) is not None:
            return True
        if getattr(agent, "cycle_state", None) is not None:
            return True
        return False

    @staticmethod
    def _core_snapshot(core: Any) -> dict[str, Any]:
        return {
            "state": {key: value.detach().clone() for key, value in core.state_dict().items()},
            "shared_frozen": getattr(core, "_shared_frozen", None),
            "requires_grad": {name: bool(parameter.requires_grad) for name, parameter in core.named_parameters()},
            "gradients": {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in core.named_parameters()
            },
            "module_training": {name: bool(module.training) for name, module in core.named_modules()},
            "behavior_attributes": {
                name: copy.deepcopy(getattr(core, name))
                for name in ("temperature", "epsilon", "local_slots", "shared_groups")
                if hasattr(core, name)
            },
        }

    @staticmethod
    def _core_matches_snapshot(core: Any, snapshot: Mapping[str, Any]) -> bool:
        current = core.state_dict()
        before = snapshot["state"]
        if set(current) != set(before):
            return False
        if any(not torch.equal(current[key], before[key]) for key in before):
            return False
        if getattr(core, "_shared_frozen", None) != snapshot["shared_frozen"]:
            return False
        current_flags = {name: bool(parameter.requires_grad) for name, parameter in core.named_parameters()}
        if current_flags != snapshot["requires_grad"]:
            return False
        current_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach()
            for name, parameter in core.named_parameters()
        }
        if set(current_gradients) != set(snapshot["gradients"]):
            return False
        for name, gradient in current_gradients.items():
            before = snapshot["gradients"][name]
            if gradient is None or before is None:
                if gradient is not before:
                    return False
            elif not torch.equal(gradient, before):
                return False
        if {name: bool(module.training) for name, module in core.named_modules()} != snapshot["module_training"]:
            return False
        if any(getattr(core, name, object()) != value for name, value in snapshot["behavior_attributes"].items()):
            return False
        return True

    @staticmethod
    def _capture_transaction(agent: Any, controller: "H037Controller") -> dict[str, Any]:
        optimizer = agent.optimizer
        torch_cuda_rng = None
        if torch.cuda.is_initialized():
            torch_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        return {
            "core_object": agent.core,
            "optimizer_object": optimizer,
            "core": H037Controller._core_snapshot(agent.core),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "optimizer_step_count": copy.deepcopy(getattr(optimizer, "step_count", 0)),
            "agent_fields": H037Controller._capture_agent_fields(agent),
            "controller_version": int(controller.parent_version),
            "controller_group": controller.parent_group,
            "controller_fingerprint": controller.parent_fingerprint,
            "controller_evidence_object": controller.evidence,
            "controller_evidence": copy.deepcopy(controller.evidence.state_dict()),
            "controller_counters": dict(controller.counters),
            "controller_attempt_used": bool(controller.eligible_attempt_used),
            "controller_last_event": copy.deepcopy(controller.last_event),
            "numpy_rng": copy.deepcopy(np.random.get_state()),
            "python_rng": copy.deepcopy(random.getstate()),
            "torch_rng": torch.get_rng_state().clone(),
            "torch_cuda_rng": torch_cuda_rng,
        }

    def _transaction_matches(self, agent: Any, snapshot: Mapping[str, Any]) -> bool:
        if agent.core is not snapshot["core_object"] or agent.optimizer is not snapshot["optimizer_object"]:
            return False
        if not self._core_matches_snapshot(agent.core, snapshot["core"]):
            return False
        if not self._state_equal(agent.optimizer.state_dict(), snapshot["optimizer"]):
            return False
        if getattr(agent.optimizer, "step_count", 0) != snapshot["optimizer_step_count"]:
            return False
        if not self._agent_fields_match(agent, snapshot["agent_fields"]):
            return False
        if self.parent_version != snapshot["controller_version"] or self.parent_group != snapshot["controller_group"]:
            return False
        if self.parent_fingerprint != snapshot["controller_fingerprint"]:
            return False
        if self.evidence is not snapshot["controller_evidence_object"]:
            return False
        if not self._state_equal(self.evidence.state_dict(), snapshot["controller_evidence"]):
            return False
        if self.counters != snapshot["controller_counters"]:
            return False
        if self.eligible_attempt_used != snapshot["controller_attempt_used"]:
            return False
        if not self._state_equal(self.last_event, snapshot["controller_last_event"]):
            return False
        if not self._state_equal(np.random.get_state(), snapshot["numpy_rng"]):
            return False
        if not self._state_equal(random.getstate(), snapshot["python_rng"]):
            return False
        if not torch.equal(torch.get_rng_state(), snapshot["torch_rng"]):
            return False
        cuda_rng = snapshot["torch_cuda_rng"]
        if cuda_rng is not None:
            if not torch.cuda.is_initialized() or len(cuda_rng) != torch.cuda.device_count():
                return False
            if any(not torch.equal(current, saved) for current, saved in zip(torch.cuda.get_rng_state_all(), cuda_rng)):
                return False
        return True

    def _restore_transaction(self, agent: Any, snapshot: Mapping[str, Any]) -> None:
        agent.core = snapshot["core_object"]
        agent.optimizer = snapshot["optimizer_object"]
        agent.core.load_state_dict(snapshot["core"]["state"], strict=True)
        if snapshot["core"]["shared_frozen"] is not None:
            agent.core.freeze_shared(bool(snapshot["core"]["shared_frozen"]))
        for name, parameter in agent.core.named_parameters():
            if name in snapshot["core"]["requires_grad"]:
                parameter.requires_grad_(bool(snapshot["core"]["requires_grad"][name]))
            if name in snapshot["core"]["gradients"]:
                gradient = snapshot["core"]["gradients"][name]
                parameter.grad = None if gradient is None else gradient.detach().clone()
        for name, module in agent.core.named_modules():
            if name in snapshot["core"]["module_training"]:
                module.training = bool(snapshot["core"]["module_training"][name])
        for name, value in snapshot["core"]["behavior_attributes"].items():
            setattr(agent.core, name, copy.deepcopy(value))
        agent.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        agent.optimizer.step_count = copy.deepcopy(snapshot["optimizer_step_count"])
        self._restore_agent_fields(agent, snapshot["agent_fields"])
        self.parent_version = int(snapshot["controller_version"])
        self.parent_group = snapshot["controller_group"]
        self.parent_fingerprint = snapshot["controller_fingerprint"]
        evidence_object = snapshot["controller_evidence_object"]
        restored_evidence = H037Evidence.from_state_dict(snapshot["controller_evidence"])
        evidence_object.__dict__.clear()
        evidence_object.__dict__.update(copy.deepcopy(vars(restored_evidence)))
        self.evidence = evidence_object
        self.counters = dict(snapshot["controller_counters"])
        self.eligible_attempt_used = bool(snapshot["controller_attempt_used"])
        self.last_event = copy.deepcopy(snapshot["controller_last_event"])
        np.random.set_state(copy.deepcopy(snapshot["numpy_rng"]))
        random.setstate(copy.deepcopy(snapshot["python_rng"]))
        torch.set_rng_state(snapshot["torch_rng"].clone())
        if snapshot["torch_cuda_rng"] is not None and torch.cuda.is_initialized():
            torch.cuda.set_rng_state_all([state.clone() for state in snapshot["torch_cuda_rng"]])

    @staticmethod
    def _clear_optimizer_state(agent: Any, slots: Sequence[int], *, shared: bool) -> None:
        parameters = [agent.core._nodes[str(int(slot) + 1)].A for slot in slots]
        if shared:
            parameters.extend(list(agent.core.shared_parameters()))
        for parameter in parameters:
            agent.optimizer.state.pop(parameter, None)
            parameter.grad = None

    @staticmethod
    def _capacity_receipt(
        value: Any,
        *,
        proposed_parent_version: int,
    ) -> tuple[bool, dict[str, Any], str]:
        if not isinstance(value, Mapping):
            return False, {}, "capacity_check_unverifiable"
        required = ("fits", "package_bytes", "budget_bytes", "controller_state_included", "parent_version")
        if any(key not in value for key in required):
            return False, _json_value(value), "capacity_check_receipt_incomplete"
        try:
            package_bytes = _strict_int(value["package_bytes"], "capacity.package_bytes", minimum=0)
            budget_bytes = _strict_int(value["budget_bytes"], "capacity.budget_bytes", minimum=1)
            receipt_version = _strict_int(value["parent_version"], "capacity.parent_version", minimum=0)
        except ValueError:
            return False, _json_value(value), "capacity_check_receipt_invalid"
        if not isinstance(value["fits"], (bool, np.bool_)):
            return False, _json_value(value), "capacity_check_receipt_invalid"
        includes_state = value["controller_state_included"] is True
        if not includes_state or receipt_version != proposed_parent_version:
            return False, _json_value(value), "capacity_check_did_not_include_proposed_controller_state"
        # An explicit byte count above the stated cap is sufficient evidence
        # for the learner-package stop reason. A bare ``fits=False`` with
        # in-budget byte counts is an unverifiable/inconsistent receipt.
        if package_bytes > budget_bytes:
            return False, _json_value(value), "complete_package_exceeds_budget"
        if value["fits"] is not True:
            return False, _json_value(value), "capacity_check_receipt_inconsistent"
        return True, _json_value(value), "capacity_ok"

    def _event(
        self,
        *,
        status: str,
        reason: str,
        started_wall: float,
        started_cpu: float,
        **fields: Any,
    ) -> dict[str, Any]:
        result = {
            "format": "h037_event_v1",
            "status": str(status),
            "reason": str(reason),
            "parent_version": int(self.parent_version),
            "counters": {key: int(value) for key, value in self.counters.items()},
            "cost": {
                "wall_s": max(0.0, time.perf_counter() - started_wall),
                "cpu_s": max(0.0, time.process_time() - started_cpu),
            },
        }
        result.update(_json_value(fields))
        self.last_event = result
        return result

    def _skip(self, reason: str, started_wall: float, started_cpu: float, **fields: Any) -> dict[str, Any]:
        self.counters["skipped"] += 1
        return self._event(status="skip", reason=reason, started_wall=started_wall, started_cpu=started_cpu, **fields)

    def _execution_error(
        self,
        reason: str,
        started_wall: float,
        started_cpu: float,
        *,
        error: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        self.counters["errors"] += 1
        error_info = (
            {"type": "H037ExecutionError", "message": str(reason)}
            if error is None
            else _json_value(error)
        )
        return self._event(
            status="error",
            reason=reason,
            started_wall=started_wall,
            started_cpu=started_cpu,
            error=error_info,
            error_count=int(self.counters["errors"]),
            **fields,
        )

    def pre_reset(
        self,
        agent: Any,
        allocation: Mapping[str, Any],
        *,
        capacity_check: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Attempt one same-group handoff before the runner resets an evicted slot.

        Required callback contract::

            capacity_check(agent, *, controller, controller_state, actor_data,
                           actor_bytes, victim_slot, resident_slots,
                           old_parent_version, proposed_parent_version)

        It must measure the complete projected package and return a mapping
        containing ``fits``, ``package_bytes``, ``budget_bytes``,
        ``controller_state_included=True``, and the proposed ``parent_version``.
        The controller state supplied to it already contains the proposed
        parent version and bounded H037 evidence/counters, but excludes event
        logs. A missing, invalid, or unverifiable receipt is an execution error
        that the runner must persist and stop on.
        """
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        self.counters["pre_reset_calls"] += 1
        try:
            if not isinstance(allocation, Mapping):
                return self._execution_error(
                    "invalid_allocation",
                    started_wall,
                    started_cpu,
                    error={"type": "InvalidAllocation", "message": "allocation must be a mapping"},
                )
            allocated = allocation.get("allocated")
            if not isinstance(allocated, (bool, np.bool_)):
                return self._execution_error(
                    "invalid_allocation_flag",
                    started_wall,
                    started_cpu,
                    error={"type": "InvalidAllocation", "message": "allocated flag must be boolean"},
                )
            if not allocated:
                return self._execution_error(
                    "allocation_did_not_replace_slot",
                    started_wall,
                    started_cpu,
                    error={"type": "InvalidAllocation", "message": "pre_reset requires a completed replacement allocation"},
                )
            evicted_slot_value = allocation.get("evicted_slot")
            evicted_task = allocation.get("evicted_task")
            if evicted_slot_value is None or not isinstance(evicted_task, str) or not evicted_task:
                return self._execution_error(
                    "allocation_has_no_victim",
                    started_wall,
                    started_cpu,
                    error={"type": "InvalidAllocation", "message": "replacement allocation is missing a valid victim"},
                )
            victim_slot = _strict_int(evicted_slot_value, "evicted_slot", minimum=0)
            allocated_slot = _strict_int(allocation.get("slot"), "allocation.slot", minimum=0)
            if victim_slot >= int(agent.slots) or allocated_slot != victim_slot:
                return self._execution_error(
                    "allocation_victim_slot_mismatch",
                    started_wall,
                    started_cpu,
                    error={"type": "InvalidAllocation", "message": "allocation slot does not match its evicted slot"},
                    victim_slot=victim_slot,
                )
            if self._has_unfinished_rollout(agent):
                return self._execution_error(
                    "unfinished_rollout",
                    started_wall,
                    started_cpu,
                    error={"type": "PPOBoundaryError", "message": "pre_reset cannot run with an unfinished rollout"},
                    victim_slot=victim_slot,
                )
            if self.eligible_attempt_used:
                return self._skip("single_eligible_attempt_already_used", started_wall, started_cpu, victim_slot=victim_slot)
            if capacity_check is None or not callable(capacity_check):
                return self._execution_error(
                    "capacity_check_unavailable",
                    started_wall,
                    started_cpu,
                    error={"type": "CapacityCheckUnavailable", "message": "capacity_check callback is missing or not callable"},
                    victim_slot=victim_slot,
                )

            self._sync_parent_identity(agent)

            parent_group = int(agent.parent_group)
            if parent_group not in (0, 1) or parent_group != self.parent_group:
                return self._skip("parent_group_is_not_a_single_known_group", started_wall, started_cpu,
                                  victim_slot=victim_slot, parent_group=parent_group)
            logical_tags = np.asarray(agent.slot_group_tags)
            if logical_tags.shape != (int(agent.slots),):
                return self._execution_error(
                    "invalid_logical_group_tags",
                    started_wall,
                    started_cpu,
                    error={"type": "InvalidLearnerState", "message": "logical group tags do not match the slot count"},
                    victim_slot=victim_slot,
                )
            logical_victim = int(logical_tags[victim_slot])
            physical_victim = int(agent.core.group_for_slot(victim_slot))
            if logical_victim != parent_group:
                return self._skip("victim_logical_group_does_not_match_parent", started_wall, started_cpu,
                                  victim_slot=victim_slot, logical_group=logical_victim, parent_group=parent_group)
            if physical_victim != 0:
                return self._skip("victim_is_not_bound_to_physical_group_zero", started_wall, started_cpu,
                                  victim_slot=victim_slot, physical_group=physical_victim)
            if int(getattr(agent.core, "shared_groups", -1)) != 1 or int(agent.slots) != 2:
                return self._execution_error(
                    "unsupported_actor_layout",
                    started_wall,
                    started_cpu,
                    error={"type": "InvalidActorLayout", "message": "H037 requires two slots and one shared group"},
                    slots=int(agent.slots),
                    shared_groups=int(getattr(agent.core, "shared_groups", -1)),
                )

            candidate_evidence, evidence_stats, evidence_reason, resident_slots = self._build_candidate_evidence(
                agent, allocation, victim_slot, parent_group
            )
            if evidence_reason is not None or candidate_evidence is None:
                return self._skip(evidence_reason or "evidence_unavailable", started_wall, started_cpu,
                                  victim_slot=victim_slot, evidence=evidence_stats)

            # The single eligible attempt is consumed before any candidate call.
            # It remains consumed after rejection or exception to prevent retries.
            self.eligible_attempt_used = True
            self.counters["candidate_attempts"] += 1
            snapshot = self._capture_transaction(agent, self)
            old_parent_version = int(self.parent_version)
            local = np.asarray(agent.core.read_local_matrix(), dtype=np.float32).copy()
            shared = np.asarray(agent.core.read_shared_matrix(), dtype=np.float32).copy()
            groups = agent.core.slot_groups.detach().cpu().numpy().astype(np.int8, copy=True)
            try:
                candidate = propose_h037(
                    agent.core,
                    victim_slot=victim_slot,
                    local=local.copy(),
                    shared=shared.copy(),
                    groups=groups.copy(),
                    evidence=candidate_evidence,
                    config=self.candidate_config,
                )
                if not self._transaction_matches(agent, snapshot):
                    raise RuntimeError("propose_h037 modified transactional state despite its pure interface")
                if not isinstance(candidate, Mapping):
                    raise TypeError("propose_h037 must return a mapping")
                candidate_status = candidate.get("status")
                candidate_accepted = candidate.get("accepted")
                if candidate_status == "error" and candidate_accepted is False:
                    # A candidate execution fault is not a scientific refusal.
                    # Restore all tentative actor/controller/RNG state, retain
                    # the consumed attempt and error counter, and stop upstream.
                    self._restore_transaction(agent, snapshot)
                    self.counters["errors"] += 1
                    candidate_error = candidate.get("error")
                    if not isinstance(candidate_error, Mapping):
                        candidate_error = {
                            "type": "CandidateExecutionError",
                            "message": str(candidate.get("reason", "candidate execution failed")),
                        }
                    return self._event(
                        status="error",
                        reason=str(candidate.get("reason", "candidate_execution_error")),
                        started_wall=started_wall,
                        started_cpu=started_cpu,
                        victim_slot=victim_slot,
                        victim_task=str(evicted_task),
                        evidence=evidence_stats,
                        metrics=candidate.get("metrics", {}),
                        candidate_error=candidate_error,
                        error=_json_value(candidate_error),
                        parent_version_before=old_parent_version,
                        parent_version_after=int(self.parent_version),
                        error_count=int(self.counters["errors"]),
                    )
                if (
                    candidate_status == "reject"
                    and candidate_accepted is False
                    and "error" in candidate
                    and candidate.get("error") is None
                ):
                    # A rejection still consumes this one attempt, but any
                    # incidental candidate-side state change is rolled back.
                    self._restore_transaction(agent, snapshot)
                    self.counters["rejected"] += 1
                    return self._event(
                        status="reject",
                        reason=str(candidate.get("reason", "candidate_rejected")),
                        started_wall=started_wall,
                        started_cpu=started_cpu,
                        victim_slot=victim_slot,
                        victim_task=str(evicted_task),
                        evidence=evidence_stats,
                        metrics=candidate.get("metrics", {}),
                        error=None,
                        parent_version_before=old_parent_version,
                        parent_version_after=int(self.parent_version),
                    )
                if (
                    candidate_status != "accept"
                    or candidate_accepted is not True
                    or "error" not in candidate
                    or candidate.get("error") is not None
                ):
                    raise ValueError("propose_h037 returned an invalid candidate result protocol")
                actor_data = candidate.get("actor_bytes")
                if not isinstance(actor_data, (bytes, bytearray, memoryview)):
                    raise ValueError("accepted candidate is missing encoded actor bytes")
                actor_data = bytes(actor_data)
                if not actor_data or len(actor_data) > int(self.config["actor_budget_bytes"]):
                    raise ValueError("candidate actor payload is empty or exceeds actor budget")
                decoded_local, decoded_shared, decoded_groups = decode_actor(actor_data)
                if decoded_local.shape != local.shape or decoded_shared.shape != shared.shape or decoded_groups.shape != groups.shape:
                    raise ValueError("candidate actor layout does not match the live core")
                if not np.array_equal(decoded_groups.astype(np.int8), groups):
                    raise ValueError("candidate changed slot bindings outside the H037 parent update")

                # Install only the decoder's result. Candidate evaluation and
                # commit therefore share the exact serialized actor values.
                agent.core.install_local_matrix(decoded_local)
                agent.core.install_shared_matrix(decoded_shared)
                with torch.no_grad():
                    agent.core.slot_groups.copy_(torch.as_tensor(
                        decoded_groups.astype(np.int64),
                        device=agent.core.slot_groups.device,
                        dtype=agent.core.slot_groups.dtype,
                    ))
                self._clear_optimizer_state(agent, resident_slots, shared=True)
                proposed_parent_version = old_parent_version + 1
                self.parent_version = proposed_parent_version
                self.parent_group = parent_group
                self.parent_fingerprint = self._parent_identity(agent)[1]
                controller_state = self.state_dict()
                callback_snapshot = self._capture_transaction(agent, self)
                capacity_result = capacity_check(
                    agent,
                    controller=self,
                    controller_state=controller_state,
                    actor_data=actor_data,
                    actor_bytes=len(actor_data),
                    victim_slot=victim_slot,
                    resident_slots=resident_slots,
                    old_parent_version=old_parent_version,
                    proposed_parent_version=proposed_parent_version,
                )
                if not self._transaction_matches(agent, callback_snapshot):
                    raise RuntimeError("capacity_check modified transactional state")
                capacity_ok, capacity_info, capacity_reason = self._capacity_receipt(
                    capacity_result,
                    proposed_parent_version=proposed_parent_version,
                )
                if not capacity_ok:
                    self._restore_transaction(agent, snapshot)
                    if capacity_reason == "complete_package_exceeds_budget":
                        self.counters["skipped"] += 1
                        return self._event(
                            status="skip",
                            reason=capacity_reason,
                            started_wall=started_wall,
                            started_cpu=started_cpu,
                            victim_slot=victim_slot,
                            victim_task=str(evicted_task),
                            evidence=evidence_stats,
                            metrics=candidate.get("metrics", {}),
                            capacity=capacity_info,
                            error=None,
                            parent_version_before=old_parent_version,
                            parent_version_after=int(self.parent_version),
                            actor_bytes=len(actor_data),
                        )
                    self.counters["errors"] += 1
                    return self._event(
                        status="error",
                        reason=capacity_reason,
                        started_wall=started_wall,
                        started_cpu=started_cpu,
                        victim_slot=victim_slot,
                        victim_task=str(evicted_task),
                        evidence=evidence_stats,
                        metrics=candidate.get("metrics", {}),
                        capacity=capacity_info,
                        error={"type": "CapacityReceiptError", "message": capacity_reason},
                        parent_version_before=old_parent_version,
                        parent_version_after=int(self.parent_version),
                        actor_bytes=len(actor_data),
                        error_count=int(self.counters["errors"]),
                    )
                self.counters["accepted"] += 1
                return self._event(
                    status="accept",
                    reason=str(candidate.get("reason", "accepted")),
                    started_wall=started_wall,
                    started_cpu=started_cpu,
                    victim_slot=victim_slot,
                    victim_task=str(evicted_task),
                    evidence=evidence_stats,
                    metrics=candidate.get("metrics", {}),
                    error=None,
                    capacity=capacity_info,
                    actor_bytes=len(actor_data),
                    candidate_sha256=candidate.get("candidate_sha256"),
                    parent_version_before=old_parent_version,
                    parent_version_after=proposed_parent_version,
                    resident_optimizer_states_cleared=list(resident_slots),
                )
            except Exception as exc:
                self._restore_transaction(agent, snapshot)
                self.counters["errors"] += 1
                return self._event(
                    status="error",
                    reason=f"{type(exc).__name__}: {exc}",
                    started_wall=started_wall,
                    started_cpu=started_cpu,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    victim_slot=victim_slot,
                    victim_task=str(evicted_task),
                    evidence=evidence_stats,
                    parent_version_before=old_parent_version,
                    parent_version_after=int(self.parent_version),
                    error_count=int(self.counters["errors"]),
                )
        except Exception as exc:
            self.counters["errors"] += 1
            return self._event(
                status="error",
                reason=f"{type(exc).__name__}: {exc}",
                started_wall=started_wall,
                started_cpu=started_cpu,
                error={"type": type(exc).__name__, "message": str(exc)},
                error_count=int(self.counters["errors"]),
            )


__all__ = ["H037Controller"]

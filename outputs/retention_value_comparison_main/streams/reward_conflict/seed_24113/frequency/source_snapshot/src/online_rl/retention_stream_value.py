"""Pure repeated-retention actions and bounded online statistics.

This module does not run an environment or retain candidate policies. It
provides physical shared-parent interventions, compact per-task history, and
past-only selectors for a repeated-retention stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from online_rl.storage_decision import compensated_single_parent_install, effective_matrices


GROUPS = (0, 1)
ACTIONS = ("keep", "replace", "mean", "none")
BINDINGS = (-2, -1, 0, 1)
RECENT_CAPACITY = 8
FEATURE_NAMES = (
    "bias",
    "candidate_minus_incumbent_recent_group_frequency",
    "candidate_minus_incumbent_recovery_ewma_over_0_02",
    "normalized_log_candidate_minus_parent_source_fit",
    "decisions_since_parent_write_over_16",
    "recent_eviction_fraction8",
)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer")
    result = int(array)
    limits = np.iinfo(np.int64)
    if result < limits.min or result > limits.max:
        raise ValueError(f"{name} exceeds int64 capacity")
    return result


def _group(value: Any, name: str = "group") -> int:
    result = _integer(value, name)
    if result not in GROUPS:
        raise ValueError(f"{name} must be 0 or 1")
    return result


def _binding(value: Any, name: str = "binding") -> int:
    result = _integer(value, name)
    if result not in BINDINGS:
        raise ValueError(f"{name} must be one of {BINDINGS}")
    return result


def _matrix(value: Any, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3 by 3 matrix or length-9 vector") from exc
    if raw.shape == (9,):
        raw = raw.reshape(3, 3)
    if raw.shape != (3, 3) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite 3 by 3 matrix or length-9 vector")
    result = np.asarray(raw, dtype=np.float32).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _actor_arrays(local: Any, shared: Any, physical_groups: Any, logical_tags: Any):
    """Validate the single-parent actor layout and return its effective rows."""
    effective = effective_matrices(local, shared, physical_groups)
    tags = np.asarray(logical_tags)
    if tags.shape != (len(effective),):
        raise ValueError("logical_tags must have one entry per local slot")
    if tags.dtype.kind not in "iu" or np.any((tags < -1) | (tags > 1)):
        raise ValueError("logical_tags must contain only -1, 0, or 1")
    groups = np.asarray(physical_groups)
    if groups.dtype.kind not in "iu" or np.any((groups != -1) & (groups != 0)):
        raise ValueError("physical_groups must contain only -1 or 0")
    return effective


def apply_retention_action(
    local: Any,
    shared: Any,
    physical_groups: Any,
    logical_tags: Any,
    incumbent_binding: int,
    candidate: Any,
    candidate_group: int,
    action: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one repeated-retention action and preserve resident policies.

    ``incumbent_binding`` is ``-2`` for wildcard, ``-1`` for no parent, or
    group ``0`` / ``1``. ``keep`` returns copies of the current physical arrays
    directly, avoiding a needless install or codec round trip.
    """
    binding = _binding(incumbent_binding, "incumbent_binding")
    group = _group(candidate_group, "candidate_group")
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}")
    before = _actor_arrays(local, shared, physical_groups, logical_tags)
    local_array = np.asarray(local)
    shared_array = np.asarray(shared)
    physical_array = np.asarray(physical_groups)
    tags = np.asarray(logical_tags)
    current_parent = _matrix(shared_array[0], "shared parent")
    candidate_parent = _matrix(candidate, "candidate")

    if action == "keep":
        local_result = local_array.copy()
        shared_result = shared_array.copy()
        physical_result = physical_array.copy()
        assigned_slots = np.flatnonzero(physical_array == 0).astype(int).tolist()
        parent = current_parent
        parent_group = binding
    else:
        if action == "replace":
            parent, parent_group = candidate_parent, group
        elif action == "mean":
            incumbent = current_parent if binding != -1 else np.zeros((3, 3), dtype=np.float32)
            parent, parent_group = (incumbent + candidate_parent) / 2.0, -2
        else:
            parent, parent_group = np.zeros((3, 3), dtype=np.float32), -1
        local_result, shared_result, physical_result, _ = compensated_single_parent_install(
            local_array,
            shared_array,
            physical_array,
            tags,
            parent=parent,
            parent_group=parent_group,
        )
        assigned_slots = [
            slot for slot, logical_group in enumerate(tags.tolist())
            if parent_group == -2 or (parent_group >= 0 and int(logical_group) == parent_group)
        ]

    after = effective_matrices(local_result, shared_result, physical_result)
    per_slot = np.max(np.abs(after - before), axis=(1, 2, 3)).astype(float).tolist()
    details = {
        "action": action,
        "parent_group": int(parent_group),
        "incumbent_binding": binding,
        "candidate_group": group,
        "bound_slots": [int(slot) for slot in assigned_slots],
        "max_effective_preservation_error": float(max(per_slot, default=0.0)),
        "per_slot_preservation_error": per_slot,
    }
    return local_result, shared_result, physical_result, details


class BoundedRetentionStats:
    """Fixed-capacity task history for repeated parent-retention decisions.

    Bout records use the schedule fields ``task``, ``group``, ``allocated``,
    ``evicted_task``, and ``return_first_256``. Groups can be fixed at
    construction or learned once per task from its first record.
    """

    FORMAT_VERSION = 1
    __slots__ = (
        "tasks", "_task_index", "_task_groups", "seen", "visits",
        "recovery_ewma", "evictions", "recent_groups", "recent_evictions",
        "recent_length", "parent_source_fit", "parent_written_decision",
    )

    def __init__(self, tasks: Sequence[str], task_groups: Mapping[str, int] | None = None) -> None:
        if isinstance(tasks, (str, bytes)):
            raise ValueError("tasks must contain exactly six unique task names")
        try:
            task_names = tuple(tasks)
        except TypeError as exc:
            raise ValueError("tasks must contain exactly six unique task names") from exc
        if (
            len(task_names) != 6
            or any(not isinstance(task, str) or not task for task in task_names)
            or len(set(task_names)) != 6
        ):
            raise ValueError("tasks must contain exactly six unique nonempty task names")
        self.tasks = task_names
        self._task_index = {task: index for index, task in enumerate(task_names)}
        self._task_groups = np.full(6, -1, dtype=np.int8)
        if task_groups is not None:
            if not isinstance(task_groups, Mapping) or set(task_groups) != set(task_names):
                raise ValueError("task_groups must map exactly the six configured tasks")
            self._task_groups = np.asarray(
                [_group(task_groups[task], f"task_groups[{task!r}]") for task in task_names],
                dtype=np.int8,
            )
        self.seen = np.zeros(6, dtype=np.bool_)
        self.visits = np.zeros(6, dtype=np.int64)
        self.recovery_ewma = np.zeros(6, dtype=np.float64)
        self.evictions = np.zeros(6, dtype=np.int64)
        self.recent_groups = np.zeros(RECENT_CAPACITY, dtype=np.int8)
        self.recent_evictions = np.zeros(RECENT_CAPACITY, dtype=np.uint8)
        self.recent_length = 0
        self.parent_source_fit = 0.0
        self.parent_written_decision = -1

    @property
    def task_groups(self) -> dict[str, int]:
        """Return the currently known task-to-group assignments."""
        return {
            task: int(self._task_groups[index])
            for index, task in enumerate(self.tasks)
            if int(self._task_groups[index]) in GROUPS
        }

    @property
    def recent_group_values(self) -> np.ndarray:
        return self.recent_groups[: self.recent_length].copy()

    @property
    def recent_eviction_values(self) -> np.ndarray:
        return self.recent_evictions[: self.recent_length].copy()

    def _record_fields(
        self, record: Mapping[str, Any], *, store_group: bool = False,
    ) -> tuple[int, int, bool, str | None, float]:
        if not isinstance(record, Mapping):
            raise ValueError("bout record must be a mapping")
        task = record.get("task")
        if not isinstance(task, str) or task not in self._task_index:
            raise ValueError("bout task must be one of the six configured tasks")
        index = self._task_index[task]
        group_value = record.get("group")
        if group_value is None:
            group = int(self._task_groups[index])
            if group not in GROUPS:
                raise ValueError("bout record must provide group when task_groups is not fixed")
        else:
            group = _group(group_value)
            known_group = int(self._task_groups[index])
            if known_group in GROUPS and known_group != group:
                raise ValueError(f"task {task!r} changed its fixed group")
            if store_group:
                self._task_groups[index] = group

        allocated_value = record.get("allocated")
        if not isinstance(allocated_value, (bool, np.bool_)):
            raise ValueError("allocated must be a boolean")
        allocated = bool(allocated_value)
        evicted_value = record.get("evicted_task")
        if evicted_value in (None, ""):
            evicted_task = None
        elif isinstance(evicted_value, str) and evicted_value in self._task_index:
            evicted_task = evicted_value
        else:
            raise ValueError("evicted_task must be None or one of the six configured tasks")
        if evicted_task is not None and not allocated:
            raise ValueError("a natural eviction requires an allocated bout")

        if "return_first_256" not in record:
            raise ValueError("bout record must contain return_first_256")
        raw_return = np.asarray(record["return_first_256"])
        if raw_return.ndim != 0 or raw_return.dtype.kind not in "fiu":
            raise ValueError("return_first_256 must be a finite real number")
        reward_rate = float(raw_return) / 256.0
        if not np.isfinite(reward_rate):
            raise ValueError("return_first_256 must be a finite real number")
        return index, group, allocated, evicted_task, reward_rate

    def recurrence_info(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Describe recurrence and eviction before :meth:`observe_bout` mutates state."""
        index, group, allocated, evicted_task, reward_rate = self._record_fields(record)
        seen_before = bool(self.seen[index])
        return {
            "task": self.tasks[index],
            "group": group,
            "seen_before": seen_before,
            "visits_before": int(self.visits[index]),
            "allocated": allocated,
            "recurrence": bool(seen_before and allocated),
            "natural_eviction": evicted_task is not None,
            "evicted_task": evicted_task,
            "first256_reward_per_step": reward_rate,
        }

    def observe_bout(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Update bounded per-task EWMA and recent allocation/eviction history."""
        info = self.recurrence_info(record)
        index, group, _allocated, evicted_task, reward_rate = self._record_fields(record, store_group=True)
        if self.visits[index] == np.iinfo(np.int64).max:
            raise OverflowError("task visit count exceeds int64 capacity")
        self.recovery_ewma[index] = (
            reward_rate if not self.seen[index]
            else 0.75 * float(self.recovery_ewma[index]) + 0.25 * reward_rate
        )
        self.seen[index] = True
        self.visits[index] += 1
        if evicted_task is not None:
            victim_index = self._task_index[evicted_task]
            if self.evictions[victim_index] == np.iinfo(np.int64).max:
                raise OverflowError("task eviction count exceeds int64 capacity")
            self.evictions[victim_index] += 1

        evicted = np.uint8(evicted_task is not None)
        if self.recent_length < RECENT_CAPACITY:
            offset = self.recent_length
            self.recent_groups[offset] = group
            self.recent_evictions[offset] = evicted
            self.recent_length += 1
        else:
            self.recent_groups[:-1] = self.recent_groups[1:]
            self.recent_evictions[:-1] = self.recent_evictions[1:]
            self.recent_groups[-1] = group
            self.recent_evictions[-1] = evicted
        return info

    def record_parent_write(self, source_fit: float, decision_index: int) -> None:
        """Store the source-fit and decision index for the currently written parent."""
        fit_array = np.asarray(source_fit)
        if fit_array.ndim != 0 or fit_array.dtype.kind not in "fiu":
            raise ValueError("source_fit must be a finite nonnegative real number")
        fit = float(fit_array)
        if not np.isfinite(fit) or fit < 0:
            raise ValueError("source_fit must be a finite nonnegative real number")
        index = _integer(decision_index, "decision_index")
        if index < 0:
            raise ValueError("decision_index must be nonnegative")
        if index < self.parent_written_decision:
            raise ValueError("parent write decision indices must be nondecreasing")
        self.parent_source_fit = fit
        self.parent_written_decision = index

    def group_frequencies(self) -> dict[int, float]:
        """Return frequencies over the observed trailing group history."""
        recent = self.recent_groups[: self.recent_length]
        if self.recent_length == 0:
            return {group: 0.0 for group in GROUPS}
        return {group: float(np.count_nonzero(recent == group) / self.recent_length) for group in GROUPS}

    def features(
        self,
        candidate_group: int,
        incumbent_binding: int,
        candidate_source_fit: float,
        decision_index: int,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return six bounded float64 features and their stable names."""
        candidate_group = _group(candidate_group, "candidate_group")
        binding = _binding(incumbent_binding, "incumbent_binding")
        candidate_fit_array = np.asarray(candidate_source_fit)
        if candidate_fit_array.ndim != 0 or candidate_fit_array.dtype.kind not in "fiu":
            raise ValueError("candidate_source_fit must be a finite nonnegative real number")
        candidate_fit = float(candidate_fit_array)
        if not np.isfinite(candidate_fit) or candidate_fit < 0:
            raise ValueError("candidate_source_fit must be a finite nonnegative real number")
        decision = _integer(decision_index, "decision_index")
        if decision < 0:
            raise ValueError("decision_index must be nonnegative")

        frequencies = self.group_frequencies()
        candidate_frequency = frequencies[candidate_group]
        if binding == -2:
            incumbent_frequency = 1.0
            incumbent_groups = GROUPS
        elif binding == -1:
            incumbent_frequency = 0.0
            incumbent_groups = ()
        else:
            incumbent_frequency = frequencies[binding]
            incumbent_groups = (binding,)

        candidate_tasks = self.seen & (self._task_groups == candidate_group)
        incumbent_tasks = self.seen & np.isin(self._task_groups, np.asarray(incumbent_groups, dtype=np.int8))
        candidate_recovery = float(np.mean(self.recovery_ewma[candidate_tasks])) if np.any(candidate_tasks) else 0.0
        incumbent_recovery = float(np.mean(self.recovery_ewma[incumbent_tasks])) if np.any(incumbent_tasks) else 0.0
        recovery_gap = float(np.clip((candidate_recovery - incumbent_recovery) / 0.02, -2.0, 2.0))
        source_fit_gap = float(np.clip(
            (np.log1p(candidate_fit) - np.log1p(self.parent_source_fit)) / np.log(8193.0),
            -1.0,
            1.0,
        ))
        decisions_since_write = float(np.clip((decision - self.parent_written_decision) / 16.0, 0.0, 2.0))
        recent_eviction_fraction = (
            float(np.mean(self.recent_evictions[: self.recent_length]))
            if self.recent_length else 0.0
        )
        result = np.asarray([
            1.0,
            candidate_frequency - incumbent_frequency,
            recovery_gap,
            source_fit_gap,
            decisions_since_write,
            recent_eviction_fraction,
        ], dtype=np.float64)
        return result, FEATURE_NAMES

    def state_dict(self) -> dict[str, Any]:
        """Return a fixed-capacity numeric snapshot with task names as metadata."""
        return {
            "format_version": self.FORMAT_VERSION,
            "tasks": self.tasks,
            "task_groups": self._task_groups.copy(),
            "seen": self.seen.copy(),
            "visits": self.visits.copy(),
            "recovery_ewma": self.recovery_ewma.copy(),
            "evictions": self.evictions.copy(),
            "recent_groups": self.recent_groups.copy(),
            "recent_evictions": self.recent_evictions.copy(),
            "recent_length": int(self.recent_length),
            "parent_source_fit": float(self.parent_source_fit),
            "parent_written_decision": int(self.parent_written_decision),
        }

    @classmethod
    def from_state_dict(cls, state: Any) -> "BoundedRetentionStats":
        if not isinstance(state, Mapping) or state.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported retention stream stats state")
        result = cls(state.get("tasks"))
        groups = _int_array(state.get("task_groups"), "task_groups", (6,), np.int8)
        if np.any((groups < -1) | (groups > 1)):
            raise ValueError("task_groups must contain only -1, 0, or 1")
        result._task_groups = groups
        result.seen = _bool_array(state.get("seen"), "seen", (6,))
        result.visits = _int_array(state.get("visits"), "visits", (6,), np.int64)
        result.recovery_ewma = _finite_array(state.get("recovery_ewma"), "recovery_ewma", (6,))
        result.evictions = _int_array(state.get("evictions"), "evictions", (6,), np.int64)
        result.recent_groups = _int_array(state.get("recent_groups"), "recent_groups", (RECENT_CAPACITY,), np.int8)
        result.recent_evictions = _int_array(state.get("recent_evictions"), "recent_evictions", (RECENT_CAPACITY,), np.uint8)
        length = _integer(state.get("recent_length"), "recent_length")
        if not 0 <= length <= RECENT_CAPACITY:
            raise ValueError("recent_length must be in [0, 8]")
        if np.any(result.visits < 0) or np.any(result.evictions < 0):
            raise ValueError("visits and evictions cannot be negative")
        if not np.array_equal(result.seen, result.visits > 0):
            raise ValueError("seen must match whether visits is positive")
        if np.any((result.recent_groups[:length] < 0) | (result.recent_groups[:length] > 1)):
            raise ValueError("recent_groups must contain only 0 or 1")
        if np.any(result.recent_evictions > 1):
            raise ValueError("recent_evictions must contain only 0 or 1")
        if np.any(result.recent_groups[length:] != 0) or np.any(result.recent_evictions[length:] != 0):
            raise ValueError("unused recent-history capacity must be zero")
        result.recent_length = length

        fit_array = np.asarray(state.get("parent_source_fit"))
        if fit_array.ndim != 0 or fit_array.dtype.kind not in "fiu":
            raise ValueError("parent_source_fit must be finite and nonnegative")
        fit = float(fit_array)
        if not np.isfinite(fit) or fit < 0:
            raise ValueError("parent_source_fit must be finite and nonnegative")
        parent_decision = _integer(state.get("parent_written_decision"), "parent_written_decision")
        if parent_decision < -1:
            raise ValueError("parent_written_decision must be at least -1")
        result.parent_source_fit = fit
        result.parent_written_decision = parent_decision
        return result

    def numeric_nbytes(self) -> int:
        """Return bytes for the complete bounded numeric state."""
        arrays = (
            self._task_groups,
            self.seen,
            self.visits,
            self.recovery_ewma,
            self.evictions,
            self.recent_groups,
            self.recent_evictions,
        )
        scalar_bytes = 2 * np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize
        return int(sum(array.nbytes for array in arrays) + scalar_bytes)


def _bool_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a boolean array") from exc
    if raw.shape != shape or raw.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return np.array(raw, dtype=np.bool_, copy=True)


def _int_array(value: Any, name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer array") from exc
    if raw.shape != shape or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with shape {shape}")
    limits = np.iinfo(dtype)
    if raw.size and (np.any(raw < limits.min) or np.any(raw > limits.max)):
        raise ValueError(f"{name} exceeds {np.dtype(dtype).name} capacity")
    return np.asarray(raw, dtype=dtype).copy()


def _finite_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array with shape {shape}")
    result = np.asarray(raw, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _recent_group_frequencies(stats: BoundedRetentionStats) -> dict[int, float]:
    return stats.group_frequencies()


def frequency_action(
    stats: BoundedRetentionStats,
    candidate_group: int,
    incumbent_binding: int,
) -> str:
    """Replace only when the candidate group's recent frequency is larger."""
    if not isinstance(stats, BoundedRetentionStats):
        raise ValueError("stats must be BoundedRetentionStats")
    group = _group(candidate_group, "candidate_group")
    binding = _binding(incumbent_binding, "incumbent_binding")
    frequencies = _recent_group_frequencies(stats)
    candidate_frequency = frequencies[group]
    if binding == -2:
        incumbent_frequency = 1.0
    elif binding == -1:
        incumbent_frequency = 0.0
    else:
        incumbent_frequency = frequencies[binding]
    return "replace" if candidate_frequency > incumbent_frequency else "keep"


def _group_evidence(evidence: Any, group: int) -> Mapping[str, Any]:
    arrays = getattr(evidence, "arrays", None)
    if callable(arrays):
        result = arrays(group)
    elif isinstance(evidence, Mapping):
        result = evidence.get(group, evidence.get(str(group)))
    else:
        raise ValueError("evidence must provide arrays(group) or a group mapping")
    if not isinstance(result, Mapping):
        raise ValueError(f"evidence for group {group} is missing")
    return result


def generalized_lower_decision(
    evidence: Any,
    score_fn: Callable[[Mapping[str, Any], Any, Any], Mapping[str, Any]],
    stats: BoundedRetentionStats,
    incumbent: Any,
    parent_binding: int,
    candidate: Any,
    candidate_group: int,
) -> tuple[str, dict[str, Any]]:
    """Apply a groupwise lower-bound comparison for repeated retention.

    Each group is scored against the incumbent only when its binding covers
    that group, and against the candidate only for the candidate's group.
    This generalizes the old one-write old-parent/new-candidate comparison.
    """
    if not isinstance(stats, BoundedRetentionStats):
        raise ValueError("stats must be BoundedRetentionStats")
    if not callable(score_fn):
        raise ValueError("score_fn must be callable")
    binding = _binding(parent_binding, "parent_binding")
    group = _group(candidate_group, "candidate_group")
    incumbent_parent = _matrix(incumbent, "incumbent")
    candidate_parent = _matrix(candidate, "candidate")
    zero = np.zeros((3, 3), dtype=np.float32)
    frequencies = _recent_group_frequencies(stats)
    group_scores: dict[str, dict[str, Any]] = {}
    baselines: dict[str, list[list[float]]] = {}
    replacements: dict[str, list[list[float]]] = {}
    values: dict[int, float] = {}
    all_valid = True

    for current_group in GROUPS:
        baseline = incumbent_parent if binding == -2 or binding == current_group else zero
        replacement = candidate_parent if current_group == group else zero
        baselines[str(current_group)] = baseline.tolist()
        replacements[str(current_group)] = replacement.tolist()
        score = score_fn(_group_evidence(evidence, current_group), replacement, baseline)
        if not isinstance(score, Mapping):
            raise ValueError("score_fn must return a mapping containing lower")
        raw_lower = score.get("lower")
        try:
            lower = float(raw_lower)
        except (TypeError, ValueError, OverflowError):
            lower = float("nan")
        valid = bool(np.isfinite(lower))
        group_scores[str(current_group)] = {"lower": lower if valid else None, "valid": valid}
        if valid:
            values[current_group] = lower
        else:
            all_valid = False

    weighted_lower = (
        float(sum(frequencies[group_id] * values[group_id] for group_id in GROUPS))
        if all_valid else None
    )
    action = "replace" if weighted_lower is not None and weighted_lower > 0.0 else "keep"
    diagnostics = {
        "format": "generalized_repeated_retention_lower_v1",
        "action": action,
        "selector": "generalized_lower",
        "parent_binding": binding,
        "candidate_group": group,
        "recent_group_frequency": {str(key): frequencies[key] for key in GROUPS},
        "group_scores": group_scores,
        "frequency_weighted_lower": weighted_lower,
        "all_group_scores_valid": all_valid,
        "baseline_coefficients_by_group": baselines,
        "candidate_coefficients_by_group": replacements,
        "rule": "replace iff all group lower scores are finite and recent-frequency weighted lower is positive; otherwise keep",
        "comparison": "each group uses its currently bound incumbent and the candidate only in candidate_group",
    }
    return action, diagnostics


__all__ = [
    "ACTIONS",
    "BINDINGS",
    "FEATURE_NAMES",
    "BoundedRetentionStats",
    "apply_retention_action",
    "frequency_action",
    "generalized_lower_decision",
]

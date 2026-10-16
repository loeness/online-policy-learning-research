"""Bounded parent-group recycling for two-group shared write policies.

This helper only handles parent-capacity reuse. It does not infer task labels,
estimate utility, run training, or claim optimal organization. A caller must
finish the current PPO cycle and reset actor optimizer state before recycling.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from operator import index as as_index
from typing import Any

import numpy as np

from .write_decision_evidence import EvidenceProfile, GroupAssignment, RewardAwareGroups


_GROUPS = 2
_STATE_VERSION = 1
_CAPACITY_FAILURES = {"capacity_full", "max_groups_reached"}
_COUNTER_MAX = int(np.iinfo(np.uint64).max)


@dataclass(frozen=True)
class ParentRecycleResult:
    """Result of a bounded attempt to reuse one parent group."""

    recycled: bool
    reason: str
    grouper: RewardAwareGroups
    assignment: GroupAssignment
    group: int | None = None
    previous_group_version: int | None = None
    group_version: int | None = None
    affected_slots: tuple[int, ...] = ()
    removed_slots: tuple[int, ...] = ()
    added_slots: tuple[int, ...] = ()
    cleared_shadow_slots: tuple[int, ...] = ()
    affected_generations: tuple[tuple[int, int], ...] = ()
    seen_updates: tuple[tuple[int, bool], ...] = ()
    committed_profile_slots: tuple[int, ...] = ()


class ParentBudgetManager:
    """Fixed-size LRU and version counters for exactly two shared parents."""

    def __init__(self) -> None:
        self.last_used = np.zeros(_GROUPS, dtype=np.uint64)
        self.touch_count = np.zeros(_GROUPS, dtype=np.uint64)
        self.version = np.zeros(_GROUPS, dtype=np.uint64)

    @property
    def nbytes(self) -> int:
        return int(self.last_used.nbytes + self.touch_count.nbytes + self.version.nbytes)

    def touch(self, group: int, step: int) -> None:
        """Mark one parent used at a nondecreasing environment step."""
        group_index = _integer(group, "group", minimum=0, maximum=_GROUPS - 1)
        step_index = _integer(step, "step", minimum=0, maximum=_COUNTER_MAX)
        if step_index < int(self.last_used[group_index]):
            raise ValueError("step must not move backwards for a parent group")
        if int(self.touch_count[group_index]) >= _COUNTER_MAX:
            raise OverflowError("parent touch counter is exhausted")
        self.last_used[group_index] = np.uint64(step_index)
        self.touch_count[group_index] += np.uint64(1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "groups": _GROUPS,
            "last_used": self.last_used.copy(),
            "touch_count": self.touch_count.copy(),
            "parent_version": self.version.copy(),
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ParentBudgetManager":
        if not isinstance(state, Mapping) or state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported ParentBudgetManager state")
        if state.get("groups") != _GROUPS:
            raise ValueError("ParentBudgetManager state must contain exactly two groups")
        result = cls()
        fields = (
            ("last_used", result.last_used),
            ("touch_count", result.touch_count),
            ("parent_version", result.version),
        )
        for name, target in fields:
            value = np.asarray(state.get(name))
            if value.shape != (_GROUPS,) or value.dtype != np.uint64:
                raise ValueError(f"{name} must be a uint64 vector of length two")
            target[...] = value
        if "nbytes" in state and int(state["nbytes"]) != result.nbytes:
            raise ValueError("state nbytes does not match parent budget storage")
        return result

    def try_recycle(
        self,
        core,
        grouper: RewardAwareGroups,
        profiles: Mapping[str, Any],
        current_slot: int,
        slot_generations: Any,
        step: int,
    ) -> ParentRecycleResult:
        """Attempt a capacity-only recycle using visible profiles and slot IDs."""
        return try_recycle(
            self,
            core,
            grouper,
            profiles,
            current_slot=current_slot,
            slot_generations=slot_generations,
            step=step,
        )


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    try:
        result = as_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(result)


def _profile_arrays(profiles: Mapping[str, Any], slots: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not isinstance(profiles, Mapping):
        raise TypeError("profiles must be an index-based mapping of fixed-size arrays")
    specifications = {
        "movement_counts": ((slots, 3, 4, 4), np.dtype(np.uint16)),
        "reward_xtx": ((slots, 3, 3), np.dtype(np.float32)),
        "reward_xty": ((slots, 3), np.dtype(np.float32)),
        "reward_n": ((slots,), np.dtype(np.float32)),
        "seen": ((slots,), np.dtype(np.uint8)),
        "shadow_group": ((slots,), np.dtype(np.int8)),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in specifications.items():
        raw = profiles.get(name)
        if not isinstance(raw, np.ndarray):
            raise TypeError(f"profiles[{name!r}] must be a NumPy array")
        value = np.asarray(raw)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"profiles[{name!r}] must have dtype {dtype} and shape {shape}")
        arrays[name] = value
    for name in ("reward_xtx", "reward_xty", "reward_n"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"profiles[{name!r}] must contain finite values")
    if np.any(arrays["seen"] > 1):
        raise ValueError("profiles['seen'] must contain only zero or one")
    if np.any(arrays["shadow_group"] < -1) or np.any(arrays["shadow_group"] >= _GROUPS):
        raise ValueError("profiles['shadow_group'] values must be -1, 0, or 1")
    return arrays, arrays["seen"].astype(bool, copy=False)


def _profile_for_slot(arrays: Mapping[str, np.ndarray], slot: int) -> EvidenceProfile:
    return EvidenceProfile(
        arrays["movement_counts"][slot].copy(),
        arrays["reward_xtx"][slot].copy(),
        arrays["reward_xty"][slot].copy(),
        np.float32(arrays["reward_n"][slot]),
    )


def _new_parent_probe(grouper: RewardAwareGroups) -> RewardAwareGroups:
    config = dict(grouper.config)
    # Parent reuse requires both dynamic and reward agreement even when the
    # live learner's baseline grouper uses dynamics-only assignment.
    config["mode"] = "reward"
    return RewardAwareGroups(**config)


def _compatible_partner(
    current_profile: EvidenceProfile,
    partner_profile: EvidenceProfile,
    grouper: RewardAwareGroups,
) -> bool:
    probe = _new_parent_probe(grouper)
    first = probe.assign(current_profile, commit=True)
    if first.group < 0:
        return False
    second = probe.assign(partner_profile, commit=False)
    return second.group == first.group and second.reason == "matched"


def _clear_parent_statistics(grouper: RewardAwareGroups, group: int) -> RewardAwareGroups:
    state = grouper.state_dict()
    for name in ("movement_counts", "reward_xtx", "reward_xty"):
        state[name][group].fill(0)
    state["reward_n"][group] = np.float32(0.0)
    state["active"][group] = 0
    state["nbytes"] = grouper.nbytes
    return RewardAwareGroups.from_state_dict(state)


def _not_recycled(
    grouper: RewardAwareGroups,
    assignment: GroupAssignment,
    reason: str,
) -> ParentRecycleResult:
    return ParentRecycleResult(
        recycled=False,
        reason=reason,
        grouper=grouper,
        assignment=assignment,
    )


def try_recycle(
    manager: ParentBudgetManager,
    core,
    grouper: RewardAwareGroups,
    profiles: Mapping[str, Any],
    *,
    current_slot: int,
    slot_generations: Any,
    step: int,
) -> ParentRecycleResult:
    """Reuse a parent only after a fresh assignment reports group capacity full.

    This mutates ``core`` and ``profiles['shadow_group']`` only on success.
    Individual slot evidence and slot generations are retained. The caller
    owns anchor provenance by parent version and must not combine rows across
    parent versions when scoring a later shared change.
    """
    if not isinstance(manager, ParentBudgetManager):
        raise TypeError("manager must be a ParentBudgetManager")
    if not isinstance(grouper, RewardAwareGroups) or grouper.max_groups != _GROUPS:
        raise ValueError("grouper must be a two-parent RewardAwareGroups instance")
    slots = int(getattr(core, "local_slots", len(getattr(core, "slot_groups", ()))))
    if slots < 2 or int(getattr(core, "shared_groups", -1)) != _GROUPS:
        raise ValueError("core must provide local slots and exactly two shared groups")
    if len(core.slot_groups) != slots:
        raise ValueError("core slot group count does not match local_slots")
    current = _integer(current_slot, "current_slot", minimum=0, maximum=slots - 1)
    step_index = _integer(step, "step", minimum=0, maximum=_COUNTER_MAX)
    generations = np.asarray(slot_generations)
    if generations.shape != (slots,) or generations.dtype.kind not in "iu":
        raise ValueError(f"slot_generations must be an integer vector of length {slots}")
    # Routers commonly use -1 for vacant local slots. Such slots may have
    # stale profile rows, but they are not resident evidence and cannot vote.
    if generations.dtype.kind == "i" and np.any(generations < -1):
        raise ValueError("slot_generations values must be -1 or nonnegative")
    arrays, seen = _profile_arrays(profiles, slots)
    if not arrays["shadow_group"].flags.writeable:
        raise ValueError("profiles['shadow_group'] must be writable for a successful recycle")
    current_profile = _profile_for_slot(arrays, current)
    original_assignment = grouper.assign(current_profile, commit=False)
    if original_assignment.reason not in _CAPACITY_FAILURES:
        return _not_recycled(grouper, original_assignment, "not_capacity_full")
    if core.group_for_slot(current) != -1:
        return _not_recycled(grouper, original_assignment, "current_slot_not_ungrouped")
    if int(generations[current]) < 0:
        return _not_recycled(grouper, original_assignment, "current_slot_not_resident")
    if not seen[current]:
        return _not_recycled(grouper, original_assignment, "current_profile_not_resident")

    groups = np.asarray(core.slot_groups.detach().cpu().numpy(), dtype=np.int64)
    resident_slots_by_group = {
        group: tuple(int(slot) for slot in np.flatnonzero(groups == group))
        for group in range(_GROUPS)
    }
    empty_parent_groups = [
        group for group, members in resident_slots_by_group.items() if not members
    ]
    chosen_group = min(
        empty_parent_groups if empty_parent_groups else range(_GROUPS),
        key=lambda group: (int(manager.last_used[group]), group),
    )

    partner_slots: list[int] = []
    if not empty_parent_groups:
        for candidate_slot in range(slots):
            if (
                candidate_slot == current
                or not seen[candidate_slot]
                or int(generations[candidate_slot]) < 0
            ):
                continue
            if groups[candidate_slot] != -1:
                continue
            if _compatible_partner(
                current_profile,
                _profile_for_slot(arrays, candidate_slot),
                grouper,
            ):
                partner_slots.append(candidate_slot)
                break
        if not partner_slots:
            return _not_recycled(grouper, original_assignment, "no_compatible_unpaired_profile")

    removed_slots = resident_slots_by_group[chosen_group]
    previous_group_version = int(manager.version[chosen_group])
    if int(manager.version[chosen_group]) >= _COUNTER_MAX:
        raise OverflowError("parent version counter is exhausted")
    if step_index < int(manager.last_used[chosen_group]):
        raise ValueError("step must not move backwards for a parent group")
    if int(manager.touch_count[chosen_group]) >= _COUNTER_MAX:
        raise OverflowError("parent touch counter is exhausted")
    replacement_grouper = _clear_parent_statistics(grouper, chosen_group)
    replacement_assignment = replacement_grouper.assign(current_profile, commit=True)
    if replacement_assignment.group != chosen_group:
        return _not_recycled(grouper, original_assignment, "replacement_parent_not_available")
    if partner_slots:
        partner_assignment = replacement_grouper.assign(
            _profile_for_slot(arrays, partner_slots[0]), commit=True
        )
        if partner_assignment.group != chosen_group:
            return _not_recycled(grouper, original_assignment, "partner_not_assigned_to_recycled_parent")

    # Make the parameter mutations only after all evidence and group assignment
    # checks have succeeded. set_slot_group compensates every affected local A.
    for slot in removed_slots:
        core.set_slot_group(slot, -1, preserve_effective=True)
    if any(core.group_for_slot(slot) == chosen_group for slot in range(slots)):
        raise RuntimeError("old parent still has resident policy members after detachment")
    shared = core.read_shared_matrix()
    shared[chosen_group].fill(0)
    core.install_shared_matrix(shared)

    added_slots = (current, *partner_slots)
    for slot in added_slots:
        core.set_slot_group(slot, chosen_group, preserve_effective=True)

    shadow = arrays["shadow_group"]
    cleared_shadow_slots = tuple(int(slot) for slot in np.flatnonzero(shadow == chosen_group))
    shadow[shadow == chosen_group] = -1
    for slot in added_slots:
        shadow[slot] = chosen_group

    manager.version[chosen_group] += np.uint64(1)
    manager.touch(chosen_group, step_index)
    affected = tuple(sorted(set((*removed_slots, *added_slots, *cleared_shadow_slots))))
    affected_generations = tuple((slot, int(generations[slot])) for slot in affected)
    return ParentRecycleResult(
        recycled=True,
        reason="recycled_empty_parent" if empty_parent_groups else "recycled_lru_parent",
        grouper=replacement_grouper,
        assignment=replacement_assignment,
        group=chosen_group,
        previous_group_version=previous_group_version,
        group_version=int(manager.version[chosen_group]),
        affected_slots=affected,
        removed_slots=removed_slots,
        added_slots=tuple(added_slots),
        cleared_shadow_slots=cleared_shadow_slots,
        affected_generations=affected_generations,
        seen_updates=tuple((slot, bool(seen[slot])) for slot in affected),
        committed_profile_slots=tuple(added_slots),
    )


__all__ = ["ParentBudgetManager", "ParentRecycleResult", "try_recycle"]

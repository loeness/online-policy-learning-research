"""Fixed-capacity, label-free archive of reusable rollout evidence.

The archive stores the latest visible evidence profile for each anonymous
prototype and a bounded uniform reservoir of finalized transition rows. A
prototype match requires supported movement agreement and an identifiable,
compatible reward model. Slot and generation values are retained only as row
provenance; they never determine prototype identity or clear archived rows.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from operator import index as as_index
from typing import Any

import numpy as np

from .shared_write_grouping import PROFILE_SHAPE, supported_directions
from .write_decision_evidence import (
    EvidenceProfile,
    RewardAwareGroups,
    _profile_fields,
)


OBSERVATION_SIZE = 13
ACTION_COUNT = 4
_STATE_VERSION = 1
_MIN_CELL_SUPPORT = 2
_ARCHIVE_MIN_OVERLAP = 6
_ARCHIVE_MIN_AGREEMENT = 0.9
_ARCHIVE_MIN_REWARD_COSINE = 0.5
_MAX_INT64 = int(np.iinfo(np.int64).max)
_MIN_INT64 = int(np.iinfo(np.int64).min)

# The support-aware comparator abstains when the reward features are not
# identifiable or when either fitted coefficient vector has no magnitude.
_REWARD_COMPARATOR = RewardAwareGroups(max_groups=1, reward_support_aware=True)


def _positive_integer(value: object, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = as_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1 or (maximum is not None and result > maximum):
        suffix = f" no greater than {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return int(result)


def _bounded_fraction(value: object, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]") from exc
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return result


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = as_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if result < _MIN_INT64 or result > _MAX_INT64:
        raise ValueError(f"{name} exceeds int64 storage")
    return int(result)


def _row_value(row: Any, name: str, *aliases: str, default: Any = ...) -> Any:
    for key in (name, *aliases):
        if isinstance(row, Mapping) and key in row:
            return row[key]
        if hasattr(row, key):
            return getattr(row, key)
    if default is not ...:
        return default
    raise ValueError(f"transition row is missing {name!r}")


def _profile_arrays(
    profile: EvidenceProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    if not isinstance(profile, EvidenceProfile):
        raise TypeError("profile must be an EvidenceProfile")
    counts, xtx, xty, reward_n = _profile_fields(profile)
    return counts, xtx, xty, reward_n


def profiles_compatible(
    first: EvidenceProfile,
    second: EvidenceProfile,
    *,
    min_overlap: int = 4,
    min_agreement: float = 0.6,
    reward_cosine: float = 0.5,
) -> dict[str, bool | int | float]:
    """Compare two visible profiles, abstaining on unsupported evidence.

    Movement directions count as supported when a cell has at least two
    observations with at least 80 percent directional purity. Reward
    coefficients are compared only over features that both profiles support
    and identify. If the common reward vector is not identifiable, comparison
    is explicitly marked unavailable and the profiles are incompatible.
    """

    overlap_required = _positive_integer(min_overlap, "min_overlap")
    agreement_required = _bounded_fraction(min_agreement, "min_agreement", 0.0, 1.0)
    reward_required = _bounded_fraction(reward_cosine, "reward_cosine", -1.0, 1.0)
    first_counts, first_xtx, first_xty, _ = _profile_arrays(first)
    second_counts, second_xtx, second_xty, _ = _profile_arrays(second)

    first_directions = supported_directions(first_counts, min_cell_support=_MIN_CELL_SUPPORT)
    second_directions = supported_directions(second_counts, min_cell_support=_MIN_CELL_SUPPORT)
    common = (first_directions >= 0) & (second_directions >= 0)
    overlap = int(np.count_nonzero(common))
    agreement = (
        float(np.count_nonzero(first_directions[common] == second_directions[common]) / overlap)
        if overlap
        else 0.0
    )

    reward_score = _REWARD_COMPARATOR._reward_similarity(
        first_xtx, first_xty, second_xtx, second_xty
    )
    is_reward_comparable = bool(reward_score.get("reward_comparable", 0))
    cosine = float(reward_score.get("reward_cosine", 0.0))
    compatible = bool(
        overlap >= overlap_required
        and agreement >= agreement_required
        and is_reward_comparable
        and cosine >= reward_required
    )
    return {
        "compatible": compatible,
        "overlap": overlap,
        "agreement": agreement,
        "reward_cosine": cosine,
        "reward_comparable": is_reward_comparable,
    }


class RetentionEvidenceArchive:
    """A fixed table of anonymous prototypes and per-prototype row reservoirs."""

    STATE_VERSION = _STATE_VERSION

    def __init__(
        self,
        max_prototypes: int = 8,
        per_prototype: int = 24,
        seed: int = 0,
    ) -> None:
        self.max_prototypes = _positive_integer(max_prototypes, "max_prototypes")
        self.per_prototype = _positive_integer(per_prototype, "per_prototype")
        self.seed = _integer(seed, "seed", minimum=0)

        prototype_count = self.max_prototypes
        reservoir_shape = (prototype_count, self.per_prototype)
        self._movement_counts = np.zeros((prototype_count, *PROFILE_SHAPE), dtype=np.uint16)
        self._reward_xtx = np.zeros((prototype_count, 3, 3), dtype=np.float32)
        self._reward_xty = np.zeros((prototype_count, 3), dtype=np.float32)
        self._reward_n = np.zeros(prototype_count, dtype=np.float32)
        self._active = np.zeros(prototype_count, dtype=np.uint8)
        self._versions = np.zeros(prototype_count, dtype=np.int64)
        self._last_used = np.full(prototype_count, -1, dtype=np.int64)
        self._sizes = np.zeros(prototype_count, dtype=np.int32)
        self._seen = np.zeros(prototype_count, dtype=np.int64)
        # Transition rows do not always carry episode_id. Negative synthetic
        # IDs preserve row clustering without colliding with ordinary
        # nonnegative environment episode counters.
        self._implicit_episode_ids = np.full(prototype_count, -1, dtype=np.int64)

        self._obs = np.zeros((*reservoir_shape, OBSERVATION_SIZE), dtype=np.float32)
        self._actions = np.zeros(reservoir_shape, dtype=np.int8)
        self._behavior_log_probs = np.zeros(reservoir_shape, dtype=np.float32)
        self._behavior_probs = np.zeros((*reservoir_shape, ACTION_COUNT), dtype=np.float32)
        self._advantages = np.zeros(reservoir_shape, dtype=np.float32)
        self._returns = np.zeros(reservoir_shape, dtype=np.float32)
        self._episode_ids = np.zeros(reservoir_shape, dtype=np.int64)
        self._source_slots = np.full(reservoir_shape, -1, dtype=np.int64)
        self._source_generations = np.full(reservoir_shape, -1, dtype=np.int64)
        self._rng = np.random.default_rng(self.seed)

    @property
    def nbytes(self) -> int:
        """Bytes in the fixed numeric tables, excluding Python and RNG state."""

        arrays = (
            self._movement_counts,
            self._reward_xtx,
            self._reward_xty,
            self._reward_n,
            self._active,
            self._versions,
            self._last_used,
            self._sizes,
            self._seen,
            self._implicit_episode_ids,
            self._obs,
            self._actions,
            self._behavior_log_probs,
            self._behavior_probs,
            self._advantages,
            self._returns,
            self._episode_ids,
            self._source_slots,
            self._source_generations,
        )
        return int(sum(array.nbytes for array in arrays))

    def active_indices(self) -> tuple[int, ...]:
        """Return active prototype cells in ascending numeric order."""

        return tuple(int(index) for index in np.flatnonzero(self._active))

    def _check_index(self, index: object) -> int:
        slot = _integer(index, "prototype", minimum=0)
        if slot >= self.max_prototypes:
            raise ValueError(f"prototype must be in [0, {self.max_prototypes - 1}]")
        if not self._active[slot]:
            raise ValueError(f"prototype {slot} is inactive")
        return slot

    def profile(self, index: int) -> EvidenceProfile:
        """Return a copy of the latest visible profile in one active cell."""

        slot = self._check_index(index)
        return EvidenceProfile(
            movement_counts=self._movement_counts[slot].copy(),
            reward_xtx=self._reward_xtx[slot].copy(),
            reward_xty=self._reward_xty[slot].copy(),
            reward_n=np.float32(self._reward_n[slot]),
        )

    def arrays(self, index: int) -> dict[str, np.ndarray]:
        """Return copies of sampled rows and their optional source metadata."""

        slot = self._check_index(index)
        size = int(self._sizes[slot])
        return {
            "obs": self._obs[slot, :size].copy(),
            "action": self._actions[slot, :size].copy(),
            "behavior_log_prob": self._behavior_log_probs[slot, :size].copy(),
            "behavior_probs": self._behavior_probs[slot, :size].copy(),
            "advantage": self._advantages[slot, :size].copy(),
            "return": self._returns[slot, :size].copy(),
            "episode_id": self._episode_ids[slot, :size].copy(),
            "source_slot": self._source_slots[slot, :size].copy(),
            "source_generation": self._source_generations[slot, :size].copy(),
        }

    def _clear_cell(self, slot: int) -> None:
        self._movement_counts[slot].fill(0)
        self._reward_xtx[slot].fill(0)
        self._reward_xty[slot].fill(0)
        self._reward_n[slot] = 0
        self._obs[slot].fill(0)
        self._actions[slot].fill(0)
        self._behavior_log_probs[slot].fill(0)
        self._behavior_probs[slot].fill(0)
        self._advantages[slot].fill(0)
        self._returns[slot].fill(0)
        self._episode_ids[slot].fill(0)
        self._source_slots[slot].fill(-1)
        self._source_generations[slot].fill(-1)
        self._sizes[slot] = 0
        self._seen[slot] = 0
        self._implicit_episode_ids[slot] = -1
        self._last_used[slot] = -1

    def _write_profile(self, slot: int, profile: EvidenceProfile) -> None:
        counts, xtx, xty, reward_n = _profile_arrays(profile)
        self._movement_counts[slot] = counts
        self._reward_xtx[slot] = xtx
        self._reward_xty[slot] = xty
        self._reward_n[slot] = reward_n

    @staticmethod
    def _validate_row(
        row: Any, *, fallback_episode_id: int
    ) -> tuple[np.ndarray, int, np.float32, np.ndarray, np.float32, np.float32, int, int, int, bool]:
        obs = np.asarray(_row_value(row, "obs"), dtype=np.float32)
        if obs.shape != (OBSERVATION_SIZE,) or not np.isfinite(obs).all():
            raise ValueError(f"transition obs must be a finite vector of length {OBSERVATION_SIZE}")

        action = _integer(_row_value(row, "action"), "action", minimum=0)
        if action >= ACTION_COUNT:
            raise ValueError(f"action must be in [0, {ACTION_COUNT - 1}]")

        log_prob = float(_row_value(row, "behavior_log_prob", "log_prob"))
        if not math.isfinite(log_prob) or log_prob > 1e-6:
            raise ValueError("behavior_log_prob must be finite and nonpositive")
        probabilities = np.asarray(
            _row_value(row, "behavior_probs", "probabilities"), dtype=np.float32
        )
        if probabilities.shape != (ACTION_COUNT,) or not np.isfinite(probabilities).all():
            raise ValueError(f"behavior_probs must be a finite vector of length {ACTION_COUNT}")
        total = float(probabilities.sum(dtype=np.float64))
        if (
            np.any(probabilities < 0)
            or total <= 0.0
            or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-4)
        ):
            raise ValueError("behavior_probs must be a probability distribution")
        probabilities = probabilities / np.float32(total)

        advantage = float(_row_value(row, "advantage"))
        episode_return = float(_row_value(row, "return", "returns"))
        if not math.isfinite(advantage) or not math.isfinite(episode_return):
            raise ValueError("finalized advantage and return must be finite")
        episode_id = _integer(
            _row_value(row, "episode_id", default=fallback_episode_id), "episode_id"
        )
        done_value = _row_value(row, "done", "terminated", default=False)
        if not isinstance(done_value, (bool, np.bool_)):
            raise ValueError("transition done must be a boolean")

        source_slot = _integer(
            _row_value(row, "source_slot", "slot", default=-1), "source_slot"
        )
        source_generation = _integer(
            _row_value(row, "source_generation", "generation", default=-1),
            "source_generation",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            log_prob32 = np.float32(log_prob)
            advantage32 = np.float32(advantage)
            return32 = np.float32(episode_return)
        if not np.isfinite(log_prob32) or not np.isfinite(advantage32) or not np.isfinite(return32):
            raise ValueError("transition values must be representable as float32")
        return (
            obs.copy(),
            action,
            log_prob32,
            probabilities.copy(),
            advantage32,
            return32,
            episode_id,
            source_slot,
            source_generation,
            bool(done_value),
        )

    def _add_rows(self, slot: int, rows: Sequence[Any]) -> None:
        for row in rows:
            (
                observation,
                action,
                log_prob,
                probabilities,
                advantage,
                episode_return,
                episode_id,
                source_slot,
                source_generation,
                done,
            ) = self._validate_row(
                row, fallback_episode_id=int(self._implicit_episode_ids[slot])
            )
            if done:
                self._implicit_episode_ids[slot] -= 1
            seen = int(self._seen[slot])
            if int(self._sizes[slot]) < self.per_prototype:
                index = int(self._sizes[slot])
                self._sizes[slot] += 1
            else:
                replacement = int(self._rng.integers(0, seen + 1))
                if replacement >= self.per_prototype:
                    self._seen[slot] += 1
                    continue
                index = replacement

            self._obs[slot, index] = observation
            self._actions[slot, index] = action
            self._behavior_log_probs[slot, index] = log_prob
            self._behavior_probs[slot, index] = probabilities
            self._advantages[slot, index] = advantage
            self._returns[slot, index] = episode_return
            self._episode_ids[slot, index] = episode_id
            self._source_slots[slot, index] = source_slot
            self._source_generations[slot, index] = source_generation
            self._seen[slot] += 1

    @staticmethod
    def _diagnostic_reason(scores: Sequence[Mapping[str, bool | int | float]]) -> str:
        if not scores or max(int(score["overlap"]) for score in scores) < _ARCHIVE_MIN_OVERLAP:
            return "insufficient_support"
        direction_candidates = [
            score
            for score in scores
            if int(score["overlap"]) >= _ARCHIVE_MIN_OVERLAP
            and float(score["agreement"]) >= _ARCHIVE_MIN_AGREEMENT
        ]
        if not direction_candidates:
            return "movement_conflict"
        if any(not bool(score["reward_comparable"]) for score in direction_candidates):
            return "reward_unidentifiable"
        return "reward_conflict"

    def observe(
        self,
        profile: EvidenceProfile,
        rows: Sequence[Any],
        *,
        step: int,
    ) -> dict[str, Any]:
        """Match or allocate one profile and add finalized visible rollout rows.

        A matching cell keeps its bounded row reservoir and replaces its
        profile with the newest window summary. A new cell starts a fresh
        reservoir. When full, the least recently observed cell is replaced;
        slot and generation changes in input rows do not clear any cell.
        """

        _profile_arrays(profile)
        if isinstance(rows, (str, bytes, bytearray, Mapping)):
            raise TypeError("rows must be a sequence of transition rows")
        try:
            iter(rows)
        except TypeError as exc:
            raise TypeError("rows must be a sequence of transition rows") from exc
        observation_step = _integer(step, "step", minimum=0)

        active = self.active_indices()
        latest_touch = max((int(self._last_used[index]) for index in active), default=-1)
        if latest_touch >= _MAX_INT64:
            raise ValueError("archive LRU counter exhausted int64 storage")
        scores = [
            (slot, profiles_compatible(
                profile,
                self.profile(slot),
                min_overlap=_ARCHIVE_MIN_OVERLAP,
                min_agreement=_ARCHIVE_MIN_AGREEMENT,
                reward_cosine=_ARCHIVE_MIN_REWARD_COSINE,
            ))
            for slot in active
        ]
        matches = [(slot, score) for slot, score in scores if bool(score["compatible"])]
        if matches:
            # Prefer the strongest evidence match; stable cell order breaks ties.
            slot, _ = max(
                matches,
                key=lambda pair: (
                    int(pair[1]["overlap"]),
                    float(pair[1]["agreement"]),
                    float(pair[1]["reward_cosine"]),
                    -pair[0],
                ),
            )
            matched = True
            evicted = False
            reason = "compatible_profile_match"
        else:
            free = np.flatnonzero(self._active == 0)
            evicted = not len(free)
            if evicted:
                slot = min(active, key=lambda index: (int(self._last_used[index]), index))
            else:
                slot = int(free[0])
            matched = False
            diagnostic = self._diagnostic_reason([score for _, score in scores])
            if not active:
                reason = "created_first"
            elif evicted:
                reason = f"lru_replaced_{diagnostic}"
            else:
                reason = f"created_new_cell_{diagnostic}"
            previous_version = int(self._versions[slot])
            self._clear_cell(slot)
            self._versions[slot] = previous_version + 1
            self._active[slot] = 1

        self._write_profile(slot, profile)
        self._last_used[slot] = max(observation_step, latest_touch + 1)
        self._add_rows(slot, rows)
        return {
            "prototype": int(slot),
            "version": int(self._versions[slot]),
            "matched": bool(matched),
            "evicted": bool(evicted),
            "reason": reason,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a bounded, pickle-friendly snapshot of every fixed buffer."""

        return {
            "version": self.STATE_VERSION,
            "max_prototypes": self.max_prototypes,
            "per_prototype": self.per_prototype,
            "seed": self.seed,
            "movement_counts": self._movement_counts.copy(),
            "reward_xtx": self._reward_xtx.copy(),
            "reward_xty": self._reward_xty.copy(),
            "reward_n": self._reward_n.copy(),
            "active": self._active.copy(),
            "versions": self._versions.copy(),
            "last_used": self._last_used.copy(),
            "sizes": self._sizes.copy(),
            "seen": self._seen.copy(),
            "implicit_episode_ids": self._implicit_episode_ids.copy(),
            "obs": self._obs.copy(),
            "actions": self._actions.copy(),
            "behavior_log_probs": self._behavior_log_probs.copy(),
            "behavior_probs": self._behavior_probs.copy(),
            "advantages": self._advantages.copy(),
            "returns": self._returns.copy(),
            "episode_ids": self._episode_ids.copy(),
            "source_slots": self._source_slots.copy(),
            "source_generations": self._source_generations.copy(),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RetentionEvidenceArchive":
        """Restore and validate a fixed-capacity archive snapshot."""

        if not isinstance(state, Mapping) or state.get("version") != cls.STATE_VERSION:
            raise ValueError("unsupported RetentionEvidenceArchive state")
        try:
            archive = cls(
                max_prototypes=state["max_prototypes"],
                per_prototype=state["per_prototype"],
                seed=state.get("seed", 0),
            )
            specifications = {
                "movement_counts": (archive._movement_counts, np.uint16),
                "reward_xtx": (archive._reward_xtx, np.float32),
                "reward_xty": (archive._reward_xty, np.float32),
                "reward_n": (archive._reward_n, np.float32),
                "active": (archive._active, np.uint8),
                "versions": (archive._versions, np.int64),
                "last_used": (archive._last_used, np.int64),
                "sizes": (archive._sizes, np.int32),
                "seen": (archive._seen, np.int64),
                "implicit_episode_ids": (archive._implicit_episode_ids, np.int64),
                "obs": (archive._obs, np.float32),
                "actions": (archive._actions, np.int8),
                "behavior_log_probs": (archive._behavior_log_probs, np.float32),
                "behavior_probs": (archive._behavior_probs, np.float32),
                "advantages": (archive._advantages, np.float32),
                "returns": (archive._returns, np.float32),
                "episode_ids": (archive._episode_ids, np.int64),
                "source_slots": (archive._source_slots, np.int64),
                "source_generations": (archive._source_generations, np.int64),
            }
            for name, (target, dtype) in specifications.items():
                value = np.asarray(state[name])
                if value.shape != target.shape or value.dtype != dtype:
                    raise ValueError(f"invalid archive {name} buffer")
                if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
                    raise ValueError(f"archive {name} buffer contains non-finite values")
                target[...] = value

            if np.any(archive._active > 1):
                raise ValueError("archive active values must be zero or one")
            if np.any(archive._sizes < 0) or np.any(archive._sizes > archive.per_prototype):
                raise ValueError("archive reservoir sizes are out of range")
            if np.any(archive._seen < archive._sizes) or np.any(archive._seen < 0):
                raise ValueError("archive seen counts are invalid")
            if np.any(archive._versions < 0):
                raise ValueError("archive versions are invalid")
            if np.any(archive._last_used < -1):
                raise ValueError("archive last_used values are invalid")
            if not np.isfinite(archive._reward_n).all() or np.any(archive._reward_n < 0):
                raise ValueError("archive reward counts are invalid")
            if not np.allclose(
                archive._reward_xtx,
                archive._reward_xtx.transpose(0, 2, 1),
                rtol=0.0,
                atol=1e-4,
            ):
                raise ValueError("archive reward_xtx values must be symmetric")
            if np.any(archive._actions < 0) or np.any(archive._actions >= ACTION_COUNT):
                raise ValueError("archive actions are out of range")
            if np.any(archive._behavior_log_probs > 1e-6):
                raise ValueError("archive behavior log probabilities must be nonpositive")
            if np.any(archive._behavior_probs < 0):
                raise ValueError("archive behavior probabilities cannot be negative")

            for slot in range(archive.max_prototypes):
                size = int(archive._sizes[slot])
                active = bool(archive._active[slot])
                if active != (archive._versions[slot] > 0):
                    # Inactive cells can keep a version counter after eviction.
                    if active or archive._versions[slot] == 0 and not active:
                        raise ValueError("archive active cells need a positive version")
                if active and int(archive._last_used[slot]) < 0:
                    raise ValueError("active archive cells need a nonnegative last_used value")
                if not active and (size or archive._seen[slot]):
                    raise ValueError("inactive archive cells cannot contain rows")
                if size:
                    probs = archive._behavior_probs[slot, :size].astype(np.float64)
                    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-4):
                        raise ValueError("archive behavior probabilities must sum to one")
                if active:
                    _profile_fields(
                        EvidenceProfile(
                            archive._movement_counts[slot],
                            archive._reward_xtx[slot],
                            archive._reward_xty[slot],
                            np.float32(archive._reward_n[slot]),
                        )
                    )

            if "nbytes" in state and state["nbytes"] != archive.nbytes:
                raise ValueError("archive nbytes does not match fixed storage")
            archive._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
            return archive
        except KeyError as exc:
            raise ValueError(f"archive state is missing {exc.args[0]!r}") from exc


__all__ = ["RetentionEvidenceArchive", "profiles_compatible"]

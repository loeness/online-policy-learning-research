"""Bounded evidence and proposal filters for shared-write decisions.

This module contains no environment or policy mutation.  Anchors are finalized
rollout samples used only to assess a proposed shared-policy change; callers
must not feed them back into PPO training.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


OBSERVATION_SIZE = 13
ACTION_COUNT = 4
ANCHOR_FIELDS = (
    "obs",
    "action",
    "behavior_log_prob",
    "behavior_probs",
    "advantage",
    "return",
    "generation",
    "episode_id",
)


def _row_value(row: Any, name: str, *aliases: str) -> Any:
    for key in (name, *aliases):
        if isinstance(row, Mapping) and key in row:
            return row[key]
        if hasattr(row, key):
            return getattr(row, key)
    raise ValueError(f"anchor row is missing {name!r}")


class AnchorBank:
    """Fixed-size per-slot uniform reservoirs for finalized PPO rollouts.

    ``add`` accepts mappings or objects with ``obs``, ``action``, ``log_prob``
    (or ``behavior_log_prob``), ``probabilities`` (or ``behavior_probs``),
    ``advantage``, ``returns`` (or ``return``), and ``episode_id`` fields.
    Rows are copied into bounded arrays. A new generation clears that slot
    before its first row is stored; evidence never crosses slot reuse.

    ``arrays(slot, generation)`` returns a dictionary of copied NumPy arrays.
    On an empty or generation-mismatched slot it returns the same keys with
    zero-length arrays. Floating-point evidence is stored as float32; actions
    use int8 and generation / episode identifiers use int64.
    """

    STATE_VERSION = 1

    def __init__(self, slots: int = 6, per_slot: int = 24, *, seed: int | None = None) -> None:
        if isinstance(slots, bool) or int(slots) != slots or slots < 1:
            raise ValueError("slots must be a positive integer")
        if isinstance(per_slot, bool) or int(per_slot) != per_slot or per_slot < 1:
            raise ValueError("per_slot must be a positive integer")
        self.slots = int(slots)
        self.per_slot = int(per_slot)
        shape = (self.slots, self.per_slot)
        self._obs = np.zeros((*shape, OBSERVATION_SIZE), dtype=np.float32)
        self._actions = np.zeros(shape, dtype=np.int8)
        self._behavior_log_probs = np.zeros(shape, dtype=np.float32)
        self._behavior_probs = np.zeros((*shape, ACTION_COUNT), dtype=np.float32)
        self._advantages = np.zeros(shape, dtype=np.float32)
        self._returns = np.zeros(shape, dtype=np.float32)
        self._generations = np.full(shape, -1, dtype=np.int64)
        self._episode_ids = np.zeros(shape, dtype=np.int64)
        self._sizes = np.zeros(self.slots, dtype=np.int32)
        self._seen = np.zeros(self.slots, dtype=np.int64)
        self._active_generations = np.full(self.slots, -1, dtype=np.int64)
        self._rng = np.random.default_rng(seed)

    @property
    def nbytes(self) -> int:
        """Bytes occupied by the fixed NumPy buffers (excluding Python/RNG state)."""
        arrays = (
            self._obs,
            self._actions,
            self._behavior_log_probs,
            self._behavior_probs,
            self._advantages,
            self._returns,
            self._generations,
            self._episode_ids,
            self._sizes,
            self._seen,
            self._active_generations,
        )
        return int(sum(array.nbytes for array in arrays))

    def _check_slot(self, slot: int) -> int:
        if isinstance(slot, bool) or int(slot) != slot or not 0 <= int(slot) < self.slots:
            raise ValueError(f"slot must be an integer in [0, {self.slots - 1}]")
        return int(slot)

    def _clear_slot(self, slot: int) -> None:
        self._obs[slot].fill(0)
        self._actions[slot].fill(0)
        self._behavior_log_probs[slot].fill(0)
        self._behavior_probs[slot].fill(0)
        self._advantages[slot].fill(0)
        self._returns[slot].fill(0)
        self._generations[slot].fill(-1)
        self._episode_ids[slot].fill(0)
        self._sizes[slot] = 0
        self._seen[slot] = 0

    def add(self, slot: int, generation: int, rows: Sequence[Any]) -> None:
        """Reservoir-sample finalized rows from one live slot generation."""
        slot = self._check_slot(slot)
        if isinstance(generation, bool) or int(generation) != generation or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        generation = int(generation)
        if self._active_generations[slot] != generation:
            self._clear_slot(slot)
            self._active_generations[slot] = generation

        for row in rows:
            row_generation = _row_value(row, "generation") if (
                (isinstance(row, Mapping) and "generation" in row)
                or hasattr(row, "generation")
            ) else generation
            if isinstance(row_generation, bool) or int(row_generation) != row_generation:
                raise ValueError("row generation must be an integer")
            if int(row_generation) != generation:
                raise ValueError("anchor row generation does not match the requested generation")

            observation = np.asarray(_row_value(row, "obs"), dtype=np.float32)
            if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
                raise ValueError(f"anchor obs must be a finite vector of length {OBSERVATION_SIZE}")
            action_value = _row_value(row, "action")
            if isinstance(action_value, bool) or int(action_value) != action_value or not 0 <= int(action_value) < ACTION_COUNT:
                raise ValueError(f"anchor action must be an integer in [0, {ACTION_COUNT - 1}]")
            log_prob = float(_row_value(row, "behavior_log_prob", "log_prob"))
            behavior_probs = np.asarray(
                _row_value(row, "behavior_probs", "probabilities"), dtype=np.float32
            )
            if not math.isfinite(log_prob) or log_prob > 1e-6:
                raise ValueError("behavior_log_prob must be finite and nonpositive")
            if behavior_probs.shape != (ACTION_COUNT,) or not np.isfinite(behavior_probs).all():
                raise ValueError(f"behavior_probs must be a finite vector of length {ACTION_COUNT}")
            if np.any(behavior_probs < 0) or not np.isclose(float(behavior_probs.sum()), 1.0, atol=1e-4, rtol=0):
                raise ValueError("behavior_probs must be a probability distribution")
            advantage = float(_row_value(row, "advantage"))
            episode_return = float(_row_value(row, "return", "returns"))
            episode_id = _row_value(row, "episode_id")
            if not math.isfinite(advantage) or not math.isfinite(episode_return):
                raise ValueError("anchor advantage and return must be finite")
            if isinstance(episode_id, bool) or int(episode_id) != episode_id:
                raise ValueError("episode_id must be an integer")

            seen = int(self._seen[slot])
            if int(self._sizes[slot]) < self.per_slot:
                index = int(self._sizes[slot])
                self._sizes[slot] += 1
            else:
                replacement = int(self._rng.integers(0, seen + 1))
                if replacement >= self.per_slot:
                    self._seen[slot] += 1
                    continue
                index = replacement

            self._obs[slot, index] = observation
            self._actions[slot, index] = int(action_value)
            self._behavior_log_probs[slot, index] = log_prob
            self._behavior_probs[slot, index] = behavior_probs / behavior_probs.sum()
            self._advantages[slot, index] = advantage
            self._returns[slot, index] = episode_return
            self._generations[slot, index] = generation
            self._episode_ids[slot, index] = int(episode_id)
            self._seen[slot] += 1

    def arrays(self, slot: int, generation: int) -> dict[str, np.ndarray]:
        """Return copied, nonempty row arrays for the requested generation."""
        slot = self._check_slot(slot)
        if isinstance(generation, bool) or int(generation) != generation or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        size = int(self._sizes[slot]) if int(self._active_generations[slot]) == int(generation) else 0
        return {
            "obs": self._obs[slot, :size].copy(),
            "action": self._actions[slot, :size].copy(),
            "behavior_log_prob": self._behavior_log_probs[slot, :size].copy(),
            "behavior_probs": self._behavior_probs[slot, :size].copy(),
            "advantage": self._advantages[slot, :size].copy(),
            "return": self._returns[slot, :size].copy(),
            "generation": self._generations[slot, :size].copy(),
            "episode_id": self._episode_ids[slot, :size].copy(),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a bounded, pickle-friendly snapshot of buffers and RNG state."""
        return {
            "version": self.STATE_VERSION,
            "slots": self.slots,
            "per_slot": self.per_slot,
            "obs": self._obs.copy(),
            "actions": self._actions.copy(),
            "behavior_log_probs": self._behavior_log_probs.copy(),
            "behavior_probs": self._behavior_probs.copy(),
            "advantages": self._advantages.copy(),
            "returns": self._returns.copy(),
            "generations": self._generations.copy(),
            "episode_ids": self._episode_ids.copy(),
            "sizes": self._sizes.copy(),
            "seen": self._seen.copy(),
            "active_generations": self._active_generations.copy(),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "AnchorBank":
        """Restore a validated bounded snapshot."""
        if not isinstance(state, Mapping) or state.get("version") != cls.STATE_VERSION:
            raise ValueError("unsupported anchor bank state")
        try:
            bank = cls(int(state["slots"]), int(state["per_slot"]))
            expected = {
                "obs": (bank._obs, np.float32),
                "actions": (bank._actions, np.int8),
                "behavior_log_probs": (bank._behavior_log_probs, np.float32),
                "behavior_probs": (bank._behavior_probs, np.float32),
                "advantages": (bank._advantages, np.float32),
                "returns": (bank._returns, np.float32),
                "generations": (bank._generations, np.int64),
                "episode_ids": (bank._episode_ids, np.int64),
                "sizes": (bank._sizes, np.int32),
                "seen": (bank._seen, np.int64),
                "active_generations": (bank._active_generations, np.int64),
            }
            for name, (target, dtype) in expected.items():
                value = np.asarray(state[name])
                if value.shape != target.shape or value.dtype != dtype:
                    raise ValueError(f"invalid anchor bank {name} buffer")
                if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
                    raise ValueError(f"anchor bank {name} buffer contains non-finite values")
                target[...] = value
            if np.any(bank._sizes < 0) or np.any(bank._sizes > bank.per_slot):
                raise ValueError("anchor bank sizes are out of range")
            if np.any(bank._seen < bank._sizes):
                raise ValueError("anchor bank seen counts are smaller than stored sizes")
            for slot in range(bank.slots):
                size = int(bank._sizes[slot])
                active = int(bank._active_generations[slot])
                if active < -1:
                    raise ValueError("anchor bank active generation is invalid")
                if size and (active < 0 or not np.all(bank._generations[slot, :size] == active)):
                    raise ValueError("anchor bank row generations disagree with active generation")
                if size == 0 and active == -1 and bank._seen[slot] != 0:
                    raise ValueError("empty anchor bank slot has a nonzero seen count")
            bank._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
            return bank
        except KeyError as exc:
            raise ValueError(f"anchor bank state is missing {exc.args[0]!r}") from exc


def _probability_rows(value: Any, name: str, samples: int) -> np.ndarray:
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.shape != (samples, ACTION_COUNT) or not np.isfinite(probabilities).all():
        raise ValueError(f"{name} must have shape (N, {ACTION_COUNT}) and finite values")
    if np.any(probabilities < 0):
        raise ValueError(f"{name} cannot contain negative values")
    totals = probabilities.sum(axis=1)
    if np.any(totals <= 0) or not np.allclose(totals, 1.0, atol=1e-4, rtol=0):
        raise ValueError(f"each row of {name} must sum to one")
    return probabilities / totals[:, None]


def _forward_kl(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tiny = np.finfo(np.float64).tiny
    safe_source = np.maximum(source, tiny)
    safe_target = np.maximum(target, tiny)
    terms = np.where(source > 0, source * (np.log(safe_source) - np.log(safe_target)), 0.0)
    return np.maximum(terms.sum(axis=1), 0.0)


def _ratios_from_log_prob(probabilities: np.ndarray, actions: np.ndarray, behavior_logp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = probabilities[np.arange(len(actions)), actions]
    log_ratio = np.full(len(actions), -np.inf, dtype=np.float64)
    positive = selected > 0
    log_ratio[positive] = np.log(selected[positive]) - behavior_logp[positive]
    # Unsupported extreme ratios are marked as clipped below and force support_ok=False.
    numeric_limit = math.log(1e100)
    numeric_clipped = log_ratio > numeric_limit
    ratio = np.zeros(len(actions), dtype=np.float64)
    finite = np.isfinite(log_ratio)
    ratio[finite] = np.exp(np.clip(log_ratio[finite], -745.0, numeric_limit))
    return ratio, numeric_clipped


def evaluate_change(
    incumbent_probs: Any,
    candidate_probs: Any,
    actions: Any,
    behavior_logp: Any,
    behavior_probs: Any,
    advantages: Any,
    episode_ids: Any,
    *,
    min_ess: float = 8.0,
    kl_limit: float = 0.05,
    clip_range: float = 0.2,
    min_episodes: int = 2,
) -> dict[str, Any]:
    """Score a candidate against one batch of behavior-policy evidence.

    Importance ratios use ``exp(behavior_logp)`` for the sampled action, never
    the possibly rounded ``behavior_probs[action]``. ``behavior_kl`` is the
    mean exact per-state ``KL(behavior || candidate)``. ``mean_kl`` and
    ``max_kl`` are the mean and maximum ``KL(incumbent || candidate)``. The
    score is the PPO clipped-surrogate difference using the same behavior
    advantages for both policies. ``episode_se`` is a cluster-robust empirical
    standard error over ``episode_ids``; it is not a confidence guarantee.
    """
    incumbent_raw = np.asarray(incumbent_probs)
    if incumbent_raw.ndim != 2 or incumbent_raw.shape[1:] != (ACTION_COUNT,):
        raise ValueError(f"incumbent_probs must have shape (N, {ACTION_COUNT})")
    samples = int(incumbent_raw.shape[0])
    if samples < 1:
        raise ValueError("at least one evidence row is required")
    incumbent = _probability_rows(incumbent_probs, "incumbent_probs", samples)
    candidate = _probability_rows(candidate_probs, "candidate_probs", samples)
    behavior = _probability_rows(behavior_probs, "behavior_probs", samples)
    action_values = np.asarray(actions)
    if action_values.shape != (samples,) or not np.issubdtype(action_values.dtype, np.integer):
        raise ValueError("actions must be an integer vector of length N")
    if np.any(action_values < 0) or np.any(action_values >= ACTION_COUNT):
        raise ValueError(f"actions must be in [0, {ACTION_COUNT - 1}]")
    action_values = action_values.astype(np.int64, copy=False)
    logp = np.asarray(behavior_logp, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    episodes = np.asarray(episode_ids)
    if logp.shape != (samples,) or not np.isfinite(logp).all() or np.any(logp > 1e-6):
        raise ValueError("behavior_logp must be a finite nonpositive vector of length N")
    if adv.shape != (samples,) or not np.isfinite(adv).all():
        raise ValueError("advantages must be a finite vector of length N")
    if episodes.shape != (samples,) or not np.issubdtype(episodes.dtype, np.integer):
        raise ValueError("episode_ids must be an integer vector of length N")
    if isinstance(min_ess, bool) or not math.isfinite(float(min_ess)) or min_ess < 0:
        raise ValueError("min_ess must be finite and nonnegative")
    if isinstance(kl_limit, bool) or not math.isfinite(float(kl_limit)) or kl_limit < 0:
        raise ValueError("kl_limit must be finite and nonnegative")
    if not math.isfinite(float(clip_range)) or not 0 <= clip_range < 1:
        raise ValueError("clip_range must be in [0, 1)")
    if isinstance(min_episodes, bool) or int(min_episodes) != min_episodes or min_episodes < 1:
        raise ValueError("min_episodes must be a positive integer")

    candidate_ratio, numeric_clip_candidate = _ratios_from_log_prob(candidate, action_values, logp)
    incumbent_ratio, numeric_clip_incumbent = _ratios_from_log_prob(incumbent, action_values, logp)
    lower, upper = 1.0 - clip_range, 1.0 + clip_range
    candidate_surrogate = np.minimum(candidate_ratio * adv, np.clip(candidate_ratio, lower, upper) * adv)
    incumbent_surrogate = np.minimum(incumbent_ratio * adv, np.clip(incumbent_ratio, lower, upper) * adv)
    row_deltas = candidate_surrogate - incumbent_surrogate
    mean_delta = float(np.mean(row_deltas))

    unique_episodes, inverse = np.unique(episodes, return_inverse=True)
    episode_count = int(len(unique_episodes))
    if episode_count < 2:
        episode_se = float("inf")
    else:
        centered = row_deltas - mean_delta
        cluster_sums = np.bincount(inverse, weights=centered, minlength=episode_count)
        variance = (episode_count / (episode_count - 1.0)) * float(np.dot(cluster_sums, cluster_sums)) / (samples * samples)
        episode_se = float(math.sqrt(max(variance, 0.0)))

    log_weights = np.full(samples, -np.inf, dtype=np.float64)
    selected_candidate = candidate[np.arange(samples), action_values]
    positive = selected_candidate > 0
    log_weights[positive] = np.log(selected_candidate[positive]) - logp[positive]
    max_log_weight = float(np.max(log_weights))
    if not math.isfinite(max_log_weight):
        ess = 0.0
    else:
        scaled_weights = np.exp(log_weights - max_log_weight)
        squared_sum = float(np.dot(scaled_weights, scaled_weights))
        ess = float(scaled_weights.sum() ** 2 / squared_sum) if squared_sum > 0 else 0.0

    behavior_kl_rows = _forward_kl(behavior, candidate)
    incumbent_kl_rows = _forward_kl(incumbent, candidate)
    behavior_kl = float(np.mean(behavior_kl_rows))
    mean_kl = float(np.mean(incumbent_kl_rows))
    max_kl = float(np.max(incumbent_kl_rows))
    weight_clip_fraction = float(np.mean((candidate_ratio < lower) | (candidate_ratio > upper)))
    numeric_clipped = bool(np.any(numeric_clip_candidate) or np.any(numeric_clip_incumbent))
    support_ok = bool(
        ess >= float(min_ess)
        and episode_count >= int(min_episodes)
        and behavior_kl <= float(kl_limit)
        and max_kl <= float(kl_limit)
        and not numeric_clipped
    )
    return {
        "mean_delta": mean_delta,
        "episode_se": episode_se,
        "ess": ess,
        "behavior_kl": behavior_kl,
        "max_kl": max_kl,
        "mean_kl": mean_kl,
        "weight_clip_fraction": weight_clip_fraction,
        "support_ok": support_ok,
        "sample_count": samples,
        "episode_count": episode_count,
    }


@dataclass(frozen=True)
class ProjectionResult:
    """Sequential projection output and finite-pass feasibility report."""

    delta: np.ndarray
    satisfied: bool
    max_violation: float
    passes_used: int


def project_delta(
    delta9: Any,
    reference_gradients: Any,
    *,
    tolerance: float = 0.0,
    passes: int = 3,
) -> ProjectionResult:
    """Project an advantage-ascent delta toward ``g dot delta >= -tolerance``.

    This is a finite-pass sequential half-space projection, inspired by GEM /
    PCGrad. It does not reverse the proposed direction or guarantee all
    constraints are satisfied; callers must inspect ``satisfied``.
    """
    original = np.asarray(delta9, dtype=np.float64)
    if original.size != 9 or not np.isfinite(original).all():
        raise ValueError("delta9 must contain exactly nine finite values")
    delta = original.reshape(9).copy()
    gradients = np.asarray(reference_gradients, dtype=np.float64)
    if gradients.size == 0:
        gradients = np.empty((0, 9), dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[1] != 9 or not np.isfinite(gradients).all():
        raise ValueError("reference_gradients must be a finite matrix with shape (K, 9)")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if isinstance(passes, bool) or int(passes) != passes or passes < 0:
        raise ValueError("passes must be a nonnegative integer")

    used = 0
    for pass_index in range(int(passes)):
        used = pass_index + 1
        changed = False
        for gradient in gradients:
            norm_squared = float(np.dot(gradient, gradient))
            if norm_squared == 0.0:
                continue
            dot = float(np.dot(gradient, delta))
            if dot < -float(tolerance):
                delta += ((-float(tolerance) - dot) / norm_squared) * gradient
                changed = True
        if not changed:
            break

    violations = np.maximum(-float(tolerance) - gradients @ delta, 0.0)
    max_violation = float(np.max(violations)) if len(violations) else 0.0
    satisfied = max_violation <= 1e-10
    return ProjectionResult(delta=delta, satisfied=satisfied, max_violation=max_violation, passes_used=used)

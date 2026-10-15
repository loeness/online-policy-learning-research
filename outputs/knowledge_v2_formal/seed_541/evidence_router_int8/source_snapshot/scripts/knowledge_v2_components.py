"""Explicit implementation fixes and nonblocking compression for V2 controls.

These components deliberately leave the archived V1 runner unchanged.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from scripts import run_knowledge_loop as loop

V1_CONSOLIDATE = loop._consolidate


class PreservingTrialArm(loop.OnlineArm):
    """Reset only the allocated row; preserve other in-flight PPO updates."""

    def _reset_allocated_slot(self, slot: int, *, evicted: bool) -> None:
        trial = loop._read_matrix(self.core)
        super()._reset_allocated_slot(slot, evicted=evicted)
        if self.arm in loop.NEURAL_ARMS and not self.shared_head:
            trial[slot].fill(0.)
            loop._install_matrix(self.core, trial, shared=False)


def behavior_relative_surrogate(agent, matrix, validation):
    """Finite-sample IS difference, using the actual sampling distribution.

    This is a diagnostic score, not a lower bound on policy value improvement.
    The advantages and state distribution still belong to the trial policy.
    """
    selected = [r for r in validation if r.slot >= 0 and
                r.generation == agent.slot_generation[r.slot]]
    if not selected:
        return float("nan"), 0
    observations = np.stack([r.obs for r in selected])
    actions = np.asarray([r.action for r in selected], dtype=np.int64)
    behavior = np.exp(np.asarray([r.log_prob for r in selected]))
    original = loop._read_matrix(agent.core)
    incumbent = loop._matrix_probs(agent, agent.committed_matrix, observations, restore=original)
    candidate = loop._matrix_probs(agent, matrix, observations, restore=original)
    indexes = np.arange(len(actions))
    ratio_candidate = candidate[indexes, actions] / np.clip(behavior, 1e-12, 1.)
    ratio_incumbent = incumbent[indexes, actions] / np.clip(behavior, 1e-12, 1.)
    advantage = np.asarray([r.advantage for r in selected], dtype=np.float64)
    # Both hypotheses use the same behavior-relative reference and clipping.
    def surrogate(ratio):
        return np.minimum(ratio * advantage,
                          np.clip(ratio, 1 - loop.CLIP_RANGE, 1 + loop.CLIP_RANGE) * advantage)
    return float(np.mean(surrogate(ratio_candidate) - surrogate(ratio_incumbent))), len(selected)


def nonblocking_consolidate(agent, validation, *, cycle):
    """Commit local learning independently; share only for a measured byte gain.

    Each shared candidate must be smaller than the independent int8 package
    and have bounded policy distortion against that fallback on old anchors
    and the current holdout. No return guarantee is inferred from action KL.
    """
    if agent.arm != "evidence_gate":
        return V1_CONSOLIDATE(agent, validation, cycle=cycle)
    trial = loop._read_matrix(agent.core)
    fallback = loop.encode(trial, "int8", loop.ACTOR_BUDGET_BYTES)
    selected = fallback
    record: dict[str, Any] = {
        "step": cycle * loop.CONSOLIDATE_EVERY, "arm": agent.arm,
        "kind": "nonblocking_consolidation", "validation_samples": len(validation),
        "actor_bytes_before": agent.compact.nbytes, "actor_budget_bytes": loop.ACTOR_BUDGET_BYTES,
        "fallback": "int8", "selected_candidate": "int8", "candidates": [],
    }
    anchors = agent.anchors.active_items(agent.slot_generation)
    visible = [r.obs for r in validation if r.slot >= 0 and
               r.generation == agent.slot_generation[r.slot]]
    obs = np.stack([a["obs"] for a in anchors] + visible) if anchors or visible else np.empty((0, 13), np.float32)
    if len(obs):
        fallback_probs = loop._matrix_probs(agent, fallback.decode(), obs, restore=trial)
        for method in ("global_mean_sparse", "centroid2_sparse", "truncated_svd"):
            try:
                candidate = loop.encode(trial, method, min(loop.ACTOR_BUDGET_BYTES, fallback.nbytes - 1))
            except ValueError:
                continue
            probs = loop._matrix_probs(agent, candidate.decode(), obs, restore=trial)
            kl = loop._kl_rows(fallback_probs, probs)
            mean_kl, worst_kl = float(kl.mean()), float(kl.max())
            # A fixed distortion gate for compression; PPO learning is retained
            # through the independent fallback even if every shared trial fails.
            feasible = candidate.nbytes < fallback.nbytes and mean_kl <= .002 and worst_kl <= .02
            record["candidates"].append({"method": method, "bytes": candidate.nbytes,
                                          "mean_kl": mean_kl, "max_kl": worst_kl,
                                          "eligible": bool(feasible), "observations": len(obs)})
            if feasible and candidate.nbytes < selected.nbytes:
                selected = candidate
    agent.compact = selected
    agent.committed_method = selected.method
    agent.committed_matrix = selected.decode()
    loop._install_matrix(agent.core, agent.committed_matrix, shared=False)
    agent.counts["accepted"] += 1
    agent.counts["codec_commits"] += 1
    if validation:
        observations = np.stack([r.obs for r in validation])
        probs = loop._policy_probs(agent.core, observations)
        for row, p in zip(validation, probs):
            if row.slot >= 0 and row.generation == agent.slot_generation[row.slot]:
                agent.anchors.add(row.slot, row.generation, row.obs, p)
    reset = loop._reset_actor_optimizer(agent)
    record.update({"selected_codec": selected.method, "selected_candidate": selected.method,
                   "accepted": True, "decision": "shared" if selected.method != "int8" else "independent_fallback",
                   "actor_bytes_after": selected.nbytes, "residual_optimizer_parameters_reset": reset})
    return record

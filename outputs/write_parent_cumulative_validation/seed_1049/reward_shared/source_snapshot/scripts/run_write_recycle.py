"""Run the write stream with bounded, evidence-based shared-parent recycling.

This module installs its hooks as soon as it is imported.  It intentionally
leaves the fixed-capacity ``run_write_stream`` entry point unchanged; invoking
this file selects the separate recycling cohort.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.write_parent_budget import ParentBudgetManager
from online_rl.write_cumulative_evidence import CumulativeEvidence
from scripts import run_write_stream as stream


PROTOCOL_PATH = ROOT / "docs" / "WRITE_RECYCLE_PROTOCOL.md"
SOURCE_ADDITIONS = (
    Path("scripts/run_write_recycle.py"),
    Path("src/online_rl/write_parent_budget.py"),
    Path("src/online_rl/write_cumulative_evidence.py"),
    Path("docs/WRITE_RECYCLE_PROTOCOL.md"),
)

_ORIGINALS: dict[str, Any] = {
    "load_config": stream._load_config,
    "new_agent": stream._new_agent,
    "record_visible_event": stream._record_visible_event,
    "reset_allocated_slot": stream._reset_allocated_slot,
    "maybe_assign_group": stream._maybe_assign_group,
    "route_before_action": stream._route_before_action,
    "save_checkpoint": stream._save_stream_checkpoint,
    "restore_checkpoint": stream.restore_stream_checkpoint,
}
_INSTALLED = False
_FORCE_NO_RECYCLE = False


class _CaptureWriter:
    """Capture the original assigner's optional single CSV row."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def writerow(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))


def _write_rows(writer: Any, rows: list[dict[str, Any]]) -> None:
    if writer is not None:
        for row in rows:
            writer.writerow(row)


def _clear_affected_anchors(agent: Any, result: Any) -> None:
    """Drop evidence whose shared-parent identity changed during recycling."""
    state = agent.anchors.state_dict()
    generations = dict(result.affected_generations)
    row_fields = (
        "obs", "actions", "behavior_log_probs", "behavior_probs",
        "advantages", "returns", "generations", "episode_ids",
    )
    for slot in result.affected_slots:
        for field in row_fields:
            state[field][slot].fill(-1 if field == "generations" else 0)
        state["sizes"][slot] = 0
        state["seen"][slot] = 0
        state["active_generations"][slot] = generations.get(int(slot), -1)
    agent.anchors = stream.AnchorBank.from_state_dict(state)


def _recycle_event(result: Any, manager: ParentBudgetManager) -> dict[str, Any]:
    return {
        "recycled": bool(result.recycled),
        "reason": str(result.reason),
        "group": result.group,
        "previous_group_version": result.previous_group_version,
        "group_version": result.group_version,
        "parent_versions": manager.version.astype(np.uint64, copy=False).tolist(),
        "affected_slots": list(result.affected_slots),
        "removed_slots": list(result.removed_slots),
        "added_slots": list(result.added_slots),
    }


def _wrapped_load_config() -> dict[str, Any]:
    config = dict(_ORIGINALS["load_config"]())
    config.setdefault("cumulative_profiles", True)
    config["parent_recycle"] = not _FORCE_NO_RECYCLE and bool(config.get("parent_recycle", True))
    return config


def _wrapped_new_agent(condition: str, seed: int, config: Mapping[str, Any], base_config: Mapping[str, Any]):
    run_config = dict(config)
    run_config.setdefault("cumulative_profiles", True)
    run_config["parent_recycle"] = not _FORCE_NO_RECYCLE and bool(run_config.get("parent_recycle", True))
    agent = _ORIGINALS["new_agent"](condition, seed, run_config, base_config)
    agent.run_config["parent_recycle"] = bool(run_config["parent_recycle"])
    agent.parent_budget_manager = ParentBudgetManager()
    agent.cumulative_evidence = (
        CumulativeEvidence(agent.slots)
        if (
            agent.run_config.get("phase") == "known"
            and bool(agent.run_config.get("cumulative_profiles", True))
        ) else None
    )
    agent.counts.setdefault("parent_recycles", 0)
    agent.counts.setdefault("parent_detached_slots", 0)
    agent.counts.setdefault("parent_joined_slots", 0)
    return agent


def _wrapped_record_visible_event(
    agent: Any, obs, action, reward, next_obs, terminated, *, task_name, probe
):
    result = _ORIGINALS["record_visible_event"](
        agent, obs, action, reward, next_obs, terminated,
        task_name=task_name, probe=probe,
    )
    agent._parent_bootstrap_route_pending = bool(
        not probe
        and not result.get("switched", False)
        and result.get("allocated_slot") is None
    )
    evidence = getattr(agent, "cumulative_evidence", None)
    if agent.run_config.get("phase") == "known" and evidence is not None:
        slot = agent.current_known_slot
        if slot is not None:
            evidence.observe(
                int(slot), obs, int(action), float(reward), next_obs, bool(terminated)
            )
    return result


def _wrapped_reset_allocated_slot(agent: Any, slot: int, *, reason: str) -> int:
    generation = _ORIGINALS["reset_allocated_slot"](agent, slot, reason=reason)
    evidence = getattr(agent, "cumulative_evidence", None)
    if evidence is not None:
        evidence.clear(int(slot))
    return generation


def _wrapped_route_before_action(agent: Any, obs: np.ndarray) -> dict[str, Any]:
    route = _ORIGINALS["route_before_action"](agent, obs)
    is_bootstrap_route = bool(getattr(agent, "_parent_bootstrap_route_pending", False))
    if is_bootstrap_route:
        agent._parent_bootstrap_route_pending = False
    slot = int(route.get("slot", -1))
    manager = getattr(agent, "parent_budget_manager", None)
    if (
        is_bootstrap_route
        or manager is None
        or not bool(agent.run_config.get("parent_recycle", True))
        or slot < 0
        or slot >= int(agent.core.local_slots)
    ):
        return route
    group = int(agent.core.group_for_slot(slot))
    if group < 0:
        return route
    # raw_steps advances for both probes and training interactions.  Repeated
    # route lookups within one action cannot inflate the parent frequency.
    step = int(agent.counts.get("raw_steps", agent.counts.get("environment_steps", 0)))
    if int(manager.touch_count[group]) == 0 or int(manager.last_used[group]) != step:
        manager.touch(group, step)
    return route


def _wrapped_maybe_assign_group(
    agent: Any,
    slot: int,
    profile: object,
    assignments_writer=None,
    *,
    bout: int = -1,
) -> dict[str, Any] | None:
    cumulative = getattr(agent, "cumulative_evidence", None)
    if agent.run_config.get("phase") == "known" and cumulative is not None:
        profile = cumulative.profile_for_slot(int(slot))
    capture = _CaptureWriter()
    outcome = _ORIGINALS["maybe_assign_group"](
        agent, slot, profile, capture, bout=bout
    )
    if not isinstance(outcome, Mapping) or outcome.get("reason") not in {
        "max_groups_reached", "capacity_full",
    }:
        _write_rows(assignments_writer, capture.rows)
        return outcome

    cycle = agent.cycle_state
    at_boundary = (
        not agent.rollout
        and int(agent.cycle_step) == 0
        and (cycle is None or int(cycle.get("fit_rows", 0)) == 0)
    )
    if (
        not at_boundary
        or int(agent.core.shared_groups) != 2
        or not bool(agent.run_config.get("parent_recycle", True))
        or getattr(agent, "parent_budget_manager", None) is None
    ):
        _write_rows(assignments_writer, capture.rows)
        return outcome

    manager = agent.parent_budget_manager
    result = manager.try_recycle(
        agent.core,
        agent.grouper,
        agent.profiles,
        current_slot=int(slot),
        slot_generations=agent.slot_generation,
        step=int(agent.counts.get("raw_steps", agent.counts.get("environment_steps", 0))),
    )
    event = _recycle_event(result, manager)
    old_scores: Any = {}
    if capture.rows:
        try:
            old_scores = json.loads(str(capture.rows[-1].get("scores", "{}")))
        except (TypeError, ValueError):
            old_scores = {"original_scores": capture.rows[-1].get("scores")}
    scores = {"capacity_assignment": old_scores, "parent_recycle": event}
    profile_rows = float(
        profile["reward_n"] if isinstance(profile, Mapping) else profile.reward_n
    )

    if not result.recycled:
        row = dict(capture.rows[-1]) if capture.rows else {
            "phase": agent.run_config["phase"], "bout": bout, "slot": int(slot),
            "generation": int(agent.slot_generation[slot]), "group": -1,
            "reason": str(outcome["reason"]), "profile_rows": profile_rows,
            "committed": 0,
        }
        row["scores"] = json.dumps(scores, ensure_ascii=False)
        _write_rows(assignments_writer, [row])
        return outcome

    agent.grouper = result.grouper
    added = set(int(value) for value in result.added_slots)
    for affected_slot in result.affected_slots:
        affected_slot = int(affected_slot)
        if affected_slot in added:
            generation = int(agent.slot_generation[affected_slot])
            agent.grouped_generation[affected_slot] = generation
            agent.profile_reasons[affected_slot] = str(result.reason)
            support = int(agent.profiles["movement_counts"][affected_slot].sum(dtype=np.uint64))
            reward_n = int(agent.profiles["reward_n"][affected_slot])
            agent.group_attempt_support[affected_slot] = np.uint64(support + reward_n)
        elif int(agent.core.group_for_slot(affected_slot)) < 0:
            agent.grouped_generation[affected_slot] = -1
            agent.profile_reasons[affected_slot] = "parent_recycled_detached"
            support = int(agent.profiles["movement_counts"][affected_slot].sum(dtype=np.uint64))
            reward_n = int(agent.profiles["reward_n"][affected_slot])
            agent.group_attempt_support[affected_slot] = np.uint64(support + reward_n)

    _clear_affected_anchors(agent, result)
    agent.counts["parent_recycles"] += 1
    agent.counts["parent_detached_slots"] += len(result.removed_slots)
    agent.counts["parent_joined_slots"] += len(result.added_slots)
    if "group_abstentions" in agent.counts:
        agent.counts["group_abstentions"] = max(0, int(agent.counts["group_abstentions"]) - 1)
    if "group_max_abstentions" in agent.counts and outcome.get("reason") == "max_groups_reached":
        agent.counts["group_max_abstentions"] = max(
            0, int(agent.counts["group_max_abstentions"]) - 1
        )
    agent.counts["effective_preserving_bindings"] = int(
        agent.counts.get("effective_preserving_bindings", 0)
    ) + len(result.added_slots)

    stream._clear_optimizer_state(agent)
    roundtrip = stream.step1.roundtrip_actor(agent, agent.run_config)
    assignment_scores = {
        "assignment": result.assignment.scores,
        "parent_recycle": event,
        "actor_roundtrip": roundtrip,
    }
    row = {
        "phase": agent.run_config["phase"], "bout": bout, "slot": int(slot),
        "generation": int(agent.slot_generation[slot]), "group": int(result.group),
        "reason": str(result.reason), "binding": "preserve_effective",
        "profile_rows": profile_rows,
        "scores": json.dumps(assignment_scores, ensure_ascii=False), "committed": 1,
    }
    _write_rows(assignments_writer, [row])
    return {
        "group": int(result.group), "reason": str(result.reason),
        "binding": "preserve_effective", "committed": True,
        "parent_recycle": event,
    }


def _wrapped_save_checkpoint(agent: Any, path: Path, foundation_sha: str, capacity_rows: list[dict[str, Any]]):
    result = _ORIGINALS["save_checkpoint"](agent, path, foundation_sha, capacity_rows)
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manager = getattr(agent, "parent_budget_manager", ParentBudgetManager())
    payload["parent_budget_manager"] = manager.state_dict()
    cumulative = getattr(agent, "cumulative_evidence", None)
    payload["cumulative_evidence"] = (
        cumulative.state_dict() if cumulative is not None else None
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    foundation_bytes = Path(agent.base_config["foundation"]).stat().st_size
    package_cap = int(agent.run_config["package_budget_bytes"])
    full_bytes = stream.full_persistent_bytes(foundation_bytes, len(data))
    if full_bytes > package_cap:
        checkpoint_path.unlink(missing_ok=True)
        if capacity_rows and capacity_rows[-1] is result:
            capacity_rows.pop()
        raise ValueError(
            f"W0 plus parent-manager checkpoint is {full_bytes} bytes, "
            f"exceeding package cap {package_cap}"
        )
    checkpoint_path.write_bytes(data)
    added_capacity = {
        "checkpoint_bytes": len(data),
        "foundation_bytes": int(foundation_bytes),
        "full_persistent_bytes": int(full_bytes),
        "package_budget_bytes": package_cap,
        "parent_state_bytes": int(manager.nbytes),
        "cumulative_state_bytes": int(cumulative.nbytes) if cumulative is not None else 0,
    }
    result.update(added_capacity)
    if capacity_rows:
        capacity_rows[-1].update(added_capacity)
    return result


def _wrapped_restore_checkpoint(path: str | Path, base_config: Mapping[str, Any] | None = None):
    agent = _ORIGINALS["restore_checkpoint"](path, base_config)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("parent_budget_manager")
    agent.parent_budget_manager = (
        ParentBudgetManager.from_state_dict(state)
        if state is not None else ParentBudgetManager()
    )
    agent.counts.setdefault("parent_recycles", 0)
    agent.counts.setdefault("parent_detached_slots", 0)
    agent.counts.setdefault("parent_joined_slots", 0)
    cumulative_state = checkpoint.get("cumulative_evidence")
    if cumulative_state is not None:
        agent.cumulative_evidence = CumulativeEvidence.from_state_dict(cumulative_state)
    elif not hasattr(agent, "cumulative_evidence"):
        agent.cumulative_evidence = (
            CumulativeEvidence(agent.slots)
            if (
                agent.run_config.get("phase") == "known"
                and bool(agent.run_config.get("cumulative_profiles", True))
            ) else None
        )
    return agent


def install_hooks() -> None:
    """Install the recycling wrapper into the stream module once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    stream._load_config = _wrapped_load_config
    stream._new_agent = _wrapped_new_agent
    stream._record_visible_event = _wrapped_record_visible_event
    stream._reset_allocated_slot = _wrapped_reset_allocated_slot
    stream._maybe_assign_group = _wrapped_maybe_assign_group
    stream._route_before_action = _wrapped_route_before_action
    stream._save_stream_checkpoint = _wrapped_save_checkpoint
    stream.restore_stream_checkpoint = _wrapped_restore_checkpoint
    stream.PROTOCOL_PATH = PROTOCOL_PATH
    stream.SOURCE_FILES = tuple(dict.fromkeys((*stream.SOURCE_FILES, *SOURCE_ADDITIONS)))
    _INSTALLED = True


def uninstall_hooks() -> None:
    """Restore the stock module functions; primarily useful for isolated tests."""
    global _INSTALLED
    if not _INSTALLED:
        return
    stream._load_config = _ORIGINALS["load_config"]
    stream._new_agent = _ORIGINALS["new_agent"]
    stream._record_visible_event = _ORIGINALS["record_visible_event"]
    stream._reset_allocated_slot = _ORIGINALS["reset_allocated_slot"]
    stream._maybe_assign_group = _ORIGINALS["maybe_assign_group"]
    stream._route_before_action = _ORIGINALS["route_before_action"]
    stream._save_stream_checkpoint = _ORIGINALS["save_checkpoint"]
    stream.restore_stream_checkpoint = _ORIGINALS["restore_checkpoint"]
    stream.PROTOCOL_PATH = ROOT / "docs" / "WRITE_STREAM_PROTOCOL.md"
    additions = set(SOURCE_ADDITIONS)
    stream.SOURCE_FILES = tuple(path for path in stream.SOURCE_FILES if path not in additions)
    _INSTALLED = False


install_hooks()


def main() -> None:
    global _FORCE_NO_RECYCLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("known", "unknown"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bout-steps", type=int)
    parser.add_argument("--domain", choices=("navigation", "detour"))
    parser.add_argument("--local-slots", type=int, choices=(6, 7))
    parser.add_argument("--evaluation-targets", type=int, choices=(6, 12))
    parser.add_argument("--foundation", type=Path, help="override the W0 foundation checkpoint")
    parser.add_argument(
        "--no-recycle", action="store_true",
        help="keep the fixed-parent assignment behavior while retaining cumulative profiles",
    )
    args = parser.parse_args()
    _FORCE_NO_RECYCLE = bool(args.no_recycle)
    config = stream._load_config()
    config["parent_recycle"] = not _FORCE_NO_RECYCLE and bool(config.get("parent_recycle", True))
    # run_cohort reloads its config, so persist this invocation's mode through
    # the installed loader while leaving the stock config file untouched.
    _FORCE_NO_RECYCLE = not bool(config["parent_recycle"])
    result = stream.run_cohort(
        args.phase, args.output, selection_path=args.selection, workers=args.workers,
        conditions=args.conditions, seeds=args.seeds, bout_steps=args.bout_steps,
        domain=args.domain, local_slots=args.local_slots,
        evaluation_targets=args.evaluation_targets, foundation=args.foundation,
    )
    print(json.dumps({"phase": result["phase"], "conditions": result["conditions"],
                      "seeds": result["seeds"], "cohort_cost": result["cohort_cost"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

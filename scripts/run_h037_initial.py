"""Draft H037 paired online harness, for an independently reviewed server run.

This executable is intentionally not launched by import. It retains the known
task/group navigation protocol and compares one possible same-group eviction
handoff against LRU. The supplied example has unset gates and cannot start.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import platform
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import torch

from online_rl.h037_evidence import H037Evidence
from online_rl.h037_integration import H037Controller
from online_rl.storage_decision import bootstrap_schedule, compensated_single_parent_install, task_name
from scripts import run_knowledge_loop as loop
from scripts import run_storage_decision as sd
from scripts import run_write_stream as ws


class H037StopError(RuntimeError):
    """Raised after a critical H037 event is recorded and before slot reset."""

    def __init__(self, event: dict[str, Any], *, event_persisted: bool) -> None:
        self.event = dict(event)
        self.event_persisted = bool(event_persisted)
        super().__init__(
            "H037 stopped before reset: "
            f"status={event.get('status')!r}, reason={event.get('reason')!r}, "
            f"event_persisted={self.event_persisted}"
        )


_H037_EVENT_STATUSES = frozenset({"accept", "reject", "skip", "error"})


def _checked_h037_event(value: Any) -> tuple[dict[str, Any], bool]:
    """Return a serializable event and whether its shape requires stopping."""
    if not isinstance(value, Mapping):
        return {
            "format": "h037_runner_event_error_v1",
            "status": "error",
            "reason": "invalid_h037_event_mapping",
            "observed_type": type(value).__name__,
        }, True

    event = dict(value)
    status = event.get("status")
    if not isinstance(status, str) or status not in _H037_EVENT_STATUSES:
        try:
            status_text = str(status)
        except Exception:
            status_text = f"<{type(status).__name__}>"
        try:
            observed = json.loads(json.dumps(event, ensure_ascii=False, default=sd._json_default, allow_nan=False))
        except Exception as exc:
            observed = {
                "type": type(value).__name__,
                "repr": repr(value)[:1000],
                "serialization_error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "format": "h037_runner_event_error_v1",
            "status": "error",
            "reason": "invalid_h037_event_status",
            "observed_status": status_text,
            "observed_event": observed,
        }, True

    try:
        json.dumps(event, ensure_ascii=False, default=sd._json_default, allow_nan=False)
    except Exception as exc:
        return {
            "format": "h037_runner_event_error_v1",
            "status": "error",
            "reason": "unserializable_h037_event",
            "observed_status": status,
            "serialization_error": f"{type(exc).__name__}: {exc}",
        }, True
    return event, False


def _config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != "h037_initial_v0_draft":
        raise ValueError("invalid H037 draft config format")
    if raw.get("pair_kind") not in ("dynamics", "reward_conflict"):
        raise ValueError("pair_kind must be dynamics or reward_conflict")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a fixed nonnegative integer")
    for key in ("bootstrap_steps_per_task", "bout_steps"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 256:
            raise ValueError(f"{key} must be a positive multiple of 256")
    for key in ("capacity_per_slot", "min_train_rows", "min_holdout_rows", "package_budget_bytes",
                "max_environment_steps", "max_cpu_seconds", "max_wall_seconds"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a predeclared positive integer")
    candidate = raw.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("candidate config must be an object")
    for key in ("lambda_old", "ridge", "delta_v", "epsilon_old", "epsilon_res", "compensation_atol"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"candidate.{key} must be a predeclared finite nonnegative number")
    if float(candidate["delta_v"]) <= 0:
        raise ValueError("candidate.delta_v must be positive")
    if float(candidate["compensation_atol"]) <= 0:
        raise ValueError("candidate.compensation_atol must be positive")
    if float(candidate["epsilon_res"]) > 1:
        raise ValueError("candidate.epsilon_res cannot exceed a probability difference of 1")
    for key in ("max_iter", "max_eval", "actor_budget_bytes"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"candidate.{key} must be a positive integer")
    if candidate.get("codec") != "int8" or candidate["actor_budget_bytes"] != 89:
        raise ValueError("this harness is restricted to the existing 89-byte int8 actor")
    if raw["capacity_per_slot"] < raw["min_train_rows"] + raw["min_holdout_rows"]:
        raise ValueError("per-slot capacity must cover the two configured evidence splits")
    fraction = raw.get("holdout_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    if 2 * (60 * raw["capacity_per_slot"] + 89) > raw["package_budget_bytes"]:
        raise ValueError("the raw per-slot evidence arrays already exceed the entire package budget")
    planned_steps = 2 * raw["bootstrap_steps_per_task"] + 10 * raw["bout_steps"]
    if planned_steps > raw["max_environment_steps"]:
        raise ValueError(f"planned {planned_steps} interactions exceed the configured limit")
    return raw


def _bundle(agent: Any, foundation_sha: str) -> dict[str, Any]:
    payload = sd._capture_learner(agent, foundation_sha)
    payload["h037_controller"] = agent.h037_controller.state_dict()
    # Include the complete bounded stream state, including state which this
    # particular controlled runner does not ordinarily query after restore.
    # This makes the charged checkpoint and the common prefix agree.
    payload["h037_stream_aux"] = {
        "profiles": {key: value.copy() for key, value in agent.profiles.items()},
        "profile_reasons": list(agent.profile_reasons),
        "grouped_generation": agent.grouped_generation.copy(),
        "group_attempt_support": agent.group_attempt_support.copy(),
        "windows": [window.state_dict() for window in agent.windows],
        "grouper": agent.grouper.state_dict(),
        "anchors": agent.anchors.state_dict(),
        "current_known_slot": getattr(agent, "current_known_slot", None),
        "reward_by_task": dict(agent.reward_by_task),
        "steps_by_task": dict(agent.steps_by_task),
        "observed_reward_by_task": dict(agent.observed_reward_by_task),
        "observed_steps_by_task": dict(agent.observed_steps_by_task),
        "goal_episodes_by_task": dict(agent.goal_episodes_by_task),
        "complete_episodes_by_task": dict(agent.complete_episodes_by_task),
        "positive_complete_episodes_by_task": dict(agent.positive_complete_episodes_by_task),
        "next_episode_id": int(agent.next_episode_id),
        "decision_cpu_s": float(agent.decision_cpu_s),
        "evaluation_cpu_s": float(agent.evaluation_cpu_s),
    }
    return payload


def _restore_stream_aux(agent: Any, payload: dict[str, Any]) -> None:
    aux = payload["h037_stream_aux"]
    for name, target in agent.profiles.items():
        value = np.asarray(aux["profiles"][name])
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"invalid restored profile {name}")
        target[...] = value
    agent.profile_reasons = list(aux["profile_reasons"])
    agent.grouped_generation[...] = aux["grouped_generation"]
    agent.group_attempt_support[...] = aux["group_attempt_support"]
    agent.windows = [ws.EvidenceWindow.from_state_dict(item) for item in aux["windows"]]
    if len(agent.windows) != agent.slots:
        raise ValueError("restored window count differs from slot count")
    agent.grouper = ws.RewardAwareGroups.from_state_dict(aux["grouper"])
    agent.anchors = ws.AnchorBank.from_state_dict(aux["anchors"])
    for name in ("current_known_slot", "reward_by_task", "steps_by_task",
                 "observed_reward_by_task", "observed_steps_by_task",
                 "goal_episodes_by_task", "complete_episodes_by_task",
                 "positive_complete_episodes_by_task", "next_episode_id",
                 "decision_cpu_s", "evaluation_cpu_s"):
        setattr(agent, name, copy.deepcopy(aux[name]))


def _same_state(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and bool(torch.equal(left, right)))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and bool(np.array_equal(left, right)))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and set(left) == set(right)
                and all(_same_state(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same_state(a, b) for a, b in zip(left, right)))
    return bool(left == right)


def _serialized(payload: dict[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _attach(agent: Any, controller: H037Controller, *, enabled: bool, capacity_check: Any,
            event_path: Path | None = None) -> None:
    agent.h037_controller = controller
    agent.h037_enabled = bool(enabled)
    agent.h037_capacity_check = capacity_check
    agent.h037_event_path = event_path
    old_add = agent.evidence.add_rollout

    def add_rollout(rows: Any, *, group: int, task: str, bout: int) -> None:
        old_add(rows, group=group, task=task, bout=bout)
        if rows:
            slot, generation = int(rows[0].slot), int(rows[0].generation)
            if any(int(row.slot) != slot or int(row.generation) != generation for row in rows):
                raise RuntimeError("H037 requires each rollout to contain one slot generation")
            controller.record_rollout(agent, rows, slot=slot, generation=generation, task=task, group=group)

    agent.evidence.add_rollout = add_rollout


def _prepare_task(agent: Any, task: str, group: int) -> tuple[int, dict[str, Any]]:
    # Kept in step with the controlled storage runner; resolve itself mutates
    # cache mappings. The H037 transaction starts only after it has returned.
    allocation = agent.cache.resolve(task)
    slot = int(allocation["slot"])
    inherited = False
    if allocation["allocated"]:
        if agent.h037_enabled and allocation["evicted_slot"] is not None:
            raw_event = agent.h037_controller.pre_reset(agent, allocation, capacity_check=agent.h037_capacity_check)
            event, invalid_event = _checked_h037_event(raw_event)
            must_stop = (
                invalid_event
                or event.get("status") == "error"
                or event.get("reason") == "complete_package_exceeds_budget"
            )
            event_path = agent.h037_event_path
            if event_path is not None:
                with event_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, default=sd._json_default) + "\n")
                    stream.flush()
                    if must_stop:
                        os.fsync(stream.fileno())
            if must_stop:
                # An error or a verified complete-package cap failure ends the
                # campaign before the allocation reset or the next env step.
                # Missing/failed event persistence also fails closed here.
                raise H037StopError(event, event_persisted=event_path is not None)
        ws._reset_allocated_slot(agent, slot, reason="known_task_lru_allocation")
        agent.runtime_counts["slot_allocations"] += 1
        evicted = int(allocation["evicted_slot"] is not None)
        agent.runtime_counts["slot_evictions"] += evicted
        agent.counts["slot_evictions"] += evicted
        agent.slot_group_tags[slot] = int(group)
        inherited = bool(agent.parent_group == -2 or agent.parent_group == int(group))
        agent.core.set_slot_group(slot, 0 if inherited else -1, preserve_effective=False)
        if inherited:
            agent.runtime_counts["prior_bindings"] = agent.runtime_counts.get("prior_bindings", 0) + 1
    agent.current_known_slot = slot
    allocation["inherited_parent"] = inherited
    allocation["physical_group_at_start"] = int(agent.core.group_for_slot(slot))
    allocation["parent_group_at_start"] = int(agent.parent_group)
    return slot, allocation


def _records(mapping: dict[str, Any], entries: list[tuple[int, int]], steps: int, start: int) -> list[dict[str, Any]]:
    return [
        {"bout": start + index, "task": task_name(mapping, group, variant),
         "group": group, "variant": variant, "steps": steps}
        for index, (group, variant) in enumerate(entries)
    ]


def _run(config_path: Path, output: Path, seed: int, run_id: str) -> dict[str, Any]:
    run_cpu_start, run_wall_start = time.process_time(), time.perf_counter()
    config_path = config_path.resolve(strict=True)
    if not config_path.is_relative_to(ROOT.resolve()):
        raise ValueError("configuration must be inside the project")
    config = _config(config_path)
    if seed != config.get("seed"):
        raise ValueError("CLI seed differs from the fixed exploratory configuration")
    output = output.resolve()
    if output == ROOT.resolve() or not output.is_relative_to(ROOT.resolve()):
        raise ValueError("output must be a fresh directory inside the project")
    output.mkdir(parents=True, exist_ok=False)
    (output / ".h037_run_id").write_text(run_id, encoding="ascii")

    # This is a draft runner, not an independent audit or final experiment.
    sd.SOURCE_FILES = sd.SOURCE_FILES + (
        Path("scripts/run_h037_initial.py"), Path("src/online_rl/h037_candidate.py"),
        Path("src/online_rl/h037_evidence.py"), Path("src/online_rl/h037_integration.py"),
        Path("src/online_rl/retention_behavioral_candidate.py"), Path("configs/c1.json"),
        Path("requirements.txt"), Path("requirements-lock.txt"),
        Path("tests/test_h037_initial.py"), Path("tests/test_h037_transactions.py"),
    )
    source_manifest = sd._snapshot_sources(output, config_path)
    base = loop.load_base_config()
    foundation = Path(base["foundation"])
    foundation_sha = sd._sha(foundation)
    canonical_config = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    (output / "deployment_config.json").write_bytes(canonical_config)
    storage_config = json.loads((ROOT / "configs/storage_decision.json").read_text(encoding="utf-8"))
    storage_config["package_budget_bytes"] = int(config["package_budget_bytes"])
    storage_config["torch_threads"] = 1

    def capacity_check(agent: Any, *, actor_data: bytes, proposed_parent_version: int, **_context: Any) -> dict[str, Any]:
        checkpoint_bytes = len(_serialized(_bundle(agent, foundation_sha)))
        total = foundation.stat().st_size + checkpoint_bytes + len(actor_data) + len(canonical_config)
        return {"fits": total <= config["package_budget_bytes"], "package_bytes": total,
                "budget_bytes": config["package_budget_bytes"], "controller_state_included": True,
                "parent_version": proposed_parent_version,
                "checkpoint_bytes": checkpoint_bytes, "actor_bytes": len(actor_data),
                "foundation_bytes": foundation.stat().st_size, "config_bytes": len(canonical_config)}

    def write_package(folder: Path, agent: Any) -> dict[str, int]:
        folder.mkdir(parents=True, exist_ok=False)
        actor, _, _ = sd._encode_live_actor(agent)
        checkpoint = _serialized(_bundle(agent, foundation_sha))
        total = foundation.stat().st_size + len(checkpoint) + len(actor) + len(canonical_config)
        if total > config["package_budget_bytes"]:
            raise ValueError(f"actual package {total} exceeds configured cap {config['package_budget_bytes']}")
        (folder / "checkpoint.pt").write_bytes(checkpoint)
        (folder / "actor.bin").write_bytes(actor)
        return {"package_bytes": total, "checkpoint_bytes": len(checkpoint), "actor_bytes": len(actor),
                "foundation_bytes": foundation.stat().st_size, "config_bytes": len(canonical_config)}

    mapping = sd.pair_mapping(seed, config["pair_kind"])
    bootstrap = bootstrap_schedule(mapping, config["bootstrap_steps_per_task"])
    env, runtime = sd._runtime_initial(seed, bootstrap[0]["task"])
    try:
        agent = sd._new_agent(seed, base, storage_config)
        controller_config = {
            "format": "h037_controller_v1",
            **config["candidate"],
            "holdout_fraction": float(config["holdout_fraction"]),
            "min_victim_train_rows": config["min_train_rows"],
            "min_victim_holdout_rows": config["min_holdout_rows"],
            "min_old_train_rows": config["min_train_rows"],
            "min_old_holdout_rows": config["min_holdout_rows"],
            "min_resident_holdout_rows": config["min_holdout_rows"],
        }
        controller = H037Controller(controller_config, H037Evidence(config["capacity_per_slot"], slots=2))
        _attach(agent, controller, enabled=False, capacity_check=capacity_check)
        initial_actor, _, _ = sd._encode_live_actor(agent)
        initial_package = capacity_check(agent, actor_data=initial_actor,
                                         proposed_parent_version=controller.parent_version)
        if not initial_package["fits"]:
            raise ValueError("initial complete package exceeds the configured budget before interaction")
    except BaseException:
        env.close()
        raise
    original_prepare = sd._prepare_task
    sd._prepare_task = _prepare_task
    try:
        sd._run_schedule(agent, env, runtime, bootstrap, folder=output / "bootstrap", phase="bootstrap")
        local, shared, groups = ws._actor_arrays(agent.core)
        parent = sd.common_mean_effective(local, shared, groups, [0, 1])
        new_local, new_shared, new_groups, _ = compensated_single_parent_install(
            local, shared, groups, agent.slot_group_tags, parent=parent, parent_group=0)
        agent.core.install_local_matrix(new_local)
        agent.core.install_shared_matrix(new_shared)
        with torch.no_grad():
            agent.core.slot_groups.copy_(torch.as_tensor(new_groups, dtype=torch.long))
        agent.parent_group = 0
        sd._encode_live_actor(agent)
        ws._clear_optimizer_state(agent)
        # The controller notices the newly formed parent at the next rollout
        # and advances its version before accepting new evidence.
        warmup = _records(mapping, [(0, 0), (0, 1)], config["bout_steps"], 2)
        sd._run_schedule(agent, env, runtime, warmup, folder=output / "warmup", phase="warmup")
        if agent.rollout:
            raise RuntimeError("prefix ends with an incomplete PPO rollout")
        prefix_package = write_package(output / "prefix", agent)
        prefix_payload = _bundle(agent, foundation_sha)
        prefix_simulator = sd._capture_simulator(env, runtime)
        sd._write_json(output / "prefix" / "simulator.json", prefix_simulator)
        future = _records(mapping, [(0, 2), (0, 0), (0, 1), (1, 0)], config["bout_steps"], 4)
        branches = {}
        for name, enabled in (("lru", False), ("handoff", True)):
            branch_cpu_start, branch_wall_start = time.process_time(), time.perf_counter()
            restored = sd._restore_learner(prefix_payload, base, storage_config)
            _restore_stream_aux(restored, prefix_payload)
            restored_controller = H037Controller.from_state_dict(prefix_payload["h037_controller"])
            event_folder = output / name
            event_folder.mkdir(parents=True, exist_ok=False)
            _attach(restored, restored_controller, enabled=enabled, capacity_check=capacity_check,
                    event_path=event_folder / "handoff_events.jsonl")
            branch_env, branch_runtime = sd._restore_simulator(prefix_simulator)
            try:
                # Compare the full in-scope learner and simulator, including
                # the action, behavior, cache and environment RNG states,
                # before allowing either branch to take its first step.
                if not _same_state(_bundle(restored, foundation_sha), prefix_payload):
                    raise RuntimeError(f"{name} learner did not restore the shared prefix")
                if not _same_state(sd._capture_simulator(branch_env, branch_runtime), prefix_simulator):
                    raise RuntimeError(f"{name} simulator did not restore the shared prefix")
                expected_victim = task_name(mapping, 0, 0)
                if restored.cache.eviction != "lru" or future[0]["task"] in restored.cache.task_to_slot:
                    raise RuntimeError("the first continuation must allocate under LRU")
                victim_index = min(range(restored.cache.capacity),
                                   key=lambda i: (int(restored.cache.last_used[i]), i))
                if restored.cache.slot_to_task[victim_index] != expected_victim:
                    raise RuntimeError("the scheduled first LRU victim is not the fixed target task")
                bouts = sd._run_schedule(restored, branch_env, branch_runtime, future,
                                         folder=output / name / "stream", phase="future", action=name)
                package = write_package(output / name / "final", restored)
                branches[name] = {"bouts": bouts, "package": package,
                                  "handoff_counters": restored_controller.counters,
                                  "last_handoff_event": restored_controller.last_event,
                                  "environment_steps": sum(int(record["steps"]) for record in bouts),
                                  "cpu_seconds": time.process_time() - branch_cpu_start,
                                  "wall_seconds": time.perf_counter() - branch_wall_start}
            finally:
                branch_env.close()
        victim_task = task_name(mapping, 0, 0)
        primary = {}
        for name in ("lru", "handoff"):
            record = branches[name]["bouts"][1]
            if record["task"] != victim_task or int(record["steps"]) != int(config["bout_steps"]):
                raise RuntimeError("primary victim revisit does not match the fixed schedule")
            primary[name] = float(record["return_full"]) / int(record["steps"])
        result = {"status": "EXPLORATORY_UNAUDITED", "seed": seed, "mapping": mapping,
                  "initial_package": initial_package, "prefix_package": prefix_package, "branches": branches,
                  "source_manifest": source_manifest,
                  "primary_metric": {"name": "first_victim_revisit_reward_per_step",
                                     "task": victim_task, "window_steps": int(config["bout_steps"]),
                                     "lru": primary["lru"], "handoff": primary["handoff"],
                                     "handoff_minus_lru": primary["handoff"] - primary["lru"]},
                  "common_prefix_restore_verified": True,
                  "environment": {"python": sys.version, "platform": platform.platform(),
                                  "machine": platform.machine(), "numpy": np.__version__,
                                  "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                                  "gpu_used": False},
                  "prefix_environment_steps": sum(int(row["steps"]) for row in bootstrap + warmup),
                  "total_cpu_seconds_before_summary": time.process_time() - run_cpu_start,
                  "total_wall_seconds_before_summary": time.perf_counter() - run_wall_start,
                  "interpretation": "Engineering draft; no confirmatory statistical claim."}
        sd._write_json(output / "summary.json", result)
        return result
    finally:
        sd._prepare_task = original_prepare
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    # One process and one CPU thread only. RLIMIT_CPU covers this process's
    # accumulated CPU; the real-time timer starts after config parsing. A
    # supervising process is still needed to account for launch overhead and
    # enforce the overall campaign / temporary-storage reservation.
    if sys.platform != "linux":
        raise RuntimeError("this draft can run only in the approved Linux environment")
    import resource
    import signal

    limits = _config(args.config.resolve(strict=True))
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_CPU)
    requested_soft = int(limits["max_cpu_seconds"])
    requested_hard = requested_soft + 1
    if old_hard != resource.RLIM_INFINITY and requested_hard > old_hard:
        raise RuntimeError("existing CPU hard limit is below the configured reservation")
    resource.setrlimit(resource.RLIMIT_CPU, (requested_soft, requested_hard))

    def stop_at_limit(signum: int, _frame: Any) -> None:
        # BaseException avoids being mistaken for a recoverable candidate error
        # by transaction handlers which intentionally catch ordinary Exception.
        raise SystemExit(f"H037 campaign limit reached (signal {signum})")

    signal.signal(signal.SIGXCPU, stop_at_limit)
    signal.signal(signal.SIGALRM, stop_at_limit)
    signal.setitimer(signal.ITIMER_REAL, float(limits["max_wall_seconds"]))
    run_id = uuid.uuid4().hex
    process_cpu_before = time.process_time()
    process_wall_before = time.perf_counter()
    try:
        result = _run(args.config, args.output, args.seed, run_id)
    except BaseException as exc:
        # Keep already written artifacts. Only write in the fresh directory
        # created for this invocation and bearing its unique marker.
        target = args.output.resolve()
        marker = target / ".h037_run_id"
        try:
            if target.is_relative_to(ROOT.resolve()) and marker.is_file() and marker.read_text(encoding="ascii") == run_id:
                sd._write_json(target / "failure.json", {
                    "status": "FAILED_OR_INTERRUPTED", "reason": f"{type(exc).__name__}: {exc}",
                    "seed": args.seed, "cpu_seconds_since_limit_setup": time.process_time() - process_cpu_before,
                    "wall_seconds_since_limit_setup": time.perf_counter() - process_wall_before,
                    "note": "Partial accounting only; preserve all artifacts and use the campaign supervisor ledger.",
                })
        except Exception:
            pass
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": result["status"], "seed": result["seed"], "output": os.fspath(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

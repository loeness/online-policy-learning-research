"""Generate one unlabeled C1 behavior stream and compare task-free routers.

Each learner is evaluated offline against the same recorded stream. The
observer-only context IDs are held out of all learner calls and used only in
the post-hoc metrics written beside the per-step traces.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.c1_envs import C1NavigationEnv, c1_contexts  # noqa: E402
from online_rl.c1_strong_models import C1ModelController, load_source_prior  # noqa: E402
from online_rl.envs import NavigationEnv  # noqa: E402
from online_rl.experiment import BlankEnv, make_model  # noqa: E402
from online_rl.task_free_router import TaskFreeRouter  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


PROTOCOL_PATH = ROOT / "docs" / "KNOWLEDGE_ORGANIZATION_PROTOCOL.md"
OUTPUTS = ROOT / "outputs"
ALGORITHM_VARIANTS = (
    ("robust_bank", "robust"),
    ("naive_bank", "naive"),
    ("no_bank_window", "no_bank"),
)
WINDOWS = (8, 32, 64)
_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
SOURCE_FILES = (
    ROOT / "src" / "online_rl" / "task_free_router.py",
    ROOT / "scripts" / "run_task_free_identification.py",
)
INPUT_FILES = (
    ROOT / "data" / "foundation.pt",
    ROOT / "data" / "pretrain_source.npz",
    ROOT / "data" / "pretrain_source.json",
    PROTOCOL_PATH,
)
DEPENDENCY_FILES = (
    ROOT / "src" / "online_rl" / "c1_envs.py",
    ROOT / "src" / "online_rl" / "c1_strong_models.py",
    ROOT / "src" / "online_rl" / "envs.py",
    ROOT / "src" / "online_rl" / "experiment.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_sources(output_dir: Path) -> dict[str, Any]:
    """Preserve the exact executable source and input hashes at phase launch."""

    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    for source_path in SOURCE_FILES:
        relative = source_path.relative_to(ROOT)
        destination = snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        source_hashes[str(relative)] = _sha256_file(destination)
    input_hashes: dict[str, str] = {}
    for input_path in INPUT_FILES:
        relative = input_path.relative_to(ROOT)
        input_hashes[str(relative)] = _sha256_file(input_path)
    dependency_hashes: dict[str, str] = {}
    for dependency_path in DEPENDENCY_FILES:
        relative = dependency_path.relative_to(ROOT)
        dependency_hashes[str(relative)] = _sha256_file(dependency_path)
    return {
        "source_snapshot_directory": str(snapshot_dir.relative_to(ROOT)),
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "dependency_sha256": dependency_hashes,
    }


class FrozenFoundationActor:
    """Small inference wrapper around the existing frozen foundation actor."""

    def __init__(self, seed: int = 0) -> None:
        env = DummyVecEnv([lambda: BlankEnv(NavigationEnv())])
        model = make_model(env, {"seed": int(seed), "ent_coef": 0.06})
        weights = torch.load(ROOT / "data" / "foundation.pt", map_location="cpu", weights_only=True)
        model.policy.load_state_dict(weights)
        model.policy.set_training_mode(False)
        self.policy = model.policy
        self.env = env
        self._torch = torch

    def probabilities(self, observation: np.ndarray, temperature: float = 2.0) -> np.ndarray:
        with self._torch.no_grad():
            values = np.zeros(13, dtype=np.float32)
            values[:4] = np.asarray(observation, dtype=np.float32)
            tensor = self._torch.as_tensor(values[None], dtype=self._torch.float32)
            features = self.policy.extract_features(tensor)
            latent = self.policy.mlp_extractor.forward_actor(features)
            logits = self.policy.action_net(latent)[0].to(self._torch.float64) / float(temperature)
            probs = self._torch.softmax(logits, dim=-1).cpu().numpy()
        return np.asarray(probs, dtype=np.float64)

    def close(self) -> None:
        self.env.close()


def _protocol_hash() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _generate_stream(seed: int, minimum_segment: int, maximum_segment: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run the common behavior once, then log arrays for isolated diagnostics."""

    schedule_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed + 100_003)
    names = list(c1_contexts()) * 2
    schedule_rng.shuffle(names)
    segment_lengths = schedule_rng.integers(minimum_segment, maximum_segment + 1, size=len(names))
    registry = c1_contexts()
    env = C1NavigationEnv(registry[names[0]])
    observation, _ = env.reset(seed=seed)
    actor = FrozenFoundationActor(seed)
    behavior = C1ModelController(
        "adaptive_assignment",
        window=32,
        source_prior=load_source_prior(),
        exploration_rate=0.08,
    )

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    context_ids: list[int] = []
    segment_ids: list[int] = []
    segment_steps: list[int] = []
    reset_after: list[bool] = []
    schedule_records: list[dict[str, int | str]] = []
    global_step = 0
    start = time.perf_counter()
    cpu_start = time.process_time()

    for segment_id, (name, segment_length) in enumerate(zip(names, segment_lengths.tolist())):
        env.set_context(registry[name])
        schedule_records.append({
            "segment": segment_id,
            "true_context": name,
            "length": int(segment_length),
            "start_step": global_step,
        })
        for local_step in range(int(segment_length)):
            base = actor.probabilities(observation)
            command_probs = behavior.action_probabilities(observation, base)
            command_probs = 0.85 * command_probs + 0.15 / 4.0
            command_probs /= command_probs.sum()
            action = int(action_rng.choice(4, p=command_probs))
            next_observation, reward, terminated, truncated, _info = env.step(action)
            # The behavior controller sees the same unlabelled transition as a
            # normal online agent. Hidden schedule labels are stored separately.
            behavior.observe(observation, action, next_observation, reward)
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            next_observations.append(np.asarray(next_observation, dtype=np.float32).copy())
            actions.append(action)
            rewards.append(float(reward))
            context_ids.append(list(registry).index(name))
            segment_ids.append(segment_id)
            segment_steps.append(local_step)
            reset_after.append(bool(terminated or truncated))
            global_step += 1
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset()

    elapsed = time.perf_counter() - start
    actor.close()
    env.close()
    stream = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        # Observer-only truth, intentionally kept separate from learner inputs.
        "context_ids": np.asarray(context_ids, dtype=np.int8),
        "segment_ids": np.asarray(segment_ids, dtype=np.int8),
        "segment_steps": np.asarray(segment_steps, dtype=np.int16),
        "reset_after": np.asarray(reset_after, dtype=np.bool_),
    }
    details = {
        "seed": int(seed),
        "behavior": "C1ModelController(adaptive_assignment, window=32) + frozen foundation actor + 0.15 uniform action mixture",
        "steps": int(global_step),
        "generation_seconds_once": float(elapsed),
        "generation_cpu_seconds_once": float(time.process_time() - cpu_start),
        "schedule": schedule_records,
    }
    return stream, details


class WindowOnlyPredictor:
    """Recent-window action-to-direction counts with no persistent slots."""

    def __init__(self, window: int, alpha: float = 0.5) -> None:
        self.window = int(window)
        self.alpha = float(alpha)
        self.events: list[tuple[int, int, int]] = []

    @staticmethod
    def _cell(obs: np.ndarray) -> tuple[int, int]:
        return int(round(float(obs[0]) * 8)), int(round(float(obs[1]) * 8))

    def predict(self, obs: np.ndarray, action: int) -> tuple[np.ndarray, bool]:
        x, _ = self._cell(obs)
        band = x // 3
        counts = np.zeros(4, dtype=np.float64)
        for event_band, event_action, direction in self.events:
            if event_band == band and event_action == action:
                counts[direction] += 1
        untried = bool(counts.sum() == 0)
        probabilities = counts + self.alpha
        return probabilities / probabilities.sum(), untried

    def observe(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> bool:
        before = self._cell(obs)
        after = self._cell(next_obs)
        delta = (after[0] - before[0], after[1] - before[1])
        if delta == (0, 0):
            return True
        try:
            direction = _DELTAS.index(delta)
        except ValueError as exc:
            raise ValueError(f"unexpected movement delta {delta}") from exc
        self.events.append((before[0] // 3, int(action), direction))
        if len(self.events) > self.window:
            del self.events[0]
        return False

    def nbytes(self) -> dict[str, int]:
        python_bytes = sys.getsizeof(self.events)
        python_bytes += sum(sys.getsizeof(event) + sum(sys.getsizeof(value) for value in event) for event in self.events)
        return {"recent_window": int(python_bytes), "total": int(python_bytes)}


def _ari(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    n = len(true_labels)
    if n < 2:
        return float("nan")
    contingency: dict[tuple[int, int], int] = {}
    true_counts: dict[int, int] = {}
    predicted_counts: dict[int, int] = {}
    for truth, predicted in zip(true_labels.tolist(), predicted_labels.tolist()):
        pair = (int(truth), int(predicted))
        contingency[pair] = contingency.get(pair, 0) + 1
        true_counts[pair[0]] = true_counts.get(pair[0], 0) + 1
        predicted_counts[pair[1]] = predicted_counts.get(pair[1], 0) + 1
    choose2 = lambda value: value * (value - 1) / 2
    observed = sum(choose2(value) for value in contingency.values())
    true_pairs = sum(choose2(value) for value in true_counts.values())
    predicted_pairs = sum(choose2(value) for value in predicted_counts.values())
    total_pairs = choose2(n)
    expected = true_pairs * predicted_pairs / total_pairs
    maximum = 0.5 * (true_pairs + predicted_pairs)
    denominator = maximum - expected
    return float((observed - expected) / denominator) if denominator else 1.0


def _posthoc_metrics(
    stream: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    prediction_losses: list[float],
    prediction_correct: list[bool],
    movement_predictions: int,
    censored_count: int,
    memory: dict[str, int],
    elapsed_seconds: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    truth = stream["context_ids"].astype(np.int64)
    routes = np.asarray([row.get("generation", -1) for row in rows], dtype=np.int64)
    physical_slots = np.asarray([row["slot"] for row in rows], dtype=np.int64)
    resolved = routes >= 0
    resolved_truth = truth[resolved]
    resolved_routes = routes[resolved]
    purity = 0.0
    completeness_by_context: list[float] = []
    if len(resolved_truth):
        purity = sum(
            int(np.max(np.bincount(resolved_truth[resolved_routes == cluster])))
            for cluster in np.unique(resolved_routes)
        ) / len(resolved_truth)
        for label in np.unique(truth):
            label_routes = routes[(truth == label) & resolved]
            completeness_by_context.append(
                float(np.max(np.bincount(label_routes)) / len(label_routes)) if len(label_routes) else 0.0
            )

    schedule = metadata["schedule"]
    first_visit_segment: dict[int, int] = {}
    first_visit_counts: dict[int, dict[int, int]] = {}
    for record in schedule:
        label = list(c1_contexts()).index(str(record["true_context"]))
        if label not in first_visit_segment:
            first_visit_segment[label] = int(record["segment"])
            start = int(record["start_step"])
            end = start + int(record["length"])
            counts: dict[int, int] = {}
            for route in routes[start:end].tolist():
                if route >= 0:
                    counts[int(route)] = counts.get(int(route), 0) + 1
            first_visit_counts[label] = counts

    # Observer-only one-to-one assignment: a shared cluster generation can
    # match at most one hidden context in this diagnostic.
    first_visit_edges = [
        (count, label, cluster)
        for label, counts in first_visit_counts.items()
        for cluster, count in counts.items()
    ]
    first_visit_edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    first_visit_cluster: dict[int, int] = {}
    cluster_to_context: dict[int, int] = {}
    for count, label, cluster in first_visit_edges:
        if count <= 0 or label in first_visit_cluster or cluster in cluster_to_context:
            continue
        first_visit_cluster[label] = cluster
        cluster_to_context[cluster] = label

    revisit_total = 0
    revisit_correct = 0
    revisit_latency: list[int] = []
    revisit_latency_censored_steps = 0
    revisit_segments_missed = 0
    revisit_segments_not_evaluable = 0
    warmup_gate_accepted = 0
    warmup_gate_total = 0
    revisit_gate_accepted = 0
    revisit_gate_total = 0
    gate_correct = 0
    for record in schedule:
        segment_id = int(record["segment"])
        label = list(c1_contexts()).index(str(record["true_context"]))
        start = int(record["start_step"])
        end = start + int(record["length"])
        if first_visit_segment.get(label) == segment_id:
            for row in rows[start:end]:
                warmup_gate_total += 1
                accepted = row["generation"] >= 0 and row["confidence"] >= 0.80 and row["support"] >= 0.50
                warmup_gate_accepted += int(accepted)
                gate_correct += int(accepted and cluster_to_context.get(row["generation"]) == label)
        elif label in first_visit_cluster:
            revisit_total += end - start
            expected_cluster = first_visit_cluster[label]
            found = None
            for offset, route in enumerate(routes[start:end].tolist()):
                if route == expected_cluster:
                    found = offset
                    break
            if found is not None:
                revisit_correct += int(np.sum(routes[start:end] == expected_cluster))
                revisit_latency.append(found)
            else:
                revisit_segments_missed += 1
                revisit_latency_censored_steps += end - start
            for row in rows[start:end]:
                revisit_gate_total += 1
                accepted = row["generation"] >= 0 and row["confidence"] >= 0.80 and row["support"] >= 0.50
                revisit_gate_accepted += int(accepted)
                gate_correct += int(accepted and cluster_to_context.get(row["generation"]) == label)
        elif first_visit_segment.get(label) is not None:
            revisit_segments_not_evaluable += 1

    false_route_changes = 0
    for record in schedule:
        start = int(record["start_step"])
        end = start + int(record["length"])
        prior = None
        for route in routes[start:end].tolist():
            if route < 0:
                continue
            if prior is not None and route != prior:
                false_route_changes += 1
            prior = route

    clusters_to_truth: dict[int, list[int]] = {}
    for cluster in np.unique(routes[routes >= 0]).tolist():
        clusters_to_truth[int(cluster)] = truth[routes == cluster].tolist()
    modal_clusters_by_truth: dict[int, set[int]] = {}
    for cluster, labels in clusters_to_truth.items():
        modal = int(np.bincount(np.asarray(labels, dtype=np.int64)).argmax())
        modal_clusters_by_truth.setdefault(modal, set()).add(cluster)
    redundant_clusters = sum(max(0, len(cluster_set) - 1) for cluster_set in modal_clusters_by_truth.values())

    resolved_changes = sum(
        1 for previous, current in zip(routes[:-1], routes[1:])
        if previous >= 0 and current >= 0 and previous != current
    )
    result = {
        "steps": int(len(routes)),
        "transition_accuracy": float(np.mean(prediction_correct)) if prediction_correct else None,
        "transition_logloss": float(np.mean(prediction_losses)) if prediction_losses else None,
        "moving_predictions_scored": int(movement_predictions),
        "wall_censored_transitions": int(censored_count),
        "route_coverage": float(np.mean(resolved)) if len(resolved) else 0.0,
        "resolved_route_purity_by_slot_generation": float(purity),
        "resolved_route_completeness_macro_by_true_context": float(np.mean(completeness_by_context)) if completeness_by_context else 0.0,
        "offline_ari_including_unresolved": _ari(truth, routes),
        "known_context_revisit_generation_reuse": (
            float(revisit_correct / revisit_total) if revisit_total else None
        ),
        "revisit_change_latency_mean_steps_successful_only": float(np.mean(revisit_latency)) if revisit_latency else None,
        "revisit_change_latency_mean_steps_censored_at_segment_length": (
            float((sum(revisit_latency) + revisit_latency_censored_steps) / (len(revisit_latency) + revisit_segments_missed))
            if revisit_latency or revisit_segments_missed else None
        ),
        "revisit_segments_with_expected_generation": int(len(revisit_latency)),
        "revisit_segments_missed": int(revisit_segments_missed),
        "revisit_latency_censored_steps": int(revisit_latency_censored_steps),
        "revisit_segments_not_evaluable_no_warmup_generation": int(revisit_segments_not_evaluable),
        "route_changes_inside_true_stable_segment": int(false_route_changes),
        "all_stream_resolved_route_changes": int(resolved_changes),
        "redundant_slot_generations_by_posthoc_true_context": int(redundant_clusters),
        "unique_route_slot_generations": int(len(np.unique(routes[routes >= 0]))),
        "physical_slot_ids_used": sorted(int(value) for value in np.unique(physical_slots[physical_slots >= 0]).tolist()),
        "warmup_gate_accepted_fraction": float(warmup_gate_accepted / warmup_gate_total) if warmup_gate_total else 0.0,
        "revisit_gate_accepted_fraction": float(revisit_gate_accepted / revisit_gate_total) if revisit_gate_total else 0.0,
        "gate_accepted_generation_correct_fraction_under_injective_warmup_mapping": float(gate_correct / (warmup_gate_accepted + revisit_gate_accepted)) if (warmup_gate_accepted + revisit_gate_accepted) else None,
        "posthoc_injective_warmup_mapping": {str(cluster): int(label) for cluster, label in cluster_to_context.items()},
        "gate_thresholds": {"confidence_at_least": 0.80, "support_at_least": 0.50},
        "average_pre_action_confidence": float(np.mean([row["confidence"] for row in rows])) if rows else 0.0,
        "average_pre_action_support": float(np.mean([row["support"] for row in rows])) if rows else 0.0,
        "pre_action_untried_fraction": float(np.mean([row["untried_action"] for row in rows])) if rows else 0.0,
        "allocations": int(metadata.get("allocations", 0)),
        "evictions": int(metadata.get("evictions", 0)),
        "switches": int(metadata.get("switches", 0)),
        "censored_seen_by_router": int(metadata.get("censored", 0)),
        "active_slots_at_end": metadata.get("active_slots", []),
        "memory_bytes": memory,
        "evaluation_seconds": float(elapsed_seconds),
        "mean_seconds_per_step": float(elapsed_seconds / len(routes)) if len(routes) else 0.0,
        "active_exploration": "not tested; the same behavior stream is used for all methods",
    }
    return result


def _evaluate_router(
    learner_stream: dict[str, np.ndarray],
    *,
    kind: str,
    mode: str,
    window: int,
    output_path: Path,
) -> dict[str, Any]:
    router = None if mode == "no_bank" else TaskFreeRouter(max_contexts=6, window=window, mode=mode)
    predictor = WindowOnlyPredictor(window) if router is None else None
    observations = learner_stream["observations"]
    actions = learner_stream["actions"]
    rewards = learner_stream["rewards"]
    next_observations = learner_stream["next_observations"]
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    correct: list[bool] = []
    movement_predictions = 0
    censored_count = 0
    slot_generations = [-1] * 6
    next_generation = 0
    start_time = time.perf_counter()
    cpu_start = time.process_time()
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as zipped:
        writer = None
        for index in range(len(actions)):
            obs = observations[index]
            action = int(actions[index])
            nxt = next_observations[index]
            before = (int(round(float(obs[0]) * 8)), int(round(float(obs[1]) * 8)))
            after = (int(round(float(nxt[0]) * 8)), int(round(float(nxt[1]) * 8)))
            delta = (after[0] - before[0], after[1] - before[1])
            actual_direction = _DELTAS.index(delta) if delta in _DELTAS else None

            if router is not None:
                query_start = time.perf_counter()
                route = router.route(obs)
                probs = router.predict_direction(obs, action)
                query_seconds = time.perf_counter() - query_start
                routed_slot = int(route["slot"])
                route_generation = slot_generations[routed_slot] if routed_slot >= 0 else -1
                confidence = float(route["confidence"])
                support = float(route["support"])
                route_reason = str(route["reason"])
                if routed_slot >= 0:
                    cell_support = router._cell_total(routed_slot, before[0] // 3, action)
                    untried = cell_support == 0
                else:
                    untried = True
            else:
                query_start = time.perf_counter()
                probs, untried = predictor.predict(obs, action)
                query_seconds = time.perf_counter() - query_start
                routed_slot = -1
                route_generation = -1
                confidence = 0.0
                support = float(len(predictor.events) / window)
                route_reason = "no_context_bank"

            is_censored = actual_direction is None
            if is_censored:
                censored_count += 1
                predicted_correct = None
                logloss = None
            else:
                movement_predictions += 1
                logloss = -math.log(max(float(probs[actual_direction]), 1e-12))
                predicted_correct = int(int(np.argmax(probs)) == actual_direction)
                losses.append(float(logloss))
                correct.append(bool(predicted_correct))

            observe_start = time.perf_counter()
            if router is not None:
                observe_info = router.observe(obs, action, float(rewards[index]), nxt)
            else:
                was_censored = predictor.observe(obs, action, nxt)
                observe_info = {
                    "allocated_slot": None,
                    "evicted_slot": None,
                    "switched": False,
                    "censored": was_censored,
                    "reason": "wall_censored" if was_censored else "observed",
                }
            allocated_slot = observe_info.get("allocated_slot")
            allocated_generation = None
            if allocated_slot is not None:
                slot_generations[int(allocated_slot)] = next_generation
                allocated_generation = next_generation
                next_generation += 1
            observe_seconds = time.perf_counter() - observe_start
            row = {
                "step": index,
                "action": action,
                "band": int(before[0] // 3),
                "slot": routed_slot,
                "generation": int(route_generation),
                "allocated_generation": allocated_generation,
                "confidence": confidence,
                "support": support,
                "reason": route_reason,
                "untried_action": bool(untried),
                "actual_direction_observer_only": actual_direction,
                "predicted_direction": int(np.argmax(probs)),
                "transition_correct": predicted_correct,
                "transition_logloss": logloss,
                "wall_censored": is_censored,
                "allocated_slot": allocated_slot,
                "evicted_slot": observe_info.get("evicted_slot"),
                "switched": bool(observe_info.get("switched", False)),
                "query_seconds": query_seconds,
                "observe_seconds": observe_seconds,
            }
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(zipped, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)

    elapsed = time.perf_counter() - start_time
    memory = router.nbytes() if router is not None else predictor.nbytes()
    if router is not None:
        memory["state_dict_json_bytes"] = len(
            json.dumps(router.state_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    else:
        memory["state_dict_json_bytes"] = len(
            json.dumps(predictor.events, separators=(",", ":")).encode("utf-8")
        )
    router_metadata = {
        "allocations": router.allocations if router is not None else 0,
        "evictions": router.evictions if router is not None else 0,
        "switches": router.switches if router is not None else 0,
        "censored": router._censored if router is not None else censored_count,
        "active_slots": list(router.active_slots) if router is not None else [],
    }
    return {
        "rows": rows,
        "prediction_losses": losses,
        "prediction_correct": correct,
        "movement_predictions": movement_predictions,
        "censored_count": censored_count,
        "memory": memory,
        "elapsed_seconds": elapsed,
        "cpu_seconds": float(time.process_time() - cpu_start),
        "router_metadata": router_metadata,
        "algorithm": kind,
        "window": int(window),
        "per_step_output": str(output_path.relative_to(ROOT)),
    }


def run_phase(phase: str, output_dir: Path, *, seeds: list[int] | None = None) -> dict[str, Any]:
    if phase == "pilot":
        seeds = [331] if seeds is None else seeds
        min_segment = max_segment = 64
    elif phase == "formal":
        seeds = [331, 347, 359] if seeds is None else seeds
        min_segment, max_segment = 384, 640
    else:
        raise ValueError("phase must be pilot or formal")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    provenance = _snapshot_sources(output_dir)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process = psutil.Process(os.getpid())
    rss_start = int(process.memory_info().rss)
    phase_start = time.perf_counter()
    phase_cpu_start = time.process_time()
    all_records: list[dict[str, Any]] = []
    seed_manifests: list[dict[str, Any]] = []

    for seed in seeds:
        stream, details = _generate_stream(seed, min_segment, max_segment)
        stream_path = output_dir / f"stream_seed_{seed}.npz"
        np.savez_compressed(stream_path, **stream)
        details["stream_file"] = str(stream_path.relative_to(ROOT))
        details["stream_sha256"] = hashlib.sha256(stream_path.read_bytes()).hexdigest()
        seed_manifests.append(details)
        # The learner receives only this four-field view. Observer truth and
        # switch boundaries stay outside the online evaluation call.
        learner_stream = {
            key: stream[key]
            for key in ("observations", "actions", "rewards", "next_observations")
        }
        for kind, mode in ALGORITHM_VARIANTS:
            for window in WINDOWS:
                file_path = output_dir / f"steps_seed_{seed}_{kind}_w{window}.csv.gz"
                diagnostics = _evaluate_router(
                    learner_stream,
                    kind=kind,
                    mode=mode,
                    window=window,
                    output_path=file_path,
                )
                result = _posthoc_metrics(
                    stream,
                    diagnostics["rows"],
                    diagnostics["prediction_losses"],
                    diagnostics["prediction_correct"],
                    diagnostics["movement_predictions"],
                    diagnostics["censored_count"],
                    diagnostics["memory"],
                    diagnostics["elapsed_seconds"],
                    {**details, **diagnostics["router_metadata"]},
                )
                result.update({
                    "algorithm": kind,
                    "window": int(window),
                    "per_step_output": diagnostics["per_step_output"],
                })
                result["seed"] = seed
                result["stream_file"] = str(stream_path.relative_to(ROOT))
                all_records.append(result)
                print(json.dumps({
                    "phase": phase,
                    "seed": seed,
                    "algorithm": kind,
                    "window": window,
                    "route_coverage": result["route_coverage"],
                    "ari": result["offline_ari_including_unresolved"],
                    "allocations": result["allocations"],
                }), flush=True)

    elapsed = time.perf_counter() - phase_start
    rss_end = int(process.memory_info().rss)
    manifest = {
        "phase": phase,
        **provenance,
        "protocol_sha256": _protocol_hash(),
        "protocol_file": str(PROTOCOL_PATH.relative_to(ROOT)),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "numpy": np.__version__,
        "seeds": seeds,
        "segment_lengths_inclusive": [min_segment, max_segment],
        "streams": seed_manifests,
        "generated_environment_interactions_once_per_seed": sum(item["steps"] for item in seed_manifests),
        "offline_algorithm_evaluations": len(all_records),
        "active_exploration": "not tested",
        "elapsed_seconds_including_stream_generation_and_offline_evaluation": float(elapsed),
        "cpu_seconds_including_stream_generation_and_offline_evaluation": float(time.process_time() - phase_cpu_start),
        "process_rss_start_bytes": rss_start,
        "process_rss_end_bytes": rss_end,
        "records": all_records,
    }
    _write_json(output_dir / "summary.json", manifest)
    _write_json(output_dir / "metrics.json", {"records": all_records})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = OUTPUTS / ("knowledge_identification_pilot" if args.phase == "pilot" else "knowledge_identification_formal")
    manifest = run_phase(args.phase, output, seeds=args.seeds)
    print(json.dumps({"phase": args.phase, "output": str(output), "records": len(manifest["records"])}), flush=True)


if __name__ == "__main__":
    main()

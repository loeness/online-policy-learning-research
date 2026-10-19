"""Frozen supplemental C1 neural experiments.

This module reuses the original C1 environment, history router, PPO rollout
buffer, and no-gradient evaluator.  It adds resumable stage 2/3 orchestration,
full shared-actor fine tuning, a seen-aware route fallback for the original
flat/dictionary layouts, and explicit continual-state accounting.

No experiment is started on import.  Long-running jobs are launched only by
``scripts/run_c1_supplement.py``.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .c1_continual_controls import (
    C1ContinualPPO,
    C1FullFineTunePolicy,
    export_payload as export_full_payload,
)
from .c1_envs import C1NavigationEnv, c1_contexts, context_index
from .c1_experiment import C1Stream, EP_FIELDS, STEP_FIELDS, augment, evaluate, new_router
from .c1_modular_controls import FlatGroupedAdditivePolicy, GroupedDictionaryPolicy
from .c1_policy import C1ResidualPolicy
from .compression import export_payload as export_residual_payload
from .compression_experiment import PackedSB3Adapter, actor_hash
from .envs import NavigationEnv
from .experiment import BlankEnv, VersionedRolloutBuffer, make_model, seed_everything, write_json
from .policy import TreePolicy, tensor_bytes


_INITIAL_SEEN = (True, True, False, True, True, False)
_FAMILY_MEMBERS = {7: (1, 2, 3), 8: (4, 5, 6)}
_STEP3_COMPRESSED = ("flat", "dictionary")


class C1SeenAwareResidualPolicy(C1ResidualPolicy):
    """C1 flat/dictionary residuals with an explicit seen-leaf fallback mask.

    The base residual implementation uses the first two siblings whenever a
    parent route is evaluated.  This supplemental control instead averages
    every sibling exposed so far.  The six Boolean entries are persistent
    checkpoint state and are counted in complete and live-state capacity.
    Tree routes retain their trainable family residuals and do not use this
    mask.
    """

    def _make_residual_layout(self, *, like: torch.Tensor) -> None:
        super()._make_residual_layout(like=like)
        self._register_seen_mask_for_instance(like)

    def _register_seen_mask_for_instance(self, like: torch.Tensor) -> None:
        if "seen_mask" in self._buffers:
            del self._buffers["seen_mask"]
        self.register_buffer(
            "seen_mask",
            torch.tensor(_INITIAL_SEEN, dtype=torch.bool, device=like.device),
            persistent=True,
        )

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "C1SeenAwareResidualPolicy":
        result = C1ResidualPolicy.from_spec.__func__(cls, spec)
        mask = spec.get("supplemental_seen_mask", list(_INITIAL_SEEN))
        if len(mask) != 6:
            raise ValueError("supplemental seen mask must contain six entries")
        result.seen_mask.copy_(torch.as_tensor(mask, dtype=torch.bool, device=result.seen_mask.device))
        return result

    def to_spec(self) -> dict[str, Any]:
        spec = super().to_spec()
        spec["supplemental_layout"] = "c1_seen_aware_residual_v1"
        spec["supplemental_seen_mask"] = self.seen_mask.detach().cpu().tolist()
        return spec

    def _fallback_correction(self, family_parent: int, local_phi: torch.Tensor) -> torch.Tensor:
        members = _FAMILY_MEMBERS[int(family_parent)]
        selected = [member for member in members if bool(self.seen_mask[member - 1])]
        if not selected:
            raise RuntimeError(f"family {family_parent} has no seen endpoint for fallback")
        with torch.no_grad():
            matrix = torch.stack([self._effective_matrix(endpoint) for endpoint in selected]).mean(0)
        return torch.nn.functional.linear(local_phi, matrix)

    def initialize_heldout(self, endpoint: int) -> None:
        super().initialize_heldout(endpoint)
        self.mark_seen(endpoint)

    def mark_seen(self, endpoint: int) -> None:
        endpoint = int(endpoint)
        if endpoint not in range(1, 7):
            raise ValueError("endpoint must be one through six")
        self.seen_mask[endpoint - 1] = True

    def storage_report(self) -> dict[str, int | str]:
        report = super().storage_report()
        report["supplemental_seen_mask_bytes"] = int(self.seen_mask.numel() * self.seen_mask.element_size())
        report["seen_endpoint_count"] = int(self.seen_mask.sum().item())
        return report


class _SupplementStream(C1Stream):
    """C1 stream that snapshots behavior before an action transition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_policy_observation: np.ndarray | None = None
        self.completed_episodes = self.successful_episodes = 0

    def _augmented(self, obs):
        value = super()._augmented(obs)
        self.last_policy_observation = np.asarray(value, dtype=np.float32).copy()
        return value

    def step(self, action):
        model = self.model
        if (isinstance(model, C1ContinualPPO) and model.bc_coefficient > 0.0
                and self.last_policy_observation is not None):
            # Called before C1Stream.step applies the action.  The reservoir
            # stores only the pre-action observation and current probabilities.
            model.observe_online(self.last_policy_observation)
        result = super().step(action)
        if result[2] or result[3]:
            self.completed_episodes += 1
            self.successful_episodes += int(result[4]["success"])
        return result


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_protocol(config_path: str | Path = "configs/c1_supplement.json") -> tuple[dict[str, Any], dict[str, Any]]:
    supplement = _load_json(config_path)
    base_path = Path(supplement["base_config"])
    base = _load_json(base_path)
    return supplement, base


def _source_hashes(config_path: str | Path, supplement: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, str]:
    paths = [
        Path(config_path),
        Path(supplement["base_config"]),
        Path(supplement["foundation"] if "foundation" in supplement else base["foundation"]),
        Path(base["dataset"]),
        Path("docs/C1_SUPPLEMENT_PROTOCOL.md"),
        Path("src/online_rl/c1_supplement_experiment.py"),
        Path("src/online_rl/c1_experiment.py"),
        Path("src/online_rl/c1_continual_controls.py"),
        Path("src/online_rl/c1_modular_controls.py"),
        Path("src/online_rl/c1_policy.py"),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _foundation(config: Mapping[str, Any], seed: int):
    foundation_config = {**config, "seed": int(seed)}
    model = make_model(DummyVecEnv([lambda: BlankEnv(NavigationEnv())]), foundation_config)
    state = torch.load(config["foundation"], map_location="cpu", weights_only=True)
    model.policy.load_state_dict(state, strict=True)
    return model


def _make_core(config: Mapping[str, Any], seed: int, method: str, tier: Mapping[str, Any], params: Mapping[str, Any]):
    seed_everything(seed)
    reference = _foundation(config, seed)
    epsilon = float(config["exploration_epsilon"])
    temperature = float(config.get("temperature", 2.0))
    leaf_rank = int(tier.get("leaf_rank", 2))
    dictionary_rank = int(tier.get("dictionary_rank", 12))
    try:
        if method == "tree":
            core = C1ResidualPolicy.from_reference(
                reference.policy, method="tree", leaf_rank=leaf_rank,
                dictionary_rank=dictionary_rank, exploration_epsilon=epsilon,
                temperature=temperature,
            )
        elif method in _STEP3_COMPRESSED:
            core = C1SeenAwareResidualPolicy.from_reference(
                reference.policy, method=method, leaf_rank=leaf_rank,
                dictionary_rank=dictionary_rank, exploration_epsilon=epsilon,
                temperature=temperature,
            )
        elif method == "grouped_dictionary":
            # This representation declares its own three/six atom grid; map
            # the C1 low/high tiers onto that supported grouped basis size.
            grouped_rank = 3 if dictionary_rank <= 6 else 6
            core = GroupedDictionaryPolicy.from_reference(
                reference.policy, leaf_rank=leaf_rank, dictionary_rank=grouped_rank,
                exploration_epsilon=epsilon, temperature=temperature,
            )
        elif method == "flat_grouped":
            core = FlatGroupedAdditivePolicy.from_reference(
                reference.policy, leaf_rank=leaf_rank,
                exploration_epsilon=epsilon, temperature=temperature,
            )
        elif method in ("full", "full_plain"):
            core = C1FullFineTunePolicy.from_reference(
                reference.policy, route_conditioned=(method == "full"),
                epsilon=epsilon, temperature=temperature,
            )
        else:
            raise ValueError(f"unknown C1 supplemental method {method!r}")
    finally:
        reference.env.close()

    learning_rate = float(params["learning_rate"])
    if isinstance(core, C1FullFineTunePolicy):
        optimizer = core.make_optimizer(learning_rate=learning_rate)
    else:
        optimizer = core.make_optimizer(learning_rate=learning_rate, transfer=False)
    return core, optimizer


def _is_full(core: torch.nn.Module) -> bool:
    return isinstance(core, C1FullFineTunePolicy)


def _policy_digest(core: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(core.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _continual_state(model) -> dict[str, Any] | None:
    if isinstance(model, C1ContinualPPO):
        return model.continual_state_dict()
    return None


def _make_payload(core, optimizer, router, metadata, continual_state=None):
    if _is_full(core):
        return export_full_payload(
            core, optimizer, router.snapshot(), metadata,
            replay_state=continual_state,
        )
    payload = export_residual_payload(core, optimizer, router.snapshot(), metadata)
    payload["supplemental_continual_state"] = continual_state
    return payload


def _restore_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = payload.get("spec", {})
    if spec.get("architecture") == "C1FullFineTunePolicy-v1":
        restored = C1FullFineTunePolicy.from_payload(payload)
    elif spec.get("layout") == "c1_grouped_dictionary_v1":
        restored = GroupedDictionaryPolicy.from_payload(payload)
    elif spec.get("layout") == "c1_grouped_additive_v1":
        restored = FlatGroupedAdditivePolicy.from_payload(payload)
    elif spec.get("supplemental_layout") == "c1_seen_aware_residual_v1":
        restored = C1SeenAwareResidualPolicy.from_payload(payload)
    else:
        restored = C1ResidualPolicy.from_payload(payload)
    restored["continual_state"] = payload.get("replay") or payload.get("supplemental_continual_state")
    del payload
    return restored


def _save_bundle(path: Path, core, optimizer, router, metadata, continual_state=None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_make_payload(core, optimizer, router, metadata, continual_state), path)
    return int(path.stat().st_size)


def _router_bytes(snapshot: Mapping[str, Any]) -> int:
    return len(json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8"))


def _continual_bytes(state: Mapping[str, Any] | None) -> dict[str, int]:
    if not state:
        return {"behavior_reservoir_bytes": 0, "ewc_fisher_bytes": 0, "ewc_reference_bytes": 0,
                "continual_metadata_bytes": 0}
    reservoir = state.get("behavior_reservoir") or {}
    arrays = reservoir.get("observations"), reservoir.get("teacher_probabilities")
    reservoir_bytes = sum(int(value.nbytes) for value in arrays if isinstance(value, np.ndarray))
    ewc_state = state.get("online_ewc") or {}
    fisher = ewc_state.get("fisher") or {}
    reference = ewc_state.get("reference") or {}
    return {
        "behavior_reservoir_bytes": reservoir_bytes,
        "ewc_fisher_bytes": tensor_bytes(fisher),
        "ewc_reference_bytes": tensor_bytes(reference),
        "continual_metadata_bytes": _router_bytes({k: v for k, v in state.items()
                                                    if k not in ("behavior_reservoir", "online_ewc")}),
    }


def _storage_snapshot(core, optimizer, router, model, rollout_buffer) -> dict[str, Any]:
    continual = _continual_state(model)
    if _is_full(core):
        representation = core.storage_report(optimizer)
        adaptive_bytes = int(representation["adaptive_actor_tensor_bytes"])
        trainable_bytes = sum(int(p.numel() * p.element_size()) for p in core.parameters() if p.requires_grad)
    else:
        representation = core.storage_report()
        residual_optimizer = sum(
            tensor_bytes(state) for parameter, state in optimizer.state.items()
            if getattr(parameter, "tree_residual", False)
        )
        adaptive_bytes = int(representation.get("residual_value_bytes", 0))
        adaptive_bytes += int(representation.get("coordinate_index_bytes", 0)) + residual_optimizer
        trainable_bytes = sum(int(p.numel() * p.element_size()) for p in core.parameters() if p.requires_grad)
        representation["optimizer_tensor_bytes"] = tensor_bytes(optimizer.state_dict())
        representation["residual_optimizer_tensor_bytes"] = residual_optimizer
        representation["adaptive_tensor_bytes"] = adaptive_bytes

    router_snapshot = router.snapshot()
    rollout_bytes = sum(
        int(value.nbytes) for value in vars(rollout_buffer).values() if isinstance(value, np.ndarray)
    ) if rollout_buffer is not None else 0
    continual_report = _continual_bytes(continual)
    live_lower_bound = (
        tensor_bytes(core.state_dict()) + tensor_bytes(optimizer.state_dict())
        + tensor_bytes([parameter.grad for parameter in core.parameters()]) + rollout_bytes
        + continual_report["behavior_reservoir_bytes"] + continual_report["ewc_fisher_bytes"]
        + continual_report["ewc_reference_bytes"] + _router_bytes(router_snapshot)
    )
    return {
        "representation": representation,
        "adaptive_parameter_and_optimizer_bytes": adaptive_bytes,
        "trainable_parameter_bytes": trainable_bytes,
        "policy_tensor_bytes": tensor_bytes(core.state_dict()),
        "optimizer_tensor_bytes": tensor_bytes(optimizer.state_dict()),
        "router_bytes": _router_bytes(router_snapshot),
        "rollout_bytes": rollout_bytes,
        **continual_report,
        "explicit_live_state_lower_bound_bytes": live_lower_bound,
        "parameter_update_scope": getattr(core, "parameter_update_scope", "residual_adapters_and_critic"),
        "shared_actor_copies": 1,
    }


class _SupplementCapacityCallback(BaseCallback):
    def __init__(self, stream, core, optimizer, router, output: Path, *, agent_cap: int,
                 adaptive_cap: int | None, outside_adaptive_budget: bool):
        super().__init__()
        self.stream, self.core, self.optimizer, self.router = stream, core, optimizer, router
        self.output = output
        self.agent_cap, self.adaptive_cap = int(agent_cap), adaptive_cap
        self.outside_adaptive_budget = bool(outside_adaptive_budget)
        self.serialized_peak = self.rollout_peak = self.live_peak = self.rss_peak = 0
        self.rows: list[dict[str, Any]] = []
        self._writer = None
        self._file = None

    def _on_training_start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "capacity.csv"
        exists = path.exists()
        self._file = path.open("a", newline="", encoding="utf-8")
        fields = ["global_step", "serialized_core_bytes", "rollout_bytes", "complete_agent_bytes",
                  "adaptive_bytes", "explicit_live_state_lower_bound_bytes", "rss_bytes"]
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        if not exists:
            self._writer.writeheader()
        self.measure()

    def _on_rollout_start(self):
        self.measure()

    def _on_step(self):
        return True

    def _on_training_end(self):
        try:
            self.measure()
        finally:
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None

    def measure(self):
        continual = _continual_state(self.model)
        snapshot = _storage_snapshot(self.core, self.optimizer, self.router, self.model,
                                     self.model.rollout_buffer)
        payload = _make_payload(
            self.core, self.optimizer, self.router,
            {"global_timesteps": int(self.model.num_timesteps),
             "stream_steps": int(self.stream.total_steps),
             "last_observation": self.model._last_obs,
             "last_episode_starts": self.model._last_episode_starts},
            continual,
        )
        storage = io.BytesIO()
        torch.save(payload, storage)
        serialized = int(storage.tell())
        rollout = int(snapshot["rollout_bytes"])
        total = serialized + rollout
        self.serialized_peak = max(self.serialized_peak, serialized)
        self.rollout_peak = max(self.rollout_peak, rollout)
        self.live_peak = max(self.live_peak, int(snapshot["explicit_live_state_lower_bound_bytes"]))
        self.rss_peak = max(self.rss_peak, int(psutil.Process().memory_info().rss))
        row = {"global_step": int(self.model.num_timesteps), "serialized_core_bytes": serialized,
               "rollout_bytes": rollout, "complete_agent_bytes": total,
               "adaptive_bytes": int(snapshot["adaptive_parameter_and_optimizer_bytes"]),
               "explicit_live_state_lower_bound_bytes": int(snapshot["explicit_live_state_lower_bound_bytes"]),
               "rss_bytes": self.rss_peak}
        self.rows.append(row)
        if self._writer is not None:
            self._writer.writerow(row)
        if not self.outside_adaptive_budget and self.adaptive_cap is not None and row["adaptive_bytes"] > self.adaptive_cap:
            raise RuntimeError(f"compressed adaptive budget exceeded: {row['adaptive_bytes']}>{self.adaptive_cap}")
        if total > self.agent_cap:
            raise RuntimeError(f"complete agent budget exceeded: {total}>{self.agent_cap}")

    def metrics(self) -> dict[str, Any]:
        return {"max_serialized_core_bytes": self.serialized_peak,
                "max_rollout_bytes": self.rollout_peak,
                "max_complete_agent_bytes": max((r["complete_agent_bytes"] for r in self.rows), default=0),
                "max_explicit_live_state_lower_bound_bytes": self.live_peak,
                "max_sampled_process_rss_bytes": self.rss_peak,
                "last_capacity_sample": self.rows[-1] if self.rows else None}


def _algorithm(core, optimizer, env, config, seed, control, coefficient, source_dataset, initialize_source=True):
    common = dict(
        policy=TreePolicy, env=env, learning_rate=float(config["learning_rate"]),
        n_steps=256, batch_size=64, n_epochs=4, gamma=0.98, gae_lambda=0.95,
        ent_coef=float(config["ent_coef"]), clip_range=0.2, target_kl=0.06,
        seed=int(seed), device="cpu", verbose=0,
    )
    if control in ("bc", "ewc"):
        model = C1ContinualPPO(
            **common, replay_capacity=128 if control == "bc" else 1,
            bc_coefficient=float(coefficient) if control == "bc" else 0.0,
            bc_batch_size=64, replay_seed=int(seed) + 13,
            ewc_coefficient=float(coefficient) if control == "ewc" else 0.0,
            ewc_interval=2048, ewc_batch_size=64, ewc_seed=int(seed) + 19,
            policy_kwargs={"max_nodes": 9},
            rollout_buffer_class=VersionedRolloutBuffer,
        )
    else:
        model = PPO(
            **common, policy_kwargs={"max_nodes": 9},
            rollout_buffer_class=VersionedRolloutBuffer,
        )
    model.policy = PackedSB3Adapter(core, optimizer)
    if control == "ewc" and initialize_source:
        if not _is_full(core):
            raise TypeError("online EWC requires the full shared actor")
        with np.load(source_dataset) as source:
            source_observations = source["observations"].copy()
            source_actions = source["actions"].copy()
        model.initialize_source_ewc(
            source_observations, source_actions, sample_size=64, seed=int(seed) + 23,
        )
    return model


def _attach_trace(stream, output, handles):
    output.mkdir(parents=True, exist_ok=True)
    for name, fields, attribute in (("steps", STEP_FIELDS, "steps_writer"), ("episodes", EP_FIELDS, "episodes_writer")):
        handle = (output / f"{name}.csv").open("w", newline="", encoding="utf-8")
        handles.append(handle)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        setattr(stream, attribute, writer)


def _count_optimizer_steps(optimizer, counts):
    # LeafAdam.step calls Adam.step. PyTorch may wrap both class methods, so
    # a post-hook can be invoked at the inner and outer level for one update.
    # LeafAdam increments its own counter only after the actual Adam call.
    if hasattr(optimizer, "step_count"):
        previous = [int(optimizer.step_count)]
        def count_leaf_step(*args, **kwargs):
            current = int(optimizer.step_count)
            counts["steps"] += current - previous[0]
            previous[0] = current
        return optimizer.register_step_post_hook(count_leaf_step)
    return optimizer.register_step_post_hook(lambda *args, **kwargs: counts.__setitem__("steps", counts["steps"] + 1))


def _sequence_schedule(base: Mapping[str, Any], stage: Mapping[str, Any]) -> dict[str, list[tuple[str, int]]]:
    return {
        "seen": [(name, int(stage["seen_bout_steps"])) for name in base["seen_sequence"]],
        "heldout": [(name, int(stage["heldout_bout_steps"])) for name in base["heldout_sequence"]],
        "revisit": [(name, int(stage["revisit_bout_steps"])) for name in base["revisit_sequence"]],
        "history": [(name, int(stage["history_bout_steps"])) for name in base["history_sequence"]],
    }


def _score_result(result: Mapping[str, Any]) -> float:
    if result.get("status") != "complete":
        return float("nan")
    final = result["final_eval"]
    adapted = result["adapted_eval"]
    final_mean = statistics.mean(row["success_rate"] for row in final.values())
    old_mean = statistics.mean(adapted[name]["success_rate"] for name in ("A0", "A1", "B0", "B1"))
    heldout_mean = statistics.mean(adapted[name]["success_rate"] for name in ("A2", "B2"))
    return 0.50 * final_mean + 0.25 * old_mean + 0.25 * heldout_mean


def run_arm(
    *,
    supplement: Mapping[str, Any],
    base: Mapping[str, Any],
    output: str | Path,
    seed: int,
    method: str,
    tier_name: str,
    tier: Mapping[str, Any],
    params: Mapping[str, Any],
    stage_name: str,
    control: str = "none",
    coefficient: float = 0.0,
    validation: bool = False,
    config_path: str | Path = "configs/c1_supplement.json",
) -> dict[str, Any]:
    """Run one neural arm, preserving checkpoint restore and all measured state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        return _load_json(result_path)

    schedule_name = "validation_schedule" if validation else "formal_schedule"
    stage = supplement[schedule_name]
    schedules = _sequence_schedule(base, stage)
    cap_key = "full_reference_agent_budget_bytes" if _is_method_full(method) else "original_agent_budget_bytes"
    complete_cap = int(supplement[cap_key] if _is_method_full(method) else supplement[cap_key][tier_name])
    adaptive_cap = None if _is_method_full(method) else int(supplement["original_adaptive_budget_bytes"][tier_name])
    outside_adaptive = _is_method_full(method)
    if _is_method_full(method) and control == "none":
        budget_label = "outside_original_adaptive_budget_full_shared_actor_reference"
    elif _is_method_full(method):
        budget_label = "outside_original_adaptive_budget_full_shared_actor_continual_reference"
    else:
        budget_label = "original_compressed_budget"

    frozen = {
        "stage": stage_name, "method": method, "control": control, "coefficient": float(coefficient),
        "seed": int(seed), "tier": tier_name, "tier_config": dict(tier), "params": dict(params),
        "schedule": stage, "validation": bool(validation), "complete_agent_cap_bytes": complete_cap,
        "adaptive_parameter_optimizer_cap_bytes": adaptive_cap,
        "budget_classification": budget_label,
        "input_sha256": _source_hashes(config_path, supplement, base),
    }
    write_json(output / "config.json", frozen)
    core = optimizer = model = stream = vector_env = history_env = None
    handles = []
    try:
        core, optimizer = _make_core(base, seed, method, tier, params)
        router = new_router(base)
        oracle_schedule = schedules["seen"] + schedules["heldout"] + schedules["revisit"]
        stream = _SupplementStream(oracle_schedule, seed + 17, router, mode="oracle")
        _attach_trace(stream, output / "oracle_stream", handles)
        vector_env = DummyVecEnv([lambda: stream])
        model_config = {**base, **params, "seed": int(seed)}
        initialization_cpu = time.process_time()
        model = _algorithm(core, optimizer, vector_env, model_config, seed, control, coefficient,
                           base["dataset"])
        initialization_cpu = time.process_time() - initialization_cpu
        update_counter = {"steps": 0}
        update_hook = _count_optimizer_steps(optimizer, update_counter)
        initial_actor_hash = actor_hash(core)
        stream.model = model
        if isinstance(model, C1ContinualPPO):
            model.num_timesteps = 0

        cap_callback = _SupplementCapacityCallback(
            stream, core, optimizer, router, output,
            agent_cap=complete_cap, adaptive_cap=adaptive_cap,
            outside_adaptive_budget=outside_adaptive,
        )
        training_records: dict[str, Any] = {}
        train_start = time.perf_counter()
        process_start = time.process_time()

        def learn_phase(name: str, phase_schedule: list[tuple[str, int]]) -> dict[str, Any]:
            steps = sum(count for _, count in phase_schedule)
            before_timesteps = int(model.num_timesteps)
            before_stream = int(stream.total_steps)
            before_updates = int(model._n_updates)
            before_actual_updates = update_counter["steps"]
            before_reward = stream.cumulative_return
            before_completed, before_successful = stream.completed_episodes, stream.successful_episodes
            start = time.perf_counter()
            cpu_start = time.process_time()
            model.learn(steps, callback=cap_callback, reset_num_timesteps=False, progress_bar=False)
            elapsed = time.perf_counter() - start
            metrics = {"steps": int(stream.total_steps - before_stream),
                       "global_timesteps": int(model.num_timesteps),
                       "optimizer_updates": update_counter["steps"] - before_actual_updates,
                       "ppo_epochs": int(model._n_updates - before_updates),
                       "process_cpu_s": time.process_time() - cpu_start,
                       "completed_episodes": stream.completed_episodes - before_completed,
                       "successful_episodes": stream.successful_episodes - before_successful,
                       "elapsed_s": elapsed, "schedule": phase_schedule,
                       "online_reward": float(stream.cumulative_return - before_reward)}
            if metrics["steps"] != steps or int(model.num_timesteps) - before_timesteps != steps:
                raise RuntimeError(f"phase {name} collected an unexpected number of interactions")
            metrics["capacity"] = cap_callback.metrics()
            metrics["checkpoint_bytes"] = _save_bundle(output / name / "agent.pt", core, optimizer, router,
                {"global_timesteps": int(model.num_timesteps), "phase": name, "method": method,
                 "actual_optimizer_steps": update_counter["steps"]}, _continual_state(model))
            return metrics

        training_records["seen"] = learn_phase("seen", schedules["seen"])
        seen_eval = _evaluate_many(core, seed, int(stage["eval_steps"]), output / "seen_eval", ("A0", "A1", "B0", "B1"))
        for context in base["heldout_sequence"]:
            route = context_index(context) + 1
            if method == "full_plain":
                continue
            initializer = getattr(core, "initialize_heldout", None)
            if callable(initializer):
                initializer(route)

        foundation_metrics = {}
        for context in base["heldout_sequence"]:
            eval_seed = int(seed) * 1000 + context_index(context)
            parent_route = 7 if context.startswith("A") else 8
            foundation_core, foundation_optimizer = _make_foundation_comparator(base, seed)
            foundation_metrics[context] = evaluate(
                foundation_core, context, eval_seed, int(stage["cold_eval_steps"]),
                output / "cold" / f"{context}_pristine_W0.json", route=0,
            )
            reuse_metrics = evaluate(
                core, context, eval_seed, int(stage["cold_eval_steps"]),
                output / "cold" / f"{context}_seen_parent_route.json", route=parent_route,
            )
            foundation_metrics[context]["comparator"] = "pristine external W0; route 0"
            foundation_metrics[context]["reuse"] = reuse_metrics
            foundation_metrics[context]["adapted_path"] = evaluate(
                core, context, eval_seed, int(stage["cold_eval_steps"]),
                output / "cold" / f"{context}_initialized_path.json", route=context_index(context) + 1,
            )
            del foundation_core, foundation_optimizer

        training_records["heldout_initialization"] = {
            "heldout_routes_initialized": [context_index(name) + 1 for name in base["heldout_sequence"]
                                           if method != "full_plain"],
            "seen_mask": _seen_mask(core),
            "seen_mask_bytes": _seen_mask_bytes(core),
        }
        training_records["heldout"] = learn_phase("heldout", schedules["heldout"])
        adapted_eval = _evaluate_many(core, seed, int(stage["eval_steps"]), output / "adapted_eval")
        training_records["revisit"] = learn_phase("revisit", schedules["revisit"])
        final_eval = _evaluate_many(core, seed, int(stage["eval_steps"]), output / "final_eval")

        pre_history_metadata = {
            "method": method, "control": control, "coefficient": float(coefficient),
            "seed": int(seed), "phase": "revisit", "global_timesteps": int(model.num_timesteps),
            "stream_steps": int(stream.total_steps), "seen_mask": _seen_mask(core),
            "actual_optimizer_steps": update_counter["steps"],
        }
        revisit_checkpoint_bytes = _save_bundle(
            output / "revisit" / "agent.pt", core, optimizer, router, pre_history_metadata,
            _continual_state(model),
        )
        # Restore all policy, named optimizer, route table, reservoir, EWC, and
        # RNG metadata from disk before hidden-history training.
        update_hook.remove()
        vector_env.close()
        cap_callback.model = cap_callback.core = cap_callback.optimizer = cap_callback.router = cap_callback.stream = None
        model = vector_env = stream = core = optimizer = router = None
        gc.collect()
        restored = _restore_payload(output / "revisit" / "agent.pt")
        core, optimizer = restored["policy"], restored["optimizer"]
        router = type(new_router(base)).from_snapshot(restored["router"])
        router.clear_history()

        history_stream = _SupplementStream(schedules["history"], seed + 317, router,
                                            mode="calibrated_history")
        _attach_trace(history_stream, output / "history", handles)
        history_env = DummyVecEnv([lambda: history_stream])
        history_config = {**base, **params, "seed": int(seed) + 317}
        model = _algorithm(core, optimizer, history_env, history_config, seed + 317,
                           control, coefficient, base["dataset"], initialize_source=False)
        update_hook = _count_optimizer_steps(optimizer, update_counter)
        history_stream.model = model
        model.num_timesteps = int(restored["metadata"].get("global_timesteps", 0))
        continual_state = restored.get("continual_state")
        if isinstance(model, C1ContinualPPO) and continual_state is not None:
            model.load_continual_state_dict(continual_state)
        history_cap_callback = _SupplementCapacityCallback(
            history_stream, core, optimizer, router, output / "history",
            agent_cap=complete_cap, adaptive_cap=adaptive_cap,
            outside_adaptive_budget=outside_adaptive,
        )
        history_start = time.perf_counter()
        history_before_updates = int(model._n_updates)
        history_before_actual_updates = update_counter["steps"]
        history_cpu_start = time.process_time()
        bank_hash = hashlib.sha256(router.direction_counts.tobytes() + router.outcome_counts.tobytes()).hexdigest()
        model.learn(sum(value for _, value in schedules["history"]), callback=history_cap_callback,
                    reset_num_timesteps=False, progress_bar=False)
        history_elapsed = time.perf_counter() - history_start
        if bank_hash != hashlib.sha256(router.direction_counts.tobytes() + router.outcome_counts.tobytes()).hexdigest():
            raise RuntimeError("Hidden phase changed calibration bank")
        history_metrics = {
            "online_steps": int(history_stream.total_steps),
            "online_cumulative_return": float(history_stream.cumulative_return),
            "online_elapsed_s": history_elapsed,
            "online_process_cpu_s": time.process_time() - history_cpu_start,
            "optimizer_updates": update_counter["steps"] - history_before_actual_updates,
            "ppo_epochs": int(model._n_updates - history_before_updates),
            "completed_episodes": history_stream.completed_episodes,
            "successful_episodes": history_stream.successful_episodes,
            "success_rate": history_stream.successful_episodes / max(1, history_stream.completed_episodes),
            "mode": "calibrated_history", "schedule": schedules["history"],
            "route_exact_accuracy": history_stream.exact_correct / max(1, history_stream.total_steps),
            "route_family_accuracy": history_stream.family_correct / max(1, history_stream.total_steps),
            "parent_fallback_fraction": history_stream.parent_count / max(1, history_stream.total_steps),
            "foundation_fallback_fraction": history_stream.root_count / max(1, history_stream.total_steps),
            "switch_first_correct": history_stream.switch_delays,
            "unknown_switches_not_given_to_router": True,
            "calibration_bank_frozen": True,
            "bank_calibration_interactions_added": 0,
            **history_cap_callback.metrics(),
        }
        history_eval = _evaluate_many(core, seed, int(stage["eval_steps"]), output / "history_eval")
        history_bundle_bytes = _save_bundle(
            output / "history" / "agent.pt", core, optimizer, router,
            {"method": method, "control": control, "seed": int(seed),
             "phase": "history", "global_timesteps": int(model.num_timesteps),
             "stream_steps": int(history_stream.total_steps), "seen_mask": _seen_mask(core)},
            _continual_state(model),
        )
        total_elapsed = time.perf_counter() - train_start
        total_cpu = time.process_time() - process_start
        result = {
            "status": "complete", "method": method, "control": control,
            "coefficient": float(coefficient), "seed": int(seed), "tier": tier_name,
            "params": dict(params), "stage": stage_name, "validation": bool(validation),
            "budget_classification": budget_label,
            "complete_agent_cap_bytes": complete_cap,
            "adaptive_parameter_optimizer_cap_bytes": adaptive_cap,
            "training": training_records, "seen_eval": seen_eval,
            "cold": foundation_metrics, "adapted_eval": adapted_eval,
            "final_eval": final_eval, "history": history_metrics,
            "history_eval": history_eval,
            "history_scope": "actual hidden-label continuation using frozen oracle-calibrated transition bank; separate from oracle evaluation",
            "source_fisher": (model.online_ewc.storage_report() if control == "ewc" and model.online_ewc else None),
            "continual_state": _continual_bytes(_continual_state(model)),
            "representation": _storage_snapshot(core, optimizer, router, model, model.rollout_buffer),
            "training_interactions": sum(row["steps"] for row in training_records.values()
                                          if isinstance(row, dict) and "steps" in row)
                                      + history_metrics["online_steps"],
            "evaluation_interactions": int(stage["eval_steps"]) * 22 + int(stage["cold_eval_steps"]) * 6,
            "revisit_checkpoint_bytes": revisit_checkpoint_bytes,
            "history_checkpoint_bytes": history_bundle_bytes,
            "training_elapsed_s": total_elapsed,
            "training_process_cpu_s": total_cpu,
            "online_learning_cpu_s": sum(row.get("process_cpu_s", 0) for row in training_records.values()) + history_metrics["online_process_cpu_s"],
            "online_learning_elapsed_s": sum(row.get("elapsed_s", 0) for row in training_records.values()) + history_elapsed,
            "initialization_cpu_s": initialization_cpu,
            "actual_optimizer_steps": update_counter["steps"],
            "base_actor_unchanged": actor_hash(core) == initial_actor_hash,
            "max_complete_agent_bytes": max(cap_callback.metrics()["max_complete_agent_bytes"], history_metrics["max_complete_agent_bytes"]),
            "gradient_update_scope": getattr(core, "parameter_update_scope", "residual_adapters_and_critic"),
            "foundation_actor_changed": (actor_hash(core) != _foundation_actor_digest(base, seed)) if _is_full(core) else False,
            "actor_hash_after_training": actor_hash(core),
            "seen_mask": _seen_mask(core),
            "seen_mask_bytes": _seen_mask_bytes(core),
            "capacity_accounting": {
                "includes": ["policy", "optimizer", "router", "continual replay/Fisher/reference", "rollout", "route embeddings/seen mask"],
                "full_actor_adaptive_cap_comparison": "outside original adaptive caps; Pareto reference",
                "reservoir_capacity": 128 if control == "bc" else 0,
                "bc128_array_bytes": 8704 if control == "bc" else 0,
                "ewc_source_observations": 64 if control == "ewc" else 0,
                "live_lower_bound_bytes": history_metrics["max_explicit_live_state_lower_bound_bytes"],
                "serialized_restore_verified": True,
            },
        }
        write_json(result_path, result)
        return result
    except Exception as exc:
        failure = {
            "status": "failed", "method": method, "control": control,
            "coefficient": float(coefficient), "seed": int(seed), "tier": tier_name,
            "params": dict(params), "stage": stage_name, "validation": bool(validation),
            "budget_classification": budget_label,
            "complete_agent_cap_bytes": complete_cap,
            "adaptive_parameter_optimizer_cap_bytes": adaptive_cap,
            "error_type": type(exc).__name__, "error": str(exc),
            "partial_capacity": (cap_callback.metrics() if "cap_callback" in locals() else None),
        }
        write_json(result_path, failure)
        return failure
    finally:
        for handle in handles:
            handle.close()
        if history_env is not None:
            history_env.close()
        if vector_env is not None:
            try:
                vector_env.close()
            except Exception:
                pass
        for value in (model, vector_env, stream, core, optimizer):
            if value is not None:
                del value
        gc.collect()


def _is_method_full(method: str) -> bool:
    return method in ("full", "full_plain")


def _seen_mask(core) -> list[bool] | None:
    mask = getattr(core, "seen_mask", None)
    if mask is None:
        mask = getattr(core, "known_endpoint_mask", None)
    return mask.detach().cpu().tolist() if torch.is_tensor(mask) else None


def _seen_mask_bytes(core) -> int:
    mask = getattr(core, "seen_mask", None)
    if mask is None:
        mask = getattr(core, "known_endpoint_mask", None)
    return int(mask.numel() * mask.element_size()) if torch.is_tensor(mask) else 0


def _make_foundation_comparator(config, seed):
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    torch_rng = torch.get_rng_state()
    reference = _foundation(config, seed)
    try:
        core = C1FullFineTunePolicy.from_reference(
            reference.policy, route_conditioned=True,
            epsilon=float(config["exploration_epsilon"]),
            temperature=float(config.get("temperature", 2.0)),
        )
        # This comparator has no optimizer and is never trained.  Route zero
        # is the pristine shared W0 policy, independent of the trained arm.
        return core, None
    finally:
        reference.env.close()
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)


def _foundation_actor_digest(config, seed):
    core, _ = _make_foundation_comparator(config, seed)
    digest = actor_hash(core)
    del core
    return digest


def _evaluate_many(core, seed, steps, directory, names=("A0", "A1", "A2", "B0", "B1", "B2")):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        name: evaluate(core, name, int(seed) * 1000 + context_index(name), int(steps), directory / f"{name}.json")
        for name in names
    }


def _candidate_result_path(root: Path, method: str, control: str, param_index: int, seed: int) -> Path:
    candidate = f"candidate_{param_index:02d}"
    if control != "none":
        candidate = f"{candidate}_{control}"
    return root / "validation" / method / candidate / f"s{seed}"


def _select(records: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    candidates = []
    for row in records:
        candidate = row["candidate"]
        result = row["result"]
        score = _score_result(result)
        candidates.append({"candidate": candidate, "params": row["params"], "score": score,
                           "status": result.get("status"), "result_path": row["result_path"]})
    complete = [row for row in candidates if row["status"] == "complete" and math.isfinite(row["score"])]
    selected = max(complete, key=lambda row: (row["score"], -row["candidate"])) if complete else None
    return {
        "selection_formula": config["validation_selection"],
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_params": None if selected is None else selected["params"],
        "selected_score": None if selected is None else selected["score"],
        "candidates": candidates,
    }


def _run_validations(root, supplement, base, method, tier_name, tier, params_grid,
                     *, control="none", coefficients=None, config_path="configs/c1_supplement.json"):
    seed = int(supplement["validation_seeds"][0])
    rows = []
    for index, params in enumerate(params_grid):
        coefficient = 0.0 if coefficients is None else float(coefficients[index])
        path = _candidate_result_path(root, method, control, index, seed)
        result = run_arm(
            supplement=supplement, base=base, output=path, seed=seed,
            method=method, tier_name=tier_name, tier=tier, params=params,
            stage_name=root.name, control=control, coefficient=coefficient,
            validation=True, config_path=config_path,
        )
        rows.append({"candidate": index, "params": dict(params), "coefficient": coefficient,
                     "result": result, "result_path": str(path / "result.json")})
        print(json.dumps({"suite": "validation", "method": method, "control": control, "candidate": index, "status": result["status"], "score": _score_result(result), "error": result.get("error")}), flush=True)
        write_json(root / "validation_progress.json", rows)
    selection = _select(rows, supplement)
    if coefficients is not None and selection["selected_candidate"] is not None:
        selection["selected_coefficient"] = float(coefficients[selection["selected_candidate"]])
    write_json(root / "selection.json", selection)
    return selection


def _run_formal(root, supplement, base, method, tier_name, tier, params, *, control="none",
                coefficient=0.0, config_path="configs/c1_supplement.json"):
    results = []
    for seed in supplement["formal_seeds"]:
        path = root / "formal" / method / control / tier_name / f"s{seed}"
        result = run_arm(
            supplement=supplement, base=base, output=path, seed=int(seed), method=method,
            tier_name=tier_name, tier=tier, params=params, stage_name=root.name,
            control=control, coefficient=coefficient, validation=False,
            config_path=config_path,
        )
        results.append(result)
        print(json.dumps({"suite": "formal", "method": method, "control": control, "tier": tier_name, "seed": seed, "status": result["status"], "score": _score_result(result), "error": result.get("error")}), flush=True)
        write_json(root / f"formal_progress_{tier_name}.json", results)
    return results


def run_step2(supplement, base, output, *, config_path="configs/c1_supplement.json"):
    """Tune tree/grouped-dictionary, then run both frozen capacities."""
    root = Path(output) / "step2_grouping"
    root.mkdir(parents=True, exist_ok=True)
    selections = {}
    methods = ("tree", "grouped_dictionary")
    for method in methods:
        selection = _run_validations(
            root / method, supplement, base, method, "high", base["tiers"]["high"],
            supplement["compressed_candidates"], config_path=config_path,
        )
        selections[method] = selection
        if selection["selected_params"] is not None:
            for tier_name in ("low", "high"):
                _run_formal(root / method, supplement, base, method, tier_name,
                            base["tiers"][tier_name], selection["selected_params"],
                            config_path=config_path)
    frozen = {"stage": "step2_grouping", "selections": selections,
              "formal_seeds": supplement["formal_seeds"], "config_sha256": _source_hashes(config_path, supplement, base)}
    write_json(root / "frozen_settings.json", frozen)
    return frozen


def run_step3(supplement, base, output, *, config_path="configs/c1_supplement.json"):
    """Tune compressed policies and evaluate full, EWC, and bounded replay controls."""
    root = Path(output) / "step3_tuning_and_continual"
    root.mkdir(parents=True, exist_ok=True)
    selected: dict[str, Any] = {}
    validation_seed = int(supplement["validation_seeds"][0])

    for method in _STEP3_COMPRESSED:
        selection = _run_validations(
            root / method, supplement, base, method, "high", base["tiers"]["high"],
            supplement["compressed_candidates"], config_path=config_path,
        )
        selected[method] = selection
        if selection["selected_params"] is not None:
            for tier_name in ("low", "high"):
                _run_formal(root / method, supplement, base, method, tier_name,
                            base["tiers"][tier_name], selection["selected_params"],
                            config_path=config_path)

    full_selection = _run_validations(
        root / "full", supplement, base, "full", "high", base["tiers"]["high"],
        supplement["full_candidates"], config_path=config_path,
    )
    selected["full"] = full_selection
    full_params = full_selection["selected_params"]
    if full_params is not None:
        selected["full_formal"] = _run_formal(
            root / "full", supplement, base, "full", "high", base["tiers"]["high"],
            full_params, config_path=config_path,
        )
        for control, key in (("ewc", "ewc_coefficients"), ("bc", "bc_coefficients")):
            coefficients = [float(value) for value in supplement[key]]
            selection = _run_validations(
                root / f"full_{control}", supplement, base, "full", "high", base["tiers"]["high"],
                [full_params] * len(coefficients), control=control, coefficients=coefficients,
                config_path=config_path,
            )
            selected[f"full_{control}"] = selection
            if selection.get("selected_coefficient") is not None:
                selected[f"full_{control}_formal"] = _run_formal(
                    root / f"full_{control}", supplement, base, "full", "high", base["tiers"]["high"],
                    full_params, control=control, coefficient=selection["selected_coefficient"],
                    config_path=config_path,
                )

    tree_selection_path = Path(output) / "step2_grouping" / "tree" / "selection.json"
    if tree_selection_path.exists():
        tree_selection = _load_json(tree_selection_path)
        tree_params = tree_selection.get("selected_params")
    else:
        tree_params = None
    if tree_params is not None:
        coefficients = [float(value) for value in supplement["bc_coefficients"]]
        tree_bc_selection = _run_validations(
            root / "tree_bc", supplement, base, "tree", "high", base["tiers"]["high"],
            [tree_params] * len(coefficients), control="bc", coefficients=coefficients,
            config_path=config_path,
        )
        selected["tree_bc"] = tree_bc_selection
        if tree_bc_selection.get("selected_coefficient") is not None:
            for tier_name in ("low", "high"):
                selected[f"tree_bc_formal_{tier_name}"] = _run_formal(
                    root / "tree_bc", supplement, base, "tree", tier_name,
                    base["tiers"][tier_name], tree_params, control="bc",
                    coefficient=tree_bc_selection["selected_coefficient"], config_path=config_path,
                )

    frozen = {
        "stage": "step3_tuning_and_continual", "selections": selected,
        "formal_seeds": supplement["formal_seeds"],
        "validation_seed": validation_seed,
        "source_sha256": _source_hashes(config_path, supplement, base),
        "continual_labels": {"bc": "bounded PPO + behavior cloning; CLEAR-inspired, not CLEAR reproduction",
                             "ewc": "task-free diagonal empirical-Fisher EWC for PPO"},
    }
    write_json(root / "frozen_settings.json", frozen)
    return frozen


def run_stage(stage: str, config_path="configs/c1_supplement.json"):
    if stage not in ("step2", "step3"):
        raise ValueError("stage must be 'step2' or 'step3'; step 1 is owned by the model-control stage")
    supplement, base = load_protocol(config_path)
    output = Path(supplement["output"])
    if stage == "step2":
        return run_step2(supplement, base, output, config_path=config_path)
    return run_step3(supplement, base, output, config_path=config_path)


def _is_method_full(method: str) -> bool:
    return method in ("full", "full_plain")


__all__ = [
    "C1SeenAwareResidualPolicy", "run_arm", "run_stage", "run_step2", "run_step3",
    "load_protocol", "_make_core", "_restore_payload", "_sequence_schedule", "_score_result",
]

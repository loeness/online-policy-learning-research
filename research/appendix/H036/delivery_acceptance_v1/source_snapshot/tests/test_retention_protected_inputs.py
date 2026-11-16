"""Synthetic-only checks for the H036 frozen input adapter."""
from __future__ import annotations

import copy
import hashlib
import io
import random
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl import retention_behavioral_candidate as behavioral  # noqa: E402
from online_rl import retention_protected_candidate as protected  # noqa: E402
from online_rl import retention_protected_inputs as inputs  # noqa: E402
from online_rl.storage_decision import pair_mapping, task_name  # noqa: E402
from online_rl.write_decision_policy import decode_actor, encode_actor  # noqa: E402


CONFIG = inputs.validate_config()
SEED = CONFIG["seeds"][0]
FOUNDATION_SHA256 = CONFIG["foundation"]["sha256"]


def _random_obs(rng: np.random.Generator, count: int, route: int) -> np.ndarray:
    result = np.zeros((count, 13), dtype=np.float32)
    result[:, 0] = rng.uniform(0.01, 0.99, size=count).astype(np.float32)
    result[:, 1:4] = rng.normal(0.0, 0.35, size=(count, 3)).astype(np.float32)
    result[:, -1] = np.float32(route)
    return result


def synthetic_payload(seed: int = SEED):
    """Build a complete, deterministic checkpoint and two-source sidecar.

    The fixture intentionally leaves only a short new-group PPO FIFO, so the
    adapter must bind new sources from the sidecar and current cache only.
    """
    rng = np.random.default_rng(9041 + int(seed))
    state: dict[str, torch.Tensor] = {
        "mlp_extractor.policy_net.0.weight": torch.tensor(rng.normal(0, 0.08, (64, 4)).astype(np.float32)),
        "mlp_extractor.policy_net.0.bias": torch.tensor(rng.normal(0, 0.03, 64).astype(np.float32)),
        "mlp_extractor.policy_net.2.weight": torch.tensor(rng.normal(0, 0.05, (64, 64)).astype(np.float32)),
        "mlp_extractor.policy_net.2.bias": torch.tensor(rng.normal(0, 0.03, 64).astype(np.float32)),
        "action_net.weight": torch.tensor(rng.normal(0, 0.07, (4, 64)).astype(np.float32)),
        "action_net.bias": torch.tensor(rng.normal(0, 0.03, 4).astype(np.float32)),
        "source_basis": torch.tensor(rng.normal(0, 0.025, (3, 65)).astype(np.float32)),
    }
    local = rng.normal(0.0, 0.09, size=(2, 27)).astype(np.float32)
    shared = rng.normal(0.0, 0.07, size=(1, 9)).astype(np.float32)
    physical_groups = np.asarray([-1, -1], dtype=np.int8)
    actor = encode_actor(local, shared, physical_groups, codec="int8", budget_bytes=89).data
    decoded_local, decoded_shared, decoded_groups = decode_actor(actor)
    state["_nodes.1.A"] = torch.tensor(decoded_local[0].reshape(3, 3, 3).copy())
    state["_nodes.2.A"] = torch.tensor(decoded_local[1].reshape(3, 3, 3).copy())
    state["_shared_coefficients.0.C"] = torch.tensor(decoded_shared[0].reshape(3, 3).copy())
    state["slot_groups"] = torch.tensor(decoded_groups.astype(np.int64))

    pair = pair_mapping(seed, "dynamics")
    old_tasks = [task_name(pair, 0, variant) for variant in (0, 1, 2)]
    new_tasks = [task_name(pair, 1, variant) for variant in (0, 1, 2)]
    old_count = 144
    old_obs = _random_obs(rng, old_count, route=1)
    old_bouts = np.repeat(np.arange(4, 16, dtype=np.int32), 12)
    old_task = np.asarray([old_tasks[index % 3] for index in range(old_count)], dtype="U4")
    old_probs = np.full((old_count, 4), 0.25, dtype=np.float32)

    new_obs = np.concatenate((_random_obs(rng, 128, route=2), _random_obs(rng, 128, route=1)))
    owners = np.repeat(np.asarray([0, 1], dtype=np.int8), 128)
    owner_slots = np.where(owners == 0, 1, 0).astype(np.int8)
    sidecar = {
        "observations": new_obs,
        "owner_indices": owners,
        "source_slots": owner_slots,
        "generations": np.where(owner_slots == 0, 5, 9).astype(np.int64),
        "bouts": np.concatenate((np.full(128, 20, dtype=np.int64), np.full(128, 21, dtype=np.int64))),
        "global_steps": np.concatenate((np.arange(100, 228, dtype=np.int64), np.arange(300, 428, dtype=np.int64))),
    }
    # Only a few recent new-group PPO rows remain. Source identity comes from
    # the sidecar plus cache mapping, not from requiring 128 group-1 PPO rows.
    group1_obs = new_obs[-5:].copy()
    group1_task = np.asarray([new_tasks[1]] * len(group1_obs), dtype="U4")
    group1_bout = np.full(len(group1_obs), 21, dtype=np.int32)
    evidence = {
        "capacity": 256,
        "groups": {
            0: {
                "obs": old_obs,
                "action": np.zeros(old_count, dtype=np.int64),
                "behavior_probs": old_probs,
                "behavior_log_prob": np.full(old_count, -np.log(4), dtype=np.float32),
                "advantage": np.zeros(old_count, dtype=np.float32),
                "returns": np.zeros(old_count, dtype=np.float32),
                "episode_id": np.arange(old_count, dtype=np.int64),
                "task": old_task,
                "bout": old_bouts,
            },
            1: {
                "obs": group1_obs,
                "action": np.zeros(len(group1_obs), dtype=np.int64),
                "behavior_probs": np.full((len(group1_obs), 4), 0.25, dtype=np.float32),
                "behavior_log_prob": np.full(len(group1_obs), -np.log(4), dtype=np.float32),
                "advantage": np.zeros(len(group1_obs), dtype=np.float32),
                "returns": np.zeros(len(group1_obs), dtype=np.float32),
                "episode_id": np.arange(len(group1_obs), dtype=np.int64),
                "task": group1_task,
                "bout": group1_bout,
            },
        },
    }
    cache_tasks = [new_tasks[2], new_tasks[1]]
    cache = {
        "capacity": 2,
        "eviction": "lru",
        "task_to_slot": {task: slot for slot, task in enumerate(cache_tasks)},
        "slot_to_task": cache_tasks.copy(),
        "last_used": np.asarray([21, 22], dtype=np.int64),
        "frequency": np.asarray([80, 81], dtype=np.uint64),
        "clock": 22,
        "rng_state": np.random.default_rng(seed + 5).bit_generator.state,
        "nbytes": 32,
    }
    checkpoint = {
        "format": "storage_decision_learner_v1",
        "seed": int(seed),
        "foundation_sha256": FOUNDATION_SHA256,
        "core_state": state,
        "optimizer_state": {
            "state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.ones(2)}},
            "param_groups": [{"lr": 0.003, "params": [0]}],
        },
        "optimizer_step_count": 3,
        "shared_frozen": True,
        "counts": {"updates": 3},
        "runtime_counts": {"environment_steps": 1200},
        "slot_generation": np.asarray([5, 9], dtype=np.int64),
        "fitted_steps": np.asarray([128, 256], dtype=np.uint64),
        "cache": cache,
        "evidence": evidence,
        "recent_groups": [0, 1, 0, 1, 1, 0, 1, 1],
        "parent_group": 0,
        "slot_group_tags": np.asarray([1, 1], dtype=np.int8),
        "action_rng": np.random.default_rng(seed + 1).bit_generator.state,
        "behavior_rng": np.random.default_rng(seed + 2).bit_generator.state,
        "torch_rng": torch.get_rng_state().clone(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    return checkpoint, actor, sidecar


def _prepare(checkpoint, actor, sidecar, *, seed=SEED):
    return inputs.prepare_from_payload(
        checkpoint,
        actor,
        sidecar,
        foundation_bytes=CONFIG["foundation"]["bytes"],
        foundation_sha256=FOUNDATION_SHA256,
        input_identity={"logical_case": f"s{seed}_dynamics_late", "kind": "synthetic"},
    )


def _write_prefix(path: Path, checkpoint, actor: bytes, sidecar) -> dict[str, str]:
    path.mkdir(parents=True)
    checkpoint_path = path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    actor_path = path / "actor.bin"
    actor_path.write_bytes(actor)
    evidence_path = path / "evidence.npz"
    with evidence_path.open("wb") as handle:
        np.savez_compressed(handle, **sidecar)
    return {
        "checkpoint.pt": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "actor.bin": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
        "evidence.npz": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    }


def test_prepare_returns_frozen_96_32_splits_and_complete_prefix_copy():
    checkpoint, actor, sidecar = synthetic_payload()
    before_core = {key: value.clone() for key, value in checkpoint["core_state"].items()}
    before_obs = {key: value.copy() for key, value in sidecar.items()}
    prepared = _prepare(checkpoint, actor, sidecar)

    assert prepared["format"] == "h036_prepared_v1"
    assert prepared["seed"] == SEED
    assert prepared["new_observations"].shape == (256, 13)
    assert prepared["new_observations"].dtype == np.float32
    assert prepared["new_owners"].dtype == np.int64
    assert np.array_equal(prepared["new_owners"], np.repeat([0, 1], 128))
    assert prepared["new_train_indices"].tolist() == list(range(96)) + list(range(128, 224))
    assert prepared["new_holdout_indices"].tolist() == list(range(96, 128)) + list(range(224, 256))
    assert prepared["old_train_indices"].tolist() == list(range(96))
    assert prepared["old_holdout_indices"].tolist() == list(range(96, 128))
    assert prepared["old_observations"].shape == (128, 13)
    assert np.array_equal(prepared["old_observations"], checkpoint["evidence"]["groups"][0]["obs"][-128:])
    assert prepared["new_effective"].shape == (2, 3, 3, 3)
    assert prepared["mean64"].dtype == np.float64
    assert np.array_equal(prepared["mean64"], prepared["new_effective"].mean(axis=(0, 1), dtype=np.float64))
    assert len(prepared["prefix_actor_bytes"]) == 89
    assert prepared["prefix_evidence_bytes"]
    assert prepared["prefix_checkpoint"] is not checkpoint
    assert "optimizer_state" in prepared["prefix_checkpoint"]
    assert prepared["owner_tasks"] == (checkpoint["cache"]["slot_to_task"][1], checkpoint["cache"]["slot_to_task"][0])
    for name, value in before_core.items():
        assert torch.equal(checkpoint["core_state"][name], value)
    for name, value in before_obs.items():
        assert np.array_equal(sidecar[name], value)
    assert not prepared["new_observations"].flags.writeable
    assert not prepared["new_teacher"].flags.writeable
    assert not prepared["new_features"].base_logits.flags.writeable


def test_teacher_uses_each_owners_slot_and_observation_band_with_equal_mean():
    checkpoint, actor, sidecar = synthetic_payload()
    prepared = _prepare(checkpoint, actor, sidecar)
    bands = behavioral._bands_from_observations(prepared["new_observations"])
    expected = behavioral._teacher_probabilities(
        prepared["new_features"],
        prepared["new_effective"],
        prepared["new_owners"],
        bands,
    )
    assert np.allclose(prepared["new_teacher"], expected, rtol=0.0, atol=0.0)
    assert np.allclose(prepared["new_teacher"].sum(axis=1), 1.0)
    assert np.allclose(prepared["old_teacher"].sum(axis=1), 1.0)
    assert np.array_equal(prepared["owner_slots"], [1, 0])
    assert np.array_equal(prepared["physical_groups"], [-1, -1])
    assert np.array_equal(prepared["logical_tags"], [1, 1])


def test_old_keep_teacher_is_route_neutral_and_excludes_resident_local_matrices():
    checkpoint, actor, sidecar = synthetic_payload()
    baseline = _prepare(checkpoint, actor, sidecar)

    route_changed = copy.deepcopy(checkpoint)
    route_changed["evidence"]["groups"][0]["obs"][-128:, -1] = np.float32(99.0)
    routed = _prepare(route_changed, actor, sidecar)
    assert np.array_equal(baseline["old_teacher"], routed["old_teacher"])

    changed_local = copy.deepcopy(checkpoint)
    changed_local["core_state"]["_nodes.2.A"][0, 0, 0] += 3.0
    local = np.stack([
        changed_local["core_state"][f"_nodes.{slot + 1}.A"].numpy().reshape(27)
        for slot in range(2)
    ])
    shared = changed_local["core_state"]["_shared_coefficients.0.C"].numpy().reshape(1, 9)
    actor_local = encode_actor(local, shared, np.asarray([-1, -1], dtype=np.int8), codec="int8", budget_bytes=89).data
    decoded_local, decoded_shared, _ = decode_actor(actor_local)
    changed_local["core_state"]["_nodes.1.A"] = torch.tensor(decoded_local[0].reshape(3, 3, 3).copy())
    changed_local["core_state"]["_nodes.2.A"] = torch.tensor(decoded_local[1].reshape(3, 3, 3).copy())
    changed_local["core_state"]["_shared_coefficients.0.C"] = torch.tensor(decoded_shared[0].reshape(3, 3).copy())
    local_changed = _prepare(changed_local, actor_local, sidecar)
    assert np.array_equal(baseline["old_teacher"], local_changed["old_teacher"])
    assert not np.array_equal(baseline["new_teacher"], local_changed["new_teacher"])


def test_prepare_rejects_bad_generation_and_source_order():
    checkpoint, actor, sidecar = synthetic_payload()
    bad_generation = {name: value.copy() for name, value in sidecar.items()}
    bad_generation["generations"][0] += 1
    with pytest.raises(ValueError, match="generation"):
        _prepare(checkpoint, actor, bad_generation)

    bad_steps = {name: value.copy() for name, value in sidecar.items()}
    bad_steps["global_steps"][1] = bad_steps["global_steps"][0]
    with pytest.raises(ValueError, match="global_steps"):
        _prepare(checkpoint, actor, bad_steps)

    bad_bouts = {name: value.copy() for name, value in sidecar.items()}
    bad_bouts["bouts"][129] = bad_bouts["bouts"][128] - 1
    with pytest.raises(ValueError, match="bouts"):
        _prepare(checkpoint, actor, bad_bouts)

    bad_old_order = copy.deepcopy(checkpoint)
    bad_old_order["evidence"]["groups"][0]["bout"][-2] = 0
    with pytest.raises(ValueError, match="nondecreasing"):
        _prepare(bad_old_order, actor, sidecar)


def test_prepare_rejects_short_old_fifo_seed_mismatch_and_non_int8_actor(tmp_path: Path):
    checkpoint, actor, sidecar = synthetic_payload()
    short_old = copy.deepcopy(checkpoint)
    for name, values in short_old["evidence"]["groups"][0].items():
        short_old["evidence"]["groups"][0][name] = values[:127].copy()
    with pytest.raises(ValueError, match="128 to capacity"):
        _prepare(short_old, actor, sidecar)

    float_actor = encode_actor(
        np.stack([checkpoint["core_state"][f"_nodes.{slot + 1}.A"].numpy().reshape(27) for slot in range(2)]),
        checkpoint["core_state"]["_shared_coefficients.0.C"].numpy().reshape(1, 9),
        np.asarray([-1, -1], dtype=np.int8),
        codec="float32",
        budget_bytes=1024,
    ).data
    with pytest.raises(ValueError, match="89 bytes|int8"):
        _prepare(checkpoint, float_actor, sidecar)

    prefix = tmp_path / "one_prefix"
    _write_prefix(prefix, checkpoint, actor, sidecar)
    with pytest.raises(ValueError, match="expected_seed"):
        inputs.prepare_inputs(prefix, config=CONFIG, expected_seed=CONFIG["seeds"][1])


def test_prepare_inputs_reads_only_three_files_and_identity_survives_relocation(tmp_path: Path):
    checkpoint, actor, sidecar = synthetic_payload()
    first = tmp_path / "first" / "prefix"
    second = tmp_path / "relocated" / "prefix"
    hashes = _write_prefix(first, checkpoint, actor, sidecar)
    _write_prefix(second, checkpoint, actor, sidecar)
    for root in (first, second):
        (root / "observer_prefix.json").write_text("not an input", encoding="utf-8")
        (root / "summary.json").write_text("not an input", encoding="utf-8")
        (root / "simulator.json").write_text("not an input", encoding="utf-8")
        (root / "future_rewards.json").write_text("not an input", encoding="utf-8")

    left = inputs.prepare_inputs(
        first,
        config=CONFIG,
        expected_seed=SEED,
        expected_identity=hashes,
        logical_case=f"s{SEED}_dynamics_late",
    )
    right = inputs.prepare_inputs(
        second,
        config=CONFIG,
        expected_seed=SEED,
        expected_identity=hashes,
    )
    assert left["prepared_id"] == right["prepared_id"]
    assert protected._prepared_hash(left) == protected._prepared_hash(right)
    assert left["input_identity"]["prefix_path"] != right["input_identity"]["prefix_path"]
    assert left["input_identity"]["logical_case"] == f"s{SEED}_dynamics_late"
    assert left["memory_cpu_wall"]["evidence_npz_bytes"] == len(left["prefix_evidence_bytes"])
    assert left["prefix_checkpoint_serialized_bytes"] > 0
    assert all((root / name).exists() for root in (first, second) for name in (
        "observer_prefix.json", "summary.json", "simulator.json", "future_rewards.json"
    ))

    wrong_hashes = dict(hashes)
    wrong_hashes["actor.bin"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        inputs.prepare_inputs(first, config=CONFIG, expected_identity=wrong_hashes)
    with pytest.raises(ValueError, match="expected_identity must map"):
        inputs.prepare_inputs(first, config=CONFIG)
    with pytest.raises(ValueError, match="logical_case must match"):
        inputs.prepare_inputs(first, config=CONFIG, expected_identity=hashes, logical_case="wrong_case")


def test_config_rejects_unregistered_seed_list():
    bad_config = copy.deepcopy(CONFIG)
    bad_config["seeds"][0] += 1
    with pytest.raises(ValueError, match="seeds"):
        inputs.validate_config(bad_config)

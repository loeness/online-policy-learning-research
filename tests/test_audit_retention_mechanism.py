from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.retention_stream_value import apply_retention_action
from online_rl.storage_decision import effective_matrices, pair_mapping, semantic_hash
from scripts.audit_retention_mechanism import (
    Audit,
    _check_episode_ledger,
    _check_rng_transform,
    expected_plan,
    future_schedule,
    future_seed,
    horizon_totals,
    prefix_schedule,
)


CONFIG = {
    "bootstrap_steps": 1024,
    "bout_steps": 512,
    "candidate_source_steps": 1024,
    "future_blocks": 8,
    "rollout_steps": 256,
    "future_seed_offset": 40_000_000,
    "future_seed_multiplier": 1000,
    "horizons": [1536, 6144, 12288],
}


def test_frozen_main_and_pilot_budgets_reconcile() -> None:
    main = expected_plan(
        "main", seeds=[25101, 25102, 25103, 25104, 25107, 25108],
        pair_kinds=["dynamics", "reward_conflict"], histories=["old_major", "new_major"],
        futures=["old_major", "new_major"], replicates=[0, 1],
    )
    pilot = expected_plan(
        "pilot", seeds=[25089], pair_kinds=["dynamics"], histories=["old_major"],
        futures=["old_major"], replicates=[0, 1], future_blocks=4,
    )
    assert (main["case_count"], main["branch_count"], main["environment_steps_total"],
            main["ppo_rollouts_total"]) == (24, 384, 4_890_624, 19_104)
    assert (pilot["case_count"], pilot["branch_count"], pilot["environment_steps_total"],
            pilot["ppo_rollouts_total"]) == (1, 8, 56_320, 220)


def test_prefix_tail_candidate_and_fixed_future_schedule() -> None:
    mapping = pair_mapping(25101, "dynamics")
    prefix = prefix_schedule(mapping, CONFIG, "old_major")
    assert len(prefix) == 10
    assert sum(row["steps"] for row in prefix) == 7168
    assert [(row["group"], row["variant"], row["steps"]) for row in prefix[-2:]] == [
        (1, 0, 1024), (1, 1, 1024),
    ]
    future = future_schedule(mapping, "new_major", CONFIG)
    assert len(future) == 24
    assert sum(row["steps"] for row in future) == 12_288
    for block in range(8):
        group = [row["group"] for row in future[block * 3:block * 3 + 3]]
        variant = [row["variant"] for row in future[block * 3:block * 3 + 3]]
        assert group == [0, 1, 1]
        assert variant == [block % 3, block % 3, (block + 1) % 3]
    assert future_seed(25101, 0, CONFIG) != future_seed(25101, 1, CONFIG)
    assert future_seed(25101, 0, CONFIG) == future_seed(25101, 0, CONFIG)


def test_nested_horizons_use_one_raw_branch_ledger() -> None:
    rows = [{"reward": 0.25} for _ in range(12_288)]
    totals = horizon_totals(rows, CONFIG["horizons"])
    assert totals["1536"]["reward_total"] == 384.0
    assert totals["6144"]["reward_total"] == 1536.0
    assert totals["12288"]["reward_total"] == 3072.0
    assert sum(rows[index]["reward"] for index in range(len(rows))) == totals["12288"]["reward_total"]
    assert sum(int(value["steps"]) for value in totals.values()) > len(rows)


def test_generic_retention_actions_preserve_current_resident_effective_matrices() -> None:
    local = np.linspace(-0.25, 0.25, 54, dtype=np.float32).reshape(2, 27)
    shared = np.linspace(-0.1, 0.1, 9, dtype=np.float32).reshape(1, 9)
    physical = np.asarray([0, -1], dtype=np.int8)
    logical = np.asarray([0, 1], dtype=np.int8)
    candidate = np.full((3, 3), 0.125, dtype=np.float32)
    before = effective_matrices(local, shared, physical)
    bindings = {"keep": 0, "replace": 1, "mean": -2, "none": -1}
    for action, expected_binding in bindings.items():
        new_local, new_shared, new_physical, details = apply_retention_action(
            local, shared, physical, logical, 0, candidate, 1, action,
        )
        assert details["parent_group"] == expected_binding
        assert np.allclose(effective_matrices(new_local, new_shared, new_physical), before, atol=1e-7)
        assert np.array_equal(new_shared[0].reshape(3, 3), shared[0].reshape(3, 3)) if action == "keep" else True


def test_rng_transform_preserves_current_episode_and_reproduces_hashes() -> None:
    original_seed = 91
    replicate = 1
    expected_seed = future_seed(original_seed, replicate, CONFIG)
    checkpoint = {
        "seed": original_seed,
        "action_rng": np.random.default_rng(original_seed + 1).bit_generator.state,
        "behavior_rng": np.random.default_rng(original_seed + 2).bit_generator.state,
        "numpy_rng": np.random.RandomState(original_seed).get_state(),
        "python_rng": random.Random(original_seed).getstate(),
        "torch_rng": torch.Generator(device="cpu").manual_seed(original_seed).get_state(),
    }
    simulator = {
        "environment": {
            "np_random_bit_generator": "PCG64",
            "np_random_state": {"bit_generator": "PCG64", "state": {"state": 17, "inc": 3}},
        },
        "driver": {"obs": [0.5], "episode_ordinal": 2, "episode_length": 9,
                   "episode_reset_seed": 123456},
    }
    env_hash = semantic_hash({"bit_generator": "PCG64", "state": simulator["environment"]["np_random_state"]})
    before = {
        "agent_seed": original_seed,
        "action_rng_sha256": semantic_hash(checkpoint["action_rng"]),
        "behavior_rng_sha256": semantic_hash(checkpoint["behavior_rng"]),
        "numpy_global_rng_sha256": semantic_hash(checkpoint["numpy_rng"]),
        "python_global_rng_sha256": semantic_hash(checkpoint["python_rng"]),
        "torch_global_rng_sha256": semantic_hash(checkpoint["torch_rng"]),
        "environment_rng_sha256": env_hash,
        "episode_reset_seed": 123456,
        "episode_ordinal": 2,
        "episode_length": 9,
    }
    after = {
        "agent_seed": expected_seed,
        "action_rng_sha256": semantic_hash(np.random.default_rng(expected_seed + 0xA17).bit_generator.state),
        "behavior_rng_sha256": semantic_hash(np.random.default_rng(expected_seed + 0xB3E).bit_generator.state),
        "numpy_global_rng_sha256": semantic_hash(np.random.RandomState(expected_seed).get_state()),
        "python_global_rng_sha256": semantic_hash(random.Random(expected_seed).getstate()),
        "torch_global_rng_sha256": semantic_hash(torch.Generator(device="cpu").manual_seed(expected_seed).get_state()),
        "environment_rng_sha256": env_hash,
        "episode_reset_seed": 123456,
        "episode_ordinal": 2,
        "episode_length": 9,
    }
    transformed = dict(checkpoint)
    transformed.update({
        "seed": expected_seed,
        "action_rng": np.random.default_rng(expected_seed + 0xA17).bit_generator.state,
        "behavior_rng": np.random.default_rng(expected_seed + 0xB3E).bit_generator.state,
        "numpy_rng": np.random.RandomState(expected_seed).get_state(),
        "python_rng": random.Random(expected_seed).getstate(),
        "torch_rng": torch.Generator(device="cpu").manual_seed(expected_seed).get_state(),
    })
    record = {
        "seed": original_seed,
        "replicate": replicate,
        "future_seed": expected_seed,
        "post_rng_transform_state_hash": semantic_hash({"learner": transformed, "simulator": simulator}),
        "rng_transform": {
            "before": before,
            "after": after,
            "environment_rng_preserved": True,
            "current_episode_preserved": True,
            "reseed": {
                "future_seed": expected_seed,
                "action_rng_seed": expected_seed + 0xA17,
                "action_rng_stream": 0xA17,
                "behavior_rng_seed": expected_seed + 0xB3E,
                "behavior_rng_stream": 0xB3E,
                "numpy_global_seed": expected_seed,
                "python_global_seed": expected_seed,
                "torch_global_seed": expected_seed,
                "environment_rng_action": "preserved until a normal episode reset",
            },
        },
    }
    audit = Audit(Path("synthetic"))
    assert _check_rng_transform(audit, record, checkpoint, simulator, "synthetic", CONFIG) == expected_seed
    assert audit.errors == []


def test_episode_ledger_checks_unreset_fork_and_normal_reset(tmp_path: Path) -> None:
    future = 40_091_001
    initial_seed = 123456
    reset_after_completion = future * 1_000_003 + 4 * 7919 + 0x5E7
    rows = [
        {"episode_id": "3", "reset_seed": str(initial_seed), "reward": "1.0",
         "environment_done": "0", "terminated": "0", "truncated": "0",
         "obs": "[1.0]", "next_obs": "[2.0]", "action": "keep", "future_regime": "old_major"},
        {"episode_id": "3", "reset_seed": str(initial_seed), "reward": "2.0",
         "environment_done": "1", "terminated": "1", "truncated": "0",
         "obs": "[2.0]", "next_obs": "[3.0]", "action": "keep", "future_regime": "old_major"},
        {"episode_id": "4", "reset_seed": str(reset_after_completion), "reward": "0.5",
         "environment_done": "0", "terminated": "0", "truncated": "0",
         "obs": "[9.0]", "next_obs": "[10.0]", "action": "keep", "future_regime": "old_major"},
    ]
    with (tmp_path / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "episode_id", "reset_seed", "next_reset_seed", "length", "return_value",
            "terminated", "truncated", "action", "future_regime",
        ])
        writer.writeheader()
        writer.writerow({
            "episode_id": 3, "reset_seed": initial_seed,
            "next_reset_seed": reset_after_completion,
            "length": 3, "return_value": 100.0,
            "terminated": 1, "truncated": 0,
            "action": "keep", "future_regime": "old_major",
        })
    start = {"driver": {"episode_ordinal": 3, "episode_length": 1,
                         "episode_reset_seed": initial_seed, "obs": [1.0]}}
    final = {"driver": {"episode_ordinal": 4, "episode_length": 1,
                         "episode_reset_seed": reset_after_completion, "obs": [10.0]},
             "environment": {"episode_length": 1}}
    audit = Audit(tmp_path)
    result = _check_episode_ledger(
        audit, tmp_path, rows, final, seed=91, agent_seed=future,
        start_simulator=start, action="keep", future="old_major", where="synthetic",
    )
    assert result["completed_episodes"] == 1
    assert audit.errors == []

"""Pure schedule, reproducibility, and snapshot-contract tests for Stage 1."""
from __future__ import annotations

import json

import pytest
import torch

from scripts import run_retention_mechanism as mechanism


def _raw_config() -> dict:
    return json.loads(mechanism.CONFIG_PATH.read_text(encoding="utf-8"))


def test_phase_configs_and_interaction_ledgers_match_frozen_counts() -> None:
    raw = _raw_config()
    main = mechanism._phase_config(raw, pilot=False)
    pilot = mechanism._phase_config(raw, pilot=True)

    assert main["phase"] == "main"
    assert mechanism._expected_interactions(main) == raw["expected_main"]
    assert pilot["phase"] == "pilot"
    assert pilot["seeds"] == [25089]
    assert pilot["future_blocks"] == 4
    assert pilot["horizons"] == [1536, 6144]
    assert mechanism._expected_interactions(pilot) == raw["expected_pilot"]


@pytest.mark.parametrize("history_regime", ["old_major", "new_major"])
def test_prefix_schedule_preserves_bootstrap_and_candidate_fit_exposure(history_regime: str) -> None:
    config = mechanism._phase_config(_raw_config(), pilot=False)
    mapping = mechanism.pair_mapping(25101, "dynamics")
    schedule = mechanism.build_prefix_schedule(mapping, config, history_regime)

    assert len(schedule) == 10
    assert [(row["group"], row["variant"], row["steps"]) for row in schedule[:2]] == [
        (0, 0, 1024),
        (0, 1, 1024),
    ]
    expected_history = mechanism.template_schedule(
        mapping, history_regime, kind="history", steps=512, start_bout=2,
    )
    assert [(row["group"], row["variant"]) for row in schedule[2:]] == [
        (row["group"], row["variant"]) for row in expected_history
    ]
    assert [row["steps"] for row in schedule[2:8]] == [512] * 6
    assert [(row["group"], row["variant"], row["steps"]) for row in schedule[8:]] == [
        (1, 0, 1024),
        (1, 1, 1024),
    ]
    assert sum(int(row["steps"]) for row in schedule) == 7168


@pytest.mark.parametrize(
    ("future_regime", "dominant", "minority"),
    [("old_major", 0, 1), ("new_major", 1, 0)],
)
def test_future_blocks_use_declared_minority_then_dominant_pair(
    future_regime: str, dominant: int, minority: int,
) -> None:
    mapping = mechanism.pair_mapping(25107, "reward_conflict")
    schedule = mechanism.build_future_schedule(
        mapping, future_regime, blocks=8, bout_steps=512, start_bout=10,
    )

    assert len(schedule) == 24
    for block in range(8):
        rows = schedule[block * 3:(block + 1) * 3]
        variant = block % 3
        assert [(row["group"], row["variant"]) for row in rows] == [
            (minority, variant),
            (dominant, variant),
            (dominant, (variant + 1) % 3),
        ]
        assert [row["bout"] for row in rows] == list(range(10 + 3 * block, 13 + 3 * block))
        assert all(row["steps"] == 512 and row["future_block"] == block for row in rows)


def test_future_seed_and_action_execution_order_are_deterministic() -> None:
    config = mechanism._phase_config(_raw_config(), pilot=False)
    seed = mechanism.future_seed(config, 25101, 1)
    assert seed == 40_000_000 + 1000 * 25101 + 1
    assert seed != mechanism.future_seed(config, 25101, 0)

    first, order_seed = mechanism._branch_specs(config, "s25101_dynamics_old_major")
    repeated, repeated_order_seed = mechanism._branch_specs(config, "s25101_dynamics_old_major")
    assert order_seed == repeated_order_seed
    assert [row["branch_id"] for row in first] == [row["branch_id"] for row in repeated]
    assert len(first) == 16
    assert len({row["branch_id"] for row in first}) == 16
    assert {
        (row["future_regime"], row["replicate"], row["action"])
        for row in first
    } == {
        (future, replicate, action)
        for future in config["future_regimes"]
        for replicate in config["replicates"]
        for action in config["actions"]
    }


def test_serialized_prefix_payload_contract_and_hash_check(tmp_path) -> None:
    learner = {"format": "test-learner", "tensor": torch.arange(3)}
    simulator = {"driver": {"obs": [0.25, 0.5], "episode_length": 7}}
    torch.save(learner, tmp_path / "checkpoint.pt")
    (tmp_path / "simulator.json").write_text(json.dumps(simulator), encoding="utf-8")
    (tmp_path / "actor.bin").write_bytes(b"a" * 89)
    expected_hash = mechanism.semantic_hash({"learner": learner, "simulator": simulator})

    loaded_learner, loaded_simulator, actor = mechanism._load_prefix_payloads(tmp_path, expected_hash)
    assert mechanism.semantic_hash({"learner": loaded_learner, "simulator": loaded_simulator}) == expected_hash
    assert len(actor) == 89
    assert loaded_learner["tensor"] is not learner["tensor"]

    with pytest.raises(RuntimeError, match="do not match"):
        mechanism._load_prefix_payloads(tmp_path, "0" * 64)


def test_package_artifact_paths_are_relative_posix_paths(tmp_path) -> None:
    output = tmp_path
    prefix = output / "cases" / "s25101_dynamics_old_major" / "prefix"
    prefix.mkdir(parents=True)
    (prefix / "checkpoint.pt").write_bytes(b"checkpoint")
    (prefix / "actor.bin").write_bytes(b"actor")
    (prefix / "simulator.json").write_text("{}", encoding="utf-8")

    artifacts = mechanism._package_artifacts(prefix, output)

    assert artifacts["checkpoint"]["path"] == "cases/s25101_dynamics_old_major/prefix/checkpoint.pt"
    assert artifacts["actor"]["path"] == "cases/s25101_dynamics_old_major/prefix/actor.bin"
    assert artifacts["simulator"]["path"] == "cases/s25101_dynamics_old_major/prefix/simulator.json"
    assert all("\\" not in record["path"] for record in artifacts.values())
    assert all(len(record["sha256"]) == 64 for record in artifacts.values())


def test_campaign_cpu_accounting_does_not_double_count_inline_cases() -> None:
    inline = mechanism._campaign_cpu_accounting("inline", 12.0, [10.0])
    assert inline == {
        "campaign_parent_cpu_s": 12.0,
        "campaign_case_cpu_s_sum": 10.0,
        "campaign_child_case_cpu_s": 0.0,
        "campaign_cpu_s": 12.0,
    }

    process_pool = mechanism._campaign_cpu_accounting("process_pool", 2.0, [4.0, 5.0])
    assert process_pool["campaign_cpu_s"] == 11.0
    assert process_pool["campaign_child_case_cpu_s"] == 9.0
    assert mechanism._campaign_cpu_accounting("process_pool", 2.0, [None])["campaign_cpu_s"] is None

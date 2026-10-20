"""Static schedule, provenance, and mocked-helper checks for the candidate harness."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_retention_candidate as candidate
from scripts import run_retention_mechanism as mechanism
from scripts import run_storage_decision as storage
from scripts import run_write_stream as ws
from online_rl.storage_decision import pair_mapping, task_name


def _raw_config() -> dict:
    return json.loads(candidate.CONFIG_PATH.read_text(encoding="utf-8"))


def test_phase_counts_match_frozen_main_and_pilot_budgets() -> None:
    raw = _raw_config()
    main = candidate._phase_config(raw, pilot=False)
    pilot = candidate._phase_config(raw, pilot=True)

    assert candidate._expected_interactions(main) == raw["expected_main"]
    assert raw["expected_main"] == {
        "cases": 24,
        "branches": 480,
        "early_prefix_steps": 7168,
        "late_prefix_steps": 29696,
        "prefix_steps": 442368,
        "steps_per_branch": 6144,
        "branch_steps": 2949120,
        "total_steps": 3391488,
    }
    assert candidate._expected_interactions(pilot) == raw["expected_pilot"]
    assert raw["expected_pilot"] == {
        "cases": 2,
        "branches": 10,
        "early_prefix_steps": 7168,
        "late_prefix_steps": 29696,
        "prefix_steps": 36864,
        "steps_per_branch": 6144,
        "branch_steps": 61440,
        "total_steps": 98304,
    }
    assert candidate._validate_frozen_expected(main, candidate._expected_interactions(main)) == raw["expected_main"]
    assert candidate._validate_frozen_expected(pilot, candidate._expected_interactions(pilot)) == raw["expected_pilot"]


@pytest.mark.parametrize("seed", [25201, 25202])
@pytest.mark.parametrize("state", ["early", "late"])
def test_prefix_plan_uses_parity_history_and_exact_candidate_tail(seed: int, state: str) -> None:
    config = candidate._phase_config(_raw_config(), pilot=False)
    mapping = pair_mapping(seed, "dynamics")
    history_regime = candidate.history_regime_for_seed(seed)
    plan = candidate.build_prefix_plan(mapping, config, state, history_regime)

    assert history_regime == ("old_major" if seed % 2 == 0 else "new_major")
    assert plan["steps"] == (7168 if state == "early" else 29696)
    assert len(plan["bootstrap"]) == 2
    assert [row["steps"] for row in plan["bootstrap"]] == [1024, 1024]
    assert len(plan["history"]) == 8
    assert [row["steps"] for row in plan["history"][:6]] == [512] * 6
    tail_steps = 1024 if state == "early" else 4096
    assert [(row["group"], row["variant"], row["steps"]) for row in plan["candidate_sources"]] == [
        (1, 0, tail_steps),
        (1, 1, tail_steps),
    ]
    assert [row["bout"] for row in plan["candidate_sources"]] == [
        row["bout"] for row in plan["history"][-2:]
    ]

    if state == "early":
        assert plan["warmup_pairs"] == []
    else:
        assert [(pair["cycle"], pair["group"], pair["write_applied_after_pair"]) for pair in plan["warmup_pairs"]] == [
            (cycle, group, group == 0)
            for cycle in range(4)
            for group in (1, 0)
        ]
        for pair in plan["warmup_pairs"]:
            assert [row["variant"] for row in pair["schedule"]] == [0, 1]
            assert [row["steps"] for row in pair["schedule"]] == [1024, 1024]
            assert all(row["group"] == pair["group"] for row in pair["schedule"])


def test_branch_grid_is_complete_and_execution_order_is_repeatable() -> None:
    config = candidate._phase_config(_raw_config(), pilot=False)
    first, first_seed = candidate._branch_specs(config, "s25201_dynamics_early")
    second, second_seed = candidate._branch_specs(config, "s25201_dynamics_early")

    assert first_seed == second_seed
    assert [row["branch_id"] for row in first] == [row["branch_id"] for row in second]
    assert len(first) == 20
    assert len({row["branch_id"] for row in first}) == 20
    assert {
        (row["future_regime"], row["replicate"], row["arm"])
        for row in first
    } == {
        (future, replicate, arm)
        for future in config["future_regimes"]
        for replicate in config["replicates"]
        for arm in candidate.ARMS
    }


def test_evidence_buffer_keeps_only_last_ordered_rows_per_source() -> None:
    owner_rows = []
    for owner in (0, 1):
        owner_rows.append([
            {
                "obs": np.full(13, owner + step / 1000, dtype=np.float32),
                "slot": owner,
                "generation": 20 + owner,
                "bout": 30 + owner,
                "global_step": 10000 * owner + step,
            }
            for step in range(256)
        ])

    evidence = candidate._evidence_arrays(owner_rows, 128)
    assert evidence["observations"].shape == (256, 13)
    assert evidence["owner_indices"].tolist() == [0] * 128 + [1] * 128
    assert evidence["source_slots"].tolist() == [0] * 128 + [1] * 128
    assert evidence["generations"].tolist() == [20] * 128 + [21] * 128
    assert evidence["global_steps"].tolist() == list(range(128, 256)) + list(range(10128, 10256))
    assert evidence["bouts"].tolist() == [30] * 128 + [31] * 128


def test_candidate_source_helpers_use_mechanism_slot_lookup_with_mock_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seed = 25201
    mapping = pair_mapping(seed, "dynamics")
    tasks = [task_name(mapping, 1, variant) for variant in (0, 1)]
    local = np.arange(2 * 27, dtype=np.float32).reshape(2, 27) / 100.0
    shared = np.zeros((1, 9), dtype=np.float32)
    physical_groups = np.asarray([-1, -1], dtype=np.int8)
    monkeypatch.setattr(ws, "_actor_arrays", lambda core: (local.copy(), shared.copy(), physical_groups.copy()))

    agent = SimpleNamespace(
        core=object(),
        cache=SimpleNamespace(slot_to_task=tasks),
        slot_generation=np.asarray([7, 8], dtype=np.uint64),
        fitted_steps=np.asarray([1024, 1024], dtype=np.uint64),
        slot_group_tags=np.asarray([1, 1], dtype=np.int8),
    )
    sources = [
        {"task": task, "variant": variant, "bout": 40 + variant}
        for variant, task in enumerate(tasks)
    ]
    prefix_bouts = [
        {
            "task": task,
            "group": 1,
            "bout": 40 + variant,
            "slot": variant,
            "generation": 7 + variant,
            "steps": 1024,
        }
        for variant, task in enumerate(tasks)
    ]

    records, slots, effective = candidate._candidate_source_records(
        agent, mapping, sources, prefix_bouts, expected_steps=1024,
    )
    assert slots == [0, 1]
    assert [row["task"] for row in records] == tasks
    assert [row["generation"] for row in records] == [7, 8]
    assert [row["latest_source_bout"] for row in records] == [40, 41]
    assert np.array_equal(effective, local.reshape(2, 3, 3, 3))


def test_pair_mean_install_uses_mechanism_lookup_and_compensates_mock_residents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = pair_mapping(25202, "reward_conflict")
    tasks = [task_name(mapping, 0, variant) for variant in (0, 1)]

    class FakeCore:
        def __init__(self) -> None:
            self.local = np.stack((np.full(27, 0.25), np.full(27, -0.1))).astype(np.float32)
            self.shared = np.zeros((1, 9), dtype=np.float32)
            self.slot_groups = torch.tensor([0, 0], dtype=torch.long)

        def install_local_matrix(self, value: np.ndarray) -> None:
            self.local = np.asarray(value, dtype=np.float32).reshape(2, 27).copy()

        def install_shared_matrix(self, value: np.ndarray) -> None:
            self.shared = np.asarray(value, dtype=np.float32).reshape(1, 9).copy()

    core = FakeCore()
    agent = SimpleNamespace(
        core=core,
        cache=SimpleNamespace(slot_to_task=tasks),
        slot_generation=np.asarray([2, 3], dtype=np.uint64),
        fitted_steps=np.asarray([1024, 1024], dtype=np.uint64),
        slot_group_tags=np.asarray([0, 0], dtype=np.int8),
        parent_group=-1,
    )

    monkeypatch.setattr(
        ws,
        "_actor_arrays",
        lambda value: (value.local.copy(), value.shared.copy(), value.slot_groups.detach().cpu().numpy().astype(np.int8)),
    )
    monkeypatch.setattr(storage, "_encode_live_actor", lambda _agent: (b"actor", 0.0, [0.0, 0.0]))
    cleared: list[object] = []
    monkeypatch.setattr(ws, "_clear_optimizer_state", lambda value: cleared.append(value))

    before_effective = core.local.reshape(2, 3, 3, 3).copy()
    result = candidate._install_pair_mean(agent, mapping, group=0, label="mock_old_pair")

    assert result["source_slots"] == [0, 1]
    assert [row["task"] for row in result["source_records"]] == tasks
    assert np.allclose(result["mean_parent"], before_effective.mean(axis=(0, 1)), rtol=0.0, atol=1e-7)
    assert agent.parent_group == 0
    assert core.slot_groups.tolist() == [0, 0]
    assert cleared == [agent]
    assert result["encoded_actor_bytes"] == len(b"actor")


def test_source_snapshot_manifest_excludes_machine_local_instruction_files() -> None:
    assert all(
        not any(part.casefold() in {"agent.md", "agents.md"} for part in path.parts)
        for path in candidate.SOURCE_FILES
    )


def test_cli_entrypoint_is_after_case_worker_definition() -> None:
    tree = ast.parse(Path(candidate.__file__).read_text(encoding="utf-8"))
    case_worker = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_case_worker"
    )
    guard = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    assert guard.lineno > case_worker.end_lineno

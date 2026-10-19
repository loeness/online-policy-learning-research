from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest
import torch

from online_rl.retention_stream_value import BoundedRetentionStats
from online_rl.storage_decision import pair_mapping, task_name
from scripts import run_retention_value_stream as runner


def _config():
    return json.loads((ROOT / "configs" / "retention_value_stream.json").read_text(encoding="utf-8"))


def test_stage_grid_uses_frozen_pilot_and_full_comparison_grid():
    config = _config()
    calibration = runner._stream_grid(config, "calibration", True)
    comparison = runner._stream_grid(config, "comparison", True)

    assert len(calibration) == 1
    assert calibration[0]["method"] == "random"
    assert calibration[0]["decisions"] == 4
    assert len(comparison) == 10
    assert [item["method"] for item in comparison] == list(runner.METHODS)
    assert all(item["pair_kind"] == "dynamics" and item["decisions"] == 4 for item in comparison)


def test_outcome_schedule_is_three_bouts_with_eight_block_dominance_switch():
    mapping = pair_mapping(24101, "dynamics")
    first = runner._outcome_schedule(mapping, 24101, 0, bout_start=4)
    seventh = runner._outcome_schedule(mapping, 24101, 7, bout_start=25)
    eighth = runner._outcome_schedule(mapping, 24101, 8, bout_start=28)

    assert [(item["group"], item["variant"]) for item in first] == [
        (0, 0), (1, 0), (1, 1),
    ]
    assert [item["group"] for item in seventh] == [0, 1, 1]
    assert [item["group"] for item in eighth] == [1, 0, 0]
    assert [item["bout"] for item in eighth] == [28, 29, 30]
    assert all(item["steps"] == 512 for item in first + seventh + eighth)


def test_candidate_from_group_one_uses_effective_live_coefficients_when_parent_is_group_zero():
    mapping = pair_mapping(24101, "dynamics")
    task0 = task_name(mapping, 1, 0)
    task1 = task_name(mapping, 1, 1)

    class Core:
        def __init__(self):
            self.local = np.stack((np.ones(27, dtype=np.float32), np.full(27, 3, dtype=np.float32)))
            self.shared = np.full((1, 9), 5, dtype=np.float32)
            self.slot_groups = torch.as_tensor([-1, -1], dtype=torch.long)

        def read_local_matrix(self):
            return self.local.copy()

        def read_shared_matrix(self):
            return self.shared.copy()

    agent = SimpleNamespace(
        core=Core(),
        cache=SimpleNamespace(
            task_to_slot={task0: 0, task1: 1},
            slot_to_task=[task0, task1],
        ),
        slot_generation=np.asarray([2, 7], dtype=np.int64),
        fitted_steps=np.asarray([512, 1024], dtype=np.uint64),
        slot_group_tags=np.asarray([1, 1], dtype=np.int8),
        parent_group=0,
    )
    records = [
        {"task": task0, "group": 1, "slot": 0, "generation": 2, "bout": 10},
        {"task": task1, "group": 1, "slot": 1, "generation": 7, "bout": 11},
    ]

    candidate, group, source_fit, provenance = runner._current_candidate(agent, records, mapping)

    assert group == 1
    assert source_fit == 768.0
    # Group 1 slots are physically unbound while a group 0 parent is current.
    # The candidate therefore averages local-only coefficients (1 and 3).
    assert np.allclose(candidate, np.full((3, 3), 2.0, dtype=np.float32))
    assert provenance["candidate_matrix_persisted"] is False
    assert [source["variant"] for source in provenance["sources"]] == [0, 1]


def test_candidate_rejects_a_stale_generation_and_an_underfit_source():
    mapping = pair_mapping(24101, "reward_conflict")
    task0 = task_name(mapping, 0, 0)
    task1 = task_name(mapping, 0, 1)

    class Core:
        def read_local_matrix(self):
            return np.zeros((2, 27), dtype=np.float32)

        def read_shared_matrix(self):
            return np.zeros((1, 9), dtype=np.float32)

        slot_groups = torch.as_tensor([-1, -1], dtype=torch.long)

    agent = SimpleNamespace(
        core=Core(),
        cache=SimpleNamespace(task_to_slot={task0: 0, task1: 1}, slot_to_task=[task0, task1]),
        slot_generation=np.asarray([4, 6], dtype=np.int64),
        fitted_steps=np.asarray([512, 511], dtype=np.uint64),
        slot_group_tags=np.asarray([0, 0], dtype=np.int8),
    )
    records = [
        {"task": task0, "group": 0, "slot": 0, "generation": 3, "bout": 2},
        {"task": task1, "group": 0, "slot": 1, "generation": 6, "bout": 3},
    ]

    with pytest.raises(ValueError, match="generation is no longer resident"):
        runner._current_candidate(agent, records, mapping)

    records[0]["generation"] = 4
    with pytest.raises(ValueError, match="only 511 fitted steps"):
        runner._current_candidate(agent, records, mapping)


def test_selected_learners_share_matched_exploration_streams():
    spec = {"seed": 24101, "pair_kind": "dynamics", "method": "learned_linear"}
    first = runner._selector_rngs(spec)
    second = runner._selector_rngs({**spec, "method": "learned_intercept"})

    assert runner._rng_state(first) == runner._rng_state(second)
    # Both streams consume both draws each turn, even when one branch explores.
    assert first["exploration"].random() == second["exploration"].random()
    assert first["exploration_action"].integers(4) == second["exploration_action"].integers(4)


def test_cohort_cpu_cost_does_not_double_count_serial_stream_cpu():
    serial = runner._cohort_costs(
        run_wall_s=10.0,
        parent_process_cpu_s=6.0,
        child_cpu_s=4.0,
        workers_actual=1,
        serial_execution=True,
    )
    assert serial == {
        "wall_s": 10.0,
        "child_cpu_s": 4.0,
        "orchestrator_cpu_s": 2.0,
        "total_cpu_s": 6.0,
        "workers_actual": 1,
    }

    parallel = runner._cohort_costs(
        run_wall_s=12.0,
        parent_process_cpu_s=3.0,
        child_cpu_s=8.0,
        workers_actual=4,
        serial_execution=False,
    )
    assert parallel["orchestrator_cpu_s"] == 3.0
    assert parallel["total_cpu_s"] == 11.0

    single_process_pool = runner._cohort_costs(
        run_wall_s=12.0,
        parent_process_cpu_s=0.5,
        child_cpu_s=8.0,
        workers_actual=1,
        serial_execution=False,
    )
    assert single_process_pool["orchestrator_cpu_s"] == 0.5
    assert single_process_pool["total_cpu_s"] == 8.5


def test_summary_reuses_protocol_hash_captured_in_frozen_config(tmp_path, monkeypatch):
    protocol_hash_calls = 0

    def fake_sha256(path):
        nonlocal protocol_hash_calls
        if Path(path) == runner.PROTOCOL_PATH:
            protocol_hash_calls += 1
            return "protocol-at-freeze" if protocol_hash_calls == 1 else "protocol-reread"
        return "config-at-freeze"

    monkeypatch.setattr(runner, "_sha256_file", fake_sha256)
    monkeypatch.setattr(runner, "_load_config", lambda _path: {"max_workers": 1})
    monkeypatch.setattr(runner, "_stream_grid", lambda _config, _stage, _pilot: [])

    output = tmp_path / "cohort"
    summary = runner.run_stage(
        "calibration",
        output,
        pilot=True,
        config_path=tmp_path / "config.json",
    )
    frozen = json.loads((output / "frozen_config.json").read_text(encoding="utf-8"))

    assert protocol_hash_calls == 1
    assert frozen["protocol_sha256"] == "protocol-at-freeze"
    assert summary["protocol_sha256"] == frozen["protocol_sha256"]


def test_keep_does_not_reinstall_or_requantize_the_live_actor(monkeypatch):
    mapping = pair_mapping(24101, "dynamics")
    tasks, task_groups = runner._task_groups(mapping)
    stats = BoundedRetentionStats(tasks, task_groups)
    stats.record_parent_write(1024.0, 0)

    class Core:
        def __init__(self):
            self.local = np.arange(54, dtype=np.float32).reshape(2, 27) / 100.0
            self.shared = np.arange(9, dtype=np.float32).reshape(1, 9) / 100.0
            self.slot_groups = torch.as_tensor([0, -1], dtype=torch.long)

        def read_local_matrix(self):
            return self.local.copy()

        def read_shared_matrix(self):
            return self.shared.copy()

        def install_local_matrix(self, matrix):
            self.local = np.asarray(matrix, dtype=np.float32).copy()

        def install_shared_matrix(self, matrix):
            self.shared = np.asarray(matrix, dtype=np.float32).copy()

    core = Core()
    agent = SimpleNamespace(
        core=core,
        slot_group_tags=np.asarray([0, 1], dtype=np.int8),
        parent_group=0,
        parent_version=1,
        parent_source_fit=1024.0,
    )
    before_local = core.local.copy()
    before_shared = core.shared.copy()
    monkeypatch.setattr(runner.ws, "_clear_optimizer_state", lambda _agent: None)

    result = runner._parent_action(
        agent, stats, np.ones((3, 3), dtype=np.float32), 1, 512.0, "keep", 0,
    )

    assert np.array_equal(core.local, before_local)
    assert np.array_equal(core.shared, before_shared)
    assert agent.parent_group == 0
    assert agent.parent_version == 1
    assert stats.parent_source_fit == 1024.0
    assert result["details"]["codec"]["roundtrip_applied"] is False
    assert result["details"]["codec"]["max_effective_error"] == 0.0

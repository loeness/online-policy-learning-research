"""Focused S1b schedule, probe, grouping, and checkpoint checks."""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.shared_write_grouping import DynamicsGrouper, empty_profile  # noqa: E402
from scripts import run_shared_write_auto as runner  # noqa: E402


def _profile() -> np.ndarray:
    profile = empty_profile()
    for band in range(3):
        for action in range(4):
            direction = (action + band) % 4
            profile[band, action, direction] = 5
    return profile


class FakeEnv:
    """Small deterministic grid stream for probe plan tests."""

    def __init__(self, context):
        self.context = context
        self.x = 4
        self.y = 4
        self.steps = 0

    def _obs(self):
        return np.asarray([self.x / 8.0, self.y / 8.0, 0.5, 0.5], dtype=np.float32)

    def reset(self, *, seed=None):
        self.x, self.y, self.steps = 4, 4, 0
        return self._obs(), {}

    def step(self, action):
        directions = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        dx, dy = directions[int(action)]
        self.x = min(8, max(0, self.x + dx))
        self.y = min(8, max(0, self.y + dy))
        self.steps += 1
        truncated = self.steps == 19
        return self._obs(), float(action), False, truncated, {}

    def close(self):
        pass


class FakeCore:
    def __init__(self):
        self.assignments = {}

    def set_slot_group(self, slot, group, *, preserve_effective=True):
        self.assignments[int(slot)] = int(group)


def _probe_writer(stream: io.StringIO):
    fields = [
        "probe_step", "context", "slot", "episode", "reset_seed", "episode_action_seed",
        "episode_action_index", "obs", "action", "reward", "next_obs", "terminated",
        "truncated", "profile_counted", "interaction_cost",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return writer


def test_frozen_phase_schedules_and_budget_contract():
    raw = runner.load_config()
    pilot, pilot_seeds = runner.resolve_phase_config(raw, "pilot")
    formal, formal_seeds = runner.resolve_phase_config(raw, "formal")
    assert pilot_seeds == [827]
    assert pilot["seen_sequence"] == ["A0", "B0", "A1", "B1"]
    assert pilot["bout_steps"] == pilot["heldout_steps"] == 2048
    assert formal_seeds == [677, 691, 709]
    assert formal["seen_sequence"] == ["A0", "B0", "A1", "B1", "A0", "B0", "A1", "B1"]
    assert formal["heldout_sequence"] == ["A2", "B2"]
    assert formal["bout_steps"] == formal["heldout_steps"] == 4096
    assert formal["probe_steps_per_context"] * 6 == 3072
    assert formal["conditions"] == ["independent", "oracle_active", "auto_active", "auto_frozen"]
    assert formal["actor_budget_bytes"] == 1024
    assert formal["whole_package_cap_bytes"] == 256 * 1024
    assert formal["grouper"] == {
        "max_groups": 2,
        "min_overlap": 4,
        "min_agreement": 0.6,
        "min_cell_support": 2,
        "novelty_max_agreement": 0.3,
    }


def test_probe_is_fixed_uniform_transition_stream_and_counts_separately():
    stream_a, stream_b = io.StringIO(), io.StringIO()
    counts_a, counts_b = {"probe_steps": 0}, {"probe_steps": 0}
    probe_a = runner.collect_probe(
        "A0", 2, 827, steps=512, writer=_probe_writer(stream_a), counts=counts_a, env_factory=FakeEnv
    )
    probe_b = runner.collect_probe(
        "A0", 2, 827, steps=512, writer=_probe_writer(stream_b), counts=counts_b, env_factory=FakeEnv
    )
    assert probe_a["steps"] == probe_b["steps"] == 512
    assert counts_a == counts_b == {"probe_steps": 512}
    np.testing.assert_array_equal(probe_a["profile"], probe_b["profile"])
    assert stream_a.getvalue() == stream_b.getvalue()

    rows = list(csv.DictReader(io.StringIO(stream_a.getvalue())))
    for episode in sorted({int(row["episode"]) for row in rows}):
        episode_rows = [row for row in rows if int(row["episode"]) == episode]
        action_rng = np.random.default_rng(int(episode_rows[0]["episode_action_seed"]))
        expected_actions = action_rng.integers(0, 4, len(episode_rows))
        assert [int(row["action"]) for row in episode_rows] == expected_actions.tolist()
        assert [int(row["episode_action_index"]) for row in episode_rows] == list(range(len(episode_rows)))
    assert all(int(row["interaction_cost"]) == 1 for row in rows)
    assert all(len(json.loads(row["obs"])) == 4 for row in rows)
    assert all(len(json.loads(row["next_obs"])) == 4 for row in rows)
    assert sum(int(row["profile_counted"]) for row in rows) > 0


def test_automatic_assignment_uses_profile_without_family_or_slot_lookup():
    grouper = DynamicsGrouper()
    agent = SimpleNamespace(
        condition="auto_active",
        seed=827,
        core=FakeCore(),
        grouper=grouper,
    )
    first = runner.assign_probe_profile(agent, "A0", 4, _profile())
    second = runner.assign_probe_profile(agent, "B0", 1, _profile())
    assert first["assigned_group"] == 0
    assert second["assigned_group"] == 0
    assert second["reason"] == "matched"
    assert agent.core.assignments == {4: 0, 1: 0}
    assert second["observer_family_audit_only"] == "B"


def test_auto_active_endpoint_checkpoint_restores_with_s1b_factory(tmp_path):
    torch.set_num_threads(1)
    config = runner.load_config()
    config.update(condition="auto_active", seed=809)
    agent, slots, base = runner.make_agent("auto_active", 809, config)
    assert agent.core.shared_groups == 2
    assert agent.core.slot_groups.tolist() == [-1] * 6
    profile = _profile()
    decision = runner.assign_probe_profile(agent, "A0", slots["A0"], profile)
    agent.probe_profiles["A0"] = profile.copy()
    agent.group_decisions.append(decision)
    with torch.no_grad():
        agent.core._nodes["1"].A.fill_(0.012)
        agent.core._shared_coefficients[decision["assigned_group"]].C.fill_(0.04)
    agent.optimizer.step_count = 17
    folder = tmp_path / "seed_809" / "auto_active"
    folder.mkdir(parents=True)
    runner.write_json(folder / "config.json", config)
    sizes = runner.save(agent, folder, "final", config, slots, base)

    restored, restored_slots, _ = runner.restore_run(folder)
    assert restored_slots == slots
    assert restored.condition == "auto_active"
    assert restored.grouper is not None
    assert restored.grouper.active_groups == (0,)
    assert restored.core.group_for_slot(slots["A0"]) == 0
    assert restored.optimizer.step_count == 17
    np.testing.assert_array_equal(restored.core.read_local_matrix(), agent.core.read_local_matrix())
    np.testing.assert_array_equal(restored.core.read_shared_matrix(), agent.core.read_shared_matrix())
    assert restored.probe_profiles["A0"].dtype == np.uint16
    assert sizes["actor_bytes"] == 738
    assert sizes["whole_package_bytes"] <= config["whole_package_cap_bytes"]
    assert restored.core.source_basis.requires_grad is False


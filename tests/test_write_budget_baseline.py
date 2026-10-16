from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.write_decision_gate import AnchorBank  # noqa: E402
from scripts import run_write_budget_baseline as baseline  # noqa: E402
from scripts import run_write_stream as stream  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_stream_hooks():
    baseline.install_hooks()
    yield
    baseline.uninstall_hooks()


def test_default_budgets_match_training_and_probe_interaction_counts():
    assert baseline.default_bout_steps("known") == 2_144
    assert baseline.default_bout_steps("unknown") == 2_048
    assert baseline.expected_training_interactions("known") == 51_456
    assert baseline.expected_training_interactions("unknown") == 49_152
    assert baseline.expected_training_interactions("known", 7) == 168
    with pytest.raises(ValueError, match="phase"):
        baseline.default_bout_steps("other")


def test_configuration_is_explicitly_probe_free_g0_and_budget_bounded():
    config = baseline._wrapped_load_config()

    assert config["budget_baseline"] is True
    assert config["all_rows_fit"] is True
    assert config["probe_steps"] == 0
    assert config["local_slots"] == 7
    assert config["fast_local_learning_rate"] == 0.004
    assert config["actor_budget_bytes"] == 236
    assert config["independent7_actor_budget_bytes"] == 236
    assert config["package_budget_bytes"] == 262_144
    assert stream.PROTOCOL_PATH == ROOT / "docs" / "WRITE_BUDGET_BASELINE.md"
    assert Path("scripts/run_write_budget_baseline.py") in stream.SOURCE_FILES
    assert Path("docs/WRITE_BUDGET_BASELINE.md") in stream.SOURCE_FILES


def test_real_g0_agent_uses_nonzero_internal_window_and_real_anchor_bank():
    config = baseline._wrapped_load_config()
    config.update({"phase": "known", "condition": baseline.CONDITION, "seed": 117})
    base_config = stream.loop.load_base_config()

    agent = baseline._wrapped_new_agent(
        baseline.CONDITION, 117, config, base_config
    )

    assert agent.external_condition == baseline.CONDITION
    assert agent.condition == "independent_fast"
    assert agent.slots == 7
    assert agent.core.shared_groups == 0
    assert agent.run_config["probe_steps"] == 0
    assert all(window.capacity == 128 for window in agent.windows)
    assert isinstance(agent.anchors, AnchorBank)
    assert (agent.anchors.slots, agent.anchors.per_slot) == (7, 24)

    obs = np.zeros(13, dtype=np.float32)
    agent.anchors.add(0, 0, [{
        "obs": obs,
        "action": 1,
        "log_prob": float(np.log(0.25)),
        "probabilities": np.full(4, 0.25, dtype=np.float32),
        "advantage": 1.0,
        "returns": 1.0,
        "episode_id": 3,
        "generation": 0,
    }])
    rows = agent.anchors.arrays(0, 0)
    assert rows["obs"].shape == (1, 13)
    assert rows["action"].tolist() == [1]


def test_generation_reset_keeps_probe_budget_zero_and_window_capacity_valid():
    config = baseline._wrapped_load_config()
    config.update({"phase": "known", "condition": baseline.CONDITION, "seed": 118})
    agent = baseline._wrapped_new_agent(
        baseline.CONDITION, 118, config, stream.loop.load_base_config()
    )

    generation = baseline._wrapped_reset_allocated_slot(
        agent, 0, reason="unit_test"
    )

    assert generation == 1
    assert agent.run_config["probe_steps"] == 0
    assert agent.windows[0].capacity == 128


def test_all_four_fresh_rollouts_are_forwarded_once_as_fit(monkeypatch, tmp_path):
    seen: list[tuple[str, int, object]] = []

    def capture(agent, **_kwargs):
        seen.append((agent.rollout_phase, len(agent.rollout), agent.rollout))

    monkeypatch.setitem(baseline._ORIGINALS, "process_rollout", capture)
    agent = SimpleNamespace(
        run_config={"budget_baseline": True},
        rollout=[],
        rollout_phase=None,
    )
    chunks = [[object() for _ in range(256)] for _ in range(4)]
    for index, chunk in enumerate(chunks):
        agent.rollout = chunk
        agent.rollout_phase = "fit" if index < 3 else "holdout"
        baseline._wrapped_process_rollout(agent, folder=tmp_path)

    assert [phase for phase, _, _ in seen] == ["fit"] * 4
    assert [size for _, size, _ in seen] == [256] * 4
    assert all(actual is expected for (_, _, actual), expected in zip(seen, chunks))


def test_cohort_freezes_flags_and_runs_only_g0_without_training(monkeypatch, tmp_path):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        '{"format":"write_decision_selection_v1",'
        '"selected_conditions":["trial_gate","projected_gate"]}',
        encoding="utf-8",
    )

    def fake_run_one(config, _selection, output):
        output.mkdir(parents=True, exist_ok=True)
        stream._write_json(output / "config.json", dict(config))
        steps = int(config["bout_steps"]) * len(stream.STREAM_SEQUENCE)
        return {
            "phase": config["phase"],
            "condition": config["condition"],
            "seed": int(config["seed"]),
            "counts": {
                "raw_steps": steps,
                "probe_steps": 0,
                "fit_steps": steps,
                "holdout_steps": 0,
                "rollouts_trained": 96,
                "minibatch_updates": 480,
            },
            "costs": {
                "training_steps": steps,
                "probe_steps": 0,
                "evaluation_steps": 17,
                "total_interactions": steps + 17,
                "decision_cpu_s": 0.25,
                "cpu_s": 2.5,
            },
            "limitations": [],
        }

    monkeypatch.setitem(baseline._ORIGINALS, "run_one", fake_run_one)
    output = tmp_path / "cohort"
    result = baseline.run_cohort(
        "known", output, selection_path=selection_path, workers=1,
        seeds=[1031], bout_steps=3,
    )

    frozen = json.loads((output / "frozen_config.json").read_text(encoding="utf-8"))
    run_config = json.loads(
        (output / "seed_1031" / baseline.CONDITION / "config.json").read_text(encoding="utf-8")
    )
    run_summary = json.loads(
        (output / "seed_1031" / baseline.CONDITION / "summary.json").read_text(encoding="utf-8")
    )
    assert frozen["conditions"] == [baseline.CONDITION]
    assert frozen["budget_baseline"] is True
    assert frozen["probe_steps"] == 0
    assert frozen["all_rows_fit"] is True
    assert run_config["budget_baseline"] is True
    assert result["conditions"] == [baseline.CONDITION]
    assert run_summary["budget_comparison"]["expected_training_interactions"] == 72
    assert run_summary["budget_comparison"]["paid_probe_interactions"] == 0
    assert run_summary["budget_comparison"]["evaluation_interactions_separate"] == 17
    assert run_summary["budget_comparison"]["compute_matched"] is False
    assert run_summary["budget_comparison"]["optimization_accounting"]["minibatch_updates"] == 480


def test_sample_log_probability_matches_policy_distribution():
    probs = np.asarray([0.1, 0.6, 0.2, 0.1], dtype=np.float32)
    action, log_prob = stream._sample(probs, np.random.default_rng(44))

    assert 0 <= action < 4
    assert log_prob == pytest.approx(float(np.log(probs[action])), abs=1e-7)

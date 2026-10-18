"""Unit checks for common-prefix source-dose bookkeeping (no environment rollout)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_source_dose as source_dose  # noqa: E402


def _config(**overrides):
    config = {
        "format": "storage_decision_v1",
        "bootstrap_steps": 2048,
        "bout_steps": 1024,
        "source_capture_steps": 1024,
    }
    config.update(overrides)
    return config


def test_source_dose_config_requires_rollout_aligned_capture_within_bootstrap():
    valid = source_dose.validate_config(_config())
    assert valid["source_capture_steps"] == 1024
    with pytest.raises(ValueError, match="multiple of rollout_steps"):
        source_dose.validate_config(_config(source_capture_steps=1000))
    with pytest.raises(ValueError, match="cannot exceed bootstrap_steps"):
        source_dose.validate_config(_config(bootstrap_steps=512, source_capture_steps=1024))


def test_pilot_uses_top_level_pilot_capture_steps():
    config = _config(
        pilot_source_capture_steps=256,
        pilot={
            "seeds": [20089],
            "pair_kinds": ["dynamics"],
            "history_regimes": ["old_major"],
            "bootstrap_steps": 512,
            "bout_steps": 256,
        },
    )
    normalized = source_dose._phase_config(config, pilot=True)
    assert normalized["bootstrap_steps"] == 512
    assert normalized["bout_steps"] == 256
    assert normalized["source_capture_steps"] == 256


def test_captured_source_mean_averages_two_sources_and_all_three_bands():
    first = np.arange(27, dtype=np.float32)
    second = (np.arange(27, dtype=np.float32) + 27.0)
    candidate = source_dose._mean_captured_sources({
        0: {"effective_matrix_27": first},
        1: {"effective_matrix_27": second},
    })
    expected = np.stack((first.reshape(3, 3, 3), second.reshape(3, 3, 3))).mean(
        axis=(0, 1), dtype=np.float64,
    ).astype(np.float32)
    np.testing.assert_array_equal(candidate, expected)
    assert candidate.shape == (3, 3)


def test_observer_captures_only_old_source_at_requested_post_rollout_step():
    class _Core:
        def __init__(self):
            self.local = np.arange(54, dtype=np.float32).reshape(2, 27)
            self.shared = np.full((1, 9), 9.0, dtype=np.float32)
            self.slot_groups = torch.tensor([-1, -1], dtype=torch.long)

        def read_local_matrix(self):
            return self.local.copy()

        def read_shared_matrix(self):
            return self.shared.copy()

    captures = {}
    observer = source_dose._make_source_observer(
        captures, seed=20101, pair_kind="dynamics", capture_steps=1024,
    )
    agent = SimpleNamespace(
        core=_Core(),
        slot_generation=np.asarray([3, 4], dtype=np.int64),
        fitted_steps=np.asarray([768, 1024], dtype=np.uint64),
        runtime_counts={"environment_steps": 3072, "rollouts_trained": 12, "minibatch_updates": 40},
    )
    kwargs = {
        "agent": agent,
        "phase": "bootstrap",
        "bout_record": {"task": source_dose.base.task_name(source_dose.base.pair_mapping(20101, "dynamics"), 0, 1),
                        "group": 0, "bout": 1},
        "slot": 1,
        "generation": 4,
        "bout_step": 1024,
        "rollout_index": 3,
        "rollout_rows": 256,
        "update": {"updates": 2},
    }
    observer(**{**kwargs, "phase": "future"})
    observer(**{**kwargs, "bout_step": 768})
    assert captures == {}
    observer(**kwargs)
    assert set(captures) == {1}
    row = captures[1]
    assert row["fitted_steps"] == 1024
    assert row["global_environment_steps"] == 3072
    assert row["last_rollout_updates"] == 2
    np.testing.assert_array_equal(row["effective_matrix_27"], agent.core.local[1])


def test_persisted_step_ledger_counts_shared_prefix_and_both_branch_sets(tmp_path):
    case = tmp_path / "cases" / "s1_dynamics_old_major"
    paths = {
        case / "prefix" / "steps.csv": 12,
        case / "branches" / "keep__old_major" / "steps.csv": 8,
        case / "dose_1024" / "branches" / "mean__new_major" / "steps.csv": 4,
    }
    for path, count in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("global_step", "reward"))
            writer.writerows((index, 0.0) for index in range(count))
    counts = source_dose._persisted_step_counts(case)
    assert counts == {
        "prefix_environment_steps": 12,
        "base_branch_environment_steps": 8,
        "source_1024_branch_environment_steps": 4,
        "branch_environment_steps": 12,
        "environment_steps_total": 24,
    }


def test_only_hash_identical_replace_or_none_branch_is_reusable():
    branch = {
        "action": "replace",
        "future_regime": "old_major",
        "action_state_hash": "same-state",
    }
    rows = [branch, {**branch, "action": "none", "action_state_hash": "none-state"}]
    assert source_dose._matching_reusable_branch(
        rows, action="replace", future_regime="old_major", action_state_hash="same-state",
    ) is branch
    assert source_dose._matching_reusable_branch(
        rows, action="replace", future_regime="old_major", action_state_hash="different-state",
    ) is None
    assert source_dose._matching_reusable_branch(
        rows, action="keep", future_regime="old_major", action_state_hash="same-state",
    ) is None
    assert source_dose._matching_reusable_branch(
        [branch, dict(branch)], action="replace", future_regime="old_major", action_state_hash="same-state",
    ) is None


def test_dose_selector_contract_freezes_correct_candidate_before_future():
    captures = {
        0: {"effective_matrix_27": np.ones(27, dtype=np.float32)},
        1: {"effective_matrix_27": np.full(27, 3.0, dtype=np.float32)},
    }
    frozen = {}
    calls = []

    def original(history, evidence, old, new, score):
        calls.append(float(np.mean(old)))
        return {"past_only_selectors": {"surrogate_gain": "replace" if np.mean(old) < 3 else "keep"}}

    observed = source_dose._make_decision_observer(captures, frozen, original)
    result = observed([0] * 8, object(), np.full((3, 3), 4.0), np.zeros((3, 3)), object())
    assert result["past_only_selectors"]["surrogate_gain"] == "keep"
    assert frozen["low"]["past_only_selectors"]["surrogate_gain"] == "replace"
    assert calls == [4.0, 2.0]
    with pytest.raises(RuntimeError, match="already frozen"):
        observed([0] * 8, object(), np.ones((3, 3)), np.zeros((3, 3)), object())

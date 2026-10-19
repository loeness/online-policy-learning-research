from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.retention_stream_value import (  # noqa: E402
    FEATURE_NAMES,
    BoundedRetentionStats,
    apply_retention_action,
    frequency_action,
    generalized_lower_decision,
)
from online_rl.storage_decision import effective_matrices  # noqa: E402


TASKS = ["A0+", "A1+", "A2+", "B0+", "B1+", "B2+"]
TASK_GROUPS = {task: (0 if task.startswith("A") else 1) for task in TASKS}


def _bout(
    task: str,
    group: int,
    first256: float,
    *,
    allocated: bool = True,
    evicted_task: str | None = None,
) -> dict:
    return {
        "task": task,
        "group": group,
        "allocated": allocated,
        "evicted_task": evicted_task,
        "return_first_256": first256,
    }


def _stats_with_history() -> BoundedRetentionStats:
    stats = BoundedRetentionStats(TASKS, TASK_GROUPS)
    stats.observe_bout(_bout("A0+", 0, 256.0))
    stats.observe_bout(_bout("B0+", 1, 258.56))
    stats.observe_bout(_bout("B1+", 1, 263.68, evicted_task="B0+"))
    return stats


def test_repeated_actions_keep_residents_pre_codec_and_handle_both_groups() -> None:
    local = np.arange(54, dtype=np.float32).reshape(2, 27) / 10.0
    shared = np.arange(9, dtype=np.float32).reshape(1, 9) / 5.0
    physical_groups = np.asarray([0, -1], dtype=np.int8)
    logical_tags = np.asarray([0, 1], dtype=np.int8)
    candidate = (np.arange(9, dtype=np.float32).reshape(3, 3) + 2.0) / 7.0
    before = effective_matrices(local, shared, physical_groups)

    kept_local, kept_shared, kept_groups, kept_info = apply_retention_action(
        local, shared, physical_groups, logical_tags, 0, candidate, 1, "keep"
    )
    np.testing.assert_array_equal(kept_local, local)
    np.testing.assert_array_equal(kept_shared, shared)
    np.testing.assert_array_equal(kept_groups, physical_groups)
    assert kept_info["parent_group"] == 0
    np.testing.assert_array_equal(effective_matrices(kept_local, kept_shared, kept_groups), before)

    expected_bindings = {"replace": 1, "mean": -2, "none": -1}
    for action, expected_binding in expected_bindings.items():
        result_local, result_shared, result_groups, info = apply_retention_action(
            local, shared, physical_groups, logical_tags, 0, candidate, 1, action
        )
        assert info["parent_group"] == expected_binding
        np.testing.assert_allclose(effective_matrices(result_local, result_shared, result_groups), before, atol=1e-6)
        assert info["max_effective_preservation_error"] <= 1e-6

    replaced = apply_retention_action(
        local, shared, physical_groups, logical_tags, 0, candidate, 1, "replace"
    )
    assert replaced[2].tolist() == [-1, 0]
    np.testing.assert_allclose(replaced[1][0], candidate.reshape(-1))

    # Repeated replacement can bind back to group 0 after binding group 1.
    back_to_old_group = apply_retention_action(
        replaced[0], replaced[1], replaced[2], logical_tags, 1,
        np.full((3, 3), -0.5, dtype=np.float32), 0, "replace",
    )
    assert back_to_old_group[3]["parent_group"] == 0
    assert back_to_old_group[2].tolist() == [0, -1]
    np.testing.assert_allclose(
        effective_matrices(*back_to_old_group[:3]), before, atol=1e-6
    )


def test_stats_track_recurrence_eviction_ewma_and_fixed_capacity_state() -> None:
    stats = BoundedRetentionStats(TASKS, TASK_GROUPS)
    expected_numeric_bytes = 6 + 6 + 6 * 8 + 6 * 8 + 6 * 8 + 8 + 8 + 2 * 8 + 8
    assert stats.numeric_nbytes() == expected_numeric_bytes
    first = _bout("A0+", 0, 256.0)
    assert stats.recurrence_info(first)["recurrence"] is False
    first_info = stats.observe_bout(first)
    assert first_info["seen_before"] is False
    assert stats.seen[0]
    assert stats.visits[0] == 1
    assert stats.recovery_ewma[0] == pytest.approx(1.0)

    stats.observe_bout(_bout("B0+", 1, 258.56))
    recurrence = _bout("A0+", 0, 512.0, evicted_task="B0+")
    before_info = stats.recurrence_info(recurrence)
    assert before_info["seen_before"] is True
    assert before_info["allocated"] is True
    assert before_info["recurrence"] is True
    assert before_info["natural_eviction"] is True
    stats.observe_bout(recurrence)

    assert stats.visits[0] == 2
    assert stats.recovery_ewma[0] == pytest.approx(1.25)
    assert stats.evictions[3] == 1
    assert stats.recent_group_values.tolist() == [0, 1, 0]
    assert stats.recent_eviction_values.tolist() == [0, 0, 1]

    for index in range(10):
        task = TASKS[index % 6]
        stats.observe_bout(_bout(task, TASK_GROUPS[task], float(index)))
    assert stats.recent_length == 8
    assert len(stats.recent_group_values) == len(stats.recent_eviction_values) == 8

    snapshot = stats.state_dict()
    restored = BoundedRetentionStats.from_state_dict(snapshot)
    assert stats.numeric_nbytes() == restored.numeric_nbytes()
    for key in (
        "task_groups", "seen", "visits", "recovery_ewma", "evictions",
        "recent_groups", "recent_evictions",
    ):
        np.testing.assert_array_equal(snapshot[key], restored.state_dict()[key])
    assert restored.parent_source_fit == stats.parent_source_fit
    assert restored.parent_written_decision == stats.parent_written_decision
    snapshot["visits"][:] = 999
    assert np.max(restored.visits) < 999


def test_features_follow_declared_frequency_recovery_fit_age_and_eviction_rules() -> None:
    stats = _stats_with_history()
    stats.record_parent_write(source_fit=1024.0, decision_index=2)
    features, names = stats.features(
        candidate_group=1,
        incumbent_binding=0,
        candidate_source_fit=8192.0,
        decision_index=18,
    )

    assert names == FEATURE_NAMES
    assert features.dtype == np.float64
    assert features.shape == (6,)
    assert features[0] == 1.0
    assert features[1] == pytest.approx(1.0 / 3.0)
    assert features[2] == pytest.approx(1.0)
    assert features[3] == pytest.approx(
        (np.log1p(8192.0) - np.log1p(1024.0)) / np.log(8193.0)
    )
    assert features[4] == pytest.approx(1.0)
    assert features[5] == pytest.approx(1.0 / 3.0)

    stats.record_parent_write(source_fit=8192.0, decision_index=18)
    restored = BoundedRetentionStats.from_state_dict(stats.state_dict())
    np.testing.assert_array_equal(
        restored.features(1, 0, 1024.0, 34)[0],
        stats.features(1, 0, 1024.0, 34)[0],
    )


def test_frequency_rule_respects_group_wildcard_and_none_bindings() -> None:
    stats = _stats_with_history()
    assert frequency_action(stats, candidate_group=1, incumbent_binding=0) == "replace"
    assert frequency_action(stats, candidate_group=1, incumbent_binding=-2) == "keep"
    assert frequency_action(stats, candidate_group=1, incumbent_binding=-1) == "replace"
    assert frequency_action(stats, candidate_group=0, incumbent_binding=1) == "keep"


def test_generalized_lower_scores_current_group_bindings_and_requires_both_valid() -> None:
    stats = _stats_with_history()
    evidence = {0: {"group": 0}, 1: {"group": 1}}
    incumbent = np.ones((3, 3), dtype=np.float32)
    candidate = np.full((3, 3), 2.0, dtype=np.float32)
    calls = []

    def score_fn(group_evidence, candidate_coefficients, baseline_coefficients):
        group = group_evidence["group"]
        calls.append((group, candidate_coefficients.copy(), baseline_coefficients.copy()))
        lower = float(candidate_coefficients.sum() - baseline_coefficients.sum())
        return {"lower": lower}

    action, diagnostics = generalized_lower_decision(
        evidence, score_fn, stats, incumbent, 0, candidate, 1
    )
    assert action == "replace"
    assert diagnostics["format"] == "generalized_repeated_retention_lower_v1"
    assert diagnostics["all_group_scores_valid"] is True
    assert diagnostics["group_scores"]["0"]["lower"] == pytest.approx(-9.0)
    assert diagnostics["group_scores"]["1"]["lower"] == pytest.approx(18.0)
    assert diagnostics["frequency_weighted_lower"] == pytest.approx(9.0)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][1], np.zeros((3, 3), dtype=np.float32))
    np.testing.assert_array_equal(calls[0][2], incumbent)
    np.testing.assert_array_equal(calls[1][1], candidate)
    np.testing.assert_array_equal(calls[1][2], np.zeros((3, 3), dtype=np.float32))

    def unsupported_score(_group_evidence, _candidate_coefficients, _baseline_coefficients):
        return {"lower": None}

    invalid_action, invalid_diagnostics = generalized_lower_decision(
        evidence, unsupported_score, stats, incumbent, 0, candidate, 1
    )
    assert invalid_action == "keep"
    assert invalid_diagnostics["frequency_weighted_lower"] is None
    assert invalid_diagnostics["all_group_scores_valid"] is False


def test_stats_validate_fixed_task_mapping_and_records() -> None:
    with pytest.raises(ValueError, match="exactly six"):
        BoundedRetentionStats(TASKS[:5], TASK_GROUPS)
    with pytest.raises(ValueError, match="exactly the six"):
        BoundedRetentionStats(TASKS, {"A0+": 0})

    stats = BoundedRetentionStats(TASKS)
    record = _bout("A0+", 0, 256.0)
    stats.recurrence_info(record)  # Lookup alone does not count the first visit.
    assert stats.task_groups == {}
    stats.observe_bout(record)  # Observation fixes the task's group.
    assert stats.task_groups == {"A0+": 0}
    with pytest.raises(ValueError, match="changed its fixed group"):
        stats.observe_bout(_bout("A0+", 1, 256.0))
    with pytest.raises(ValueError, match="return_first_256"):
        stats.observe_bout({"task": "A1+", "group": 0, "allocated": False})

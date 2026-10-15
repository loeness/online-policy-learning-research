from scripts.report_shared_write_conflict import _eval_metrics, _run_metrics


def _evaluation_row(stage, mode, return_value, *, goal_reached="0", task_success=""):
    return {
        "stage": stage,
        "mode": mode,
        "episode": "1",
        "return": str(return_value),
        "goal_reached": goal_reached,
        "task_success": task_success,
        "steps": "1",
    }


def test_cold_transfer_is_separate_from_minus_adaptation_and_goal_metadata():
    evaluations = []
    values = {
        "initial": {"plus": (6.0, "1"), "minus": (2.0, "")},
        "after_plus": {"plus": (5.0, "1"), "minus": (-3.0, "")},
        "after_minus": {"plus": (4.0, "0"), "minus": (4.0, "")},
    }
    for stage, modes in values.items():
        for mode, (return_value, task_success) in modes.items():
            evaluations.append(_evaluation_row(
                stage,
                mode,
                return_value,
                goal_reached="1" if stage == "after_minus" and mode == "minus" else task_success or "0",
                task_success=task_success,
            ))

    metrics = _run_metrics(
        "shared_conflict",
        {},
        evaluations,
        [{"mode": "minus", "reward": "0.25"}],
        {},
        None,
        None,
    )

    assert metrics["minus_cold_transfer_signed_return"] == -5.0
    assert metrics["minus_adaptation_gain_after_plus"] == 7.0
    assert metrics["plus_return_retention_delta"] == -1.0
    assert metrics["minus_goal_termination_rate_after_minus_metadata"] == 1.0
    assert metrics["minus_goal_metric_label"] == "environment termination metadata, not task success"

    summarized = _eval_metrics(evaluations)
    assert summarized["after_minus"]["minus"]["goal_success_rate"] is None
    assert summarized["after_minus"]["minus"]["goal_termination_rate_metadata"] == 1.0

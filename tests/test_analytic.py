import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.analytic import AnalyticController  # noqa: E402


@pytest.fixture(scope="module")
def source_responses() -> np.ndarray:
    with np.load(Path(__file__).resolve().parents[1] / "data" / "pretrain_source.npz") as archive:
        return archive["source_responses"]


def _observe_rotation(
    controller: AnalyticController,
    source_responses: np.ndarray,
    action: int,
    rotation: int,
    slip: int = 0,
) -> None:
    before = np.asarray([0.5, 0.5, 0.75, 0.75], dtype=np.float32)
    after = before.copy()
    effective_action = (action + rotation + slip) % 4
    after[:2] += source_responses[effective_action]
    controller.observe(before, action, after)


@pytest.mark.parametrize("rotation", [1, -1])
def test_estimates_quarter_turn_and_inverts_foundation_actions(source_responses, rotation):
    controller = AnalyticController(source_responses)
    for action in (0, 1, 2, 3):
        _observe_rotation(controller, source_responses, action, rotation)
    assert controller.estimated_rotation == 0

    _observe_rotation(controller, source_responses, 0, rotation)
    assert controller.estimated_rotation == rotation
    for action in (1, 2, 3):
        _observe_rotation(controller, source_responses, action, rotation)

    assert controller.estimated_rotation == rotation
    assert [controller.map_action(action) for action in range(4)] == [
        (action - rotation) % 4 for action in range(4)
    ]


def test_estimate_adapts_after_rotation_switch(source_responses):
    controller = AnalyticController(source_responses, window=8, minimum_events=5)
    for action in range(8):
        _observe_rotation(controller, source_responses, action % 4, rotation=1)
    assert controller.estimated_rotation == 1

    for action in range(8):
        _observe_rotation(controller, source_responses, action % 4, rotation=-1)
    assert controller.estimated_rotation == -1
    assert controller.map_action(0) == 1


def test_wall_collisions_do_not_add_evidence(source_responses):
    controller = AnalyticController(source_responses)
    before = np.asarray([0.0, 0.5, 0.75, 0.75], dtype=np.float32)
    after = before.copy()
    for action in range(10):
        controller.observe(before, action % 4, after)

    assert controller.estimated_rotation == 0
    assert controller.map_action(2) == 2
    assert controller.snapshot()["events"] == []


def test_recent_history_is_bounded_and_snapshot_restores_state(source_responses):
    controller = AnalyticController(source_responses, window=3, minimum_events=2)
    for action in range(5):
        _observe_rotation(controller, source_responses, action % 4, rotation=1)

    snapshot = controller.snapshot()
    assert len(snapshot["events"]) == 3
    restored = AnalyticController.from_snapshot(json.loads(json.dumps(snapshot)))
    assert restored.estimated_rotation == controller.estimated_rotation == 1
    assert restored.map_action(0) == controller.map_action(0)

    for action in range(4):
        _observe_rotation(controller, source_responses, action, rotation=-1)
        _observe_rotation(restored, source_responses, action, rotation=-1)
    assert restored.snapshot() == controller.snapshot()


def test_adjacent_slips_do_not_overturn_consistent_rotation(source_responses):
    controller = AnalyticController(source_responses)
    slips = {2: 1, 11: -1}
    for event_index in range(16):
        action = event_index % 4
        _observe_rotation(
            controller,
            source_responses,
            action,
            rotation=1,
            slip=slips.get(event_index, 0),
        )

    assert controller.estimated_rotation == 1


def test_controller_rejects_invalid_configuration_and_action(source_responses):
    with pytest.raises(ValueError):
        AnalyticController(source_responses, window=4, minimum_events=5)
    with pytest.raises(ValueError):
        AnalyticController(source_responses, window=0)
    controller = AnalyticController(source_responses)
    with pytest.raises(ValueError):
        controller.map_action(4)

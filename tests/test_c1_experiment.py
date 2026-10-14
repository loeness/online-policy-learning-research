"""Fast integration checks for C1 stream and policy runner contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.c1_envs import C1_CONTEXTS, context_index  # noqa: E402
from online_rl.c1_experiment import C1Stream, load_bundle, train  # noqa: E402
from online_rl.c1_policy import C1ResidualPolicy  # noqa: E402
from online_rl.c1_routing import C1HistoryRouter  # noqa: E402
from online_rl.compression_experiment import actor_hash  # noqa: E402
from online_rl.envs import NavigationEnv  # noqa: E402
from online_rl.experiment import BlankEnv, make_model  # noqa: E402


torch.set_num_threads(1)


class _Rows:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def writerow(self, row: dict[str, object]) -> None:
        self.rows.append(row)


class _ClockRouter:
    """Router spy whose next route can only depend on completed observations."""

    def __init__(self) -> None:
        self.events: list[tuple[np.ndarray, int, np.ndarray]] = []
        self.calibration_writes = 0
        self.route_inputs: list[np.ndarray] = []

    @property
    def confidence(self) -> float:
        return 0.5

    def route(self, obs: np.ndarray) -> int:
        value = np.asarray(obs, dtype=np.float32)
        assert value.shape == (4,)
        self.route_inputs.append(value.copy())
        return len(self.events)

    def features(self) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    def observe(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> None:
        before = np.asarray(obs, dtype=np.float32)
        after = np.asarray(next_obs, dtype=np.float32)
        assert before.shape == after.shape == (4,)
        self.events.append((before.copy(), int(action), after.copy()))

    def observe_calibration(self, *args: object) -> None:
        self.calibration_writes += 1


@pytest.mark.parametrize("name", tuple(C1_CONTEXTS))
def test_oracle_stream_calibrates_the_matching_zero_based_profile(name: str) -> None:
    router = C1HistoryRouter(window=8)
    stream = C1Stream([(name, 1)], seed=41, router=router, mode="oracle")
    stream.reset()
    stream.step(0)

    totals = router.outcome_counts.sum(axis=(1, 2, 3))
    expected_profile = context_index(name)
    assert totals[expected_profile] == 1
    assert np.delete(totals, expected_profile).sum() == 0
    stream.close()


def test_hidden_stream_routes_before_action_then_updates_from_that_transition() -> None:
    router = _ClockRouter()
    rows = _Rows()
    stream = C1Stream(
        [("A0", 1), ("B0", 1)],
        seed=52,
        router=router,
        mode="calibrated_history",
        steps_writer=rows,
    )

    obs, _ = stream.reset()
    assert int(obs[-1]) == 0
    after_a, _, _, _, info_a = stream.step(0)
    assert info_a["success"] is False
    assert rows.rows[0]["context"] == "A0"
    assert rows.rows[0]["route"] == 0
    assert int(after_a[-1]) == 1
    assert len(router.events) == 1

    # The first B0 action uses the route computed after the preceding A0
    # transition. No B0 transition has been observed at that point.
    after_b, _, _, _, info_b = stream.step(0)
    assert info_b["success"] is False
    assert rows.rows[1]["context"] == "B0"
    assert rows.rows[1]["route"] == int(after_a[-1]) == 1
    assert int(after_b[-1]) == 2
    assert len(router.events) == 2
    assert router.calibration_writes == 0
    assert all(len(event) == 3 for event in router.events)
    assert all(value.shape == (4,) for event in router.events for value in (event[0], event[2]))
    assert len(router.route_inputs) == 3
    stream.close()


def _make_smoke_core(seed: int = 23) -> tuple[C1ResidualPolicy, torch.optim.Optimizer]:
    config = {"learning_rate": 0.01, "ent_coef": 0.06, "seed": seed}
    base = make_model(
        DummyVecEnv([lambda: BlankEnv(NavigationEnv())]),
        config,
    )
    core = C1ResidualPolicy.from_reference(
        base.policy,
        method="flat",
        leaf_rank=1,
        dictionary_rank=6,
        exploration_epsilon=0.05,
    )
    base.env.close()
    optimizer = core.make_optimizer(learning_rate=0.01, transfer=False)
    return core, optimizer


def test_real_ppo_smoke_freezes_foundation_updates_residuals_and_restores(tmp_path: Path) -> None:
    torch.manual_seed(23)
    np.random.seed(23)
    core, optimizer = _make_smoke_core()
    before_actor = actor_hash(core)
    before_residuals = [parameter.detach().clone() for parameter in core._residual_parameters()]
    router = C1HistoryRouter(window=16, confidence_threshold=0.8)
    config = {
        "learning_rate": 0.01,
        "ent_coef": 0.06,
        "route_window": 16,
        "route_confidence": 0.8,
    }

    metrics = train(
        core,
        optimizer,
        router,
        config,
        [("A0", 256)],
        seed=23,
        output=tmp_path / "ppo_smoke",
        mode="oracle",
    )

    assert metrics["online_steps"] == 256
    assert metrics["optimizer_updates"] > 0
    assert metrics["base_actor_unchanged"] is True
    assert actor_hash(core) == before_actor
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(before_residuals, core._residual_parameters())
    )

    restored_core, restored_optimizer, restored_router = load_bundle(tmp_path / "ppo_smoke" / "agent.pt")
    assert actor_hash(restored_core) == before_actor
    assert restored_optimizer.step_count == optimizer.step_count
    for name, tensor in core.state_dict().items():
        torch.testing.assert_close(restored_core.state_dict()[name], tensor)
    assert restored_router.outcome_counts.sum() == 256
    assert any(parameter in restored_optimizer.state for parameter in restored_core._residual_parameters())


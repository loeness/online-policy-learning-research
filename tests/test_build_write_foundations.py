"""Lightweight checks for foundation data preparation and evaluation budgets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_write_foundations as build_foundations  # noqa: E402


def test_prepare_bc_data_keeps_whole_episode_mod_ten_split(tmp_path: Path) -> None:
    path = tmp_path / "tiny_source.npz"
    observations = np.arange(48, dtype=np.float32).reshape(12, 4) / 48.0
    episodes = np.asarray([0, 0, 1, 1, 2, 2, 9, 9, 10, 10, 11, 11], dtype=np.int32)
    targets = np.full((12, 4), 0.25, dtype=np.float32)
    returns = np.arange(12, dtype=np.float32)
    np.savez_compressed(
        path,
        observations=observations,
        episodes=episodes,
        targets=targets,
        returns=returns,
    )

    prepared = build_foundations._prepare_bc_data(path)

    assert prepared["observations"].shape == (12, 13)
    assert np.array_equal(prepared["observations"][:, :4], observations)
    assert np.count_nonzero(prepared["observations"][:, 4:]) == 0
    assert np.array_equal(prepared["train_ids"], [2, 3, 4, 5, 6, 7, 10, 11])
    assert np.array_equal(prepared["valid_ids"], [0, 1, 8, 9])
    assert len(prepared["dataset_sha256"]) == 64


class _FakeEnv:
    def __init__(self) -> None:
        self.episode_steps = 0
        self.reset_seeds: list[int | None] = []
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        self.episode_steps = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: int):
        assert action == 0
        self.episode_steps += 1
        terminated = self.episode_steps == 3
        return (
            np.zeros(4, dtype=np.float32),
            1.0,
            terminated,
            False,
            {"success": terminated},
        )

    def close(self) -> None:
        self.closed = True


class _FakeModel:
    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, ...]] = []

    def predict(self, observation, deterministic: bool):
        assert deterministic is True
        self.seen_shapes.append(tuple(observation.shape))
        return np.asarray(0), None


def test_fixed_step_evaluation_counts_partial_episode_exactly(monkeypatch) -> None:
    env = _FakeEnv()
    model = _FakeModel()
    monkeypatch.setattr(build_foundations, "_make_evaluation_env", lambda context: env)

    metrics = build_foundations.evaluate_fixed_steps(
        model,
        "normal",
        seed=73,
        steps=5,
    )

    assert metrics["steps"] == metrics["step_budget"] == 5
    assert metrics["completed_episodes"] == 1
    assert metrics["episodes"] == 2
    assert metrics["mean_return"] == 3.0
    assert metrics["success_rate"] == 1.0
    assert metrics["partial_episode_return"] == 2.0
    assert metrics["partial_episode_steps"] == 2
    assert len(model.seen_shapes) == 5
    assert set(model.seen_shapes) == {(13,)}
    assert env.reset_seeds == [73, 74]
    assert env.closed


def test_nonempty_output_is_rejected_before_loading_training_config(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        build_foundations,
        "load_base_config",
        lambda: pytest.fail("must reject the output before loading config"),
    )

    with pytest.raises(FileExistsError, match="nonempty"):
        build_foundations.build_foundations(output, seeds=(17,), updates=1)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"

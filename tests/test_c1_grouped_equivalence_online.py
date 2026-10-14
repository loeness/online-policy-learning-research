"""Gated paired real-PPO parity check for the C1 flat grouping control.

Run only after the Step 2 rollout gate is explicitly opened:

    $env:C1_RUN_GROUPED_PPO_PARITY = "1"
    .\.venv\Scripts\python.exe -m pytest -q tests/test_c1_grouped_equivalence_online.py

The test writes a compact evidence record below outputs/c1_supplement. It is
skipped during ordinary test runs so Step 1 work cannot accidentally trigger
the paired online interactions.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from online_rl.c1_experiment import C1Stream, foundation, new_router  # noqa: E402
from online_rl.c1_modular_controls import FlatGroupedAdditivePolicy  # noqa: E402
from online_rl.c1_policy import C1ResidualPolicy  # noqa: E402
from online_rl.compression import export_payload  # noqa: E402
from online_rl.compression_experiment import PackedSB3Adapter  # noqa: E402
from online_rl.experiment import make_model, seed_everything, write_json  # noqa: E402
from online_rl.policy import tensor_bytes  # noqa: E402


torch.set_num_threads(1)


class _StepRows:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def writerow(self, row: dict[str, Any]) -> None:
        self.rows.append(dict(row))


class _TraceAdapter(PackedSB3Adapter):
    """Keep per-action logits and sampled actions from real PPO collection."""

    def __init__(self, core, optimizer) -> None:
        super().__init__(core, optimizer)
        self.trace: list[dict[str, Any]] = []

    def forward(self, obs, deterministic=False):
        result = super().forward(obs, deterministic=deterministic)
        with torch.no_grad():
            batched = self.core._as_batch(obs)
            latent = self.core.mlp_extractor.forward_actor(self.core.extract_features(batched))
            logits = self.core.action_net(latent) + self.core._correction(batched, latent)
            probabilities = torch.softmax(logits / self.core.temperature, dim=-1)
            if self.core.epsilon:
                probabilities = (1.0 - self.core.epsilon) * probabilities + self.core.epsilon / 4.0
            self.trace.append(
                {
                    "observation": batched.detach().cpu().numpy().copy(),
                    "logits": logits.detach().cpu().numpy().copy(),
                    "probabilities": probabilities.detach().cpu().numpy().copy(),
                    "actions": result[0].detach().cpu().numpy().copy(),
                }
            )
        return result


def _digest(value: Any) -> str:
    digest = hashlib.sha256()

    def feed(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"array\0")
            digest.update(str(array.dtype).encode())
            digest.update(repr(tuple(array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                feed(key)
                feed(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                feed(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode())
            digest.update(b"\0")

    feed(value)
    return digest.hexdigest()


def _assert_policy_equal(left, right) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    assert list(left_state) == list(right_state)
    for name, value in left_state.items():
        assert torch.equal(value, right_state[name]), name


def _assert_optimizer_equal(left_core, left_optimizer, right_core, right_optimizer) -> None:
    left = left_core.optimizer_manifest(left_optimizer)
    right = right_core.optimizer_manifest(right_optimizer)
    assert left["param_groups"] == right["param_groups"]
    assert left["step_count"] == right["step_count"]
    assert left["state"].keys() == right["state"].keys()
    for name, values in left["state"].items():
        assert values.keys() == right["state"][name].keys()
        for key, value in values.items():
            other = right["state"][name][key]
            if torch.is_tensor(value):
                assert torch.equal(value, other), f"{name}.{key}"
            else:
                assert value == other, f"{name}.{key}"


def _checkpoint_summary(core, optimizer, router, phase: str) -> dict[str, Any]:
    payload = export_payload(core, optimizer, router.snapshot(), {"phase": phase})
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    manifest = core.optimizer_manifest(optimizer)
    return {
        "phase": phase,
        "policy_tensor_sha256": _digest(core.state_dict()),
        "optimizer_manifest_sha256": _digest(manifest),
        "serialized_payload_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "policy_tensor_bytes": core.persistent_tensor_bytes(),
        "optimizer_tensor_bytes": tensor_bytes(optimizer.state_dict()),
    }


def _trace_arrays(adapter: _TraceAdapter) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([record[key] for record in adapter.trace], axis=0)
        for key in ("observation", "logits", "probabilities", "actions")
    }


def _run_arm(core, optimizer, base_config: dict[str, Any], seed: int) -> dict[str, Any]:
    schedule = [("A0", 256), ("B0", 256), ("A2", 256), ("B2", 256)]
    router = new_router({"route_window": 32, "route_confidence": 0.8})
    rows = _StepRows()
    stream = C1Stream(schedule, seed=seed + 17, router=router, mode="oracle", steps_writer=rows)
    vec = DummyVecEnv([lambda: stream])

    seed_everything(seed)
    model = make_model(vec, {**base_config, "seed": seed})
    adapter = _TraceAdapter(core, optimizer)
    model.policy = adapter
    stream.model = model

    model.learn(512, reset_num_timesteps=True, progress_bar=False)
    seen_checkpoint = _checkpoint_summary(core, optimizer, router, "seen")
    core.initialize_heldout(3)
    core.initialize_heldout(6)
    heldout_checkpoint = _checkpoint_summary(core, optimizer, router, "heldout_initialized")
    model.learn(512, reset_num_timesteps=False, progress_bar=False)
    final_checkpoint = _checkpoint_summary(core, optimizer, router, "continuation_complete")

    vec.close()
    return {
        "trace": _trace_arrays(adapter),
        "rows": rows.rows,
        "router": router.snapshot(),
        "optimizer_steps": int(optimizer.step_count),
        "seen_checkpoint": seen_checkpoint,
        "heldout_checkpoint": heldout_checkpoint,
        "final_checkpoint": final_checkpoint,
    }


def _phase_summary(rows: list[dict[str, Any]], start: int, stop: int) -> dict[str, Any]:
    selected = [row for row in rows if start < int(row["step"]) <= stop]
    return {
        "steps": len(selected),
        "contexts": sorted({str(row["context"]) for row in selected}),
        "return": float(sum(float(row["reward"]) for row in selected)),
        "completed_successes": sum(int(row["success"]) for row in selected if row["success"] != ""),
        "actions": [int(row["action"]) for row in selected],
        "routes": [int(row["route"]) for row in selected],
    }


@pytest.mark.skipif(
    os.environ.get("C1_RUN_GROUPED_PPO_PARITY") != "1",
    reason="Step 2 paired PPO interactions are gated; set C1_RUN_GROUPED_PPO_PARITY=1 after authorization",
)
def test_tree_and_flat_grouped_control_have_paired_real_ppo_equivalence() -> None:
    supplement = json.loads((ROOT / "configs" / "c1_supplement.json").read_text(encoding="utf-8"))
    base_path = ROOT / supplement["base_config"]
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    seed = int(supplement["validation_seeds"][0])
    temperature = float(supplement["temperature"])
    epsilon = float(supplement["exploration_epsilon"])
    base_config["foundation"] = str(ROOT / base_config["foundation"])
    base_config.update(temperature=temperature, exploration_epsilon=epsilon)

    source = foundation(base_config)
    results: dict[str, Any] = {
        "protocol": "C1 supplement Step 2 paired PPO equivalence",
        "status": "completed",
        "seed": seed,
        "temperature": temperature,
        "exploration_epsilon": epsilon,
        "schedule": [("A0", 256), ("B0", 256), ("A2", 256), ("B2", 256)],
        "phase_steps": {"seen": 512, "heldout_after_initialize": 512},
        "tiers": {},
    }

    for tier_name, rank in (("low", 1), ("high", 2)):
        seed_everything(seed)
        tree = C1ResidualPolicy.from_reference(
            source.policy,
            method="tree",
            leaf_rank=rank,
            exploration_epsilon=epsilon,
            temperature=temperature,
        )
        flat = FlatGroupedAdditivePolicy.from_tree(tree)
        tree_optimizer = tree.make_optimizer(
            learning_rate=float(base_config["learning_rate"]), transfer=False
        )
        flat_optimizer = flat._optimizer_from_manifest(tree.optimizer_manifest(tree_optimizer))
        _assert_policy_equal(tree, flat)

        tree_result = _run_arm(tree, tree_optimizer, base_config, seed)
        flat_result = _run_arm(flat, flat_optimizer, base_config, seed)
        tree_trace, flat_trace = tree_result["trace"], flat_result["trace"]
        for key in ("observation", "actions"):
            assert np.array_equal(tree_trace[key], flat_trace[key]), key
        for key in ("logits", "probabilities"):
            np.testing.assert_allclose(tree_trace[key], flat_trace[key], rtol=0.0, atol=1e-7)

        seen_tree = _phase_summary(tree_result["rows"], 0, 512)
        seen_flat = _phase_summary(flat_result["rows"], 0, 512)
        heldout_tree = _phase_summary(tree_result["rows"], 512, 1024)
        heldout_flat = _phase_summary(flat_result["rows"], 512, 1024)
        assert seen_tree == seen_flat
        assert heldout_tree == heldout_flat
        assert seen_tree["steps"] == heldout_tree["steps"] == 512

        _assert_policy_equal(tree, flat)
        _assert_optimizer_equal(tree, tree_optimizer, flat, flat_optimizer)
        assert tree_result["router"] == flat_result["router"]
        assert tree_result["optimizer_steps"] == flat_result["optimizer_steps"]
        for key in ("policy_tensor_bytes", "optimizer_tensor_bytes"):
            assert tree_result["final_checkpoint"][key] == flat_result["final_checkpoint"][key]

        results["tiers"][tier_name] = {
            "leaf_rank": rank,
            "seen": seen_tree,
            "heldout_continuation": heldout_tree,
            "trace": {
                "action_trace_equal": True,
                "observation_trace_equal": True,
                "max_logit_abs_difference": float(np.max(np.abs(tree_trace["logits"] - flat_trace["logits"]))),
                "max_probability_abs_difference": float(np.max(np.abs(tree_trace["probabilities"] - flat_trace["probabilities"]))),
                "captured_actions": int(tree_trace["actions"].shape[0]),
            },
            "optimizer_steps": tree_result["optimizer_steps"],
            "optimizer_manifest_sha256": tree_result["final_checkpoint"]["optimizer_manifest_sha256"],
            "policy_tensor_sha256": tree_result["final_checkpoint"]["policy_tensor_sha256"],
            "checkpoint_hashes": {
                "tree": {
                    "seen": tree_result["seen_checkpoint"],
                    "heldout_initialized": tree_result["heldout_checkpoint"],
                    "final": tree_result["final_checkpoint"],
                },
                "flat": {
                    "seen": flat_result["seen_checkpoint"],
                    "heldout_initialized": flat_result["heldout_checkpoint"],
                    "final": flat_result["final_checkpoint"],
                },
            },
        }

    source.env.close()
    output = ROOT / supplement["output"] / "step2_grouping" / "equivalence" / "ppo_parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, results)

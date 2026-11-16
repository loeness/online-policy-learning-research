"""Synthetic-only tests for the bounded H033 input adapters."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl import h033_inputs as inputs  # noqa: E402
from online_rl import h033_diagnostic as diagnostic  # noqa: E402
from online_rl import retention_behavioral_candidate as behavioral  # noqa: E402
from online_rl.c1_policy import _contrast_basis  # noqa: E402
from online_rl.policy import SplitNetwork  # noqa: E402
from online_rl.storage_decision import pair_mapping  # noqa: E402
from online_rl.write_decision_policy import decode_actor, encode_actor  # noqa: E402


CONFIG = json.loads((ROOT / "configs" / "h033_diagnostic_v1.json").read_text(encoding="utf-8"))
SEEDS = CONFIG["seeds"]


def _random_obs(rng: np.random.Generator, count: int, route: int = 1) -> np.ndarray:
    obs = np.zeros((count, 13), dtype=np.float32)
    obs[:, 0] = rng.uniform(0.0, 1.0, size=count).astype(np.float32)
    obs[:, 1:4] = rng.normal(0.0, 0.4, size=(count, 3)).astype(np.float32)
    obs[:, -1] = np.float32(route)
    return obs


def _synthetic_inputs(seed: int = SEEDS[0]):
    rng = np.random.default_rng(129)
    state: dict[str, torch.Tensor] = {
        "mlp_extractor.policy_net.0.weight": torch.tensor(rng.normal(0, 0.08, (64, 4)).astype(np.float32)),
        "mlp_extractor.policy_net.0.bias": torch.tensor(rng.normal(0, 0.03, 64).astype(np.float32)),
        "mlp_extractor.policy_net.2.weight": torch.tensor(rng.normal(0, 0.05, (64, 64)).astype(np.float32)),
        "mlp_extractor.policy_net.2.bias": torch.tensor(rng.normal(0, 0.03, 64).astype(np.float32)),
        "action_net.weight": torch.tensor(rng.normal(0, 0.07, (4, 64)).astype(np.float32)),
        "action_net.bias": torch.tensor(rng.normal(0, 0.03, 4).astype(np.float32)),
        "source_basis": torch.tensor(rng.normal(0, 0.025, (3, 65)).astype(np.float32)),
    }
    local = rng.normal(0, 0.06, (2, 27)).astype(np.float32)
    shared = rng.normal(0, 0.04, (1, 9)).astype(np.float32)
    groups = np.asarray([-1, -1], dtype=np.int8)
    encoded = encode_actor(local, shared, groups, codec="int8", budget_bytes=89).data
    decoded_local, decoded_shared, decoded_groups = decode_actor(encoded)
    state["_nodes.1.A"] = torch.tensor(decoded_local[0].reshape(3, 3, 3).copy())
    state["_nodes.2.A"] = torch.tensor(decoded_local[1].reshape(3, 3, 3).copy())
    state["_shared_coefficients.0.C"] = torch.tensor(decoded_shared[0].reshape(3, 3).copy())
    state["slot_groups"] = torch.tensor(decoded_groups.astype(np.int64))

    old_obs = _random_obs(rng, 20, route=7)
    checkpoint = {
        "format": "storage_decision_learner_v1",
        "seed": int(seed),
        "core_state": state,
        "shared_frozen": True,
        "slot_generation": np.asarray([5, 9], dtype=np.int64),
        "fitted_steps": np.asarray([128, 256], dtype=np.uint64),
        "slot_group_tags": np.asarray([1, 1], dtype=np.int8),
        "parent_group": 0,
        "recent_groups": [0, 1, 0, 1, 1, 0, 1, 1],
        "evidence": {"capacity": 256, "groups": {0: {"obs": old_obs}, 1: {"obs": old_obs.copy()}}},
    }

    owners = np.repeat(np.asarray([0, 1], dtype=np.int8), 128)
    owner_slots = np.where(owners == 0, 1, 0).astype(np.int8)
    observations = np.concatenate([_random_obs(rng, 128, route=2), _random_obs(rng, 128, route=1)])
    sidecar = {
        "observations": observations,
        "owner_indices": owners,
        "source_slots": owner_slots,
        "generations": np.where(owner_slots == 0, 5, 9).astype(np.int64),
        "bouts": np.repeat(np.asarray([21, 22], dtype=np.int64), 128),
        "global_steps": np.arange(1000, 1256, dtype=np.int64),
    }
    return checkpoint, encoded, sidecar


class _LightweightFrozenCore(torch.nn.Module):
    """Actor-only shell using the repository's actual policy components."""

    def __init__(self):
        super().__init__()
        self.mlp_extractor = SplitNetwork(features_dim=4)
        self.action_net = torch.nn.Linear(64, 4)
        self.register_buffer("source_basis", torch.zeros((3, 65), dtype=torch.float32))
        self.register_buffer(
            "_contrast_basis", _contrast_basis(dtype=torch.float32, device="cpu"), persistent=False
        )
        self.temperature = 2.0
        self.epsilon = 0.05

    def extract_features(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


def _config() -> dict:
    return copy.deepcopy(CONFIG)


def _synthetic_analysis_rows(*, omit_one: bool = False) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        family = pair_mapping(seed, "dynamics")["families"]["old"]
        history = "old_major" if seed % 2 == 0 else "new_major"
        for future in ("old_major", "new_major", "balanced"):
            for replica in (0, 1):
                for arm in ("keep", "replace_behavioral"):
                    if omit_one and seed == SEEDS[0] and future == "old_major" and replica == 0 and arm == "keep":
                        continue
                    rows.append(
                        {
                            "seed": seed,
                            "case_id": f"s{seed}_dynamics_late",
                            "pair_kind": "dynamics",
                            "state": "late",
                            "family": family,
                            "history_regime": history,
                            "future_regime": future,
                            "replicate": replica,
                            "arm": arm,
                            "horizon": 6144,
                            "reward_per_step": float(seed / 1000 + replica / 100 + (arm == "replace_behavioral") / 10),
                            "action_install": {"post_action_sentinel": "must_not_escape"},
                            "new_source_teacher_kl_after_codec": 987654.0,
                            "utility_rate": -987654.0,
                            "costs_delta": {"sentinel": True},
                        }
                    )
    # Irrelevant report rows are ignored before validation and never returned.
    rows.extend(
        [
            {"arm": "replace_mean", "horizon": 6144, "reward_per_step": 1e99, "post_action": "ignore"},
            {"arm": "keep", "horizon": 1536, "reward_per_step": -1e99, "post_action": "ignore"},
        ]
    )
    return rows


def test_synthetic_saved_prefix_runs_original_fitter_without_forbidden_files(tmp_path: Path):
    checkpoint, actor_data, sidecar = _synthetic_inputs()
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    torch.save(checkpoint, prefix / "checkpoint.pt")
    (prefix / "actor.bin").write_bytes(actor_data)
    np.savez_compressed(prefix / "evidence.npz", **sidecar)
    before = {name: hashlib.sha256((prefix / name).read_bytes()).hexdigest() for name in (
        "checkpoint.pt", "actor.bin", "evidence.npz"
    )}

    row = inputs.extract_prefix_features(prefix, SEEDS[0])

    assert set(row) == {"seed", "old_family", "history_regime", "features", "provenance"}
    assert row["seed"] == SEEDS[0]
    assert set(row["features"]) == set(diagnostic.FEATURE_NAMES)
    assert all(np.isfinite(value) for value in row["features"].values())
    assert row["provenance"]["candidate_fit_diagnostics"]["min_rows_per_owner"] == 16
    assert row["provenance"]["sidecar_rows_by_owner"] == [128, 128]
    assert row["provenance"]["old_evidence_rows"] == 20
    assert not (prefix / "observer_prefix.json").exists()
    assert not (prefix / "simulator.json").exists()
    assert not (prefix / "summary.json").exists()
    after = {name: hashlib.sha256((prefix / name).read_bytes()).hexdigest() for name in before}
    assert after == before


def test_adapter_frozen_features_match_original_extractor_bitwise():
    checkpoint, _, sidecar = _synthetic_inputs()
    state = checkpoint["core_state"]
    core = _LightweightFrozenCore()
    actor_state = {name: value for name, value in state.items() if name in core.state_dict()}
    incompatible = core.load_state_dict(actor_state, strict=False)
    assert set(incompatible.missing_keys) == {
        "mlp_extractor.value_net.0.weight",
        "mlp_extractor.value_net.0.bias",
        "mlp_extractor.value_net.2.weight",
        "mlp_extractor.value_net.2.bias",
    }
    assert incompatible.unexpected_keys == []

    expected = behavioral.extract_frozen_policy_features(core, sidecar["observations"])
    actual = inputs._frozen_features(state, sidecar["observations"])
    for name in ("base_logits", "source_psi", "contrast_basis"):
        assert np.array_equal(getattr(actual, name), getattr(expected, name)), name
    assert actual.temperature == expected.temperature == 2.0
    assert actual.epsilon == expected.epsilon == 0.05


def test_route_neutrality_old_local_clearing_and_equal_owner_weight(monkeypatch):
    checkpoint, actor_data, sidecar = _synthetic_inputs()

    def fixed_fit(observations, owners, matrices, features, *, min_rows_per_owner):
        assert min_rows_per_owner == 16
        assert np.bincount(owners, minlength=2).tolist() == [128, 128]
        return np.zeros((3, 3), dtype=np.float32), {"status": "synthetic"}

    monkeypatch.setattr(behavioral, "fit_behavioral_candidate", fixed_fit)
    original = inputs.extract_from_payload(checkpoint, actor_data, sidecar, SEEDS[0])

    route_changed = copy.deepcopy(checkpoint)
    route_changed["evidence"]["groups"][0]["obs"][:, -1] = np.linspace(
        1.0, 8.0, len(route_changed["evidence"]["groups"][0]["obs"]), dtype=np.float32
    )
    route_neutral = inputs.extract_from_payload(route_changed, actor_data, sidecar, SEEDS[0])
    assert route_neutral["features"] == pytest.approx(original["features"])

    local_changed = copy.deepcopy(checkpoint)
    local = np.stack([local_changed["core_state"][f"_nodes.{slot + 1}.A"].numpy().reshape(27) for slot in range(2)])
    shared = local_changed["core_state"]["_shared_coefficients.0.C"].numpy().reshape(1, 9)
    new_local = local + np.float32(0.08)
    groups = local_changed["core_state"]["slot_groups"].numpy().astype(np.int8)
    changed_actor = encode_actor(new_local, shared, groups, codec="int8", budget_bytes=89).data
    decoded_local, _, _ = decode_actor(changed_actor)
    for slot in range(2):
        local_changed["core_state"][f"_nodes.{slot + 1}.A"] = torch.tensor(
            decoded_local[slot].reshape(3, 3, 3).copy()
        )
    old_prior_only = inputs.extract_from_payload(local_changed, changed_actor, sidecar, SEEDS[0])
    assert old_prior_only["features"]["old_prior_loss"] == pytest.approx(
        original["features"]["old_prior_loss"], abs=1e-12
    )

    assert inputs._owner_equal_mean(np.asarray([1.0, 1.0, 3.0]), np.asarray([0, 0, 1])) == pytest.approx(2.0)


def test_rejects_wrong_seed_recent_group_wrap_and_physical_binding():
    checkpoint, actor_data, sidecar = _synthetic_inputs()
    with pytest.raises(ValueError, match="checkpoint seed"):
        inputs.extract_from_payload(checkpoint, actor_data, sidecar, SEEDS[1])

    checkpoint_bad_recent = copy.deepcopy(checkpoint)
    checkpoint_bad_recent["recent_groups"] = [0, 1, 256, 0, 1, 0, 1, 0]
    with pytest.raises(ValueError, match="recent_groups"):
        inputs.extract_from_payload(checkpoint_bad_recent, actor_data, sidecar, SEEDS[0])

    checkpoint_bad_slots = copy.deepcopy(checkpoint)
    checkpoint_bad_slots["core_state"]["slot_groups"] = torch.tensor([255, -1], dtype=torch.int64)
    with pytest.raises(ValueError, match="slot_groups"):
        inputs.extract_from_payload(checkpoint_bad_slots, actor_data, sidecar, SEEDS[0])


def test_rejects_codec_and_generation_mismatches():
    checkpoint, actor_data, sidecar = _synthetic_inputs()
    local = np.stack([checkpoint["core_state"][f"_nodes.{slot + 1}.A"].numpy().reshape(27) for slot in range(2)])
    local[0, 0] += np.float32(0.1)
    shared = checkpoint["core_state"]["_shared_coefficients.0.C"].numpy().reshape(1, 9)
    groups = checkpoint["core_state"]["slot_groups"].numpy().astype(np.int8)
    mismatched_actor = encode_actor(local, shared, groups, codec="int8", budget_bytes=89).data
    with pytest.raises(ValueError, match="actor.bin local matrices"):
        inputs.extract_from_payload(checkpoint, mismatched_actor, sidecar, SEEDS[0])

    mismatched_sidecar = {key: value.copy() for key, value in sidecar.items()}
    mismatched_sidecar["generations"][0] += 1
    with pytest.raises(ValueError, match="generation"):
        inputs.extract_from_payload(checkpoint, actor_data, mismatched_sidecar, SEEDS[0])


def test_build_packet_uses_exact_seed_prefix_paths_and_strata(monkeypatch, tmp_path: Path):
    producer = tmp_path / "producer"
    producer.mkdir()
    visited: list[Path] = []

    def fake_extract(prefix: Path, seed: int) -> dict:
        expected = producer / "cases" / f"s{seed}_dynamics_late" / "prefix"
        assert prefix == expected
        visited.append(prefix)
        mapping = pair_mapping(seed, "dynamics")
        return {
            "seed": seed,
            "old_family": mapping["families"]["old"],
            "history_regime": "old_major" if seed % 2 == 0 else "new_major",
            "features": {name: 0.0 for name in diagnostic.FEATURE_NAMES},
            "provenance": {"synthetic": True},
        }

    monkeypatch.setattr(inputs, "extract_prefix_features", fake_extract)
    packet = inputs.build_feature_packet(producer, _config())
    assert packet["format"] == "h033_features_v1"
    assert [row["seed"] for row in packet["rows"]] == SEEDS
    assert len(visited) == 16
    assert packet["metadata"]["input_allowlist"] == ["checkpoint.pt", "actor.bin", "evidence.npz"]


def test_outcome_adapter_filters_labels_and_discards_post_action_fields(tmp_path: Path):
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps({"branch_horizon_observations": _synthetic_analysis_rows()}), encoding="utf-8"
    )

    packet = inputs.load_outcome_packet(analysis, _config())

    assert packet["format"] == "h033_outcomes_v1"
    assert len(packet["rows"]) == 16 * 3 * 2 * 2
    assert all(
        set(row) == {"seed", "future_regime", "replica", "arm", "horizon", "reward_per_step"}
        for row in packet["rows"]
    )
    assert all(row["horizon"] == 6144 for row in packet["rows"])
    assert all(row["arm"] in ("keep", "replace_behavioral") for row in packet["rows"])
    assert all("post_action_sentinel" not in json.dumps(row) for row in packet["rows"])
    diagnostic.make_targets(packet, _config())


def test_outcome_adapter_rejects_inconsistent_metadata_and_incomplete_pairs(tmp_path: Path):
    rows = _synthetic_analysis_rows()
    rows[0]["family"] = "not-the-old-family"
    analysis = tmp_path / "bad-family.json"
    analysis.write_text(json.dumps({"branch_horizon_observations": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="old family"):
        inputs.load_outcome_packet(analysis, _config())

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(
        json.dumps({"branch_horizon_observations": _synthetic_analysis_rows(omit_one=True)}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly 192 rows"):
        inputs.load_outcome_packet(incomplete, _config())

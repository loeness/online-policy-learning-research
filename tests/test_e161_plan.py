import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.storage_decision import pair_mapping
from scripts import run_retention_candidate as candidate
from scripts import run_retention_mechanism as mechanism


def _config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_e161_config_is_a_balanced_late_only_plan_with_frozen_counts():
    raw = _config("e161.json")
    config = candidate.validate_config(raw)

    assert config["seed_status"] in {"provisional_pending_root_seed_audit", "frozen"}
    assert len(config["seeds"]) == len(set(config["seeds"])) == 16
    assert config["pair_kinds"] == ["dynamics"]
    assert config["states"] == ["late"]
    assert config["arms"] == ["keep", "replace_mean", "replace_behavioral"]
    assert config["future_regimes"] == ["old_major", "new_major", "balanced"]
    assert config["report"]["primary_future_regimes"] == ["old_major", "new_major"]
    assert config["report"]["confirmatory"] is True
    assert config["report"]["bootstrap_resamples"] == 10_000
    assert config["report"]["bootstrap_seed"] == 2026102102
    assert config["report"]["practical_reward_per_step"] == 0.002
    assert config["execution_order_seed"] == 2026102101
    provenance = candidate._experiment_provenance(config)
    assert provenance["producer_protocol_path"] == "docs/RETENTION_CANDIDATE_PROTOCOL.md"
    assert provenance["experiment_plan_path"] == "experiments/exp161/README.md"
    assert len(provenance["producer_protocol_sha256"]) == 64
    assert len(provenance["experiment_plan_sha256"]) == 64

    expected = {
        "cases": 16,
        "branches": 288,
        "early_prefix_steps": 7168,
        "late_prefix_steps": 29696,
        "prefix_steps": 475136,
        "steps_per_branch": 6144,
        "branch_steps": 1769472,
        "total_steps": 2244608,
    }
    assert candidate._expected_interactions(config) == expected
    assert config["expected_main"] == expected


def test_e161_seed_families_and_history_are_balanced():
    config = candidate.validate_config(_config("e161.json"))
    cells = Counter()
    for seed in config["seeds"]:
        family = pair_mapping(seed, "dynamics")["families"]["old"]
        history = candidate.history_regime_for_seed(seed)
        cells[(family, history)] += 1
    assert cells == Counter({("A", "old_major"): 4, ("A", "new_major"): 4,
                             ("B", "old_major"): 4, ("B", "new_major"): 4})


def test_late_prefix_and_future_regimes_have_expected_bout_counts():
    config = candidate.validate_config(_config("e161.json"))
    seed = config["seeds"][0]
    mapping = pair_mapping(seed, "dynamics")
    prefix = candidate.build_prefix_plan(
        mapping, config, "late", candidate.history_regime_for_seed(seed),
    )
    assert prefix["steps"] == 29696
    assert len(prefix["schedule"]) == 26

    for regime in ("old_major", "new_major", "balanced"):
        rows = mechanism.build_future_schedule(
            mapping, regime, blocks=4, bout_steps=512, start_bout=0,
        )
        assert len(rows) == 12
        assert sum(row["steps"] for row in rows) == 6144
        if regime in ("old_major", "new_major"):
            dominant = 0 if regime == "old_major" else 1
            minority = 1 - dominant
            expected = []
            for block in range(4):
                variant = block % 3
                expected.extend(((minority, variant), (dominant, variant), (dominant, (variant + 1) % 3)))
            assert [(row["group"], row["variant"]) for row in rows] == expected
        else:
            dominant_by_block = [rows[index * 3 + 1]["group"] for index in range(4)]
            assert dominant_by_block == [0, 1, 0, 1]
            assert Counter(row["group"] for row in rows) == Counter({0: 6, 1: 6})


def test_branch_matrix_and_previous_config_validation_are_preserved():
    config = candidate.validate_config(_config("e161.json"))
    branches, _ = candidate._branch_specs(config, "s26201_dynamics_late")
    assert len(branches) == 18
    assert {(row["future_regime"], row["replicate"], row["arm"]) for row in branches} == {
        (future, replicate, arm)
        for future in config["future_regimes"]
        for replicate in config["replicates"]
        for arm in config["arms"]
    }

    old_config = candidate.validate_config(_config("retention_candidate.json"))
    assert old_config["states"] == ["early", "late"]
    assert old_config["arms"] == ["keep", "replace_mean", "replace_behavioral", "merge_mean", "merge_behavioral"]
    assert old_config["future_regimes"] == ["old_major", "new_major"]

    invalid = _config("e161.json")
    invalid["future_regimes"] = ["balanced", "old_major"]
    with pytest.raises(ValueError, match="future_regimes"):
        candidate.validate_config(invalid)


def test_provisional_seed_config_cannot_start_a_run(tmp_path):
    raw = _config("e161.json")
    raw["seed_status"] = "provisional_pending_root_seed_audit"
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="seed_status must be frozen"):
        candidate.run(raw, output)
    assert not output.exists()

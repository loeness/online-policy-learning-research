"""Zero-environment coverage for E161 schedule and confirmatory analysis."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts import audit_retention_candidate as audit
from scripts import report_retention_candidate as report


ROOT = Path(__file__).resolve().parents[1]
E161_SEEDS = [26201, 26202, 26203, 26204, 26205, 26206, 26207, 26208,
              26209, 26210, 26211, 26212, 26213, 26215, 26216, 26218]


def _e161_config() -> dict:
    config = json.loads((ROOT / "configs" / "retention_candidate.json").read_text(encoding="utf-8"))
    config.update({
        "analysis_role": "confirmatory_E161",
        "seeds": list(E161_SEEDS),
        "pair_kinds": ["dynamics"],
        "states": ["late"],
        "history_regime_by_seed": "old_major_if_even_else_new_major",
        "history_regimes": ["old_major", "new_major"],
        "future_regimes": ["old_major", "new_major", "balanced"],
        "arms": ["keep", "replace_mean", "replace_behavioral"],
        "replicates": [0, 1],
        "horizons": [1536, 6144],
    })
    config["report"].update({
        "primary_future_regimes": ["old_major", "new_major"],
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 2026102102,
        "primary_horizon": 6144,
        "practical_reward_per_step": 0.002,
        "primary_contrasts": list(report.E161_PRIMARY_CONTRASTS),
    })
    return config


def _warmup_rows() -> list[dict]:
    rows = []
    for cycle in range(4):
        for group in (1, 0):
            before = f"parent-{cycle}-{group}-before"
            rows.append({
                "cycle": cycle,
                "group": group,
                "write_applied": group == 0,
                "write_rule": "replace_old_group_only_after_pair",
                "parent_binding_before_after": [0, 0],
                "parent_sha256_before_after": [before, before if group == 1 else f"{before}-after"],
                "parent_coefficients_unchanged": group == 1,
                "codec_max_coefficient_error": 0.0 if group == 1 else 0.125,
            })
    return rows


def _e161_summary() -> dict:
    config = _e161_config()
    cases, branches = [], []
    for index, seed in enumerate(E161_SEEDS):
        history = "old_major" if seed % 2 == 0 else "new_major"
        family = "A" if index < 8 else "B"
        case_id = f"s{seed}_dynamics_late"
        fit = {"mean_candidate_sha256": f"mean-{seed}",
               "behavioral_candidate_sha256": f"behavioral-{seed}"}
        prefix = {"prefix_state_hash": f"prefix-{seed}", "candidate_fit": fit,
                  "warmup_pair_records": _warmup_rows()}
        cases.append({"case_id": case_id, "seed": seed, "pair_kind": "dynamics", "state": "late",
                      "history_regime": history, "mapping": {"families": {"old": family}},
                      "prefix": prefix, "errors": []})
        for future in config["future_regimes"]:
            for replicate in config["replicates"]:
                schedule = [{"block": 0, "task": f"{future}-replicate-{replicate}"}]
                for arm in config["arms"]:
                    branches.append({
                        "branch_id": f"{case_id}_{future}_r{replicate}_{arm}",
                        "case_id": case_id,
                        "seed": seed,
                        "pair_kind": "dynamics",
                        "state": "late",
                        "history_regime": history,
                        "future_regime": future,
                        "replicate": replicate,
                        "arm": arm,
                        "future_seed": config["future_seed_offset"] + config["future_seed_multiplier"] * seed + replicate,
                        "future_schedule": schedule,
                        "prefix_state_hash": prefix["prefix_state_hash"],
                        "restored_state_hash": prefix["prefix_state_hash"],
                        "prefix_restore_verified": True,
                        "prefix_bouts_included_in_reward": False,
                        "candidate_fit_reference": fit,
                    })
    return {"format": report.SUMMARY_FORMAT, "phase": "main", "config": config,
            "seeds": list(E161_SEEDS), "case_count": len(cases), "branch_count": len(branches),
            "cases": cases, "branches": branches, "errors": []}


def test_balanced_schedule_alternates_dominant_group_and_rejects_unknown_regimes() -> None:
    config = {"future_blocks": 4, "bout_steps": 512}
    rows = audit._expected_future_schedule(26201, "dynamics", "balanced", 10, config)

    assert len(rows) == 12
    assert [rows[block * 3 + 1]["group"] for block in range(4)] == [0, 1, 0, 1]
    assert sum(row["group"] == 0 for row in rows) == 6
    assert sum(row["group"] == 1 for row in rows) == 6
    with pytest.raises(ValueError, match="unknown future regime"):
        audit._expected_future_schedule(26201, "dynamics", "other", 10, config)


def test_relative_file_resolves_posix_and_windows_manifest_separators(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    target = root / "nested" / "source.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"saved source")

    assert audit._relative_file(root, "nested/source.bin") == target.resolve()
    assert audit._relative_file(root, r"nested\source.bin") == target.resolve()


@pytest.mark.parametrize("relative", [
    "",
    "/outside/file.bin",
    r"\outside\file.bin",
    r"\\server\share\file.bin",
    r"C:\outside\file.bin",
    "C:/outside/file.bin",
    r"C:outside\file.bin",
    "../outside/file.bin",
    r"..\outside\file.bin",
    "nested/../../outside/file.bin",
])
def test_relative_file_rejects_absolute_drive_and_parent_paths(tmp_path: Path, relative: str) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()

    assert audit._relative_file(root, relative) is None


def test_relative_file_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"outside")
    link = root / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    assert audit._relative_file(root, "outside-link/secret.bin") is None


def test_e161_design_uses_configured_late_only_grid_and_balanced_seed_strata() -> None:
    design, cases, branches, families = report._design(_e161_summary())

    assert design["active"]["seeds"] == E161_SEEDS
    assert design["active"]["states"] == ["late"]
    assert design["active"]["arms"] == ["keep", "replace_mean", "replace_behavioral"]
    assert design["active"]["future_regimes"] == ["old_major", "new_major", "balanced"]
    assert len(cases) == 16
    assert len(branches) == 288
    assert {family: list(families.values()).count(family) for family in ("A", "B")} == {"A": 8, "B": 8}
    assert design["expected_counts"]["prefix_steps"] == 475136
    assert design["expected_counts"]["branches"] == 288
    assert design["expected_counts"]["total_steps"] == 2244608


def test_e161_main_estimand_excludes_balanced_and_keeps_seed_level_rows() -> None:
    config = _e161_config()
    config["report"]["bootstrap_resamples"] = 40
    seeds = list(E161_SEEDS)
    families = {seed: ("A" if index < 8 else "B") for index, seed in enumerate(seeds)}
    active = {"seeds": seeds, "pair_kinds": ["dynamics"], "states": ["late"],
              "future_regimes": ["old_major", "new_major", "balanced"], "replicates": [0, 1],
              "arms": list(config["arms"]), "horizons": [1536, 6144]}
    observations, recovery = [], []
    for index, seed in enumerate(seeds):
        jitter = (index % 4 - 1.5) * 0.0001
        for future in active["future_regimes"]:
            d1 = {"old_major": 0.004, "new_major": 0.006, "balanced": -0.5}[future] + jitter
            for replicate in active["replicates"]:
                baseline = seed * 0.000001 + replicate * 0.00001
                values = {"keep": baseline, "replace_mean": baseline + 0.0003,
                          "replace_behavioral": baseline + 0.0003 + d1}
                for horizon in active["horizons"]:
                    for arm, value in values.items():
                        observations.append({"seed": seed, "pair_kind": "dynamics", "state": "late",
                                             "future_regime": future, "replicate": replicate, "arm": arm,
                                             "horizon": horizon, "reward_per_step": value})
                for arm, value in values.items():
                    for population in ("allocated_only", "all_bouts"):
                        recovery.append({"seed": seed, "pair_kind": "dynamics", "state": "late",
                                         "future_regime": future, "replicate": replicate, "arm": arm,
                                         "population": population,
                                         "return_first_256_per_step": value})

    result = report._compute_effects(observations, recovery, families, active, config, "main")
    d1 = next(row for row in result["effects"] if row["horizon"] == 6144
              and row["contrast"] == "replace_behavioral-minus-replace_mean")["reward_per_step"]
    d2 = next(row for row in result["effects"] if row["horizon"] == 6144
              and row["contrast"] == "replace_behavioral-minus-keep")["reward_per_step"]
    balanced = next(row for row in result["future_scoped_effects"] if row["horizon"] == 6144
                    and row["future_regime"] == "balanced"
                    and row["contrast"] == "replace_behavioral-minus-replace_mean")["reward_per_step"]

    assert d1["mean"] == pytest.approx(0.005)
    assert d2["mean"] == pytest.approx(0.0053)
    assert balanced["mean"] == pytest.approx(-0.5)
    assert len(d1["per_seed"]) == 16
    assert result["primary_future_regimes"] == ["old_major", "new_major"]
    assert len(result["recovery_future_effects"]) == 18


def test_stratified_welch_statistics_and_zero_standard_error_are_deterministic() -> None:
    assert report._student_t_quantile(0.95, 7.0) == pytest.approx(1.8945786050900073)
    assert report._student_t_quantile(0.975, 7.0) == pytest.approx(2.3646242515927853)
    families = {seed: ("A" if index < 8 else "B") for index, seed in enumerate(E161_SEEDS)}
    values = {seed: (0.01 + index * 0.001 if index < 8 else 0.03 + index * 0.002)
              for index, seed in enumerate(E161_SEEDS)}
    stat = report._e161_stratified_stat(values, families, resamples=100, bootstrap_seed=18)
    values_a = [values[seed] for seed in E161_SEEDS[:8]]
    values_b = [values[seed] for seed in E161_SEEDS[8:]]
    mean_a = sum(values_a) / 8
    mean_b = sum(values_b) / 8
    var_a = sum((value - mean_a) ** 2 for value in values_a) / 7
    var_b = sum((value - mean_b) ** 2 for value in values_b) / 7
    component_a, component_b = var_a / 8, var_b / 8
    expected_se = math.sqrt(component_a + component_b) / 2
    expected_df = (component_a + component_b) ** 2 / (component_a**2 / 7 + component_b**2 / 7)
    assert stat["mean"] == pytest.approx((mean_a + mean_b) / 2)
    assert stat["standard_error"] == pytest.approx(expected_se)
    assert stat["welch_degrees_freedom"] == pytest.approx(expected_df)
    assert stat["ci95"][0] < stat["mean"] < stat["ci95"][1]
    assert len(stat["bootstrap_ci95_sensitivity"]) == 2
    constant = report._e161_stratified_stat({seed: 0.003 for seed in E161_SEEDS}, families,
                                             resamples=20, bootstrap_seed=18)
    assert constant["standard_error"] == 0.0
    assert constant["lower95_one_sided"] == 0.003
    assert constant["ci95"] == [0.003, 0.003]
    assert constant["welch_degrees_freedom"] is None
    assert constant["welch_degrees_freedom_infinite"] is True
    assert constant["welch_t_statistic_degenerate"] == "positive_infinity"


def test_iut_direction_and_point_threshold_both_gate_advancement() -> None:
    config = _e161_config()

    def effects(mean_d1: float, lower_d1: float, mean_d2: float, lower_d2: float) -> dict:
        rows = []
        for name, mean, lower in zip(report.E161_PRIMARY_CONTRASTS,
                                     (mean_d1, mean_d2), (lower_d1, lower_d2)):
            rows.append({"horizon": 6144, "pair_kind": "dynamics", "state": "late",
                         "contrast": name, "reward_per_step": {
                             "mean": mean, "lower95_one_sided": lower, "ci95": [lower, mean + 0.001],
                             "welch_degrees_freedom": 10.0,
                         }})
        return {"primary_horizon": 6144, "effects": rows}

    passed = report._e161_scientific_decision(effects(0.004, 0.001, 0.006, 0.002), config)
    inconclusive = report._e161_scientific_decision(effects(0.004, 0.001, 0.006, -0.001), config)
    small = report._e161_scientific_decision(effects(0.0019, 0.0005, 0.006, 0.002), config)

    assert passed["status"] == "ADVANCE_TO_FULL_STREAM_VALIDATION"
    assert passed["technical_audit_status"] == "PASS"
    assert passed["intersection_union_direction_pass"] is True
    assert inconclusive["status"] == "INCONCLUSIVE"
    assert inconclusive["intersection_union_direction_pass"] is False
    assert small["status"] == "SMALL_CONDITION_SPECIFIC_GAIN"
    assert small["advancement_threshold_pass"] is False


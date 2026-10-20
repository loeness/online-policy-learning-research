"""Semantic-pairing and provenance tests for the offline candidate reporter."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import report_retention_candidate as report


ROOT = Path(__file__).resolve().parents[1]


def _pilot_config() -> dict:
    config = json.loads((ROOT / "configs" / "retention_candidate.json").read_text(encoding="utf-8"))
    for key, value in config["pilot"].items():
        config[key] = copy.deepcopy(value)
    config["phase"] = "pilot"
    return config


def _warmup_records() -> list[dict]:
    rows = []
    for cycle in range(4):
        for group in (1, 0):
            before = f"parent-{cycle}-{group}-before"
            after = before if group == 1 else f"parent-{cycle}-{group}-after"
            rows.append({
                "cycle": cycle,
                "group": group,
                "write_applied": group == 0,
                "write_rule": "replace_old_group_only_after_pair",
                "parent_binding_before_after": [0, 0],
                "parent_sha256_before_after": [before, after],
                "parent_coefficients_unchanged": group == 1,
                "codec_max_coefficient_error": 0.0 if group == 1 else 0.125,
            })
    return rows


def _pilot_design_summary() -> dict:
    config = _pilot_config()
    seed = int(config["seeds"][0])
    pair = str(config["pair_kinds"][0])
    history = "old_major" if seed % 2 == 0 else "new_major"
    cases = []
    branches = []
    for state in config["states"]:
        case_id = f"s{seed}_{pair}_{state}"
        fit = {
            "mean_candidate_sha256": f"mean-{state}",
            "behavioral_candidate_sha256": f"behavioral-{state}",
            "source_buffer_sha256": f"buffer-{state}",
        }
        prefix = {
            "prefix_state_hash": f"prefix-{state}",
            "candidate_fit": fit,
            "warmup_pair_records": _warmup_records() if state == "late" else [],
        }
        case = {
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair,
            "state": state,
            "history_regime": history,
            "mapping": {"families": {"old": "A", "new": "B"}},
            "prefix": prefix,
        }
        cases.append(case)
        for future in config["future_regimes"]:
            for replicate in config["replicates"]:
                schedule = [{"block": 0, "task": "old-source-0", "future_regime": future}]
                for arm in config["arms"]:
                    branch_id = f"{case_id}_{future}_r{replicate}_{arm}"
                    branches.append({
                        "branch_id": branch_id,
                        "case_id": case_id,
                        "seed": seed,
                        "pair_kind": pair,
                        "state": state,
                        "history_regime": history,
                        "future_regime": future,
                        "replicate": replicate,
                        "arm": arm,
                        "future_seed": int(config["future_seed_offset"]) + int(config["future_seed_multiplier"]) * seed + int(replicate),
                        "future_schedule": schedule,
                        "prefix_state_hash": prefix["prefix_state_hash"],
                        "restored_state_hash": prefix["prefix_state_hash"],
                        "prefix_restore_verified": True,
                        "prefix_bouts_included_in_reward": False,
                        "candidate_fit_reference": {
                            "mean_candidate_sha256": fit["mean_candidate_sha256"],
                            "behavioral_candidate_sha256": fit["behavioral_candidate_sha256"],
                        },
                    })
    return {
        "format": report.SUMMARY_FORMAT,
        "phase": "pilot",
        "config": config,
        "seeds": [seed],
        "case_count": len(cases),
        "branch_count": len(branches),
        "cases": cases,
        "branches": branches,
        "errors": [],
    }


def test_design_rejects_a_missing_arm_condition() -> None:
    summary = _pilot_design_summary()
    summary["branches"].pop()
    summary["branch_count"] -= 1

    with pytest.raises(ValueError, match="branch grid mismatch"):
        report._design(summary)


def test_design_requires_the_same_future_schedule_for_paired_arms() -> None:
    summary = _pilot_design_summary()
    branch = summary["branches"][0]
    branch["future_schedule"] = [{"block": 0, "task": "different-task", "future_regime": "old_major"}]

    with pytest.raises(ValueError, match="future schedule differs across arms"):
        report._design(summary)


def test_effects_average_future_replicas_within_seed_before_bootstrap() -> None:
    seeds = [25201, 25202, 25203, 25204, 25205, 25206]
    families = {seed: "A" if index < 3 else "B" for index, seed in enumerate(seeds)}
    active = {
        "seeds": seeds,
        "pair_kinds": ["dynamics"],
        "states": ["early", "late"],
        "future_regimes": ["old_major", "new_major"],
        "replicates": [0, 1],
        "horizons": [6144],
    }
    arm_effect = {
        "keep": 0.0,
        "replace_mean": 0.01,
        "replace_behavioral": 0.02,
        "merge_mean": 0.03,
        "merge_behavioral": 0.04,
    }
    observations = []
    recovery = []
    for seed in seeds:
        for state_index, state in enumerate(active["states"]):
            for future_index, future in enumerate(active["future_regimes"]):
                for replicate in active["replicates"]:
                    for arm, effect in arm_effect.items():
                        base = seed * 0.0001 + state_index * 0.05 + future_index * 0.02 + replicate * 0.005
                        observations.append({
                            "seed": seed, "pair_kind": "dynamics", "state": state,
                            "future_regime": future, "replicate": replicate, "arm": arm,
                            "horizon": 6144, "reward_per_step": base + effect,
                        })
                        for population in ("allocated_only", "all_bouts"):
                            recovery.append({
                                "seed": seed, "pair_kind": "dynamics", "state": state,
                                "future_regime": future, "replicate": replicate, "arm": arm,
                                "population": population,
                                "return_first_256_per_step": base + effect,
                            })

    result = report._compute_effects(
        observations,
        recovery,
        families,
        active,
        {"report": {"bootstrap_resamples": 100, "bootstrap_seed": 123, "primary_horizon": 6144}},
        "main",
    )
    primary = next(row for row in result["effects"] if row["state"] == "early"
                   and row["contrast"] == "replace_behavioral-minus-replace_mean")
    stat = primary["reward_per_step"]
    assert stat["mean"] == pytest.approx(0.01)
    assert stat["seed_count"] == 6
    assert len(stat["per_seed"]) == 6
    assert [row["value"] for row in stat["per_seed"]] == pytest.approx([0.01] * 6)


def _evidence_case(tmp_path: Path) -> tuple[dict, dict, dict, dict]:
    case_id = "s25189_dynamics_early"
    prefix_folder = tmp_path / "cases" / case_id / "prefix"
    prefix_folder.mkdir(parents=True)
    observations = np.zeros((256, 13), dtype=np.float32)
    band_values = np.asarray([0.1] * 42 + [0.4] * 43 + [0.8] * 43, dtype=np.float32)
    observations[:128, 0] = band_values
    observations[128:, 0] = band_values
    arrays = {
        "observations": observations,
        "owner_indices": np.repeat(np.asarray([0, 1], dtype=np.int8), 128),
        "source_slots": np.repeat(np.asarray([0, 1], dtype=np.int8), 128),
        "generations": np.repeat(np.asarray([7, 8], dtype=np.int64), 128),
        "bouts": np.repeat(np.asarray([5, 6], dtype=np.int64), 128),
        "global_steps": np.concatenate([np.arange(1000, 1128), np.arange(2000, 2128)]).astype(np.int64),
    }
    path = prefix_folder / "evidence.npz"
    np.savez_compressed(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"path": "evidence.npz", "bytes": path.stat().st_size, "sha256": digest,
                "rows": 256, "rows_by_owner": [128, 128]}
    sources = [
        {"slot": 0, "generation": 7, "source_bout": 5, "source_time_fitted_steps": 1024,
         "fitted_steps": 1024, "latest_source_global_step": 1127},
        {"slot": 1, "generation": 8, "source_bout": 6, "source_time_fitted_steps": 1024,
         "fitted_steps": 1024, "latest_source_global_step": 2127},
    ]
    fit = {
        "source_buffer_sha256": digest,
        "source_buffer_bytes": path.stat().st_size,
        "mean_candidate_sha256": "mean-candidate",
        "behavioral_candidate_sha256": "behavioral-candidate",
        "band_counts_by_owner": [[42, 43, 43], [42, 43, 43]],
    }
    observer = {
        "format": "retention_candidate_observer_prefix_v1",
        "accessible_to_deployed_selector": False,
        "included_in_deployment_capacity": False,
        "current_mean_candidate_sha256": "mean-candidate",
        "current_behavioral_candidate_sha256": "behavioral-candidate",
        "candidate_sources": [
            {"slot": 0, "generation": 7, "source_bout": 5, "source_time_fitted_steps": 1024,
             "fitted_steps": 1024, "latest_source_global_step": 1127},
            {"slot": 1, "generation": 8, "source_bout": 6, "source_time_fitted_steps": 1024,
             "fitted_steps": 1024, "latest_source_global_step": 2127},
        ],
        "candidate_fit": {"behavioral_candidate_sha256": "behavioral-candidate"},
        "evidence_buffer": {
            **metadata,
            "source_slots": arrays["source_slots"].astype(np.int64).tolist(),
            "generations": arrays["generations"].astype(np.int64).tolist(),
            "bouts": arrays["bouts"].astype(np.int64).tolist(),
            "global_steps": arrays["global_steps"].astype(np.int64).tolist(),
        },
    }
    (prefix_folder / "observer_prefix.json").write_text(json.dumps(observer), encoding="utf-8")
    artifact = {"path": f"cases/{case_id}/prefix/evidence.npz", "sha256": digest,
                "bytes": path.stat().st_size}
    prefix = {
        "candidate_sources": sources,
        "evidence_buffer": metadata,
        "artifacts": {"evidence": artifact},
    }
    case = {"case_id": case_id, "state": "early", "observer_prefix_path": f"cases/{case_id}/prefix/observer_prefix.json"}
    config = {"evidence_rows_per_source": 128, "early_source_steps": 1024, "late_source_steps": 4096}
    return case, prefix, fit, config


def test_evidence_audit_checks_source_generation_and_band_coverage(tmp_path: Path) -> None:
    case, prefix, fit, config = _evidence_case(tmp_path)

    result = report._audit_evidence_buffer(tmp_path, case, prefix, fit, config)

    assert result["rows"] == 256
    assert result["rows_by_owner"] == [128, 128]
    assert result["band_counts_by_owner"] == [[42, 43, 43], [42, 43, 43]]
    assert result["source_generations_verified"] is True

    prefix["candidate_sources"][0]["generation"] = 9
    with pytest.raises(ValueError, match="do not match owner 0 source"):
        report._audit_evidence_buffer(tmp_path, case, prefix, fit, config)


def test_source_manifest_rejects_machine_local_agent_instructions(tmp_path: Path) -> None:
    summary = {"source_manifest": {
        "excluded_local_instructions": ["Agent.md", "AGENTS.md"],
        "sources": {"nested/AGENTS.md": {"sha256": "unused"}},
    }}

    with pytest.raises(ValueError, match="machine-local agent instructions"):
        report._source_hashes(tmp_path, summary, tmp_path / "summary.json", "summary-hash")


def test_audit_history_preserves_superseded_and_final_record_hashes(tmp_path: Path) -> None:
    run_root = tmp_path / "retention_candidate_pilot"
    snapshot = run_root / "source_snapshot"
    snapshot.mkdir(parents=True)
    manifest_path = snapshot / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    audit_root = tmp_path / "retention_candidate_audit"
    audit_root.mkdir()
    base = {
        "format": "retention_candidate_audit_v1",
        "phase": "pilot",
        "cohort": str(run_root.resolve()),
        "source_manifest_sha256": manifest_sha,
        "coverage": {"cases": 2, "branches": 10, "expected_cases": 2, "expected_branches": 10},
        "raw_counts": {"environment_steps_total": 98304},
        "integrity_errors": [],
        "diagnostic_warnings": [],
        "status": "passed",
    }
    initial_path = audit_root / "pilot.json"
    final_path = audit_root / "pilot_final.json"
    initial_path.write_text(json.dumps(base), encoding="utf-8")
    final_path.write_text(json.dumps(base), encoding="utf-8")
    os.utime(initial_path, (100, 100))
    os.utime(final_path, (200, 200))

    records = report._audit_record_history(
        run_root,
        "pilot",
        {"cases": 2, "branches": 10, "total_steps": 98304},
        manifest_sha,
    )

    assert len(records) == 2
    assert records[0]["superseded_by_later_record"] is True
    assert "not an experiment failure" in records[0]["review_context"]
    assert records[0]["sha256"] == hashlib.sha256(initial_path.read_bytes()).hexdigest()
    assert records[1]["declared_status"] == "passed"
    assert records[1]["superseded_by_later_record"] is False

import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import report_retention_mechanism as report  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {"path": path.relative_to(path.parents[3]).as_posix() if len(path.parents) > 3 else path.name,
              "sha256": _sha(path), "rows": len(rows)}
    if "reward" in fieldnames:
        result["reward_sum"] = sum(float(row["reward"]) for row in rows)
    return result


def _write_ledger(root: Path, relative: str, fieldnames: list[str], rows: list[dict]) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    meta = {"path": relative, "sha256": _sha(path), "rows": len(rows), "exists": True}
    if "reward" in fieldnames:
        meta["reward_sum"] = sum(float(row["reward"]) for row in rows)
    if "updates" in fieldnames:
        meta.update({
            "rollout_count": len(rows),
            "trained_rows": sum(int(row["rows"]) for row in rows),
            "minibatch_updates": sum(int(row["updates"]) for row in rows),
        })
    return meta


def _make_pilot(root: Path) -> Path:
    config = json.loads((ROOT / "configs" / "retention_mechanism.json").read_text(encoding="utf-8"))
    case_id = "s25089_dynamics_old_major"
    prefix_hash = "a" * 64
    prefix_steps = 7168
    prefix_rewards = [{"phase": "prefix", "global_step": i + 1, "reward": 0.0}
                      for i in range(prefix_steps)]
    prefix_rollouts = [{"rows": 256, "updates": 2} for _ in range(prefix_steps // 256)]
    prefix_episodes = [{"episode_id": 0, "return": 0.0}]
    prefix_ledgers = {
        "steps_csv": _write_ledger(root, f"cases/{case_id}/prefix/steps.csv",
                                   ["phase", "global_step", "reward"], prefix_rewards),
        "episodes_csv": _write_ledger(root, f"cases/{case_id}/prefix/episodes.csv",
                                      ["episode_id", "return"], prefix_episodes),
        "rollouts_csv": _write_ledger(root, f"cases/{case_id}/prefix/rollouts.csv",
                                      ["rows", "updates"], prefix_rollouts),
    }
    source_fits = {
        "old_verified": True,
        "candidate_verified": True,
        "verified": True,
        "old_sources": [
            {"task": "old_a", "group": 0, "slot": 0, "generation": 1, "source_time_fitted_steps": 1024,
             "fitted_steps": 1024, "later_same_generation_fitted_steps": 0,
             "task_lifetime_observed_steps": 1024, "effective_matrix": [[0.1, 0.2], [0.3, 0.4]]},
            {"task": "old_b", "group": 0, "slot": 1, "generation": 1, "source_time_fitted_steps": 1024,
             "fitted_steps": 1024, "later_same_generation_fitted_steps": 0,
             "task_lifetime_observed_steps": 1024, "effective_matrix": [[0.2, 0.3], [0.4, 0.5]]},
        ],
        "candidate_sources": [
            {"task": "new_a", "group": 1, "slot": 0, "generation": 2, "source_time_fitted_steps": 1024,
             "fitted_steps": 1024, "later_same_generation_fitted_steps": 0,
             "task_lifetime_observed_steps": 1024, "effective_matrix": [[0.3, 0.4], [0.5, 0.6]]},
            {"task": "new_b", "group": 1, "slot": 1, "generation": 2, "source_time_fitted_steps": 1024,
             "fitted_steps": 1024, "later_same_generation_fitted_steps": 0,
             "task_lifetime_observed_steps": 1024, "effective_matrix": [[0.4, 0.5], [0.6, 0.7]]},
        ],
    }
    prefix_ppo_counts = {
        "rollouts_attempted": 28, "rollouts_completed": 28, "rollouts_failed": 0,
        "minibatch_updates": 56, "partial_minibatch_updates": 0,
    }
    branch_rows = []
    for replicate in (0, 1):
        for action in report.ACTIONS:
            branch_id = f"old_major__r{replicate}__{action}"
            rate = {
                "keep": 0.10 + 0.001 * replicate,
                "replace": 0.11 + 0.001 * replicate,
                "mean": 0.09 + 0.001 * replicate,
                "none": 0.105 + 0.001 * replicate,
            }[action]
            future_steps = [
                {"phase": "future", "action": action, "future_regime": "old_major",
                 "global_step": prefix_steps + i + 1, "reward": rate}
                for i in range(6144)
            ]
            rollouts = [{"rows": 256, "updates": 1} for _ in range(6144 // 256)]
            branch_root = f"cases/{case_id}/branches/{branch_id}"
            future_schedule = [{"bout": i, "steps": 512, "task": "future_task"} for i in range(12)]
            ledgers = {
                "steps_csv": _write_ledger(root, f"{branch_root}/steps.csv",
                                           ["phase", "action", "future_regime", "global_step", "reward"], future_steps),
                "episodes_csv": _write_ledger(root, f"{branch_root}/episodes.csv",
                                              ["episode_id", "return"], []),
                "rollouts_csv": _write_ledger(root, f"{branch_root}/rollouts.csv",
                                              ["rows", "updates"], rollouts),
            }
            horizons = []
            for horizon in (1536, 6144):
                updates = horizon // 256
                horizons.append({
                    "steps": horizon, "reward_sum": rate * horizon, "reward_per_step": rate,
                    "simulation_cpu_s": horizon / 10000, "action_install_cpu_s": 0.01,
                    "utility_rate": rate - 0.02, "raw_environment_steps": horizon,
                    "rollouts_trained": updates, "minibatch_updates": updates,
                })
            branch_rows.append({
                "branch_id": branch_id, "case_id": case_id, "seed": 25089, "pair_kind": "dynamics",
                "history_regime": "old_major", "future_regime": "old_major", "replicate": replicate,
                "action": action, "future_seed": 40000000 + 1000 * 25089 + replicate,
                "future_schedule": future_schedule,
                "prefix_state_hash": prefix_hash, "restored_state_hash": prefix_hash,
                "prefix_restore_verified": True,
                "rng_transform": {"environment_rng_preserved": True, "current_episode_preserved": True},
                "shared_parent_invariant": {"exactly_fixed": True, "max_coefficient_drift": 0.0},
                "prefix_bouts_included_in_reward": False,
                "raw_environment_steps": 6144,
                "ppo_call_counts": {
                    "rollouts_attempted": 24, "rollouts_completed": 24, "rollouts_failed": 0,
                    "minibatch_updates": 24, "partial_minibatch_updates": 0,
                },
                "counts_delta": {"optimizer_step_count": 24, "partial_rollout_rows": 0},
                "raw_ledgers": ledgers, "horizons": horizons,
                "errors": [],
            })
    case = {
        "case_id": case_id, "seed": 25089, "pair_kind": "dynamics", "history_regime": "old_major",
        "mapping": {"pair_kind": "dynamics", "families": {"old": "A", "new": "B"},
                    "preferences": {"old": 1, "new": 1}, "groups": {"old": 0, "new": 1}},
        "prefix": {"steps": prefix_steps, "reward_total": 0.0, "prefix_state_hash": prefix_hash,
                   "source_fits": source_fits, "ppo_call_counts": prefix_ppo_counts,
                   "raw_ledgers": prefix_ledgers},
        "branches": branch_rows, "costs": {}, "errors": [],
    }
    summary = {
        "format": report.SUMMARY_FORMAT, "phase": "pilot", "config": config, "source_manifest": {},
        "seeds": [25089], "case_count": 1, "branch_count": len(branch_rows),
        "cases": [case], "branches": branch_rows, "errors": [], "costs": {"environment_steps": 56320},
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root / "summary.json"


def test_pilot_report_recounts_hashed_ledgers_and_writes_three_panel_exports(tmp_path: Path) -> None:
    source = tmp_path / "pilot"
    source.mkdir()
    _make_pilot(source)

    result = report.analyze(source, tmp_path / "report")

    assert result["audit"]["passed"] is True
    assert result["case_count"] == 1
    assert result["branch_count"] == 8
    assert result["costs_recount"]["total_environment_steps"] == 56320
    assert result["costs_recount"]["branch_rollouts_trained"] == 8 * 24
    assert result["costs_recount"]["raw_ledger_count"] == 3 + 8 * 3
    effect = next(row for row in result["action_effects"]
                  if row["action"] == "replace" and row["horizon"] == 6144)
    assert effect["reward_per_step"]["mean"] == pytest.approx(0.01)
    cross = result["cross_replica_choice"]
    assert cross["stable_past_only_ranking"] is False
    old_future = cross["by_horizon_relation_future"]["dynamics"]["6144"]["old_major"]
    assert old_future["metrics"]["selected_minus_keep_reward_per_step"]["mean"] == pytest.approx(0.01)
    assert old_future["metrics"]["selected_minus_replace_reward_per_step"]["mean"] == pytest.approx(0.0)
    assert len(old_future["per_seed"]) == 1
    assert old_future["per_seed"][0]["family"] == "A"
    assert result["input_hashes"]["raw_ledger_sha256"]
    output = tmp_path / "report"
    for filename in ("analysis.json", "report.md", "figure.png", "figure.pdf", "figure.svg"):
        assert (output / filename).is_file()
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "not be treated as a stable past-only action ranking" in markdown
    assert "utility-rate difference" in markdown
    assert "Per-seed cross-replica comparisons" in markdown
    assert "not estimated for the single-seed pilot" in markdown


def test_corrupted_raw_steps_hash_is_rejected_before_output_creation(tmp_path: Path) -> None:
    source = tmp_path / "pilot"
    source.mkdir()
    summary_path = _make_pilot(source)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    step_path = source / summary["branches"][0]["raw_ledgers"]["steps_csv"]["path"]
    with step_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        report.analyze(source, tmp_path / "report")

    assert not (tmp_path / "report").exists()


def test_main_effects_average_nested_conditions_within_seed_before_bootstrap() -> None:
    seeds = list(report.MAIN_SEEDS)
    families = {seed: ("A" if i < 3 else "B") for i, seed in enumerate(seeds)}
    active = {
        "seeds": seeds, "pair_kinds": ["dynamics"], "history_regimes": ["old_major", "new_major"],
        "future_regimes": ["old_major", "new_major"], "replicates": [0, 1],
        "horizons": [1536],
    }
    config = {"report": {"bootstrap_resamples": 50, "bootstrap_seed": 123,
                         "practical_reward_per_step": 0.002}}
    observations = []
    for seed in seeds:
        for history in active["history_regimes"]:
            for future in active["future_regimes"]:
                gap = 0.005 if future == "new_major" else 0.001
                for replicate in active["replicates"]:
                    for action in report.ACTIONS:
                        rate = 0.10 + (gap if action == "replace" else 0.002 if action == "mean" else -0.001 if action == "none" else 0)
                        observations.append({
                            "seed": seed, "pair_kind": "dynamics", "history_regime": history,
                            "future_regime": future, "replicate": replicate, "action": action,
                            "horizon": 1536, "reward_per_step": rate,
                            "cpu_seconds_per_1000_steps": 1.0, "utility_rate": rate - 0.5,
                        })

    effects, subgroups, interactions = report._effects(observations, families, config, active, "main")

    replace = next(row for row in effects if row["action"] == "replace")
    assert replace["reward_per_step"]["mean"] == pytest.approx(0.003)
    assert replace["reward_per_step"]["seed_count"] == 6
    interaction = interactions[0]
    assert interaction["reward_per_step"]["mean"] == pytest.approx(0.004)
    assert len(subgroups) == 2 * 2 * 3


def test_ties_prefer_keep_and_declared_action_order() -> None:
    action, tied, margin = report._choose({name: 1.0 for name in report.ACTIONS})
    assert action == "keep"
    assert tied == list(report.ACTIONS)
    assert margin == 0
    action, tied, margin = report._choose({"keep": 0.1, "replace": 0.2, "mean": 0.2, "none": 0.0})
    assert action == "replace"
    assert tied == ["replace", "mean"]
    assert margin == 0


def test_interval_spanning_practical_reference_is_marked_inconclusive() -> None:
    stat = {"interval": [0.001, 0.003]}
    result = report._annotate_reward_reference(stat, {"report": {"practical_reward_per_step": 0.002}})
    assert result["practical_interval_interpretation"].startswith("inconclusive:")

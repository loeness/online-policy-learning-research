from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import report_source_followup as report


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_steps(path: Path, reward_rate: float, *, rows: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["phase", "global_step", "bout", "bout_step", "taskobserver", "reward"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(rows):
            writer.writerow({
                "phase": "future",
                "global_step": 1001 + index,
                "bout": index // 256 + 1,
                "bout_step": index % 256 + 1,
                "taskobserver": "A0+",
                "reward": reward_rate,
            })


def _branch(action: str, future: str, reward_rate: float, *, alias: bool = False) -> dict:
    total = reward_rate * 512
    if alias:
        return {
            "format": "storage_source_dose_branch_alias_v1",
            "action": action,
            "future_regime": future,
            "independent_observation": False,
            "reuse_of": f"../../../branches/{action}__{future}/summary.json",
            "logical_environment_steps": 512,
            "environment_steps": 0,
            "logical_reward_total": total,
            "logical_reward_per_step": reward_rate,
            "logical_counts": {"environment_steps": 512, "minibatch_updates": 10},
            "counts": {"environment_steps": 0, "minibatch_updates": 0},
            "actual_costs": {"environment_steps": 0, "cpu_s": 0.01, "wall_s": 0.02},
        }
    return {
        "format": "storage_source_dose_branch_v1",
        "action": action,
        "future_regime": future,
        "environment_steps": 512,
        "reward_total": total,
        "reward_per_step": reward_rate,
        "counts": {"environment_steps": 512, "minibatch_updates": 10},
        "costs": {"cpu_s": 0.1, "wall_s": 0.2},
        "capacity": {"full_persistent_bytes": 220000, "package_budget_bytes": 262144, "actor_bytes": 89},
        "capacity_after_action": {"full_persistent_bytes": 219000, "package_budget_bytes": 262144, "actor_bytes": 89},
        "codec_max_coefficient_error": 0.001,
        "codec_effective_errors": [0.001, 0.002],
        "final_codec_max_coefficient_error": 0.0015,
        "final_codec_effective_errors": [0.0012, 0.0013],
        "prequantized_behavior_drift": {"max_probability_absolute_drift": 0.00001, "max_kl_reference_to_candidate": 0.0000001},
        "postquantized_behavior_drift": {"max_probability_absolute_drift": 0.0002, "max_kl_reference_to_candidate": 0.000003},
    }


def _make_source(root: Path, *, seeds=range(1, 9), missing: tuple[int, str, str, str, str] | None = None) -> Path:
    source = root / "source"
    (source / "cases").mkdir(parents=True)
    cases = []
    for seed in seeds:
        for history in report.REGIMES:
            case_id = f"s{seed}_dynamics_{history}"
            base = []
            low = []
            for fi, future in enumerate(report.REGIMES):
                base_keep = 0.1 + seed * 0.001 + (history == "new_major") * 0.002 + fi * 0.003
                base_d = 0.001 + fi * 0.002
                for action in report.ACTIONS:
                    rate = {
                        "keep": base_keep,
                        "replace": base_keep + base_d,
                        "mean": base_keep + 0.0005,
                        "none": base_keep - 0.0005,
                    }[action]
                    if missing == (seed, history, "old2048", action, future):
                        continue
                    base.append(_branch(action, future, rate))
                    if action == "replace":
                        low.append(_branch(action, future, rate, alias=True))
                    elif action == "none":
                        low.append(_branch(action, future, rate, alias=True))
                    else:
                        low_rate = rate - 0.002 if action == "keep" else rate - 0.001
                        low.append(_branch(action, future, low_rate))
            case_record = {
                "case_id": case_id,
                "seed": seed,
                "pair_kind": "dynamics",
                "history_regime": history,
                "base_branches": base,
                "source_1024_branches": low,
                "selectors": {"frequency": "keep", "recency": "replace", "surrogate_gain": "keep", "surrogate_lower": "replace"},
                "costs": {"prefix_environment_steps": 300, "environment_steps_total": 1000},
                "errors": [],
            }
            cases.append(case_record)
            case_dir = source / "cases" / case_id
            _write_json(case_dir / "decision.json", {
                "selectors": case_record["selectors"],
                "selector_input_candidate_set": "old_candidate_2048",
                "prefix_state_hash": "prefix-hash-001",
            })
            _write_json(case_dir / "dose_1024" / "decision.json", {
                "selectors": case_record["selectors"],
                "selector_input_candidate_set": "old_candidate_2048",
            })
            _write_json(case_dir / "dose_1024" / "observer_candidate.json", {
                "format": "storage_source_dose_observer_v1",
                "source_capture_steps": 256,
                "capture_timing": "after PPO update, int8 actor roundtrip, and actor optimizer-state clear",
                "candidate_coefficients": 9,
                "sources": {
                    "0": {"task": "A0+", "bout_step": 256, "global_environment_steps": 256, "fitted_steps": 256, "rollouts_trained": 1, "minibatch_updates": 4},
                    "1": {"task": "A1+", "bout_step": 256, "global_environment_steps": 512, "fitted_steps": 256, "rollouts_trained": 2, "minibatch_updates": 8},
                },
            })
            _write_json(case_dir / "prefix" / "summary.json", {"capacity": {"full_persistent_bytes": 220000, "package_budget_bytes": 262144}})
            for b in base:
                _write_json(case_dir / "branches" / f"{b['action']}__{b['future_regime']}" / "summary.json", {
                    **b,
                    "capacity": {"full_persistent_bytes": 220000, "package_budget_bytes": 262144, "actor_bytes": 89},
                })
                _write_steps(case_dir / "branches" / f"{b['action']}__{b['future_regime']}" / "steps.csv", b["reward_per_step"])
            for b in low:
                branch_path = case_dir / "dose_1024" / "branches" / f"{b['action']}__{b['future_regime']}" / "summary.json"
                _write_json(branch_path, b)

    summary = {
        "format": report.SOURCE_FORMAT,
        "phase": "main",
        "config": {
            "seeds": list(seeds),
            "pair_kinds": ["dynamics"],
            "history_regimes": list(report.REGIMES),
            "future_regimes": list(report.REGIMES),
            "actions": list(report.ACTIONS),
            "package_budget_bytes": 262144,
            "report": {"bootstrap_seed": 20261018},
        },
        "seeds": list(seeds),
        "cases": cases,
        "errors": [],
        "costs": {
            "prefix_environment_steps": 4800,
            "environment_steps_total": 16000,
            "logical_environment_steps_total": 20000,
            "branch_environment_steps": 11200,
            "cpu_s": 100.0,
            "wall_s": 80.0,
        },
    }
    _write_json(source / "summary.json", summary)
    _write_json(source / "config.json", summary["config"])
    return source


def _make_ablation(root: Path, source: Path, *, omit_audit: bool = False) -> Path:
    out = root / "ablation"
    cases = []
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    for source_case in source_summary["cases"]:
        case_id = source_case["case_id"]
        seed = source_case["seed"]
        history = source_case["history_regime"]
        branches = []
        for future in report.REGIMES:
            for action in ("replace", "none"):
                future_index = report.REGIMES.index(future)
                train_gap = 0.001 + future_index * 0.002 + 0.0005
                frozen_rate = 0.05 + seed * 0.001
                no_gradient_gap = train_gap - (0.004 + future_index * 0.004)
                branch = {
                    "learning_mode": "no_gradient",
                    "source_case_id": case_id,
                    "prefix_state_hash": "prefix-hash-001",
                    "action": action,
                    "future_regime": future,
                    "environment_steps": 512,
                    "reward_total": (frozen_rate + (no_gradient_gap if action == "replace" else 0.0)) * 512,
                    "counts": {"environment_steps": 512, "minibatch_updates": 0},
                    "minibatch_updates": 0,
                    "fitted_steps_not_increased": not omit_audit,
                    "matched_ppo_branch": {"source_case_id": case_id, "action": action, "future_regime": future, "summary": f"branches/{action}__{future}/summary.json", "summary_sha256": "abc"},
                    "costs": {"cpu_s": 0.01, "wall_s": 0.02},
                }
                if not omit_audit:
                    branch.update({"critic_unchanged": True, "w0_unchanged": True})
                branches.append(branch)
                _write_steps(
                    out / "cases" / case_id / "branches" / f"{action}__{future}" / "steps.csv",
                    branch["reward_total"] / branch["environment_steps"],
                )
        cases.append({
            "case_id": case_id,
            "source_case_id": case_id,
            "seed": seed,
            "pair_kind": source_case["pair_kind"],
            "history_regime": history,
            "prefix_state_hash": "prefix-hash-001",
            "branch_results": branches,
        })
    _write_json(out / "summary.json", {
        "format": "learning_interaction_ablation_v1",
        "phase": "pilot",
        "learning_mode": "no_gradient",
        "cases": cases,
        "case_count": len(cases),
        "branch_count": len(cases) * 4,
        "physical_branch_count": len(cases) * 4,
        "errors": [],
        "costs": {
            "new_prefix_environment_steps": 0,
            "new_branch_environment_steps": len(cases) * 4 * 512,
            "new_environment_steps_total": len(cases) * 4 * 512,
            "source_reference_prefix_environment_steps": len(cases) * 300,
            "source_reference_branch_environment_steps": len(cases) * 8 * 512,
            "source_reference_environment_steps_total": len(cases) * 4 * 300 + len(cases) * 8 * 512,
            "cpu_s": 25,
            "wall_s": 20,
            "branch_cpu_s": 24,
            "branch_wall_s": 19,
        },
    })
    return out


def test_source_estimands_use_alias_logical_reward_and_seed_clusters(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    analysis = report.analyze_source(source)
    result = analysis["relations"]["dynamics"]["estimands"]
    assert result["B"]["all"]["n_seeds"] == 8
    assert result["B"]["all"]["mean"] == pytest.approx(0.002)
    assert result["B"]["all"]["predeclared_classification"].startswith("supports")
    assert result["Q"]["mean"] == pytest.approx(0.0)
    assert result["preference_reversal_by_dose"]["old1024"]["preference_reversal"] is False
    assert analysis["costs_and_integrity"]["verified_alias_branch_count_observed"] == 64
    assert analysis["costs_and_integrity"]["physical_future_branch_count_observed"] == 192
    assert analysis["replace_dose_reuse"]["all_observed_replace_branches_reused"] is True
    assert analysis["replace_dose_reuse"]["alias_old1024_replace_branches"] == 8 * 2 * 2
    assert analysis["source_capture"]["candidate_float32_bytes_per_case"] == 36
    rates = analysis["relations"]["dynamics"]["mean_action_reward_rates"]
    assert set(rates["old2048"]) == set(report.ACTIONS)
    assert rates["old2048"]["keep"]["old_major"]["mean"] == pytest.approx(0.1055)
    assert rates["old1024"]["keep"]["old_major"]["mean"] == pytest.approx(0.1035)
    drift = analysis["codec_probability_drift"]["doses"]
    assert drift["old2048"]["replace"]["physical_branch_count"] == 8 * 2 * 2
    assert drift["old2048"]["replace"]["postquantized_max_probability_absolute_drift"]["p95"] == pytest.approx(0.0002)
    assert drift["old1024"]["replace"]["physical_branch_count"] == 0
    # A 1024 map based on the 2048 candidate cannot support the dose-specific
    # surrogate claims, while its history-only rules remain valid.
    low = analysis["relations"]["dynamics"]["selectors"]["old1024"]
    assert low["frequency"]["primary"]["keep"]["available"] is True
    assert low["surrogate_gain"].get("available") is False


def test_missing_branch_is_counted_and_not_imputed_as_zero(tmp_path: Path) -> None:
    source = _make_source(tmp_path, missing=(1, "old_major", "old2048", "keep", "old_major"))
    analysis = report.analyze_source(source)
    result = analysis["relations"]["dynamics"]["estimands"]["B"]["all"]
    assert result["n_seeds"] == 7
    assert analysis["relations"]["dynamics"]["incomplete_or_missing_seeds"] == [1]
    assert any("missing keep/old_major logical branch" in issue for case in analysis["case_integrity"] for issue in case["issues"])


def test_mechanism_step_reader_rejects_prefix_rows(tmp_path: Path) -> None:
    path = tmp_path / "steps.csv"
    path.write_text(
        "phase,global_step,bout,taskobserver,reward\nprefix,1,0,A0+,0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="future-only"):
        report._read_future_trace(path)


def test_write_report_creates_nonoverwriting_png_pdf_and_json(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "report"
    report.write_report(source, output)
    for name in ("analysis.json", "report.md", "source_followup_estimands.png", "source_followup_estimands.pdf"):
        assert (output / name).is_file()
        assert (output / name).stat().st_size > 0
    with pytest.raises(FileExistsError):
        report.write_report(source, output)


def test_ablation_T_uses_paired_seed_and_rejects_missing_no_gradient_audit(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    ablation = _make_ablation(tmp_path, source)
    analysis = report.write_report(source, tmp_path / "report-with-ablation", ablation=ablation)
    t = analysis["ablation"]["relations"]["dynamics"]["T"]["all_future"]["old_major"]
    assert t["n_seeds"] == 8
    assert t["mean"] == pytest.approx(0.004)
    assert t["practical_classification"] == "interval entirely above +0.002"
    balanced = analysis["ablation"]["relations"]["dynamics"]["T"]["all_future"]["balanced"]
    assert balanced["n_seeds"] == 8
    assert balanced["mean"] == pytest.approx(0.006)
    assert balanced["S_train"]["mean"] - balanced["S_frozen"]["mean"] == pytest.approx(balanced["mean"])
    assert analysis["ablation"]["relations"]["dynamics"]["T"]["all_future"]["new_major"]["mean"] == pytest.approx(0.008)
    costs = analysis["ablation"]["costs"]
    assert costs["new_prefix_environment_steps"] == 0
    assert costs["new_future_environment_steps"] == 8 * 2 * 4 * 512
    assert costs["new_logical_environment_steps"] == 8 * 2 * 4 * 512
    assert costs["source_ppo_reference_branch_count"] == 8 * 2 * 4
    assert costs["source_ppo_reference_environment_steps"] == 8 * 2 * 4 * 512
    assert costs["source_ppo_reference_cpu_s"] == pytest.approx(8 * 2 * 4 * 0.1)
    assert costs["source_ppo_reference_wall_s"] == pytest.approx(8 * 2 * 4 * 0.2)
    assert costs["merged_actual_environment_steps"] == 16000 + 8 * 2 * 4 * 512
    assert costs["merged_logical_environment_steps"] == 20000 + 8 * 2 * 4 * 512
    report_text = (tmp_path / "report-with-ablation" / "report.md").read_text(encoding="utf-8")
    assert "Report-generation environment steps: 0" in report_text
    assert "after PPO update, int8 actor roundtrip" in report_text
    assert "Ablation incremental CPU / wall seconds | 25 / 20" in report_text
    assert "All 32 observed 1024-step `replace` branches are hash-verified aliases" in report_text
    mechanism = analysis["ablation"]["mechanism"]
    assert mechanism["prefix_included"] is False
    assert mechanism["future_step_logs_read"] == 8 * 2 * 2 * 4
    assert mechanism["relations"]["dynamics"]["early_steps"]["old_major"]["256"]["T"]["mean"] == pytest.approx(0.004)
    assert mechanism["relations"]["dynamics"]["per_bout"]["old_major"][0]["T"]["mean"] == pytest.approx(0.004)
    assert not mechanism["issues"]
    assert (tmp_path / "report-with-ablation" / "learning_interaction_T.pdf").is_file()
    assert (tmp_path / "report-with-ablation" / "learning_mechanism_curves.pdf").is_file()

    bad_ablation = _make_ablation(tmp_path / "bad", source, omit_audit=True)
    bad = report._analyze_ablation(source, bad_ablation)
    assert bad["issues"]
    assert bad["relations"]["dynamics"]["T"]["all_future"]["old_major"]["n_seeds"] == 0

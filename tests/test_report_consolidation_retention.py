from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_consolidation_retention import (  # noqa: E402
    analyze_summary,
    build_report_markdown,
    generate_report,
    main,
)


SEEDS = [1009, 1013, 1019]
METHODS = ["local_only", "mean", "weighted", "random"]
TARGETS = ["A0+", "A2+", "B2+", "A0-"]


def _summary(*, failed_audit: bool = True) -> dict[str, object]:
    seed_offsets = {1009: -20.0, 1013: 15.0, 1019: 80.0}
    method_offsets = {"local_only": 0.0, "mean": 4.0, "weighted": 1.0, "random": 0.5}
    records = []
    for seed in SEEDS:
        for method in METHODS:
            targets = {}
            for task_index, task in enumerate(TARGETS):
                base = seed_offsets[seed] + method_offsets[method] + task_index * 2.0
                episode_returns = [base + episode * 0.25 for episode in range(8)]
                negative = task.endswith("-")
                targets[task] = {
                    "episode_returns": episode_returns,
                    "episode_lengths": [10 + episode for episode in range(8)],
                    # A negative-target goal is deliberately not positive success.
                    "goal_flags": [True] * 8 if negative else [episode % 2 == 0 for episode in range(8)],
                    "positive_success_flags": [False] * 8 if negative else [episode % 2 == 0 for episode in range(8)],
                    "mean_signed_return": sum(episode_returns) / 8,
                    "mean_goal_rate": 1.0 if negative else 0.5,
                    "positive_success_rate": 0.0 if negative else 0.5,
                    "evaluation_steps": 80,
                }
            purge_passed = not (failed_audit and seed == 1013 and method == "weighted")
            records.append(
                {
                    "seed": seed,
                    "method": method,
                    "path": f"seed_{seed}/{method}",
                    "proposal": {"kind": method},
                    "invariance": {
                        "raw_logit_max_error": 0.0,
                        "raw_probability_max_error": 1e-8,
                        "int8_logit_max_error": 0.01,
                        "int8_mean_KL": 0.001,
                    },
                    "purge_audit": {
                        "passed": purge_passed,
                        "checks": {"proposal_deleted": purge_passed, "source_preserved": True},
                    },
                    "capacity": {
                        "actor_bytes": 230,
                        "checkpoint_bytes": 500,
                        "foundation_bytes": 100,
                        "full_persistent_bytes": 600,
                        "package_budget_bytes": 4096,
                    },
                    "targets": targets,
                    "cpu_s": 0.02,
                    "wall_s": 0.1,
                    "actor_frozen_during_eval": True,
                    "source_unchanged": True,
                }
            )
    return {
        "format": "consolidation_retention_stage1_v1",
        "status": "completed",
        "seeds": SEEDS,
        "methods": METHODS,
        "targets": TARGETS,
        "records": records,
        "cohort_cost": {
            "source_training_steps_reused": 98304,
            "new_training_steps": 0,
            "new_evaluation_steps": 960,
            "new_total_interactions": 960,
        },
        "audit": {
            "passed": not failed_audit,
            "failures": ["synthetic source audit failure"] if failed_audit else [],
        },
        "source_audits": [{"name": "synthetic manifest", "passed": not failed_audit}],
        "analysis_role": "exploratory",
    }


def _write_summary(root: Path, summary: dict[str, object]) -> Path:
    root.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def _postrun_audit(summary: dict[str, object]) -> dict[str, object]:
    records = summary["records"]
    capacity_records = []
    for record in records:
        capacity = record["capacity"]
        capacity_records.append(
            {
                "seed": record["seed"],
                "method": record["method"],
                "checkpoint_bytes": capacity["checkpoint_bytes"],
                "foundation_bytes": capacity["foundation_bytes"],
                "actor_sidecar_bytes": capacity["actor_bytes"],
                "original_full_persistent_bytes": capacity["full_persistent_bytes"],
                "full_physical_bytes": (
                    capacity["checkpoint_bytes"]
                    + capacity["foundation_bytes"]
                    + capacity["actor_bytes"]
                ),
                "package_budget_bytes": capacity["package_budget_bytes"],
                "passed": True,
            }
        )
    return {
        "format": "consolidation_retention_postrun_audit_v1",
        "passed": True,
        "capacity_correction": {
            "scope": "W0 + checkpoint + generated actor sidecar; observer logs excluded",
            "raw_summary_retained_unchanged": True,
            "records": capacity_records,
        },
        "cohort_cost": {
            "main_new_evaluation_steps": 227,
            "pilot_new_evaluation_steps": 33,
            "total_new_interactions": 260,
            "new_training_steps": 0,
            "main_episodes": 24,
        },
        "source_cost": {
            "historical_training_steps_reused": 200,
            "historical_probe_steps_reused": 30,
            "historical_evaluation_steps_reused": 40,
            "historical_total_interactions_reused": 270,
        },
        "pairing": {
            "reset_and_action_seeds_match": True,
            "common_uniform_rows": 1234,
            "uniform_mismatches": 0,
        },
        "integrity": {
            "snapshot_files_verified": 24,
            "checkpoints_verified": 12,
            "source_files_verified": 3,
        },
        "checkpoint_contract": {
            "runtime_frozen_and_hash_unchanged": True,
            "standalone_restore_must_explicitly_freeze": True,
            "note": "freeze after restore",
        },
    }


def test_analysis_aggregates_seed_means_and_keeps_all_paired_seed_contrasts():
    analysis = analyze_summary(_summary(failed_audit=False))
    a0 = next(row for row in analysis["targets"] if row["target"] == "A0+")
    weighted = next(row for row in a0["method_summaries"] if row["method"] == "weighted")
    returns = weighted["metrics"]["mean_signed_return"]
    expected_seed_means = [
        sum(seed_offset + 1.0 + episode * 0.25 for episode in range(8)) / 8
        for seed_offset in (-20.0, 15.0, 80.0)
    ]
    assert len(returns["per_seed"]) == 3
    assert [row["value"] for row in returns["per_seed"]] == pytest.approx(expected_seed_means)
    assert returns["mean_across_seeds"] == pytest.approx(sum(expected_seed_means) / 3)
    assert returns["min_across_seeds"] == pytest.approx(min(expected_seed_means))
    assert returns["max_across_seeds"] == pytest.approx(max(expected_seed_means))

    contrast = next(row for row in a0["paired_contrasts"] if row["name"] == "weighted_minus_local_only")
    return_differences = contrast["metrics"]["mean_signed_return"]
    assert [row["seed"] for row in return_differences["per_seed"]] == SEEDS
    assert [row["difference"] for row in return_differences["per_seed"]] == pytest.approx([1, 1, 1])
    assert return_differences["mean_across_seeds"] == pytest.approx(1.0)
    assert analysis["interpretation"]["unit"].startswith("seed;")


def test_negative_target_goal_flags_do_not_count_as_positive_success():
    analysis = analyze_summary(_summary(failed_audit=False))
    negative = next(row for row in analysis["targets"] if row["target"] == "A0-")
    local = next(row for row in negative["method_summaries"] if row["method"] == "local_only")

    assert local["metrics"]["mean_goal_rate"]["mean_across_seeds"] == 1.0
    assert local["metrics"]["positive_success_rate"]["mean_across_seeds"] == 0.0


def test_markdown_surfaces_failed_audits_and_exploratory_limits():
    analysis = analyze_summary(_summary(failed_audit=True))
    report = build_report_markdown(analysis)

    assert analysis["audit"]["summary_passed"] is False
    assert analysis["audit"]["purge_failure_count"] == 1
    assert "synthetic source audit failure" in report
    assert "seed 1013 / `weighted`" in report
    assert "不做显著性检验" in report
    assert "删除全部局部修正，仅剩 W0 + 零共享" in report
    assert "A0+、A2+" in report and "6/6" in report
    assert "Mean − W0 only" in report and "Evidence weighted − W0 only" in report
    assert "不用于选择算法" in report
    assert "A2 source 未见过，但 A2+ 用作 exploratory validation" in report
    assert "B2+ 与 A0-" in report
    assert "positive_success_rate 在 A0− 没有解释意义" in report
    assert "Frobenius 范数匹配 `weighted`" in report


def test_generate_report_writes_analysis_markdown_png_and_pdf(tmp_path):
    summary = _summary(failed_audit=True)
    input_dir = _write_summary(tmp_path / "input", summary)
    postrun_audit = _postrun_audit(summary)
    (input_dir / "postrun_audit.json").write_text(
        json.dumps(postrun_audit), encoding="utf-8"
    )
    output_dir = tmp_path / "report"

    assert main(["--input", str(input_dir), "--output", str(output_dir)]) == 0

    assert {path.name for path in output_dir.iterdir()} == {
        "analysis.json",
        "report.md",
        "figure.png",
        "figure.pdf",
    }
    report_data = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    assert report_data["format"] == "consolidation_retention_analysis_v1"
    assert report_data["audit"]["purge_failure_count"] == 1
    assert report_data["postrun_audit"] == postrun_audit
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "## 事后物理文件核算" in markdown
    assert "原日志 Full persistent bytes" in markdown
    assert "Full physical bytes" in markdown
    assert "新增交互 260" in markdown
    assert "历史 source 交互复用 270" in markdown
    assert "共同 uniform rows 1234，mismatches 0" in markdown
    assert "verified snapshots 24" in markdown
    assert "runner 实际 freeze 且 hash 保持不变为 是" in markdown
    assert (output_dir / "figure.png").stat().st_size > 1000
    assert (output_dir / "figure.pdf").stat().st_size > 1000


def test_postrun_capacity_audit_must_match_original_capacity_and_total(tmp_path):
    summary = _summary(failed_audit=False)
    audit = _postrun_audit(summary)
    audit["capacity_correction"]["records"][0]["original_full_persistent_bytes"] = 601

    with pytest.raises(ValueError, match="original_full_persistent_bytes does not match"):
        analyze_summary(summary, postrun_audit=audit)


def test_generate_report_refuses_nonempty_output_directory(tmp_path):
    input_dir = _write_summary(tmp_path / "input", _summary(failed_audit=False))
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="nonempty report directory"):
        generate_report(input_dir, output_dir)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert list(output_dir.iterdir()) == [marker]

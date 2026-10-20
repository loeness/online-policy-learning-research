"""Offline encoding diagnosis for a completed audited retention-candidate run."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
ARMS = ("keep", "replace_mean", "replace_behavioral", "merge_mean", "merge_behavioral")
CONTRASTS = (
    ("replace_behavioral_minus_replace_mean", "replace_behavioral", "replace_mean"),
    ("replace_behavioral_minus_keep", "replace_behavioral", "keep"),
    ("replace_mean_minus_keep", "replace_mean", "keep"),
    ("merge_behavioral_minus_merge_mean", "merge_behavioral", "merge_mean"),
    ("merge_behavioral_minus_keep", "merge_behavioral", "keep"),
    ("merge_mean_minus_keep", "merge_mean", "keep"),
)

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None

def _mean(values: Sequence[Any]) -> float | None:
    nums = [float(value) for value in values if _num(value) is not None]
    return float(np.mean(nums)) if nums else None

def _mean_tree(values: Sequence[Any]) -> Any:
    """Average numeric leaves while retaining saved owner/resident arrays."""
    first = values[0]
    if isinstance(first, Mapping):
        return {key: _mean_tree([value.get(key) for value in values]) for key in first}
    if isinstance(first, list):
        return [_mean_tree([value[index] for value in values]) for index in range(len(first))]
    if _num(first) is not None:
        return _mean(values)
    return first

def _stat(values: Sequence[Any]) -> dict[str, Any]:
    nums = [float(value) for value in values if _num(value) is not None]
    return {"n": len(nums), "missing": len(values) - len(nums),
            "mean": float(np.mean(nums)) if nums else None, "median": float(np.median(nums)) if nums else None,
            "min": float(np.min(nums)) if nums else None, "max": float(np.max(nums)) if nums else None}

def _spectra(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.isfinite(matrix).all():
        raise ValueError("training source_psi must be finite with three columns")
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    cond = float(singular[0] / singular[-1]) if rank == 3 and singular[-1] > 0 else None
    return {"rows": len(matrix), "rank": rank, "rank_deficient": rank < matrix.shape[1],
            "singular_values": [float(value) for value in singular], "condition_number_2": cond}

def _conditioning(run_root: Path, case: Mapping[str, Any], config: Mapping[str, Any],
                  manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    # These audited helpers recreate only the saved frozen policy core/features.
    from scripts import audit_retention_candidate as audit
    from online_rl import retention_behavioral_candidate as behavioral
    case_id = str(case["case_id"])
    prefix = run_root / "cases" / case_id / "prefix"
    checkpoint_path, evidence_path = prefix / "checkpoint.pt", prefix / "evidence.npz"
    checkpoint = audit._checkpoint(checkpoint_path)
    with np.load(evidence_path, allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        owners = np.asarray(data["owner_indices"], dtype=np.int64)
    if owners.shape != (len(observations),):
        raise ValueError(f"owner/evidence mismatch in {case_id}")
    features = behavioral.extract_frozen_policy_features(
        audit._instantiate_core(checkpoint, config, manifest), observations,
    )
    psi = np.asarray(features.source_psi, dtype=np.float64)
    behavior_fit = case["candidate_fit"]["behavioral"]
    used = behavior_fit["rows_used_by_owner"]
    train_rows: list[np.ndarray] = []
    owner_spectra = []
    for owner in (0, 1):
        rows = np.flatnonzero(owners == owner)
        selected = rows[-int(used[owner]):] if int(used[owner]) else rows[:0]
        split = min(int(math.floor(0.75 * len(selected))), max(0, len(selected) - 1))
        train = selected[:split] if len(selected) > 1 else selected
        train_rows.append(train)
        owner_spectra.append(_spectra(psi[train]))
    pooled = np.concatenate(train_rows)
    result = {
        "source_psi_train_pooled": _spectra(psi[pooled]),
        "source_psi_train_by_owner": owner_spectra,
        "source_psi_rank_saved": behavior_fit.get("source_psi_rank"),
    }
    hashes = {"checkpoint_sha256": _sha(checkpoint_path), "evidence_sha256": _sha(evidence_path)}
    return result, hashes

def _action(raw: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    saved = raw.get("action_install", {})
    before = np.asarray(saved.get("pre_action_parent"), dtype=np.float64)
    precodec = np.asarray(saved.get("prequantized_parent"), dtype=np.float64)
    postcodec = np.asarray(saved.get("postquantized_parent"), dtype=np.float64)
    if any(array.shape != (3, 3) or not np.isfinite(array).all() for array in (before, precodec, postcodec)):
        raise ValueError(f"bad saved parent coefficients in {raw.get('branch_id')}")
    kl_before, kl_after = diagnostic.get("new_source_teacher_kl_before_codec"), diagnostic.get("new_source_teacher_kl_after_codec")
    pre_drift, post_drift = diagnostic.get("prequantized_behavior_drift"), diagnostic.get("postquantized_behavior_drift")
    pre_mean = kl_before.get("mean") if isinstance(kl_before, Mapping) else None
    post_mean = kl_after.get("mean") if isinstance(kl_after, Mapping) else None
    return {
        "candidate_norm": _num(saved.get("candidate_norm")),
        "pre_action_parent_norm": _num(saved.get("pre_action_parent_norm")),
        "prequantized_parent_norm": _num(saved.get("prequantized_parent_norm")),
        "postquantized_parent_norm": _num(saved.get("postquantized_parent_norm")),
        "parent_change_install_l2": float(np.linalg.norm(precodec - before)),
        "parent_change_codec_l2": float(np.linalg.norm(postcodec - precodec)),
        "parent_change_total_l2": float(np.linalg.norm(postcodec - before)),
        "pre_action_current_parent_teacher_kl": diagnostic.get("pre_action_new_source_teacher_kl"),
        "installed_parent_teacher_kl_precodec": kl_before,
        "installed_parent_teacher_kl_postcodec": kl_after,
        "installed_parent_codec_kl_change_mean": (float(post_mean) - float(pre_mean)) if _num(pre_mean) is not None and _num(post_mean) is not None else None,
        "install_effective_error_by_resident": saved.get("prequantized_effective_errors"),
        "codec_effective_error_by_resident": saved.get("codec_effective_errors"),
        "codec_max_coefficient_error": _num(saved.get("codec_max_coefficient_error")),
        "prequantized_behavior_drift": pre_drift,
        "postquantized_behavior_drift": post_drift,
    }

def _reward_contrasts(rows: Sequence[Mapping[str, Any]], horizon: int, case_ids: set[str]) -> dict[str, Any]:
    rewards: dict[tuple[str, str, int, str], float] = {}
    for row in rows:
        if int(row.get("horizon", -1)) == horizon:
            key = (str(row["case_id"]), str(row["future_regime"]), int(row["replicate"]), str(row["arm"]))
            if key in rewards:
                raise ValueError(f"duplicate reward record {key}")
            rewards[key] = float(row["reward_per_step"])
    result = {}
    for case_id in sorted(case_ids):
        pairs = sorted({(future, rep) for cid, future, rep, _ in rewards if cid == case_id})
        if not pairs:
            raise ValueError(f"missing primary-horizon rewards for {case_id}")
        contrasts = {}
        for name, left, right in CONTRASTS:
            diffs = []
            for future, rep in pairs:
                a, b = (case_id, future, rep, left), (case_id, future, rep, right)
                if a not in rewards or b not in rewards:
                    raise ValueError(f"unpaired reward cells {a} / {b}")
                diffs.append(rewards[a] - rewards[b])
            contrasts[name] = {"paired_mean_reward_per_step_difference": _mean(diffs),
                               "matched_future_replicate_count": len(diffs)}
        result[case_id] = contrasts
    return result

def analyze(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    report_path = Path(input_path).resolve()
    report = _json(report_path)
    if report.get("format") != "retention_candidate_report_v1" or report.get("phase") != "main":
        raise ValueError("input must be analysis.json for the completed main run")
    if report.get("audit", {}).get("passed") is not True or (report.get("case_count"), report.get("branch_count")) != (24, 480):
        raise ValueError("the existing audit must pass for the complete 24-prefix / 480-branch grid")
    inputs = report.get("input_hashes", {})
    summary_path = Path(str(inputs.get("summary_path", ""))).resolve()
    if not summary_path.is_file() or _sha(summary_path) != inputs.get("summary_sha256"):
        raise ValueError("the saved source summary is missing or its report hash does not match")
    summary = _json(summary_path)
    if summary.get("format") != "retention_candidate_summary_v1" or summary.get("phase") != "main":
        raise ValueError("report source must be a completed main-phase summary")
    config, manifest = summary.get("config", {}), summary.get("source_manifest", {})
    fit_rows = report.get("diagnostics", {}).get("candidate_fit_by_case", [])
    diag_rows = report.get("diagnostics", {}).get("branch_codec_kl_by_arm", [])
    raw_rows = summary.get("branches", [])
    if len(fit_rows) != 24 or len(diag_rows) != 480 or len(raw_rows) != 480:
        raise ValueError("report or source summary does not contain the full main grid")
    fit_by_case = {str(row["case_id"]): row for row in fit_rows}
    if len(fit_by_case) != 24:
        raise ValueError("candidate-fit case identifiers are not unique")
    def branch_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (int(row["seed"]), str(row["pair_kind"]), str(row["state"]),
                str(row["future_regime"]), int(row["replicate"]), str(row["arm"]))
    diag_by_key = {branch_key(row): row for row in diag_rows}
    raw_by_key = {branch_key(row): row for row in raw_rows}
    if len(diag_by_key) != 480 or len(raw_by_key) != 480 or set(diag_by_key) != set(raw_by_key):
        raise ValueError("report and source summary branch identities differ")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, raw in raw_by_key.items():
        case_id = str(raw["case_id"])
        if case_id not in fit_by_case:
            raise ValueError(f"branch references missing prefix: {case_id}")
        grouped.setdefault((case_id, key[5]), []).append(_action(raw, diag_by_key[key]))
    by_case_arm: dict[str, dict[str, Any]] = {case_id: {} for case_id in fit_by_case}
    expected_repetitions = len(report.get("future_regimes", [])) * len(report.get("replicates", []))
    for (case_id, arm), rows in grouped.items():
        if arm not in ARMS or len(rows) != expected_repetitions:
            raise ValueError(f"incomplete branch repetition grid for {case_id} / {arm}")
        by_case_arm[case_id][arm] = _mean_tree(rows)
        by_case_arm[case_id][arm]["matched_branch_count"] = len(rows)
    reward_rows = _reward_contrasts(report.get("branch_horizon_observations", []),
                                   int(report["primary_horizon"]), set(fit_by_case))
    hashes: dict[str, Any] = {}
    cases = []
    for case_id, fit in sorted(fit_by_case.items()):
        conditioning, file_hashes = _conditioning(summary_path.parent, fit, config, manifest)
        hashes[case_id] = file_hashes
        saved_bands = fit.get("source_band_counts_by_owner")
        if set(by_case_arm[case_id]) != set(ARMS):
            raise ValueError(f"prefix {case_id} is missing one or more arms")
        split = fit["precodec_fit_split_kl"]
        mean_norm = _num(fit["candidate_norms"].get("mean"))
        behavior_norm = _num(fit["candidate_norms"].get("behavioral"))
        mean_norm_near_zero = mean_norm is None or mean_norm <= 1e-8
        fit_metrics = {
            "mean_candidate_norm": mean_norm, "behavioral_candidate_norm": behavior_norm,
            "behavioral_to_mean_norm_ratio": behavior_norm / mean_norm if not mean_norm_near_zero and behavior_norm is not None else None,
            "behavioral_to_mean_norm_ratio_status": "mean_norm_near_zero" if mean_norm_near_zero else "finite",
            "train_kl_mean_initial": split.get("train_mean_initial"), "train_kl_mean_behavioral": split.get("train_mean_fitted"),
            "train_kl_behavioral_minus_mean": float(split["train_mean_fitted"]) - float(split["train_mean_initial"]),
            "train_kl_by_owner_initial": split.get("train_by_owner_initial"), "train_kl_by_owner_behavioral": split.get("train_by_owner_fitted"),
            "holdout_kl_mean_initial": split.get("holdout_mean_initial"), "holdout_kl_mean_behavioral": split.get("holdout_mean_fitted"),
            "holdout_kl_behavioral_minus_mean": float(split["holdout_mean_fitted"]) - float(split["holdout_mean_initial"]),
            "holdout_kl_by_owner_initial": split.get("holdout_by_owner_initial"), "holdout_kl_by_owner_behavioral": split.get("holdout_by_owner_fitted"),
            "band_counts_by_owner": saved_bands, "source_psi_rank_saved": conditioning["source_psi_rank_saved"],
        }
        cases.append({"case_id": case_id,
                      "identity": {key: fit.get(key) for key in ("seed", "family", "pair_kind", "state")},
                      "candidate_fit": fit_metrics, "feature_conditioning": conditioning,
                      "actions_by_arm": by_case_arm[case_id],
                      "paired_reward_contrasts": reward_rows[case_id]})
    summary_by_arm = {}
    action_scalar_fields = ("candidate_norm", "parent_change_install_l2", "parent_change_codec_l2",
                            "parent_change_total_l2", "installed_parent_codec_kl_change_mean",
                            "codec_max_coefficient_error")
    for arm in ARMS:
        rows = [case["actions_by_arm"][arm] for case in cases]
        summary_by_arm[arm] = {field: _stat([row.get(field) for row in rows]) for field in action_scalar_fields}
        summary_by_arm[arm]["precodec_installed_parent_teacher_kl_mean"] = _stat([
            row.get("installed_parent_teacher_kl_precodec", {}).get("mean") for row in rows])
        summary_by_arm[arm]["postcodec_installed_parent_teacher_kl_mean"] = _stat([
            row.get("installed_parent_teacher_kl_postcodec", {}).get("mean") for row in rows])
        summary_by_arm[arm]["postcodec_max_probability_drift"] = _stat([
            row.get("postquantized_behavior_drift", {}).get("max_probability_absolute_drift") for row in rows])
    fit_summary = {field: _stat([case["candidate_fit"].get(field) for case in cases]) for field in (
        "behavioral_to_mean_norm_ratio", "train_kl_behavioral_minus_mean", "holdout_kl_behavioral_minus_mean")}
    reward_summary = {name: _stat([case["paired_reward_contrasts"][name]["paired_mean_reward_per_step_difference"] for case in cases])
                      for name, _, _ in CONTRASTS}
    script_path = Path(__file__).resolve()
    result = {
        "format": "retention_candidate_encoding_diagnostics_v1", "phase": "main",
        "scope": "zero-interaction descriptive summary of all prefixes and arms; no refit, selection, tuning, or causal analysis",
        "primary_horizon": int(report["primary_horizon"]),
        "counts": {"prefixes": len(cases), "arms_per_prefix": len(ARMS), "prefix_arm_cells": 24 * 5, "report_branches": len(diag_by_key)},
        "input_hashes": {"report_analysis_json_sha256": _sha(report_path), "summary_json_sha256": _sha(summary_path),
                         "summary_sha256_recorded_by_report": inputs["summary_sha256"],
                         "reporter_sha256": inputs.get("reporter_sha256"),
                         "runner_sha256": inputs.get("runner_sha256"),
                         "behavioral_fit_module_sha256": inputs.get("behavioral_fit_module_sha256"),
                         "source_manifest_canonical_sha256": inputs.get("source_manifest_canonical_sha256"),
                         "diagnostic_script_sha256": _sha(script_path), "prefix_files_by_case": hashes},
        "interpretation_limits": [
            "Coefficient norms depend on frozen feature coordinates; magnitude alone is not a quality or complexity measure.",
            "Pre-action teacher KL describes the inherited current parent, not the arithmetic-mean candidate.",
            "Paired reward contrasts are descriptive associations, not mediation or causal evidence; lower KL does not prove reward gain.",
        ],
        "fit_summary_across_24_prefixes": fit_summary,
        "action_summary_across_24_prefixes_by_arm": summary_by_arm,
        "paired_reward_summary_across_24_prefixes": reward_summary,
        "cases": cases,
    }
    output = Path(output_path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"preserve existing diagnostic output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="audited analysis.json from report_retention_candidate.py")
    parser.add_argument("--output", type=Path, required=True, help="new or empty diagnostic output directory")
    args = parser.parse_args(argv)
    result = analyze(args.input, args.output)
    print(json.dumps({"phase": result["phase"], "prefixes": result["counts"]["prefixes"],
                      "arms_per_prefix": result["counts"]["arms_per_prefix"], "output": str(Path(args.output).resolve())}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

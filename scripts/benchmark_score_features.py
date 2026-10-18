"""Benchmark shared feature computation in the frozen storage score path.

This is a checkpoint-only diagnostic. It does not construct or step an
environment, train an agent, or modify the production scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "outputs" / "storage_decision_main"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "score_feature_benchmark"
DEFAULT_REPEATS = 25
SCORE_TOLERANCE = 1e-7
ZERO_PARENT = np.zeros((3, 3), dtype=np.float32)
SCORE_FIELDS = ("min_rows", "min_blocks", "min_ess_ratio", "max_mean_kl", "clip_ratio", "stderr_multiplier")


def paired_coefficient_probs(
    core: Any,
    observations: Any,
    candidate: Any,
    baseline: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate candidate and baseline while sharing candidate-independent features."""

    obs_array = np.asarray(observations)
    if obs_array.ndim != 2 or obs_array.shape[1] != 13:
        raise ValueError("counterfactual observations must be N by 13")
    if len(obs_array) == 0:
        empty = np.empty((0, 4), dtype=np.float32)
        return empty.copy(), empty

    obs = torch.as_tensor(obs_array, dtype=torch.float32)
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        phi = torch.cat((latent, torch.ones_like(latent[:, :1])), dim=1)
        psi = torch.nn.functional.linear(phi, core.source_basis)
        base_logits = core.action_net(latent)
        contrast_basis_t = core._contrast_basis.T

        def evaluate(coefficients: Any) -> torch.Tensor:
            coeff = torch.as_tensor(np.asarray(coefficients), dtype=psi.dtype)
            if coeff.shape == (3, 3):
                contrast = torch.nn.functional.linear(psi, coeff)
            elif coeff.shape == (3, 3, 3):
                bands = torch.floor(obs[:, 0].clamp(0, 1) * 3).long().clamp(0, 2)
                contrast = torch.bmm(coeff[bands], psi.unsqueeze(-1)).squeeze(-1)
            else:
                raise ValueError("effective coefficients must be 3x3 or 3x3x3")
            logits = base_logits + contrast @ contrast_basis_t
            probs = torch.softmax(logits / core.temperature, dim=-1)
            return (1.0 - core.epsilon) * probs + core.epsilon / 4.0

        candidate_probs = evaluate(candidate)
        baseline_probs = evaluate(baseline)
    return candidate_probs.cpu().numpy(), baseline_probs.cpu().numpy()


def paired_feature_tensor_bytes(core: Any, observations: Any) -> int:
    """Bytes held by the paired evaluator's observation-side feature tensors."""

    obs_array = np.asarray(observations)
    if obs_array.ndim != 2 or obs_array.shape[1] != 13:
        raise ValueError("counterfactual observations must be N by 13")
    if len(obs_array) == 0:
        return 0
    obs = torch.as_tensor(obs_array, dtype=torch.float32)
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        phi = torch.cat((latent, torch.ones_like(latent[:, :1])), dim=1)
        psi = torch.nn.functional.linear(phi, core.source_basis)
        base_logits = core.action_net(latent)
    return int(sum(t.numel() * t.element_size() for t in (obs, latent, phi, psi, base_logits)))


def make_paired_score_fn(
    core: Any,
    config: Mapping[str, Any],
    score_function: Callable[..., dict[str, Any]],
) -> Callable[[Mapping[str, Any], Any, Any], dict[str, Any]]:
    gate = config["gate"]
    options = {field: gate[field] for field in SCORE_FIELDS}

    def score(evidence: Mapping[str, Any], candidate: Any, baseline: Any) -> dict[str, Any]:
        candidate_probs, baseline_probs = paired_coefficient_probs(
            core, evidence["obs"], candidate, baseline
        )
        return score_function(
            dict(evidence), candidate_probs, baseline_probs, **options
        )

    return score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = resolved.as_posix()
    if root is not None:
        try:
            display = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": display, "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _load_source_runner(repo_root: Path, data_root: Path) -> Any:
    """Load the exact run snapshot, with current-source fallback for omitted imports."""

    snapshot_root = data_root / "source_snapshot"
    snapshot_src = snapshot_root / "src"
    current_src = repo_root / "src"
    if not (snapshot_root / "scripts" / "run_storage_decision.py").is_file():
        raise FileNotFoundError("storage decision source snapshot is missing its runner")

    if str(current_src) not in sys.path:
        sys.path.insert(0, str(current_src))
    package = importlib.import_module("online_rl")
    package.__path__ = [str(snapshot_src / "online_rl"), str(current_src / "online_rl")]

    for entry in (str(repo_root), str(snapshot_src), str(snapshot_root)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    scripts_package = importlib.import_module("scripts")
    scripts_package.__path__ = [
        str(snapshot_root / "scripts"),
        str(repo_root / "scripts"),
    ]

    module_name = "scripts.run_storage_decision"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if snapshot_root.resolve() not in loaded_path.parents:
            del sys.modules[module_name]
            loaded = None
    runner = loaded or importlib.import_module(module_name)
    runner_path = Path(runner.__file__).resolve()
    if snapshot_root.resolve() not in runner_path.parents:
        raise RuntimeError("benchmark did not load the preserved storage decision runner")
    return runner


def _load_base_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "c1.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    for name in ("foundation", "dataset"):
        result[name] = str((repo_root / result[name]).resolve())
    return result


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "sum": float(sum(values)),
    }


def _score_difference(reference: Mapping[str, Any], paired: Mapping[str, Any]) -> dict[str, Any]:
    numeric_differences: dict[str, float | None] = {}
    nullable_mismatches: list[str] = []
    for field in ("gain", "lower", "stderr"):
        left, right = reference.get(field), paired.get(field)
        if left is None and right is None:
            numeric_differences[field] = None
        elif left is None or right is None:
            numeric_differences[field] = None
            nullable_mismatches.append(field)
        else:
            numeric_differences[field] = abs(float(left) - float(right))

    flag_fields = ("status", "supported", "reason", "coverage", "blocks", "rows")
    flag_mismatches = [
        field for field in flag_fields if reference.get(field) != paired.get(field)
    ]
    required_mismatches = [
        field for field in ("gain", "lower")
        if numeric_differences[field] is not None
        and numeric_differences[field] > SCORE_TOLERANCE
    ]
    required_mismatches.extend(nullable_mismatches)
    return {
        "numeric_abs_differences": numeric_differences,
        "nullable_mismatches": nullable_mismatches,
        "flag_mismatches": flag_mismatches,
        "required_mismatches": required_mismatches,
    }


def _cached_selector_output(
    runner: Any,
    history: list[int],
    evidence: Any,
    old_parent: np.ndarray,
    new_candidate: np.ndarray,
    old_score: Mapping[str, Any],
    new_score: Mapping[str, Any],
) -> dict[str, Any]:
    scores = iter((dict(old_score), dict(new_score)))
    return runner.decide_from_past(
        history, evidence, old_parent, new_candidate, lambda *_args: next(scores)
    )


def _module_records(repo_root: Path, snapshot_root: Path) -> list[dict[str, Any]]:
    paths: dict[str, Path] = {}
    allowed_roots = (snapshot_root.resolve(), (repo_root / "src").resolve(), (repo_root / "scripts").resolve())
    for name, module in sorted(sys.modules.items()):
        if not (name.startswith("online_rl.") or name.startswith("scripts.")):
            continue
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        if not path.is_file() or not any(root == path or root in path.parents for root in allowed_roots):
            continue
        paths[path.as_posix()] = path
    return [_file_record(path, root=repo_root) for path in sorted(paths.values(), key=lambda item: item.as_posix())]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _render_report(result: Mapping[str, Any]) -> str:
    benchmark = result["timing"]
    eq = result["equivalence"]
    memory = result["memory"]
    scale = result["cohort_scale"]
    rows = [
        "# Frozen score feature benchmark",
        "",
        "This checkpoint-only diagnostic compares the preserved score function with a paired-feature evaluator. It did not construct or step an environment, and did not train or modify an agent.",
        f"Timing context: {result['timing_context']}",
        "",
        f"- Cases: {result['case_count']} storage decision prefixes; scored group rows: {result['score_case_count']} (old and new groups per case).",
        f"- Repetitions: {benchmark['repeats']} after one warm-up per method and score case; torch threads: {benchmark['torch_threads']}.",
        f"- CPU timer: {result['execution']['cpu_clock']['implementation']} (reported resolution {result['execution']['cpu_clock']['reported_resolution_s']:.1g} s); per-call CPU values may be quantized, so interpret aggregate CPU means with wall timings.",
        f"- Equivalence: {'PASS' if eq['passed'] else 'FAIL'}; max probability error: {eq['max_probability_abs_error']:.3g}; max gain error: {eq['max_gain_abs_error']:.3g}; max lower error: {eq['max_lower_abs_error']:.3g}.",
        f"- Mean CPU per score: reference {benchmark['reference']['cpu_s']['mean']:.6f} s; paired {benchmark['paired']['cpu_s']['mean']:.6f} s; paired saving {benchmark['paired_delta']['cpu_saved_per_score_s']['mean']:.6f} s.",
        f"- Estimated saving per full case using the actual 2 score calls: {scale['estimated_saved_cpu_per_case_s']:.6f} s ({scale['estimated_saving_fraction_of_prefix_case_cpu']:.4%} of mean prefix CPU and {scale['estimated_saving_fraction_of_full_case_cpu']:.4%} of mean full case CPU). These are scaled estimates from scorer timing, not an end-to-end cohort speedup measurement.",
        f"- Paired feature workspace: {memory['temporary_feature_tensor_bytes_per_score']['mean']:.0f} bytes per score (mean); added persistent bytes: {memory['added_persistent_bytes']} bytes.",
        "",
        "## Timing setup",
        "",
        f"- Runner import: {result['setup']['runner_import_wall_s']:.3f} wall s / {result['setup']['runner_import_cpu_s']:.3f} CPU s.",
        f"- Checkpoint/model restore: {result['setup']['checkpoint_restore_wall_s']:.3f} wall s / {result['setup']['checkpoint_restore_cpu_s']:.3f} CPU s across {result['case_count']} cases.",
        f"- Scorer construction mean per case: reference {result['setup']['reference_scorer_construction_wall_s_mean']:.6f} wall s; paired {result['setup']['paired_scorer_construction_wall_s_mean']:.6f} wall s.",
        f"- Timed scorer totals over {benchmark['repeats']} rounds: reference {benchmark['reference']['cpu_s']['sum']:.3f} CPU s; paired {benchmark['paired']['cpu_s']['sum']:.3f} CPU s.",
        "",
        "## Correctness",
        "",
        f"- Probability, gain/lower, support, stored score, and selector checks passed: {eq['passed']}.",
        f"- Probability max absolute error: {eq['max_probability_abs_error']:.9g}; gain max absolute error: {eq['max_gain_abs_error']:.9g}; lower max absolute error: {eq['max_lower_abs_error']:.9g}.",
        f"- Support-flag mismatches: {eq['support_flag_mismatch_count']}; stored decision-score mismatches: {eq['stored_score_mismatch_count']}; selector mismatches: {eq['selector_mismatch_count']}; timed-output mismatches: {eq['timed_output_mismatch_count']}.",
        "",
        "## Per-score cases",
        "",
        "| Case | Group | Rows | Reference CPU (ms) | Paired CPU (ms) | Saved CPU (ms) | Max probability error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["per_score_case"]:
        rows.append(
            f"| {item['case_id']} | {item['group']} | {item['rows']} | "
            f"{1000 * item['reference_cpu_s_mean']:.3f} | "
            f"{1000 * item['paired_cpu_s_mean']:.3f} | "
            f"{1000 * item['saved_cpu_s_mean']:.3f} | "
            f"{item['max_probability_abs_error']:.3g} |"
        )
    rows.extend(
        [
            "",
            "## Memory and scope",
            "",
            f"- Mean / min / max temporary feature tensor bytes per score: {memory['temporary_feature_tensor_bytes_per_score']['mean']:.0f} / {memory['temporary_feature_tensor_bytes_per_score']['min']:.0f} / {memory['temporary_feature_tensor_bytes_per_score']['max']:.0f}.",
            "- Added persistent cache state: 0 bytes. No cache survives a score call and no learner/checkpoint state is changed.",
            "- This is common-subexpression reuse within the same candidate/baseline score. It is not a SparseForest local-structure result and does not support a core-contribution claim.",
            "- The comparison holds the frozen checkpoint, evidence arrays, candidate, baseline, and surrogate_gain logic fixed. Timing covers scorer calls only; setup and model restore are reported separately.",
            "",
            "## Mismatches",
            "",
        ]
    )
    mismatches = result["equivalence"]["mismatches"]
    if mismatches:
        rows.append("Mismatches are retained in report.json and listed here:")
        rows.append("")
        for mismatch in mismatches:
            rows.append(f"- {json.dumps(mismatch, ensure_ascii=False, default=_json_default)}")
    else:
        rows.append("None.")
    rows.extend(["", "Full hashes, per-case records, timing order, and detailed metrics are in report.json.", ""])
    return "\n".join(rows)


def run_benchmark(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    repeats: int = DEFAULT_REPEATS,
    order_seed: int = 20261018,
    timing_context: str = "unspecified; consult the run log for concurrent workload status",
) -> dict[str, Any]:
    if repeats < 20:
        raise ValueError("repeats must be at least 20")
    repo_root = ROOT.resolve()
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"benchmark output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    import_cpu_start, import_wall_start = time.process_time(), time.perf_counter()
    runner = _load_source_runner(repo_root, data_root)
    runner_import_cpu_s = time.process_time() - import_cpu_start
    runner_import_wall_s = time.perf_counter() - import_wall_start

    config_path = data_root / "config.json"
    summary_path = data_root / "summary.json"
    snapshot_manifest_path = data_root / "source_snapshot" / "manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cohort_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base_config = _load_base_config(repo_root)
    foundation_path = Path(base_config["foundation"])
    if not foundation_path.is_file():
        raise FileNotFoundError(f"configured foundation file is missing: {foundation_path}")
    foundation_sha = _sha256(foundation_path)

    case_dirs = sorted((data_root / "cases").iterdir(), key=lambda path: path.name)
    if len(case_dirs) != 32 or int(cohort_summary.get("case_count", -1)) != 32:
        raise ValueError("expected exactly 32 completed storage decision cases")
    summary_cases = {str(item["case_id"]): item for item in cohort_summary["cases"]}
    if len(summary_cases) != 32:
        raise ValueError("main cohort summary must contain all 32 cases")
    if set(summary_cases) != {path.name for path in case_dirs}:
        raise ValueError("main cohort summary case IDs do not match the 32 case directories")

    restore_cpu_start, restore_wall_start = time.process_time(), time.perf_counter()
    case_states: list[dict[str, Any]] = []
    score_cases: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = [
        _file_record(config_path, root=repo_root),
        _file_record(summary_path, root=repo_root),
        _file_record(snapshot_manifest_path, root=repo_root),
        _file_record(repo_root / "configs" / "c1.json", root=repo_root),
        _file_record(foundation_path, root=repo_root),
    ]

    for case_dir in case_dirs:
        case_id = case_dir.name
        checkpoint_path = case_dir / "prefix" / "checkpoint.pt"
        decision_path = case_dir / "decision.json"
        if not checkpoint_path.is_file() or not decision_path.is_file():
            raise FileNotFoundError(f"case {case_id} is missing prefix/checkpoint.pt or decision.json")
        input_records.extend(
            [_file_record(checkpoint_path, root=repo_root), _file_record(decision_path, root=repo_root)]
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("foundation_sha256") != foundation_sha:
            raise ValueError(f"foundation hash does not match prefix checkpoint for {case_id}")
        agent = runner._restore_learner(checkpoint, base_config, config)
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("case_id") != case_id:
            raise ValueError(f"decision case ID does not match directory for {case_id}")
        if not np.array_equal(agent.recent_groups, decision["selector_diagnostics"]["recent_group_ids"]):
            raise ValueError(f"recent group evidence differs from decision record for {case_id}")
        old_parent = np.asarray(decision["old_parent"], dtype=np.float32)
        new_candidate = np.asarray(decision["new_candidate"], dtype=np.float32)
        base_construction_wall_start = time.perf_counter()
        reference_score = runner._score_fn(agent.core, config)
        reference_construction_wall_s = time.perf_counter() - base_construction_wall_start
        paired_construction_wall_start = time.perf_counter()
        paired_score = make_paired_score_fn(agent.core, config, runner.surrogate_gain)
        paired_construction_wall_s = time.perf_counter() - paired_construction_wall_start
        groups: dict[int, dict[str, Any]] = {}
        for group, label, candidate, baseline in (
            (0, "old_zero_vs_old_parent", ZERO_PARENT, old_parent),
            (1, "new_parent_vs_zero", new_candidate, ZERO_PARENT),
        ):
            evidence = agent.evidence.arrays(group)
            rows = len(evidence["action"])
            expected_rows = decision["evidence_rows_per_group"]["old" if group == 0 else "new"]
            if rows != 256 or int(expected_rows) != rows:
                raise ValueError(f"case {case_id} group {group} does not have the frozen 256 evidence rows")
            candidate = np.asarray(candidate, dtype=np.float32)
            baseline = np.asarray(baseline, dtype=np.float32)

            reference_candidate_probs = runner._coefficient_probs(agent.core, evidence["obs"], candidate)
            reference_baseline_probs = runner._coefficient_probs(agent.core, evidence["obs"], baseline)
            paired_candidate_probs, paired_baseline_probs = paired_coefficient_probs(
                agent.core, evidence["obs"], candidate, baseline
            )
            candidate_error = float(np.max(np.abs(reference_candidate_probs - paired_candidate_probs), initial=0.0))
            baseline_error = float(np.max(np.abs(reference_baseline_probs - paired_baseline_probs), initial=0.0))
            reference_value = reference_score(evidence, candidate, baseline)
            paired_value = paired_score(evidence, candidate, baseline)
            score_diff = _score_difference(reference_value, paired_value)
            expected_score = decision["selector_diagnostics"]["surrogate"][label]
            stored_diff = _score_difference(expected_score, reference_value)
            key = f"{case_id}/{group}"
            groups[group] = {
                "label": label,
                "reference_score": reference_value,
                "paired_score": paired_value,
            }
            score_cases.append(
                {
                    "key": key,
                    "case_id": case_id,
                    "group": group,
                    "label": label,
                    "rows": rows,
                    "evidence": evidence,
                    "candidate": candidate,
                    "baseline": baseline,
                    "reference_fn": reference_score,
                    "paired_fn": paired_score,
                    "reference_value": reference_value,
                    "paired_value": paired_value,
                    "candidate_probability_abs_error": candidate_error,
                    "baseline_probability_abs_error": baseline_error,
                    "score_difference": score_diff,
                    "stored_score_difference": stored_diff,
                    "feature_tensor_bytes": paired_feature_tensor_bytes(agent.core, evidence["obs"]),
                }
            )

        reference_selector = _cached_selector_output(
            runner, list(agent.recent_groups), agent.evidence, old_parent, new_candidate,
            groups[0]["reference_score"], groups[1]["reference_score"],
        )
        paired_selector = _cached_selector_output(
            runner, list(agent.recent_groups), agent.evidence, old_parent, new_candidate,
            groups[0]["paired_score"], groups[1]["paired_score"],
        )
        case_states.append(
            {
                "case_id": case_id,
                "reference_selectors": reference_selector["past_only_selectors"],
                "paired_selectors": paired_selector["past_only_selectors"],
                "stored_selectors": decision["selectors"],
                "paired_scorer_construction_wall_s": paired_construction_wall_s,
                "reference_scorer_construction_wall_s": reference_construction_wall_s,
            }
        )
        del agent, checkpoint

    checkpoint_restore_cpu_s = time.process_time() - restore_cpu_start
    checkpoint_restore_wall_s = time.perf_counter() - restore_wall_start

    mismatches: list[dict[str, Any]] = []
    max_probability_error = 0.0
    max_gain_error = 0.0
    max_lower_error = 0.0
    support_flag_mismatch_count = 0
    stored_score_mismatch_count = 0
    selector_mismatch_count = 0
    for item in score_cases:
        probability_error = max(
            item["candidate_probability_abs_error"], item["baseline_probability_abs_error"]
        )
        max_probability_error = max(max_probability_error, probability_error)
        diff = item["score_difference"]
        gain_error = diff["numeric_abs_differences"]["gain"]
        lower_error = diff["numeric_abs_differences"]["lower"]
        if gain_error is not None:
            max_gain_error = max(max_gain_error, gain_error)
        if lower_error is not None:
            max_lower_error = max(max_lower_error, lower_error)
        if diff["flag_mismatches"]:
            support_flag_mismatch_count += 1
        if probability_error > SCORE_TOLERANCE or diff["required_mismatches"] or diff["flag_mismatches"]:
            mismatches.append(
                {
                    "kind": "reference_vs_paired",
                    "score_case": item["key"],
                    "probability_abs_error": probability_error,
                    **diff,
                }
            )
        stored_diff = item["stored_score_difference"]
        if stored_diff["required_mismatches"] or stored_diff["flag_mismatches"]:
            stored_score_mismatch_count += 1
            mismatches.append(
                {
                    "kind": "reference_vs_saved_decision",
                    "score_case": item["key"],
                    **stored_diff,
                }
            )
    for case_state in case_states:
        if case_state["reference_selectors"] != case_state["paired_selectors"]:
            selector_mismatch_count += 1
            mismatches.append(
                {
                    "kind": "reference_vs_paired_selectors",
                    "case_id": case_state["case_id"],
                    "reference": case_state["reference_selectors"],
                    "paired": case_state["paired_selectors"],
                }
            )
        if case_state["reference_selectors"] != case_state["stored_selectors"]:
            selector_mismatch_count += 1
            mismatches.append(
                {
                    "kind": "reference_vs_saved_selectors",
                    "case_id": case_state["case_id"],
                    "reference": case_state["reference_selectors"],
                    "stored": case_state["stored_selectors"],
                }
            )

    # Warm each implementation once on every real evidence group before timing.
    for item in score_cases:
        item["reference_fn"](item["evidence"], item["candidate"], item["baseline"])
        item["paired_fn"](item["evidence"], item["candidate"], item["baseline"])

    order_rng = random.Random(order_seed)
    timing_samples: list[dict[str, Any]] = []
    paired_order_counts = {"reference_first": 0, "paired_first": 0}
    timed_output_mismatch_count = 0
    per_key_samples: dict[str, list[dict[str, Any]]] = {item["key"]: [] for item in score_cases}
    for repeat in range(repeats):
        round_cases = list(score_cases)
        order_rng.shuffle(round_cases)
        for item in round_cases:
            reference_first = bool(order_rng.getrandbits(1))
            method_order = ("reference", "paired") if reference_first else ("paired", "reference")
            paired_order_counts["reference_first" if reference_first else "paired_first"] += 1
            measurements: dict[str, dict[str, float]] = {}
            for method in method_order:
                score_fn = item["reference_fn"] if method == "reference" else item["paired_fn"]
                cpu_start, wall_start = time.process_time(), time.perf_counter()
                measured_value = score_fn(item["evidence"], item["candidate"], item["baseline"])
                measurements[method] = {
                    "cpu_s": time.process_time() - cpu_start,
                    "wall_s": time.perf_counter() - wall_start,
                }
                expected_value = item["reference_value"] if method == "reference" else item["paired_value"]
                timed_diff = _score_difference(expected_value, measured_value)
                if timed_diff["required_mismatches"] or timed_diff["flag_mismatches"]:
                    timed_output_mismatch_count += 1
            sample = {
                "repeat": repeat,
                "case_id": item["case_id"],
                "group": item["group"],
                "order": list(method_order),
                "reference_cpu_s": measurements["reference"]["cpu_s"],
                "paired_cpu_s": measurements["paired"]["cpu_s"],
                "reference_wall_s": measurements["reference"]["wall_s"],
                "paired_wall_s": measurements["paired"]["wall_s"],
                "cpu_saved_s": measurements["reference"]["cpu_s"] - measurements["paired"]["cpu_s"],
                "wall_saved_s": measurements["reference"]["wall_s"] - measurements["paired"]["wall_s"],
            }
            timing_samples.append(sample)
            per_key_samples[item["key"]].append(sample)

    reference_cpu = [row["reference_cpu_s"] for row in timing_samples]
    paired_cpu = [row["paired_cpu_s"] for row in timing_samples]
    reference_wall = [row["reference_wall_s"] for row in timing_samples]
    paired_wall = [row["paired_wall_s"] for row in timing_samples]
    cpu_saved = [row["cpu_saved_s"] for row in timing_samples]
    wall_saved = [row["wall_saved_s"] for row in timing_samples]
    per_score_output: list[dict[str, Any]] = []
    for item in score_cases:
        samples = per_key_samples[item["key"]]
        per_score_output.append(
            {
                "case_id": item["case_id"],
                "group": item["group"],
                "label": item["label"],
                "rows": item["rows"],
                "max_probability_abs_error": max(
                    item["candidate_probability_abs_error"], item["baseline_probability_abs_error"]
                ),
                "score_difference": item["score_difference"],
                "stored_score_difference": item["stored_score_difference"],
                "feature_tensor_bytes": item["feature_tensor_bytes"],
                "reference_cpu_s_mean": statistics.fmean(row["reference_cpu_s"] for row in samples),
                "paired_cpu_s_mean": statistics.fmean(row["paired_cpu_s"] for row in samples),
                "saved_cpu_s_mean": statistics.fmean(row["cpu_saved_s"] for row in samples),
                "reference_wall_s_mean": statistics.fmean(row["reference_wall_s"] for row in samples),
                "paired_wall_s_mean": statistics.fmean(row["paired_wall_s"] for row in samples),
                "saved_wall_s_mean": statistics.fmean(row["wall_saved_s"] for row in samples),
            }
        )

    cohort_case_count = int(cohort_summary["case_count"])
    mean_prefix_case_cpu_s = float(cohort_summary["costs"]["prefix_cpu_s"]) / cohort_case_count
    mean_full_case_cpu_s = float(cohort_summary["costs"]["cpu_s"]) / cohort_case_count
    estimated_saved_cpu_per_case_s = statistics.fmean(cpu_saved) * 2.0
    equivalence_passed = (
        max_probability_error <= SCORE_TOLERANCE
        and max_gain_error <= SCORE_TOLERANCE
        and max_lower_error <= SCORE_TOLERANCE
        and support_flag_mismatch_count == 0
        and stored_score_mismatch_count == 0
        and selector_mismatch_count == 0
        and timed_output_mismatch_count == 0
    )
    source_snapshot_root = data_root / "source_snapshot"
    script_path = Path(__file__).resolve()
    module_records = _module_records(repo_root, source_snapshot_root)
    input_records = sorted(input_records, key=lambda item: item["path"])
    input_records.append(_file_record(script_path, root=repo_root))
    provenance = {
        "script": _file_record(script_path, root=repo_root),
        "preserved_source_snapshot_root": source_snapshot_root.as_posix(),
        "source_snapshot_manifest": _file_record(snapshot_manifest_path, root=repo_root),
        "loaded_code_modules": module_records,
        "inputs": input_records,
        "foundation_sha256_checked_against_all_prefix_checkpoints": foundation_sha,
        "local_instruction_or_pdf_contents_copied": False,
    }
    feature_bytes = [item["feature_tensor_bytes"] for item in score_cases]
    reference_construction = [item["reference_scorer_construction_wall_s"] for item in case_states]
    paired_construction = [item["paired_scorer_construction_wall_s"] for item in case_states]
    result: dict[str, Any] = {
        "format": "score_feature_benchmark_v1",
        "status": "completed" if equivalence_passed else "completed_with_mismatches",
        "timing_context": timing_context,
        "case_count": len(case_states),
        "score_case_count": len(score_cases),
        "actual_score_calls_per_cohort_case": 2,
        "execution": {
            "checkpoint_only": True,
            "environment_constructed": False,
            "environment_steps": 0,
            "ppo_updates": 0,
            "torch_threads": torch.get_num_threads(),
            "repeats": repeats,
            "warmup_per_method_per_score_case": 1,
            "ab_order_seed": order_seed,
            "ab_order_counts": paired_order_counts,
            "timed_calls_per_method": len(timing_samples),
            "timing_scope": "reference _score_fn or paired scorer call, including probability calculation and the same surrogate_gain reduction",
            "cpu_clock": {
                "implementation": time.get_clock_info("process_time").implementation,
                "reported_resolution_s": time.get_clock_info("process_time").resolution,
                "note": "Per-call process CPU samples may be quantized on this Windows host; use aggregated CPU means and wall timings for interpretation.",
            },
        },
        "setup": {
            "runner_import_cpu_s": runner_import_cpu_s,
            "runner_import_wall_s": runner_import_wall_s,
            "checkpoint_restore_cpu_s": checkpoint_restore_cpu_s,
            "checkpoint_restore_wall_s": checkpoint_restore_wall_s,
            "mean_checkpoint_restore_cpu_s": checkpoint_restore_cpu_s / len(case_states),
            "mean_checkpoint_restore_wall_s": checkpoint_restore_wall_s / len(case_states),
            "reference_scorer_construction_wall_s_mean": statistics.fmean(reference_construction),
            "paired_scorer_construction_wall_s_mean": statistics.fmean(paired_construction),
        },
        "timing": {
            "repeats": repeats,
            "torch_threads": torch.get_num_threads(),
            "reference": {"cpu_s": _stats(reference_cpu), "wall_s": _stats(reference_wall)},
            "paired": {"cpu_s": _stats(paired_cpu), "wall_s": _stats(paired_wall)},
            "paired_delta": {
                "cpu_saved_per_score_s": _stats(cpu_saved),
                "wall_saved_per_score_s": _stats(wall_saved),
                "cpu_total_saved_s": float(sum(cpu_saved)),
                "wall_total_saved_s": float(sum(wall_saved)),
            },
            "paired_speedup_ratio_cpu": (
                float(sum(reference_cpu) / sum(paired_cpu)) if sum(paired_cpu) > 0 else None
            ),
            "paired_speedup_ratio_wall": (
                float(sum(reference_wall) / sum(paired_wall)) if sum(paired_wall) > 0 else None
            ),
        },
        "memory": {
            "temporary_feature_tensor_bytes_per_score": _stats([float(value) for value in feature_bytes]),
            "temporary_feature_bytes_definition": "live float32 observation, latent, phi, psi, and base_logits tensors in one paired scorer call; excludes outputs and allocator-reserved memory",
            "added_persistent_bytes": 0,
            "persistent_cache_created": False,
        },
        "cohort_scale": {
            "basis": "estimated from mean paired timing delta per scorer multiplied by the real 2 score calls per prefix case; not a full-cohort rerun",
            "mean_prefix_case_cpu_s": mean_prefix_case_cpu_s,
            "mean_full_case_cpu_s": mean_full_case_cpu_s,
            "estimated_saved_cpu_per_case_s": estimated_saved_cpu_per_case_s,
            "estimated_saving_fraction_of_prefix_case_cpu": (
                estimated_saved_cpu_per_case_s / mean_prefix_case_cpu_s if mean_prefix_case_cpu_s > 0 else None
            ),
            "estimated_saving_fraction_of_full_case_cpu": (
                estimated_saved_cpu_per_case_s / mean_full_case_cpu_s if mean_full_case_cpu_s > 0 else None
            ),
            "main_cohort_case_count": cohort_case_count,
            "main_cohort_total_cpu_s": float(cohort_summary["costs"]["cpu_s"]),
            "main_cohort_prefix_cpu_s": float(cohort_summary["costs"]["prefix_cpu_s"]),
            "estimated_cohort_total_saving_s": estimated_saved_cpu_per_case_s * cohort_case_count,
        },
        "equivalence": {
            "passed": equivalence_passed,
            "tolerance": SCORE_TOLERANCE,
            "probability_max_abs_error_limit": SCORE_TOLERANCE,
            "gain_lower_max_abs_error_limit": SCORE_TOLERANCE,
            "max_probability_abs_error": max_probability_error,
            "max_gain_abs_error": max_gain_error,
            "max_lower_abs_error": max_lower_error,
            "support_flag_mismatch_count": support_flag_mismatch_count,
            "stored_score_mismatch_count": stored_score_mismatch_count,
            "selector_mismatch_count": selector_mismatch_count,
            "timed_output_mismatch_count": timed_output_mismatch_count,
            "mismatches": mismatches,
        },
        "per_score_case": per_score_output,
        "timing_samples": timing_samples,
        "claim_boundary": "Same-score common-subexpression optimization only. This is not SparseForest local maintenance, structural novelty, or a core contribution claim.",
        "provenance": provenance,
    }
    (output_root / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    (output_root / "report.md").write_text(_render_report(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--order-seed", type=int, default=20261018)
    parser.add_argument("--timing-context", type=str, default="unspecified; consult the run log for concurrent workload status")
    args = parser.parse_args(argv)
    result = run_benchmark(
        args.data_root, args.output_root, repeats=args.repeats, order_seed=args.order_seed,
        timing_context=args.timing_context,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "case_count": result["case_count"],
                "score_case_count": result["score_case_count"],
                "equivalence": result["equivalence"]["passed"],
                "mean_cpu_saved_per_score_s": result["timing"]["paired_delta"]["cpu_saved_per_score_s"]["mean"],
                "estimated_saved_cpu_per_case_s": result["cohort_scale"]["estimated_saved_cpu_per_case_s"],
                "report": (args.output_root / "report.md").resolve().as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["equivalence"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Make frozen, descriptive plots for the ordered retention cohorts.

This is an offline plotting utility. It reads the completed S1 report and/or
S2/S3 stream summaries and logs, starts no environment, and adds no experiment
steps. Outputs go only to the requested output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


LOWER_FORMAT = "surrogate_lower_confirmation_report_v1"
STREAM_FORMAT = "retention_value_stream_summary_v1"
COHORT_CONFIG_FORMAT = "retention_value_stream_run_config_v1"
RELATIONS = ("dynamics", "reward_conflict")
COMPARISON_METHODS = (
    "learned_linear", "learned_intercept", "random", "keep", "replace",
    "mean", "frequency", "lower", "shared_ppo", "independent",
)
BLOCK_CURVE_METHODS = (
    "learned_linear", "learned_intercept", "mean", "replace", "shared_ppo", "independent",
)
METHOD_LABELS = {
    "learned_linear": "Learned linear",
    "learned_intercept": "Learned intercept",
    "random": "Uniform random",
    "keep": "Keep",
    "replace": "Replace",
    "mean": "Mean",
    "frequency": "Frequency",
    "lower": "Generalized lower",
    "shared_ppo": "Direct shared PPO",
    "independent": "Independent",
}
RELATION_LABELS = {"dynamics": "Dynamics", "reward_conflict": "Reward conflict"}
TAIL_SIZE = 32


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL log: {path}") from exc
    return rows


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    result = _number(value, name)
    if not result.is_integer() or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class InputHashes:
    """Track exact files consumed by this plotting run."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, Any]] = {}
        self.missing: list[dict[str, str]] = []

    def add(self, path: Path | None, role: str) -> Path | None:
        if path is None:
            self.missing.append({"role": role, "path": "<not supplied>"})
            return None
        resolved = path.resolve()
        if not resolved.is_file():
            self.missing.append({"role": role, "path": str(resolved)})
            return None
        key = str(resolved)
        row = self.files.get(key)
        if row is None:
            row = {
                "path": key,
                "sha256": _sha256(resolved),
                "size_bytes": resolved.stat().st_size,
                "roles": [],
            }
            self.files[key] = row
        if role not in row["roles"]:
            row["roles"].append(role)
        return resolved

    def as_list(self) -> list[dict[str, Any]]:
        return [self.files[key] for key in sorted(self.files)]


def _resolve_from_root(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_cohort(
    input_path: str | Path,
    stage: str,
    hashes: InputHashes,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    source = Path(input_path).resolve()
    summary_path = source / "summary.json" if source.is_dir() else source
    summary_path = hashes.add(summary_path, f"{stage}_summary")
    if summary_path is None:
        raise FileNotFoundError(f"expected summary.json under {source}")
    summary = _read_json(summary_path)
    if summary.get("format") != STREAM_FORMAT:
        raise ValueError(f"{summary_path} has an unexpected format")
    if summary.get("stage") != stage:
        raise ValueError(f"{summary_path} must describe stage {stage!r}")
    if summary.get("pilot") is not False:
        raise ValueError(f"{stage} plots require the main cohort; pilot streams are plumbing only")

    root = summary_path.parent
    frozen_path = _resolve_from_root(root, summary.get("frozen_config_path"))
    if frozen_path is None:
        frozen_path = root / "frozen_config.json"
    frozen_path = hashes.add(frozen_path, f"{stage}_frozen_config")
    if frozen_path is None:
        raise FileNotFoundError(f"missing frozen_config.json for {stage} cohort")
    frozen = _read_json(frozen_path)
    if frozen.get("format") == COHORT_CONFIG_FORMAT:
        if frozen.get("stage") != stage or frozen.get("pilot") is not False:
            raise ValueError(f"{frozen_path} is not the frozen main {stage} design")
        config = frozen.get("config")
    else:
        config = frozen.get("config", frozen)
    if not isinstance(config, dict):
        raise ValueError(f"{frozen_path} does not contain a configuration object")
    if summary.get("protocol_sha256") and frozen.get("protocol_sha256"):
        if summary["protocol_sha256"] != frozen["protocol_sha256"]:
            raise ValueError(f"{stage} summary and frozen configuration have different protocol hashes")

    design = config.get(stage)
    if not isinstance(design, dict):
        raise ValueError(f"frozen config.{stage} must be an object")
    seeds_value = design.get("seeds")
    if not isinstance(seeds_value, list) or not seeds_value:
        raise ValueError(f"frozen config.{stage}.seeds must be a non-empty list")
    seeds = [_integer(seed, f"config.{stage}.seeds[]") for seed in seeds_value]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"frozen config.{stage}.seeds contains duplicates")
    pair_kinds_value = design.get("pair_kinds", config.get("pair_kinds"))
    if not isinstance(pair_kinds_value, list) or tuple(pair_kinds_value) != RELATIONS:
        raise ValueError(f"frozen config.{stage} must include relations {RELATIONS}")
    methods_value = design.get("methods")
    expected_methods = ("random",) if stage == "calibration" else COMPARISON_METHODS
    if not isinstance(methods_value, list) or tuple(methods_value) != expected_methods:
        raise ValueError(f"frozen config.{stage}.methods must match the prespecified method order")
    decisions = _integer(design.get("decisions"), f"config.{stage}.decisions", minimum=1)
    expected_decisions = 64 if stage == "calibration" else 48
    if decisions != expected_decisions:
        raise ValueError(f"main {stage} cohort must have {expected_decisions} decisions per stream")
    expected_seed_count = 6 if stage == "calibration" else 4
    if len(seeds) != expected_seed_count:
        raise ValueError(f"main {stage} cohort must have {expected_seed_count} seeds")

    streams_value = summary.get("streams")
    if not isinstance(streams_value, list):
        raise ValueError(f"{summary_path} must contain a streams list")
    streams: dict[tuple[int, str, str], dict[str, Any]] = {}
    for offset, row in enumerate(streams_value):
        if not isinstance(row, dict):
            raise ValueError(f"summary.streams[{offset}] must be an object")
        try:
            key = (
                _integer(row.get("seed"), f"streams[{offset}].seed"),
                str(row.get("pair_kind")),
                str(row.get("method")),
            )
        except ValueError:
            continue
        if key in streams:
            raise ValueError(f"duplicate stream key {key} in {summary_path}")
        streams[key] = row
    design_meta = {
        "stage": stage,
        "seeds": seeds,
        "relations": list(pair_kinds_value),
        "methods": list(methods_value),
        "decisions": decisions,
        "expected_streams": len(seeds) * len(pair_kinds_value) * len(methods_value),
        "summary_protocol_sha256": summary.get("protocol_sha256"),
        "frozen_protocol_sha256": frozen.get("protocol_sha256"),
    }
    return summary, streams, root, design_meta


def _log_path(root: Path, stream: Mapping[str, Any], name: str) -> Path | None:
    return _resolve_from_root(root, stream.get(f"{name}_path"))


def _save_figure(fig: Any, output: Path, stem: str, figures: list[dict[str, Any]]) -> None:
    for extension in ("png", "pdf", "svg"):
        path = output / f"{stem}.{extension}"
        options: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            options["dpi"] = 300
        fig.savefig(path, **options)
        figures.append({"name": path.name, "format": extension})


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires Matplotlib in the selected Python environment") from exc
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _lower_data(input_path: str | Path, hashes: InputHashes) -> dict[str, Any]:
    source = Path(input_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError("--lower must be a confirmation report directory")
    analysis_path = hashes.add(source / "analysis.json", "lower_confirmation_analysis")
    seed_path = hashes.add(source / "per_seed.csv", "lower_confirmation_per_seed")
    if analysis_path is None or seed_path is None:
        raise FileNotFoundError("--lower requires analysis.json and per_seed.csv")
    analysis = _read_json(analysis_path)
    if analysis.get("format") != LOWER_FORMAT or analysis.get("status") != "complete_confirmation":
        raise ValueError("--lower is not a complete surrogate-lower confirmation report")
    primary = analysis.get("primary")
    if not isinstance(primary, dict) or primary.get("contrast") != "surrogate_lower minus keep":
        raise ValueError("lower report primary contrast is not surrogate_lower minus keep")
    estimate = primary.get("estimate")
    if not isinstance(estimate, dict):
        raise ValueError("lower report lacks a primary estimate")
    n = _integer(estimate.get("n"), "lower primary estimate n", minimum=1)
    if n != 16:
        raise ValueError(f"lower confirmation requires n=16 seeds, got {n}")
    mean = _number(estimate.get("mean"), "lower primary mean")
    ci = estimate.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("lower primary estimate must contain a two-value ci95")
    ci_low, ci_high = (_number(ci[0], "lower ci95 lower"), _number(ci[1], "lower ci95 upper"))
    if ci_low > mean or mean > ci_high:
        raise ValueError("lower primary mean must lie within its ci95")

    rows: list[dict[str, Any]] = []
    try:
        with seed_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"seed", "old_family", "observed_history_mean_lower_minus_keep"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("per_seed.csv lacks required seed, old_family, or lower-minus-keep columns")
            for row in reader:
                family = row["old_family"]
                if family not in ("A", "B"):
                    raise ValueError(f"invalid lower-report old family: {family!r}")
                rows.append({
                    "seed": _integer(row["seed"], "per_seed.seed"),
                    "family": family,
                    "difference": _number(row["observed_history_mean_lower_minus_keep"], "per_seed difference"),
                })
    except OSError as exc:
        raise ValueError(f"cannot read lower per-seed CSV: {seed_path}") from exc
    rows.sort(key=lambda row: (row["family"], row["seed"]))
    if len(rows) != 16 or len({row["seed"] for row in rows}) != 16:
        raise ValueError(f"per_seed.csv must have 16 unique seed rows, got {len(rows)}")
    counts = {family: sum(row["family"] == family for row in rows) for family in ("A", "B")}
    if counts != {"A": 8, "B": 8}:
        raise ValueError(f"lower confirmation requires 8 seeds per old family, got {counts}")
    computed_mean = float(_mean([row["difference"] for row in rows]))
    if not math.isclose(computed_mean, mean, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("per-seed lower-minus-keep mean disagrees with analysis.json")
    return {"rows": rows, "n": n, "mean": mean, "ci95": [ci_low, ci_high], "family_counts": counts}


def _plot_lower(data: Mapping[str, Any], output: Path, figures: list[dict[str, Any]]) -> dict[str, Any]:
    plt = _pyplot()
    rows = list(data["rows"])
    family_styles = {
        "A": {"color": "#0072B2", "marker": "o", "label": "Old family A"},
        "B": {"color": "#D55E00", "marker": "s", "label": "Old family B"},
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.7), constrained_layout=True)
    for index, row in enumerate(rows):
        style = family_styles[row["family"]]
        ax.scatter(index, row["difference"], s=54, color=style["color"], marker=style["marker"],
                   edgecolor="white", linewidth=0.5, zorder=3,
                   label=style["label"] if index == 0 or rows[index - 1]["family"] != row["family"] else None)
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="-", label="Zero")
    ax.axhline(0.002, color="#009E73", linewidth=1.25, linestyle="--", label="Practical threshold (+0.002)")
    mean = float(data["mean"])
    low, high = (float(value) for value in data["ci95"])
    ax.errorbar(16.5, mean, yerr=[[mean - low], [high - mean]], fmt="D", markersize=7,
                color="#222222", ecolor="#222222", capsize=4, linewidth=1.5,
                label="Overall mean and stratified 95% CI", zorder=5)
    ax.set_xticks([*range(16), 16.5], [str(row["seed"]) for row in rows] + ["Overall mean\n(n=16)"])
    ax.set_xlim(-0.6, 17.5)
    bounds = [row["difference"] for row in rows] + [low, high, 0.0, 0.002]
    span = max(bounds) - min(bounds)
    margin = max(0.00015, span * 0.10)
    ax.set_ylim(min(bounds) - margin, max(bounds) + margin)
    ax.set_xlabel("Seed (points grouped by old-family mapping)")
    ax.set_ylabel("Surrogate lower minus keep (reward per step)")
    ax.set_title("Independent surrogate-lower confirmation")
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.55, alpha=0.45)
    ax.legend(frameon=False, ncol=2, loc="best")
    _save_figure(fig, output, "s1_lower_confirmation", figures)
    plt.close(fig)
    return {
        "statistical_unit": "seed",
        "n": int(data["n"]),
        "family_counts": data["family_counts"],
        "mean": mean,
        "ci95": [low, high],
        "practical_threshold": 0.002,
        "points": rows,
    }


def _decision_map(rows: Sequence[Mapping[str, Any]], context: str) -> dict[int, Mapping[str, Any]]:
    mapped: dict[int, Mapping[str, Any]] = {}
    for offset, row in enumerate(rows):
        decision = _integer(row.get("decision_index"), f"{context}[{offset}].decision_index")
        if decision in mapped:
            raise ValueError(f"{context} contains duplicate decision_index {decision}")
        mapped[decision] = row
    return mapped


def _calibration_stream_score(
    row: Mapping[str, Any],
    root: Path,
    hashes: InputHashes,
    decisions: int,
) -> dict[str, Any]:
    stream_id = str(row.get("stream_id", "unknown stream"))
    if row.get("status") != "completed" or row.get("completed_decisions") != decisions:
        raise ValueError(f"{stream_id}: stream is not complete")
    pre_path = hashes.add(_log_path(root, row, "preactions"), f"calibration_preactions:{stream_id}")
    feedback_path = hashes.add(_log_path(root, row, "feedback"), f"calibration_feedback:{stream_id}")
    if pre_path is None or feedback_path is None:
        raise FileNotFoundError(f"{stream_id}: missing preactions or feedback JSONL")
    preactions = _decision_map(_read_jsonl(pre_path), f"{stream_id}.preactions")
    feedback = _decision_map(_read_jsonl(feedback_path), f"{stream_id}.feedback")
    expected = set(range(decisions))
    if set(preactions) != expected or set(feedback) != expected:
        raise ValueError(f"{stream_id}: expected exactly {decisions} matched preaction and feedback records")
    squared = {"linear": [], "intercept": []}
    tail_indices = range(decisions - TAIL_SIZE, decisions)
    for index in tail_indices:
        pre = preactions[index]
        observed = feedback[index]
        action = pre.get("selected_action")
        if action not in ("keep", "replace", "mean", "none") or observed.get("selected_action") != action:
            raise ValueError(f"{stream_id}: selected action mismatch at decision {index}")
        target = _number(observed.get("utility_target"), f"{stream_id}.feedback[{index}].utility_target")
        predictions = pre.get("predictions")
        if not isinstance(predictions, dict):
            raise ValueError(f"{stream_id}: missing pre-action model predictions at decision {index}")
        for model in ("linear", "intercept"):
            action_values = predictions.get(model)
            if not isinstance(action_values, dict) or action not in action_values:
                raise ValueError(f"{stream_id}: missing {model} prediction for selected action {action}")
            prediction = _number(action_values[action], f"{stream_id}.{model}.{action} prediction")
            squared[model].append((prediction - target) ** 2)
    if any(len(values) != TAIL_SIZE for values in squared.values()):
        raise ValueError(f"{stream_id}: final {TAIL_SIZE} prequential labels are incomplete")
    linear_mse = float(_mean(squared["linear"]))
    intercept_mse = float(_mean(squared["intercept"]))
    return {
        "seed": _integer(row.get("seed"), f"{stream_id}.seed"),
        "pair_kind": str(row.get("pair_kind")),
        "linear_mse": linear_mse,
        "intercept_mse": intercept_mse,
        "difference": linear_mse - intercept_mse,
        "scored_decisions": TAIL_SIZE,
        "tail_decision_indices_zero_based": [decisions - TAIL_SIZE, decisions - 1],
    }


def _calibration_data(
    input_path: str | Path,
    hashes: InputHashes,
) -> dict[str, Any]:
    summary, streams, root, design = _load_cohort(input_path, "calibration", hashes)
    expected = {
        (seed, relation, "random")
        for seed in design["seeds"] for relation in design["relations"]
    }
    unexpected = set(streams) - expected
    if unexpected:
        raise ValueError(f"calibration summary contains unexpected stream keys: {sorted(unexpected)}")
    scores: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for key in sorted(expected):
        row = streams.get(key)
        if row is None:
            invalid.append({"seed": key[0], "pair_kind": key[1], "reason": "missing stream summary"})
            continue
        try:
            scores.append(_calibration_stream_score(row, root, hashes, design["decisions"]))
        except (ValueError, FileNotFoundError) as exc:
            invalid.append({"seed": key[0], "pair_kind": key[1], "reason": str(exc)})
    for relation in design["relations"]:
        count = sum(row["pair_kind"] == relation for row in scores)
        if count == 0:
            raise ValueError(f"calibration has no complete selected-label streams for {relation}")
    return {"design": design, "scores": scores, "invalid_streams": invalid}


def _plot_calibration(data: Mapping[str, Any], output: Path, figures: list[dict[str, Any]]) -> dict[str, Any]:
    plt = _pyplot()
    design = data["design"]
    colors = plt.get_cmap("tab10")
    seed_color = {seed: colors(index % 10) for index, seed in enumerate(sorted(design["seeds"]))}
    scores = list(data["scores"])
    fig, ax = plt.subplots(figsize=(8.5, 5.7), constrained_layout=True)
    all_values = [row["difference"] for row in scores]
    for relation_index, relation in enumerate(design["relations"]):
        relation_rows = sorted(
            (row for row in scores if row["pair_kind"] == relation),
            key=lambda row: row["seed"],
        )
        jitter = {seed: ((index - (len(design["seeds"]) - 1) / 2) * 0.045)
                  for index, seed in enumerate(design["seeds"])}
        for row in relation_rows:
            ax.scatter(relation_index + jitter[row["seed"]], row["difference"], s=54,
                       color=seed_color[row["seed"]], edgecolor="white", linewidth=0.5,
                       zorder=3, label=f"Seed {row['seed']}" if relation_index == 0 else None)
        mean = _mean([row["difference"] for row in relation_rows])
        if mean is not None:
            ax.scatter(relation_index, mean, marker="D", s=74, color="#222222",
                       edgecolor="white", linewidth=0.5, zorder=5,
                       label="Relation mean" if relation_index == 0 else None)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    group_counts = {
        relation: sum(row["pair_kind"] == relation for row in scores)
        for relation in design["relations"]
    }
    ax.set_xticks(range(len(design["relations"])), [
        f"{RELATION_LABELS[relation]}\n(n={group_counts[relation]}/{len(design['seeds'])})"
        for relation in design["relations"]
    ])
    ax.set_xlim(-0.45, len(design["relations"]) - 0.55)
    if all_values:
        lower, upper = min([*all_values, 0.0]), max([*all_values, 0.0])
        margin = max((upper - lower) * 0.12, 1e-5)
        ax.set_ylim(lower - margin, upper + margin)
    ax.set_ylabel("MSE(linear) - MSE(intercept)\n(final 32 of 64 selected labels)")
    ax.set_title(
        "Prequential prediction error on the selected action\n"
        "Negative values favor linear; each dot is one seed stream",
        pad=10,
    )
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.55, alpha=0.45)
    ax.legend(frameon=False, ncol=4, loc="best")
    _save_figure(fig, output, "s2_calibration_selected_mse", figures)
    plt.close(fig)
    return {
        "statistical_unit": "stream (seed x relation)",
        "tail_decision_indices_zero_based": [32, 63],
        "scores_per_stream": TAIL_SIZE,
        "expected_seeds_per_relation": len(design["seeds"]),
        "valid_stream_counts": group_counts,
        "scores": scores,
        "invalid_streams": data["invalid_streams"],
    }


def _comparison_stream_metric(row: Mapping[str, Any], expected_steps: int) -> dict[str, Any]:
    stream_id = str(row.get("stream_id", "unknown stream"))
    decisions = _integer(row.get("completed_decisions"), f"{stream_id}.completed_decisions")
    if row.get("status") != "completed" or decisions != 48:
        raise ValueError(f"{stream_id}: stream is not complete")
    costs = row.get("costs")
    if not isinstance(costs, dict):
        raise ValueError(f"{stream_id}: missing cost summary")
    steps = _integer(costs.get("environment_steps"), f"{stream_id}.environment_steps", minimum=1)
    if steps != expected_steps:
        raise ValueError(f"{stream_id}: expected {expected_steps} whole-stream steps, got {steps}")
    total = _number(row.get("total_reward_sum"), f"{stream_id}.total_reward_sum")
    warmup = _number(row.get("warmup_return_sum"), f"{stream_id}.warmup_return_sum")
    outcome = _number(row.get("outcome_return_sum"), f"{stream_id}.outcome_return_sum")
    if not math.isclose(total, warmup + outcome, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError(f"{stream_id}: total reward does not equal warmup plus outcomes")
    return {
        "seed": _integer(row.get("seed"), f"{stream_id}.seed"),
        "pair_kind": str(row.get("pair_kind")),
        "method": str(row.get("method")),
        "steps": steps,
        "total_reward_sum": total,
        "whole_stream_return_per_step": total / steps,
    }


def _block_rates(
    row: Mapping[str, Any],
    root: Path,
    hashes: InputHashes,
    decisions: int,
    expected_block_steps: int,
) -> list[float]:
    stream_id = str(row.get("stream_id", "unknown stream"))
    path = hashes.add(_log_path(root, row, "feedback"), f"comparison_feedback:{stream_id}")
    if path is None:
        raise FileNotFoundError(f"{stream_id}: missing feedback JSONL")
    feedback = _decision_map(_read_jsonl(path), f"{stream_id}.feedback")
    expected = set(range(decisions))
    if set(feedback) != expected:
        raise ValueError(f"{stream_id}: expected exactly {decisions} outcome feedback records")
    rates: list[float] = []
    for index in range(decisions):
        record = feedback[index]
        block_steps = _integer(record.get("block_steps"), f"{stream_id}.feedback[{index}].block_steps", minimum=1)
        if block_steps != expected_block_steps:
            raise ValueError(f"{stream_id}: expected {expected_block_steps} steps at decision {index}, got {block_steps}")
        block_return = _number(record.get("block_return_sum"), f"{stream_id}.feedback[{index}].block_return_sum")
        rate = block_return / block_steps
        if "reward_rate" in record:
            logged_rate = _number(record["reward_rate"], f"{stream_id}.feedback[{index}].reward_rate")
            if not math.isclose(rate, logged_rate, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"{stream_id}: reward_rate disagrees with block return at decision {index}")
        rates.append(rate)
    return rates


def _comparison_data(
    input_path: str | Path,
    hashes: InputHashes,
) -> dict[str, Any]:
    summary, streams, root, design = _load_cohort(input_path, "comparison", hashes)
    expected_keys = {
        (seed, relation, method)
        for seed in design["seeds"]
        for relation in design["relations"]
        for method in design["methods"]
    }
    unexpected = set(streams) - expected_keys
    if unexpected:
        raise ValueError(f"comparison summary contains unexpected stream keys: {sorted(unexpected)}")
    frozen_path = _resolve_from_root(root, summary.get("frozen_config_path")) or root / "frozen_config.json"
    frozen_path = hashes.add(frozen_path, "comparison_frozen_config")
    if frozen_path is None:
        raise FileNotFoundError("missing frozen comparison configuration")
    frozen = _read_json(frozen_path)
    config = frozen.get("config", frozen)
    if not isinstance(config, dict):
        raise ValueError("comparison frozen config must contain its nested config object")
    bootstrap_steps = _integer(config.get("bootstrap_steps"), "config.bootstrap_steps", minimum=1)
    bout_steps = _integer(config.get("bout_steps"), "config.bout_steps", minimum=1)
    outcome_bouts = _integer(config.get("outcome_bouts"), "config.outcome_bouts", minimum=1)
    # Warm-up fits two group-0 tasks for bootstrap_steps and two group-1 tasks
    # for bout_steps; bootstrap_steps is per task, not the full warm-up total.
    warmup_steps_expected = 2 * bootstrap_steps + 2 * bout_steps
    whole_steps_expected = warmup_steps_expected + design["decisions"] * outcome_bouts * bout_steps
    expected_outcome_steps = design["decisions"] * outcome_bouts * bout_steps
    expected_block_steps = outcome_bouts * bout_steps
    metrics: list[dict[str, Any]] = []
    rates: dict[tuple[int, str, str], list[float]] = {}
    invalid: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        row = streams.get(key)
        if row is None:
            invalid.append({"seed": key[0], "pair_kind": key[1], "method": key[2], "reason": "missing stream summary"})
            continue
        try:
            metrics.append(_comparison_stream_metric(row, whole_steps_expected))
        except ValueError as exc:
            invalid.append({"seed": key[0], "pair_kind": key[1], "method": key[2], "reason": str(exc)})
            continue
        if key[2] in BLOCK_CURVE_METHODS:
            try:
                rates[key] = _block_rates(row, root, hashes, design["decisions"], expected_block_steps)
            except (ValueError, FileNotFoundError) as exc:
                invalid.append({"seed": key[0], "pair_kind": key[1], "method": key[2], "reason": str(exc)})
    return {
        "design": design,
        "whole_steps_expected": whole_steps_expected,
        "outcome_steps_expected": expected_outcome_steps,
        "metrics": metrics,
        "block_rates": rates,
        "invalid_streams": invalid,
    }


def _plot_comparison(data: Mapping[str, Any], output: Path, figures: list[dict[str, Any]]) -> dict[str, Any]:
    plt = _pyplot()
    design = data["design"]
    metrics = list(data["metrics"])
    seeds = list(design["seeds"])
    seed_colors = {seed: plt.get_cmap("tab10")(index % 10) for index, seed in enumerate(seeds)}
    method_colors = {method: plt.get_cmap("tab10")(index % 10) for index, method in enumerate(COMPARISON_METHODS)}

    fig, axes = plt.subplots(1, len(design["relations"]), figsize=(17, 7), sharey=True,
                             constrained_layout=True)
    if len(design["relations"]) == 1:
        axes = [axes]
    by_relation_method: dict[str, dict[str, dict[str, Any]]] = {}
    seed_offsets = {seed: (index - (len(seeds) - 1) / 2) * 0.085 for index, seed in enumerate(seeds)}
    for axis, relation in zip(axes, design["relations"]):
        relation_summary: dict[str, dict[str, Any]] = {}
        for method_index, method in enumerate(COMPARISON_METHODS):
            points = sorted(
                (row for row in metrics if row["pair_kind"] == relation and row["method"] == method),
                key=lambda row: row["seed"],
            )
            values = [row["whole_stream_return_per_step"] for row in points]
            for point in points:
                axis.scatter(method_index + seed_offsets[point["seed"]], point["whole_stream_return_per_step"],
                             s=36, color=seed_colors[point["seed"]], edgecolor="white", linewidth=0.4,
                             alpha=0.92, zorder=3)
            mean = _mean(values)
            if mean is not None:
                axis.scatter(method_index, mean, marker="D", s=54, color="#222222",
                             edgecolor="white", linewidth=0.4, zorder=5)
            relation_summary[method] = {
                "mean_whole_stream_return_per_step": mean,
                "n_complete_streams": len(points),
                "n_expected_streams": len(seeds),
                "points": points,
            }
        by_relation_method[relation] = relation_summary
        axis.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.85)
        axis.set_xticks(range(len(COMPARISON_METHODS)), [
            f"{METHOD_LABELS[method]}\n(n={relation_summary[method]['n_complete_streams']}/{len(seeds)})"
            for method in COMPARISON_METHODS
        ], rotation=40, ha="right")
        axis.set_xlim(-0.55, len(COMPARISON_METHODS) - 0.45)
        axis.set_title(f"{RELATION_LABELS[relation]} (seed unit; n={len(seeds)} planned)")
        axis.grid(axis="y", color="#B8B8B8", linewidth=0.55, alpha=0.4)
        axis.set_xlabel("Prespecified method")
    axes[0].set_ylabel("Whole-stream return per environment step\n(includes warm-up and outcomes)")
    fig.suptitle("Repeated-retention comparison: all 10 prespecified methods")
    seed_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=seed_colors[seed], label=f"Seed {seed}")
        for seed in seeds
    ]
    seed_handles.append(plt.Line2D([], [], marker="D", linestyle="none", color="#222222", label="Method mean"))
    fig.legend(handles=seed_handles, frameon=False, ncol=min(len(seed_handles), 5),
               loc="outside lower center")
    _save_figure(fig, output, "s3_comparison_whole_stream", figures)
    plt.close(fig)

    curve_fig, curve_axes = plt.subplots(1, len(design["relations"]), figsize=(15.5, 6.3), sharey=True,
                                         constrained_layout=True)
    if len(design["relations"]) == 1:
        curve_axes = [curve_axes]
    curve_counts: dict[str, dict[str, int]] = {}
    block_rates = data["block_rates"]
    for axis, relation in zip(curve_axes, design["relations"]):
        relation_counts: dict[str, int] = {}
        for method in BLOCK_CURVE_METHODS:
            series_by_seed = [
                block_rates[(seed, relation, method)]
                for seed in seeds if (seed, relation, method) in block_rates
            ]
            relation_counts[method] = len(series_by_seed)
            means = [
                _mean([series[decision] for series in series_by_seed])
                for decision in range(design["decisions"])
            ]
            axis.plot(
                range(1, design["decisions"] + 1), means,
                color=method_colors[method], linewidth=1.8,
                marker="o", markersize=2.6, markevery=4,
                label=f"{METHOD_LABELS[method]} (n={len(series_by_seed)})",
            )
        axis.axvspan(0.5, 16.5, color="#B8B8B8", alpha=0.22, zorder=0)
        axis.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.8)
        axis.set_xlim(0.5, design["decisions"] + 0.5)
        axis.set_xlabel("Decision block")
        axis.set_title(RELATION_LABELS[relation])
        axis.grid(axis="y", color="#B8B8B8", linewidth=0.55, alpha=0.4)
        curve_counts[relation] = relation_counts
    curve_axes[0].set_ylabel("Mean outcome block reward per step")
    curve_fig.suptitle(
        "Outcome-block trajectory by decision (mean across complete streams)\n"
        "Shading marks decisions 1–16, when learned selectors explore uniformly",
    )
    curve_handles, curve_labels = curve_axes[0].get_legend_handles_labels()
    curve_fig.legend(curve_handles, curve_labels, frameon=False, ncol=3,
                     loc="outside lower center")
    _save_figure(curve_fig, output, "s3_comparison_block_returns", figures)
    plt.close(curve_fig)

    return {
        "whole_stream_statistical_unit": "stream (seed x relation x method); summaries are over seeds, not decisions",
        "whole_stream_metric": "(warmup return + outcome return) / all environment steps",
        "whole_stream_steps_per_stream": int(data["whole_steps_expected"]),
        "whole_stream": by_relation_method,
        "block_curve_statistical_unit": "seed streams; each decision point is the mean across complete streams",
        "block_curve_decisions": design["decisions"],
        "block_curve_methods_fixed_before_results": list(BLOCK_CURVE_METHODS),
        "block_curve_stream_counts": curve_counts,
        "invalid_streams": data["invalid_streams"],
    }


def _guard_output(output: Path, input_roots: Sequence[Path]) -> None:
    resolved = output.resolve()
    for root in input_roots:
        root = root.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(f"output directory must be outside experiment input directory {root}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not any((args.lower, args.calibration, args.comparison)):
        raise ValueError("provide at least one of --lower, --calibration, or --comparison")
    input_roots = [Path(value).resolve() for value in (args.lower, args.calibration, args.comparison) if value]
    _guard_output(args.output, input_roots)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hashes = InputHashes()
    figures: list[dict[str, Any]] = []
    plot_data: dict[str, Any] = {}

    if args.lower:
        lower = _lower_data(args.lower, hashes)
        plot_data["lower_confirmation"] = _plot_lower(lower, output, figures)
    if args.calibration:
        calibration = _calibration_data(args.calibration, hashes)
        plot_data["calibration"] = _plot_calibration(calibration, output, figures)
    if args.comparison:
        comparison = _comparison_data(args.comparison, hashes)
        plot_data["comparison"] = _plot_comparison(comparison, output, figures)

    metadata = {
        "format": "ordered_retention_plot_metadata_v1",
        "plotting_script": "scripts/plot_ordered_retention.py",
        "inputs": {
            "lower": None if not args.lower else str(Path(args.lower).resolve()),
            "calibration": None if not args.calibration else str(Path(args.calibration).resolve()),
            "comparison": None if not args.comparison else str(Path(args.comparison).resolve()),
        },
        "input_files": hashes.as_list(),
        "missing_input_files": hashes.missing,
        "generated_figures": figures,
        "plots": plot_data,
        "additional_environment_steps": 0,
        "reinforcement_learning_or_environment_started": False,
    }
    metadata_path = output / "plot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=Path, help="completed S1 confirmation report directory")
    parser.add_argument("--calibration", type=Path, help="completed S2 calibration cohort directory")
    parser.add_argument("--comparison", type=Path, help="completed S3 comparison cohort directory")
    parser.add_argument("--output", type=Path, required=True, help="separate directory for plots and metadata")
    args = parser.parse_args()
    try:
        metadata = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "generated_figures": [row["name"] for row in metadata["generated_figures"]],
        "metadata": str((args.output.resolve() / "plot_metadata.json")),
        "input_file_count": len(metadata["input_files"]),
        "additional_environment_steps": 0,
    }, indent=2))


if __name__ == "__main__":
    main()

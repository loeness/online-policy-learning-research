"""Generate an evidence-based Markdown report and static figures from run logs.

The module intentionally reads only the JSON and CSV artifacts written by the
experiment runner. It does not construct illustrative or synthetic results.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_METHOD_ORDER = (
    "frozen",
    "full",
    "global_residual",
    "history_keep",
    "history_reset",
    "oracle_keep",
)
_RETURN = "return"
_SUCCESS = "success"
_CONFIDENCE = "route_confidence"
_CAPACITY_FIELDS = (
    "policy_bytes",
    "agent_state_bytes",
    "rollout_bytes",
    "rss_bytes",
)


def generate_report(output_dir: str | Path) -> Path:
    """Write ``report.md`` and actual-data PNG/PDF figures under *output_dir*.

    ``output_dir`` can be one run directory containing ``episodes.csv`` or a
    parent directory containing run directories. Runs are combined at the
    seed/method level so uncertainty bands and paired comparisons never treat
    individual episodes as independent samples.
    """

    root = Path(output_dir).expanduser().resolve()
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(
            f"No episodes.csv found in {root}; reporting requires actual run data."
        )

    runs = [_read_run(run_dir) for run_dir in run_dirs]
    # A suite directory can carry metadata that is absent from each leaf run,
    # such as source pretraining evaluation and the aggregate Stage A gate.
    root_summary = _read_json(root / "summary.json")
    root_config = _read_json(root / "config.json")
    rows = [row for run in runs for row in run["episodes"]]
    if not rows:
        raise ValueError(f"No episode records found in {root}; no figures were created.")

    # Import after validating the inputs so a missing/empty run does not create
    # any report artifacts. Agg keeps this suitable for headless experiment runs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    output_rows = [row for row in rows if not row["_is_eval"]]
    methods = _ordered_methods({row["_method"] for row in rows})
    seed_method_stage = _seed_method_rows(rows)
    stage_a = _stage_rows(output_rows, "A")
    stage_b = _stage_rows(output_rows, "B")
    stage_b_context = [row for row in stage_b if not _as_bool(row.get("mixed_context"))]
    visits = _phase_b_visits(stage_b_context)

    figure_path = root / "learning_summary.png"
    pdf_path = root / "learning_summary.pdf"
    _make_figure(plt, figure_path, pdf_path, stage_a, visits, methods)

    report_path = root / "report.md"
    report_path.write_text(
        _build_report(
            root=root,
            runs=runs,
            rows=rows,
            output_rows=output_rows,
            seed_method_stage=seed_method_stage,
            stage_a=stage_a,
            visits=visits,
            methods=methods,
            root_summary=root_summary,
            root_config=root_config,
            figure_path=figure_path,
            pdf_path=pdf_path,
        ),
        encoding="utf-8",
    )
    return report_path


def _find_run_dirs(root: Path) -> list[Path]:
    if root.is_file():
        raise NotADirectoryError(root)
    if (root / "episodes.csv").is_file():
        return [root]
    if not root.exists():
        return []
    return sorted({path.parent for path in root.rglob("episodes.csv")})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {"_value": value}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _read_run(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    raw_episodes = _read_csv(run_dir / "episodes.csv")
    episodes: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_episodes):
        row = dict(raw)
        row["_seed"] = _first_nonempty(raw.get("seed"), config.get("seed"), run_dir.name)
        row["_method"] = str(
            _first_nonempty(raw.get("method"), config.get("method"), "unknown-method")
        )
        row["_stage"] = _normalise_stage(raw.get("stage"))
        row["_is_eval"] = _as_bool(raw.get("is_eval"))
        row["_row_index"] = index
        row["_episode"] = _as_number(raw.get("episode"))
        row["_global_steps"] = _as_number(raw.get("global_steps"))
        for key in (
            _RETURN,
            _SUCCESS,
            _CONFIDENCE,
            "steps",
            "elapsed_s",
            "update_count",
            *_CAPACITY_FIELDS,
        ):
            row[f"_{key}"] = _as_number(raw.get(key))
        episodes.append(row)

    raw_capacity = _read_csv(run_dir / "capacity.csv")
    capacity: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_capacity):
        row = dict(raw)
        row["_seed"] = _first_nonempty(raw.get("seed"), config.get("seed"), run_dir.name)
        row["_method"] = str(
            _first_nonempty(raw.get("method"), config.get("method"), "unknown-method")
        )
        row["_row_index"] = index
        # Keep every bytes-like field so runner-specific capacity measurements
        # remain visible without assuming a single capacity.csv schema.
        for key, value in raw.items():
            if "byte" in key.lower():
                row[f"_{key}"] = _as_number(value)
        capacity.append(row)
    return {
        "path": run_dir,
        "config": config,
        "summary": summary,
        "episodes": episodes,
        "capacity": capacity,
    }


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return "unknown"


def _normalise_stage(value: Any) -> str | None:
    token = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    if token in {"a", "phasea", "stagea"} or token.startswith(("phasea", "stagea")):
        return "A"
    if token in {"b", "phaseb", "stageb"} or token.startswith(("phaseb", "stageb")):
        return "B"
    return None


def _as_number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    token = str(value).strip().lower()
    if token in {"true", "yes", "y", "t"}:
        return 1.0
    if token in {"false", "no", "n", "f"}:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def _ordered_methods(methods: Iterable[str]) -> list[str]:
    values = set(methods)
    preferred = [method for method in _METHOD_ORDER if method in values]
    return preferred + sorted(values - set(preferred))


def _seed_method_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str | None], list[dict[str, Any]]]:
    """Group observations without merging Stage A and Stage B outcomes."""

    grouped: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["_seed"]), str(row["_method"]), row.get("_stage"))].append(row)
    return grouped


def _stage_rows(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["_stage"] == stage]
    return sorted(selected, key=_row_order)


def _row_order(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        row.get("_global_steps") if row.get("_global_steps") is not None else math.inf,
        row.get("_episode") if row.get("_episode") is not None else math.inf,
        int(row.get("_row_index", 0)),
    )


def _phase_b_visits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create first/last-five summaries for each recorded B bout/visit.

    Context novelty is a transparent proxy: a context's first appearance for
    each seed/method is an initial (unknown-route) exposure; later appearances
    are revisits. Episodes within each exposure are never pooled across seeds.
    """

    grouped = _seed_method_rows(rows)
    result: list[dict[str, Any]] = []
    for (seed, method, _stage), seed_rows in grouped.items():
        ordered = sorted(seed_rows, key=_row_order)
        blocks = _visit_blocks(ordered)
        seen_contexts: set[str] = set()
        for exposure_index, block in enumerate(blocks, start=1):
            context = _text(block[0].get("context")) or "(missing context)"
            novelty = "first-seen" if context not in seen_contexts else "revisit"
            seen_contexts.add(context)
            bout = _text(block[0].get("bout")) or "?"
            visit = block[0].get("_visit_label", "?")
            early_rows = block[:5]
            late_rows = block[-5:]
            result.append(
                {
                    "seed": seed,
                    "method": method,
                    "bout": bout,
                    "visit": str(visit),
                    "order": exposure_index,
                    "context": context,
                    "label": f"b{bout}/v{visit} · {context}",
                    "novelty": novelty,
                    "early_n": len(early_rows),
                    "late_n": len(late_rows),
                    "early_return": _mean([row.get(f"_{_RETURN}") for row in early_rows]),
                    "late_return": _mean([row.get(f"_{_RETURN}") for row in late_rows]),
                    "early_success": _mean([row.get(f"_{_SUCCESS}") for row in early_rows]),
                    "late_success": _mean([row.get(f"_{_SUCCESS}") for row in late_rows]),
                    "early_confidence": _mean([row.get(f"_{_CONFIDENCE}") for row in early_rows]),
                }
            )
    return result


def _visit_blocks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    last_key: tuple[str, str, str] | None = None
    occurrences: dict[tuple[str, str], int] = defaultdict(int)
    active: list[dict[str, Any]] = []
    for row in rows:
        bout = _text(row.get("bout")) or "?"
        context = _text(row.get("context")) or "(missing context)"
        explicit_visit = _text(row.get("visit"))
        sequence_key = (bout, context, explicit_visit)
        if sequence_key != last_key:
            if active:
                blocks.append(active)
            active = []
            if explicit_visit:
                visit_label = explicit_visit
            else:
                occurrence_key = (bout, context)
                occurrences[occurrence_key] += 1
                visit_label = str(occurrences[occurrence_key])
            row["_visit_label"] = visit_label
            last_key = sequence_key
        else:
            row["_visit_label"] = active[0].get("_visit_label", "?")
        active.append(row)
    if active:
        blocks.append(active)
    return blocks


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def _curve_by_seed(
    stage_rows: list[dict[str, Any]], metric: str
) -> dict[tuple[str, str], dict[float, float]]:
    grouped = _seed_method_rows(stage_rows)
    result: dict[tuple[str, str], dict[float, float]] = {}
    for key, seed_rows in grouped.items():
        seed, method, _stage = key
        values: dict[float, list[float]] = defaultdict(list)
        for index, row in enumerate(sorted(seed_rows, key=_row_order), start=1):
            x = row.get("_episode")
            if x is None:
                x = float(index)
            y = row.get(f"_{metric}")
            if y is not None:
                values[float(x)].append(float(y))
        result[(seed, method)] = {x: float(np.mean(ys)) for x, ys in values.items()}
    return result


def _seed_level_values(
    values: Iterable[tuple[str, float | None]],
) -> dict[str, float]:
    collected: dict[str, list[float]] = defaultdict(list)
    for seed, value in values:
        if value is not None:
            collected[str(seed)].append(float(value))
    return {seed: float(np.mean(items)) for seed, items in collected.items() if items}


def _bootstrap_ci(values: list[float], seed: int = 1701, draws: int = 4000) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(np.asarray(values)[samples], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _paired_bootstrap(
    left: dict[str, float], right: dict[str, float], seed: int = 2407, draws: int = 10000
) -> tuple[int, float | None, tuple[float, float] | None]:
    common = sorted(set(left) & set(right))
    if not common:
        return 0, None, None
    differences = np.asarray([left[key] - right[key] for key in common], dtype=float)
    estimate = float(np.mean(differences))
    if len(common) < 2:
        return len(common), estimate, None
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(common), size=(draws, len(common)))
    bootstrap = np.mean(differences[samples], axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return len(common), estimate, (float(low), float(high))


def _make_figure(
    plt: Any,
    png_path: Path,
    pdf_path: Path,
    stage_a: list[dict[str, Any]],
    visits: list[dict[str, Any]],
    methods: list[str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    colors = {method: palette(index % 10) for index, method in enumerate(methods)}
    _plot_stage_a_curve(axes[0, 0], stage_a, methods, colors, _RETURN, "Stage A training return")
    _plot_stage_a_curve(axes[0, 1], stage_a, methods, colors, _SUCCESS, "Stage A training success")
    _plot_b_visits(axes[1, 0], visits, methods, colors)
    _plot_recurrence(axes[1, 1], visits, methods, colors)
    fig.suptitle("Online adaptation results (seed-level uncertainty)", fontsize=14)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_stage_a_curve(
    ax: Any,
    rows: list[dict[str, Any]],
    methods: list[str],
    colors: dict[str, Any],
    metric: str,
    title: str,
) -> None:
    curves = _curve_by_seed(rows, metric)
    plotted = False
    for method in methods:
        by_seed = {seed: curve for (seed, name), curve in curves.items() if name == method}
        xs = sorted({x for curve in by_seed.values() for x in curve})
        means: list[float] = []
        lows: list[float | None] = []
        highs: list[float | None] = []
        kept_x: list[float] = []
        for x in xs:
            values = [curve[x] for curve in by_seed.values() if x in curve]
            if not values:
                continue
            kept_x.append(x)
            means.append(float(np.mean(values)))
            ci = _bootstrap_ci(values, seed=1701 + int(x) + len(method))
            lows.append(ci[0] if ci else None)
            highs.append(ci[1] if ci else None)
        if not kept_x:
            continue
        plotted = True
        line, = ax.plot(kept_x, means, color=colors[method], linewidth=1.8, label=method)
        if len(by_seed) > 1:
            low_arr = np.asarray([mean if low is None else low for mean, low in zip(means, lows)])
            high_arr = np.asarray([mean if high is None else high for mean, high in zip(means, highs)])
            ax.fill_between(kept_x, low_arr, high_arr, color=colors[method], alpha=0.15, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Mean per episode" if metric == _RETURN else "Success fraction")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No Stage A training records", ha="center", va="center", transform=ax.transAxes)


def _visit_sort_key(visit: dict[str, Any]) -> tuple[float, float, str, str]:
    bout = _as_number(visit.get("bout"))
    number = _as_number(visit.get("visit"))
    return (
        bout if bout is not None else math.inf,
        number if number is not None else math.inf,
        str(visit.get("bout")),
        str(visit.get("label")),
    )


def _plot_b_visits(ax: Any, visits: list[dict[str, Any]], methods: list[str], colors: dict[str, Any]) -> None:
    """Plot per-exposure first-five and last-five returns, averaged by seed."""

    if not visits:
        ax.text(0.5, 0.5, "No Stage B training records", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Stage B return by bout / visit")
        return
    ordered_events = sorted({visit["label"] for visit in visits}, key=lambda label: _label_event_sort(label, visits))
    positions = {label: index for index, label in enumerate(ordered_events)}
    for method in methods:
        for metric_name, field, linestyle, marker in (
            ("early first 5", "early_return", "-", "o"),
            ("late last 5", "late_return", "--", "s"),
        ):
            grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for visit in visits:
                if visit["method"] == method and visit.get(field) is not None:
                    grouped[visit["label"]][visit["seed"]].append(float(visit[field]))
            xs: list[int] = []
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for label in ordered_events:
                seed_values = grouped.get(label, {})
                values = [float(np.mean(items)) for items in seed_values.values() if items]
                if not values:
                    continue
                xs.append(positions[label])
                means.append(float(np.mean(values)))
                ci = _bootstrap_ci(values, seed=2701 + positions[label] + len(method) + len(metric_name))
                lows.append(ci[0] if ci else means[-1])
                highs.append(ci[1] if ci else means[-1])
            if xs:
                ax.plot(xs, means, color=colors[method], linestyle=linestyle, marker=marker,
                        markersize=3.5, linewidth=1.5, label=f"{method} · {metric_name}")
                if len(lows) and any(low != mean for low, mean in zip(lows, means)):
                    ax.fill_between(xs, lows, highs, color=colors[method], alpha=0.09, linewidth=0)
    ax.set_title("Stage B return by bout / visit")
    ax.set_xlabel("Recorded bout / visit / context (exposures aligned by label)")
    ax.set_ylabel("Mean return per exposure")
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(ordered_events, rotation=60, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)


def _label_event_sort(label: str, visits: list[dict[str, Any]]) -> tuple[float, float, str]:
    matching = next((visit for visit in visits if visit["label"] == label), {})
    key = _visit_sort_key(matching)
    return key[0], key[1], label


def _recurrence_values(visits: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for visit in visits:
        grouped[(visit["seed"], visit["method"])].append(visit)
    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, seed_visits in grouped.items():
        # _phase_b_visits preserves the chronology established from global_steps
        # and episode order. Numeric label sorting is only for chart axes and may
        # not reflect chronology for runner-specific visit labels.
        ordered = sorted(seed_visits, key=lambda visit: int(visit.get("order", 0)))
        first_by_context: dict[str, float] = {}
        deltas: dict[str, list[float]] = defaultdict(list)
        for visit in ordered:
            early = visit.get("early_return")
            if early is None:
                continue
            context = visit["context"]
            if context not in first_by_context:
                first_by_context[context] = float(early)
            elif visit["novelty"] == "revisit":
                deltas[context].append(float(early) - first_by_context[context])
        result[key] = {context: float(np.mean(values)) for context, values in deltas.items() if values}
    return result


def _plot_recurrence(ax: Any, visits: list[dict[str, Any]], methods: list[str], colors: dict[str, Any]) -> None:
    if not visits:
        ax.text(0.5, 0.5, "No repeated Stage B contexts", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Recurrence recovery")
        return
    recurrences = _recurrence_values(visits)
    context_order: list[str] = []
    for visit in sorted(visits, key=_visit_sort_key):
        if visit["novelty"] == "revisit" and visit["context"] not in context_order:
            context_order.append(visit["context"])
    if not context_order:
        ax.text(0.5, 0.5, "No repeated Stage B contexts", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Recurrence recovery")
        return
    for method in methods:
        xs: list[int] = []
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for index, context in enumerate(context_order):
            values = [mapping[context] for (seed, name), mapping in recurrences.items()
                      if name == method and context in mapping]
            if values:
                xs.append(index)
                means.append(float(np.mean(values)))
                ci = _bootstrap_ci(values, seed=3701 + index + len(method))
                lows.append(ci[0] if ci else means[-1])
                highs.append(ci[1] if ci else means[-1])
        if xs:
            ax.plot(xs, means, marker="o", linewidth=1.7, color=colors[method], label=method)
            if len(xs) and any(low != mean for low, mean in zip(lows, means)):
                ax.fill_between(xs, lows, highs, color=colors[method], alpha=0.15, linewidth=0)
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle=":")
    ax.set_title("Recurrence recovery (early revisit − first exposure)")
    ax.set_xlabel("Revisited context")
    ax.set_ylabel("Return difference")
    ax.set_xticks(range(len(context_order)))
    ax.set_xticklabels(context_order, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def _build_report(
    *,
    root: Path,
    runs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    seed_method_stage: dict[tuple[str, str, str | None], list[dict[str, Any]]],
    stage_a: list[dict[str, Any]],
    visits: list[dict[str, Any]],
    methods: list[str],
    root_summary: dict[str, Any],
    root_config: dict[str, Any],
    figure_path: Path,
    pdf_path: Path,
) -> str:
    lines = [
        "# Online adaptation report",
        "",
        f"Input directory: `{_relative_or_absolute(root)}`. This report was generated from logged run data.",
        "",
        f"![Stage A and Stage B metrics]({figure_path.name})",
        "",
        f"Vector figure: [{pdf_path.name}]({pdf_path.name})",
        "",
        "## Data coverage",
        "",
        f"- Run directories: {len(runs)}",
        f"- Episode records: {len(rows)} ({sum(1 for row in rows if row['_is_eval'])} marked evaluation)",
        f"- Training records used in curves and outcome summaries: {len(output_rows)}",
        f"- Methods: {', '.join(methods) if methods else 'none recorded'}",
        f"- Seed identifiers: {len({str(row['_seed']) for row in rows})}",
        f"- Mixed-context episode records retained in overall counts: {sum(1 for row in rows if _as_bool(row.get('mixed_context')))}",
        f"- Distinct seed/method/stage groups: {len(seed_method_stage)}",
        "- Curves first average within each seed and then across seeds. Confidence bands are 95% percentile bootstrap intervals resampled over seeds; episode rows are not treated as independent samples.",
        "- Stage A and Stage B are independent seed/method groups. Stage B per-context exposure metrics exclude rows marked `mixed_context=true`; overall run totals still use the runner's cumulative summary values.",
        "- Stage B records are grouped by each seed/method's recorded bout, visit, and context. Within each exposure, early means use the first up to 5 episodes and late means use the last up to 5. For exposures shorter than 10 episodes, these windows can overlap.",
        "",
        "## Initial source pretraining evaluation",
        "",
    ]

    pretrain_values: list[tuple[str, Any]] = []
    if "pretrain" in root_summary:
        pretrain_values.append((str(root.name), root_summary["pretrain"]))
    else:
        pretrain_values.extend(
            (str(run["path"].name), run["summary"]["pretrain"])
            for run in runs
            if "pretrain" in run["summary"]
        )
    if pretrain_values:
        for run_name, value in pretrain_values:
            lines.append(f"### `{_md(run_name)}`")
            lines.append("")
            lines.extend(_json_fence(value))
            lines.append("")
    else:
        lines.extend(["No `summary.pretrain` value was present in the run summaries.", ""])

    lines.extend(["## Stage A deployment training", ""])
    lines.extend(_stage_a_table(stage_a, methods))
    lines.append("")
    lines.append("### Baseline learning gate")
    lines.append("")
    gate_entries = _gate_entries(runs, root_summary, root_config)
    if gate_entries:
        lines.extend(["| Run | Gate result | Evidence |", "|---|---|---|"])
        for run_name, status, evidence in gate_entries:
            lines.append(f"| `{_md(run_name)}` | {status} | {_md(evidence)} |")
    else:
        lines.append("Gate status was not explicitly recorded in the available summary/config JSON, so attainment cannot be determined without inventing a threshold.")
    lines.append("")
    lines.append("### Evaluation episodes")
    lines.append("")
    lines.extend(_evaluation_table(runs, rows, methods))
    lines.append("")

    lines.extend(["## Stage B unknown-route and revisit metrics", ""])
    lines.append(
        "Operational definition: for each seed and method, the first chronological appearance of a recorded context in Stage B is an initial/unknown-route exposure; later exposures to that context are revisits. This is a logged-context proxy, not a claim about internal route knowledge."
    )
    lines.append("")
    lines.extend(_phase_b_table(visits, methods))
    lines.append("")
    lines.append("### Paired history retention comparison")
    lines.append("")
    lines.extend(_paired_history_table(visits))
    lines.append("")

    lines.extend(["## Runtime, interaction, and capacity", ""])
    lines.extend(_resource_table(runs, methods, root_summary))
    lines.extend([
        "",
        "Runtime, interaction, updates, and evaluation costs come from each run's `summary.json`; per-episode cumulative `elapsed_s` values are not summed into a runtime estimate. `peak_agent_state_bytes` is reported as recorded and is not added to `rollout_bytes`, because the agent-state total already includes the rollout buffer. Process RSS is reported separately. The maximum recorded payload/RSS is not proof of all memory usage.",
        "",
        "## Protocol and limitations",
        "",
        "- Three declared configurations were selected on validation seed 901. The initial pilot (11, 23, 37) exposed a duplicate-node routing bug and was retained. The corrected implementation was rerun on fresh seeds (41, 53, 67) with the selected hyperparameters unchanged. This is an implementation verification pilot, not a pristine confirmatory study.",
        "- The recorded evaluation_elapsed_s includes evaluation and output serialization; it is not isolated evaluation compute time. Online elapsed time excludes this output phase.",
        "- The environment is a small synthetic navigation task. A small number of seeds and rotation/slip context shifts do not establish generalization to other worlds or shift families.",
        "- There is no Phase C result here, so these data do not establish a shared-capacity conclusion.",
        "- `oracle_keep` is diagnostic only and is excluded from formal method claims.",
        "- The evaluated scope is a root plus leaf persistent residual bank with history routing. These runs do not establish parent-level sharing, compression efficiency, or a safe policy-improvement gate. The runner's `gain_gate` field describes experimental online trials; it is not the Stage A baseline gate.",
        "- When fewer than 5 paired seeds are available, the seed-bootstrap intervals and paired contrasts are exploratory. No statistical-significance claim is made.",
        "- Raw `episodes.csv` and `capacity.csv` files are retained in their run directories for review; this report does not replace them.",
        "",
        "## Software and procedural sources",
        "",
        "- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and [PPO](https://arxiv.org/abs/1707.06347); [Gymnasium](https://gymnasium.farama.org/). Installed versions are recorded in requirements-lock.txt.",
        "- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) inspired the experiment loop; the original language-model trainer was not executed.",
        "- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065). The stable-baselines3 and matplotlib skills supported integration and reporting; this citation does not imply external validation of the results.",
        "",
        "## Runner-supplied summaries",
        "",
    ])
    if root_summary and root_summary not in [run["summary"] for run in runs]:
        lines.append(f"### Suite summary `{_md(root.name)}/summary.json`")
        lines.append("")
        lines.extend(_json_fence(root_summary))
        lines.append("")
    if root_config:
        lines.append(f"### Suite config `{_md(root.name)}/config.json`")
        lines.append("")
        lines.extend(_json_fence(root_config))
        lines.append("")
    for run in runs:
        summary = run["summary"]
        if not summary:
            continue
        lines.append(f"### `{_md(run['path'].name)}`")
        lines.append("")
        lines.extend(_json_fence(summary))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _stage_a_table(stage_a_rows: list[dict[str, Any]], methods: list[str]) -> list[str]:
    by_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    grouped_rows = _seed_method_rows(stage_a_rows)
    for (_seed, method, _stage), seed_rows in grouped_rows.items():
        selected = [row for row in seed_rows if not row["_is_eval"]]
        returns = [row.get(f"_{_RETURN}") for row in selected]
        successes = [row.get(f"_{_SUCCESS}") for row in selected]
        return_value = _mean(returns)
        success_value = _mean(successes)
        if return_value is not None or success_value is not None:
            by_method[method].append(
                {"return": return_value if return_value is not None else math.nan,
                 "success": success_value if success_value is not None else math.nan}
            )
    lines = [
        "| Method | Seeds | Training return (mean ± seed SD) | Success fraction (mean ± seed SD) |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        entries = by_method.get(method, [])
        if not entries:
            lines.append(f"| {method} | 0 | n/a | n/a |")
            continue
        returns = [entry["return"] for entry in entries if math.isfinite(entry["return"])]
        successes = [entry["success"] for entry in entries if math.isfinite(entry["success"])]
        lines.append(
            f"| {method} | {len(entries)} | {_mean_sd(returns)} | {_mean_sd(successes)} |"
        )
    return lines


def _evaluation_table(
    runs: list[dict[str, Any]], rows: list[dict[str, Any]], methods: list[str]
) -> list[str]:
    lines = [
        "| Stage | Method | Seed | Evaluation | Episodes | Steps | Mean return | Success fraction |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        summary = run["summary"]
        stage = _normalise_stage(summary.get("stage", run["config"].get("stage")))
        method = str(summary.get("method", run["config"].get("method", "unknown-method")))
        seed = summary.get("seed", run["config"].get("seed", "unknown"))
        evaluation = summary.get("evaluation")
        if stage not in {"A", "B"} or not isinstance(evaluation, dict):
            continue
        for name, value in evaluation.items():
            if not isinstance(value, dict):
                continue
            lines.append(
                "| {stage} | {method} | {seed} | {name} | {episodes} | {steps} | {mean_return} | {success} |".format(
                    stage=stage,
                    method=_md(method),
                    seed=_md(seed),
                    name=_md(name),
                    episodes=_fmt(value.get("episodes"), integer=True),
                    steps=_fmt(value.get("steps"), integer=True),
                    mean_return=_fmt(value.get("mean_return")),
                    success=_fmt(value.get("success_rate")),
                )
            )
    # Preserve evaluation rows if a runner also writes them to episodes.csv.
    for row in rows:
        if row["_is_eval"] and row["_stage"] in {"A", "B"}:
            lines.append(
                f"| {row['_stage']} | {_md(row['_method'])} | {_md(row['_seed'])} | episodes.csv | 1 | {_fmt(row.get('_steps'), integer=True)} | {_fmt(row.get(f'_{_RETURN}'))} | {_fmt(row.get(f'_{_SUCCESS}'))} |"
            )
    if len(lines) == 2:
        return ["No evaluation metrics were recorded in per-run summaries or `is_eval=true` episode rows."]
    return lines


def _phase_b_table(visits: list[dict[str, Any]], methods: list[str]) -> list[str]:
    lines = [
        "| Method | Exposure type | Seed count | Early return | Late return | Early success | Late success | Early route confidence |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        for novelty in ("first-seen", "revisit"):
            selected = [visit for visit in visits if visit["method"] == method and visit["novelty"] == novelty]
            if not selected:
                continue
            seed_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for visit in selected:
                for field in ("early_return", "late_return", "early_success", "late_success", "early_confidence"):
                    value = visit.get(field)
                    if value is not None:
                        seed_metrics[visit["seed"]][field].append(float(value))
            field_means: dict[str, list[float]] = defaultdict(list)
            for metrics in seed_metrics.values():
                for field, values in metrics.items():
                    field_means[field].append(float(np.mean(values)))
            lines.append(
                "| {method} | {novelty} | {n} | {early} | {late} | {early_s} | {late_s} | {conf} |".format(
                    method=method,
                    novelty="Initial context exposure (unknown-route proxy)" if novelty == "first-seen" else "Context revisit",
                    n=len(seed_metrics),
                    early=_mean_sd(field_means["early_return"]),
                    late=_mean_sd(field_means["late_return"]),
                    early_s=_mean_sd(field_means["early_success"]),
                    late_s=_mean_sd(field_means["late_success"]),
                    conf=_mean_sd(field_means["early_confidence"]),
                )
            )
    if len(lines) == 2:
        return ["No Stage B training exposures were available to summarize."]
    return lines


def _paired_history_table(visits: list[dict[str, Any]]) -> list[str]:
    lines = ["| Outcome (history_keep − history_reset) | Paired seeds | Estimate | 95% paired seed-bootstrap interval | Reading |", "|---|---:|---:|---:|---|"]
    method_values: dict[str, dict[str, float]] = {"history_keep": {}, "history_reset": {}}
    for method in method_values:
        selected = [visit for visit in visits if visit["method"] == method and visit["novelty"] == "first-seen"]
        method_values[method] = _seed_level_values((visit["seed"], visit["early_return"]) for visit in selected)
    n, estimate, interval = _paired_bootstrap(method_values["history_keep"], method_values["history_reset"])
    lines.append(_paired_row("Initial exposure early return", n, estimate, interval))

    recurrence = _recurrence_values(visits)
    keep: dict[str, list[float]] = defaultdict(list)
    reset: dict[str, list[float]] = defaultdict(list)
    for (seed, method), context_values in recurrence.items():
        target = keep if method == "history_keep" else reset if method == "history_reset" else None
        if target is not None:
            target[seed].extend(context_values.values())
    keep_means = {seed: float(np.mean(values)) for seed, values in keep.items() if values}
    reset_means = {seed: float(np.mean(values)) for seed, values in reset.items() if values}
    n, estimate, interval = _paired_bootstrap(keep_means, reset_means, seed=2408)
    lines.append(_paired_row("Recurrence early-return recovery", n, estimate, interval))
    lines.append("")
    lines.append("Paired intervals resample matched seed clusters; they do not resample episode rows. An interval crossing zero is not a significance test, and with fewer than 5 pairs these contrasts are exploratory.")
    return lines


def _paired_row(label: str, n: int, estimate: float | None, interval: tuple[float, float] | None) -> str:
    if estimate is None:
        return f"| {label} | 0 | n/a | n/a | Insufficient matched seeds |"
    interval_text = "n/a" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
    reading = "Exploratory (< 5 pairs)" if n < 5 else "Descriptive paired estimate"
    return f"| {label} | {n} | {estimate:.3f} | {interval_text} | {reading} |"


def _resource_table(
    runs: list[dict[str, Any]], methods: list[str], root_summary: dict[str, Any]
) -> list[str]:
    """Summarize suite costs from run summaries, keeping stages separate."""

    suite_runs = root_summary.get("runs")
    if not isinstance(suite_runs, list) or not all(isinstance(item, dict) for item in suite_runs):
        suite_runs = [run["summary"] for run in runs]
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, summary in enumerate(suite_runs):
        leaf = runs[index] if index < len(runs) else None
        stage = _normalise_stage(
            summary.get("stage", leaf["config"].get("stage") if leaf else None)
        )
        method = str(
            summary.get("method", leaf["config"].get("method", "unknown-method") if leaf else "unknown-method")
        )
        seed = str(
            summary.get("seed", leaf["config"].get("seed", leaf["path"].name) if leaf else index)
        )
        if stage in {"A", "B"}:
            grouped[(stage, method)][seed].append(summary)

    lines = [
        "| Stage | Method | Seeds | Online steps (mean ± seed SD) | Online time s (mean ± seed SD) | Updates (mean ± seed SD) | Online cumulative return (mean ± seed SD) | Evaluation steps (mean ± seed SD) | Evaluation time s (mean ± seed SD) | Peak agent state bytes | Peak process RSS bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_names = (
        "online_steps",
        "online_elapsed_s",
        "updates",
        "online_cumulative_return",
        "evaluation_steps",
        "evaluation_elapsed_s",
    )
    for stage in ("A", "B"):
        for method in methods:
            seed_runs = grouped.get((stage, method), {})
            if not seed_runs:
                continue
            per_seed: dict[str, dict[str, float | None]] = {}
            for seed, summaries in seed_runs.items():
                values: dict[str, float | None] = {}
                for field in metric_names:
                    parsed = [_as_number(summary.get(field)) for summary in summaries]
                    valid = [value for value in parsed if value is not None]
                    values[field] = float(np.mean(valid)) if valid else None
                for field in ("peak_agent_state_bytes", "peak_process_rss_bytes"):
                    parsed = [_as_number(summary.get(field)) for summary in summaries]
                    values[field] = max((value for value in parsed if value is not None), default=None)
                per_seed[seed] = values

            def mean_sd_for(field: str) -> str:
                return _mean_sd(
                    [float(values[field]) for values in per_seed.values() if values[field] is not None]
                )

            max_agent = max(
                (values["peak_agent_state_bytes"] for values in per_seed.values()
                 if values["peak_agent_state_bytes"] is not None),
                default=None,
            )
            max_rss = max(
                (values["peak_process_rss_bytes"] for values in per_seed.values()
                 if values["peak_process_rss_bytes"] is not None),
                default=None,
            )
            lines.append(
                "| {stage} | {method} | {seeds} | {online_steps} | {online_time} | {updates} | {cumulative} | {evaluation_steps} | {evaluation_time} | {agent} | {rss} |".format(
                    stage=stage,
                    method=_md(method),
                    seeds=len(per_seed),
                    online_steps=mean_sd_for("online_steps"),
                    online_time=mean_sd_for("online_elapsed_s"),
                    updates=mean_sd_for("updates"),
                    cumulative=mean_sd_for("online_cumulative_return"),
                    evaluation_steps=mean_sd_for("evaluation_steps"),
                    evaluation_time=mean_sd_for("evaluation_elapsed_s"),
                    agent=_fmt(max_agent, integer=True),
                    rss=_fmt(max_rss, integer=True),
                )
            )
    if len(lines) == 2:
        lines.append("| n/a | n/a | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _gate_entries(
    runs: list[dict[str, Any]],
    root_summary: dict[str, Any],
    root_config: dict[str, Any],
) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    found = _find_gate(root_summary)
    if found is None:
        found = _find_gate(root_config)
    if found is not None:
        key, value = found
        status, evidence = _gate_status(value)
        entries.append(("suite summary/config", status, f"{key}: {evidence}"))
    for run in runs:
        found = _find_gate(run["summary"])
        if found is None:
            found = _find_gate(run["config"])
        if found is not None:
            key, value = found
            status, evidence = _gate_status(value)
            if not any(existing[2] == f"{key}: {evidence}" for existing in entries):
                entries.append((run["path"].name, status, f"{key}: {evidence}"))
    return entries


def _find_gate(value: Any, prefix: str = "") -> tuple[str, Any] | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            path = f"{prefix}.{key}" if prefix else str(key)
            if "gate" in normalized and any(token in normalized for token in ("learn", "baseline", "stagea", "phasea")):
                return path, child
        for key, child in value.items():
            found = _find_gate(child, f"{prefix}.{key}" if prefix else str(key))
            if found is not None:
                return found
    return None


def _gate_status(value: Any) -> tuple[str, str]:
    if isinstance(value, bool):
        return ("達成" if value else "未達成"), str(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"pass", "passed", "met", "achieved", "true", "yes"}:
            return "達成", value
        if token in {"fail", "failed", "not met", "unmet", "not achieved", "false", "no"}:
            return "未達成", value
        return "未判定", value
    if isinstance(value, dict):
        for field in ("passed", "pass", "met", "achieved", "ok", "success"):
            if field in value:
                status, _ = _gate_status(value[field])
                return status, _json_inline(value)
        for field in ("status", "result", "outcome"):
            if field in value:
                status, _ = _gate_status(value[field])
                return status, _json_inline(value)
        return "未判定", _json_inline(value)
    return "未判定", _json_inline(value)


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {sd:.3f}"


def _fmt(value: Any, integer: bool = False) -> str:
    number = _as_number(value)
    if number is None:
        return "n/a"
    if integer:
        return f"{int(round(number)):,}"
    return f"{number:.3f}"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _json_inline(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_fence(value: Any) -> list[str]:
    return ["```json", json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), "```"]


def _relative_or_absolute(path: Path) -> str:
    return str(path)

"""Create a focused seed-level figure from write investigation analysis JSON.

Example:
    python scripts/plot_write_investigation_focus.py \
        --analysis outputs/write_investigation_report/analysis.json \
        --output outputs/write_investigation_figures

The expected input follows ``write_investigation_report_v1``. A progress
analysis can be passed while cohorts are still running; unavailable panels or
conditions are labeled as pending and are never plotted as zero.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


_COLORS = {
    "independent_fast": "#4477AA",
    "dynamics_shared": "#EE6677",
    "reward_shared": "#228833",
    "trial_gate_relaxed": "#CCBB44",
    "projected_gate": "#AA3377",
    "trial_rollback": "#66CCEE",
    "independent7_lru": "#4477AA",
    "independent7_lfu": "#EE6677",
    "full_use_baseline": "#666666",
}
_CONDITION_LABELS = {
    "independent_fast": "Independent fast",
    "dynamics_shared": "Dynamics shared",
    "reward_shared": "Reward shared",
    "trial_gate_relaxed": "Trial gate relaxed",
    "projected_gate": "Projected gate",
    "trial_rollback": "Trial rollback",
    "independent7_lru": "Independent 7 LRU",
    "independent7_lfu": "Independent 7 LFU",
}


def _read_analysis(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        analysis = json.load(handle)
    if not isinstance(analysis, dict) or analysis.get("format") != "write_investigation_report_v1":
        raise ValueError(
            f"unsupported analysis format in {path}; expected write_investigation_report_v1"
        )
    if not isinstance(analysis.get("cohorts"), dict):
        raise ValueError(f"analysis has no cohort map: {path}")
    return analysis


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _pending_map(analysis: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in analysis.get("pending", []):
        if isinstance(item, dict) and item.get("name"):
            result[str(item["name"])] = str(item.get("status") or "pending")
    return result


def _seed_values(
    analysis: dict[str, Any],
    pending: dict[str, str],
    cohort_name: str,
    condition: str,
    metric: str,
) -> tuple[list[tuple[str, float]], str | None]:
    cohorts = analysis["cohorts"]
    cohort = cohorts.get(cohort_name)
    if not isinstance(cohort, dict):
        status = pending.get(cohort_name, "pending; no top-level cohort record in this report")
        return [], f"Pending: {status}"

    conditions = cohort.get("by_condition")
    condition_summary = conditions.get(condition) if isinstance(conditions, dict) else None
    if not isinstance(condition_summary, dict):
        status = pending.get(cohort_name)
        suffix = f" ({status})" if status else ""
        return [], f"Pending: {condition} row unavailable{suffix}"

    seed_metrics = condition_summary.get("seed_metrics")
    if not isinstance(seed_metrics, dict):
        return [], "Metric unavailable in completed records"

    values: list[tuple[str, float]] = []
    for seed, metrics in seed_metrics.items():
        if not isinstance(metrics, dict):
            continue
        value = _number(metrics.get(metric))
        if value is not None:
            values.append((str(seed), value))
    values.sort(key=lambda item: (0, int(item[0])) if item[0].lstrip("-").isdigit() else (1, item[0]))
    if not values:
        return [], f"Metric unavailable: {metric}"
    return values, None


def _condition_spec(
    label: str,
    cohort: str,
    condition: str,
    color: str | None = None,
) -> dict[str, str]:
    return {
        "label": label,
        "cohort": cohort,
        "condition": condition,
        "color": color or _COLORS.get(condition, "#4477AA"),
    }


def _condition_label(condition: str) -> str:
    return _CONDITION_LABELS.get(condition, condition.replace("_", " "))


def _stream_specs(phase: str, conditions: list[str]) -> list[dict[str, str]]:
    cohort = f"write_stream_{phase}"
    specs = [_condition_spec(_condition_label(condition), cohort, condition) for condition in conditions]
    specs.append(
        _condition_spec(
            "G0 full-use baseline",
            f"write_stream_{phase}_budget_baseline",
            "independent7_lru",
            _COLORS["full_use_baseline"],
        )
    )
    return specs


def _parent_validation_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    values_by_condition: dict[str, list[tuple[str, float]]] = {}
    for pair in analysis.get("pairing", {}).get("recycle_cumulative", {}).get("pairs", []):
        if not isinstance(pair, dict):
            continue
        if not str(pair.get("left", "")).startswith("write_parent_recycle_validation:"):
            continue
        if not str(pair.get("right", "")).startswith("write_parent_cumulative_validation:"):
            continue
        condition = str(pair["left"]).split(":", 1)[1]
        for row in pair.get("paired_deltas", []):
            if not isinstance(row, dict):
                continue
            delta = row.get("delta_left_minus_right")
            value = _number(delta.get("online_reward_per_step")) if isinstance(delta, dict) else None
            if value is not None:
                values_by_condition.setdefault(condition, []).append((str(row.get("seed", "?")), value))

    rows = []
    for condition in sorted(values_by_condition):
        rows.append(
            {
                "label": _condition_label(condition),
                "values": sorted(
                    values_by_condition[condition],
                    key=lambda item: (0, int(item[0])) if item[0].lstrip("-").isdigit() else (1, item[0]),
                ),
                "color": _COLORS.get(condition, "#4477AA"),
                "missing": None,
            }
        )
    return rows


def _format_mean(value: float) -> str:
    return f"{value:.3g}"


def _draw_panel(
    ax: Any,
    title: str,
    rows: list[dict[str, Any]],
    *,
    xlabel: str,
    include_zero: bool = False,
    empty_message: str | None = None,
    subtitle: str | None = None,
) -> None:
    title_pad = 34 if subtitle else 10
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=title_pad)
    if subtitle:
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#555555",
            linespacing=1.15,
            clip_on=False,
        )
    available = any(row.get("values") for row in rows)
    if not available:
        ax.set_axis_off()
        message = empty_message or "Pending: required cohort data is not in this report yet."
        ax.text(
            0.03,
            0.54,
            message,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            color="#555555",
            linespacing=1.5,
        )
        return

    y_positions = list(range(len(rows) - 1, -1, -1))
    all_values: list[float] = []
    tick_labels: list[str] = []
    for row, y in zip(rows, y_positions):
        values = row.get("values", [])
        if not values:
            state = row.get("missing") or "Pending"
            state = "Pending" if str(state).lower().startswith("pending") else "Metric unavailable"
            tick_labels.append(f"{row['label']} · {state}")
            continue

        numbers = [value for _, value in values]
        all_values.extend(numbers)
        mean = sum(numbers) / len(numbers)
        all_values.append(mean)
        tick_labels.append(
            f"{textwrap.fill(row['label'], width=22)}\nn={len(numbers)}; mean={_format_mean(mean)}"
        )
        if len(values) == 1:
            offsets = [0.0]
        else:
            offsets = [
                -0.13 + 0.26 * index / (len(values) - 1)
                for index in range(len(values))
            ]
        ax.scatter(
            numbers,
            [y + offset for offset in offsets],
            s=34,
            marker="o",
            color=row.get("color", "#4477AA"),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.scatter(
            [mean],
            [y],
            s=62,
            marker="D",
            color="#202020",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )

    ax.set_yticks(y_positions, tick_labels)
    top_extra = 0.6 if subtitle else 0.0
    ax.set_ylim(-0.65, len(rows) - 0.35 + top_extra)
    ax.tick_params(axis="y", labelsize=8, length=0, pad=7)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="x", color="#B7C0C8", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#7B858E")
    ax.tick_params(axis="x", labelsize=8)

    if all_values:
        low, high = min(all_values), max(all_values)
        if include_zero:
            low, high = min(low, 0.0), max(high, 0.0)
            ax.axvline(0.0, color="#858585", linestyle="--", linewidth=1.0, zorder=1)
        span = high - low
        scale = max(abs(low), abs(high))
        pad = span * 0.15 if span > 0 else scale * 0.15 if scale > 0 else 1e-9
        ax.set_xlim(low - pad, high + pad)


def _empty_cohort_note(names: list[str], pending: dict[str, str]) -> str:
    notes = ["Pending cohort data:"]
    for name in names:
        status = pending.get(name, "no top-level cohort record in this report")
        readable_name = name.replace("_", " ")
        readable_status = status.replace("_", " ")
        notes.append(
            textwrap.fill(
                f"{readable_name}: {readable_status}",
                width=38,
                subsequent_indent="  ",
            )
        )
    return "\n".join(notes)


def _make_focus_figure(analysis: dict[str, Any]) -> plt.Figure:
    pending = _pending_map(analysis)
    cohorts = analysis["cohorts"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 12.6), squeeze=False)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.87, wspace=0.62, hspace=0.62)

    step1_specs = [
        _condition_spec(_condition_label(label), "write_decision_formal", label)
        for label in (
            "independent_fast",
            "dynamics_shared",
            "reward_shared",
            "trial_gate_relaxed",
            "projected_gate",
        )
    ]
    step1_specs.append(
        _condition_spec(
            "Trial rollback · causal control",
            "write_decision_causal_control",
            "trial_rollback",
        )
    )
    step1_rows = []
    for spec in step1_specs:
        values, missing = _seed_values(
            analysis,
            pending,
            spec["cohort"],
            spec["condition"],
            "online_reward_per_step",
        )
        step1_rows.append({**spec, "values": values, "missing": missing})
    _draw_panel(
        axes[0, 0],
        "A · Step 1 formal arms and causal control",
        step1_rows,
        xlabel="Online reward per training step · train-only",
        empty_message=_empty_cohort_note(
            ["write_decision_formal", "write_decision_causal_control"], pending
        ),
    )

    known_specs = _stream_specs(
        "known", ["independent7_lru", "reward_shared", "trial_gate_relaxed", "projected_gate"]
    )
    known_rows = []
    for spec in known_specs:
        values, missing = _seed_values(
            analysis,
            pending,
            spec["cohort"],
            spec["condition"],
            "online_reward_per_stream_step",
        )
        known_rows.append({**spec, "values": values, "missing": missing})
    _draw_panel(
        axes[0, 1],
        "B · Known-boundary stream",
        known_rows,
        xlabel="Online reward per interaction · training + probe",
        empty_message=_empty_cohort_note(
            ["write_stream_known", "write_stream_known_budget_baseline"], pending
        ),
    )

    unknown_specs = _stream_specs(
        "unknown", ["independent7_lru", "independent7_lfu", "reward_shared", "trial_gate_relaxed"]
    )
    unknown_rows = []
    for spec in unknown_specs:
        values, missing = _seed_values(
            analysis,
            pending,
            spec["cohort"],
            spec["condition"],
            "online_reward_per_stream_step",
        )
        unknown_rows.append({**spec, "values": values, "missing": missing})
    _draw_panel(
        axes[0, 2],
        "C · Unknown-boundary stream",
        unknown_rows,
        xlabel="Online reward per interaction · training + probe",
        empty_message=_empty_cohort_note(
            ["write_stream_unknown", "write_stream_unknown_budget_baseline"], pending
        ),
    )

    recycle_rows = _parent_validation_rows(analysis)
    parent_names = ["write_parent_recycle_validation", "write_parent_cumulative_validation"]
    if recycle_rows:
        validation_means = {
            row["label"]: sum(value for _, value in row["values"]) / len(row["values"])
            for row in recycle_rows
            if row.get("values")
        }
        validation_conditions = {"Reward shared", "Trial gate relaxed"}
        parent_subtitle = None
        if validation_conditions.issubset(validation_means) and all(
            validation_means[condition] < 0 for condition in validation_conditions
        ):
            parent_subtitle = "Both validation means < 0; formal/unknown expansion skipped by preset gate"
        _draw_panel(
            axes[1, 0],
            "D · Parent recycle-fixed vs cumulative",
            recycle_rows,
            xlabel="Δ online reward per training step · recycle − cumulative",
            include_zero=True,
            subtitle=(
                "Both validation means < 0.\nPreset gate skipped formal/unknown follow-up."
                if parent_subtitle
                else None
            ),
        )
    else:
        validation_cohorts_present = all(name in cohorts for name in parent_names)
        if validation_cohorts_present:
            empty_parent_note = (
                "Parent validation cohorts are present, but no matched same-seed, "
                "same-condition pairs are available in this report."
            )
        else:
            empty_parent_note = _empty_cohort_note(parent_names, pending)
        _draw_panel(
            axes[1, 0],
            "D · Parent recycle-fixed vs cumulative",
            [],
            xlabel="Δ online reward per training step · recycle − cumulative",
            include_zero=True,
            empty_message=empty_parent_note + "\nAwaiting matched seed and condition pairs.",
        )

    for axis, foundation_seed in ((axes[1, 1], 17), (axes[1, 2], 29)):
        cohort_name = f"write_detour_w0_{foundation_seed}"
        specs = [
            _condition_spec(
                _condition_label(condition), cohort_name, condition
            )
            for condition in ("trial_gate_relaxed", "reward_shared", "independent7_lru")
        ]
        specs.append(
            _condition_spec(
                "G0 full-use baseline",
                f"write_detour_budget_w0_{foundation_seed}",
                "independent7_lru",
                _COLORS["full_use_baseline"],
            )
        )
        detour_rows = []
        for spec in specs:
            values, missing = _seed_values(
                analysis,
                pending,
                spec["cohort"],
                spec["condition"],
                "online_reward_per_stream_step",
            )
            detour_rows.append({**spec, "values": values, "missing": missing})
        _draw_panel(
            axis,
            f"{'E' if foundation_seed == 17 else 'F'} · Detour stream · foundation W0 {foundation_seed}",
            detour_rows,
            xlabel="Online reward per interaction · training + probe",
            empty_message=_empty_cohort_note([cohort_name], pending),
        )

    fig.suptitle(
        "Write investigation · seed-level focus",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor="#4477AA", markeredgecolor="white", label="Individual deployment seed",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=7,
            markerfacecolor="#202020", markeredgecolor="white", label="Mean across available seeds",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.918), ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.065,
        "Dots show seed-level values; diamonds show descriptive means. No confidence intervals or significance claims.\n"
        "Step 1 and parent validation use train-only reward; stream panels include training + probe interactions.\n"
        "Detour results remain separate for foundation W0 17 and W0 29.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#454545",
        linespacing=1.35,
    )
    return fig


def build_focus_figure(analysis_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    analysis_file = Path(analysis_path).resolve()
    output_path = Path(output_dir).resolve()
    if not analysis_file.is_file():
        raise FileNotFoundError(f"analysis JSON does not exist: {analysis_file}")
    output_path.mkdir(parents=True, exist_ok=True)

    analysis = _read_analysis(analysis_file)
    figure = _make_focus_figure(analysis)
    png_path = output_path / "write_investigation_focus.png"
    pdf_path = output_path / "write_investigation_focus.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("outputs/write_investigation_report/analysis.json"),
        help="write_investigation_report_v1 analysis JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/write_investigation_figures"),
        help="directory for the focused PNG and PDF figures",
    )
    args = parser.parse_args(argv)
    png_path, pdf_path = build_focus_figure(args.analysis, args.output)
    print(f"Wrote {png_path} and {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

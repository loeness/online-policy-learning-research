"""Run the two frozen detour full-use baseline cohorts after the main pipeline.

The coordinator waits for the primary follow-up pipeline to finish and for its
W0-29 detour summary to exist. It then runs the W0-17 and W0-29 budget
baselines sequentially, preserving per-command stdout and stderr logs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ROOT_PIPELINE_STATUS = OUTPUTS / "write_execution_logs" / "followups_status.json"
ROOT_W0_29_SUMMARY = OUTPUTS / "write_detour_w0_29" / "summary.json"
LOG_DIR = OUTPUTS / "write_detour_budget_logs"
STATUS_PATH = LOG_DIR / "coordinator_status.json"
SELECTION = "outputs/write_decision_selection/selection.json"
DEPLOYMENT_SEEDS = ("1061", "1063")
FOUNDATION_SEEDS = (17, 29)
POLL_SECONDS = 15


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_status(state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    with STATUS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _preflight() -> None:
    if LOG_DIR.exists() and any(LOG_DIR.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing coordinator logs or artifacts: {LOG_DIR}"
        )
    for seed in FOUNDATION_SEEDS:
        for relative in (
            f"write_detour_budget_w0_{seed}",
            f"write_detour_budget_w0_{seed}_report",
        ):
            path = OUTPUTS / relative
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite existing output; preserve it and inspect manually: {path}"
                )


def _wait_for_primary_pipeline(state: dict[str, Any]) -> None:
    last_status = None
    while True:
        if not ROOT_PIPELINE_STATUS.is_file():
            root_status = "status_file_unavailable"
            summary_exists = ROOT_W0_29_SUMMARY.is_file()
        else:
            payload = _read_json(ROOT_PIPELINE_STATUS)
            root_status = str(payload.get("status", "unknown")) if isinstance(payload, dict) else "invalid"
            summary_exists = ROOT_W0_29_SUMMARY.is_file()

        if root_status == "failed":
            state.update(
                phase="blocked_by_primary_pipeline_failure",
                root_pipeline_status=root_status,
                root_w0_29_summary_exists=summary_exists,
            )
            _write_status(state)
            raise RuntimeError(
                "primary follow-up pipeline status is failed; budget training was not started"
            )

        if root_status == "complete" and summary_exists:
            state.update(
                phase="primary_pipeline_complete",
                root_pipeline_status=root_status,
                root_w0_29_summary_exists=True,
            )
            _write_status(state)
            print("Primary pipeline complete and W0-29 summary exists; starting sequential baselines.", flush=True)
            return

        if root_status != last_status:
            print(
                f"Waiting for primary pipeline: status={root_status}, "
                f"W0-29 summary exists={summary_exists}.",
                flush=True,
            )
            last_status = root_status
        state.update(
            phase="waiting_for_primary_pipeline",
            root_pipeline_status=root_status,
            root_w0_29_summary_exists=summary_exists,
        )
        _write_status(state)
        time.sleep(POLL_SECONDS)


def _run_step(
    state: dict[str, Any],
    *,
    label: str,
    command: list[str],
) -> None:
    stdout_path = LOG_DIR / f"{label}.stdout.log"
    stderr_path = LOG_DIR / f"{label}.stderr.log"
    state["phase"] = f"running_{label}"
    state["current_command"] = command
    state["current_stdout"] = str(stdout_path.relative_to(ROOT))
    state["current_stderr"] = str(stderr_path.relative_to(ROOT))
    _write_status(state)
    print(f"Starting {label}.", flush=True)
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    state.setdefault("steps", []).append(
        {
            "label": label,
            "returncode": completed.returncode,
            "stdout": str(stdout_path.relative_to(ROOT)),
            "stderr": str(stderr_path.relative_to(ROOT)),
            "finished_at": _now(),
        }
    )
    state.pop("current_command", None)
    state.pop("current_stdout", None)
    state.pop("current_stderr", None)
    _write_status(state)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} exited with code {completed.returncode}; preserved logs at "
            f"{stdout_path} and {stderr_path}"
        )


def _run_cohort(state: dict[str, Any], foundation_seed: int) -> None:
    name = f"write_detour_budget_w0_{foundation_seed}"
    output = f"outputs/{name}"
    foundation = f"outputs/write_decision_foundations/seed_{foundation_seed}/foundation.pt"
    train_command = [
        sys.executable,
        "scripts/run_write_budget_baseline.py",
        "--phase",
        "known",
        "--selection",
        SELECTION,
        "--domain",
        "detour",
        "--foundation",
        foundation,
        "--seeds",
        *DEPLOYMENT_SEEDS,
        "--workers",
        "2",
        "--output",
        output,
    ]
    _run_step(state, label=f"w0_{foundation_seed}_baseline", command=train_command)
    if not (OUTPUTS / name / "summary.json").is_file():
        raise FileNotFoundError(
            f"baseline runner returned success without writing {OUTPUTS / name / 'summary.json'}"
        )

    report_command = [
        sys.executable,
        "scripts/report_write_stream.py",
        "--input",
        output,
        "--output",
        f"outputs/{name}_report",
    ]
    _run_step(state, label=f"w0_{foundation_seed}_report", command=report_command)
    if not (OUTPUTS / f"{name}_report" / "analysis.json").is_file():
        raise FileNotFoundError(
            f"report command returned success without writing "
            f"{OUTPUTS / f'{name}_report' / 'analysis.json'}"
        )


def main() -> int:
    _preflight()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "started_at": _now(),
        "status": "running",
        "phase": "initializing",
        "foundation_seeds": list(FOUNDATION_SEEDS),
        "deployment_seeds": [int(seed) for seed in DEPLOYMENT_SEEDS],
        "max_training_workers": 2,
        "steps": [],
    }
    _write_status(state)
    try:
        _wait_for_primary_pipeline(state)
        for foundation_seed in FOUNDATION_SEEDS:
            _run_cohort(state, foundation_seed)
        state.update(status="complete", phase="complete")
        _write_status(state)
        print(f"Completed both detour full-use baseline cohorts; logs are in {LOG_DIR}.", flush=True)
        return 0
    except Exception as error:
        state.update(status="failed", phase="failed", error=str(error))
        _write_status(state)
        raise
    finally:
        state["finished_or_updated_at"] = _now()
        _write_status(state)


if __name__ == "__main__":
    raise SystemExit(main())

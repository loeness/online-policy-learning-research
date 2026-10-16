"""Execute the already-declared follow-up matrix with at most two learners.

The coordinator and per-stage logs survive a chat/tool-session interruption.
It never edits completed experiments or selects rules from formal seeds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LOGS = OUTPUTS / "write_execution_logs"
SELECTION = "outputs/write_decision_selection/selection.json"
STATUS = LOGS / "followups_status.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main(wait_pid=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f")
    history = LOGS / "attempt_history" / attempt
    if STATUS.exists():
        history.mkdir(parents=True, exist_ok=True)
        (history / STATUS.name).write_bytes(STATUS.read_bytes())
    state = {"started": datetime.now(timezone(timedelta(hours=8))).isoformat(),
             "max_training_workers": 2, "steps": [], "status": "running"}
    write(STATUS, state)

    def run(label, command, cohort=None):
        if cohort and (OUTPUTS / cohort / "summary.json").exists():
            state["steps"].append({"label": label, "status": "already_complete", "cohort": cohort})
            write(STATUS, state)
            return
        if command[0] == "scripts/report_write_stream.py":
            report_dir = (ROOT / command[command.index("--output") + 1]).resolve()
            if (report_dir / "analysis.json").is_file():
                state["steps"].append({"label": label, "status": "already_complete_report",
                                       "output": str(report_dir)})
                write(STATUS, state)
                return
        entry = {"label": label, "command": command, "status": "running", "cohort": cohort}
        state["steps"].append(entry)
        write(STATUS, state)
        print(label, flush=True)
        started = time.perf_counter()
        for suffix in ("stdout.log", "stderr.log"):
            previous = LOGS / f"{label}.{suffix}"
            if previous.exists():
                history.mkdir(parents=True, exist_ok=True)
                (history / previous.name).write_bytes(previous.read_bytes())
        with (LOGS / f"{label}.stdout.log").open("w", encoding="utf-8") as out, \
             (LOGS / f"{label}.stderr.log").open("w", encoding="utf-8") as err:
            process = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=out, stderr=err,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        entry.update(status="complete" if process.returncode == 0 else "failed",
                     returncode=process.returncode, elapsed_s=time.perf_counter() - started)
        write(STATUS, state)
        if process.returncode:
            raise RuntimeError(f"{label} failed; see {LOGS / (label + '.stderr.log')}")

    try:
        if wait_pid:
            import psutil
            deadline = time.monotonic() + 3600
            while not (OUTPUTS / "write_stream_unknown/summary.json").exists():
                if not psutil.pid_exists(wait_pid) or time.monotonic() > deadline:
                    raise RuntimeError("Unknown-route resumption exited before its summary became available")
                time.sleep(5)
        run("unknown_report", ["scripts/report_write_stream.py", "--input", "outputs/write_stream_unknown",
                               "--output", "outputs/write_stream_unknown_report"])

        common = ["--phase", "known", "--selection", SELECTION, "--conditions", "trial_gate_relaxed", "reward_shared",
                  "--seeds", "1049", "1051", "--bout-steps", "1024", "--workers", "2"]
        for variant, flag in (("cumulative", ["--no-recycle"]), ("recycle", [])):
            name = f"write_parent_{variant}_validation"
            run(f"parent_{variant}_validation", ["scripts/run_write_recycle.py", *common, *flag,
                                                 "--output", f"outputs/{name}"], name)

        def means(name):
            records = json.loads((OUTPUTS / name / "summary.json").read_text(encoding="utf-8"))["records"]
            return {condition: sum(r["reward_per_step"] for r in records if r["condition"] == condition) /
                    sum(r["condition"] == condition for r in records)
                    for condition in ("trial_gate_relaxed", "reward_shared")}
        fixed = means("write_parent_cumulative_validation")
        recycled = means("write_parent_recycle_validation")
        gains = {condition: recycled[condition] - fixed[condition] for condition in fixed}
        best = max(gains, key=lambda condition: (gains[condition], condition))
        admitted = any(gain > 0 for gain in gains.values())
        chosen = {"validation_seeds": [1049, 1051], "fixed_mean_reward": fixed,
                  "recycle_mean_reward": recycled, "paired_mean_gain": gains,
                  "formal_admitted": admitted, "best_validation_condition": best,
                  "rule": "Prespecified positive validation mean gain; no formal ranking tunes thresholds."}
        write(OUTPUTS / "write_parent_selection/selection.json", chosen)
        for variant in ("cumulative", "recycle"):
            run(f"parent_{variant}_validation_report", ["scripts/report_write_stream.py", "--input",
                f"outputs/write_parent_{variant}_validation", "--output", f"outputs/write_parent_{variant}_validation_report"])

        formal_gains = {}
        if admitted:
            for variant, flag in (("cumulative", ["--no-recycle"]), ("recycle", [])):
                name = f"write_parent_{variant}_formal"
                run(f"parent_{variant}_formal", ["scripts/run_write_recycle.py", "--phase", "known", "--selection", SELECTION,
                    "--conditions", "trial_gate_relaxed", "reward_shared", "--seeds", "1093", "1097", "1103",
                    "--bout-steps", "2048", "--workers", "2", *flag, "--output", f"outputs/{name}"], name)
                run(f"parent_{variant}_formal_report", ["scripts/report_write_stream.py", "--input", f"outputs/{name}",
                                                       "--output", f"outputs/{name}_report"])
            a, b = means("write_parent_cumulative_formal"), means("write_parent_recycle_formal")
            formal_gains = {condition: b[condition] - a[condition] for condition in a}
            if any(gain > 0 for gain in formal_gains.values()):
                for variant, flag in (("cumulative", ["--no-recycle"]), ("recycle", [])):
                    name = f"write_parent_{variant}_unknown"
                    run(f"parent_{variant}_unknown", ["scripts/run_write_recycle.py", "--phase", "unknown", "--selection", SELECTION,
                        "--conditions", best, "--seeds", "1123", "1129", "1151", "--bout-steps", "2048",
                        "--workers", "2", *flag, "--output", f"outputs/{name}"], name)
                    run(f"parent_{variant}_unknown_report", ["scripts/report_write_stream.py", "--input", f"outputs/{name}",
                                                            "--output", f"outputs/{name}_report"])
                name = "write_parent_unknown_budget_baseline"
                run("parent_unknown_budget_baseline", ["scripts/run_write_budget_baseline.py", "--phase", "unknown",
                    "--selection", SELECTION, "--seeds", "1123", "1129", "1151", "--workers", "2",
                    "--output", f"outputs/{name}"], name)
                run("parent_unknown_budget_report", ["scripts/report_write_stream.py", "--input", f"outputs/{name}",
                                                      "--output", f"outputs/{name}_report"])
        else:
            write(OUTPUTS / "write_parent_selection/SKIPPED.json", {
                "formal_and_unknown_followup": "skipped per prespecified validation rule", "validation_gains": gains})
        write(OUTPUTS / "write_parent_selection/formal_decision.json", {
            "formal_mean_gains": formal_gains,
            "unknown_followup_admitted": bool(formal_gains and any(v > 0 for v in formal_gains.values()))})

        for phase in ("known", "unknown"):
            name = f"write_stream_{phase}_budget_baseline"
            run(f"{phase}_budget_baseline", ["scripts/run_write_budget_baseline.py", "--phase", phase,
                "--selection", SELECTION, "--workers", "2", "--output", f"outputs/{name}"], name)
            run(f"{phase}_budget_report", ["scripts/report_write_stream.py", "--input", f"outputs/{name}",
                                          "--output", f"outputs/{name}_report"])
        for foundation_seed in (17, 29):
            name = f"write_detour_w0_{foundation_seed}"
            run(name, ["scripts/run_write_stream.py", "--phase", "known", "--selection", SELECTION,
                "--conditions", "trial_gate_relaxed", "reward_shared", "independent7_lru", "--seeds", "1061", "1063",
                "--bout-steps", "2048", "--domain", "detour", "--foundation",
                f"outputs/write_decision_foundations/seed_{foundation_seed}/foundation.pt", "--workers", "2",
                "--output", f"outputs/{name}"], name)
            run(f"{name}_report", ["scripts/report_write_stream.py", "--input", f"outputs/{name}", "--output", f"outputs/{name}_report"])
        run("investigation_report", ["scripts/summarize_write_investigation.py", "--output", "outputs/write_investigation_report"])
        state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["updated"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        write(STATUS, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-unknown-resume-pid", type=int)
    args = parser.parse_args()
    main(args.wait_for_unknown_resume_pid)

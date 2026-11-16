"""Summarize the byte-verified run002 mirror without loading any model."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RUN = BASE / "runs" / "run002"
RAW = RUN / "remote" / "run002"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (HERE / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def span(values):
    return {"min": min(values), "max": max(values), "mean": statistics.mean(values)}


def main():
    mirror = read(RUN / "mirror_manifest.json")
    assert mirror["state"] == "REMOTE_BYTE_IDENTITIES_VERIFIED"
    report = read(RAW / "scientific_report.json")
    fit_rows, prefix_rows = [], []
    for evaluation in report["evaluations"]:
        seed = evaluation["seed"]
        row = {key: evaluation[key] for key in (
            "seed", "K_new_W0", "K_old_C", "K_old_D", "K_new_C", "K_new_D", "G_new_C", "G_new_D")}
        row.update(evaluation["conditions"])
        row["passed"] = evaluation["passed"]
        row["failed_conditions"] = ";".join(evaluation["failed_conditions"])
        for arm in ("C", "D"):
            folder = RAW / "cases" / f"s{seed}" / arm.lower()
            diag, metrics = read(folder / "diagnostics.json"), read(folder / "metrics.json")
            row[f"{arm}_returned_is_C0"] = diag["returned_is_C0"]
            row[f"{arm}_fallback_applied"] = diag["fallback_applied"]
            fit = {"seed": seed, "arm": arm, "lambda_value": metrics["lambda_value"],
                   "candidate_sha256": metrics["candidate_sha256"],
                   "returned_is_C0": diag["returned_is_C0"], "fallback_applied": diag["fallback_applied"],
                   "fallback_reason": diag["fallback_reason"], "termination_reason": diag["termination_reason"],
                   "closure_count": diag["closure_count"], "iterations": diag["iterations"],
                   "fit_cpu_subtotal_seconds": diag["cpu_time_seconds"],
                   "fit_wall_subtotal_seconds": diag["wall_time_seconds"],
                   "process_lifetime_peak_memory_bytes": diag["ownprocess_peak_memory_bytes"],
                   "teacher_array_bytes": diag["teacher_array_bytes"],
                   "temporary_array_bytes_estimate": diag["temporary_array_bytes_estimate"],
                   "optimizer_state_tensor_bytes": diag["optimizer_state_tensor_bytes"],
                   "actor_bytes": metrics["capacity"]["actor_bytes"],
                   "full_persistent_bytes": metrics["capacity"]["full_persistent_bytes"],
                   "compensation_max_abs_error": metrics["compensation"]["max_effective_preservation_error"],
                   "resident_probability_max_abs_drift": metrics["resident_drift"]["probability_max_abs"]}
            for owner, value in enumerate(metrics["resident_drift"]["KL_by_owner_all128"]):
                fit[f"resident_KL_owner{owner}"] = value
            for stage in ("pre_codec_fresh", "encoded_fresh"):
                for key in ("old_train", "old_holdout", "new_train", "new_holdout"):
                    fit[f"{stage}_{key}_KL"] = metrics[stage][key]
            for group in ("old", "new"):
                fit[f"encoding_change_{group}_holdout_KL"] = (
                    metrics["encoded_fresh"][group + "_holdout"] - metrics["pre_codec_fresh"][group + "_holdout"])
            fit_rows.append(fit)
        prefix_rows.append(row)
    assert len(prefix_rows) == 16 and len(fit_rows) == 32
    write_csv("per_prefix.csv", prefix_rows)
    write_csv("per_fit.csv", fit_rows)
    failure_seeds = {key: [r["seed"] for r in prefix_rows if not r[key]]
                     for key in report["evaluations"][0]["conditions"]}
    summary = {
        "experiment_id": "E163", "campaign": "run002", "source": "byte-verified raw mirror",
        "server_gate_status": report["status"], "pass_count": sum(r["passed"] for r in prefix_rows),
        "gate_denominator": 16, "required_pass_count": 12,
        "passed_seed_ids": [r["seed"] for r in prefix_rows if r["passed"]],
        "failed_condition_seed_ids": failure_seeds,
        "returned_C0_seed_ids": {arm: [r["seed"] for r in fit_rows if r["arm"] == arm and r["returned_is_C0"]]
                                 for arm in ("C", "D")},
        "passed_with_D_C0_seed_ids": [r["seed"] for r in prefix_rows if r["passed"] and r["D_returned_is_C0"]],
        "fallback_applied_count": sum(r["fallback_applied"] for r in fit_rows),
        "termination_counts": dict(collections.Counter(r["termination_reason"] for r in fit_rows)),
        "closure_total": sum(r["closure_count"] for r in fit_rows),
        "ranges": {key: span([r[key] for r in fit_rows]) for key in (
            "closure_count", "full_persistent_bytes", "process_lifetime_peak_memory_bytes", "teacher_array_bytes",
            "temporary_array_bytes_estimate", "optimizer_state_tensor_bytes", "compensation_max_abs_error",
            "resident_probability_max_abs_drift", "encoding_change_old_holdout_KL", "encoding_change_new_holdout_KL")},
        "resident_KL_range": span([r[k] for r in fit_rows for k in ("resident_KL_owner0", "resident_KL_owner1")]),
        "limitations": ["Gate is a KL proxy screen, not a reward estimate.",
                        "Process peak is an interpreter lifetime high-water mark, not an isolated per-fit workspace peak.",
                        "Fit stage time subtotals do not replace complete supervisor lifecycle accounting.",
                        "C0 return does not imply float32 fallback or a learned protective update."],
    }
    write_json(HERE / "summary.json", summary)
    derived = []
    for name in ("per_prefix.csv", "per_fit.csv", "summary.json"):
        data = (HERE / name).read_bytes()
        derived.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    write_json(HERE / "derived_manifest.json", {"source_mirror_manifest": "../../runs/run002/mirror_manifest.json", "files": derived})
    print(json.dumps({"prefix_rows": 16, "fit_rows": 32, "passes": summary["pass_count"],
                      "returned_C0_seed_ids": summary["returned_C0_seed_ids"]}))


if __name__ == "__main__":
    main()

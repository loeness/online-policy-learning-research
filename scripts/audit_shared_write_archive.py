"""Reconcile shared-write cohorts and verify that prior result bytes survived."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "5979956485113ef1907675baf37137d4cb2b1604"
COHORTS = (
    "knowledge_shared_write_pilot", "knowledge_shared_write_formal",
    "knowledge_shared_write_auto_pilot", "knowledge_shared_write_auto_formal",
    "knowledge_shared_write_capacity_pilot", "knowledge_shared_write_capacity_formal",
    "knowledge_shared_write_conflict_pilot", "knowledge_shared_write_conflict_formal",
)
REPORTS = (
    "knowledge_shared_write_pilot_report", "knowledge_shared_write_report",
    "knowledge_shared_write_auto_pilot_report", "knowledge_shared_write_auto_report",
    "knowledge_shared_write_capacity_pilot_report", "knowledge_shared_write_capacity_report",
    "knowledge_shared_write_conflict_pilot_report", "knowledge_shared_write_conflict_report",
)


def audit():
    old = json.loads(subprocess.check_output(["git", "show", f"{REFERENCE}:archive/manifest.json"], cwd=ROOT))
    prior = [r for r in old["files"] if r["path"].startswith(("data/", "outputs/"))]
    failures = [r["path"] for r in prior if not (ROOT/r["path"]).is_file()
                or hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest() != r["sha256"]]
    cohorts = []
    for name in COHORTS:
        d = json.loads((ROOT/"outputs"/name/"summary.json").read_text(encoding="utf-8"))
        cohorts.append({"path":f"outputs/{name}", "phase":d["phase"],
            "training_steps":d["training_steps"], "probe_steps":d.get("probe_steps",0),
            "evaluation_steps":d["evaluation_steps"], "total_interactions":d["total_interactions"]})
    reports = []
    for name in REPORTS:
        folder = ROOT/"outputs"/name
        file = folder/("report.json" if "conflict" in name else "audit.json")
        d = json.loads(file.read_text(encoding="utf-8"))
        a = d["audit"] if "conflict" in name else d
        reports.append({"path":str(file.relative_to(ROOT)).replace("\\","/"),
            "checks":len(a["checks"]), "failed_assertions":a["failed_assertions"]})
    totals = {k:sum(r[k] for r in cohorts) for k in
              ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")}
    valid = not failures and not any(r["failed_assertions"] for r in reports)
    valid = valid and all(r["training_steps"]+r["probe_steps"]+r["evaluation_steps"] == r["total_interactions"] for r in cohorts)
    result = {"valid":valid, "old_reference_commit":REFERENCE,
              "old_data_output_files_checked":len(prior), "old_files_changed_or_missing":failures,
              "cohorts":cohorts, "totals":totals, "report_audits":reports,
              "total_report_checks":sum(r["checks"] for r in reports),
              "scope":"Experiment cohorts include training, probes and evaluation. Offline replay and reports generate zero new interactions. Unit-test fixtures are outside this cohort ledger. One shared W0; controlled diagnostics, not SOTA or a completed unknown-boundary organizer."}
    output = ROOT/"outputs/shared_write_archive_audit/cohort_ledger.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"valid":valid,"old_files_checked":len(prior),"totals":totals,
                      "report_checks":result["total_report_checks"]}))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()

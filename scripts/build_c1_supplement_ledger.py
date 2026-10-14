"""Combine declared C1 experimental interaction counts without double counting."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/c1_supplement"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    formal=list(OUT.glob("step[23]*/**/formal/**/result.json"))
    validation=list(OUT.glob("step[23]*/**/validation/**/result.json"))
    for cohort in ("representation_alternatives","context_head_alternatives"):
        formal+=list((OUT/cohort).glob("**/formal/**/result.json"))
        validation+=list((OUT/cohort).glob("**/validation/**/result.json"))
    if len(formal)!=57 or len(validation)!=39:
        raise RuntimeError(f"Incomplete suite: {len(formal)} formal, {len(validation)} validation")
    fs,vs=[read(p) for p in formal],[read(p) for p in validation]
    model=read(OUT/"step1_models/summary.json")
    td=read(OUT/"step1_models/tree_diagnostic_summary.json")["runs"]
    md=read(OUT/"step1_models/model_diagnostic_summary.json")["runs"]
    source=read(OUT/"source_retention.json")
    counts={
        "formal_neural_training":sum(r.get("training_interactions",0) for r in fs),
        "validation_neural_training":sum(r.get("training_interactions",0) for r in vs),
        "formal_neural_evaluation":sum(r.get("evaluation_interactions",0) for r in fs),
        "validation_neural_evaluation":sum(r.get("evaluation_interactions",0) for r in vs),
        "formal_model_trajectories":model["formal_interactions"],
        "validation_model_trajectories":model["validation_interactions"],
        "frozen_tree_diagnostic":sum(r["online_steps"] for r in td),
        "cleared_model_diagnostic":sum(r["online_steps"] for r in md),
        "paired_group_equivalence":4096,
        "post_training_source_diagnostic":source["additional_evaluation_interactions"]}
    rows={"scope":"Declared supplemental validation/formal/diagnostic experiments only; original A/B/C1 and software integration test interactions are excluded. Counts are disjoint. Source data processing adds no new environment interactions.",
          "formal_neural_runs":len(fs),"validation_neural_runs":len(vs),
          "runtime_or_capacity_failures":sum(r["status"]!="complete" for r in fs+vs),
          "model_formal_runs":12,"model_validation_runs":4,
          "common_foundation_checkpoints":1,"new_pretraining_interactions":0,
          "counts":counts,"total_declared_interactions":sum(counts.values()),
          "optimizer_audit":read(OUT/"optimizer_step_audit.json")["corrected_runs"],
          "software_check_log":"outputs/c1_supplement/software_checks.txt"}
    (OUT/"complete_ledger.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in rows.items() if k not in ("scope","counts") }))

if __name__=="__main__":main()
